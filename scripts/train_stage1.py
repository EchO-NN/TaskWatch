#!/usr/bin/env python3
from __future__ import annotations

import argparse
import faulthandler
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from vljepa_droid.config import load_config
from vljepa_droid.data.collate import collate_droid_samples
from vljepa_droid.data.dataset import PreparedDroidDataset
from vljepa_droid.distributed.grad_sync import chunked_all_reduce_gradients
from vljepa_droid.evaluation.runner import evaluate_model, write_retrieval_metrics
from vljepa_droid.losses.infonce import bidirectional_infonce
from vljepa_droid.models.factory import build_stage1_model
from vljepa_droid.models.vljepa import VLJEPAModel
from vljepa_droid.training.checkpoint import load_checkpoint, save_checkpoint
from vljepa_droid.training.optimizer import build_optimizer


def initialize_distributed() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return rank, world_size, local_rank, device


def seed_everything(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def unwrap(model: VLJEPAModel | DistributedDataParallel) -> VLJEPAModel:
    return model.module if isinstance(model, DistributedDataParallel) else model


def dataset_from_config(config: dict, split: str, training: bool) -> PreparedDroidDataset:
    use_cache = bool(config["training"]["use_feature_cache"])
    dataset = PreparedDroidDataset(
        config["paths"]["prepared_dir"],
        split=split,
        training=training,
        feature_cache_dir=config["paths"]["feature_cache_dir"] if use_cache else None,
        resize_short_side=int(config["data"]["resize_short_side"]),
        resolution=int(config["data"]["resolution"]),
        mean=tuple(config["data"]["normalize_mean"]),
        std=tuple(config["data"]["normalize_std"]),
        seed=int(config["experiment"]["seed"]),
    )
    configured_limit = config["data"].get(f"{split}_episodes")
    if configured_limit is not None:
        configured_limit = int(configured_limit)
        if configured_limit < 1 or configured_limit > len(dataset.records):
            raise ValueError(
                f"invalid {split} episode limit {configured_limit}; "
                f"manifest contains {len(dataset.records)}"
            )
        dataset.records = dataset.records[:configured_limit]
    return dataset


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(payload, sort_keys=True) + "\n")


def synchronize_max(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor)


def verify_identical_initialization(model: VLJEPAModel, device: torch.device) -> None:
    """Permit DDP init_sync=False only after checking every trainable tensor."""
    if not dist.is_initialized():
        return
    checksums = torch.stack(
        [
            parameter.detach().float().sum()
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
    ).to(device)
    minimum = checksums.clone()
    maximum = checksums.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    if not torch.equal(minimum, maximum):
        raise RuntimeError("model initialization differs across DDP ranks")


def run_evaluation(
    model: VLJEPAModel | DistributedDataParallel,
    validation_dataset: PreparedDroidDataset,
    config: dict,
    output_dir: Path,
    step: int,
    rank: int,
    device: torch.device,
) -> None:
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        metrics = evaluate_model(
            unwrap(model),
            validation_dataset,
            device=device,
            batch_size=int(config["training"]["local_batch_size"]),
            num_workers=int(config["training"]["num_workers"]),
        )
        metrics["step"] = step
        write_retrieval_metrics(output_dir / f"retrieval_step_{step:07d}.json", metrics)
        append_jsonl(output_dir / "retrieval.jsonl", metrics)
        print(json.dumps({"evaluation": metrics}, sort_keys=True), flush=True)
    if dist.is_initialized():
        dist.barrier()


def main() -> None:
    faulthandler.enable()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--local-loss-debug",
        action="store_true",
        help="diagnostic only: disable cross-rank embedding gather",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    faulthandler_interval = int(config.get("debug", {}).get("faulthandler_interval_seconds", 900))
    if faulthandler_interval > 0:
        faulthandler.dump_traceback_later(faulthandler_interval, repeat=True)
    if args.output_dir:
        config["paths"]["output_dir"] = str(Path(args.output_dir).resolve())
    if args.local_loss_debug:
        config["loss"]["distributed_gather"] = False
    rank, world_size, local_rank, device = initialize_distributed()
    seed_everything(int(config["experiment"]["seed"]), 0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train_log = output_dir / "train.jsonl"
    if rank == 0 and train_log.exists() and not args.resume:
        raise FileExistsError(f"refusing to mix training runs in {train_log}")

    train_dataset = dataset_from_config(config, "train", True)
    validation_dataset = dataset_from_config(config, "validation", False)
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=int(config["experiment"]["seed"]),
        drop_last=bool(config["training"]["drop_last"]),
    )
    loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["local_batch_size"]),
        sampler=sampler,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=True,
        drop_last=bool(config["training"]["drop_last"]),
        collate_fn=collate_droid_samples,
    )
    if len(loader) == 0:
        raise RuntimeError("training DataLoader has no full batch")

    use_cache = bool(config["training"]["use_feature_cache"])
    base_model = build_stage1_model(
        config,
        device=device,
        include_x_encoder=not use_cache,
    )
    optimizer = build_optimizer(base_model, config)
    global_step = 0
    if args.resume:
        global_step = load_checkpoint(args.resume, model=base_model, optimizer=optimizer)
    model: VLJEPAModel | DistributedDataParallel = base_model
    gradient_process_group = None
    if world_size > 1:
        verify_identical_initialization(base_model, device)
        if config["training"].get("gradient_sync") != "chunked_gloo_allreduce":
            raise ValueError(
                "dual-Blackwell baseline requires explicit gradient_sync=chunked_gloo_allreduce"
            )
        gradient_process_group = dist.new_group(backend="gloo")
    seed_everything(int(config["experiment"]["seed"]), rank)

    if rank == 0:
        (output_dir / "resolved_config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        parameter_summary = {
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in unwrap(model).parameters()
                if parameter.requires_grad
            ),
            "frozen_parameters_present": sum(
                parameter.numel()
                for parameter in unwrap(model).parameters()
                if not parameter.requires_grad
            ),
            "world_size": world_size,
            "local_batch_size": int(config["training"]["local_batch_size"]),
        }
        (output_dir / "parameter_summary.json").write_text(
            json.dumps(parameter_summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(parameter_summary), flush=True)

    max_steps = int(config["training"]["max_steps"])
    log_every = int(config["training"]["log_every_steps"])
    eval_every = int(config["training"]["eval_every_steps"])
    save_every = int(config["training"]["save_every_steps"])
    global_batch = int(config["training"]["local_batch_size"]) * world_size
    epoch = 0
    interval_start = time.perf_counter()
    interval_start_step = global_step
    torch.cuda.reset_peak_memory_stats(device)

    if global_step == 0:
        run_evaluation(model, validation_dataset, config, output_dir, 0, rank, device)
    while global_step < max_steps:
        sampler.set_epoch(epoch)
        train_dataset.set_epoch(epoch)
        for batch in loader:
            if global_step >= max_steps:
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            inputs = {}
            if "visual_tokens" in batch:
                inputs["visual_tokens"] = batch["visual_tokens"].to(device, non_blocking=True)
            else:
                inputs["video"] = batch["video"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(target_texts=batch["target_text"], **inputs)
                result = bidirectional_infonce(
                    output.z_pred,
                    output.z_target,
                    temperature=float(config["loss"]["temperature"]),
                    distributed_gather=bool(config["loss"]["distributed_gather"]),
                )
            if not bool(torch.isfinite(result.loss)):
                raise FloatingPointError(f"non-finite loss at step {global_step + 1}")
            result.loss.backward()
            if world_size > 1:
                # SDPA backward uses auxiliary CUDA streams; fully retire those
                # kernels before handing gradient storage to NCCL on Blackwell.
                torch.cuda.synchronize(device)
                chunked_all_reduce_gradients(
                    unwrap(model),
                    chunk_elements=int(
                        config["training"].get("gradient_sync_chunk_elements", 1_048_576)
                    ),
                    process_group=gradient_process_group,
                )
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in unwrap(model).parameters() if parameter.requires_grad],
                max_norm=float(config["optimizer"]["grad_clip_norm"]),
            )
            optimizer.step()
            global_step += 1

            if global_step % log_every == 0 or global_step == 1:
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - interval_start
                interval_steps = global_step - interval_start_step
                samples_per_second = interval_steps * global_batch / max(elapsed, 1e-9)
                peak_gib = synchronize_max(torch.cuda.max_memory_allocated(device) / 2**30, device)
                payload = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss_total": float(result.loss.detach()),
                    "loss_pred_to_text": float(result.loss_pred_to_text.detach()),
                    "loss_text_to_pred": float(result.loss_text_to_pred.detach()),
                    "positive_cosine_mean": float(result.positive_cosine_mean.detach()),
                    "negative_cosine_mean": float(result.negative_cosine_mean.detach()),
                    "embedding_std_pred": float(result.embedding_std_pred.detach()),
                    "embedding_std_target": float(result.embedding_std_target.detach()),
                    "grad_norm": float(grad_norm),
                    "lr_main": optimizer.param_groups[0]["lr"],
                    "lr_yencoder": optimizer.param_groups[2]["lr"],
                    "global_batch": global_batch,
                    "samples_seen": global_batch * global_step,
                    "gpu_peak_memory_gib": peak_gib,
                    "samples_per_second": samples_per_second,
                }
                if not all(
                    math.isfinite(value)
                    for key, value in payload.items()
                    if isinstance(value, float) and not key.startswith("lr_")
                ):
                    raise FloatingPointError(f"non-finite diagnostic: {payload}")
                if rank == 0:
                    append_jsonl(train_log, payload)
                    print(json.dumps(payload, sort_keys=True), flush=True)
                interval_start = time.perf_counter()
                interval_start_step = global_step
                torch.cuda.reset_peak_memory_stats(device)

            if global_step % eval_every == 0:
                run_evaluation(
                    model,
                    validation_dataset,
                    config,
                    output_dir,
                    global_step,
                    rank,
                    device,
                )
            if global_step % save_every == 0 and rank == 0:
                save_checkpoint(
                    output_dir / f"checkpoint_step_{global_step:07d}.pt",
                    model=unwrap(model),
                    optimizer=optimizer,
                    global_step=global_step,
                    config=config,
                )
        epoch += 1

    if rank == 0:
        final_path = output_dir / f"checkpoint_step_{global_step:07d}.pt"
        if not final_path.exists():
            save_checkpoint(
                final_path,
                model=unwrap(model),
                optimizer=optimizer,
                global_step=global_step,
                config=config,
            )
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    if faulthandler_interval > 0:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
