#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from vljepa_droid.config import load_config
from vljepa_droid.data.dataset import PreparedDroidDataset
from vljepa_droid.evaluation.runner import evaluate_model, write_retrieval_metrics
from vljepa_droid.models.factory import build_stage1_model
from vljepa_droid.training.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    parser.add_argument("--ranked-examples", type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    use_cache = bool(config["training"]["use_feature_cache"])
    model = build_stage1_model(config, device=device, include_x_encoder=not use_cache)
    step = load_checkpoint(args.checkpoint, model=model)
    dataset = PreparedDroidDataset(
        config["paths"]["prepared_dir"],
        split="validation",
        training=False,
        feature_cache_dir=config["paths"]["feature_cache_dir"] if use_cache else None,
        resize_short_side=int(config["data"]["resize_short_side"]),
        resolution=int(config["data"]["resolution"]),
        mean=tuple(config["data"]["normalize_mean"]),
        std=tuple(config["data"]["normalize_std"]),
        seed=int(config["experiment"]["seed"]),
    )
    metrics = evaluate_model(
        model,
        dataset,
        device=device,
        batch_size=int(config["training"]["local_batch_size"]),
        num_workers=int(config["training"]["num_workers"]),
        ranked_example_count=args.ranked_examples,
    )
    metrics["step"] = step
    output = (
        Path(args.output) if args.output else Path(args.checkpoint).with_suffix(".retrieval.json")
    )
    write_retrieval_metrics(output, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
