#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

from vljepa_droid.config import load_config
from vljepa_droid.data.dataset import PreparedDroidDataset
from vljepa_droid.models.vjepa21_encoder import VJEPA21Encoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    dataset = PreparedDroidDataset(
        config["paths"]["prepared_dir"],
        split=args.split,
        training=False,
        resize_short_side=int(config["data"]["resize_short_side"]),
        resolution=int(config["data"]["resolution"]),
        mean=tuple(config["data"]["normalize_mean"]),
        std=tuple(config["data"]["normalize_std"]),
    )
    encoder = VJEPA21Encoder(
        vjepa_repo=config["paths"]["vjepa_repo"],
        checkpoint_path=config["paths"]["vjepa_checkpoint"],
        device=device,
    )
    cache_dir = Path(config["paths"]["feature_cache_dir"])
    completed = 0
    for index in tqdm(range(rank, len(dataset), world_size), desc=f"rank {rank} cache"):
        record = dataset.records[index]
        output_path = cache_dir / record["feature_file"]
        if output_path.is_file():
            tokens = torch.load(output_path, map_location="cpu", weights_only=True)
            if tuple(tokens.shape) == (4096, 1664) and tokens.dtype == torch.bfloat16:
                completed += 1
                continue
            raise RuntimeError(f"invalid existing feature cache: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        video = dataset[index]["video"].unsqueeze(0).to(device, non_blocking=True)
        tokens = encoder(video)[0].to(device="cpu", dtype=torch.bfloat16).contiguous()
        if tuple(tokens.shape) != (4096, 1664):
            raise RuntimeError(f"bad raw token shape: {tuple(tokens.shape)}")
        temporary = output_path.with_suffix(output_path.suffix + f".rank{rank}.tmp")
        torch.save(tokens, temporary)
        temporary.replace(output_path)
        completed += 1
    summary = {"rank": rank, "world_size": world_size, "split": args.split, "cached": completed}
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
