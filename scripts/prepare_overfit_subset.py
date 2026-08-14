#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

from vljepa_droid.config import load_config
from vljepa_droid.data.droid_rlds import DroidRLDSReader


def safe_stem(episode_id: str, source_path: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", episode_id).strip("._")
    digest = hashlib.sha1(source_path.encode("utf-8")).hexdigest()[:12]
    return f"{cleaned[:80] or 'episode'}_{digest}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = Path(config["paths"]["prepared_dir"])
    train_count = int(config["data"]["train_episodes"])
    validation_count = int(config["data"]["validation_episodes"])
    required = train_count + validation_count
    for split in ("train", "validation"):
        manifest = output_dir / f"{split}.jsonl"
        if manifest.exists():
            raise FileExistsError(
                f"refusing to overwrite prepared manifest; move it explicitly: {manifest}"
            )
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    reader = DroidRLDSReader(
        config["paths"]["rlds_dir"],
        camera_key=config["data"]["camera_key"],
        num_frames=int(config["data"]["num_frames"]),
        min_frames=int(config["data"]["num_frames"]),
        success_only=bool(config["data"]["success_only"]),
    )
    handles = {
        split: (output_dir / f"{split}.jsonl").open("x", encoding="utf-8")
        for split in ("train", "validation")
    }
    stats = {"train": 0, "validation": 0, "camera": config["data"]["camera_key"]}
    try:
        with tqdm(total=required, desc="preparing DROID episodes") as progress:
            for episode in reader.iter_episodes(shuffle_files=False):
                index = stats["train"] + stats["validation"]
                if index >= required:
                    break
                split = "train" if stats["train"] < train_count else "validation"
                stem = safe_stem(episode.episode_id, episode.source_path)
                relative_frames = f"{split}/{stem}.npy"
                np.save(output_dir / relative_frames, episode.frames, allow_pickle=False)
                record = {
                    "episode_id": episode.episode_id,
                    "source_path": episode.source_path,
                    "annotations": list(episode.annotations),
                    "total_frames": episode.total_frames,
                    "frame_indices": episode.frame_indices.tolist(),
                    "frames_file": relative_frames,
                    "feature_file": f"{split}/{stem}.pt",
                }
                handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                handles[split].flush()
                stats[split] += 1
                progress.update(1)
    finally:
        for handle in handles.values():
            handle.close()
    if stats["train"] != train_count or stats["validation"] != validation_count:
        raise RuntimeError(f"not enough eligible episodes: {stats}")
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
