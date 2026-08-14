#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

from vljepa_droid.config import load_config
from vljepa_droid.data.droid_rlds import DroidRLDSReader
from vljepa_droid.data.split import deterministic_validation_split

from prepare_overfit_subset import safe_stem


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def write_jsonl_atomic(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_total_examples(rlds_dir: Path) -> int | None:
    info = json.loads((rlds_dir / "dataset_info.json").read_text(encoding="utf-8"))
    splits = [item for item in info.get("splits", []) if item.get("name") == "train"]
    if not splits or "shardLengths" not in splits[0]:
        return None
    return sum(int(value) for value in splits[0]["shardLengths"])


def assert_disk_headroom(path: Path, minimum_free_gib: float) -> None:
    free_gib = shutil.disk_usage(path).free / 2**30
    if free_gib < minimum_free_gib:
        raise OSError(
            f"free space {free_gib:.1f} GiB is below the required "
            f"{minimum_free_gib:.1f} GiB safety floor"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--minimum-free-gib", type=float, default=250.0)
    parser.add_argument(
        "--max-eligible-episodes",
        type=int,
        default=None,
        help="development-only bound; omit for a complete DROID scan",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if config["data"].get("train_episodes") is not None:
        raise ValueError("full preparation requires data.train_episodes: null")

    output_dir = Path(config["paths"]["prepared_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    assert_disk_headroom(output_dir, args.minimum_free_gib)

    train_manifest = output_dir / "train.jsonl"
    validation_manifest = output_dir / "validation.jsonl"
    all_manifest = output_dir / "all.jsonl"
    partial_manifest = output_dir / "all.jsonl.partial"
    stats_path = output_dir / "stats.json"
    final_paths = (train_manifest, validation_manifest, all_manifest, stats_path)
    if all(path.exists() for path in final_paths):
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        print(json.dumps({"status": "already_complete", **stats}, indent=2))
        return
    if any(path.exists() for path in (train_manifest, validation_manifest, all_manifest)):
        raise FileExistsError(
            f"incomplete final manifests found; inspect them before retrying: {output_dir}"
        )

    records = read_jsonl(partial_manifest)
    seen_sources = {str(record["source_path"]) for record in records}
    if len(records) != len(seen_sources):
        raise ValueError(f"duplicate source paths in {partial_manifest}")
    for record in records:
        frames_path = output_dir / record["frames_file"]
        if not frames_path.is_file():
            raise FileNotFoundError(f"partial manifest references missing frames: {frames_path}")

    partial_handle = partial_manifest.open("a", encoding="utf-8")
    reader = DroidRLDSReader(
        config["paths"]["rlds_dir"],
        camera_key=config["data"]["camera_key"],
        num_frames=int(config["data"]["num_frames"]),
        min_frames=int(config["data"]["num_frames"]),
        success_only=bool(config["data"]["success_only"]),
    )
    total_examples = dataset_total_examples(Path(config["paths"]["rlds_dir"]))
    progress = tqdm(
        total=total_examples,
        initial=len(records),
        desc="materializing eligible DROID episodes",
        unit="episode",
    )
    try:
        for episode in reader.iter_episodes(shuffle_files=False):
            source_path = str(episode.source_path)
            if source_path in seen_sources:
                continue
            if len(records) % 1000 == 0:
                assert_disk_headroom(output_dir, args.minimum_free_gib)
            stem = safe_stem(episode.episode_id, source_path)
            relative_frames = f"samples/{stem}.npy"
            frames_path = output_dir / relative_frames
            temporary_frames = frames_path.with_suffix(".tmp.npy")
            if frames_path.exists() or temporary_frames.exists():
                raise FileExistsError(
                    f"refusing to replace existing prepared frames: {frames_path}"
                )
            np.save(temporary_frames, episode.frames, allow_pickle=False)
            os.replace(temporary_frames, frames_path)
            record = {
                "episode_id": episode.episode_id,
                "source_path": source_path,
                "annotations": list(episode.annotations),
                "total_frames": episode.total_frames,
                "frame_indices": episode.frame_indices.tolist(),
                "frames_file": relative_frames,
                "feature_file": f"samples/{stem}.pt",
            }
            partial_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            partial_handle.flush()
            records.append(record)
            seen_sources.add(source_path)
            progress.update(1)
            if (
                args.max_eligible_episodes is not None
                and len(records) >= args.max_eligible_episodes
            ):
                break
    finally:
        progress.close()
        partial_handle.close()

    if args.max_eligible_episodes is not None:
        print(
            json.dumps(
                {
                    "status": "bounded_partial",
                    "eligible_episodes": len(records),
                    "partial_manifest": str(partial_manifest),
                },
                indent=2,
            )
        )
        return

    validation_count = int(config["data"]["validation_episodes"])
    train_records, validation_records = deterministic_validation_split(
        records,
        validation_count=validation_count,
        seed=int(config["experiment"]["seed"]),
    )
    write_jsonl_atomic(train_manifest, train_records)
    write_jsonl_atomic(validation_manifest, validation_records)
    os.replace(partial_manifest, all_manifest)
    stats = {
        "eligible_episodes": len(records),
        "train_episodes": len(train_records),
        "validation_episodes": len(validation_records),
        "camera": config["data"]["camera_key"],
        "num_frames": int(config["data"]["num_frames"]),
        "split_method": "lowest seeded SHA-256 source_path hashes",
        "seed": int(config["experiment"]["seed"]),
        "train_manifest_sha256": sha256_file(train_manifest),
        "validation_manifest_sha256": sha256_file(validation_manifest),
        "free_gib_after": shutil.disk_usage(output_dir).free / 2**30,
    }
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", **stats}, indent=2))


if __name__ == "__main__":
    main()
