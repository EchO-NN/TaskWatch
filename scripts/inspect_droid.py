#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from vljepa_droid.config import load_config
from vljepa_droid.data.droid_rlds import DroidRLDSReader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    config = load_config(args.config)
    reader = DroidRLDSReader(
        config["paths"]["rlds_dir"],
        camera_key=config["data"]["camera_key"],
        num_frames=int(config["data"]["num_frames"]),
        min_frames=int(config["data"]["num_frames"]),
        success_only=bool(config["data"]["success_only"]),
    )
    for index, episode in enumerate(reader.iter_episodes()):
        print(
            json.dumps(
                {
                    "episode_id": episode.episode_id,
                    "source_path": episode.source_path,
                    "annotations": episode.annotations,
                    "total_frames": episode.total_frames,
                    "selected_shape": episode.frames.shape,
                    "frame_indices": episode.frame_indices.tolist(),
                },
                ensure_ascii=False,
            )
        )
        if index + 1 >= args.episodes:
            break


if __name__ == "__main__":
    main()
