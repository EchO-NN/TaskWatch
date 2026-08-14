from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .target_provider import DroidNativeTargetProvider, TargetTextProvider
from .transforms import preprocess_video


class PreparedDroidDataset(Dataset):
    """Random-access view of prepared 32-frame DROID samples or raw token cache."""

    def __init__(
        self,
        prepared_dir: str | Path,
        *,
        split: str,
        training: bool,
        feature_cache_dir: str | Path | None = None,
        resize_short_side: int = 292,
        resolution: int = 256,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        target_provider: TargetTextProvider | None = None,
        seed: int = 239,
    ) -> None:
        self.prepared_dir = Path(prepared_dir)
        self.feature_cache_dir = Path(feature_cache_dir) if feature_cache_dir else None
        self.training = training
        self.resize_short_side = resize_short_side
        self.resolution = resolution
        self.mean = mean
        self.std = std
        self.target_provider = target_provider or DroidNativeTargetProvider(seed)
        self.epoch = 0

        manifest_path = self.prepared_dir / f"{split}.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        with manifest_path.open("r", encoding="utf-8") as manifest_file:
            self.records = [json.loads(line) for line in manifest_file if line.strip()]
        if not self.records:
            raise ValueError(f"empty DROID manifest: {manifest_path}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        target_text = self.target_provider.get_text(
            record["episode_id"],
            record["annotations"],
            index=index,
            epoch=self.epoch,
            training=self.training,
        )
        result = {
            "target_text": target_text,
            "episode_id": record["episode_id"],
            "frame_indices": torch.tensor(record["frame_indices"], dtype=torch.long),
        }
        if self.feature_cache_dir is not None:
            feature_path = self.feature_cache_dir / record["feature_file"]
            visual_tokens = torch.load(feature_path, map_location="cpu", weights_only=True)
            result["visual_tokens"] = visual_tokens
        else:
            frames = np.load(self.prepared_dir / record["frames_file"], mmap_mode="r")
            frames_tensor = torch.from_numpy(np.array(frames, copy=True))
            result["video"] = preprocess_video(
                frames_tensor,
                resize_short_side=self.resize_short_side,
                crop_size=self.resolution,
                mean=self.mean,
                std=self.std,
            )
        return result
