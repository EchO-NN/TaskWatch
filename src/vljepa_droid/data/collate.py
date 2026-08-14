from __future__ import annotations

import torch


def collate_droid_samples(samples: list[dict]) -> dict:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    batch = {
        "target_text": [sample["target_text"] for sample in samples],
        "episode_id": [sample["episode_id"] for sample in samples],
        "frame_indices": torch.stack([sample["frame_indices"] for sample in samples]),
    }
    if "video" in samples[0]:
        batch["video"] = torch.stack([sample["video"] for sample in samples])
    if "visual_tokens" in samples[0]:
        batch["visual_tokens"] = torch.stack([sample["visual_tokens"] for sample in samples])
    return batch
