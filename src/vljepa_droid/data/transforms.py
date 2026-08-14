from __future__ import annotations

import torch
from torchvision.transforms import functional as tvf


def preprocess_video(
    frames: torch.Tensor,
    *,
    resize_short_side: int = 292,
    crop_size: int = 256,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """Deterministic V-JEPA-style preprocessing.

    Args:
        frames: uint8 tensor shaped [T,H,W,3].
    Returns:
        float32 tensor shaped [3,T,crop_size,crop_size].
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected [T,H,W,3] frames, got {tuple(frames.shape)}")
    if frames.dtype != torch.uint8:
        raise ValueError(f"expected uint8 frames, got {frames.dtype}")
    video = frames.permute(0, 3, 1, 2).float().div_(255.0)
    video = tvf.resize(video, resize_short_side, antialias=True)
    video = tvf.center_crop(video, [crop_size, crop_size])
    mean_tensor = torch.tensor(mean, dtype=video.dtype).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, dtype=video.dtype).view(1, 3, 1, 1)
    video = (video - mean_tensor) / std_tensor
    return video.permute(1, 0, 2, 3).contiguous()
