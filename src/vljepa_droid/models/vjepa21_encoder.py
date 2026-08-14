from __future__ import annotations

import gc
import sys
from pathlib import Path

import torch
import torch.nn as nn


def _clean_encoder_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        key = key.removeprefix("module.").removeprefix("backbone.")
        cleaned[key] = value
    return cleaned


def _build_meta_backbone(vjepa_repo: str | Path) -> nn.Module:
    repo = str(Path(vjepa_repo).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from app.vjepa_2_1.models import vision_transformer

    original_linspace = torch.linspace

    def cpu_linspace(*args, **kwargs):
        kwargs["device"] = "cpu"
        return original_linspace(*args, **kwargs)

    torch.linspace = cpu_linspace
    try:
        with torch.device("meta"):
            backbone = vision_transformer.vit_gigantic_xformers(
                patch_size=16,
                img_size=(256, 256),
                num_frames=32,
                tubelet_size=2,
                use_sdpa=True,
                use_SiLU=False,
                wide_SiLU=True,
                uniform_power=False,
                use_rope=True,
                img_temporal_dim_size=1,
                interpolate_rope=True,
            )
    finally:
        torch.linspace = original_linspace
    return backbone


class VJEPA21Encoder(nn.Module):
    """Frozen V-JEPA 2.1 ViT-G/16 2B producing raw visual tokens."""

    hidden_dim = 1664
    expected_tokens = 4096

    def __init__(
        self,
        *,
        vjepa_repo: str | Path,
        checkpoint_path: str | Path,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        encoder_state = _clean_encoder_state_dict(checkpoint["target_encoder"])
        backbone = _build_meta_backbone(vjepa_repo)
        incompatible = backbone.load_state_dict(encoder_state, strict=True, assign=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"strict V-JEPA load failed: {incompatible}")
        del checkpoint, encoder_state
        gc.collect()

        self.backbone = backbone.to(device=device, dtype=dtype)
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(False)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5 or tuple(video.shape[1:]) != (3, 32, 256, 256):
            raise ValueError(f"V-JEPA input must be [B,3,32,256,256], got {tuple(video.shape)}")
        device = next(self.backbone.parameters()).device
        video = video.to(device=device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            visual_tokens = self.backbone(video)
        expected = (video.shape[0], self.expected_tokens, self.hidden_dim)
        if tuple(visual_tokens.shape) != expected:
            raise RuntimeError(
                f"V-JEPA token contract failed: expected {expected}, "
                f"got {tuple(visual_tokens.shape)}"
            )
        return visual_tokens
