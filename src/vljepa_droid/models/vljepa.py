from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from .qwen3_predictor import Qwen3VLJEPAPredictor
from .vjepa21_encoder import VJEPA21Encoder
from .y_encoder import VLJEPAYEncoder


@dataclass
class VLJEPAOutput:
    z_pred: torch.Tensor
    z_target: torch.Tensor | None


class VLJEPAModel(nn.Module):
    def __init__(
        self,
        *,
        predictor: Qwen3VLJEPAPredictor,
        y_encoder: VLJEPAYEncoder,
        x_encoder: VJEPA21Encoder | None = None,
    ) -> None:
        super().__init__()
        self.x_encoder = x_encoder
        self.predictor = predictor
        self.y_encoder = y_encoder

    def train(self, mode: bool = True):
        super().train(mode)
        if self.x_encoder is not None:
            self.x_encoder.eval()
        return self

    def encode_video(
        self,
        *,
        video: torch.Tensor | None = None,
        visual_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (video is None) == (visual_tokens is None):
            raise ValueError("provide exactly one of video or visual_tokens")
        if visual_tokens is None:
            if self.x_encoder is None:
                raise RuntimeError("online video input requires an x_encoder")
            visual_tokens = self.x_encoder(video)
        return self.predictor(visual_tokens)

    def forward(
        self,
        *,
        target_texts: Sequence[str] | None = None,
        video: torch.Tensor | None = None,
        visual_tokens: torch.Tensor | None = None,
    ) -> VLJEPAOutput:
        z_pred = self.encode_video(video=video, visual_tokens=visual_tokens)
        z_target = self.y_encoder(target_texts) if target_texts is not None else None
        return VLJEPAOutput(z_pred=z_pred, z_target=z_target)
