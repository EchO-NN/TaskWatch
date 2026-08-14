from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class VLJEPAYEncoder(nn.Module):
    """Trainable EmbeddingGemma-300M semantic target encoder."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        max_length: int = 512,
        shared_dim: int = 1536,
        dtype: torch.dtype = torch.bfloat16,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        self.backbone = AutoModel.from_pretrained(
            str(model_path),
            local_files_only=True,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        if self.backbone.config.hidden_size != 768:
            raise ValueError(
                f"EmbeddingGemma output width must be 768, got {self.backbone.config.hidden_size}"
            )
        self.backbone.config.use_cache = False
        if gradient_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()
        self.target_head = nn.Linear(768, shared_dim, bias=True)

    def forward(self, texts: Sequence[str]) -> torch.Tensor:
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("target texts must be non-empty")
        device = next(self.backbone.parameters()).device
        tokens = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)
        output = self.backbone(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
            use_cache=False,
        )
        hidden = output.last_hidden_state
        mask = tokens.attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return F.normalize(self.target_head(pooled).float(), dim=-1)
