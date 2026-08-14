from __future__ import annotations

import gc
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM


class Qwen3VLJEPAPredictor(nn.Module):
    """Qwen3-4B layers 28--35 used as a non-causal VL-JEPA predictor."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        selected_layers: Sequence[int] = tuple(range(28, 36)),
        visual_dim: int = 1664,
        shared_dim: int = 1536,
        dtype: torch.dtype = torch.bfloat16,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        selected_layers = tuple(int(index) for index in selected_layers)
        if selected_layers != tuple(range(28, 36)):
            raise ValueError("strict baseline requires Qwen layers 28..35 exactly")

        full = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            local_files_only=True,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        base = full.model
        if base.config.hidden_size != 2560 or len(base.layers) != 36:
            raise ValueError(
                f"unexpected Qwen3-4B config: hidden={base.config.hidden_size}, "
                f"layers={len(base.layers)}"
            )

        self.hidden_dim = base.config.hidden_size
        self.selected_layer_indices = selected_layers
        self.embed_tokens = base.embed_tokens
        self.embed_tokens.requires_grad_(False)
        self.layers = nn.ModuleList([base.layers[index] for index in selected_layers])
        self.norm = base.norm
        self.rotary_emb = base.rotary_emb
        self.vision_projection = nn.Linear(visual_dim, self.hidden_dim, bias=True)
        self.prediction_head = nn.Linear(self.hidden_dim, shared_dim, bias=True)
        self.gradient_checkpointing = gradient_checkpointing

        for layer in self.layers:
            layer.self_attn.is_causal = False

        base.layers = nn.ModuleList()
        base.embed_tokens = None
        base.norm = None
        base.rotary_emb = None
        del full, base
        gc.collect()

    @staticmethod
    def _padding_mask(valid_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor | None:
        if bool(valid_mask.bool().all()):
            return None
        negative = torch.finfo(dtype).min
        additive = (~valid_mask.bool()).to(dtype) * negative
        length = valid_mask.shape[1]
        return additive[:, None, None, :].expand(-1, 1, length, -1).contiguous()

    def encode_sequence(
        self,
        visual_tokens: torch.Tensor,
        query_input_ids: torch.Tensor | None = None,
        query_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if visual_tokens.ndim != 3 or visual_tokens.shape[-1] != 1664:
            raise ValueError(f"expected [B,N,1664], got {tuple(visual_tokens.shape)}")
        hidden = self.vision_projection(visual_tokens)
        batch_size, visual_length, _ = hidden.shape

        valid_mask = torch.ones((batch_size, visual_length), device=hidden.device, dtype=torch.bool)
        if query_input_ids is not None:
            if query_attention_mask is None:
                raise ValueError("query_attention_mask is required with query_input_ids")
            query_hidden = self.embed_tokens(query_input_ids)
            hidden = torch.cat([hidden, query_hidden], dim=1)
            valid_mask = torch.cat(
                [valid_mask, query_attention_mask.to(device=hidden.device, dtype=torch.bool)],
                dim=1,
            )

        sequence_length = hidden.shape[1]
        position_ids = torch.arange(sequence_length, device=hidden.device).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, -1)
        position_embeddings = self.rotary_emb(hidden, position_ids)
        attention_mask = self._padding_mask(valid_mask, hidden.dtype)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training and hidden.requires_grad:

                def layer_forward(states, current_layer=layer):
                    return current_layer(
                        states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=None,
                        use_cache=False,
                        position_embeddings=position_embeddings,
                        is_causal=False,
                    )

                hidden = checkpoint(layer_forward, hidden, use_reentrant=False)
            else:
                hidden = layer(
                    hidden,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    position_embeddings=position_embeddings,
                    is_causal=False,
                )
            if isinstance(hidden, tuple):
                hidden = hidden[0]

        return self.norm(hidden), valid_mask

    def forward(
        self,
        visual_tokens: torch.Tensor,
        query_input_ids: torch.Tensor | None = None,
        query_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden, valid_mask = self.encode_sequence(
            visual_tokens,
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
        )
        mask = valid_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return F.normalize(self.prediction_head(pooled).float(), dim=-1)
