from __future__ import annotations

import torch

from vljepa_droid.models.qwen3_predictor import Qwen3VLJEPAPredictor
from vljepa_droid.models.vjepa21_encoder import VJEPA21Encoder
from vljepa_droid.models.vljepa import VLJEPAModel
from vljepa_droid.models.y_encoder import VLJEPAYEncoder


def build_stage1_model(
    config: dict,
    *,
    device: torch.device,
    include_x_encoder: bool,
) -> VLJEPAModel:
    if config["training"]["dtype"] != "bfloat16":
        raise ValueError("strict baseline currently requires bfloat16")
    predictor = Qwen3VLJEPAPredictor(
        config["paths"]["qwen_model"],
        selected_layers=config["predictor"]["selected_layers"],
        shared_dim=int(config["predictor"]["shared_embedding_dim"]),
        dtype=torch.bfloat16,
        gradient_checkpointing=bool(config["predictor"]["gradient_checkpointing"]),
    )
    y_encoder = VLJEPAYEncoder(
        config["paths"]["y_encoder_model"],
        max_length=int(config["y_encoder"]["max_length"]),
        shared_dim=int(config["predictor"]["shared_embedding_dim"]),
        dtype=torch.bfloat16,
        gradient_checkpointing=True,
    )
    x_encoder = None
    if include_x_encoder:
        x_encoder = VJEPA21Encoder(
            vjepa_repo=config["paths"]["vjepa_repo"],
            checkpoint_path=config["paths"]["vjepa_checkpoint"],
            device=device,
            dtype=torch.bfloat16,
        )
    return VLJEPAModel(
        predictor=predictor,
        y_encoder=y_encoder,
        x_encoder=x_encoder,
    ).to(device)
