from __future__ import annotations

import torch

from vljepa_droid.models.vljepa import VLJEPAModel


def build_optimizer(model: VLJEPAModel, config: dict) -> torch.optim.Optimizer:
    optimizer_config = config["optimizer"]
    main_lr = float(optimizer_config["main_lr"])
    y_encoder_lr = float(optimizer_config["y_encoder_lr"])

    predictor_parameters = [
        parameter for parameter in model.predictor.parameters() if parameter.requires_grad
    ]
    target_head_parameters = [
        parameter
        for parameter in model.y_encoder.target_head.parameters()
        if parameter.requires_grad
    ]
    y_backbone_parameters = [
        parameter for parameter in model.y_encoder.backbone.parameters() if parameter.requires_grad
    ]
    if not predictor_parameters or not target_head_parameters or not y_backbone_parameters:
        raise RuntimeError("strict Stage 1 optimizer groups must all be non-empty")

    parameter_groups = [
        {"params": predictor_parameters, "lr": main_lr, "group_name": "predictor"},
        {"params": target_head_parameters, "lr": main_lr, "group_name": "target_head"},
        {
            "params": y_backbone_parameters,
            "lr": y_encoder_lr,
            "group_name": "y_encoder_backbone",
        },
    ]
    return torch.optim.AdamW(
        parameter_groups,
        betas=tuple(float(value) for value in optimizer_config["betas"]),
        eps=float(optimizer_config["eps"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
