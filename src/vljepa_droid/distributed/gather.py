from __future__ import annotations

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def differentiable_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """Gather equal local batches while preserving gradients on every rank."""
    if not is_distributed():
        return tensor
    from torch.distributed.nn.functional import all_gather

    gathered = all_gather(tensor)
    return torch.cat(tuple(gathered), dim=0)
