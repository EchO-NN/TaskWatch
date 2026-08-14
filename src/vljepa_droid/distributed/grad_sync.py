from __future__ import annotations

import torch.distributed as dist
import torch.nn as nn


def chunked_all_reduce_gradients(
    model: nn.Module,
    *,
    chunk_elements: int = 1_048_576,
    process_group=None,
) -> None:
    """Synchronous data-parallel gradient averaging in bounded chunks.

    A Gloo process group stages each chunk through CPU. This is mathematically
    the same parameter-gradient average as DDP, while avoiding the CUDA 700 that
    this host's NCCL raises on model-sized parameter-gradient collectives.
    """
    if not dist.is_available() or not dist.is_initialized():
        return
    if chunk_elements < 1:
        raise ValueError("chunk_elements must be positive")
    world_size = dist.get_world_size(process_group)
    backend = dist.get_backend(process_group)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            raise RuntimeError(f"trainable parameter has no gradient: {name}")
        if not parameter.grad.is_contiguous():
            raise RuntimeError(f"non-contiguous gradient is unsupported: {name}")
        flattened = parameter.grad.view(-1)
        for chunk in flattened.split(chunk_elements):
            if backend == "gloo" and chunk.device.type != "cpu":
                staged = chunk.detach().to(device="cpu")
                dist.all_reduce(staged, op=dist.ReduceOp.SUM, group=process_group)
                staged.div_(world_size)
                chunk.copy_(staged.to(device=chunk.device))
            else:
                dist.all_reduce(chunk, op=dist.ReduceOp.SUM, group=process_group)
                chunk.div_(world_size)
