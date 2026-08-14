from __future__ import annotations

import os

import torch
import torch.distributed as dist

from vljepa_droid.losses.infonce import bidirectional_infonce


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.manual_seed(100 + rank)
    prediction = torch.randn(2, 1536, device=f"cuda:{local_rank}", requires_grad=True)
    target = torch.randn(2, 1536, device=f"cuda:{local_rank}", requires_grad=True)
    result = bidirectional_infonce(prediction, target, distributed_gather=True)
    result.loss.backward()
    if prediction.grad is None or target.grad is None:
        raise RuntimeError("differentiable NCCL gather lost a gradient")
    print(
        f"rank={rank} loss={result.loss.item():.6f} "
        f"pred_grad={prediction.grad.norm().item():.6f} "
        f"target_grad={target.grad.norm().item():.6f}",
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
