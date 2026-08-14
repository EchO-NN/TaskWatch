from __future__ import annotations

import os

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    value = torch.tensor([rank + 1.0], device=f"cuda:{local_rank}")
    dist.all_reduce(value)
    if value.item() != 3.0:
        raise RuntimeError(f"bad NCCL all-reduce result on rank {rank}: {value.item()}")
    print(f"rank={rank} nccl_sum={value.item()}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
