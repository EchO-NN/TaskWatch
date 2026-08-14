import os

import pytest
import torch
import torch.distributed as dist

from vljepa_droid.distributed.gather import differentiable_all_gather


@pytest.mark.distributed
def test_remote_rank_contributes_gradient() -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        pytest.skip("run with torchrun --nproc-per-node=2")
    if not dist.is_initialized():
        dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2
    local = torch.full((2, 3), float(rank + 1), requires_grad=True)
    gathered = differentiable_all_gather(local)
    other = 1 - rank
    remote_only_loss = gathered[other * 2 : (other + 1) * 2].sum()
    remote_only_loss.backward()
    torch.testing.assert_close(local.grad, torch.ones_like(local))
    dist.barrier()
    dist.destroy_process_group()
