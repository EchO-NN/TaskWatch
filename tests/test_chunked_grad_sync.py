import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from vljepa_droid.distributed.grad_sync import chunked_all_reduce_gradients


@pytest.mark.distributed
def test_chunked_gradient_average() -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        pytest.skip("run with torchrun --nproc-per-node=2")
    if not dist.is_initialized():
        dist.init_process_group("gloo")
    rank = dist.get_rank()
    model = nn.Linear(4, 3, bias=False)
    model(torch.full((2, 4), float(rank + 1))).sum().backward()
    chunked_all_reduce_gradients(model, chunk_elements=3)
    expected = torch.full_like(model.weight.grad, 3.0)
    torch.testing.assert_close(model.weight.grad, expected)
    dist.barrier()
    dist.destroy_process_group()
