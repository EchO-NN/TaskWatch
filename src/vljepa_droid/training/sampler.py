from __future__ import annotations

import itertools
from collections.abc import Iterator

from torch.utils.data import BatchSampler, Sampler


class OffsetBatchSampler(Sampler[list[int]]):
    """Skip already-consumed batches without decoding their samples."""

    def __init__(self, batch_sampler: BatchSampler) -> None:
        self.batch_sampler = batch_sampler
        self.start_batch = 0

    def set_start_batch(self, start_batch: int) -> None:
        if start_batch < 0 or start_batch > len(self.batch_sampler):
            raise ValueError(f"invalid start batch {start_batch}")
        self.start_batch = start_batch

    def __iter__(self) -> Iterator[list[int]]:
        return itertools.islice(iter(self.batch_sampler), self.start_batch, None)

    def __len__(self) -> int:
        return len(self.batch_sampler) - self.start_batch


def resume_data_position(global_step: int, steps_per_epoch: int) -> tuple[int, int]:
    if global_step < 0:
        raise ValueError(f"invalid global step {global_step}")
    if steps_per_epoch < 1:
        raise ValueError(f"invalid steps per epoch {steps_per_epoch}")
    return divmod(global_step, steps_per_epoch)
