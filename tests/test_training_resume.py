import pytest
from torch.utils.data import BatchSampler, SequentialSampler

from vljepa_droid.training.sampler import OffsetBatchSampler, resume_data_position


def test_resume_data_position_uses_global_step() -> None:
    assert resume_data_position(10_000, 4_600) == (2, 800)
    assert resume_data_position(13_800, 4_600) == (3, 0)


def test_offset_batch_sampler_skips_indices_without_loading_samples() -> None:
    base = BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True)
    sampler = OffsetBatchSampler(base)
    sampler.set_start_batch(2)

    assert len(sampler) == 1
    assert list(sampler) == [[6, 7, 8]]


@pytest.mark.parametrize("start_batch", [-1, 4])
def test_offset_batch_sampler_rejects_invalid_offset(start_batch: int) -> None:
    base = BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True)
    sampler = OffsetBatchSampler(base)

    with pytest.raises(ValueError):
        sampler.set_start_batch(start_batch)
