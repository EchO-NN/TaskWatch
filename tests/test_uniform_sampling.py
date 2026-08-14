import numpy as np
import pytest

from vljepa_droid.data.sampling import uniform_frame_indices


def test_uniform_sampling_contract() -> None:
    first = uniform_frame_indices(100, 32)
    second = uniform_frame_indices(100, 32)
    assert first.shape == (32,)
    assert first[0] == 0
    assert first[-1] == 99
    assert np.all(first[1:] >= first[:-1])
    np.testing.assert_array_equal(first, second)


def test_uniform_sampling_rejects_empty() -> None:
    with pytest.raises(ValueError):
        uniform_frame_indices(0, 32)
