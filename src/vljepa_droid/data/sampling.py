from __future__ import annotations

import numpy as np


def uniform_frame_indices(total_frames: int, num_frames: int = 32) -> np.ndarray:
    """Round a full-trajectory linspace while preserving both endpoints."""
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    indices = np.rint(np.linspace(0, total_frames - 1, num_frames)).astype(np.int64)
    if len(indices) != num_frames:
        raise AssertionError("uniform sampler returned the wrong length")
    if indices[0] != 0 or indices[-1] != total_frames - 1:
        raise AssertionError("uniform sampler must preserve trajectory endpoints")
    if np.any(indices[1:] < indices[:-1]):
        raise AssertionError("uniform sampler must be monotonic")
    return indices
