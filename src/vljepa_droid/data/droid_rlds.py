from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .sampling import uniform_frame_indices


LANGUAGE_KEYS = (
    "language_instruction",
    "language_instruction_2",
    "language_instruction_3",
)


@dataclass(frozen=True)
class DroidEpisode:
    episode_id: str
    source_path: str
    annotations: tuple[str, ...]
    frames: np.ndarray
    frame_indices: np.ndarray
    total_frames: int


def _to_python_string(value: Any) -> str:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _step_at(batched_steps: dict[str, Any], index: int) -> dict[str, Any]:
    result = {}
    for key, value in batched_steps.items():
        if isinstance(value, dict):
            result[key] = _step_at(value, index)
        else:
            result[key] = value[index]
    return result


def _iter_steps(steps: Any) -> Iterator[dict[str, Any]]:
    if isinstance(steps, dict):
        first_leaf = steps
        while isinstance(first_leaf, dict):
            first_leaf = next(iter(first_leaf.values()))
        total = int(first_leaf.shape[0])
        for index in range(total):
            yield _step_at(steps, index)
        return
    yield from steps


class DroidRLDSReader:
    """Sequential episode reader backed by the downloaded TFDS/RLDS directory."""

    def __init__(
        self,
        rlds_dir: str | Path,
        *,
        camera_key: str = "exterior_image_1_left",
        num_frames: int = 32,
        min_frames: int = 32,
        success_only: bool = True,
    ) -> None:
        self.rlds_dir = Path(rlds_dir)
        self.camera_key = camera_key
        self.num_frames = num_frames
        self.min_frames = min_frames
        self.success_only = success_only
        for required in ("dataset_info.json", "features.json"):
            if not (self.rlds_dir / required).is_file():
                raise FileNotFoundError(self.rlds_dir / required)

    def iter_episodes(self, *, shuffle_files: bool = False) -> Iterator[DroidEpisode]:
        try:
            import tensorflow as tf
            import tensorflow_datasets as tfds
        except ImportError as error:
            raise RuntimeError(
                "DROID RLDS reading requires tensorflow-cpu and tensorflow-datasets"
            ) from error

        tf.config.set_visible_devices([], "GPU")
        builder = tfds.builder_from_directory(str(self.rlds_dir))
        dataset = builder.as_dataset(split="train", shuffle_files=shuffle_files)

        for episode in dataset:
            metadata = episode["episode_metadata"]
            source_path = _to_python_string(metadata["file_path"])
            recording_path = _to_python_string(metadata["recording_folderpath"])
            episode_id = recording_path.rstrip("/").split("/")[-1] or source_path
            if self.success_only and "/success/" not in source_path:
                continue

            all_frames: list[np.ndarray] = []
            annotations: list[str] = []
            for step in _iter_steps(episode["steps"]):
                frame = step["observation"][self.camera_key]
                if hasattr(frame, "numpy"):
                    frame = frame.numpy()
                all_frames.append(np.asarray(frame, dtype=np.uint8))
                if not annotations:
                    annotations = [_to_python_string(step[key]).strip() for key in LANGUAGE_KEYS]

            total_frames = len(all_frames)
            non_empty = tuple(text for text in annotations if text)
            if total_frames < self.min_frames or not non_empty:
                continue
            indices = uniform_frame_indices(total_frames, self.num_frames)
            selected = np.stack([all_frames[int(index)] for index in indices], axis=0)
            yield DroidEpisode(
                episode_id=episode_id,
                source_path=source_path,
                annotations=non_empty,
                frames=selected,
                frame_indices=indices,
                total_frames=total_frames,
            )
