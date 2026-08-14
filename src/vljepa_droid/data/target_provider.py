from __future__ import annotations

import random
from typing import Protocol, Sequence


class TargetTextProvider(Protocol):
    def get_text(
        self,
        episode_id: str,
        annotations: Sequence[str],
        *,
        index: int,
        epoch: int,
        training: bool,
    ) -> str: ...


class DroidNativeTargetProvider:
    """Select one non-empty native DROID annotation per episode access."""

    def __init__(self, seed: int = 239) -> None:
        self.seed = seed

    def get_text(
        self,
        episode_id: str,
        annotations: Sequence[str],
        *,
        index: int,
        epoch: int,
        training: bool,
    ) -> str:
        del episode_id
        texts = [text.strip() for text in annotations if text and text.strip()]
        if not texts:
            raise ValueError("DROID episode has no non-empty native annotation")
        if not training:
            return texts[0]
        rng = random.Random(self.seed + epoch * 1_000_003 + index * 97)
        return rng.choice(texts)
