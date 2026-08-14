from __future__ import annotations

import hashlib
from collections.abc import Sequence


def deterministic_validation_split(
    records: Sequence[dict],
    *,
    validation_count: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Select an exact, order-independent validation set by source-path hash."""
    if validation_count < 1:
        raise ValueError("validation_count must be positive")
    if validation_count >= len(records):
        raise ValueError(
            f"validation_count={validation_count} requires more than "
            f"{validation_count} records, found {len(records)}"
        )
    source_paths = [str(record["source_path"]) for record in records]
    if len(source_paths) != len(set(source_paths)):
        raise ValueError("source_path values must be unique")

    def rank(record: dict) -> tuple[bytes, str]:
        source_path = str(record["source_path"])
        digest = hashlib.sha256(f"{seed}\0{source_path}".encode()).digest()
        return digest, source_path

    validation_sources = {
        str(record["source_path"]) for record in sorted(records, key=rank)[:validation_count]
    }
    train = [record for record in records if str(record["source_path"]) not in validation_sources]
    validation = sorted(
        (record for record in records if str(record["source_path"]) in validation_sources),
        key=rank,
    )
    return train, validation
