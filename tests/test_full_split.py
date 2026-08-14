import random

import pytest

from vljepa_droid.data.split import deterministic_validation_split


def make_records(count: int) -> list[dict]:
    return [
        {"episode_id": f"episode-{index}", "source_path": f"/success/{index}"}
        for index in range(count)
    ]


def test_deterministic_validation_split_is_exact_disjoint_and_order_independent() -> None:
    records = make_records(100)
    train, validation = deterministic_validation_split(records, validation_count=17, seed=239)
    shuffled = records.copy()
    random.Random(7).shuffle(shuffled)
    shuffled_train, shuffled_validation = deterministic_validation_split(
        shuffled, validation_count=17, seed=239
    )
    train_sources = {record["source_path"] for record in train}
    validation_sources = {record["source_path"] for record in validation}
    assert len(train) == 83
    assert len(validation) == 17
    assert train_sources.isdisjoint(validation_sources)
    assert train_sources | validation_sources == {record["source_path"] for record in records}
    assert validation_sources == {record["source_path"] for record in shuffled_validation}
    assert train_sources == {record["source_path"] for record in shuffled_train}


def test_deterministic_validation_split_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="requires more"):
        deterministic_validation_split(make_records(2), validation_count=2, seed=1)
    duplicate = make_records(3)
    duplicate[2]["source_path"] = duplicate[0]["source_path"]
    with pytest.raises(ValueError, match="unique"):
        deterministic_validation_split(duplicate, validation_count=1, seed=1)
