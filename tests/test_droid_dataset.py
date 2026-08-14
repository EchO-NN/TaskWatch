import json

import numpy as np

from vljepa_droid.data.dataset import PreparedDroidDataset


def test_prepared_droid_dataset_contract(tmp_path) -> None:
    frames_dir = tmp_path / "train"
    frames_dir.mkdir()
    frames = np.zeros((32, 64, 80, 3), dtype=np.uint8)
    np.save(frames_dir / "sample.npy", frames)
    record = {
        "episode_id": "sample-episode",
        "annotations": ["pick cup", "grasp cup", "lift the cup"],
        "frame_indices": list(range(32)),
        "frames_file": "train/sample.npy",
        "feature_file": "train/sample.pt",
    }
    (tmp_path / "train.jsonl").write_text(json.dumps(record) + "\n")

    dataset = PreparedDroidDataset(tmp_path, split="train", training=True)
    sample = dataset[0]
    assert sample["video"].shape == (3, 32, 256, 256)
    assert sample["target_text"] in record["annotations"]
    assert sample["episode_id"] == "sample-episode"
    dataset.set_epoch(1)
    assert dataset[0]["target_text"] in record["annotations"]

    validation = PreparedDroidDataset(tmp_path, split="train", training=False)
    assert validation[0]["target_text"] == "pick cup"
