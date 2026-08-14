import torch
import torch.nn as nn

from vljepa_droid.training.checkpoint import load_checkpoint, save_checkpoint


class DummyPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(11, 4)
        self.projection = nn.Linear(4, 3)


class DummyYEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.target_head = nn.Linear(4, 3)


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.x_encoder = nn.Linear(99, 99)
        self.predictor = DummyPredictor()
        self.y_encoder = DummyYEncoder()


def test_checkpoint_excludes_frozen_x_and_qwen_embedding(tmp_path) -> None:
    model = DummyModel()
    optimizer = torch.optim.AdamW(model.parameters())
    path = save_checkpoint(
        tmp_path / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        global_step=12,
        config={"test": True},
    )
    payload = torch.load(path, weights_only=False)
    assert "x_encoder" not in payload["model"]
    assert "embed_tokens.weight" not in payload["model"]["predictor"]
    original_embedding = model.predictor.embed_tokens.weight.detach().clone()
    with torch.no_grad():
        model.predictor.projection.weight.zero_()
    assert load_checkpoint(path, model=model) == 12
    torch.testing.assert_close(model.predictor.embed_tokens.weight, original_embedding)
    assert model.predictor.projection.weight.abs().sum() > 0
