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


def test_checkpoint_restores_optimizer_and_continuation_step(tmp_path) -> None:
    model = DummyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4)
    optimizer.zero_grad(set_to_none=True)
    loss = model.predictor.projection(model.predictor.embed_tokens.weight[:2]).sum()
    loss.backward()
    optimizer.step()

    path = save_checkpoint(
        tmp_path / "checkpoint_step_0010000.pt",
        model=model,
        optimizer=optimizer,
        global_step=10_000,
        config={"training": {"max_steps": 10_000}},
    )
    saved_optimizer = optimizer.state_dict()

    resumed_model = DummyModel()
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=9.0e-4)
    global_step = load_checkpoint(
        path,
        model=resumed_model,
        optimizer=resumed_optimizer,
    )

    assert global_step == 10_000
    resumed_state = resumed_optimizer.state_dict()
    assert resumed_state["param_groups"] == saved_optimizer["param_groups"]
    assert resumed_state["state"].keys() == saved_optimizer["state"].keys()
    for parameter_id, expected_state in saved_optimizer["state"].items():
        for key, expected_value in expected_state.items():
            actual_value = resumed_state["state"][parameter_id][key]
            if torch.is_tensor(expected_value):
                torch.testing.assert_close(actual_value, expected_value)
            else:
                assert actual_value == expected_value

    payload = torch.load(path, weights_only=False)
    assert payload["scheduler"] == {"name": "constant", "global_step": 10_000}
