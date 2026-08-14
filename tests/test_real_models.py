import os

import pytest
import torch

from vljepa_droid.models.qwen3_predictor import Qwen3VLJEPAPredictor
from vljepa_droid.models.vjepa21_encoder import VJEPA21Encoder


@pytest.mark.integration
def test_qwen_predictor_is_bidirectional() -> None:
    model_path = os.environ.get("QWEN_MODEL")
    if not model_path or not torch.cuda.is_available():
        pytest.skip("set QWEN_MODEL and run on CUDA")
    predictor = Qwen3VLJEPAPredictor(model_path, gradient_checkpointing=False).to("cuda").eval()
    torch.manual_seed(2)
    visual = torch.randn(1, 4, 1664, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(1, 2, device="cuda", dtype=torch.long)
    ids_a = torch.tensor([[100, 101]], device="cuda")
    ids_b = torch.tensor([[100, 102]], device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        states_a, _ = predictor.encode_sequence(visual, ids_a, mask)
        states_b, _ = predictor.encode_sequence(visual, ids_b, mask)
    difference = (states_a[:, :4] - states_b[:, :4]).abs().max().float()
    assert difference > 1e-6, "future query token did not influence visual states"


@pytest.mark.integration
def test_vjepa_shape_and_frozen_contract() -> None:
    repo = os.environ.get("VJEPA_REPO")
    checkpoint = os.environ.get("VJEPA_CHECKPOINT")
    if not repo or not checkpoint or not torch.cuda.is_available():
        pytest.skip("set VJEPA_REPO and VJEPA_CHECKPOINT and run on CUDA")
    encoder = VJEPA21Encoder(
        vjepa_repo=repo,
        checkpoint_path=checkpoint,
        device="cuda",
    )
    video = torch.zeros(1, 3, 32, 256, 256, device="cuda")
    tokens = encoder(video)
    assert tokens.shape == (1, 4096, 1664)
    trainable_probe = torch.nn.Linear(1664, 1, device="cuda", dtype=tokens.dtype)
    trainable_probe(tokens).mean().backward()
    assert all(parameter.grad is None for parameter in encoder.parameters())
    assert trainable_probe.weight.grad is not None
