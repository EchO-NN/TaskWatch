import torch
import torch.nn.functional as F

from vljepa_droid.losses.infonce import bidirectional_infonce


def test_matched_infonce_beats_permutation() -> None:
    torch.manual_seed(7)
    target = F.normalize(torch.randn(16, 64), dim=-1)
    prediction = target.clone().requires_grad_(True)
    matched = bidirectional_infonce(prediction, target, temperature=0.07, distributed_gather=False)
    permutation = torch.roll(target, shifts=1, dims=0)
    shuffled = bidirectional_infonce(
        prediction, permutation, temperature=0.07, distributed_gather=False
    )
    assert matched.loss < shuffled.loss
    assert matched.positive_cosine_mean > shuffled.positive_cosine_mean
    matched.loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
