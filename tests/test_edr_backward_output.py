"""Destination-writing EDR kernels retain the exact analytical gradient."""

import pytest
import torch

from angelspec.models.ops import edr


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("compiled", [False, True])
def test_backward_tile_writes_only_destination_slice(dtype, compiled):
    if compiled and not torch.cuda.is_available():
        pytest.skip("requires CUDA compilation")
    device = "cuda" if compiled else "cpu"
    torch.manual_seed(48)
    logits = torch.randn(2, 3, 19, dtype=dtype, device=device)
    target = torch.softmax(torch.randn(4, 19, device=device), dim=-1)
    indices = torch.tensor([[0, 1, 2], [1, 2, 3]], device=device)
    ids = torch.tensor([[0, 3, 5], [8, 12, 18]], device=device)
    stop_mask = torch.ones(19, dtype=torch.bool, device=device)
    stop_mask[5] = False
    non_stop = torch.rand(2, 3, device=device)
    non_stop[0, 0] = 0  # The zero-non-stop branch must retain a zero cost gradient.
    common = (
        logits, torch.logsumexp(logits.float(), -1), target, indices, ids,
        torch.arange(19, device=device), torch.rand(2, 3, device=device),
        non_stop, stop_mask, torch.rand(2, 3, device=device),
        torch.randn(2, 3, device=device), torch.randn(2, 3, device=device),
    )
    expected = edr._edr_backward_tile(*common)
    start, end = 3, 14
    args = list(common)
    args[0] = logits[..., start:end]
    args[2] = target[:, start:end]
    args[5] = common[5][start:end]
    args[8] = stop_mask[start:end]
    output = torch.full_like(logits, 42)
    destination = output[..., start:end]
    kernel = edr._compiled_edr_backward_tile if compiled else edr._edr_backward_tile
    returned = kernel(*args, destination)
    assert returned.data_ptr() == destination.data_ptr()
    torch.testing.assert_close(output[..., start:end], expected[..., start:end])
    assert bool((output[..., :start] == 42).all())
    assert bool((output[..., end:] == 42).all())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("chunk", [1, 7, 64])
@pytest.mark.parametrize("gradient_mode", ["cost", "acceptance", "both"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_streamed_backward_matches_dense_conditional_reference(dtype, chunk, gradient_mode, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA compilation")
    torch.manual_seed(36)
    target = torch.randn(4, 23, dtype=dtype, device=device)
    rows = torch.tensor([[0, 1, 2], [1, 2, 3]], device=device)
    ids = torch.tensor([[0, 2, 4], [6, 8, 22]], device=device)
    draft = torch.randn(2, 3, 23, dtype=dtype, device=device, requires_grad=True)
    reference = draft.detach().clone().requires_grad_(True)
    stops = [1, 21]
    expected = edr.edr_distribution_statistics_from_target_probabilities(
        reference, target.float().softmax(-1)[rows], ids, stops,
    )
    actual = edr.streaming_edr_distribution_statistics(
        draft, edr.prepare_edr_target_distribution(target, chunk, stopping_token_ids=stops),
        rows, ids, chunk,
    )
    cost_weights = torch.randn(2, 3, device=device)
    acceptance_weights = torch.randn(2, 3, device=device)

    def objective(statistics):
        if gradient_mode == "cost":
            return (statistics[0] * cost_weights).sum()
        if gradient_mode == "acceptance":
            return (statistics[1] * acceptance_weights).sum()
        return (statistics[0] * cost_weights + statistics[1] * acceptance_weights).sum()

    objective(expected).backward()
    objective(actual).backward()
    for result, ref in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, ref, atol=2e-6, rtol=2e-6)
    tolerance = {"atol": 2e-4, "rtol": 8e-3} if dtype == torch.bfloat16 else {
        "atol": 3e-6, "rtol": 3e-6,
    }
    torch.testing.assert_close(draft.grad, reference.grad, **tolerance)
