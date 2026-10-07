"""Distribution-scoped EDR stop masks remain exact and reusable across tiles."""

from unittest import mock

import pytest
import torch

from angelspec.models.ops import edr

VOCAB_SIZE = 11
BOUNDARY_STOPS = [0, 3, 4, 7, 8, 10, 4, 0]


def _expected_mask(stopping_token_ids, vocab_size=VOCAB_SIZE):
    stops = set(stopping_token_ids)
    return torch.tensor([token not in stops for token in range(vocab_size)], dtype=torch.bool)


def _distribution(stopping_token_ids=BOUNDARY_STOPS, *, device="cpu"):
    generator = torch.Generator().manual_seed(351)
    target_logits = torch.randn(4, VOCAB_SIZE, generator=generator).to(device)
    return edr.prepare_edr_target_distribution(
        target_logits, 4, stopping_token_ids=stopping_token_ids,
    )


def _batch(*, dtype=torch.float32):
    generator = torch.Generator().manual_seed(352)
    draft_logits = torch.randn(2, 3, VOCAB_SIZE, generator=generator).to(dtype)
    target_rows = torch.tensor([[0, 1, 2], [1, 2, 3]])
    target_ids = torch.tensor([[0, 3, 4], [7, 8, 10]])
    return draft_logits, target_rows, target_ids


def _dense_reference(draft_logits, target_logits, target_rows, target_ids, stopping_token_ids):
    """Independent dense conditional cost and acceptance, without EDR helpers."""
    draft = torch.softmax(draft_logits.float(), dim=-1)
    target = torch.softmax(target_logits.detach().float(), dim=-1)[target_rows]
    non_stopping = _expected_mask(stopping_token_ids, draft.shape[-1]).to(draft.device)
    stop_ids = torch.tensor(sorted(set(stopping_token_ids)), dtype=torch.long, device=draft.device)
    stopping_mass = target.index_select(-1, stop_ids).sum(dim=-1)
    non_stopping_mass = (1.0 - stopping_mass).clamp(min=0.0, max=1.0)
    rejection = ((target - draft).clamp_min(0.0) * non_stopping.float()).sum(dim=-1)
    denominator = torch.where(
        non_stopping_mass > 0, non_stopping_mass, torch.ones_like(non_stopping_mass),
    )
    costs = torch.where(
        non_stopping_mass > 0, rejection / denominator, torch.zeros_like(rejection),
    )
    gather_ids = target_ids.unsqueeze(-1)
    draft_realized = draft.gather(-1, gather_ids).squeeze(-1)
    target_realized = target.gather(-1, gather_ids).squeeze(-1)
    ratio = draft_realized / target_realized.clamp_min(torch.finfo(target.dtype).tiny)
    acceptance = torch.minimum(ratio, torch.ones_like(ratio))
    return costs, acceptance


@pytest.mark.parametrize("stopping_token_ids", [[], [10], [4, 4, 0], BOUNDARY_STOPS, list(range(11))])
def test_distribution_caches_independent_expected_full_vocabulary_mask(stopping_token_ids):
    distribution = _distribution(stopping_token_ids)
    mask = distribution.non_stopping_vocab
    assert mask.dtype == torch.bool
    assert mask.shape == (VOCAB_SIZE,)
    assert mask.device == distribution.logits.device
    assert distribution.non_stopping_vocab is mask
    assert not mask.requires_grad
    torch.testing.assert_close(mask, _expected_mask(stopping_token_ids), rtol=0, atol=0)
    torch.testing.assert_close(
        distribution.stopping_token_ids,
        torch.tensor(sorted(set(stopping_token_ids)), dtype=torch.long),
        rtol=0,
        atol=0,
    )


def test_masks_are_isolated_between_distributions_even_for_equal_stop_sets():
    first = _distribution([0, 4])
    equal_stops = _distribution([4, 0, 4])
    different_stops = _distribution([3, 10])
    masks = [distribution.non_stopping_vocab for distribution in (first, equal_stops, different_stops)]
    assert len({mask.untyped_storage().data_ptr() for mask in masks}) == 3
    torch.testing.assert_close(masks[0], masks[1], rtol=0, atol=0)
    torch.testing.assert_close(masks[2], _expected_mask([3, 10]), rtol=0, atol=0)


@pytest.mark.parametrize("stopping_token_ids", [[], BOUNDARY_STOPS, list(range(11))])
def test_no_grad_grad_and_backward_tiles_share_cached_mask_storage(stopping_token_ids):
    distribution = _distribution(stopping_token_ids)
    full_mask = distribution.non_stopping_vocab
    draft_logits, target_rows, target_ids = _batch()
    observations = {"no_grad": [], "grad": [], "backward": []}
    originals = {
        "no_grad": edr._edr_cost_tile,
        "grad": edr._edr_cost_and_dot_tile,
        "backward": edr._edr_backward_tile,
    }

    def recorder(phase, mask_index):
        def record(*args, **kwargs):
            observations[phase].append(args[mask_index])
            return originals[phase](*args, **kwargs)

        return record

    with (
        mock.patch.object(edr, "_edr_cost_tile", side_effect=recorder("no_grad", 5)),
        mock.patch.object(edr, "_edr_cost_and_dot_tile", side_effect=recorder("grad", 6)),
        mock.patch.object(edr, "_edr_backward_tile", side_effect=recorder("backward", 8)),
    ):
        with torch.no_grad():
            first = edr.streaming_edr_distribution_statistics(
                draft_logits, distribution, target_rows, target_ids, 4,
            )
            second = edr.streaming_edr_distribution_statistics(
                draft_logits, distribution, target_rows, target_ids, 4,
            )
        differentiable_logits = draft_logits.detach().clone().requires_grad_()
        differentiable = edr.streaming_edr_distribution_statistics(
            differentiable_logits, distribution, target_rows, target_ids, 4,
        )
        (differentiable[0].sum() + differentiable[1].sum()).backward()

    assert distribution.non_stopping_vocab is full_mask
    expected_ranges = [(0, 4), (4, 8), (8, 11)]
    for phase, repeats in (("no_grad", 2), ("grad", 1), ("backward", 1)):
        masks = observations[phase]
        assert len(masks) == repeats * len(expected_ranges)
        for mask, (start, end) in zip(masks, expected_ranges * repeats, strict=True):
            assert mask.untyped_storage().data_ptr() == full_mask.untyped_storage().data_ptr()
            assert mask.storage_offset() == full_mask.storage_offset() + start
            assert mask.shape == (end - start,)
            assert mask.stride() == (1,)
            torch.testing.assert_close(mask, _expected_mask(stopping_token_ids)[start:end])
    for first_value, second_value, grad_value in zip(first, second, differentiable, strict=True):
        torch.testing.assert_close(first_value, second_value, rtol=0, atol=0)
        torch.testing.assert_close(first_value, grad_value, rtol=0, atol=0)


@pytest.mark.parametrize("stopping_token_ids", [[], BOUNDARY_STOPS, list(range(11))])
@pytest.mark.parametrize("chunk_size", [4, 20])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cached_masks_preserve_dense_conditional_cost_and_gradient(stopping_token_ids, chunk_size, dtype):
    distribution = _distribution(stopping_token_ids)
    draft_logits, target_rows, target_ids = _batch(dtype=dtype)
    draft_logits.requires_grad_()
    reference_logits = draft_logits.detach().clone().requires_grad_()
    actual = edr.streaming_edr_distribution_statistics(
        draft_logits, distribution, target_rows, target_ids, chunk_size,
    )
    expected = _dense_reference(
        reference_logits, distribution.logits, target_rows, target_ids, stopping_token_ids,
    )
    generator = torch.Generator().manual_seed(353)
    cost_weights = torch.randn(2, 3, generator=generator)
    acceptance_weights = torch.randn(2, 3, generator=generator)
    actual_gradient = torch.autograd.grad(
        (actual[0] * cost_weights + actual[1] * acceptance_weights).sum(), draft_logits,
    )[0]
    expected_gradient = torch.autograd.grad(
        (expected[0] * cost_weights + expected[1] * acceptance_weights).sum(), reference_logits,
    )[0]
    for actual_value, expected_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_value, expected_value, atol=2e-6, rtol=2e-6)
    if dtype == torch.bfloat16:
        torch.testing.assert_close(actual_gradient, expected_gradient)
    else:
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("stopping_token_ids", [[-1], [11], [4, 4, -1], [0, 12]])
@pytest.mark.parametrize("as_tensor", [False, True])
def test_invalid_stopping_ids_still_raise_value_error(stopping_token_ids, as_tensor):
    argument = torch.tensor(stopping_token_ids) if as_tensor else stopping_token_ids
    with pytest.raises(ValueError, match=r"stopping token IDs must be in \[0, 11\)"):
        _distribution(argument)


def test_first_use_under_inference_mode_does_not_poison_later_backward():
    distribution = _distribution()
    draft_logits, target_rows, target_ids = _batch()
    with torch.inference_mode():
        inference_values = edr.streaming_edr_distribution_statistics(
            draft_logits, distribution, target_rows, target_ids, 4,
        )
        cached_mask = distribution.non_stopping_vocab
        assert not torch.is_inference(cached_mask)

    draft_logits.requires_grad_()
    training_values = edr.streaming_edr_distribution_statistics(
        draft_logits, distribution, target_rows, target_ids, 4,
    )
    (training_values[0].sum() + training_values[1].sum()).backward()
    assert distribution.non_stopping_vocab is cached_mask
    assert draft_logits.grad is not None
    assert bool(torch.isfinite(draft_logits.grad).all())
    for actual, expected in zip(training_values, inference_values, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA stop-mask transfer")
def test_static_stopping_ids_are_deduplicated_on_cpu_before_cuda_transfer():
    original_unique = torch.unique
    unique_devices = []

    def record_unique(tensor, *args, **kwargs):
        unique_devices.append(tensor.device.type)
        return original_unique(tensor, *args, **kwargs)

    with mock.patch.object(torch, "unique", side_effect=record_unique):
        stop_ids = edr._validated_stopping_token_ids([10, 4, 4, 0], 11, torch.device("cuda"))
    assert unique_devices == ["cpu"]
    assert stop_ids.device.type == "cuda"
    torch.testing.assert_close(stop_ids.cpu(), torch.tensor([0, 4, 10]), rtol=0, atol=0)
    distribution = _distribution([10, 4, 4, 0], device="cuda")
    assert distribution.non_stopping_vocab.device == distribution.logits.device
    assert distribution.non_stopping_vocab is distribution.non_stopping_vocab
    torch.testing.assert_close(distribution.non_stopping_vocab.cpu(), _expected_mask([0, 4, 10]))
