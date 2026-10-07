"""CPU-prepared EDR coefficients match an independent sampled-objective reference."""

from dataclasses import replace

import pytest
import torch

from angelspec.models.ops import edr


def _dynamic_program(length=4, width=3, acceptance=0.99):
    generator = torch.Generator().manual_seed(271)
    costs = torch.rand(length, width, generator=generator)
    accepts = torch.full_like(costs, acceptance)
    return edr.exact_edr_dynamic_program(costs, accepts, num_proposals=width)


def _sample(dynamic_program, num_anchors, random_start=0.5):
    length = dynamic_program.occupancies.shape[0]
    draw = edr.sample_edr_round_starts(
        dynamic_program.round_start_probabilities[:length],
        num_anchors,
        random_start=random_start,
    )
    return (
        draw.indices,
        draw.selected_inclusion_probabilities,
        torch.ones_like(draw.indices, dtype=torch.bool),
        draw.inverse_pps_scale,
    )


def _old_sampled_surrogate(costs, acceptance, dynamic_program, prefixes, inclusion, keep, inverse):
    """Reference sampled-surrogate arithmetic, independent of the production helpers."""
    keep = keep.to(device=costs.device, dtype=torch.bool)
    prefixes = prefixes.to(device=costs.device, dtype=torch.long)
    safe_prefixes = torch.where(keep, prefixes, torch.zeros_like(prefixes))
    weights = dynamic_program.occupancies.detach().to(costs.device)[safe_prefixes]
    survivals = dynamic_program.conditional_survivals.detach().to(costs.device)[safe_prefixes]
    advantages = dynamic_program.continuation_advantages.detach().to(costs.device)[safe_prefixes]
    learned_mask = dynamic_program.learned_mask.to(costs.device)[safe_prefixes]
    learned_mask = learned_mask & keep.unsqueeze(-1)
    round_starts = dynamic_program.round_start_probabilities.detach().to(costs.device)[
        safe_prefixes
    ]
    if inverse is None:
        ht_weights = weights
    else:
        inverse_scale = torch.as_tensor(inverse, device=costs.device, dtype=torch.float64)
        saturated = round_starts >= inverse_scale
        canceled_weights = survivals * inverse_scale
        ht_weights = torch.where(saturated.unsqueeze(-1), weights, canceled_weights)
    ht_weights = ht_weights.float()
    terms = ht_weights * (costs.float() + acceptance.float() * advantages)
    return torch.where(learned_mask, terms, torch.zeros_like(terms)).sum()


def _assert_value_and_gradient_parity(dynamic_program, sampling, *, dtype, device="cpu"):
    prefixes, inclusion, keep, inverse = sampling
    coefficients = edr.prepare_sampled_edr_surrogate_coefficients(
        dynamic_program, prefixes, inclusion, keep, inverse,
    )
    for field in (coefficients.weights, coefficients.continuation_advantages):
        assert field.device.type == "cpu"
        assert field.dtype == torch.float32
        assert not field.requires_grad
    assert coefficients.learned_mask.dtype == torch.bool
    assert coefficients.learned_mask.device.type == "cpu"
    coefficients = coefficients.to(device)
    for field in (
        coefficients.weights,
        coefficients.continuation_advantages,
        coefficients.learned_mask,
    ):
        assert field.device.type == torch.device(device).type

    generator = torch.Generator().manual_seed(272)
    shape = (prefixes.numel(), dynamic_program.occupancies.shape[1])
    costs = torch.randn(shape, generator=generator).to(device=device, dtype=dtype)
    acceptance = torch.randn(shape, generator=generator).to(device=device, dtype=dtype)
    costs.requires_grad_()
    acceptance.requires_grad_()
    reference_costs = costs.detach().clone().requires_grad_()
    reference_acceptance = acceptance.detach().clone().requires_grad_()
    expected = _old_sampled_surrogate(
        reference_costs, reference_acceptance, dynamic_program, *sampling,
    )
    actual = edr.edr_surrogate_sum_from_coefficients(costs, acceptance, coefficients)
    expected_gradients = torch.autograd.grad(expected, (reference_costs, reference_acceptance))
    actual_gradients = torch.autograd.grad(actual, (costs, acceptance))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
    return coefficients, actual_gradients


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("design", ["mixed", "uncapped", "all_fit"])
def test_cpu_preparation_matches_literal_old_surrogate_and_gradient(dtype, design):
    dynamic_program = _dynamic_program()
    sampling = _sample(dynamic_program, {"mixed": 2, "uncapped": 1, "all_fit": 8}[design])
    inclusion = sampling[1]
    if design == "mixed":
        assert bool((inclusion == 1).any())
        assert bool((inclusion < 1).any())
    elif design == "uncapped":
        assert bool((inclusion < 1).all())
    else:
        assert bool((inclusion == 1).all())
        assert sampling[3] is None
    _assert_value_and_gradient_parity(dynamic_program, sampling, dtype=dtype)


def test_reducer_preserves_original_fp32_parentheses():
    weights = torch.tensor([[0.3]])
    advantages = torch.tensor([[-99_999_992.0]])
    coefficients = edr.EDRSurrogateCoefficients(
        weights=weights,
        continuation_advantages=advantages,
        learned_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    costs = torch.tensor([[100_000_000.0]], requires_grad=True)
    acceptance = torch.ones(1, 1, requires_grad=True)
    expected = (weights * (costs.float() + acceptance.float() * advantages)).sum()
    reassociated = (weights * costs + acceptance * (weights * advantages)).sum()
    assert not torch.equal(expected, reassociated)
    actual = edr.edr_surrogate_sum_from_coefficients(costs, acceptance, coefficients)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_gradients = torch.autograd.grad(actual, (costs, acceptance))
    expected_gradients = torch.autograd.grad(expected, (costs, acceptance))
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


def _tiny_survival_case():
    base = _dynamic_program(length=3, width=2, acceptance=1.0)
    round_starts = torch.tensor([1.0, 1e-300, 0.0, 0.0], dtype=torch.float64)
    occupancies = base.occupancies.clone()
    occupancies[1] = torch.tensor([1e-300, 0.0], dtype=torch.float64)
    survivals = base.conditional_survivals.clone()
    survivals[1] = torch.tensor([1.0, 1e-30], dtype=torch.float64)
    dynamic_program = replace(
        base,
        occupancies=occupancies,
        conditional_survivals=survivals,
        round_start_probabilities=round_starts,
    )
    draw = edr.sample_edr_round_starts(round_starts[:3], 1, random_start=0.5)
    assert draw.inclusion_probabilities[1].item() == 1e-300
    sampling = (
        torch.tensor([1]),
        draw.inclusion_probabilities[1:2],
        torch.tensor([True]),
        draw.inverse_pps_scale,
    )
    return dynamic_program, sampling


def test_cpu_preparation_preserves_fp64_canceled_tiny_survival():
    dynamic_program, sampling = _tiny_survival_case()
    coefficients, gradients = _assert_value_and_gradient_parity(
        dynamic_program, sampling, dtype=torch.float32,
    )
    expected = torch.tensor([[1.0, 1e-30]])
    torch.testing.assert_close(coefficients.weights, expected, rtol=0, atol=0)
    torch.testing.assert_close(gradients[0], expected, rtol=0, atol=0)


def test_short_horizon_ignores_out_of_range_masked_padding():
    dynamic_program = _dynamic_program(length=2, width=3)
    prefixes, inclusion, keep, inverse = _sample(dynamic_program, 8)
    sampling = (
        torch.cat((prefixes, torch.tensor([-123, 999]))),
        torch.cat((inclusion, torch.zeros(2, dtype=torch.float64))),
        torch.cat((keep, torch.zeros(2, dtype=torch.bool))),
        inverse,
    )
    coefficients, gradients = _assert_value_and_gradient_parity(
        dynamic_program, sampling, dtype=torch.float32,
    )
    assert coefficients.weights.shape == (4, 3)
    assert not bool(coefficients.learned_mask[2:].any())
    for gradient in gradients:
        assert torch.count_nonzero(gradient[2:]).item() == 0
        assert torch.count_nonzero(gradient[~coefficients.learned_mask]).item() == 0


def test_prepared_coefficients_are_detached_from_dynamic_program():
    base = _dynamic_program()
    dynamic_program = replace(
        base,
        occupancies=base.occupancies.clone().requires_grad_(),
        conditional_survivals=base.conditional_survivals.clone().requires_grad_(),
        continuation_advantages=base.continuation_advantages.clone().requires_grad_(),
        round_start_probabilities=base.round_start_probabilities.clone().requires_grad_(),
    )
    _assert_value_and_gradient_parity(dynamic_program, _sample(dynamic_program, 2), dtype=torch.float32)


@pytest.mark.parametrize(
    "invalid",
    [
        "prefix_rank", "inclusion_shape", "keep_shape", "negative_prefix", "large_prefix",
        "zero_probability", "missing_inverse", "zero_inverse", "negative_inverse",
        "nan_inverse", "infinite_inverse", "vector_inverse", "mismatched_probability",
    ],
)
def test_preparation_rejects_invalid_sampling_metadata(invalid):
    dynamic_program = _dynamic_program()
    prefixes, inclusion, keep, inverse = _sample(dynamic_program, 2)
    prefixes, inclusion, keep = prefixes.clone(), inclusion.clone(), keep.clone()
    if invalid == "prefix_rank":
        prefixes = prefixes.unsqueeze(0)
    elif invalid == "inclusion_shape":
        inclusion = inclusion[:-1]
    elif invalid == "keep_shape":
        keep = keep[:-1]
    elif invalid == "negative_prefix":
        prefixes[0] = -1
    elif invalid == "large_prefix":
        prefixes[0] = dynamic_program.occupancies.shape[0]
    elif invalid == "zero_probability":
        inclusion[0] = 0
    elif invalid == "missing_inverse":
        assert bool((inclusion < 1).any())
        inverse = None
    elif invalid == "zero_inverse":
        inverse = torch.tensor(0.0, dtype=torch.float64)
    elif invalid == "negative_inverse":
        inverse = torch.tensor(-1.0, dtype=torch.float64)
    elif invalid == "nan_inverse":
        inverse = torch.tensor(float("nan"), dtype=torch.float64)
    elif invalid == "infinite_inverse":
        inverse = torch.tensor(float("inf"), dtype=torch.float64)
    elif invalid == "vector_inverse":
        inverse = torch.ones(2, dtype=torch.float64)
    elif invalid == "mismatched_probability":
        inclusion[0] *= 0.5
    with pytest.raises(ValueError):
        edr.prepare_sampled_edr_surrogate_coefficients(
            dynamic_program, prefixes, inclusion, keep, inverse,
        )


@pytest.mark.parametrize("invalid", ["rank", "acceptance_shape", "coefficient_shape"])
def test_reducer_rejects_incompatible_statistics(invalid):
    dynamic_program = _dynamic_program()
    coefficients = edr.prepare_sampled_edr_surrogate_coefficients(
        dynamic_program, *_sample(dynamic_program, 2),
    )
    costs = torch.ones_like(coefficients.weights)
    acceptance = torch.ones_like(costs)
    if invalid == "rank":
        costs, acceptance = costs.flatten(), acceptance.flatten()
    elif invalid == "acceptance_shape":
        acceptance = acceptance[:, :-1]
    elif invalid == "coefficient_shape":
        costs, acceptance = costs[:-1], acceptance[:-1]
    with pytest.raises(ValueError):
        edr.edr_surrogate_sum_from_coefficients(costs, acceptance, coefficients)


def test_coefficient_preparation_and_transfer_preserve_seeded_sampling_rng():
    dynamic_program = _dynamic_program()
    starts = dynamic_program.round_start_probabilities[:-1]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(273)
        first = edr.sample_edr_round_starts(starts, 2)
        before_preparation = torch.get_rng_state().clone()
        edr.prepare_sampled_edr_surrogate_coefficients(
            dynamic_program, first.indices, first.selected_inclusion_probabilities,
            torch.ones_like(first.indices, dtype=torch.bool), first.inverse_pps_scale,
        ).to("cpu")
        assert torch.equal(torch.get_rng_state(), before_preparation)
        second = edr.sample_edr_round_starts(starts, 2)
        final_state = torch.get_rng_state().clone()

        torch.manual_seed(273)
        reference_first = edr.sample_edr_round_starts(starts, 2)
        reference_second = edr.sample_edr_round_starts(starts, 2)
        assert torch.equal(torch.get_rng_state(), final_state)
        for actual, expected in ((first, reference_first), (second, reference_second)):
            torch.testing.assert_close(actual.indices, expected.indices, rtol=0, atol=0)
            torch.testing.assert_close(
                actual.inclusion_probabilities, expected.inclusion_probabilities, rtol=0, atol=0,
            )
            torch.testing.assert_close(actual.inverse_pps_scale, expected.inverse_pps_scale, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA coefficient transfer")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("case", ["mixed", "tiny_survival"])
def test_cuda_transferred_cpu_coefficients_match_original_gpu_arithmetic(dtype, case):
    if case == "tiny_survival":
        dynamic_program, sampling = _tiny_survival_case()
    else:
        dynamic_program = _dynamic_program()
        sampling = _sample(dynamic_program, 2)
    _assert_value_and_gradient_parity(dynamic_program, sampling, dtype=dtype, device="cuda")
