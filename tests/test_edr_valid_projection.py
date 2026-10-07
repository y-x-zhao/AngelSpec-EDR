"""Valid-only EDR projection matches the full rectangular projection and VJP."""

from copy import deepcopy
from types import MethodType
from unittest import mock

import pytest
import torch
from torch import nn

from angelspec.models.dflash import DFlashModel
from angelspec.models.dfly import DFlyModel
from angelspec.models.draft.dfly import DFlyConfig, DFlyDraftModel
from angelspec.models.draft.dspark import DSparkConfig, DSparkDraftModel
from angelspec.models.dspark import DSparkModel
from angelspec.models.ops.edr import (
    greedy_edr_distribution_statistics,
    prepare_edr_target_distribution,
    streaming_edr_distribution_statistics,
)


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model_for(kind, device="cpu", dtype=torch.float32, *, head_dim=8):
    common = dict(
        hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=32 // head_dim, num_key_value_heads=16 // head_dim,
        head_dim=head_dim, vocab_size=61,
        max_position_embeddings=256, num_target_layers=2, target_hidden_size=32,
        target_num_hidden_layers=4, target_layer_ids=[0, 2], mask_token_id=60,
    )
    options = dict(
        num_anchors=2, loss_objective="edr", edr_chunk_size=8,
        edr_vocab_chunk_size=17, edr_stop_token_ids=[1, 2],
        edr_temperature=0.7, edr_top_k=20, edr_top_p=0.8,
    )
    if kind == "dfly":
        draft = DFlyDraftModel(DFlyConfig(
            **common, block_size=8, enable_hidden_correction=True,
        ))
        model = DFlyModel(draft, block_size=7, query_includes_input_anchor=True, **options)
    elif kind == "dspark":
        draft = DSparkDraftModel(DSparkConfig(
            **common, markov_rank=4, enable_confidence_head=False,
        ))
        model = DSparkModel(
            draft, block_size=7, query_includes_input_anchor=False,
            ce_loss_alpha=0, l1_loss_alpha=0, confidence_head_alpha=0, **options,
        )
        model.edr_fused_markov_projection = True
    else:
        draft = nn.Module()
        draft.lm_head = nn.Linear(32, 61, bias=False)
        model = DFlashModel(draft, block_size=7, **options)
    return model.to(device=device, dtype=dtype)


def rectangular_statistics(
    self, *, draft_hidden, lm_head_weight, prev_token_ids, target_distribution,
    target_probability_indices, target_ids, valid_mask, draft_temperature,
    cache_rejection_mask,
    projection_indices=None,
):
    """Independent reference path: all slots reach both head and loss."""
    logits = self._compute_draft_logits(
        draft_hidden, lm_head_weight, prev_token_ids, valid_mask.shape[1],
    ).view(*valid_mask.shape, -1)
    if draft_temperature == 0:
        result = greedy_edr_distribution_statistics(
            logits, target_distribution, target_probability_indices, target_ids,
        )
    else:
        result = streaming_edr_distribution_statistics(
            logits, target_distribution, target_probability_indices, target_ids,
            self.edr_vocab_chunk_size, cache_rejection_mask=cache_rejection_mask,
            draft_temperature=draft_temperature,
        )
    return tuple(torch.where(valid_mask, value, 0) for value in result)


@pytest.mark.parametrize("device,dtype", [
    ("cpu", torch.float32), ("cuda", torch.float32), ("cuda", torch.bfloat16),
])
@pytest.mark.parametrize("kind", ["dflash", "dfly", "dspark"])
@pytest.mark.parametrize("sparse,temperature", [(False, 0.7), (True, 0.7), (False, 0.0)])
def test_valid_projection_statistics_and_all_projection_gradients(kind, sparse, temperature, device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(4301)
    model = model_for(kind, device, dtype)
    reference = deepcopy(model)
    model.edr_sparse_target = sparse
    weight = torch.randn(61, 32, device=device, dtype=dtype)
    teacher = torch.randn(11, 32, device=device, dtype=dtype)
    model.edr_temperature = temperature
    distribution = model._prepare_edr_target_distribution(teacher, weight)
    hidden = torch.randn(2, 4 * 7, 32, device=device, dtype=dtype, requires_grad=True)
    original_hidden = hidden.detach().clone().requires_grad_()
    valid = torch.tensor([
        [7, 0, 3, 1], [0, 2, 7, 0],
    ], device=device).unsqueeze(-1) > torch.arange(7, device=device)
    previous = torch.randint(0, 61, valid.shape, device=device)
    rows = torch.randint(0, 11, valid.shape, device=device)
    tokens = torch.randint(0, 61, valid.shape, device=device)
    inputs = dict(
        lm_head_weight=weight, prev_token_ids=previous, target_distribution=distribution,
        target_probability_indices=rows, target_ids=tokens, valid_mask=valid,
        draft_temperature=temperature, cache_rejection_mask=not sparse,
    )
    with torch.set_grad_enabled(temperature > 0):
        with mock.patch.object(model, "_compute_draft_logits", wraps=model._compute_draft_logits) as project:
            result = model._edr_project_valid_statistics(draft_hidden=hidden, **inputs)
        projected = project.call_args.args[0]
        assert projected.numel() // projected.shape[-1] == int(valid.sum())
        assert int(valid.sum()) < valid.numel()
        expected = rectangular_statistics(reference, draft_hidden=original_hidden, **inputs)
        tolerance = dict(atol=2e-4, rtol=8e-3) if dtype == torch.bfloat16 else dict(atol=3e-6, rtol=3e-6)
        for actual, old in zip(result, expected, strict=True):
            torch.testing.assert_close(actual, old, **tolerance)
            assert torch.equal(actual[~valid], torch.zeros_like(actual[~valid]))
        if temperature > 0:
            coefficients = torch.randn(2, *valid.shape, device=device)
            sum((value * coeff).sum() for value, coeff in zip(result, coefficients)).backward()
            sum((value * coeff).sum() for value, coeff in zip(expected, coefficients)).backward()
            torch.testing.assert_close(hidden.grad, original_hidden.grad, **tolerance)
            for (name, parameter), (_, original) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
                if original.grad is None:
                    assert parameter.grad is None, name
                else:
                    torch.testing.assert_close(parameter.grad, original.grad, **tolerance, msg=name)


def test_empty_projection_does_not_call_any_vocabulary_head():
    model = model_for("dspark")
    hidden = torch.randn(2, 14, 32, requires_grad=True)
    mask = torch.zeros(2, 2, 7, dtype=torch.bool)
    distribution = prepare_edr_target_distribution(torch.randn(5, 61), 17)
    with mock.patch.object(model, "_compute_draft_logits", side_effect=AssertionError("empty projection")):
        values = model._edr_project_valid_statistics(
            draft_hidden=hidden, lm_head_weight=torch.randn(61, 32),
            prev_token_ids=torch.zeros_like(mask, dtype=torch.long),
            target_distribution=distribution,
            target_probability_indices=torch.zeros_like(mask, dtype=torch.long),
            target_ids=torch.zeros_like(mask, dtype=torch.long), valid_mask=mask,
            draft_temperature=0.7, cache_rejection_mask=False,
        )
    sum(value.sum() for value in values).backward()
    assert hidden.grad is not None and not hidden.grad.count_nonzero()


@pytest.mark.parametrize("kind", ["dfly", "dspark"])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("full", [False, True])
def test_full_training_preserves_metrics_and_every_gradient_with_multi_turn_tails(kind, sparse, full):
    torch.manual_seed(4302)
    model = model_for(kind)
    model.edr_sparse_target = sparse
    model.edr_full_anchor_backprop = full
    reference = deepcopy(model)
    reference._edr_project_valid_statistics = MethodType(rectangular_statistics, reference)
    weight = torch.randn(61, 32)
    hidden = torch.randn(2, 18, 32)
    ids = torch.randint(3, 60, (2, 18))
    # Realized ordinary tokens lie on the filtered teacher support.
    ids[:, 1:] = (hidden[:, :-1] @ weight.T).argmax(-1)
    batch = dict(
        input_ids=ids, hidden_states_list=[torch.randn(2, 18, 32) for _ in range(2)],
        last_hidden_states=hidden, lm_head_weight=weight, target_norm=nn.Identity(),
        loss_mask=torch.tensor([
            [0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        ], dtype=torch.float32),
        attention_mask=torch.tensor([[1] * 16 + [0] * 2, [1] * 15 + [0] * 3]),
        ctx_doc_ids=torch.tensor([[0] * 6 + [1] * 10 + [-1] * 2, [0] * 11 + [1] * 4 + [-1] * 3]),
    )
    torch.manual_seed(4303)
    actual = model(**batch)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(4303)
    expected = reference(**batch)
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    for value, old in zip(actual[:5], expected[:5], strict=True):
        torch.testing.assert_close(value, old, atol=3e-5, rtol=3e-5)
    for name in expected[5]:
        torch.testing.assert_close(actual[5][name], expected[5][name], atol=3e-5, rtol=3e-5)
    actual[0].backward()
    expected[0].backward()
    for (name, parameter), (_, original) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
        if original.grad is None:
            assert parameter.grad is None, name
        else:
            torch.testing.assert_close(parameter.grad, original.grad, atol=3e-5, rtol=3e-5, msg=name)
