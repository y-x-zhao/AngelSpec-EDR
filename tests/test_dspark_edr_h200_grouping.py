"""H200 EDR grouping keeps optimizer semantics; CPU-only, no model downloads."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from angelspec.models.dflash import _create_dflash_mask_mod
from angelspec.models.draft.dspark import DSparkConfig, DSparkDraftModel
from angelspec.models.dspark import DSparkModel
from angelspec.train_single_gpu import LocalDSparkTrainer

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _model(sparse):
    config = DSparkConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        vocab_size=31, max_position_embeddings=256, num_target_layers=2,
        target_hidden_size=16, target_num_hidden_layers=2, target_layer_ids=[0, 1],
        mask_token_id=30, markov_rank=4, enable_confidence_head=True,
    )
    draft = DSparkDraftModel(config)
    draft.freeze_embedding()
    model = DSparkModel(
        draft, block_size=7, num_anchors=2, loss_objective="edr",
        ce_loss_alpha=0.0, l1_loss_alpha=0.0, confidence_head_alpha=0.0,
        fp32_lm_head=False, edr_chunk_size=16, edr_vocab_chunk_size=17,
        edr_temperature=0.7, edr_top_k=20, edr_top_p=0.8, edr_stop_token_ids=[1, 2],
        query_includes_input_anchor=False, edr_full_anchor_backprop=False,
        edr_reuse_context_cache=True, edr_rejection_cache_max_mb=0,
    )
    model.edr_fused_markov_projection = True
    model.edr_sparse_target = sparse
    return model


def _batches(group):
    generator = torch.Generator().manual_seed(3611)
    rows, length, hidden = 8, 24, 16
    teacher_weight = torch.randn(31, hidden, generator=generator) * 0.3
    teacher_weight[1:3].zero_()
    teacher_hidden = torch.randn(rows, length, hidden, generator=generator)
    input_ids = torch.randint(3, 30, (rows, length), generator=generator)
    # Argmax tokens are in both dense and compact top-k/top-p distributions.
    input_ids[:, 1:] = nn.functional.linear(teacher_hidden[:, :-1], teacher_weight).argmax(-1)
    assert not torch.isin(input_ids[:, 1:], torch.tensor([1, 2])).any()
    loss_mask = torch.zeros_like(input_ids)
    for row in range(rows):
        end = 14 + row
        loss_mask[row, 3:end] = 1
        input_ids[row, end - 1] = 1
        if row % 2:
            # Different horizon counts and lengths test PPS ordering as groups grow.
            loss_mask[row, 8:11] = 0
            input_ids[row, 7] = 2
    features = torch.randn(rows, length, 2 * hidden, generator=generator)
    for start in range(0, rows, group):
        mask = loss_mask[start:start + group]
        yield dict(
            input_ids=input_ids[start:start + group], hidden_states=features[start:start + group],
            loss_mask=mask, attention_mask=torch.ones_like(mask),
            last_hidden_states=teacher_hidden[start:start + group],
            _loss_scale=1 / rows, _teacher_weight=teacher_weight,
        )


def _accumulated_step(model, group):
    batches = list(_batches(group))
    teacher_weight = batches[0]["_teacher_weight"]
    trainer = LocalDSparkTrainer.__new__(LocalDSparkTrainer)
    trainer.args = SimpleNamespace()
    trainer.data_fetcher = SimpleNamespace(microbatches_per_item=group)
    trainer.model = trainer.dflash = model
    trainer.loss_objective = "edr"
    trainer.edr_cross_row_batch_size = group
    trainer.edr_full_anchor_backprop = False
    trainer.edr_dp_workers = 1
    trainer.num_target_layers = 2
    trainer.target_lm_head = SimpleNamespace(norm=nn.Identity())
    trainer.target_lm_head_weight = teacher_weight
    cpu_trace = []
    prepare_query = model._prepare_edr_gradient_query
    sample_anchors = model._sample_anchor_positions
    sdpa = nn.functional.scaled_dot_product_attention
    attention_mask = None

    def sample_with_mask(seq_len, *args, **kwargs):
        # Production CUDA uses isolated block-causal attention; the bare CPU
        # backbone falls back to unmasked SDPA. Materialize the production mask
        # so changing group/chunk shapes tests GPU semantics rather than that
        # deliberately unmasked CPU fallback. All real DSpark layers still run.
        nonlocal attention_mask
        anchors, keep = sample_anchors(seq_len, *args, **kwargs)
        mask_mod = _create_dflash_mask_mod(anchors, keep, seq_len, model.block_size)
        query_length = anchors.shape[1] * model.block_size
        attention_mask = mask_mod(
            torch.arange(anchors.shape[0])[:, None, None], 0,
            torch.arange(query_length)[None, :, None],
            torch.arange(seq_len + query_length)[None, None, :],
        )[:, None]
        return anchors, keep

    def masked_attention(query, key, value, **kwargs):
        assert attention_mask is not None
        return sdpa(query, key, value, attn_mask=attention_mask, **kwargs)

    def trace(*args, **kwargs):
        query = prepare_query(*args, **kwargs)
        if query is not None:
            cpu_trace.append((query.prefixes.cpu().clone(), query.coefficients.weights.cpu().clone()))
        return query

    torch.manual_seed(3612)
    totals = {}
    loss = torch.zeros(())
    with (
        mock.patch.object(model, "_prepare_edr_gradient_query", side_effect=trace),
        mock.patch.object(model, "_sample_anchor_positions", side_effect=sample_with_mask),
        mock.patch(
            "angelspec.models.draft.dflash.F.scaled_dot_product_attention", masked_attention,
        ),
    ):
        prepared_batches, count = trainer._prepare_training_batches(iter(batches), 8)
        assert count == 8 // group
        for batch in prepared_batches:
            result = model(
                input_ids=batch["input_ids"],
                hidden_states_list=trainer._split_hidden_states(batch["hidden_states"]),
                loss_mask=batch["loss_mask"], attention_mask=batch.get("attention_mask"),
                lm_head_weight=teacher_weight, last_hidden_states=batch["last_hidden_states"],
                target_norm=trainer.target_lm_head.norm,
            )
            scaled = result[0] * batch["_loss_scale"]
            scaled.backward()
            loss += scaled.detach()
            for name in ("edr_generated_tokens", "edr_weighted_cost_sum", "edr_num_horizons"):
                totals[name] = totals.get(name, 0) + result[5][name].detach()
    return loss, totals, cpu_trace, torch.get_rng_state().clone()


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "compact"])
def test_four_row_groups_preserve_accumulated_loss_gradients_and_pps_rng(sparse):
    torch.manual_seed(3610)
    original = _model(sparse)
    grouped = deepcopy(original)
    expected_loss, expected_metrics, expected_trace, expected_rng = _accumulated_step(original, 2)
    loss, metrics, trace, rng = _accumulated_step(grouped, 4)
    torch.testing.assert_close(loss, expected_loss, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(rng, expected_rng, atol=0, rtol=0)
    assert len(trace) == len(expected_trace) == 12
    for (anchors, weights), (expected_anchors, expected_weights) in zip(trace, expected_trace, strict=True):
        torch.testing.assert_close(anchors, expected_anchors, atol=0, rtol=0)
        torch.testing.assert_close(weights, expected_weights, atol=3e-5, rtol=3e-5)
    for name in expected_metrics:
        torch.testing.assert_close(metrics[name], expected_metrics[name], atol=3e-5, rtol=3e-5)
    trainable = 0
    for (name, parameter), (expected_name, expected) in zip(
        grouped.named_parameters(), original.named_parameters(), strict=True,
    ):
        assert name == expected_name and parameter.requires_grad == expected.requires_grad
        if parameter.requires_grad:
            trainable += 1
            assert parameter.grad is not None and expected.grad is not None, name
            torch.testing.assert_close(
                parameter.grad, expected.grad, atol=3e-5, rtol=3e-5,
                msg=lambda message, name=name: f"gradient {name}: {message}",
            )
        else:
            assert parameter.grad is expected.grad is None
    assert trainable > 0
    assert grouped.draft_model.markov_head.markov_w2.weight.grad.count_nonzero() > 0
