"""Verify actual DFly EDR math while eliminating duplicate context projections."""

from copy import deepcopy
from unittest import mock

import pytest
import torch
from torch import nn

from angelspec.models.dfly import DFlyModel
from angelspec.models.draft.dfly import DFlyConfig, DFlyDraftModel


def _model(reuse=True, full=False):
    config = DFlyConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=61,
        max_position_embeddings=256, num_target_layers=2, target_hidden_size=32,
        target_num_hidden_layers=4, target_layer_ids=[0, 2], block_size=8,
        mask_token_id=60, enable_hidden_correction=True,
    )
    return DFlyModel(
        DFlyDraftModel(config), block_size=7, num_anchors=2, loss_objective="edr",
        edr_chunk_size=4, edr_vocab_chunk_size=17, query_includes_input_anchor=True,
        edr_full_anchor_backprop=full, edr_stop_token_ids=[1, 2],
        edr_reuse_context_cache=reuse,
    )


def _batch(rows):
    return dict(
        input_ids=torch.randint(3, 60, (rows, 16)),
        hidden_states_list=[torch.randn(rows, 16, 32) for _ in range(2)],
        loss_mask=torch.tensor([
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        ], dtype=torch.float32)[:rows],
        attention_mask=torch.ones(rows, 16),
        lm_head_weight=torch.randn(61, 32), last_hidden_states=torch.randn(rows, 16, 32),
        target_norm=nn.Identity(),
    )


@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize("mode", ["sampled", "full", "eval"])
def test_shared_context_preserves_statistics_surrogate_and_every_parameter_gradient(rows, mode):
    torch.manual_seed(251)
    model = _model(full=mode == "full")
    reference = deepcopy(model)
    reference.edr_reuse_context_cache = False
    batch = _batch(rows)
    modes = []
    prepare = model.draft_model.prepare_context_cache

    def capture(*args, **kwargs):
        modes.append(torch.is_grad_enabled())
        return prepare(*args, **kwargs)

    statistics_name = "_edr_all_row_statistics" if rows > 1 else "_edr_all_horizon_statistics"
    statistics = getattr(model, statistics_name)
    detached_sweeps = []

    def check_statistics(*args, **kwargs):
        result = statistics(*args, **kwargs)
        records = [record for row in result[0] for record in row] if rows > 1 else result[0]
        assert records
        for record in records:
            assert not record.costs.requires_grad and not record.acceptance.requires_grad
        assert not result[1].logits.requires_grad
        detached_sweeps.append(True)
        return result

    with torch.set_grad_enabled(mode != "eval"):
        torch.manual_seed(252)
        with (
            mock.patch.object(model.draft_model, "prepare_context_cache", side_effect=capture),
            mock.patch.object(model, statistics_name, side_effect=check_statistics),
        ):
            actual = model(**batch)
        torch.manual_seed(252)
        with mock.patch.object(
            reference.draft_model, "prepare_context_cache",
            wraps=reference.draft_model.prepare_context_cache,
        ) as old_cache:
            expected = reference(**batch)
        assert modes == [mode != "eval"]
        assert old_cache.call_count == (1 if mode == "eval" else 2)
        assert detached_sweeps == [True]
        for value, original in zip(actual[:5], expected[:5], strict=True):
            torch.testing.assert_close(value, original, atol=3e-5, rtol=3e-5)
        for key in expected[5]:
            torch.testing.assert_close(actual[5][key], expected[5][key], atol=3e-5, rtol=3e-5)
        if mode != "eval":
            actual[0].backward()
            expected[0].backward()
            for (name, parameter), (_, original) in zip(
                model.named_parameters(), reference.named_parameters(), strict=True,
            ):
                if original.grad is None:
                    assert parameter.grad is None, name
                else:
                    torch.testing.assert_close(parameter.grad, original.grad, atol=3e-5, rtol=3e-5)
            for parameter in (
                model.draft_model.context_proj.weight,
                model.draft_model.layer_fusion_weights,
                model.draft_model.layers[0].self_attn.k_proj.weight,
                model.draft_model.layers[0].self_attn.v_proj.weight,
            ):
                assert parameter.grad is not None and parameter.grad.count_nonzero() > 0


def test_shared_cache_is_rebuilt_after_each_optimizer_update():
    torch.manual_seed(255)
    model = _model(full=True)
    batch = _batch(2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    snapshots = []
    prepare = model.draft_model.prepare_context_cache

    def capture(*args, **kwargs):
        result = prepare(*args, **kwargs)
        snapshots.append(result.layer_caches[0].key.detach().clone())
        return result

    with mock.patch.object(model.draft_model, "prepare_context_cache", side_effect=capture):
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            model(**batch)[0].backward()
            optimizer.step()
    assert len(snapshots) == 2
    assert not torch.equal(snapshots[0], snapshots[1])


def test_empty_supervision_does_not_build_context_cache():
    model = _model()
    batch = _batch(2)
    batch["loss_mask"].zero_()
    with mock.patch.object(model.draft_model, "prepare_context_cache") as cache:
        model(**batch)[0].backward()
    cache.assert_not_called()
