"""Internal CPU plans avoid CUDA host reads and keep the injection checks."""

from copy import deepcopy
from unittest import mock

import pytest
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from angelspec.models.dflash import _EDRValidatedAnchors
from angelspec.models.ops.edr import prepare_edr_target_distribution
from tests.test_edr_valid_projection import model_for


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class NoCudaHostReads(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if args and isinstance(args[0], torch.Tensor) and args[0].is_cuda:
            if func in (torch.ops.aten._local_scalar_dense.default, torch.ops.aten.nonzero.default):
                raise AssertionError(f"Unexpected CUDA host read: {func}")
            if func == torch.ops.aten.index.Tensor:
                assert not any(index is not None and index.dtype == torch.bool for index in args[1])
        return func(*args, **(kwargs or {}))


def make_plan(device):
    return _EDRValidatedAnchors.from_cpu(
        torch.tensor([[1, 0, 7], [2, 3, 0]]), torch.tensor([[0, 0, 6], [0, 1, 0]]),
        torch.tensor([[8, 0, 8], [3, 3, 0]]),
        torch.tensor([[True, False, True], [True, True, False]]),
        width=7, sequence_length=18, device=torch.device(device),
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_internal_anchor_and_packing_execution_does_not_read_cuda_predicates(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    model = model_for("dspark", device, head_dim=16)
    plan = make_plan(device)
    valid = torch.zeros(2, 3, 7, dtype=torch.bool, device=device)
    valid.view(-1)[plan.projection_indices] = True
    hidden = torch.randn(2, 21, 32, device=device, requires_grad=True)
    distribution = prepare_edr_target_distribution(torch.randn(10, 61, device=device), 17)
    # Inspect real eager GPU operations without dispatch-mode/compiler plumbing
    # obscuring host reads; separate parity tests exercise compiled kernels.
    with (
        NoCudaHostReads(),
        mock.patch("angelspec.models.ops.edr._use_compiled_edr_kernel", return_value=False),
    ):
        anchors, keep = model._sample_anchor_positions(
            18, torch.ones(2, 18, device=device), torch.device(device),
            injected_anchors=plan.anchors, injected_keep_mask=plan.keep_mask,
            _validated_anchors=plan,
        )
        assert anchors is plan.anchors and keep is plan.keep_mask
        result = model._edr_project_valid_statistics(
            draft_hidden=hidden, lm_head_weight=torch.randn(61, 32, device=device),
            prev_token_ids=torch.zeros_like(valid, dtype=torch.long),
            target_distribution=distribution,
            target_probability_indices=torch.zeros_like(valid, dtype=torch.long),
            target_ids=torch.zeros_like(valid, dtype=torch.long), valid_mask=valid,
            draft_temperature=0.7, cache_rejection_mask=False,
            projection_indices=plan.projection_indices,
        )
        sum(value.sum() for value in result).backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


@pytest.mark.parametrize("alter", ["anchors", "keep", "length", "rows", "device"])
def test_internal_plan_cannot_be_reused_with_other_tensors_or_shapes(alter):
    model = model_for("dspark")
    plan = make_plan("cpu")
    anchors = plan.anchors.clone() if alter == "anchors" else plan.anchors
    keep = plan.keep_mask.clone() if alter == "keep" else plan.keep_mask
    length = 19 if alter == "length" else 18
    loss_mask = torch.ones(1 if alter == "rows" else 2, length)
    with pytest.raises(ValueError, match="do not match"):
        model._sample_anchor_positions(
            length, loss_mask, torch.device("meta" if alter == "device" else "cpu"), injected_anchors=anchors,
            injected_keep_mask=keep, _validated_anchors=plan,
        )


@pytest.mark.parametrize("case", ["negative", "past_end", "padding", "document"])
def test_external_injected_anchors_remain_checked(case):
    model = model_for("dspark")
    anchors = torch.tensor([[1]])
    attention = torch.ones(1, 10)
    docs = torch.zeros(1, 10, dtype=torch.long)
    if case == "negative":
        anchors[0, 0] = -1
    elif case == "past_end":
        anchors[0, 0] = 9
    elif case == "padding":
        attention[0, 2] = 0
    else:
        docs[0, 2] = 1
    with pytest.raises(ValueError):
        model._sample_anchor_positions(
            10, torch.ones(1, 10), torch.device("cpu"),
            attention_mask=attention, ctx_doc_ids=docs,
            injected_anchors=anchors, injected_keep_mask=torch.ones_like(anchors, dtype=torch.bool),
        )


@pytest.mark.parametrize("anchor,prefix,length", [(-1, 0, 3), (17, 0, 3), (1, -1, 3), (1, 3, 3)])
def test_cpu_plan_rejects_invalid_kept_indices(anchor, prefix, length):
    with pytest.raises(ValueError, match="inside their validated horizons"):
        _EDRValidatedAnchors.from_cpu(
            torch.tensor([[anchor]]), torch.tensor([[prefix]]), torch.tensor([[length]]),
            torch.tensor([[True]]), width=7, sequence_length=18, device=torch.device("cpu"),
        )


def batch_for(device):
    torch.manual_seed(4402)
    weight = torch.randn(61, 32, device=device)
    hidden = torch.randn(4, 18, 32, device=device)
    ids = torch.randint(0, 61, (4, 18), device=device)
    ids[:, 1:] = (hidden[:, :-1] @ weight.T).argmax(-1)
    mask = torch.tensor([
        [0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],  # Final-token boundary.
        [0] * 18,  # Empty row mixed with active horizons.
        [0, 0, 1] + [0] * 15,  # Boundary-only horizon.
    ], dtype=torch.float32, device=device)
    return dict(
        input_ids=ids, hidden_states_list=[torch.randn(4, 18, 32, device=device) for _ in range(2)],
        last_hidden_states=hidden, lm_head_weight=weight, target_norm=nn.Identity(),
        loss_mask=mask,
        attention_mask=torch.tensor([[1] * 16 + [0] * 2, [1] * 18, [1] * 18, [1] * 18], device=device),
        ctx_doc_ids=torch.tensor([[0] * 6 + [1] * 10 + [-1] * 2, [0] * 11 + [1] * 7, [0] * 18, [0] * 18], device=device),
    )


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("mode", ["sampled", "full", "eval"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_validated_plans_match_checked_route_including_packed_and_empty_rows(sparse, mode, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(4401)
    # FlexAttention's CUDA kernels require head dimensions of at least 16.
    model = model_for("dspark", device, head_dim=16)
    model.edr_sparse_target = sparse
    model.edr_full_anchor_backprop = mode == "full"
    reference = deepcopy(model)
    batch = batch_for(device)
    prepare_old = reference._edr_all_row_statistics
    gradient_old = reference._edr_gradient_row_surrogate
    project = model._edr_project_valid_statistics
    phases = set()

    def checked_statistics(**kwargs):
        kwargs["_validated_horizons"] = False
        return prepare_old(**kwargs)

    def checked_gradients(**kwargs):
        kwargs["_validated_horizons"] = False
        return gradient_old(**kwargs)

    def assert_cpu_plan(**kwargs):
        expected = kwargs["valid_mask"].reshape(-1).nonzero(as_tuple=True)[0]
        assert kwargs["projection_indices"] is not None
        torch.testing.assert_close(kwargs["projection_indices"], expected, atol=0, rtol=0)
        phases.add(torch.is_grad_enabled())
        return project(**kwargs)

    def run(current):
        torch.manual_seed(4403)
        return current(**batch)

    with torch.set_grad_enabled(mode != "eval"):
        with mock.patch.object(model, "_edr_project_valid_statistics", side_effect=assert_cpu_plan):
            actual = run(model)
        rng = torch.get_rng_state().clone()
        with (
            mock.patch.object(reference, "_edr_all_row_statistics", side_effect=checked_statistics),
            mock.patch.object(reference, "_edr_gradient_row_surrogate", side_effect=checked_gradients),
        ):
            expected = run(reference)
        assert phases == ({False} if mode == "eval" else {False, True})
        torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
        for value, old in zip(actual[:5], expected[:5], strict=True):
            torch.testing.assert_close(value, old, atol=3e-5, rtol=3e-5)
        for name in expected[5]:
            torch.testing.assert_close(actual[5][name], expected[5][name], atol=3e-5, rtol=3e-5)
        if mode != "eval":
            actual[0].backward()
            expected[0].backward()
            for (name, parameter), (_, original) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
                if original.grad is None:
                    assert parameter.grad is None, name
                else:
                    torch.testing.assert_close(parameter.grad, original.grad, atol=3e-5, rtol=3e-5, msg=name)


@pytest.mark.parametrize("anchor_attention", [0.0, -1.0, float("nan")])
def test_horizon_initialization_anchor_is_checked_before_internal_bypass(anchor_attention):
    model = model_for("dspark")
    batch = batch_for("cpu")
    batch["attention_mask"] = batch["attention_mask"].float()
    batch["attention_mask"][0, 0] = anchor_attention
    with pytest.raises(ValueError, match="initialization anchor is masked as padding"):
        model(**batch)


@pytest.mark.parametrize("anchor_doc", [-1, 2])
def test_horizon_initialization_anchor_cannot_cross_document_boundary(anchor_doc):
    model = model_for("dspark")
    batch = batch_for("cpu")
    batch["ctx_doc_ids"][0, 0] = anchor_doc
    with pytest.raises(ValueError, match="document"):
        model(**batch)
