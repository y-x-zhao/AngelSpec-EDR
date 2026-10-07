"""Bounded local target prefills: real tiny Qwen3, no GPUs or downloads."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from angelspec.config.train_config import config_to_flat_args, load_config
from angelspec.data.utils import pack_loss_mask
from angelspec.models.dflash import DFlashModel
from angelspec.train_single_gpu import LocalTrainingBatches, train_single_gpu
from angelspec.training.local_target import LocalTargetFeatures


@pytest.fixture(scope="module", autouse=True)
def _small_cpu_kernels():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture
def target():
    torch.manual_seed(81)
    config = Qwen3Config(
        vocab_size=61, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=512, attention_dropout=0.0, pad_token_id=0,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "sdpa"
    return Qwen3ForCausalLM(config)


def _args(**overrides):
    args = SimpleNamespace(
        dflash_loss_objective="edr", dflash_edr_cross_row_batch_size=2,
        dflash_distill_cross_row_batch_size=2, draft_accumulation_steps=96,
        single_gpu_target_batch_size=6, single_gpu_target_max_tokens=16384,
        max_seq_length=1024, min_loss_tokens=1, length_balance_optimizer_step=False,
    )
    vars(args).update(overrides)
    return args


def _records(lengths):
    records = []
    for index, length in enumerate(lengths):
        ids = (torch.arange(length) + index) % 57 + 3
        mask = torch.tensor([0] * 4 + [1] * (length - 4))
        records.append({"input_ids": ids, "packed_loss_mask": pack_loss_mask(mask)})
    return records


def _loader(args, records, features):
    loader = LocalTrainingBatches(args, features, device="cpu")
    loader.set_step(records)
    return loader


@pytest.mark.parametrize("length_balance", [False, True])
def test_real_target_prefills_keep_features_and_reduce_48_forwards_to_16(target, length_balance):
    records = _records([12 + index % 21 for index in range(96)])
    extractor = LocalTargetFeatures(target, [0, 1])
    try:
        with mock.patch.object(target.model, "forward", wraps=target.model.forward) as forward:
            expected = list(_loader(_args(
                single_gpu_target_batch_size=0,
                length_balance_optimizer_step=length_balance,
            ), records, extractor))
            assert forward.call_count == 48
            assert {tuple(call.kwargs["input_ids"].shape) for call in forward.call_args_list} == {(2, 128)}
            forward.reset_mock()
            actual = list(_loader(_args(
                length_balance_optimizer_step=length_balance,
            ), records, extractor))
            assert forward.call_count == 16
            assert {tuple(call.kwargs["input_ids"].shape) for call in forward.call_args_list} == {(6, 128)}
        assert len(actual) == len(expected) == 48
        for batch, original in zip(actual, expected, strict=True):
            assert batch.keys() == original.keys()
            assert batch["_token_counts"] == original["_token_counts"]
            assert batch["_loss_scale"] == original["_loss_scale"] == 1 / 96
            for key in ("input_ids", "attention_mask", "loss_mask"):
                torch.testing.assert_close(batch[key], original[key], atol=0, rtol=0)
                assert batch[key].shape == (2, 128)
            for key in ("hidden_states", "last_hidden_states"):
                for positions in (batch["attention_mask"].bool(), ~batch["attention_mask"].bool()):
                    torch.testing.assert_close(
                        batch[key][positions], original[key][positions], atol=2e-6, rtol=2e-5,
                    )
                assert not batch[key].requires_grad and not batch[key].is_inference()
        assert all(parameter.grad is None for parameter in target.parameters())
        assert extractor._captured is None and extractor._final_norm_input is None
    finally:
        extractor.close()


class _TinyContextDraft(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_token_id = 60
        self.embed_tokens = nn.Embedding(61, 16)
        self.proj = nn.Linear(16, 16, bias=False)

    def extract_context_feature(self, hidden_states_list):
        return torch.stack(hidden_states_list).mean(dim=0)

    def forward(self, *, noise_embedding, context_feature, **kwargs):
        return self.proj(noise_embedding + context_feature.mean(dim=1, keepdim=True))


def test_real_frozen_features_preserve_sampled_edr_anchors_loss_and_gradient(target):
    records = _records([12, 15, 13, 16, 14, 17])
    original = DFlashModel(
        _TinyContextDraft(), block_size=7, num_anchors=2, loss_objective="edr",
        edr_chunk_size=8, edr_vocab_chunk_size=17, query_includes_input_anchor=True,
        edr_stop_token_ids=[2, 60],
    )
    coalesced = deepcopy(original)
    extractor = LocalTargetFeatures(target, [0, 1])

    def run(wrapper, target_rows):
        loader = _loader(_args(
            draft_accumulation_steps=6, single_gpu_target_batch_size=target_rows,
        ), records, extractor)
        anchors, losses, metrics = [], [], {}
        prepare = wrapper._prepare_edr_gradient_query

        def capture(*args, **kwargs):
            query = prepare(*args, **kwargs)
            if query is not None:
                anchors.append(query.prefixes.clone())
            return query

        torch.manual_seed(82)
        with mock.patch.object(wrapper, "_prepare_edr_gradient_query", side_effect=capture):
            for batch in loader:
                outputs = wrapper(
                    input_ids=batch["input_ids"],
                    hidden_states_list=list(batch["hidden_states"].split(16, dim=-1)),
                    loss_mask=batch["loss_mask"], attention_mask=batch["attention_mask"],
                    last_hidden_states=batch["last_hidden_states"],
                    lm_head_weight=target.lm_head.weight, target_norm=target.model.norm,
                )
                assert batch["_loss_scale"] == 1 / 6
                loss = outputs[0] * batch["_loss_scale"]
                losses.append(loss.detach())
                loss.backward()
                for key in ("edr_weighted_cost_sum", "edr_generated_tokens", "edr_num_horizons"):
                    metrics[key] = metrics.get(key, 0) + outputs[5][key]
        return anchors, torch.stack(losses), metrics

    try:
        expected_anchors, expected_losses, expected_metrics = run(original, 0)
        anchors, losses, metrics = run(coalesced, 6)
        assert anchors and len(anchors) == len(expected_anchors)
        for actual, expected in zip(anchors, expected_anchors, strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(losses, expected_losses)
        for key in metrics:
            torch.testing.assert_close(metrics[key], expected_metrics[key])
        for actual, expected in zip(coalesced.parameters(), original.parameters(), strict=True):
            assert actual.grad is not None and expected.grad is not None
            torch.testing.assert_close(actual.grad, expected.grad, atol=1e-7, rtol=1e-4)
        assert all(parameter.grad is None for parameter in target.parameters())
    finally:
        extractor.close()


@pytest.mark.parametrize("objective", ["decay", "dpace"])
def test_real_baseline_target_prefills_preserve_96_row_training(target, objective):
    # The shared baseline wrapper isolates target batching from draft-specific
    # heads. Small and long responses exercise both masked and sampled anchors.
    records = _records([12 + index % 36 for index in range(96)])
    original = DFlashModel(
        _TinyContextDraft(), block_size=7, num_anchors=16, loss_objective=objective,
        ce_loss_alpha=0.0, l1_loss_alpha=0.0, loss_decay_gamma=0.0,
        e2e_tv_loss_weight=float(objective == "decay"),
        lk_loss_weight=float(objective == "dpace"), lk_loss_type="hybrid", lk_eta=3.0,
        dpace_alpha=0.5, distill_mean_by_row=True, fp32_lm_head=False,
        edr_vocab_chunk_size=17, query_includes_input_anchor=False,
    )
    coalesced = deepcopy(original)
    extractor = LocalTargetFeatures(target, [0, 1])

    def run(wrapper, target_rows, max_tokens):
        loader = _loader(_args(
            dflash_loss_objective=objective, dflash_distill_cross_row_batch_size=4,
            single_gpu_target_batch_size=target_rows,
            single_gpu_target_max_tokens=max_tokens, length_balance_optimizer_step=True,
        ), records, extractor)
        trace, losses = [], []
        sample = wrapper._sample_anchor_positions
        draft_forward = wrapper.draft_model.forward

        def capture_sample(*args, **kwargs):
            anchors, keep_mask = sample(*args, **kwargs)
            trace[-1]["anchors"] = anchors.clone()
            trace[-1]["keep_mask"] = keep_mask.clone()
            return anchors, keep_mask

        def capture_draft(*args, **kwargs):
            trace[-1]["draft_shapes"] = (
                tuple(kwargs["noise_embedding"].shape),
                tuple(kwargs["context_feature"].shape),
            )
            for key in ("draft_position_ids", "context_position_ids"):
                trace[-1][key] = kwargs[key].clone()
            return draft_forward(*args, **kwargs)

        torch.manual_seed(918)
        with (
            mock.patch.object(wrapper, "_sample_anchor_positions", side_effect=capture_sample) as sampling,
            mock.patch.object(wrapper.draft_model, "forward", side_effect=capture_draft) as draft,
            mock.patch.object(target.model, "forward", wraps=target.model.forward) as forward,
        ):
            for batch in loader:
                assert batch["_loss_scale"] == 4 / 96
                trace.append({
                    "keys": set(batch),
                    "token_counts": batch["_token_counts"],
                    "rng_before": torch.get_rng_state(),
                    **{key: batch[key].clone() for key in (
                        "input_ids", "attention_mask", "loss_mask",
                    )},
                })
                for key in ("hidden_states", "last_hidden_states"):
                    assert not batch[key].requires_grad and not batch[key].is_inference()
                outputs = wrapper(
                    input_ids=batch["input_ids"],
                    hidden_states_list=list(batch["hidden_states"].split(16, dim=-1)),
                    loss_mask=batch["loss_mask"], attention_mask=batch["attention_mask"],
                    last_hidden_states=batch["last_hidden_states"],
                    lm_head_weight=target.lm_head.weight, target_norm=target.model.norm,
                )
                loss = outputs[0] * batch["_loss_scale"]
                losses.append(loss.detach())
                loss.backward()
                trace[-1]["rng_after"] = torch.get_rng_state()
            assert sampling.call_count == draft.call_count == 24
            assert forward.call_count == 96 // target_rows
            assert [tuple(call.kwargs["input_ids"].shape) for call in forward.call_args_list] == [
                (target_rows, 128)
            ] * (96 // target_rows)
        return trace, torch.stack(losses), torch.get_rng_state()

    try:
        expected_trace, expected_losses, expected_rng = run(original, 4, 16384)
        trace, losses, rng = run(coalesced, 8, 32768)
        assert len(trace) == len(expected_trace) == 24
        assert sum(entry["input_ids"].shape[0] for entry in trace) == 96
        assert sum(entry["token_counts"][0] for entry in trace) == sum(
            record["input_ids"].numel() for record in records
        )
        assert sum(entry["token_counts"][1] for entry in trace) == sum(
            record["input_ids"].numel() - 4 for record in records
        )
        assert sum(entry["token_counts"][2] for entry in trace) == 96 * 128
        assert any(not entry["keep_mask"].all() for entry in trace)
        assert any(entry["keep_mask"].all() for entry in trace)
        for actual, expected in zip(trace, expected_trace, strict=True):
            assert actual["keys"] == expected["keys"]
            assert actual["token_counts"] == expected["token_counts"]
            assert actual["draft_shapes"] == expected["draft_shapes"] == ((4, 16 * 7, 16), (4, 128, 16))
            for key in ("input_ids", "attention_mask", "loss_mask"):
                assert actual[key].shape == expected[key].shape == (4, 128)
            for key in (
                "input_ids", "attention_mask", "loss_mask", "anchors", "keep_mask",
                "draft_position_ids", "context_position_ids", "rng_before", "rng_after",
            ):
                torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        torch.testing.assert_close(rng, expected_rng, atol=0, rtol=0)
        torch.testing.assert_close(losses, expected_losses, atol=1e-7, rtol=1e-5)
        assert bool((losses > 0).all())
        for actual, expected in zip(coalesced.parameters(), original.parameters(), strict=True):
            assert actual.grad is not None and expected.grad is not None
            assert actual.grad.count_nonzero() > 0
            torch.testing.assert_close(actual.grad, expected.grad, atol=1e-7, rtol=1e-4)
        assert all(parameter.grad is None for parameter in target.parameters())
        assert extractor._captured is None and extractor._final_norm_input is None
    finally:
        extractor.close()


@pytest.mark.parametrize("length_balance", [False, True])
def test_real_baseline_eight_row_prefills_do_not_cross_padding_boundaries(target, length_balance):
    # Odd numbers of four-row groups per width force a partial target prefill
    # at each boundary, even though the 32768-token cap could fit mixed widths.
    records = _records(([12] * 12 + [130] * 20) * 3)
    extractor = LocalTargetFeatures(target, [0, 1])
    args = _args(
        dflash_loss_objective="decay", dflash_distill_cross_row_batch_size=4,
        length_balance_optimizer_step=length_balance,
    )
    try:
        args.single_gpu_target_batch_size = 4
        expected = list(_loader(args, records, extractor))
        args.single_gpu_target_batch_size = 8
        args.single_gpu_target_max_tokens = 32768
        with mock.patch.object(target.model, "forward", wraps=target.model.forward) as forward:
            actual = list(_loader(args, records, extractor))
        expected_shapes = (
            [(8, 128)] * 4 + [(4, 128)] + [(8, 256)] * 7 + [(4, 256)]
            if length_balance else
            [(8, 128), (4, 128), (8, 256), (8, 256), (4, 256)] * 3
        )
        shapes = [tuple(call.kwargs["input_ids"].shape) for call in forward.call_args_list]
        assert shapes == expected_shapes
        assert len(actual) == len(expected) == 24
        assert sum(rows * width for rows, width in shapes) == sum(
            batch["_token_counts"][2] for batch in expected
        )
        for batch, original in zip(actual, expected, strict=True):
            assert batch.keys() == original.keys()
            assert batch["_token_counts"] == original["_token_counts"]
            assert batch["_loss_scale"] == original["_loss_scale"] == 4 / 96
            assert batch["input_ids"].shape == original["input_ids"].shape
            for key in ("input_ids", "attention_mask", "loss_mask"):
                torch.testing.assert_close(batch[key], original[key], atol=0, rtol=0)
            for key in ("hidden_states", "last_hidden_states"):
                torch.testing.assert_close(batch[key], original[key], atol=2e-6, rtol=2e-5)
        assert all(parameter.grad is None for parameter in target.parameters())
        assert extractor._captured is None and extractor._final_norm_input is None
    finally:
        extractor.close()


@pytest.mark.parametrize("objective", ["edr", "decay", "dpace"])
def test_original_group_loss_scales_still_give_the_96_row_mean(objective):
    records = _records([12 + index % 9 for index in range(96)])
    loader = _loader(_args(dflash_loss_objective=objective), records, lambda *args: {})
    weight = torch.tensor(0.3, requires_grad=True)
    for batch in loader:
        mask = batch["loss_mask"].float()
        row_losses = ((weight * batch["input_ids"]).square() * mask).sum(1) / mask.sum(1)
        expected_scale = (1 if objective == "edr" else 2) / 96
        assert batch["_loss_scale"] == expected_scale
        loss = row_losses.sum() if objective == "edr" else row_losses.mean()
        (loss * batch["_loss_scale"]).backward()
    reference = torch.tensor(0.3, requires_grad=True)
    torch.stack([
        (reference * row["input_ids"][4:]).square().mean() for row in records
    ]).mean().backward()
    torch.testing.assert_close(weight.grad, reference.grad)


@pytest.mark.parametrize("max_tokens, expected_shapes", [
    (0, [(4, 128), (2, 256), (4, 128), (4, 512)]),
    (1024, [(4, 128), (2, 256), (4, 128), (2, 512), (2, 512)]),
    (1, [(2, 128), (2, 128), (2, 256), (2, 128), (2, 128), (2, 512), (2, 512)]),
])
def test_outliers_never_add_padding_or_reorder_draft_groups(max_tokens, expected_shapes):
    records = _records([12, 14, 125, 128, 129, 130, 12, 14, 18, 19, 400, 410, 420, 450])
    calls = []

    def features(ids, attention):
        calls.append(tuple(ids.shape))
        return {"hidden_states": ids.unsqueeze(-1)}

    args = _args(draft_accumulation_steps=14, single_gpu_target_max_tokens=max_tokens)
    actual = list(_loader(args, records, features))
    expected = list(_loader(_args(
        draft_accumulation_steps=14, single_gpu_target_batch_size=0,
    ), records, lambda *args: {}))
    assert calls == expected_shapes
    assert sum(rows * width for rows, width in calls) == sum(batch["_token_counts"][2] for batch in expected)
    for batch, original in zip(actual, expected, strict=True):
        assert batch["_token_counts"] == original["_token_counts"]
        for key in ("input_ids", "attention_mask", "loss_mask"):
            torch.testing.assert_close(batch[key], original[key], atol=0, rtol=0)
        assert batch["hidden_states"].shape[:2] == original["input_ids"].shape
    for rows, width in calls:
        assert rows <= 6
        assert not max_tokens or rows * width <= max_tokens or rows == 2


@pytest.mark.parametrize("max_tokens, target_rows", [(512, 4), (768, 6), (0, 6)])
def test_padded_token_budget_caps_coalescing(max_tokens, target_rows):
    calls = []
    loader = _loader(_args(single_gpu_target_max_tokens=max_tokens), _records([20] * 96),
                     lambda ids, mask: calls.append(tuple(ids.shape)) or {})
    assert sum(batch["input_ids"].shape[0] for batch in loader) == 96
    assert calls == [(target_rows, 128)] * (96 // target_rows)


def test_prefills_are_lazy_on_device_views_and_consumed_once_per_step():
    calls = []

    def features(ids, mask):
        calls.append(tuple(ids.shape))
        return {"hidden_states": ids.float().unsqueeze(-1)}

    records = _records([12] * 96)
    loader = _loader(_args(), records, features)
    iterator = iter(loader)
    assert calls == []
    first = next(iterator)
    assert calls == [(6, 128)]
    assert first["hidden_states"].untyped_storage().nbytes() == 6 * 128 * 4
    for _ in range(2):
        batch = next(iterator)
        assert batch["hidden_states"].untyped_storage().data_ptr() == first["hidden_states"].untyped_storage().data_ptr()
        assert len(calls) == 1
    assert next(iterator)["input_ids"].shape == (2, 128)
    assert len(calls) == 2
    assert sum(batch["input_ids"].shape[0] for batch in iterator) == 88
    assert len(calls) == 16
    with pytest.raises(RuntimeError, match="set_step"):
        next(iter(loader))
    loader.set_step(records)
    assert len(calls) == 16
    assert sum(batch["input_ids"].shape[0] for batch in loader) == 96
    assert len(calls) == 32


def test_input_transfer_calls_drop_from_144_to_48_without_feature_host_roundtrip():
    records = _records([12] * 96)
    transfer = torch.Tensor.to

    def run(target_rows):
        loader = _loader(_args(single_gpu_target_batch_size=target_rows), records,
                         lambda ids, mask: {"hidden_states": ids.unsqueeze(-1)})
        shapes = []

        def record(tensor, *args, **kwargs):
            shapes.append(tuple(tensor.shape))
            return transfer(tensor, *args, **kwargs)

        with (
            mock.patch.object(torch.Tensor, "to", record),
            mock.patch.object(torch.Tensor, "cpu", side_effect=AssertionError("feature host roundtrip")),
        ):
            for batch in loader:
                assert batch["hidden_states"].device == batch["input_ids"].device
        return shapes

    assert run(0) == [(2, 128)] * 144
    assert run(6) == [(6, 128)] * 48


def test_target_failure_propagates_without_reusing_partial_step_features():
    calls = []
    failure = RuntimeError("synthetic target prefill failure")

    def features(ids, mask):
        calls.append(tuple(ids.shape))
        if len(calls) == 2:
            raise failure
        return {"hidden_states": ids.unsqueeze(-1)}

    records = _records([12] * 96)
    loader = _loader(_args(), records, features)
    iterator = iter(loader)
    for _ in range(3):
        next(iterator)
    with pytest.raises(RuntimeError) as error:
        next(iterator)
    assert error.value is failure and len(calls) == 2
    with pytest.raises(StopIteration):
        next(iterator)
    with pytest.raises(RuntimeError, match="set_step"):
        next(iter(loader))
    loader.set_step(records)
    assert sum(batch["input_ids"].shape[0] for batch in loader) == 96
    assert len(calls) == 18


@pytest.mark.parametrize("name", ["single_gpu_target_batch_size", "single_gpu_target_max_tokens"])
@pytest.mark.parametrize("value", [-1, True, 1.5, "6", None])
def test_prefill_limits_require_nonnegative_integers(name, value):
    with pytest.raises(ValueError, match=name):
        LocalTrainingBatches(_args(**{name: value}), lambda *args: {}, device="cpu")


@pytest.mark.parametrize("overrides", [
    {"single_gpu_target_batch_size": 3},
    {"single_gpu_target_batch_size": 98},
    {"single_gpu_target_batch_size": 8, "draft_accumulation_steps": 6},
    {"single_gpu_target_batch_size": 128, "draft_accumulation_steps": 192},
    {"draft_accumulation_steps": 95},
    {"draft_accumulation_steps": 0},
])
def test_prefill_rows_require_whole_groups_bounded_by_the_step_and_96(overrides):
    with pytest.raises(ValueError):
        LocalTrainingBatches(_args(**overrides), lambda *args: {}, device="cpu")


@pytest.mark.parametrize("name, value", [
    ("single_gpu_target_batch_size", 3),
    ("single_gpu_target_max_tokens", -1),
])
def test_invalid_prefill_settings_fail_before_gpu_or_model_startup(name, value):
    root = Path(__file__).resolve().parents[1]
    config = load_config(
        str(root / "configs/vllm_qwen3_8b_dfly_edr.yaml"),
        cli_args=[
            "model.target_model_backend=hf", "training.training_num_gpus_per_node=1",
            "training.draft_accumulation_steps=96", "training.prefetch_depth=0",
        ],
        save_snapshot=False,
    )
    args = config_to_flat_args(config)
    args.draft_model_config = str(root / "angelspec/config/dfly_qwen3_8b_draft_config.json")
    setattr(args, name, value)
    with mock.patch("angelspec.train_single_gpu.torch.cuda.device_count", side_effect=AssertionError("GPU startup reached")):
        with pytest.raises(ValueError, match=name):
            train_single_gpu(args)


def test_zero_or_missing_target_rows_keep_the_original_path_even_with_tiny_budget():
    records = _records([12] * 96)
    for setting in (None, 0, 2):
        args = _args(single_gpu_target_max_tokens=1)
        if setting is None:
            del args.single_gpu_target_batch_size
        else:
            args.single_gpu_target_batch_size = setting
        calls = []
        loader = _loader(args, records, lambda ids, mask: calls.append(tuple(ids.shape)) or {})
        with mock.patch.object(loader, "_iter_coalesced_target_prefills", side_effect=AssertionError("default path changed")):
            assert sum(batch["input_ids"].shape[0] for batch in loader) == 96
        assert calls == [(2, 128)] * 48


def test_explicit_target_limit_can_cover_an_entire_optimizer_step():
    calls = []
    loader = _loader(_args(single_gpu_target_batch_size=96), _records([12] * 96),
                     lambda ids, mask: calls.append(tuple(ids.shape)) or {})
    assert sum(batch["input_ids"].shape[0] for batch in loader) == 96
    assert calls == [(96, 128)]
