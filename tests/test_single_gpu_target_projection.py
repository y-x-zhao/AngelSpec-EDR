"""Local teacher-projection reuse: tiny random models, no checkpoint downloads."""

from unittest import mock

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from angelspec.models.dflash import (
    DFlashModel,
    _dpace_position_weights,
    _project_unique_teacher_logits,
    _weighted_loss_mean,
)
from angelspec.training.local_target import LocalTargetFeatures


@pytest.fixture
def target():
    torch.manual_seed(71)
    config = Qwen3Config(
        vocab_size=61,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        attention_dropout=0.0,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config)


def _inputs():
    ids = torch.arange(3, 43).reshape(2, 20)
    mask = torch.ones_like(ids)
    mask[1, 13:] = 0
    ids[1, 13:] = 0
    return ids, mask


def test_feature_only_extraction_performs_no_lm_head_projection(target):
    ids, mask = _inputs()
    deferred = LocalTargetFeatures(target, [0, 2])
    try:
        with mock.patch.object(
            target.lm_head, "forward", side_effect=AssertionError("unexpected LM-head call"),
        ):
            actual = deferred(ids, mask)
        assert set(actual) == {"hidden_states", "last_hidden_states"}
        for value in actual.values():
            assert not value.requires_grad and not value.is_inference()
        # Frozen target features must remain usable by draft autograd.
        draft = torch.nn.Linear(64, 32)
        draft(actual["hidden_states"]).square().mean().backward()
        assert draft.weight.grad is not None and draft.weight.grad.count_nonzero() > 0
        assert all(parameter.grad is None for parameter in target.parameters())
        assert deferred._captured is None and deferred._final_norm_input is None
    finally:
        deferred.close()


@pytest.mark.parametrize("objective", ["e2e", "lk"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_deferred_teacher_normalizers_preserve_loss_and_gradient(target, objective, dtype):
    target.to(dtype=dtype)
    ids, mask = _inputs()
    deferred = LocalTargetFeatures(target, [0, 2])
    try:
        features = deferred(ids, mask)
        # Overlapping seven-proposal blocks, with a partly padded final block.
        positions = torch.tensor([[0, 3, 10], [0, 2, 8]])[:, :, None] + torch.arange(7)
        with torch.no_grad():
            teacher, inverse, _ = _project_unique_teacher_logits(
                target.model.norm(features["last_hidden_states"]), positions,
                target.lm_head.weight,
            )
        assert teacher.shape[0] < positions.numel(), "Exercise teacher-position deduplication"
        # Precomputed reference: exact FP32 log-partition of each unique teacher row.
        eager_logz = torch.logsumexp(teacher.float(), dim=-1)
        valid = torch.ones(2, 3, 7)
        valid[1, 2, 5:] = 0
        initial = torch.randn(2, 3, 7, target.config.vocab_size, dtype=dtype) * 0.3

        def loss(student, teacher_logz):
            if objective == "e2e":
                return DFlashModel._compute_e2e_tv_loss(
                    student, teacher, inverse, valid,
                    teacher_log_normalizers=teacher_logz,
                    vocab_chunk_size=17, mean_by_row=True,
                )[0]
            with torch.no_grad():
                confidence = student.float().softmax(-1)[..., 0]
                weights = valid * _dpace_position_weights(confidence, alpha=0.5)
            per_position = DFlashModel._compute_lk_loss(
                None, student.reshape(-1, target.config.vocab_size), teacher,
                teacher_row_indices=inverse, teacher_log_normalizers=teacher_logz,
                vocab_chunk_size=17, loss_type="hybrid", eta=3.0,
            )
            return _weighted_loss_mean(per_position, weights, batch_size=2, mean_by_row=True)

        reference = initial.clone().requires_grad_(True)
        expected_loss = loss(reference, eager_logz)
        expected_loss.backward()
        student = initial.clone().requires_grad_(True)
        normalizer_inputs = []
        from angelspec.models.ops import loss as loss_ops

        normalize = loss_ops._streaming_log_normalizers

        def record_normalizer(logits, *args, **kwargs):
            normalizer_inputs.append(logits)
            return normalize(logits, *args, **kwargs)

        with mock.patch.object(loss_ops, "_streaming_log_normalizers", record_normalizer):
            actual_loss = loss(student, None)
            actual_loss.backward()
        assert sum(tensor.data_ptr() == teacher.data_ptr() for tensor in normalizer_inputs) == 1
        torch.testing.assert_close(actual_loss, expected_loss, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(student.grad, reference.grad, atol=1e-7, rtol=1e-4)
        assert all(parameter.grad is None for parameter in target.parameters())
    finally:
        deferred.close()
