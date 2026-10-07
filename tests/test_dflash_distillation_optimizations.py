
import pytest
import torch
import torch.nn.functional as F

from angelspec.models.dflash import (
    DFlashModel,
    _dpace_position_weights,
    _project_unique_teacher_logits,
    _weighted_loss_mean,
)
from angelspec.models.ops.loss import (
    lk_tv_kl_per_pos,
    streaming_student_log_normalizers_and_argmax,
    streaming_tv_kl_per_pos,
)


def test_unique_teacher_projection_matches_repeated_projection():
    torch.manual_seed(7)
    batch_size, sequence_length, hidden_size, vocab_size = 2, 9, 5, 11
    hidden = torch.randn(batch_size, sequence_length, hidden_size)
    weight = torch.randn(vocab_size, hidden_size)
    positions = torch.tensor(
        [
            [[0, 1, 2], [1, 2, 3]],
            [[2, 3, 4], [2, 4, 5]],
        ]
    )

    gather = positions.unsqueeze(-1).expand(-1, -1, -1, hidden_size)
    repeated_hidden = torch.gather(
        hidden.unsqueeze(1).expand(-1, positions.shape[1], -1, -1),
        2,
        gather,
    )
    expected = F.linear(repeated_hidden, weight).reshape(-1, vocab_size)

    unique_logits, inverse, unique_hidden_indices = _project_unique_teacher_logits(
        hidden,
        positions,
        weight,
    )

    torch.testing.assert_close(unique_logits.index_select(0, inverse), expected)
    assert unique_hidden_indices.numel() == 8
    assert unique_logits.shape == (8, vocab_size)


def test_unique_teacher_projection_excludes_unsupervised_positions_before_projection():
    torch.manual_seed(9)
    hidden = torch.randn(2, 9, 5)
    weight = torch.randn(11, 5)
    positions = torch.tensor([
        [[0, 1, 2], [1, 2, 3]],
        [[2, 3, 4], [2, 4, 5]],
    ])
    # The two references to row 0, position 1 still share one projection.
    # Row 1 must retain its batch offset when the layout becomes ragged.
    selected = torch.tensor([0, 1, 3, 6, 11])
    logits, inverse, hidden_indices = _project_unique_teacher_logits(
        hidden, positions, weight, projection_indices=selected,
    )
    torch.testing.assert_close(hidden_indices, torch.tensor([0, 1, 11, 14]))
    torch.testing.assert_close(inverse, torch.tensor([0, 1, 1, 2, 3]))
    assert logits.shape == (4, 11)
    expected = F.linear(hidden.reshape(-1, 5)[[0, 1, 1, 11, 14]], weight)
    torch.testing.assert_close(logits.index_select(0, inverse), expected)


def test_streamed_tv_kl_matches_full_softmax_values_and_gradient():
    torch.manual_seed(11)
    student = torch.randn(7, 19, requires_grad=True)
    teacher_unique = torch.randn(4, 19)
    teacher_rows = torch.tensor([0, 1, 1, 2, 0, 3, 2])
    upstream_tv = torch.randn(7)
    upstream_kl = torch.randn(7)

    reference_student = student.detach().clone().requires_grad_(True)
    reference_tv, _ = lk_tv_kl_per_pos(
        reference_student,
        teacher_unique.index_select(0, teacher_rows),
        form="tv",
    )
    reference_kl, _ = lk_tv_kl_per_pos(
        reference_student,
        teacher_unique.index_select(0, teacher_rows),
        form="kl",
    )
    reference_loss = (reference_tv * upstream_tv + reference_kl * upstream_kl).sum()
    reference_loss.backward()

    streamed_tv, streamed_kl = streaming_tv_kl_per_pos(
        student,
        teacher_unique,
        teacher_rows,
        vocab_chunk_size=5,
    )
    streamed_loss = (streamed_tv * upstream_tv + streamed_kl * upstream_kl).sum()
    streamed_loss.backward()

    torch.testing.assert_close(streamed_tv, reference_tv, atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(streamed_kl, reference_kl, atol=5e-7, rtol=2e-6)
    torch.testing.assert_close(student.grad, reference_student.grad, atol=5e-7, rtol=3e-6)


def test_streamed_tv_kl_accepts_precomputed_unique_teacher_log_normalizers():
    torch.manual_seed(12)
    teacher_unique = torch.randn(5, 23)
    teacher_rows = torch.tensor([0, 2, 2, 4, 1, 0, 3])
    upstream = torch.randn(7)
    teacher_log_normalizers = torch.logsumexp(teacher_unique.float(), dim=-1)

    local_student = torch.randn(7, 23, requires_grad=True)
    transported_student = local_student.detach().clone().requires_grad_(True)
    local_tv, local_kl = streaming_tv_kl_per_pos(
        local_student,
        teacher_unique,
        teacher_rows,
        vocab_chunk_size=7,
    )
    transported_tv, transported_kl = streaming_tv_kl_per_pos(
        transported_student,
        teacher_unique,
        teacher_rows,
        vocab_chunk_size=7,
        teacher_log_normalizers=teacher_log_normalizers,
    )
    (local_tv * upstream + local_kl).sum().backward()
    (transported_tv * upstream + transported_kl).sum().backward()

    torch.testing.assert_close(transported_tv, local_tv)
    torch.testing.assert_close(transported_kl, local_kl)
    torch.testing.assert_close(transported_student.grad, local_student.grad)


def test_streamed_student_statistics_match_full_vocabulary_reductions():
    torch.manual_seed(15)
    logits = torch.randn(7, 23, dtype=torch.bfloat16)
    # Exercise the first-index tie behavior across separate vocabulary tiles.
    logits[0, 2] = 5
    logits[0, 19] = 5
    log_normalizers, argmax_ids = streaming_student_log_normalizers_and_argmax(
        logits,
        vocab_chunk_size=7,
    )
    torch.testing.assert_close(
        log_normalizers,
        torch.logsumexp(logits.float(), dim=-1),
    )
    torch.testing.assert_close(argmax_ids, logits.argmax(dim=-1))


def test_streamed_tv_kl_reuses_precomputed_student_log_normalizers():
    torch.manual_seed(16)
    local_student = torch.randn(7, 23, requires_grad=True)
    reused_student = local_student.detach().clone().requires_grad_(True)
    teacher = torch.randn(5, 23)
    teacher_rows = torch.tensor([0, 2, 2, 4, 1, 0, 3])
    upstream = torch.randn(7)
    student_log_normalizers, _ = streaming_student_log_normalizers_and_argmax(
        reused_student,
        vocab_chunk_size=7,
    )

    local_tv, local_kl = streaming_tv_kl_per_pos(
        local_student,
        teacher,
        teacher_rows,
        vocab_chunk_size=7,
    )
    reused_tv, reused_kl = streaming_tv_kl_per_pos(
        reused_student,
        teacher,
        teacher_rows,
        vocab_chunk_size=7,
        student_log_normalizers=student_log_normalizers,
    )
    (local_tv * upstream + local_kl).sum().backward()
    (reused_tv * upstream + reused_kl).sum().backward()

    torch.testing.assert_close(reused_tv, local_tv)
    torch.testing.assert_close(reused_kl, local_kl)
    torch.testing.assert_close(reused_student.grad, local_student.grad)


def test_streamed_e2e_tv_matches_full_softmax_values_and_gradient():
    torch.manual_seed(13)
    student = torch.randn(1, 2, 3, 17, requires_grad=True)
    teacher_unique = torch.randn(4, 17)
    teacher_rows = torch.tensor([0, 1, 1, 2, 3, 2])
    valid = torch.tensor([[[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]])

    reference_student = student.detach().clone().requires_grad_(True)
    teacher_full = teacher_unique.index_select(0, teacher_rows).view_as(student)
    teacher_probabilities = torch.softmax(teacher_full.float(), dim=-1)
    student_probabilities = torch.softmax(reference_student.float(), dim=-1)
    alpha = torch.minimum(teacher_probabilities, student_probabilities).sum(dim=-1)
    alpha_effective = alpha * valid + (1.0 - valid)
    prefix = torch.cumprod(alpha_effective, dim=-1)
    gamma_valid = valid.sum(dim=-1).clamp(min=1.0)
    accepted = (prefix * valid).sum(dim=-1)
    has_valid = (valid.sum(dim=-1) > 0).float()
    reference_loss = (
        ((1.0 - accepted / gamma_valid) * has_valid).sum()
        / has_valid.sum().clamp(min=1.0)
    )
    reference_loss.backward()

    streamed_loss, streamed_accepted = DFlashModel._compute_e2e_tv_loss(
        student,
        teacher_unique,
        teacher_rows,
        valid,
        vocab_chunk_size=6,
    )
    streamed_loss.backward()

    torch.testing.assert_close(streamed_loss, reference_loss, atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(student.grad, reference_student.grad, atol=5e-7, rtol=3e-6)
    torch.testing.assert_close(
        streamed_accepted,
        (accepted * has_valid).sum() / has_valid.sum(),
    )


def test_cross_row_distillation_preserves_independent_weighted_objectives():
    values = torch.tensor(
        [[1.0, 3.0, 9.0], [2.0, 4.0, 8.0]],
        requires_grad=True,
    )
    weights = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    grouped = _weighted_loss_mean(
        values,
        weights,
        batch_size=2,
        mean_by_row=True,
    )
    expected = ((values[0] * weights[0]).sum() / weights[0].sum())
    expected = (expected + (values[1] * weights[1]).sum() / weights[1].sum()) / 2
    torch.testing.assert_close(grouped, expected)

    grouped.backward()
    expected_gradient = torch.tensor([[0.25, 0.25, 0.0], [0.5, 0.0, 0.0]])
    torch.testing.assert_close(values.grad, expected_gradient)


@pytest.mark.parametrize("batch_size", [2, 4, 6, 8])
def test_cross_row_e2e_preserves_mean_of_row_losses_and_gradients(batch_size):
    torch.manual_seed(14)
    student = torch.randn(batch_size, 2, 3, 17, requires_grad=True)
    teacher = torch.randn(batch_size * 6, 17)
    teacher_rows = torch.arange(batch_size * 6)
    valid = torch.tensor(
        [
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
        ]
    ).repeat(batch_size // 2, 1, 1)

    grouped_loss, _ = DFlashModel._compute_e2e_tv_loss(
        student,
        teacher,
        teacher_rows,
        valid,
        vocab_chunk_size=6,
        mean_by_row=True,
    )
    grouped_loss.backward()
    grouped_gradient = student.grad.detach().clone()

    reference = student.detach().clone().requires_grad_(True)
    row_losses = []
    for row in range(batch_size):
        start = row * 6
        row_loss, _ = DFlashModel._compute_e2e_tv_loss(
            reference[row : row + 1],
            teacher[start : start + 6],
            torch.arange(6),
            valid[row : row + 1],
            vocab_chunk_size=6,
        )
        row_losses.append(row_loss)
    reference_loss = torch.stack(row_losses).mean()
    reference_loss.backward()

    torch.testing.assert_close(grouped_loss, reference_loss)
    torch.testing.assert_close(grouped_gradient, reference.grad)


@pytest.mark.parametrize("batch_size", [2, 4, 6, 8])
def test_cross_row_dpace_lk_preserves_mean_of_row_losses_and_gradients(batch_size):
    torch.manual_seed(15)
    student = torch.randn(batch_size, 2, 3, 17, requires_grad=True)
    teacher = torch.randn(batch_size * 6, 17)
    teacher_rows = torch.arange(batch_size * 6)
    valid = torch.tensor([
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
    ]).repeat(batch_size // 2, 1, 1)
    # Same detached D-PACE continuation weights as the production objective,
    # with unequal supervised lengths to catch accidental token-weighted means.
    weights = valid * _dpace_position_weights(student.detach().softmax(-1)[..., 0], alpha=0.5)
    losses = DFlashModel._compute_lk_loss(
        None, student.reshape(-1, 17), teacher, teacher_row_indices=teacher_rows,
        vocab_chunk_size=6, loss_type="hybrid", eta=3.0,
    )
    grouped_loss = _weighted_loss_mean(losses, weights, batch_size=batch_size, mean_by_row=True)
    grouped_loss.backward()

    reference = student.detach().clone().requires_grad_(True)
    reference_losses = []
    for row in range(batch_size):
        start = row * 6
        row_loss = DFlashModel._compute_lk_loss(
            None, reference[row].reshape(-1, 17), teacher[start : start + 6],
            vocab_chunk_size=6, loss_type="hybrid", eta=3.0,
        )
        reference_losses.append(_weighted_loss_mean(
            row_loss, weights[row], batch_size=1, mean_by_row=False,
        ))
    reference_loss = torch.stack(reference_losses).mean()
    reference_loss.backward()
    torch.testing.assert_close(grouped_loss, reference_loss)
    torch.testing.assert_close(student.grad, reference.grad)
