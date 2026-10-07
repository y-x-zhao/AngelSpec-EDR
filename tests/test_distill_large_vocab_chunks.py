"""Check production-sized vocabulary tiles using tiny synthetic row counts."""

import pytest
import torch

from angelspec.models.dflash import (
    DFlashModel,
    _dpace_position_weights,
    _weighted_loss_mean,
)
from angelspec.models.ops.edr import _streaming_log_normalizers


@pytest.mark.parametrize("objective", ["e2e", "lk"])
@pytest.mark.parametrize("chunk_size", [65536, 131072])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA is not available"
    )),
])
def test_large_vocab_tiles_match_full_softmax(objective, chunk_size, dtype, device):
    # Both chunk sizes have a partial final tile at Qwen3's vocabulary size.
    vocab_size = 151936
    generator = torch.Generator().manual_seed(719)
    initial = (torch.randn(2, 2, 7, vocab_size, generator=generator) * 0.4).to(
        device=device, dtype=dtype
    )
    teacher = (torch.randn(18, vocab_size, generator=generator) * 0.3).to(
        device=device, dtype=dtype
    )
    # Two overlapping blocks per sequence share frozen teacher positions.
    teacher_rows = torch.tensor([
        *range(7), *range(2, 9), *range(9, 16), *range(11, 18),
    ], device=device)
    valid = torch.ones(2, 2, 7, device=device)
    valid[1, 1, 3:] = 0
    student = initial.clone().requires_grad_(True)
    reference = initial.clone().requires_grad_(True)
    teacher_logz = _streaming_log_normalizers(teacher, chunk_size)
    student_logz = _streaming_log_normalizers(student, chunk_size)
    for logits, logz in ((teacher, teacher_logz), (student, student_logz)):
        # Independently check each tiled partition against full FP64 reduction.
        torch.testing.assert_close(
            logz.double(), logits.double().logsumexp(-1), atol=2e-6, rtol=0,
        )
    teacher_prob = teacher.float().softmax(-1)[teacher_rows].view_as(initial)
    reference_log_prob = reference.float().log_softmax(-1)
    reference_prob = reference_log_prob.exp()
    tv = 0.5 * (reference_prob - teacher_prob).abs().sum(-1)

    # TV is not differentiable at p=q, and FP32 partitions can select different
    # abs branches for near-ties. Check every probability against full softmax,
    # then use the streamed arithmetic's branch in the independent
    # full-probability autograd reference. This keeps the tight all-element
    # gradient check for near-tie coordinates as well.
    with torch.no_grad():
        streamed_p = (teacher.float() - teacher_logz[:, None]).exp()[teacher_rows].view_as(initial)
        streamed_q = (student.float() - student_logz[..., None]).exp()
        torch.testing.assert_close(streamed_p, teacher_prob, atol=0, rtol=1e-5)
        torch.testing.assert_close(streamed_q, reference_prob, atol=0, rtol=1e-5)
        reference_sign = (reference_prob - teacher_prob).sign()
        streamed_sign = (streamed_q - streamed_p).sign()
        changed = reference_sign != streamed_sign
        assert torch.all(
            ~changed | ((reference_prob - teacher_prob).abs() <= 2e-5 * teacher_prob)
        ), "TV branches may differ only inside the FP32 probability error bound"
    # Zero-valued linear correction selects that same subgradient; the
    # reference forward value remains the ordinary full-softmax objective.
    tv = tv + (
        0.5 * (streamed_sign - reference_sign) * (reference_prob - reference_prob.detach())
    ).sum(-1)

    if objective == "e2e":
        effective_alpha = (1.0 - tv) * valid + (1.0 - valid)
        accepted = (effective_alpha.cumprod(-1) * valid).sum(-1)
        reference_loss = (1.0 - accepted / valid.sum(-1)).mean()
        loss, acceptance = DFlashModel._compute_e2e_tv_loss(
            student, teacher, teacher_rows, valid,
            teacher_log_normalizers=teacher_logz,
            student_log_normalizers=student_logz.reshape(-1),
            vocab_chunk_size=chunk_size, mean_by_row=True,
        )
        torch.testing.assert_close(acceptance, accepted.mean(), atol=1e-6, rtol=1e-5)
    else:
        weights = valid * _dpace_position_weights(
            # Detached continuation weights, fixed for both implementations.
            torch.linspace(0.4, 0.9, 28, device=device).view_as(valid), alpha=0.5,
        )
        kl = (teacher_prob * (teacher_prob.clamp_min(1e-9).log() - reference_log_prob)).sum(-1)
        lam = torch.exp(-3.0 * (1.0 - tv).detach().clamp(0.0, 1.0))
        reference_per_position = lam * kl + (1.0 - lam) * tv
        # Independent row-normalized reference, without the production helper.
        reference_loss = (
            (reference_per_position * weights).sum((1, 2)) / weights.sum((1, 2))
        ).mean()
        per_position = DFlashModel._compute_lk_loss(
            None, student.reshape(-1, vocab_size), teacher,
            teacher_row_indices=teacher_rows, vocab_chunk_size=chunk_size,
            teacher_log_normalizers=teacher_logz,
            student_log_normalizers=student_logz.reshape(-1),
            loss_type="hybrid", eta=3.0,
        )
        loss = _weighted_loss_mean(per_position, weights, batch_size=2, mean_by_row=True)

    # Scale both gradients so a small absolute tolerance cannot hide missing
    # terms across a large vocabulary; also explicitly require nonzero grads.
    (100.0 * reference_loss).backward()
    (100.0 * loss).backward()
    assert student.grad is not None and student.grad.count_nonzero() > 0
    torch.testing.assert_close(loss, reference_loss, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(
        student.grad, reference.grad,
        atol=1e-9, rtol=0.01 if dtype == torch.bfloat16 else 1e-4,
    )
