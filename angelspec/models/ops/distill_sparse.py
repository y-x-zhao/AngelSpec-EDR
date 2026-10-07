"""Exact TV/KL on compact teacher support and a full-vocabulary draft.

The teacher's sampling policy may discard almost every vocabulary entry, but
the draft remains an unfiltered softmax. Only its normalizer and analytical
backward need a vocabulary pass; forward reduces over the teacher support.
"""

from __future__ import annotations

import math

import torch

from angelspec.models.ops.edr import _streaming_log_normalizers, _use_compiled_edr_kernel
from angelspec.models.ops.edr_sparse import (
    _compiled_dense_gradient_tile,
    _dense_gradient_tile,
)
from angelspec.models.ops.edr_sparse_target import EDRSparseTargetDistribution


def _support_statistics(logits, logz, token_ids, probabilities, rows, compute_kl, logit_scale):
    ids = token_ids.index_select(0, rows)
    p = probabilities.index_select(0, rows)
    logq = logits.gather(-1, ids).float() * logit_scale - logz.unsqueeze(-1)
    q = logq.exp()
    mass = p.sum(-1)
    # For normalized p and q this is 1 - sum(min(p, q)). Retaining the
    # measured teacher mass also accounts for its FP32 normalization rounding.
    tv = 0.5 * (1.0 + mass) - torch.minimum(p, q).sum(-1)
    # Subtract the constant 1/2 derivative on the complement of the support.
    # At p == q, sign(0) == 0 preserves the dense abs() zero subgradient.
    # Zero-probability padding and underflowed q contribute exactly zero.
    support_tv = 0.5 * (torch.sign(q - p) - 1.0) * q
    tv_row_scale = -support_tv.sum(-1)
    kl = torch.zeros_like(tv)
    if compute_kl:
        positive = p > 0
        logp = torch.where(positive, p, 1.0).log()
        safe_logq = torch.where(positive, logq, 0.0)
        kl = (p * (logp - safe_logq)).sum(-1)
    return tv, kl, ids, p, support_tv, tv_row_scale, mass


def _support_gradient(logits, logz, ids, p, support_tv, tv_row_scale, mass,
                      grad_tv, grad_kl, logit_scale):
    row_scale = (grad_tv * tv_row_scale + grad_kl * mass) * logit_scale
    q = (logits.gather(-1, ids).float() * logit_scale - logz.unsqueeze(-1)).exp()
    gradient = (
        q * row_scale.unsqueeze(-1)
        + (
            grad_tv.unsqueeze(-1) * support_tv
            - grad_kl.unsqueeze(-1) * p
        ) * logit_scale
    )
    return row_scale, gradient.to(logits.dtype)


_compiled_support_statistics = torch.compile(_support_statistics, dynamic=True, fullgraph=True)
_compiled_support_gradient = torch.compile(_support_gradient, dynamic=True, fullgraph=True)


class _SparseTVKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, token_ids, probabilities, rows, logz,
                vocab_chunk_size, compute_kl, logit_scale):
        kernel = _compiled_support_statistics if _use_compiled_edr_kernel(logits) else _support_statistics
        tv, kl, ids, p, support_tv, tv_row_scale, mass = kernel(
            logits, logz, token_ids, probabilities, rows, compute_kl, logit_scale,
        )
        ctx.save_for_backward(logits, logz, ids, p, support_tv, tv_row_scale, mass)
        ctx.vocab_chunk_size = vocab_chunk_size
        ctx.compute_kl = compute_kl
        ctx.logit_scale = logit_scale
        ctx.set_materialize_grads(False)
        return tv, kl

    @staticmethod
    def backward(ctx, grad_tv, grad_kl):
        if grad_tv is None and grad_kl is None:
            return (None,) * 8
        logits, logz, ids, p, support_tv, tv_row_scale, mass = ctx.saved_tensors
        if grad_tv is None:
            grad_tv = torch.zeros_like(logz)
        if grad_kl is None or not ctx.compute_kl:
            grad_kl = torch.zeros_like(logz)
        compiled = _use_compiled_edr_kernel(logits)
        correction_kernel = _compiled_support_gradient if compiled else _support_gradient
        row_scale, support_gradient = correction_kernel(
            logits, logz, ids, p, support_tv, tv_row_scale, mass,
            grad_tv, grad_kl, ctx.logit_scale,
        )
        gradient = torch.empty_like(logits)
        tile_kernel = _compiled_dense_gradient_tile if compiled else _dense_gradient_tile
        for start in range(0, logits.shape[-1], ctx.vocab_chunk_size):
            tile = slice(start, start + ctx.vocab_chunk_size)
            tile_kernel(logits[:, tile], logz, row_scale, ctx.logit_scale, gradient[:, tile])
        # IDs are unique within each row, including zero-probability padding.
        # Overwrite with the complete FP32 calculation before final BF16/FP16
        # rounding: scatter-add onto a rounded dense term loses cancellation.
        gradient.scatter_(-1, ids, support_gradient)
        return gradient, None, None, None, None, None, None, None


def sparse_tv_kl_per_pos(
    student_logits: torch.Tensor,
    target: EDRSparseTargetDistribution,
    teacher_row_indices: torch.Tensor,
    *,
    vocab_chunk_size: int,
    student_log_normalizers: torch.Tensor | None = None,
    compute_kl: bool = True,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-position TV and optional KL(p||q), with detached teacher p.

    ``target`` already embodies the teacher sampling policy. The student uses
    its complete vocabulary at ``temperature``; any supplied normalizers must
    be logsumexp(student_logits / temperature). No target-support truncation
    is applied to q or its gradient. TV has the same equality subgradient as
    0.5 * abs(p - q), up to FP32 softmax normalization rounding.
    """
    if student_logits.ndim != 2 or student_logits.shape[-1] < 1:
        raise ValueError("student logits must have shape [rows, non-empty vocabulary]")
    if student_logits.shape[-1] != target.vocab_size:
        raise ValueError("student and teacher vocabulary sizes must match")
    if not student_logits.is_floating_point():
        raise ValueError("student logits must be floating point")
    if target.probabilities.ndim != 2 or target.token_ids.shape != target.probabilities.shape:
        raise ValueError("sparse teacher IDs and probabilities must have shape [rows, support]")
    if target.token_ids.dtype != torch.long:
        raise ValueError("sparse teacher token IDs must have dtype long")
    if not target.probabilities.is_floating_point():
        raise ValueError("sparse teacher probabilities must be floating point")
    if any(t.device != student_logits.device for t in (target.token_ids, target.probabilities)):
        raise ValueError("student logits and sparse teacher tensors must be on the same device")
    if teacher_row_indices.shape != student_logits.shape[:1]:
        raise ValueError("teacher_row_indices must have one entry per student row")
    if vocab_chunk_size < 1:
        raise ValueError(f"vocab_chunk_size must be >= 1, got {vocab_chunk_size}")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("distribution-aware e2e/LK training requires finite temperature > 0")
    logit_scale = 1.0 / temperature
    rows = teacher_row_indices.to(device=student_logits.device, dtype=torch.long)
    probabilities = target.probabilities.detach().float()
    if student_log_normalizers is None:
        with torch.no_grad():
            student_log_normalizers = _streaming_log_normalizers(
                student_logits, vocab_chunk_size, logit_scale,
            )
    if student_log_normalizers.shape != student_logits.shape[:1]:
        raise ValueError("student_log_normalizers must have one value per student row")
    logz = student_log_normalizers.detach().to(device=student_logits.device, dtype=torch.float32)
    arguments = (
        student_logits, target.token_ids, probabilities, rows, logz,
        int(vocab_chunk_size), bool(compute_kl), logit_scale,
    )
    if torch.is_grad_enabled() and student_logits.requires_grad:
        return _SparseTVKL.apply(*arguments)
    kernel = _compiled_support_statistics if _use_compiled_edr_kernel(student_logits) else _support_statistics
    with torch.no_grad():
        tv, kl, *_ = kernel(
            student_logits, logz, target.token_ids, probabilities, rows, bool(compute_kl), logit_scale,
        )
        return tv, kl
