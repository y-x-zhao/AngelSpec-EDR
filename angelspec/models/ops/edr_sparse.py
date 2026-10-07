"""Exact small-support EDR statistics with a full-vocabulary draft softmax.

Only positive target probabilities can contribute to ``(p - q)+``. Reducing
on that support avoids the dense target gather and rejection-mask cache. The
draft q is not truncated: its normalizer and analytical gradient cover every
vocabulary entry, and the gradient is a dense softmax term plus sparse
corrections.
"""

from __future__ import annotations

import torch

from angelspec.models.ops.edr import _streaming_log_normalizers, _use_compiled_edr_kernel
from angelspec.models.ops.edr_sparse_target import EDRSparseTargetDistribution


def _validate_sparse_statistics(draft_logits, distribution, indices, target_ids):
    if draft_logits.ndim < 2:
        raise ValueError("draft logits must include position and vocabulary dimensions")
    if draft_logits.shape[-1] != distribution.vocab_size:
        raise ValueError("EDR target and draft vocabulary sizes must match")
    if indices.shape != draft_logits.shape[:-1] or target_ids.shape != indices.shape:
        raise ValueError("target probability indices and token IDs must match draft positions")
    if distribution.probabilities.ndim != 2 or (
        distribution.token_ids.shape != distribution.probabilities.shape
    ):
        raise ValueError("sparse target IDs and probabilities must have shape [positions, support]")
    if distribution.stop_probabilities.shape != distribution.probabilities.shape[:1]:
        raise ValueError("target stop probabilities must have one value per target position")
    if distribution.token_ids.dtype != torch.long:
        raise ValueError("sparse target token IDs must have dtype long")
    if any(t.device != draft_logits.device for t in (
        distribution.token_ids, distribution.probabilities, distribution.stop_probabilities,
        distribution.stopping_token_ids, indices, target_ids,
    )):
        raise ValueError("EDR logits, indices, and token IDs must be on the same device")


def _support_statistics(
    draft_logits, logz, support_ids, probabilities, stop_probabilities, stop_ids,
    target_rows, target_ids, logit_scale,
):
    flat_rows = target_rows.reshape(-1)
    support_shape = (*target_rows.shape, support_ids.shape[-1])
    ids = support_ids.index_select(0, flat_rows).view(support_shape)
    p = probabilities.index_select(0, flat_rows).view(support_shape)
    q = torch.exp(draft_logits.gather(-1, ids).float() * logit_scale - logz.unsqueeze(-1))
    non_stopping = ~ids.unsqueeze(-1).eq(stop_ids).any(-1)
    active = (p > q) & non_stopping
    non_stop_mass = (1.0 - stop_probabilities.index_select(0, flat_rows).view_as(target_rows)).clamp(0.0, 1.0)
    positive = non_stop_mass > 0
    safe_mass = torch.where(positive, non_stop_mass, torch.ones_like(non_stop_mass))
    costs = torch.where(
        positive,
        ((p - q).clamp_min(0.0) * non_stopping).sum(-1) / safe_mass,
        torch.zeros_like(logz),
    )
    # Negative sparse derivative w.r.t. logits before the softmax row term,
    # stored as a compact [draft positions, support] tensor.
    support_cost = torch.where(
        active & positive.unsqueeze(-1), -q / safe_mass.unsqueeze(-1), torch.zeros_like(q),
    )
    cost_dot = -support_cost.sum(-1)
    p_realized = torch.where(ids == target_ids.unsqueeze(-1), p, torch.zeros_like(p)).sum(-1)
    q_realized = torch.exp(
        draft_logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1).float() * logit_scale - logz
    )
    denominator = p_realized.clamp_min(torch.finfo(p.dtype).tiny)
    ratio = q_realized / denominator
    acceptance = torch.minimum(ratio, torch.ones_like(ratio))
    slope = torch.where(
        ratio < 1.0, torch.ones_like(ratio),
        torch.where(ratio > 1.0, torch.zeros_like(ratio), torch.full_like(ratio, 0.5)),
    )
    # Multiply before dividing: saturated ratios may overflow, but 0*q/p is 0.
    acceptance_scale = slope * q_realized / denominator
    return costs, acceptance, ids, support_cost, cost_dot, acceptance_scale


def _dense_gradient_tile(logits, logz, row_scale, logit_scale, output):
    q = torch.exp(logits.float() * logit_scale - logz.unsqueeze(-1))
    output.copy_(q * row_scale.unsqueeze(-1))


def _support_gradient(
    draft_logits, logz, support_ids, support_cost, cost_dot, acceptance_scale,
    target_ids, grad_costs, grad_acceptance, logit_scale,
):
    row_scale = (grad_costs * cost_dot - grad_acceptance * acceptance_scale) * logit_scale
    correction = grad_acceptance * acceptance_scale * logit_scale
    q_support = torch.exp(
        draft_logits.gather(-1, support_ids).float() * logit_scale - logz.unsqueeze(-1)
    )
    realized_support = support_ids == target_ids.unsqueeze(-1)
    support_gradient = (
        q_support * row_scale.unsqueeze(-1)
        + support_cost * (grad_costs * logit_scale).unsqueeze(-1)
        + realized_support * correction.unsqueeze(-1)
    )
    q_realized = torch.exp(
        draft_logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1).float() * logit_scale - logz
    )
    realized_gradient = (
        q_realized * row_scale + correction
        + (support_cost * realized_support).sum(-1) * grad_costs * logit_scale
    )
    return row_scale, support_gradient.to(draft_logits.dtype), realized_gradient.to(draft_logits.dtype)


_compiled_support_statistics = torch.compile(_support_statistics, dynamic=True, fullgraph=True)
_compiled_dense_gradient_tile = torch.compile(_dense_gradient_tile, dynamic=True, fullgraph=True)
_compiled_support_gradient = torch.compile(_support_gradient, dynamic=True, fullgraph=True)


def _forward_statistics(
    draft_logits, support_ids, probabilities, stop_probabilities, stop_ids,
    target_rows, target_ids, vocab_chunk_size, logit_scale,
):
    logz = _streaming_log_normalizers(draft_logits, vocab_chunk_size, logit_scale)
    kernel = _compiled_support_statistics if _use_compiled_edr_kernel(draft_logits) else _support_statistics
    return logz, kernel(
        draft_logits, logz, support_ids, probabilities, stop_probabilities, stop_ids,
        target_rows, target_ids, logit_scale,
    )


class _SparseEDRStatistics(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, draft_logits, support_ids, probabilities, stop_probabilities, stop_ids,
        target_rows, target_ids, vocab_chunk_size, logit_scale,
    ):
        logz, (costs, acceptance, ids, support_cost, cost_dot, acceptance_scale) = _forward_statistics(
            draft_logits, support_ids, probabilities, stop_probabilities, stop_ids,
            target_rows, target_ids, vocab_chunk_size, logit_scale,
        )
        ctx.save_for_backward(draft_logits, logz, ids, support_cost, cost_dot, acceptance_scale, target_ids)
        ctx.vocab_chunk_size = vocab_chunk_size
        ctx.logit_scale = logit_scale
        ctx.set_materialize_grads(False)
        return costs, acceptance

    @staticmethod
    def backward(ctx, grad_costs, grad_acceptance):
        if grad_costs is None and grad_acceptance is None:
            return (None,) * 9
        logits, logz, ids, support_cost, cost_dot, acceptance_scale, target_ids = ctx.saved_tensors
        if grad_costs is None:
            grad_costs = torch.zeros_like(logz)
        if grad_acceptance is None:
            grad_acceptance = torch.zeros_like(logz)
        compiled = _use_compiled_edr_kernel(logits)
        correction_kernel = _compiled_support_gradient if compiled else _support_gradient
        row_scale, support_gradient, realized_gradient = correction_kernel(
            logits, logz, ids, support_cost, cost_dot, acceptance_scale,
            target_ids, grad_costs, grad_acceptance, ctx.logit_scale,
        )
        # Write BF16/FP16 directly from each FP32 tile, never a full FP32 q or
        # gradient. Overwrite support entries with their complete FP32 result
        # rather than scatter-add onto an already rounded dense contribution.
        gradient = torch.empty_like(logits)
        tile_kernel = _compiled_dense_gradient_tile if compiled else _dense_gradient_tile
        for start in range(0, logits.shape[-1], ctx.vocab_chunk_size):
            tile = slice(start, start + ctx.vocab_chunk_size)
            tile_kernel(logits[..., tile], logz, row_scale, ctx.logit_scale, gradient[..., tile])
        gradient.scatter_(-1, ids, support_gradient)
        gradient.scatter_(-1, target_ids.unsqueeze(-1), realized_gradient.unsqueeze(-1))
        return gradient, None, None, None, None, None, None, None, None


def sparse_edr_distribution_statistics(
    draft_logits: torch.Tensor,
    distribution: EDRSparseTargetDistribution,
    target_probability_indices: torch.Tensor,
    target_ids: torch.Tensor,
    vocab_chunk_size: int,
    *,
    draft_temperature: float,
):
    _validate_sparse_statistics(draft_logits, distribution, target_probability_indices, target_ids)
    if vocab_chunk_size < 1:
        raise ValueError("vocab_chunk_size must be >= 1")
    arguments = (
        draft_logits, distribution.token_ids, distribution.probabilities,
        distribution.stop_probabilities, distribution.stopping_token_ids,
        target_probability_indices.long(), target_ids.long(), int(vocab_chunk_size),
        1.0 / draft_temperature,
    )
    if torch.is_grad_enabled() and draft_logits.requires_grad:
        return _SparseEDRStatistics.apply(*arguments)
    with torch.no_grad():
        _, values = _forward_statistics(*arguments)
        return values[0], values[1]


@torch.no_grad()
def greedy_sparse_edr_distribution_statistics(draft_logits, distribution, indices, target_ids):
    _validate_sparse_statistics(draft_logits, distribution, indices, target_ids)
    greedy_ids = draft_logits.argmax(-1)
    flat_rows = indices.long().reshape(-1)
    shape = (*indices.shape, distribution.token_ids.shape[-1])
    ids = distribution.token_ids.index_select(0, flat_rows).view(shape)
    probabilities = distribution.probabilities.index_select(0, flat_rows).view(shape)
    p_greedy = torch.where(ids == greedy_ids.unsqueeze(-1), probabilities, torch.zeros_like(probabilities)).sum(-1)
    mass = (1.0 - distribution.stop_probabilities.index_select(0, flat_rows).view_as(indices)).clamp(0.0, 1.0)
    safe_mass = torch.where(mass > 0, mass, torch.ones_like(mass))
    is_stop = greedy_ids.unsqueeze(-1).eq(distribution.stopping_token_ids).any(-1)
    costs = torch.where(
        mass > 0,
        torch.where(is_stop, torch.ones_like(mass), (1.0 - p_greedy / safe_mass).clamp(0.0, 1.0)),
        torch.zeros_like(mass),
    )
    return costs, greedy_ids.eq(target_ids).float()
