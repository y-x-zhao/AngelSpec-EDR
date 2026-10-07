"""Small-support teacher preparation for distribution-aware E2E/LK losses.

The frozen teacher projection is already deduplicated by the model. Only
supported IDs/probabilities are retained, not a [positions, vocabulary] masked
tensor. Other policies and top-k ties that exceed the compact capacity use the
dense path.
"""

from __future__ import annotations

import torch

from angelspec.models.ops.edr_sparse_target import (
    EDRSparseTargetDistribution,
    _nucleus_candidates,
)
from angelspec.utils.sampling import _scale_sampling_logits, validate_sampling_parameters

_SELECTION_ROW_CHUNK_SIZE = 256
_NUCLEUS_MAX_ELEMENTS = 4_194_304


def _compact_nucleus(
    values: torch.Tensor, top_k: int, top_p: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select an unambiguous nucleus; flag rows needing vLLM's full sort.

All top-k threshold ties must fit before this result can be used. A nucleus
cut through a tie needs the full-vocabulary sort's token ordering.
Near-cutoff cumulative sums also use that path, since changing the softmax
reduction width can move a cutoff by FP32 roundoff. The conservative margin
exceeds 512 FP32 additions (the maximum compact support) plus softmax error.
    """
    threshold = values[:, top_k - 1:top_k]
    # Mirror masked_fill(values < threshold), including NaN propagation.
    in_top_k = ~(values < threshold)
    filtered = values.masked_fill(~in_top_k, -torch.inf)
    ascending_cumulative = filtered.flip(-1).softmax(-1).cumsum(-1)
    cumulative = ascending_cumulative.flip(-1)
    # vLLM casts p to FP32 before subtracting; FP32(1 - Python(p)) can
    # otherwise select a different token at the nucleus boundary.
    cutoff = 1.0 - values.new_full((1,), top_p)
    retained = in_top_k & (cumulative > cutoff)
    retained[:, 0] = True
    split_tie = (
        (retained[:, :-1] != retained[:, 1:])
        & (values[:, :-1] == values[:, 1:])
        & torch.isfinite(values[:, :-1])
    ).any(-1)
    near_cutoff = (
        in_top_k & ((cumulative - cutoff).abs() <= 1e-4)
    ).any(-1)
    return retained, split_tie | near_cutoff, in_top_k[:, -1]


_compiled_compact_nucleus = torch.compile(_compact_nucleus, dynamic=True, fullgraph=True)


@torch.no_grad()
def prepare_distill_sparse_target(
    teacher_logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> EDRSparseTargetDistribution | None:
    """Return the actual filtered teacher distribution, or request dense fallback.

The draft is used without approximation or renormalization. For small
top-k, scan the vocabulary with topk and sort/reduce only its candidates.
Rows whose nucleus splits a logit tie or lies within a rounding margin of
the cutoff use vLLM PyTorch's full-vocabulary nucleus implementation.
This preserves token identities, not just the multiset of probabilities.
    """
    validate_sampling_parameters(temperature, top_k, top_p, allow_greedy=False)
    if teacher_logits.ndim != 2 or teacher_logits.shape[1] < 1:
        raise ValueError("teacher logits must have shape [positions, non-empty vocabulary]")
    positions, vocab_size = teacher_logits.shape
    if not 0 < top_k <= 128 or top_k >= vocab_size:
        return None
    # Reserve room for BF16 threshold ties, and one extra candidate to detect
    # overflow. The dense path handles arbitrarily wide support.
    capacity = min(vocab_size, 512, 4 * (1 << (top_k - 1).bit_length()))
    candidate_count = min(vocab_size, capacity + 1)
    values = torch.empty(
        positions, candidate_count, device=teacher_logits.device, dtype=torch.float32,
    )
    candidates = torch.empty_like(values, dtype=torch.long)
    for start in range(0, positions, _SELECTION_ROW_CHUNK_SIZE):
        end = start + _SELECTION_ROW_CHUNK_SIZE
        scaled = _scale_sampling_logits(teacher_logits[start:end], temperature)
        tile_values, tile_ids = scaled.topk(candidate_count, dim=-1, sorted=True)
        values[start:end].copy_(tile_values)
        candidates[start:end].copy_(tile_ids)
        del scaled, tile_values, tile_ids

    if top_p < 1.0:
        kernel = (
            _compiled_compact_nucleus
            if teacher_logits.is_cuda and not torch.compiler.is_compiling()
            else _compact_nucleus
        )
        retained, ambiguous, overflow = kernel(values, int(top_k), float(top_p))
    else:
        retained = ~(values < values[:, top_k - 1:top_k])
        ambiguous = torch.zeros(positions, dtype=torch.bool, device=values.device)
        overflow = retained[:, -1]
    if candidate_count <= capacity:
        overflow = torch.zeros_like(overflow)
    else:
        overflow = overflow & torch.isfinite(values[:, -1])

    # One host synchronization after scanning all teacher rows. The nonzero
    # below transfers only the ambiguous row IDs.
    if candidate_count > capacity and bool(overflow.any()):
        return None
    fallback_rows = (
        ambiguous.nonzero(as_tuple=True)[0] if top_p < 1.0
        else torch.empty(0, device=values.device, dtype=torch.long)
    )
    sort_rows = max(1, _NUCLEUS_MAX_ELEMENTS // vocab_size)
    for start in range(0, fallback_rows.numel(), sort_rows):
        rows = fallback_rows[start:start + sort_rows]
        sorted_values, sorted_ids = _nucleus_candidates(
            teacher_logits.index_select(0, rows), candidate_count,
            temperature=temperature, top_k=top_k, top_p=top_p,
        )
        candidates.index_copy_(0, rows, sorted_ids)
        retained.index_copy_(0, rows, ~torch.isneginf(sorted_values))

    token_ids = candidates[:, :capacity].contiguous()
    # Normalize the same FP32 temperature-scaled logits used for selection;
    # never round the scaled logits back to BF16.
    logits = _scale_sampling_logits(teacher_logits.gather(-1, token_ids), temperature)
    logits.masked_fill_(~retained[:, :capacity], -torch.inf)
    probabilities = (logits - logits.logsumexp(-1, keepdim=True)).exp()
    return EDRSparseTargetDistribution(
        token_ids=token_ids,
        probabilities=probabilities,
        stop_probabilities=torch.zeros(positions, device=values.device, dtype=torch.float32),
        stopping_token_ids=torch.empty(0, device=values.device, dtype=torch.long),
        vocab_size=vocab_size,
    )
