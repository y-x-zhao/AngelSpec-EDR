"""Compact, exact teacher distributions for small-top-k single-GPU EDR.

The teacher still projects over its complete vocabulary to discover its support.
Only supported token IDs and probabilities survive that bounded projection;
overlapping draft positions do not need a dense copy of teacher probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch
import torch.nn.functional as F

from angelspec.utils.sampling import (
    _scale_sampling_logits,
    _sorted_sampling_logits,
    validate_sampling_parameters,
)

if TYPE_CHECKING:
    from angelspec.models.ops.edr import EDRTargetDistribution


_PROJECTION_ROW_CHUNK_SIZE = 256
_NUCLEUS_MAX_ELEMENTS = 4_194_304
_MAX_COMPACT_SUPPORT = 512


def _nucleus_candidates(
    logits: torch.Tensor,
    candidate_count: int,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    _legacy_arithmetic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the sorted nucleus tail without a vocabulary scatter or second topk.

    Use the same full-vocabulary sort, softmax, and cumulative sum as
    sampling_logits, including its ordering of tied values. Computing the
    nucleus over compact candidates instead could pick different tied IDs.
    Only the returned candidates, not the sort workspace, survive this call.
    """
    scaled = _scale_sampling_logits(logits, temperature, _legacy_arithmetic=_legacy_arithmetic)
    sorted_logits, indices = _sorted_sampling_logits(
        scaled, top_k=top_k, top_p=top_p, _legacy_arithmetic=_legacy_arithmetic,
    )
    values = sorted_logits[:, -candidate_count:].flip(-1)
    candidates = indices[:, -candidate_count:].flip(-1)
    return values, candidates


@dataclass(frozen=True)
class EDRSparseTargetDistribution:
    """Detached teacher constants, padded with zero probabilities.

    Each row's token IDs are unique, including zero-probability padding slots.
    Top-k threshold ties can make the support wider than the configured k.
    """

    token_ids: torch.Tensor
    probabilities: torch.Tensor
    stop_probabilities: torch.Tensor
    stopping_token_ids: torch.Tensor
    vocab_size: int

    @property
    def num_positions(self) -> int:
        return self.probabilities.shape[0]

    @property
    def device(self) -> torch.device:
        return self.probabilities.device


@torch.inference_mode(False)
@torch.no_grad()
def project_edr_sparse_target_distribution(
    target_hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    vocab_chunk_size: int,
    *,
    stopping_token_ids: Sequence[int] | torch.Tensor = (),
    temperature: float = 1.0,
    top_k: int,
    top_p: float = 1.0,
) -> EDRSparseTargetDistribution | EDRTargetDistribution:
    """Project bounded position tiles and retain the exact filtered support.

    All BF16 top-k threshold ties survive, so the support can exceed k. The
    compact capacity is ``4 * k`` rounded up to a power of two (at most 512
    tokens), and one additional candidate detects overflow. A single overflow
    decision is transferred to the CPU after all position tiles; overflow
    falls back to the dense implementation.

    For top-p, bounded subtiles use the full-vocabulary sort of
    ``sampling_logits``, whose order among tied logits determines the selected
    token IDs; sorting only the compact candidates could select different IDs.
    The sorting tile size is bounded independently of the projection tile, so
    the teacher GEMM keeps large tiles. The retained representation is compact
    for either filtering policy.
    """
    # Import lazily so edr.py can dispatch on the sparse dataclass without a
    # circular import during module initialization.
    from angelspec.models.ops.edr import (
        _validated_stopping_token_ids,
        prepare_edr_target_distribution,
    )

    validate_sampling_parameters(temperature, top_k, top_p, allow_greedy=False)
    if top_k < 1:
        raise ValueError("sparse EDR target projection requires a positive top_k")
    top_k = int(top_k)
    if target_hidden.ndim != 2 or lm_head_weight.ndim != 2:
        raise ValueError("target hidden states and LM-head weight must both be two-dimensional")
    if target_hidden.shape[-1] != lm_head_weight.shape[-1]:
        raise ValueError("target hidden states and LM-head weight must share their hidden dimension")
    if lm_head_weight.shape[0] < 1 or lm_head_weight.shape[1] < 1:
        raise ValueError("LM-head weight must have nonempty vocabulary and hidden dimensions")
    if target_hidden.device != lm_head_weight.device:
        raise ValueError("target hidden states and LM-head weight must be on the same device")
    if target_hidden.dtype != lm_head_weight.dtype:
        raise ValueError("target hidden states and LM-head weight must have the same dtype")
    if not target_hidden.is_floating_point():
        raise ValueError("target hidden states and LM-head weight must be floating point")
    if vocab_chunk_size < 1:
        raise ValueError(f"vocab_chunk_size must be >= 1, got {vocab_chunk_size}")

    positions, vocab_size = target_hidden.shape[0], lm_head_weight.shape[0]
    stop_ids = _validated_stopping_token_ids(
        stopping_token_ids, vocab_size, target_hidden.device,
    )

    def dense_fallback() -> EDRTargetDistribution:
        return prepare_edr_target_distribution(
            F.linear(target_hidden, lm_head_weight),
            vocab_chunk_size,
            stopping_token_ids=stopping_token_ids,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    if top_k > 128 or top_k >= vocab_size:
        return dense_fallback()

    capacity = min(vocab_size, _MAX_COMPACT_SUPPORT, 4 * (1 << (top_k - 1).bit_length()))
    candidate_count = min(vocab_size, capacity + 1)
    token_ids = torch.empty(
        positions, capacity, device=target_hidden.device, dtype=torch.long,
    )
    probabilities = torch.empty(
        positions, capacity, device=target_hidden.device, dtype=torch.float32,
    )
    row_chunk_size = _PROJECTION_ROW_CHUNK_SIZE
    selection_row_chunk_size = row_chunk_size
    if top_p < 1:
        selection_row_chunk_size = min(
            row_chunk_size, max(1, _NUCLEUS_MAX_ELEMENTS // vocab_size),
        )
    overflow_flags = []
    for start in range(0, positions, row_chunk_size):
        projected = F.linear(target_hidden[start:start + row_chunk_size], lm_head_weight)
        for offset in range(0, projected.shape[0], selection_row_chunk_size):
            row_logits = projected[offset:offset + selection_row_chunk_size]
            if top_p < 1:
                # This includes top-k and preserves the full-vocabulary sort
                # order when nucleus filtering splits a group of tied values.
                values, candidates = _nucleus_candidates(
                    row_logits, candidate_count,
                    temperature=temperature, top_k=top_k, top_p=top_p,
                )
            else:
                selection_logits = _scale_sampling_logits(row_logits, temperature)
                values, candidates = selection_logits.topk(candidate_count, dim=-1, sorted=True)
                del selection_logits

            if top_p < 1:
                retained = ~torch.isneginf(values[:, :capacity])
                if candidate_count > capacity:
                    overflow_flags.append((~torch.isneginf(values[:, capacity])).any())
            else:
                threshold = values[:, top_k - 1:top_k]
                retained = values[:, :capacity] >= threshold
                if candidate_count > capacity:
                    overflow_flags.append((
                        (values[:, capacity] >= threshold[:, 0])
                        & ~torch.isneginf(values[:, capacity])
                    ).any())
            selected_ids = candidates[:, :capacity]
            # ``values`` already carry the sampler's FP32 temperature division;
            # only retained candidates enter the softmax.
            selected_logits = values[:, :capacity].masked_fill(~retained, -torch.inf)
            selected_probabilities = selected_logits.softmax(dim=-1, dtype=torch.float32)
            tile_start = start + offset
            tile_end = tile_start + row_logits.shape[0]
            token_ids[tile_start:tile_end].copy_(selected_ids)
            probabilities[tile_start:tile_end].copy_(selected_probabilities)
            del row_logits, values, candidates
        # Release the vocabulary-sized tile before projecting the next one or
        # allocating a dense fallback. All retained output is compact.
        del projected

    if overflow_flags and bool(torch.stack(overflow_flags).any()):
        return dense_fallback()

    if stop_ids.numel():
        is_stopping = (token_ids.unsqueeze(-1) == stop_ids).any(dim=-1)
        stop_probabilities = (probabilities * is_stopping).sum(dim=-1).clamp_(0.0, 1.0)
    else:
        stop_probabilities = torch.zeros(
            positions, device=target_hidden.device, dtype=torch.float32,
        )
    return EDRSparseTargetDistribution(
        token_ids=token_ids,
        probabilities=probabilities,
        stop_probabilities=stop_probabilities,
        stopping_token_ids=stop_ids,
        vocab_size=vocab_size,
    )
