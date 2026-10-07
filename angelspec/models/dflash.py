# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""DFlash training model: wraps the DFlash draft model with training-specific logic.

Handles anchor sampling, block-causal mask generation, noise input construction,
and cross-entropy loss with exponential decay weighting.
"""

import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from angelspec.config.distillation import distillation_distribution_aware_enabled
from angelspec.models.ops.dflash_layout import build_dflash_proposal_layout
from angelspec.models.ops.edr import (
    EDRDynamicProgram,
    EDRHorizon,
    EDRSurrogateCoefficients,
    EDRTargetDistribution,
    edr_surrogate_sum,
    edr_surrogate_sum_from_coefficients,
    exact_edr_dynamic_programs,
    extract_edr_horizons,
    greedy_edr_distribution_statistics,
    prepare_edr_target_distribution,
    prepare_sampled_edr_surrogate_coefficients,
    sample_edr_round_starts,
    sampled_edr_surrogate_sum,
    streaming_edr_distribution_statistics,
)
from angelspec.models.ops.edr_sparse_target import EDRSparseTargetDistribution
from angelspec.models.ops.flex_attention import (
    compile_friendly_create_block_mask,
    isolated_flex_attention_fallback,
)
from angelspec.models.ops.loss import (
    _kl_variant_b,
    lk_tv_kl_per_pos,
    streaming_student_log_normalizers_and_argmax,
    streaming_tv_kl_per_pos,
)
from angelspec.utils.edr_work import (
    edr_combined_gradient_anchor_slots,
    edr_gradient_anchor_slots,
)
from angelspec.utils.logging import logger
from angelspec.utils.sampling import validate_sampling_parameters

_VALID_DFLASH_LOSS_OBJECTIVES = {"decay", "dpace", "edr"}


@dataclass(frozen=True)
class _EDRHorizonBatchEntry:
    """Per-horizon offsets into one row's concatenated target distribution."""

    horizon: EDRHorizon
    target_probability_start: int
    target_distribution_offset: int
    target_probability_count: int


@dataclass(frozen=True)
class _EDRHorizonStatistics:
    """Detached statistics for one horizon inside a row-level EDR batch."""

    entry: _EDRHorizonBatchEntry
    costs: torch.Tensor
    acceptance: torch.Tensor


@dataclass(frozen=True)
class _EDRGradientQuery:
    """One horizon's gradient query and exact detached coefficients."""

    entry: _EDRHorizonBatchEntry
    dynamic_program: EDRDynamicProgram
    prefixes: torch.Tensor
    inclusion_probabilities: Optional[torch.Tensor]
    keep_mask: torch.Tensor
    inverse_pps_scale: Optional[torch.Tensor]
    coefficients: Optional[EDRSurrogateCoefficients] = None
    # CPU copy of ``prefixes`` (PPS runs on CPU), so the gradient pass can plan
    # packing without a device synchronization.
    cpu_prefixes: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class _EDRValidatedAnchors:
    """Internal CPU-checked anchors bound to the exact tensors being executed.

    Used only by builders over already validated EDR horizons. User-injected
    anchors take the fully checked path.
    """

    anchors: torch.Tensor
    keep_mask: torch.Tensor
    projection_indices: torch.Tensor
    sequence_length: int

    @classmethod
    def from_cpu(cls, anchors, prefixes, lengths, keep, *, width, sequence_length, device):
        if any(value.device.type != "cpu" for value in (anchors, prefixes, lengths, keep)):
            raise ValueError("Internal EDR anchor plans must be built on CPU")
        if anchors.ndim != 2 or any(value.shape != anchors.shape for value in (prefixes, lengths, keep)):
            raise ValueError("Internal EDR anchor plan shapes must match")
        if keep.dtype != torch.bool or anchors.dtype != torch.long:
            raise TypeError("Internal EDR anchor plans require int64 anchors and boolean validity")
        invalid = keep & (
            (anchors < 0) | (anchors >= sequence_length - 1)
            | (prefixes < 0) | (prefixes >= lengths)
            | (anchors - prefixes + lengths >= sequence_length)
        )
        if bool(invalid.any()):
            raise ValueError("Internal EDR anchors must remain inside their validated horizons")
        valid = keep.unsqueeze(-1) & (
            prefixes.unsqueeze(-1) + torch.arange(1, width + 1) <= lengths.unsqueeze(-1)
        )
        indices = valid.reshape(-1).nonzero(as_tuple=True)[0]
        return cls(
            torch.where(keep, anchors, 0).to(device, non_blocking=True),
            keep.to(device, non_blocking=True),
            indices.to(device, non_blocking=True), sequence_length,
        )


@dataclass
class EDRPreparedBatch:
    """One group's statistics and reusable context, valid until this step ends.

    No all-anchor activation graph is retained. The context cache may carry its
    small shared projection graph, and the detached teacher logits remain live
    until the selected-anchor forward consumes this object.
    """

    inputs: dict[str, Any]
    horizons: Sequence[EDRHorizon]
    statistics: list[_EDRHorizonStatistics]
    target_distribution: Optional[EDRTargetDistribution | EDRSparseTargetDistribution]
    context_cache: Any
    degenerate_horizons: int
    total_horizon_tokens: int
    owner: int
    consumed: bool = False


def _project_unique_teacher_logits(
    normalized_hidden_states: torch.Tensor,
    teacher_label_indices: torch.Tensor,
    lm_head_weight: torch.Tensor,
    projection_indices: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project each referenced teacher position exactly once.

    Overlapping DFlash anchors frequently ask for the same target position.
    ``teacher_label_indices`` is local to each batch row, so flatten a
    ``(batch, position)`` pair into one global hidden-state index, deduplicate
    those indices, and retain the inverse map needed by position-wise losses.

    Returns ``(unique_logits, inverse_indices, unique_hidden_indices)``. The
    inverse indices are flat and reconstruct the original
    ``teacher_label_indices`` layout when used with ``index_select``, or only
    its selected positions when ``projection_indices`` is supplied.
    """
    if normalized_hidden_states.ndim != 3:
        raise ValueError("normalized teacher hidden states must have shape [B, S, H]")
    if teacher_label_indices.ndim != 3:
        raise ValueError("teacher label indices must have shape [B, blocks, proposals]")
    if teacher_label_indices.shape[0] != normalized_hidden_states.shape[0]:
        raise ValueError("teacher hidden states and label indices must share batch size")

    batch_size, sequence_length, hidden_size = normalized_hidden_states.shape
    row_offsets = (
        torch.arange(batch_size, device=teacher_label_indices.device, dtype=torch.long)
        * sequence_length
    ).view(batch_size, 1, 1)
    flat_hidden_indices = (teacher_label_indices.long() + row_offsets).reshape(-1)
    if projection_indices is not None:
        flat_hidden_indices = flat_hidden_indices.index_select(0, projection_indices)
    unique_hidden_indices, inverse_indices = torch.unique(
        flat_hidden_indices,
        sorted=True,
        return_inverse=True,
    )
    unique_hidden = normalized_hidden_states.reshape(-1, hidden_size).index_select(
        0, unique_hidden_indices
    )
    unique_logits = F.linear(unique_hidden, lm_head_weight)
    return unique_logits, inverse_indices, unique_hidden_indices


def _weighted_loss_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
    *,
    batch_size: int,
    mean_by_row: bool,
) -> torch.Tensor:
    """Return the pooled token mean, or the mean of per-row means when ``mean_by_row``."""
    values = values.reshape(batch_size, -1)
    weights = weights.reshape(batch_size, -1).to(values.dtype)
    numerators = (values * weights).sum(dim=-1)
    denominators = weights.sum(dim=-1).clamp_min(1e-6)
    if mean_by_row:
        return (numerators / denominators).mean()
    return numerators.sum() / denominators.sum().clamp_min(1e-6)


def _edr_gradient_anchor_slots(selected_count: int, num_anchors: int) -> int:
    """Module-level wrapper of :func:`edr_gradient_anchor_slots`."""
    return edr_gradient_anchor_slots(selected_count, num_anchors)


def _dflash_loss_chunk() -> int:
    """Row budget for chunked draft-logit projection (0/unset => no chunking).

    Mirrors ANGELSPEC_MTP_LOSS_CHUNK. Only the decay + no-distill forward path
    honors it (the production path); dpace / distillation / subclass heads keep
    the single full-vocab projection.
    """
    return int(os.environ.get("ANGELSPEC_DFLASH_LOSS_CHUNK", "0") or 0)


def _dpace_position_weights(confidences: torch.Tensor, alpha: float) -> torch.Tensor:
    """Compute detached D-PACE weights from per-position draft confidences."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"dflash_dpace_alpha must be in [0, 1], got {alpha}")

    with torch.no_grad():
        smoothed = (1.0 - alpha) * confidences.float() + alpha
        prefix_products = torch.cumprod(smoothed, dim=-1)
        weights = torch.flip(
            torch.cumsum(torch.flip(prefix_products, dims=[-1]), dim=-1),
            dims=[-1],
        )
        return weights.to(dtype=confidences.dtype)


def _create_dflash_mask_mod(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    ctx_len: int,
    block_size: int,
    ctx_doc_ids: Optional[torch.Tensor] = None,
    pad_query_self_attention: bool = False,
):
    """Create a mask_mod function for DFlash block-causal attention.

    KV: [Context (ctx_len tokens) | Block_0 | Block_1 | ... | Block_{n-1}]
    Q:  [Block_0 | Block_1 | ... | Block_{n-1}]

    Rules:
      1. Each block sees context strictly before its anchor (kv_idx < anchor_pos)
      2. Intra-block attention is bidirectional
      3. Different blocks are invisible to each other
      4. Invalid blocks (block_keep_mask=False) see nothing, or only each
         query's own draft key when pad_query_self_attention is enabled.
         These queries remain excluded from supervision in either case.
      5. Sequence packing (ctx_doc_ids given): a block additionally only sees
         context tokens in the SAME document as its anchor (and non-padding),
         so packed docs never leak across boundaries.
    """
    num_anchors = anchor_positions.shape[1]
    packed = ctx_doc_ids is not None
    # Dynamo promotes the shape-derived ``ctx_len`` to a SymInt. FlexAttention's
    # FLASH/FA4 lowering cannot inline a symbolic scalar captured by mask_mod,
    # but it accepts a captured scalar tensor on the attention device. Using a
    # tensor keeps dynamic-length inputs on the FLASH backend without one graph
    # per sequence length.
    ctx_len_tensor = torch.scalar_tensor(
        ctx_len,
        dtype=torch.long,
        device=anchor_positions.device,
    )
    num_anchors_tensor = torch.scalar_tensor(
        num_anchors,
        dtype=torch.long,
        device=anchor_positions.device,
    )

    def dflash_mask_mod(b, h, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        # Kernels may evaluate padded query tiles past the last anchor block;
        # clamp the gather index and treat those queries as invalid blocks.
        q_in_range = q_block_id < num_anchors_tensor
        safe_q_block_id = torch.where(q_in_range, q_block_id, 0)
        anchor_pos = anchor_positions[b, safe_q_block_id]

        is_context = kv_idx < ctx_len_tensor
        mask_context = is_context & (kv_idx < anchor_pos)

        if packed:
            safe_anchor_pos = torch.where(
                (anchor_pos >= 0) & (anchor_pos < ctx_len_tensor), anchor_pos, 0
            )
            a_doc = ctx_doc_ids[b, safe_anchor_pos]
            # Clamp context index so the gather stays in-bounds for block-region
            # kv positions; only used when is_context is True.
            kv_ctx = torch.where(is_context, kv_idx, torch.zeros_like(kv_idx))
            kv_doc = ctx_doc_ids[b, kv_ctx]
            mask_context = mask_context & (a_doc >= 0) & (kv_doc == a_doc)

        is_draft = kv_idx >= ctx_len_tensor
        kv_block_id = (kv_idx - ctx_len_tensor) // block_size
        mask_draft = is_draft & (q_block_id == kv_block_id)

        is_valid_block = block_keep_mask[b, safe_q_block_id] & q_in_range
        allowed = (mask_context | mask_draft) & is_valid_block
        if pad_query_self_attention:
            # A private self-edge keeps padding finite without exposing context
            # or changing any kept query's attention (including packed rows).
            allowed = allowed | (~is_valid_block & (kv_idx == ctx_len_tensor + q_idx))
        return allowed

    suffix = "_packed" if packed else ""
    if pad_query_self_attention:
        suffix += "_pad_self"
    dflash_mask_mod.__name__ = f"dflash_mask_A{num_anchors}_B{block_size}{suffix}"
    return dflash_mask_mod


def _build_dflash_block_mask(
    *,
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    ctx_len: int,
    block_size: int,
    device: torch.device,
    ctx_doc_ids: Optional[torch.Tensor] = None,
):
    """Build DFlash's sparse attention metadata without a dense Q-by-KV grid."""
    draft_len = anchor_positions.shape[1] * block_size
    sparse_block_size = (128, 128)
    sm100 = device.type == "cuda" and torch.cuda.get_device_capability(device)[0] == 10
    if sm100:
        # SM100 FA4 processes two 128-query stages per sparse row, so metadata
        # uses 256-query blocks (KV blocks stay 128). This is kernel tiling
        # only and is independent of the DFlash ``block_size``.
        sparse_block_size = (256, 128)
        # FA4's SM100 empty-tile epilogue can propagate stale NaN/Inf from
        # unwritten TMEM into fully masked query rows. Giving invalid queries a
        # private self-edge avoids that path; valid attention and supervision
        # are unaffected.
    mask_mod = _create_dflash_mask_mod(
        anchor_positions=anchor_positions,
        block_keep_mask=block_keep_mask,
        ctx_len=ctx_len,
        block_size=block_size,
        ctx_doc_ids=ctx_doc_ids,
        pad_query_self_attention=sm100,
    )
    return compile_friendly_create_block_mask(
        mask_mod=mask_mod,
        B=anchor_positions.shape[0],
        H=None,
        Q_LEN=draft_len,
        KV_LEN=ctx_len + draft_len,
        device=device,
        BLOCK_SIZE=sparse_block_size,
        # The eager builder evaluates the mask over the full Q-by-KV grid, which
        # can exceed 100M positions per EDR chunk; the compiled builder produces
        # only the sparse block metadata.
        _compile=True,
    )


class DFlashModel(nn.Module):
    """DFlash training wrapper.

    Wraps the DFlash draft model with training-specific logic:
      - Random anchor sampling with block_keep_mask
      - Block-causal attention mask via FlexAttention
      - Noise input construction (anchor + MASK)
      - Cross-entropy loss with exponential decay weighting
      - Per-position loss_mask application
    """

    def __init__(
        self,
        draft_model,
        block_size: int = 16,
        num_anchors: int = 512,
        loss_decay_gamma: float = 7.0,
        fp32_lm_head: bool = True,
        gate_entropy_weight: float = 0.0,
        loss_objective: str = "decay",
        dpace_alpha: float = 0.5,
        ce_loss_alpha: float = 1.0,
        l1_loss_alpha: float = 0.0,
        kl_loss_weight: float = 0.0,
        kl_topk: int = 10,
        lk_loss_weight: float = 0.0,
        lk_loss_type: str = "hybrid",
        lk_eta: float = 3.0,
        e2e_tv_loss_weight: float = 0.0,
        edr_chunk_size: int = 64,
        edr_vocab_chunk_size: int = 16384,
        edr_dp_workers: int = 1,
        query_includes_input_anchor: bool = False,
        edr_full_anchor_backprop: bool = False,
        edr_stop_token_ids: Sequence[int] = (),
        edr_temperature: float = 1.0,
        edr_top_k: int = -1,
        edr_top_p: float = 1.0,
        distill_mean_by_row: bool = False,
        edr_reuse_context_cache: bool = False,
        edr_rejection_cache_max_mb: int = 0,
        distill_distribution_aware: Optional[bool] = None,
        distill_temperature: float = 1.0,
        distill_top_k: int = -1,
        distill_top_p: float = 1.0,
    ):
        super().__init__()
        loss_objective = loss_objective.lower()
        if loss_objective not in _VALID_DFLASH_LOSS_OBJECTIVES:
            valid = ", ".join(sorted(_VALID_DFLASH_LOSS_OBJECTIVES))
            raise ValueError(
                f"Unknown DFlash loss objective {loss_objective!r}; expected one of {valid}"
            )
        if not 0.0 <= dpace_alpha <= 1.0:
            raise ValueError(f"dflash_dpace_alpha must be in [0, 1], got {dpace_alpha}")

        self.draft_model = draft_model
        # block_size is the number of learned draft proposals per block.
        if block_size < 1:
            raise ValueError(f"dflash block_size must be >= 1, got {block_size}")
        self.block_size = block_size
        if num_anchors < 1:
            raise ValueError(f"dflash_num_anchors must be >= 1, got {num_anchors}")
        self.num_anchors = num_anchors
        self.loss_decay_gamma = loss_decay_gamma
        # fp32 draft logits before CE: FSDP2's bf16 compute otherwise rounds ~2% of
        # small-margin argmax decisions that CE grad and the acc metric depend on.
        self.fp32_lm_head = fp32_lm_head
        # Optional sparsity penalty on the gated_sum layer-selection gate (plan B).
        # Default 0 => no-op; only takes effect on a DFlashGatedDraftModel.
        self.gate_entropy_weight = gate_entropy_weight
        # Loss objective + optional distillation terms (unified loss architecture).
        self.loss_objective = loss_objective
        self.dpace_alpha = dpace_alpha
        self.ce_loss_alpha = float(ce_loss_alpha)
        self.l1_loss_alpha = float(l1_loss_alpha)
        # KL / LK distillation against the target's true last-layer logits. Both
        # are convex-mix coefficients in [0, 1] against CE; LK and KL are mutually
        # exclusive (LK takes precedence when both are set).
        self.kl_loss_weight = float(kl_loss_weight)
        self.kl_topk = int(kl_topk)
        self.lk_loss_weight = float(lk_loss_weight)
        self.lk_loss_type = str(lk_loss_type)
        self.lk_eta = float(lk_eta)
        # End-to-end multi-step TV loss (independent term added to the total;
        # combinable with KL/LK). 0 => off.
        self.e2e_tv_loss_weight = float(e2e_tv_loss_weight)
        self.distill_mean_by_row = bool(distill_mean_by_row)
        self.distill_distribution_aware = distillation_distribution_aware_enabled(
            distill_distribution_aware,
            loss_objective=loss_objective,
            e2e_tv_loss_weight=self.e2e_tv_loss_weight,
            lk_loss_weight=self.lk_loss_weight,
        )
        if self.distill_distribution_aware:
            if distill_temperature == 0:
                raise ValueError("Distribution-aware e2e/LK training requires temperature > 0")
            validate_sampling_parameters(
                distill_temperature, distill_top_k, distill_top_p, allow_greedy=False,
            )
        self.distill_temperature = float(distill_temperature) if self.distill_distribution_aware else 1.0
        self.distill_top_k = int(distill_top_k) if self.distill_distribution_aware else -1
        self.distill_top_p = float(distill_top_p) if self.distill_distribution_aware else 1.0
        self._distill_sampling_kwargs = dict(
            distribution_aware=self.distill_distribution_aware,
            temperature=self.distill_temperature,
            top_k=self.distill_top_k,
            top_p=self.distill_top_p,
        )
        if edr_chunk_size < 1:
            raise ValueError(f"dflash_edr_chunk_size must be >= 1, got {edr_chunk_size}")
        self.edr_chunk_size = int(edr_chunk_size)
        if edr_vocab_chunk_size < 1:
            raise ValueError(
                f"dflash_edr_vocab_chunk_size must be >= 1, got {edr_vocab_chunk_size}"
            )
        self.edr_vocab_chunk_size = int(edr_vocab_chunk_size)
        validate_sampling_parameters(edr_temperature, edr_top_k, edr_top_p)
        self.edr_temperature = float(edr_temperature)
        self.edr_top_k = int(edr_top_k)
        self.edr_top_p = float(edr_top_p)
        # Compact top-k target support for EDR statistics; set by the
        # single-GPU trainer (see train_single_gpu.py).
        self.edr_sparse_target = False
        self.edr_stop_token_ids = tuple(sorted(set(int(i) for i in edr_stop_token_ids)))
        if any(token_id < 0 for token_id in self.edr_stop_token_ids):
            raise ValueError("EDR stopping token IDs must be non-negative")
        if edr_dp_workers < 1:
            raise ValueError(f"dflash_edr_dp_workers must be >= 1, got {edr_dp_workers}")
        self.edr_dp_workers = int(edr_dp_workers)
        self.edr_full_anchor_backprop = bool(edr_full_anchor_backprop)
        self.edr_reuse_context_cache = bool(edr_reuse_context_cache)
        if edr_rejection_cache_max_mb < 0:
            raise ValueError("dflash_edr_rejection_cache_max_mb must be >= 0")
        self.edr_rejection_cache_max_bytes = int(edr_rejection_cache_max_mb) * 1024**2
        if self.edr_full_anchor_backprop and self.loss_objective != "edr":
            raise ValueError(
                "edr_full_anchor_backprop is supported only by the EDR objective"
            )
        # The query layout is a property of the drafter architecture, so every
        # objective scores the same proposal slots.
        self.query_includes_input_anchor = bool(query_includes_input_anchor)
        # With an input anchor, each query block has block_size + 1 slots: slot 0
        # carries the committed anchor token and the remaining block_size slots
        # are learned proposals. Only the learned slots are scored.
        self.query_block_size = self.block_size + int(self.query_includes_input_anchor)
        self.proposal_width = self.block_size
        self.edr_proposal_width = self.proposal_width

    def _select_learned_query_states(
        self,
        draft_hidden: torch.Tensor,
        n_blocks: int,
    ) -> torch.Tensor:
        """Drop the input-anchor query slot from the backbone output.

        The full block runs through the draft backbone because query slots
        attend bidirectionally within a block; only projection and loss drop
        slot 0. Accepts either the query width or the already-compacted
        proposal width.
        """
        if n_blocks < 1 or draft_hidden.shape[1] % n_blocks:
            raise ValueError("draft hidden length must be divisible by n_blocks")
        query_width = draft_hidden.shape[1] // n_blocks
        if query_width == self.proposal_width:
            return draft_hidden
        if not self.query_includes_input_anchor or query_width != self.query_block_size:
            raise ValueError(
                "draft hidden width must match the configured query or proposal width "
                f"({query_width} not in {{{self.query_block_size}, {self.proposal_width}}})"
            )
        bsz, _, hidden_size = draft_hidden.shape
        return (
            draft_hidden.view(bsz, n_blocks, self.query_block_size, hidden_size)[:, :, 1:, :]
            .reshape(bsz, n_blocks * self.proposal_width, hidden_size)
        )

    def _edr_gradient_anchor_slots(self, selected_count: int) -> int:
        """Choose the gradient query width for ``selected_count`` sampled starts.

        ``num_anchors`` is the maximum sample size. Short horizons have fewer
        positive-occupancy starts, so the query is padded only to the smallest
        bucket that holds the sample. The fixed bucket set bounds the number of
        compiled attention shapes and keeps every selected Horvitz--Thompson
        contribution.
        """
        return _edr_gradient_anchor_slots(selected_count, self.num_anchors)

    def _edr_combined_gradient_anchor_slots(
        self,
        selected_count: int,
        batch_capacity: int,
    ) -> int:
        """Choose the shape of the one padded tail after horizon compaction."""
        return edr_combined_gradient_anchor_slots(
            selected_count,
            self.num_anchors,
            batch_capacity,
        )

    def _prepare_edr_gradient_query(
        self,
        entry: _EDRHorizonBatchEntry,
        dynamic_program: EDRDynamicProgram,
        device: torch.device,
    ) -> Optional[_EDRGradientQuery]:
        """Finish the CPU DP/PPS handoff before transferring selected fields."""
        coefficients = None
        if self.edr_full_anchor_backprop:
            # Full-anchor mode backpropagates every round start; PPS is skipped.
            prefixes = torch.arange(entry.horizon.ordinary_length, device=device)
            cpu_prefixes = torch.arange(entry.horizon.ordinary_length)
            probabilities = None
            inverse_scale = None
        else:
            # One systematic PPS draw per horizon. DP results stay on CPU until
            # the FP64 Horvitz--Thompson coefficients of the selected starts
            # are computed; only those coefficients move to the device.
            sample = sample_edr_round_starts(
                dynamic_program.round_start_probabilities[:entry.horizon.ordinary_length],
                self.num_anchors,
            )
            prefixes = sample.indices
            cpu_prefixes = prefixes if prefixes.device.type == "cpu" else None
            probabilities = sample.selected_inclusion_probabilities
            inverse_scale = sample.inverse_pps_scale
            if not prefixes.numel():
                return None
            coefficients = prepare_sampled_edr_surrogate_coefficients(
                dynamic_program,
                prefixes,
                probabilities,
                torch.ones_like(prefixes, dtype=torch.bool),
                inverse_scale,
            ).to(device)
            prefixes = prefixes.to(device)
        if not prefixes.numel():
            return None
        return _EDRGradientQuery(
            entry=entry,
            dynamic_program=dynamic_program,
            prefixes=prefixes,
            inclusion_probabilities=probabilities,
            keep_mask=torch.ones_like(prefixes, dtype=torch.bool),
            inverse_pps_scale=inverse_scale,
            coefficients=coefficients,
            cpu_prefixes=cpu_prefixes,
        )

    def _sample_anchor_positions(
        self,
        seq_len: int,
        loss_mask: torch.Tensor,
        device: torch.device,
        attention_mask: Optional[torch.Tensor] = None,
        ctx_doc_ids: Optional[torch.Tensor] = None,
        injected_anchors: Optional[torch.Tensor] = None,
        injected_keep_mask: Optional[torch.Tensor] = None,
        _validated_anchors: Optional[_EDRValidatedAnchors] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample anchor positions per sample; returns (anchors, keep_mask).

        Random sampling returns the smallest 64/128/256/512 bucket that can
        hold the largest valid-anchor count in the microbatch, capped by
        ``self.num_anchors``. This keeps FlexAttention to a bounded shape set
        while avoiding draft-backbone work for trailing masked blocks. Samples
        with fewer valid positions use ``block_keep_mask=False`` for the
        remaining slots in the selected bucket. Injected anchors keep the
        caller-provided width.

        Args:
            seq_len: sequence length
            loss_mask: [B, seq_len] — 1 for supervised output positions
            device: torch device
            attention_mask: [B, seq_len] — 1 for non-padding tokens
            ctx_doc_ids: [B, seq_len] long — per-token document id (padding=-1)
                for sequence packing. An anchor is eligible when its immediate
                successor is supervised and belongs to the same document. Later
                proposal slots are masked individually at label construction.
            injected_anchors: [B, n_blocks] — if provided, bypass random
                sampling and use these anchors verbatim (for deterministic tests).
            injected_keep_mask: [B, n_blocks] bool — validity mask paired with
                ``injected_anchors``. If None while ``injected_anchors`` is given,
                all slots are treated as valid.

        Returns:
            anchors: [B, n_blocks] — sampled anchor positions (sorted)
            keep_mask: [B, n_blocks] — True for valid sampled anchors
        """
        bsz = loss_mask.shape[0]
        max_n = self.num_anchors

        if _validated_anchors is not None:
            plan = _validated_anchors
            if (
                injected_anchors is not plan.anchors or injected_keep_mask is not plan.keep_mask
                or plan.sequence_length != seq_len or plan.anchors.shape[0] != bsz
                or plan.anchors.device != loss_mask.device
                or plan.anchors.device.type != device.type
                or (device.index is not None and plan.anchors.device.index != device.index)
            ):
                raise ValueError("Validated EDR anchors do not match this backbone call")
            return plan.anchors, plan.keep_mask

        # Deterministic injection path: use given anchors as-is.
        if injected_anchors is not None:
            anchors = injected_anchors.to(device=device, dtype=torch.long)
            if anchors.ndim != 2 or anchors.shape[0] != bsz:
                raise ValueError("injected anchors must have shape [batch, blocks]")
            if injected_keep_mask is not None:
                keep_mask = injected_keep_mask.to(device=device, dtype=torch.bool)
            else:
                keep_mask = torch.ones_like(anchors, dtype=torch.bool)
            if keep_mask.shape != anchors.shape:
                raise ValueError("injected keep mask must match injected anchors")
            kept_anchors = anchors[keep_mask]
            if kept_anchors.numel() and bool(
                ((kept_anchors < 0) | (kept_anchors >= seq_len - 1)).any()
            ):
                raise ValueError("kept injected anchors must have an in-bounds successor token")
            if attention_mask is not None and kept_anchors.numel():
                row_ids = (
                    torch.arange(bsz, device=device).unsqueeze(1).expand_as(anchors)[keep_mask]
                )
                attention = attention_mask.to(device=device)
                if not bool(
                    (
                        (attention[row_ids, kept_anchors] > 0)
                        & (attention[row_ids, kept_anchors + 1] > 0)
                    ).all()
                ):
                    raise ValueError("kept injected anchors must precede non-padding tokens")
            if ctx_doc_ids is not None and kept_anchors.numel():
                row_ids = (
                    torch.arange(bsz, device=device).unsqueeze(1).expand_as(anchors)[keep_mask]
                )
                documents = ctx_doc_ids.to(device=device)
                anchor_documents = documents[row_ids, kept_anchors]
                successor_documents = documents[row_ids, kept_anchors + 1]
                if not bool(
                    ((anchor_documents >= 0) & (anchor_documents == successor_documents)).all()
                ):
                    raise ValueError(
                        "kept injected anchors and successors must share a packed document"
                    )
            anchors = torch.where(keep_mask, anchors, 0)
            return anchors, keep_mask

        candidate_count = seq_len - 1
        if candidate_count <= 0:
            logger.warning(
                f"Sequence too short for next-token anchor sampling (seq_len={seq_len}). "
                "Returning dummy anchors so loss is zero."
            )
            # Retain the smallest compiled shape even when the loss is empty.
            slot_count = self._edr_gradient_anchor_slots(1)
            anchors = torch.zeros(bsz, slot_count, dtype=torch.long, device=device)
            keep_mask = torch.zeros(bsz, slot_count, dtype=torch.bool, device=device)
            return anchors, keep_mask

        # The anchor is input-only. Its successor is the first learned label and
        # may be the first supervised response token after an unsupervised prompt.
        valid = loss_mask[:, 1:] > 0.5

        if attention_mask is not None:
            attention = attention_mask.to(device=device)
            valid = valid & (attention[:, :-1] > 0) & (attention[:, 1:] > 0)

        # Require the initialization anchor and first learned output to belong to
        # the same packed document. Per-slot validity handles later boundaries.
        if ctx_doc_ids is not None:
            doc = ctx_doc_ids.to(device=device)
            head_doc = doc[:, :-1]
            next_doc = doc[:, 1:]
            same_doc = (head_doc == next_doc) & (head_doc >= 0)
            valid = valid & same_doc

        valid_counts = valid.sum(dim=1)
        # Shape selection needs one scalar synchronization before the draft
        # forward. All rows in the microbatch share one bucket; per-row
        # validity is kept in keep_mask.
        max_valid_count = min(int(valid_counts.max().item()), max_n)
        slot_count = self._edr_gradient_anchor_slots(max(1, max_valid_count))

        indices = torch.arange(candidate_count, device=device).unsqueeze(0).expand(bsz, -1)
        masked_indices = torch.where(valid, indices, seq_len + 1)

        random_vals = torch.rand(bsz, candidate_count, device=device)
        random_vals = torch.where(valid, random_vals, 2.0)

        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)

        # Take only the selected bucket; pad with zeros when the sequence itself
        # has fewer candidate positions than that bucket.
        take_n = min(slot_count, gathered.shape[1])
        selected = gathered[:, :take_n].sort(dim=1).values
        if take_n < slot_count:
            pad = torch.zeros(bsz, slot_count - take_n, dtype=torch.long, device=device)
            selected = torch.cat([selected, pad], dim=1)
        anchors = selected

        keep_mask = torch.arange(slot_count, device=device).unsqueeze(
            0
        ) < valid_counts.unsqueeze(1).clamp(max=slot_count)
        anchors = torch.where(keep_mask, anchors, 0)

        return anchors, keep_mask

    def _create_position_ids(
        self,
        anchor_positions: torch.Tensor,
        seq_len: int,
        base_position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create position IDs for context and draft tokens.

        Args:
            anchor_positions: [B, n_blocks] anchor start indices.
            seq_len: context sequence length.
            base_position_ids: [B, seq_len] long — doc-local context positions
                (reset to 0 at each doc boundary) for sequence packing. When
                None (legacy path), context positions are the global
                ``arange(seq_len)`` and draft positions are ``anchor + offset``.
                When provided, context positions are taken verbatim and draft
                positions are ``base_position_ids[anchor] + offset`` so RoPE is
                doc-aware (each packed document starts at position 0).
        """
        bsz, n_blocks = anchor_positions.shape
        device = anchor_positions.device
        offsets = torch.arange(self.query_block_size, device=device).view(1, 1, -1)

        if base_position_ids is not None:
            base = base_position_ids.to(device=device, dtype=torch.long)
            context_position_ids = base
            # Doc-local base position at each anchor, then add within-block offset.
            anchor_base = torch.gather(base, 1, anchor_positions)  # [B, n_blocks]
            draft_position_ids = anchor_base.unsqueeze(-1) + offsets
            draft_position_ids = draft_position_ids.view(bsz, -1)
            return context_position_ids, draft_position_ids

        context_position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        draft_position_ids = anchor_positions.unsqueeze(-1) + offsets
        draft_position_ids = draft_position_ids.view(bsz, -1)

        return context_position_ids, draft_position_ids

    def _create_noise_embed(
        self,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Create noise embeddings: anchor token at block starts, MASK elsewhere."""
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        bs = self.query_block_size
        device = input_ids.device

        noise_ids = torch.full(
            (bsz, n * bs), self.draft_model.mask_token_id, dtype=torch.long, device=device
        )

        block_starts = torch.arange(n, device=device) * bs
        block_starts = block_starts.unsqueeze(0).expand(bsz, -1)

        valid_anchor_positions = anchor_positions.clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, valid_anchor_positions)

        flat_batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, n)
        noise_ids[flat_batch_idx, block_starts] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.tensor(self.draft_model.mask_token_id, dtype=torch.long, device=device),
        )

        return self.draft_model.embed_tokens(noise_ids)

    def _prepare_edr_context_cache(
        self,
        input_ids: torch.Tensor,
        hidden_states_list: List[torch.Tensor],
        base_position_ids: Optional[torch.Tensor],
    ):
        """Build one row's reusable context K/V when the drafter supports it."""
        if not bool(getattr(self.draft_model, "supports_context_cache", False)):
            return None

        prepare_context_cache = getattr(self.draft_model, "prepare_context_cache", None)
        cached_forward = getattr(self.draft_model, "forward_with_context_cache", None)
        if not callable(prepare_context_cache) or not callable(cached_forward):
            raise TypeError(
                "draft model advertises context-cache support without both cache methods"
            )

        context_feature = self.draft_model.extract_context_feature(hidden_states_list)
        if base_position_ids is None:
            context_position_ids = (
                torch.arange(
                    input_ids.shape[1],
                    device=input_ids.device,
                )
                .unsqueeze(0)
                .expand(input_ids.shape[0], -1)
            )
        else:
            context_position_ids = base_position_ids.to(
                device=input_ids.device,
                dtype=torch.long,
            )
        return prepare_context_cache(context_feature, context_position_ids)

    def _prepare_edr_statistics_context_cache(self, input_ids, hidden_states_list, position_ids):
        # With edr_reuse_context_cache, the cache keeps its autograd graph so the
        # gradient pass reuses the context FC/fusion/K/V projections; the no_grad
        # statistics pass only reads it. The cache is valid for one forward
        # because draft weights change at optimizer.step.
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.edr_reuse_context_cache):
            return self._prepare_edr_context_cache(input_ids, hidden_states_list, position_ids)

    def _draft_backbone(
        self,
        input_ids: torch.Tensor,
        hidden_states_list: List[torch.Tensor],
        loss_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        ctx_doc_ids: Optional[torch.Tensor] = None,
        base_position_ids: Optional[torch.Tensor] = None,
        injected_anchors: Optional[torch.Tensor] = None,
        injected_keep_mask: Optional[torch.Tensor] = None,
        draft_context_cache=None,
        _validated_anchors: Optional[_EDRValidatedAnchors] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Shared DFlash backbone (steps 1-6): context features → anchor
        sampling → noise embedding → position ids → block-causal mask → draft
        forward. ``DFlashModel.forward`` and DSpark/DFly subclasses build the
        draft hidden states this way; only the label/loss tail differs.

        Doc-aware (``ctx_doc_ids`` / ``base_position_ids``) and anchor-injection
        args are threaded through for packing and parity tests.

        Returns:
            draft_hidden: [B, n_blocks*block_size, D] pre-loss draft hidden states
            anchor_positions: [B, n_blocks] sampled anchor positions
            block_keep_mask: [B, n_blocks] bool validity of each anchor slot
            n_blocks: bucketed number of anchor slots (<= num_anchors)
        """
        seq_len = input_ids.shape[1]
        device = input_ids.device

        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len,
            loss_mask,
            device,
            attention_mask=attention_mask,
            ctx_doc_ids=ctx_doc_ids,
            injected_anchors=injected_anchors,
            injected_keep_mask=injected_keep_mask,
            _validated_anchors=_validated_anchors,
        )
        n_blocks = anchor_positions.shape[1]

        noise_embedding = self._create_noise_embed(input_ids, anchor_positions, block_keep_mask)

        context_position_ids, draft_position_ids = self._create_position_ids(
            anchor_positions, seq_len, base_position_ids=base_position_ids
        )

        block_mask = None
        if device.type == "cuda":
            block_mask = _build_dflash_block_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                ctx_len=seq_len,
                block_size=self.query_block_size,
                ctx_doc_ids=ctx_doc_ids,
                device=device,
            )

        if draft_context_cache is None:
            context_feature = self.draft_model.extract_context_feature(hidden_states_list)
            draft_hidden = self.draft_model(
                draft_input_ids=None,
                context_feature=context_feature,
                draft_position_ids=draft_position_ids,
                context_position_ids=context_position_ids,
                block_mask=block_mask,
                noise_embedding=noise_embedding,
            )
        else:
            cached_forward = getattr(self.draft_model, "forward_with_context_cache", None)
            if not callable(cached_forward):
                raise TypeError("draft context cache provided to a model without cache support")
            draft_hidden = cached_forward(
                draft_input_ids=None,
                context_cache=draft_context_cache,
                draft_position_ids=draft_position_ids,
                context_position_ids=context_position_ids,
                block_mask=block_mask,
                noise_embedding=noise_embedding,
            )

        return draft_hidden, anchor_positions, block_keep_mask, n_blocks

    @staticmethod
    def _compute_l1_loss(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """L1 distribution-distillation loss (DSpark ``l1_per_token``).

        Per-position L1 distance ``Σ_i |softmax(student)_i - softmax(teacher)_i|``
        between the full-vocab next-token distributions, which equals ``2·TV``.
        Returns [N].
        """
        tv, _ = lk_tv_kl_per_pos(student_logits, teacher_logits, form="tv")
        return 2.0 * tv

    def _compute_topk_kl_loss_variant_b(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        topk: int = 10,
    ) -> torch.Tensor:
        """Top-K KL divergence (Variant B) for DFlash distillation. Returns [N]."""
        return _kl_variant_b(student_logits, teacher_logits, topk)

    def _compute_lk_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        loss_type: str = "hybrid",
        eta: float = 3.0,
        teacher_row_indices: Optional[torch.Tensor] = None,
        teacher_log_normalizers: Optional[torch.Tensor] = None,
        student_log_normalizers: Optional[torch.Tensor] = None,
        vocab_chunk_size: int = 16384,
        distribution_aware: bool = False,
        temperature: float = 1.0,
        top_k: int = -1,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """LK (acceptance-rate) distillation loss for DFlash.

          * ``loss_type="alpha"``  →  -log( sum_i min(p_i, q_i) )
          * ``loss_type="hybrid"`` →  lambda*KL(p||q) + (1-lambda)*TV(p, q),
                lambda = exp(-eta * sg[alpha]), alpha = sum_i min(p_i, q_i).

        ``p`` = teacher (detached), ``q`` = student. Distribution-aware mode
        filters only p and tempers both; otherwise both are full-vocab at T=1.
        Returns [N] per-position LK loss.
        """
        if loss_type not in {"alpha", "hybrid"}:
            raise ValueError(
                f"Unknown lk_loss_type={loss_type!r}; expected 'alpha' or 'hybrid'."
            )
        if teacher_row_indices is None:
            teacher_row_indices = torch.arange(
                student_logits.shape[0],
                device=student_logits.device,
            )
        compute_kl = loss_type == "hybrid"
        tv, kl = streaming_tv_kl_per_pos(
            student_logits,
            teacher_logits,
            teacher_row_indices,
            vocab_chunk_size=vocab_chunk_size,
            teacher_log_normalizers=teacher_log_normalizers,
            student_log_normalizers=student_log_normalizers,
            compute_kl=compute_kl,
            distribution_aware=distribution_aware,
            temperature=temperature, top_k=top_k, top_p=top_p,
        )

        if loss_type == "alpha":
            # alpha = Σ_i min(p_i, q_i) == 1 − TV(p, q).
            alpha = (1.0 - tv).clamp_min(1e-10)  # [N]
            return -torch.log(alpha)

        if loss_type == "hybrid":
            lam = torch.exp(-eta * (1.0 - tv).detach().clamp(0.0, 1.0))
            return lam * kl + (1.0 - lam) * tv

        raise AssertionError("unreachable LK loss type")

    @staticmethod
    def _compute_e2e_tv_loss(
        student_logits_pb: torch.Tensor,
        teacher_logits: torch.Tensor,
        teacher_row_indices: torch.Tensor,
        valid_mask_pb: torch.Tensor,
        teacher_log_normalizers: Optional[torch.Tensor] = None,
        student_log_normalizers: Optional[torch.Tensor] = None,
        vocab_chunk_size: int = 16384,
        mean_by_row: bool = False,
        distribution_aware: bool = False,
        temperature: float = 1.0,
        top_k: int = -1,
        top_p: float = 1.0,
        projection_indices: Optional[torch.Tensor] = None,
    ):
        """End-to-end multi-step TV loss (γ-step accepted-length objective)::

            α_i     = 1 - TV(p_i, q_i) = Σ_v min(p_i,v, q_i,v)  ∈ (0, 1]
            L_e2e   = 1 - (1/γ) * Σ_{j=1..γ}  Π_{i=1..j} α_i

        γ = block_size. The prefix product couples steps inside a block, giving
        intrinsic per-step weighting, so this term ignores decay / flat_weights.
        Student logits are ``[B, n_blocks, block_size, V]``; teacher logits
        contain unique target positions and ``teacher_row_indices`` maps the
        flattened student layout to those rows. The validity mask is
        ``[B, n_blocks, block_size]``. With ``projection_indices``, student
        logits and the teacher map contain only selected flat positions, and
        only scalar TV values are scattered back before the block reduction.
        Distribution-aware mode filters only the teacher and tempers both
        distributions. Returns
        ``(e2e_tv_loss, accept_length)`` (the latter detached, for logging).
        """
        flat_student_logits = student_logits_pb.reshape(
            -1, student_logits_pb.shape[-1]
        )
        tv, _ = streaming_tv_kl_per_pos(
            flat_student_logits,
            teacher_logits,
            teacher_row_indices,
            vocab_chunk_size=vocab_chunk_size,
            teacher_log_normalizers=teacher_log_normalizers,
            student_log_normalizers=student_log_normalizers,
            compute_kl=False,
            distribution_aware=distribution_aware,
            temperature=temperature, top_k=top_k, top_p=top_p,
        )
        if projection_indices is not None:
            tv = tv.new_zeros(valid_mask_pb.numel()).index_copy(0, projection_indices, tv)
        alpha = (1.0 - tv).view_as(valid_mask_pb)

        # Set α:=1 on invalid slots so cumprod treats them as identity.
        m = valid_mask_pb.float()
        alpha_effective = alpha * m + (1.0 - m)
        prefix_prod = torch.cumprod(alpha_effective, dim=-1)  # [B, nb, γ]

        gamma_valid = m.sum(dim=-1).clamp(min=1.0)  # [B, nb]
        accept_length_pb = (prefix_prod * m).sum(dim=-1)  # [B, nb]
        e2e_per_block = 1.0 - accept_length_pb / gamma_valid

        block_has_valid = (m.sum(dim=-1) > 0).float()
        if mean_by_row:
            row_denominators = block_has_valid.sum(dim=-1).clamp(min=1.0)
            e2e_tv_loss = (
                (e2e_per_block * block_has_valid).sum(dim=-1) / row_denominators
            ).mean()
        else:
            denom = block_has_valid.sum().clamp(min=1.0)
            e2e_tv_loss = (e2e_per_block * block_has_valid).sum() / denom

        with torch.no_grad():
            if mean_by_row:
                accept_length = (
                    (accept_length_pb * block_has_valid).sum(dim=-1)
                    / row_denominators
                ).mean()
            else:
                accept_length = (accept_length_pb * block_has_valid).sum() / denom

        return e2e_tv_loss, accept_length.detach()

    @staticmethod
    def _distill_projection_indices(weight_mask: torch.Tensor) -> Optional[torch.Tensor]:
        """Return flat indices of supervised positions, or None for the dense path."""
        indices = weight_mask.reshape(-1).nonzero(as_tuple=True)[0]
        # A fully supervised layout needs no gather. A fully unsupervised batch
        # uses the dense path so the loss graph still yields zero gradients for
        # the projection parameters and the usual metrics.
        if indices.numel() in (0, weight_mask.numel()):
            return None
        return indices

    def _project_selected_distill_logits(
        self, draft_hidden, lm_head_weight, prev_token_ids, indices,
    ):
        hidden = draft_hidden.reshape(-1, draft_hidden.shape[-1]).index_select(0, indices)
        previous = prev_token_ids.reshape(-1).index_select(0, indices)
        # DFlash, DFly correction and DSpark Markov projections are pointwise.
        return self._compute_draft_logits(
            hidden.unsqueeze(0), lm_head_weight, previous.view(1, 1, -1), 1,
        )

    @torch.no_grad()
    def _complete_dpace_tail_confidences(
        self, confidence_nll, draft_hidden, lm_head_weight, prev_token_ids,
        target_ids, weight_mask,
    ):
        """Fill in detached confidences at masked tail positions for D-PACE.

        D-PACE weights sum prefix products over the whole block, so confidences
        at masked tail positions affect earlier valid weights. These positions
        get a detached projection only; blocks without a valid position are
        skipped.
        """
        valid = weight_mask != 0
        tails = (~valid) & valid.any(dim=-1, keepdim=True)
        indices = tails.reshape(-1).nonzero(as_tuple=True)[0]
        if not indices.numel():
            return confidence_nll
        logits = self._project_selected_distill_logits(
            draft_hidden, lm_head_weight, prev_token_ids, indices,
        )
        if self.fp32_lm_head:
            logits = logits.float()
        logits = logits.reshape(-1, logits.shape[-1])
        scale = 1.0 / self.distill_temperature
        logz, _ = streaming_student_log_normalizers_and_argmax(
            logits, self.edr_vocab_chunk_size, logit_scale=scale,
        )
        targets = target_ids.reshape(-1).index_select(0, indices)
        realized = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1).float()
        return confidence_nll.reshape(-1).index_copy(
            0, indices, logz - realized * scale,
        ).view_as(confidence_nll)

    # ------------------------------------------------------------------
    # Subclass extension hooks (no-ops for base DFlash). DSpark / TreeFlash
    # override these to inject hidden-state correction + Markov logit bias and
    # the confidence-head loss. Signatures are frozen here so subclasses attach
    # without reworking forward. See models/dspark.py.
    # ------------------------------------------------------------------
    def _compute_draft_logits(
        self,
        draft_hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        prev_token_ids: torch.Tensor,
        n_blocks: int,
    ) -> torch.Tensor:
        """Project slot-aligned hidden states to vocabulary logits.

        ``draft_hidden`` may contain any fixed projection width per block up to
        ``block_size``; ``prev_token_ids`` has the matching leading layout.
        DFlash uses the frozen LM head directly, while subclasses add pointwise
        correction or Markov bias.
        """
        return (
            self.draft_model.lm_head(draft_hidden)
            if hasattr(self.draft_model, "lm_head")
            else F.linear(draft_hidden, lm_head_weight)
        )

    def _compute_step_logits(
        self,
        draft_hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        previous_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Project one sequential proposal position during on-policy decoding.

        Base DFlash does not condition this projection on the previous token.
        DFly and DSpark override the hook for their causal correction heads.
        ``draft_hidden`` and ``previous_token_ids`` share arbitrary leading
        batch/block dimensions.
        """

        del previous_token_ids
        return (
            self.draft_model.lm_head(draft_hidden)
            if hasattr(self.draft_model, "lm_head")
            else F.linear(draft_hidden, lm_head_weight)
        )

    def _extra_distill_needed(self) -> bool:
        """Whether a subclass head needs the teacher logits even when KL/LK are
        off. DFlash: no. DSpark confidence head overrides to True."""
        return False

    @torch.no_grad()
    def _greedy_proposals_from_hidden(
        self,
        *,
        input_ids: torch.Tensor,
        draft_hidden: torch.Tensor,
        anchor_positions: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Greedily decode every learned proposal slot with causal subclass corrections."""

        bsz, n_blocks = anchor_positions.shape
        draft_hidden = self._select_learned_query_states(draft_hidden, n_blocks)
        hidden_by_block = draft_hidden.view(bsz, n_blocks, self.proposal_width, -1)
        proposals = torch.empty(
            (bsz, n_blocks, self.proposal_width),
            dtype=torch.long,
            device=draft_hidden.device,
        )
        previous_token_ids = torch.gather(
            input_ids,
            1,
            anchor_positions.clamp(min=0, max=input_ids.shape[1] - 1),
        )
        for offset in range(self.proposal_width):
            logits = self._compute_step_logits(
                hidden_by_block[:, :, offset, :],
                lm_head_weight,
                previous_token_ids,
            )
            proposal = logits.argmax(dim=-1)
            proposals[:, :, offset] = proposal
            previous_token_ids = proposal
        return proposals

    def _compute_extra_loss(
        self,
        loss: torch.Tensor,
        flat_logits: torch.Tensor,
        teacher_logits_flat: Optional[torch.Tensor],
        flat_weights: torch.Tensor,
        valid_token_count: torch.Tensor,
        prev_token_ids: torch.Tensor,
        n_blocks: int,
    ) -> Tuple[torch.Tensor, dict]:
        """Add subclass-specific loss terms on top of the DFlash objective.

        Returns ``(loss, extra_components)``, where ``extra_components`` holds
        detached scalars merged into ``loss_components`` for logging. DFlash adds
        nothing; DSpark adds e.g. ``{"confidence_loss": ...}``."""
        return loss, {}

    @torch.no_grad()
    def propose_blocks(
        self,
        input_ids: torch.Tensor,
        hidden_states_list: List[torch.Tensor],
        loss_mask: torch.Tensor,
        lm_head_weight: torch.Tensor,
        ctx_doc_ids: Optional[torch.Tensor] = None,
        base_position_ids: Optional[torch.Tensor] = None,
        injected_anchors: Optional[torch.Tensor] = None,
        injected_keep_mask: Optional[torch.Tensor] = None,
    ):
        """On-policy DFlash proposal for OPD packed tree-forward scoring (no grad).

        Runs the block-parallel draft and returns the argmax'd proposals. With
        ``query_includes_input_anchor``, slot 0 participates in block attention
        but is omitted from the returned proposal IDs.

        Returns:
            proposals: [B, n_blocks, proposal_width] long — argmax per learned slot.
            anchor_positions: [B, n_blocks] long.
            block_keep_mask: [B, n_blocks] bool — valid anchors.
        """
        seq_len = input_ids.shape[1]
        device = input_ids.device

        context_feature = self.draft_model.extract_context_feature(hidden_states_list)
        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len,
            loss_mask,
            device,
            ctx_doc_ids=ctx_doc_ids,
            injected_anchors=injected_anchors,
            injected_keep_mask=injected_keep_mask,
        )
        noise_embedding = self._create_noise_embed(input_ids, anchor_positions, block_keep_mask)
        context_position_ids, draft_position_ids = self._create_position_ids(
            anchor_positions, seq_len, base_position_ids=base_position_ids
        )
        block_mask = None
        if device.type == "cuda":
            block_mask = _build_dflash_block_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                ctx_len=seq_len,
                block_size=self.query_block_size,
                ctx_doc_ids=ctx_doc_ids,
                device=device,
            )
        draft_hidden = self.draft_model(
            draft_input_ids=None,
            context_feature=context_feature,
            draft_position_ids=draft_position_ids,
            context_position_ids=context_position_ids,
            block_mask=block_mask,
            noise_embedding=noise_embedding,
        )
        proposals = self._greedy_proposals_from_hidden(
            input_ids=input_ids,
            draft_hidden=draft_hidden,
            anchor_positions=anchor_positions,
            lm_head_weight=lm_head_weight,
        )
        return proposals, anchor_positions, block_keep_mask

    def _edr_chunk_statistics(
        self,
        *,
        input_ids: torch.Tensor,
        hidden_states_list: List[torch.Tensor],
        loss_mask: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_distribution: EDRTargetDistribution | EDRSparseTargetDistribution,
        target_probability_start: int | torch.Tensor,
        anchors: torch.Tensor,
        prefixes: torch.Tensor,
        horizon: Optional[EDRHorizon] = None,
        horizon_lengths: Optional[torch.Tensor] = None,
        target_distribution_offsets: Optional[torch.Tensor] = None,
        target_probability_counts: Optional[torch.Tensor] = None,
        block_keep_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        ctx_doc_ids: Optional[torch.Tensor],
        base_position_ids: Optional[torch.Tensor],
        draft_context_cache=None,
        draft_temperature: Optional[float] = None,
        cache_rejection_mask: bool = False,
        _validated_anchors: Optional[_EDRValidatedAnchors] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate one chunk of round starts, potentially from many horizons.

        Every anchor block remains attention-isolated. Row-level batching only
        concatenates blocks and target rows; the per-block metadata below maps
        each proposal back to its own horizon and target distribution.
        """
        if anchors.ndim not in (1, 2) or prefixes.shape != anchors.shape:
            raise ValueError("EDR anchors and prefixes must be equal rank-1 or rank-2 tensors")
        if draft_temperature is None:
            draft_temperature = self.edr_temperature
        validate_sampling_parameters(draft_temperature, -1, 1.0)
        if torch.is_grad_enabled() and draft_temperature == 0:
            validate_sampling_parameters(draft_temperature, -1, 1.0, allow_greedy=False)

        squeeze_batch = anchors.ndim == 1
        if squeeze_batch:
            anchors = anchors.unsqueeze(0)
            prefixes = prefixes.unsqueeze(0)
        batch_size, n_blocks = anchors.shape
        if input_ids.shape[0] != batch_size:
            raise ValueError(
                "EDR anchor batch size must match input_ids "
                f"({batch_size} != {input_ids.shape[0]})"
            )
        injected_anchors = anchors
        if block_keep_mask is None:
            block_keep_mask = torch.ones_like(anchors, dtype=torch.bool)
        else:
            if squeeze_batch and block_keep_mask.ndim == 1:
                block_keep_mask = block_keep_mask.unsqueeze(0)
            if block_keep_mask.shape != anchors.shape:
                raise ValueError("EDR block keep mask must match anchors")
        injected_keep_mask = block_keep_mask.to(
            device=anchors.device,
            dtype=torch.bool,
        )
        draft_hidden, _, _, _ = self._draft_backbone(
            input_ids,
            hidden_states_list,
            loss_mask,
            attention_mask=attention_mask,
            ctx_doc_ids=ctx_doc_ids,
            base_position_ids=base_position_ids,
            injected_anchors=injected_anchors,
            injected_keep_mask=injected_keep_mask,
            draft_context_cache=draft_context_cache,
            _validated_anchors=_validated_anchors,
        )

        proposal_width = self.edr_proposal_width
        proposal_layout = build_dflash_proposal_layout(
            injected_anchors,
            sequence_length=input_ids.shape[1],
            block_size=proposal_width,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            ctx_doc_ids=ctx_doc_ids,
        )
        target_ids = proposal_layout.gather_labels(input_ids)
        prev_token_ids = proposal_layout.gather_predecessor_tokens(input_ids)

        target_row_count = target_distribution.num_positions
        if target_row_count == 0:
            raise ValueError("EDR target distribution cannot be empty for a nonempty chunk")

        def block_vector(value, name: str, default: int) -> torch.Tensor:
            if value is None:
                value = default
            tensor = torch.as_tensor(
                value,
                device=input_ids.device,
                dtype=torch.long,
            )
            if tensor.ndim == 0:
                return tensor.expand(batch_size, n_blocks)
            if tensor.shape == (n_blocks,):
                return tensor.unsqueeze(0).expand(batch_size, -1)
            if tensor.shape != (batch_size, n_blocks):
                raise ValueError(
                    f"{name} must be scalar or have shape {(n_blocks,)} or "
                    f"{(batch_size, n_blocks)}"
                )
            return tensor

        probability_starts = block_vector(
            target_probability_start,
            "EDR target probability starts",
            0,
        )
        probability_offsets = block_vector(
            target_distribution_offsets,
            "EDR target distribution offsets",
            0,
        )
        probability_counts = block_vector(
            target_probability_counts,
            "EDR target probability counts",
            target_row_count,
        )
        if horizon_lengths is None:
            if horizon is None:
                raise ValueError("EDR horizon lengths are required for a batched chunk")
            horizon_lengths = torch.full(
                (batch_size, n_blocks),
                horizon.ordinary_length,
                device=input_ids.device,
                dtype=torch.long,
            )
        else:
            horizon_lengths = block_vector(
                horizon_lengths,
                "EDR horizon lengths",
                0,
            )

        relative_probability_indices = (
            proposal_layout.predecessor_indices - probability_starts.unsqueeze(-1)
        )
        target_probability_valid = (relative_probability_indices >= 0) & (
            relative_probability_indices < probability_counts.unsqueeze(-1)
        )
        safe_relative_probability_indices = torch.minimum(
            relative_probability_indices.clamp_min(0),
            (probability_counts - 1).clamp_min(0).unsqueeze(-1),
        )
        safe_target_probability_indices = (
            probability_offsets.unsqueeze(-1) + safe_relative_probability_indices
        ).clamp(
            min=0,
            max=target_row_count - 1,
        )

        proposal_offsets = torch.arange(
            1,
            1 + proposal_width,
            device=input_ids.device,
        )
        horizon_valid = prefixes.unsqueeze(-1) + proposal_offsets.unsqueeze(
            0
        ) <= horizon_lengths.unsqueeze(-1)
        valid_mask = (
            horizon_valid
            & proposal_layout.valid_mask
            & target_probability_valid
            & injected_keep_mask[:, :, None]
        )
        # Block attention ran over every query slot, including the input-anchor
        # slot. Only the pointwise projection is compacted, so padding and slots
        # beyond a horizon allocate no [N, vocab] logits.
        draft_hidden = self._select_learned_query_states(draft_hidden, n_blocks)
        costs, acceptance = self._edr_project_valid_statistics(
            draft_hidden=draft_hidden,
            lm_head_weight=lm_head_weight,
            prev_token_ids=prev_token_ids,
            target_distribution=target_distribution,
            target_probability_indices=safe_target_probability_indices,
            target_ids=target_ids,
            valid_mask=valid_mask,
            draft_temperature=draft_temperature,
            cache_rejection_mask=cache_rejection_mask,
            projection_indices=(
                _validated_anchors.projection_indices if _validated_anchors is not None else None
            ),
        )
        if squeeze_batch:
            return costs[0], acceptance[0], valid_mask[0]
        return costs, acceptance, valid_mask

    def _edr_project_valid_statistics(
        self, *, draft_hidden, lm_head_weight, prev_token_ids,
        target_distribution, target_probability_indices, target_ids, valid_mask,
        draft_temperature, cache_rejection_mask,
        projection_indices=None,
    ):
        """Project only useful learned slots; scatter scalar statistics back.

        All projection hooks are pointwise (including DFly correction and the
        DSpark Markov head), so a single compact row preserves their gradients.
        Attention has already run over the full isolated block layout.
        """
        indices = (
            valid_mask.reshape(-1).nonzero(as_tuple=True)[0]
            if projection_indices is None else projection_indices
        )
        if not indices.numel():
            # Zero statistics that stay connected to the backbone, without
            # projecting any vocabulary rows.
            zero = draft_hidden.reshape(-1)[:0].float().sum()
            empty = zero.expand(valid_mask.shape)
            return empty, empty
        hidden = draft_hidden.reshape(-1, draft_hidden.shape[-1]).index_select(0, indices)
        previous = prev_token_ids.reshape(-1).index_select(0, indices)
        logits = self._compute_draft_logits(
            hidden.unsqueeze(0), lm_head_weight, previous.view(1, 1, -1), 1,
        ).squeeze(0)
        rows = target_probability_indices.reshape(-1).index_select(0, indices)
        tokens = target_ids.reshape(-1).index_select(0, indices)
        if float(draft_temperature) == 0.0:
            costs, acceptance = greedy_edr_distribution_statistics(
                logits, target_distribution, rows, tokens,
            )
        else:
            costs, acceptance = streaming_edr_distribution_statistics(
                logits, target_distribution, rows, tokens, self.edr_vocab_chunk_size,
                cache_rejection_mask=cache_rejection_mask,
                draft_temperature=draft_temperature,
            )
        def restore(values):
            return values.new_zeros(valid_mask.numel()).index_copy(
                0, indices, values,
            ).view_as(valid_mask)

        return restore(costs), restore(acceptance)

    @torch.no_grad()
    def _prepare_edr_target_distribution(self, target_hidden, lm_head_weight):
        sampling = dict(
            stopping_token_ids=self.edr_stop_token_ids,
            temperature=self.edr_temperature, top_k=self.edr_top_k, top_p=self.edr_top_p,
        )
        if (
            self.edr_sparse_target and self.loss_objective == "edr"
            and self.edr_temperature > 0 and 1 <= self.edr_top_k <= 128
            and self.edr_top_k < lm_head_weight.shape[0]
        ):
            from angelspec.models.ops.edr_sparse_target import (
                project_edr_sparse_target_distribution,
            )

            return project_edr_sparse_target_distribution(
                target_hidden, lm_head_weight, self.edr_vocab_chunk_size, **sampling,
            )
        return prepare_edr_target_distribution(
            F.linear(target_hidden, lm_head_weight), self.edr_vocab_chunk_size, **sampling,
        )

    @torch.no_grad()
    def _edr_all_horizon_statistics(
        self,
        *,
        input_ids: torch.Tensor,
        hidden_states_list: List[torch.Tensor],
        loss_mask: torch.Tensor,
        lm_head_weight: torch.Tensor,
        normalized_target_hidden: torch.Tensor,
        horizons: List[EDRHorizon],
        attention_mask: Optional[torch.Tensor],
        ctx_doc_ids: Optional[torch.Tensor],
        base_position_ids: Optional[torch.Tensor],
        draft_context_cache=None,
    ) -> Tuple[List[_EDRHorizonStatistics], Optional[EDRTargetDistribution | EDRSparseTargetDistribution]]:
        """Batch every detached round start for one row into bounded chunks.

        Target rows and anchor blocks are concatenated across horizons. Compact
        statistics are split back into independent horizons before any Bellman
        recurrence is evaluated, preserving the exact objective.
        """
        entries: list[_EDRHorizonBatchEntry] = []
        target_hidden_chunks: list[torch.Tensor] = []
        target_offset = 0

        for horizon in horizons:
            target_count = horizon.ordinary_length
            target_start = horizon.start - 1
            entries.append(
                _EDRHorizonBatchEntry(
                    horizon=horizon,
                    target_probability_start=target_start,
                    target_distribution_offset=target_offset,
                    target_probability_count=target_count,
                )
            )
            if target_count:
                target_hidden_chunks.append(
                    normalized_target_hidden[
                        0,
                        target_start : target_start + target_count,
                    ]
                )
                target_offset += target_count

        target_distribution = None
        flat_costs = torch.zeros(
            (0, self.edr_proposal_width),
            device=input_ids.device,
            dtype=torch.float32,
        )
        flat_acceptance = torch.zeros_like(flat_costs)

        if target_hidden_chunks:
            target_hidden = torch.cat(target_hidden_chunks, dim=0)
            target_distribution = self._prepare_edr_target_distribution(target_hidden, lm_head_weight)

            prefix_chunks: list[torch.Tensor] = []
            anchor_chunks: list[torch.Tensor] = []
            horizon_length_chunks: list[torch.Tensor] = []
            probability_start_chunks: list[torch.Tensor] = []
            probability_offset_chunks: list[torch.Tensor] = []
            probability_count_chunks: list[torch.Tensor] = []
            for entry in entries:
                count = entry.target_probability_count
                if not count:
                    continue
                prefixes = torch.arange(
                    count,
                    device=input_ids.device,
                    dtype=torch.long,
                )
                anchors = prefixes + entry.horizon.start - 1
                prefix_chunks.append(prefixes)
                anchor_chunks.append(anchors)
                horizon_length_chunks.append(
                    torch.full_like(prefixes, entry.horizon.ordinary_length)
                )
                probability_start_chunks.append(
                    torch.full_like(prefixes, entry.target_probability_start)
                )
                probability_offset_chunks.append(
                    torch.full_like(prefixes, entry.target_distribution_offset)
                )
                probability_count_chunks.append(torch.full_like(prefixes, count))

            all_prefixes = torch.cat(prefix_chunks)
            all_anchors = torch.cat(anchor_chunks)
            all_horizon_lengths = torch.cat(horizon_length_chunks)
            all_probability_starts = torch.cat(probability_start_chunks)
            all_probability_offsets = torch.cat(probability_offset_chunks)
            all_probability_counts = torch.cat(probability_count_chunks)
            cost_chunks: list[torch.Tensor] = []
            acceptance_chunks: list[torch.Tensor] = []

            for chunk_start in range(0, all_prefixes.numel(), self.edr_chunk_size):
                chunk_slice = slice(chunk_start, chunk_start + self.edr_chunk_size)
                chunk_costs, chunk_acceptance, chunk_valid = self._edr_chunk_statistics(
                    input_ids=input_ids,
                    hidden_states_list=hidden_states_list,
                    loss_mask=loss_mask,
                    lm_head_weight=lm_head_weight,
                    target_distribution=target_distribution,
                    target_probability_start=all_probability_starts[chunk_slice],
                    target_distribution_offsets=all_probability_offsets[chunk_slice],
                    target_probability_counts=all_probability_counts[chunk_slice],
                    horizon_lengths=all_horizon_lengths[chunk_slice],
                    anchors=all_anchors[chunk_slice],
                    prefixes=all_prefixes[chunk_slice],
                    attention_mask=attention_mask,
                    ctx_doc_ids=ctx_doc_ids,
                    base_position_ids=base_position_ids,
                    draft_context_cache=draft_context_cache,
                )
                cost_chunks.append(
                    torch.where(chunk_valid, chunk_costs, torch.zeros_like(chunk_costs))
                )
                acceptance_chunks.append(
                    torch.where(
                        chunk_valid,
                        chunk_acceptance,
                        torch.zeros_like(chunk_acceptance),
                    )
                )

            flat_costs = torch.cat(cost_chunks, dim=0)
            flat_acceptance = torch.cat(acceptance_chunks, dim=0)

        statistics: list[_EDRHorizonStatistics] = []
        cursor = 0
        for entry in entries:
            target_count = entry.target_probability_count
            costs = flat_costs[cursor : cursor + target_count]
            acceptance = flat_acceptance[cursor : cursor + target_count]
            cursor += target_count
            padding_count = entry.horizon.ordinary_length - target_count
            if padding_count:
                padding = costs.new_zeros((padding_count, self.edr_proposal_width))
                costs = torch.cat((costs, padding), dim=0)
                acceptance = torch.cat((acceptance, padding), dim=0)
            statistics.append(
                _EDRHorizonStatistics(
                    entry=entry,
                    costs=costs,
                    acceptance=acceptance,
                )
            )

        return statistics, target_distribution

    def _edr_gradient_horizon_surrogate(
        self,
        *,
        input_ids: torch.Tensor,
        hidden_states_list: List[torch.Tensor],
        loss_mask: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_distribution: EDRTargetDistribution | EDRSparseTargetDistribution,
        queries: List[_EDRGradientQuery],
        attention_mask: Optional[torch.Tensor],
        ctx_doc_ids: Optional[torch.Tensor],
        base_position_ids: Optional[torch.Tensor],
        draft_context_cache=None,
    ) -> torch.Tensor:
        """Evaluate and split one row's EDR gradient horizons in bounded batches."""
        if not queries:
            raise ValueError("EDR gradient horizon batch cannot be empty")

        anchor_chunks: list[torch.Tensor] = []
        prefix_chunks: list[torch.Tensor] = []
        keep_chunks: list[torch.Tensor] = []
        horizon_length_chunks: list[torch.Tensor] = []
        probability_start_chunks: list[torch.Tensor] = []
        probability_offset_chunks: list[torch.Tensor] = []
        probability_count_chunks: list[torch.Tensor] = []

        for query in queries:
            entry = query.entry
            anchors = query.prefixes + entry.horizon.start - 1
            anchors = torch.where(
                query.keep_mask,
                anchors,
                torch.zeros_like(anchors),
            )
            anchor_chunks.append(anchors)
            prefix_chunks.append(query.prefixes)
            keep_chunks.append(query.keep_mask)
            horizon_length_chunks.append(
                torch.full_like(query.prefixes, entry.horizon.ordinary_length)
            )
            probability_start_chunks.append(
                torch.full_like(query.prefixes, entry.target_probability_start)
            )
            probability_offset_chunks.append(
                torch.full_like(query.prefixes, entry.target_distribution_offset)
            )
            probability_count_chunks.append(
                torch.full_like(query.prefixes, entry.target_probability_count)
            )

        all_anchors = torch.cat(anchor_chunks)
        all_prefixes = torch.cat(prefix_chunks)
        all_keep = torch.cat(keep_chunks)
        all_horizon_lengths = torch.cat(horizon_length_chunks)
        all_probability_starts = torch.cat(probability_start_chunks)
        all_probability_offsets = torch.cat(probability_offset_chunks)
        all_probability_counts = torch.cat(probability_count_chunks)

        # Full-anchor mode backpropagates every round start in edr_chunk_size
        # batches without padding (num_anchors is unused). Sampled mode pads to
        # the bounded bucket shapes.
        batch_capacity = (
            self.edr_chunk_size
            if self.edr_full_anchor_backprop
            else max(self.edr_chunk_size, self.num_anchors)
        )
        selected_total = all_prefixes.numel()
        rejection_cache_remaining = self.edr_rejection_cache_max_bytes
        cost_chunks: list[torch.Tensor] = []
        acceptance_chunks: list[torch.Tensor] = []
        valid_chunks: list[torch.Tensor] = []

        def pad_tail(
            values: torch.Tensor,
            query_slots: int,
            fill_value=0,
        ) -> torch.Tensor:
            padding = query_slots - values.numel()
            if padding == 0:
                return values
            return torch.cat(
                (values, values.new_full((padding,), fill_value)),
                dim=0,
            )

        for chunk_start in range(0, selected_total, batch_capacity):
            chunk_end = min(chunk_start + batch_capacity, selected_total)
            chunk_slice = slice(chunk_start, chunk_end)
            real_count = chunk_end - chunk_start
            if chunk_end < selected_total:
                query_slots = real_count
            elif self.edr_full_anchor_backprop:
                query_slots = real_count
            elif len(queries) == 1:
                query_slots = self._edr_gradient_anchor_slots(real_count)
            else:
                query_slots = self._edr_combined_gradient_anchor_slots(
                    real_count,
                    batch_capacity,
                )

            # All gradient chunks of this call share one rejection-cache byte
            # budget, since every cached mask stays alive until backward.
            cache_bytes = (
                query_slots * self.edr_proposal_width * target_distribution.vocab_size
            )
            cache_mask = (
                isinstance(target_distribution, EDRTargetDistribution)
                and cache_bytes <= rejection_cache_remaining
            )
            if cache_mask:
                rejection_cache_remaining -= cache_bytes
            chunk_costs, chunk_acceptance, chunk_valid = self._edr_chunk_statistics(
                input_ids=input_ids,
                hidden_states_list=hidden_states_list,
                loss_mask=loss_mask,
                lm_head_weight=lm_head_weight,
                target_distribution=target_distribution,
                target_probability_start=pad_tail(
                    all_probability_starts[chunk_slice], query_slots
                ),
                target_distribution_offsets=pad_tail(
                    all_probability_offsets[chunk_slice], query_slots
                ),
                target_probability_counts=pad_tail(
                    all_probability_counts[chunk_slice], query_slots
                ),
                horizon_lengths=pad_tail(all_horizon_lengths[chunk_slice], query_slots),
                anchors=pad_tail(all_anchors[chunk_slice], query_slots),
                prefixes=pad_tail(all_prefixes[chunk_slice], query_slots),
                block_keep_mask=pad_tail(all_keep[chunk_slice], query_slots, False),
                attention_mask=attention_mask,
                ctx_doc_ids=ctx_doc_ids,
                base_position_ids=base_position_ids,
                draft_context_cache=draft_context_cache,
                cache_rejection_mask=cache_mask,
            )
            cost_chunks.append(chunk_costs)
            acceptance_chunks.append(chunk_acceptance)
            valid_chunks.append(chunk_valid)

        # Padding exists only at the end of the final output chunk.
        all_costs = torch.cat(cost_chunks, dim=0)[:selected_total]
        all_acceptance = torch.cat(acceptance_chunks, dim=0)[:selected_total]
        all_valid = torch.cat(valid_chunks, dim=0)[:selected_total]
        surrogate = all_costs.new_zeros(())
        cursor = 0
        for query in queries:
            query_size = query.prefixes.numel()
            selected_costs = all_costs[cursor : cursor + query_size]
            selected_acceptance = all_acceptance[cursor : cursor + query_size]
            selected_valid = all_valid[cursor : cursor + query_size]
            cursor += query_size
            selected_costs = torch.where(
                selected_valid,
                selected_costs,
                torch.zeros_like(selected_costs),
            )
            selected_acceptance = torch.where(
                selected_valid,
                selected_acceptance,
                torch.zeros_like(selected_acceptance),
            )
            if self.edr_full_anchor_backprop:
                surrogate = surrogate + edr_surrogate_sum(
                    selected_costs,
                    selected_acceptance,
                    query.dynamic_program,
                )
            elif query.coefficients is not None:
                surrogate = surrogate + edr_surrogate_sum_from_coefficients(
                    selected_costs, selected_acceptance, query.coefficients,
                )
            else:
                if query.inclusion_probabilities is None:
                    raise RuntimeError(
                        "sampled EDR queries require inclusion probabilities"
                    )
                surrogate = surrogate + sampled_edr_surrogate_sum(
                    selected_costs,
                    selected_acceptance,
                    query.dynamic_program,
                    query.prefixes,
                    query.inclusion_probabilities,
                    query.keep_mask,
                    query.inverse_pps_scale,
                )

        return surrogate

    @torch.no_grad()
    def _edr_all_row_statistics(
        self,
        *,
        input_ids: torch.Tensor,
        hidden_states_list: List[torch.Tensor],
        loss_mask: torch.Tensor,
        lm_head_weight: torch.Tensor,
        normalized_target_hidden: torch.Tensor,
        horizons_by_row: Sequence[Sequence[EDRHorizon]],
        attention_mask: Optional[torch.Tensor],
        ctx_doc_ids: Optional[torch.Tensor],
        base_position_ids: Optional[torch.Tensor],
        draft_context_cache=None,
        draft_temperature: Optional[float] = None,
        _validated_horizons: bool = False,
    ) -> Tuple[List[List[_EDRHorizonStatistics]], Optional[EDRTargetDistribution | EDRSparseTargetDistribution]]:
        """Evaluate detached EDR starts from several rows in bounded batched calls.

        Each batch row owns its context and proposal blocks. The shared target
        distribution is only a concatenated lookup table; per-block offsets keep
        every probability lookup attached to its source row and horizon.
        ``edr_chunk_size`` remains a total block budget, so increasing the number
        of rows in flight does not multiply the largest vocabulary projection.
        """
        batch_size = input_ids.shape[0]
        if len(horizons_by_row) != batch_size:
            raise ValueError("EDR horizon rows must match the input batch size")
        # Validated horizons were checked on CPU once per group, so metadata and
        # packing are planned on CPU without per-chunk device reads. Other
        # callers plan on the input device.
        plan_device = torch.device("cpu") if _validated_horizons else input_ids.device

        entries_by_row: list[list[_EDRHorizonBatchEntry]] = []
        target_hidden_chunks: list[torch.Tensor] = []
        target_offset = 0

        for row_index, row_horizons in enumerate(horizons_by_row):
            row_entries: list[_EDRHorizonBatchEntry] = []
            for horizon in row_horizons:
                if horizon.batch_index != row_index:
                    raise ValueError("EDR horizon is assigned to the wrong input row")
                target_count = horizon.ordinary_length
                target_start = horizon.start - 1
                entry = _EDRHorizonBatchEntry(
                    horizon=horizon,
                    target_probability_start=target_start,
                    target_distribution_offset=target_offset,
                    target_probability_count=target_count,
                )
                row_entries.append(entry)
                if target_count:
                    target_slice = slice(target_start, target_start + target_count)
                    target_hidden_chunks.append(
                        normalized_target_hidden[row_index, target_slice]
                    )
                    target_offset += target_count
            entries_by_row.append(row_entries)

        if not target_hidden_chunks:
            return ([[] for _ in range(batch_size)], None)

        target_distribution = self._prepare_edr_target_distribution(
            torch.cat(target_hidden_chunks, dim=0), lm_head_weight,
        )

        row_vectors: list[tuple[torch.Tensor, ...]] = []
        for row_entries in entries_by_row:
            prefixes: list[torch.Tensor] = []
            anchors: list[torch.Tensor] = []
            horizon_lengths: list[torch.Tensor] = []
            probability_starts: list[torch.Tensor] = []
            probability_offsets: list[torch.Tensor] = []
            probability_counts: list[torch.Tensor] = []
            for entry in row_entries:
                count = entry.target_probability_count
                if not count:
                    continue
                prefix = torch.arange(count, device=plan_device, dtype=torch.long)
                prefixes.append(prefix)
                anchors.append(prefix + entry.horizon.start - 1)
                horizon_lengths.append(
                    torch.full_like(prefix, entry.horizon.ordinary_length)
                )
                probability_starts.append(
                    torch.full_like(prefix, entry.target_probability_start)
                )
                probability_offsets.append(
                    torch.full_like(prefix, entry.target_distribution_offset)
                )
                probability_counts.append(torch.full_like(prefix, count))

            empty = torch.empty(0, device=plan_device, dtype=torch.long)
            row_vectors.append(
                tuple(
                    torch.cat(chunks) if chunks else empty
                    for chunks in (
                        prefixes,
                        anchors,
                        horizon_lengths,
                        probability_starts,
                        probability_offsets,
                        probability_counts,
                    )
                )
            )

        # Divide the total block budget across rows so one batched call holds at
        # most edr_chunk_size blocks. Rows shorter than the longest are padded.
        blocks_per_row = max(1, self.edr_chunk_size // batch_size)
        max_blocks = max(vectors[0].numel() for vectors in row_vectors)
        cost_chunks_by_row: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
        acceptance_chunks_by_row: list[list[torch.Tensor]] = [
            [] for _ in range(batch_size)
        ]

        for chunk_start in range(0, max_blocks, blocks_per_row):
            query_slots = min(blocks_per_row, max_blocks - chunk_start)
            block_values = [
                torch.zeros(
                    (batch_size, query_slots),
                    device=plan_device,
                    dtype=torch.long,
                )
                for _ in range(6)
            ]
            block_keep = torch.zeros(
                (batch_size, query_slots),
                device=plan_device,
                dtype=torch.bool,
            )
            real_counts: list[int] = []
            for row_index, vectors in enumerate(row_vectors):
                real_count = min(query_slots, max(0, vectors[0].numel() - chunk_start))
                real_counts.append(real_count)
                if not real_count:
                    continue
                source = slice(chunk_start, chunk_start + real_count)
                for destination, values in zip(block_values, vectors, strict=True):
                    destination[row_index, :real_count] = values[source]
                block_keep[row_index, :real_count] = True

            validated_anchors = None
            if _validated_horizons:
                validated_anchors = _EDRValidatedAnchors.from_cpu(
                    block_values[1], block_values[0], block_values[2], block_keep,
                    width=self.edr_proposal_width, sequence_length=input_ids.shape[1],
                    device=input_ids.device,
                )
                block_values = [
                    validated_anchors.anchors if index == 1 else value.to(input_ids.device, non_blocking=True)
                    for index, value in enumerate(block_values)
                ]
                block_keep = validated_anchors.keep_mask
            chunk_costs, chunk_acceptance, chunk_valid = self._edr_chunk_statistics(
                input_ids=input_ids,
                hidden_states_list=hidden_states_list,
                loss_mask=loss_mask,
                lm_head_weight=lm_head_weight,
                target_distribution=target_distribution,
                prefixes=block_values[0],
                anchors=block_values[1],
                horizon_lengths=block_values[2],
                target_probability_start=block_values[3],
                target_distribution_offsets=block_values[4],
                target_probability_counts=block_values[5],
                block_keep_mask=block_keep,
                attention_mask=attention_mask,
                ctx_doc_ids=ctx_doc_ids,
                base_position_ids=base_position_ids,
                draft_context_cache=draft_context_cache,
                draft_temperature=draft_temperature,
                _validated_anchors=validated_anchors,
            )
            for row_index, real_count in enumerate(real_counts):
                if not real_count:
                    continue
                valid = chunk_valid[row_index, :real_count]
                costs = chunk_costs[row_index, :real_count]
                acceptance = chunk_acceptance[row_index, :real_count]
                cost_chunks_by_row[row_index].append(
                    torch.where(valid, costs, torch.zeros_like(costs))
                )
                acceptance_chunks_by_row[row_index].append(
                    torch.where(valid, acceptance, torch.zeros_like(acceptance))
                )

        statistics_by_row: list[list[_EDRHorizonStatistics]] = []
        for row_index, row_entries in enumerate(entries_by_row):
            empty_statistics = input_ids.new_zeros(
                (0, self.edr_proposal_width), dtype=torch.float32
            )
            flat_costs = (
                torch.cat(cost_chunks_by_row[row_index], dim=0)
                if cost_chunks_by_row[row_index]
                else empty_statistics
            )
            flat_acceptance = (
                torch.cat(acceptance_chunks_by_row[row_index], dim=0)
                if acceptance_chunks_by_row[row_index]
                else empty_statistics
            )
            row_statistics: list[_EDRHorizonStatistics] = []
            cursor = 0
            for entry in row_entries:
                count = entry.target_probability_count
                row_statistics.append(
                    _EDRHorizonStatistics(
                        entry=entry,
                        costs=flat_costs[cursor : cursor + count],
                        acceptance=flat_acceptance[cursor : cursor + count],
                    )
                )
                cursor += count
            statistics_by_row.append(row_statistics)

        return statistics_by_row, target_distribution

    def _edr_gradient_row_surrogate(
        self,
        *,
        input_ids: torch.Tensor,
        hidden_states_list: List[torch.Tensor],
        loss_mask: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_distribution: EDRTargetDistribution | EDRSparseTargetDistribution,
        queries_by_row: Sequence[Sequence[_EDRGradientQuery]],
        attention_mask: Optional[torch.Tensor],
        ctx_doc_ids: Optional[torch.Tensor],
        base_position_ids: Optional[torch.Tensor],
        draft_context_cache=None,
        _validated_horizons: bool = False,
    ) -> torch.Tensor:
        """Evaluate selected starts from several rows in bounded batched calls."""
        batch_size = input_ids.shape[0]
        if len(queries_by_row) != batch_size:
            raise ValueError("EDR gradient query rows must match the input batch size")
        cpu_plan = _validated_horizons and all(
            query.cpu_prefixes is not None for queries in queries_by_row for query in queries
        )
        plan_device = torch.device("cpu") if cpu_plan else input_ids.device

        row_vectors: list[tuple[torch.Tensor, ...]] = []
        for queries in queries_by_row:
            anchors: list[torch.Tensor] = []
            prefixes: list[torch.Tensor] = []
            keep: list[torch.Tensor] = []
            horizon_lengths: list[torch.Tensor] = []
            probability_starts: list[torch.Tensor] = []
            probability_offsets: list[torch.Tensor] = []
            probability_counts: list[torch.Tensor] = []
            for query in queries:
                entry = query.entry
                query_prefixes = query.cpu_prefixes if cpu_plan else query.prefixes
                query_keep = torch.ones_like(query_prefixes, dtype=torch.bool) if cpu_plan else query.keep_mask
                query_anchors = query_prefixes + entry.horizon.start - 1
                anchors.append(
                    torch.where(query_keep, query_anchors, torch.zeros_like(query_anchors))
                )
                prefixes.append(query_prefixes)
                keep.append(query_keep)
                horizon_lengths.append(
                    torch.full_like(query_prefixes, entry.horizon.ordinary_length)
                )
                probability_starts.append(
                    torch.full_like(query_prefixes, entry.target_probability_start)
                )
                probability_offsets.append(
                    torch.full_like(query_prefixes, entry.target_distribution_offset)
                )
                probability_counts.append(
                    torch.full_like(query_prefixes, entry.target_probability_count)
                )

            empty_long = torch.empty(0, device=plan_device, dtype=torch.long)
            empty_bool = torch.empty(0, device=plan_device, dtype=torch.bool)
            row_vectors.append(
                (
                    torch.cat(anchors) if anchors else empty_long,
                    torch.cat(prefixes) if prefixes else empty_long,
                    torch.cat(keep) if keep else empty_bool,
                    torch.cat(horizon_lengths) if horizon_lengths else empty_long,
                    torch.cat(probability_starts) if probability_starts else empty_long,
                    torch.cat(probability_offsets) if probability_offsets else empty_long,
                    torch.cat(probability_counts) if probability_counts else empty_long,
                )
            )

        batch_capacity = (
            self.edr_chunk_size
            if self.edr_full_anchor_backprop
            else max(self.edr_chunk_size, self.num_anchors)
        )
        if batch_capacity < batch_size:
            raise ValueError(
                "EDR gradient batch capacity must be at least the number of rows in flight"
            )
        blocks_per_row = max(1, batch_capacity // batch_size)
        max_blocks = max(vectors[0].numel() for vectors in row_vectors)
        rejection_cache_remaining = self.edr_rejection_cache_max_bytes
        cost_chunks_by_row: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
        acceptance_chunks_by_row: list[list[torch.Tensor]] = [
            [] for _ in range(batch_size)
        ]
        valid_chunks_by_row: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]

        for chunk_start in range(0, max_blocks, blocks_per_row):
            # Full chunks use blocks_per_row slots; the final chunk is sized to
            # the longest remaining row tail.
            query_slots = min(blocks_per_row, max_blocks - chunk_start)
            long_values = [
                torch.zeros(
                    (batch_size, query_slots),
                    device=plan_device,
                    dtype=torch.long,
                )
                for _ in range(6)
            ]
            block_keep = torch.zeros(
                (batch_size, query_slots),
                device=plan_device,
                dtype=torch.bool,
            )
            real_counts: list[int] = []
            for row_index, vectors in enumerate(row_vectors):
                real_count = min(query_slots, max(0, vectors[0].numel() - chunk_start))
                real_counts.append(real_count)
                if not real_count:
                    continue
                source = slice(chunk_start, chunk_start + real_count)
                # anchors, prefixes, then four horizon/target metadata vectors
                for destination, values in zip(
                    long_values,
                    (vectors[0], vectors[1], *vectors[3:]),
                    strict=True,
                ):
                    destination[row_index, :real_count] = values[source]
                block_keep[row_index, :real_count] = vectors[2][source]

            validated_anchors = None
            if cpu_plan:
                validated_anchors = _EDRValidatedAnchors.from_cpu(
                    long_values[0], long_values[1], long_values[2], block_keep,
                    width=self.edr_proposal_width, sequence_length=input_ids.shape[1],
                    device=input_ids.device,
                )
                long_values = [
                    validated_anchors.anchors if index == 0 else value.to(input_ids.device, non_blocking=True)
                    for index, value in enumerate(long_values)
                ]
                block_keep = validated_anchors.keep_mask
            cache_bytes = (
                batch_size * query_slots * self.edr_proposal_width
                * target_distribution.vocab_size
            )
            cache_mask = (
                isinstance(target_distribution, EDRTargetDistribution)
                and cache_bytes <= rejection_cache_remaining
            )
            if cache_mask:
                rejection_cache_remaining -= cache_bytes
            chunk_costs, chunk_acceptance, chunk_valid = self._edr_chunk_statistics(
                input_ids=input_ids,
                hidden_states_list=hidden_states_list,
                loss_mask=loss_mask,
                lm_head_weight=lm_head_weight,
                target_distribution=target_distribution,
                anchors=long_values[0],
                prefixes=long_values[1],
                horizon_lengths=long_values[2],
                target_probability_start=long_values[3],
                target_distribution_offsets=long_values[4],
                target_probability_counts=long_values[5],
                block_keep_mask=block_keep,
                attention_mask=attention_mask,
                ctx_doc_ids=ctx_doc_ids,
                base_position_ids=base_position_ids,
                draft_context_cache=draft_context_cache,
                cache_rejection_mask=cache_mask,
                _validated_anchors=validated_anchors,
            )
            for row_index, real_count in enumerate(real_counts):
                if not real_count:
                    continue
                cost_chunks_by_row[row_index].append(chunk_costs[row_index, :real_count])
                acceptance_chunks_by_row[row_index].append(
                    chunk_acceptance[row_index, :real_count]
                )
                valid_chunks_by_row[row_index].append(chunk_valid[row_index, :real_count])

        surrogate = target_distribution.stop_probabilities.new_zeros(())
        for row_index, queries in enumerate(queries_by_row):
            if not queries:
                continue
            all_costs = torch.cat(cost_chunks_by_row[row_index], dim=0)
            all_acceptance = torch.cat(acceptance_chunks_by_row[row_index], dim=0)
            all_valid = torch.cat(valid_chunks_by_row[row_index], dim=0)
            cursor = 0
            for query in queries:
                query_size = query.prefixes.numel()
                selected_valid = all_valid[cursor : cursor + query_size]
                selected_costs = torch.where(
                    selected_valid,
                    all_costs[cursor : cursor + query_size],
                    torch.zeros_like(all_costs[cursor : cursor + query_size]),
                )
                selected_acceptance = torch.where(
                    selected_valid,
                    all_acceptance[cursor : cursor + query_size],
                    torch.zeros_like(all_acceptance[cursor : cursor + query_size]),
                )
                cursor += query_size
                if self.edr_full_anchor_backprop:
                    surrogate = surrogate + edr_surrogate_sum(
                        selected_costs,
                        selected_acceptance,
                        query.dynamic_program,
                    )
                elif query.coefficients is not None:
                    surrogate = surrogate + edr_surrogate_sum_from_coefficients(
                        selected_costs, selected_acceptance, query.coefficients,
                    )
                else:
                    if query.inclusion_probabilities is None:
                        raise RuntimeError(
                            "sampled EDR queries require inclusion probabilities"
                        )
                    surrogate = surrogate + sampled_edr_surrogate_sum(
                        selected_costs,
                        selected_acceptance,
                        query.dynamic_program,
                        query.prefixes,
                        query.inclusion_probabilities,
                        query.keep_mask,
                        query.inverse_pps_scale,
                    )
        return surrogate

    def _zero_edr_loss(self) -> torch.Tensor:
        """Return a differentiable zero for an all-degenerate EDR batch."""
        for parameter in self.draft_model.parameters():
            if parameter.requires_grad:
                return parameter.reshape(-1)[0] * 0.0
        raise RuntimeError("EDR requires at least one trainable draft-model parameter")

    def _forward_edr_cross_row(
        self,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        return self.finish_edr_batch(self._prepare_edr_cross_row(**kwargs))

    def _prepare_edr_cross_row(
        self,
        *,
        input_ids: torch.Tensor,
        hidden_states_list: List[torch.Tensor],
        loss_mask: torch.Tensor,
        lm_head_weight: torch.Tensor,
        normalized_target_hidden: torch.Tensor,
        horizons: Sequence[EDRHorizon],
        attention_mask: Optional[torch.Tensor],
        ctx_doc_ids: Optional[torch.Tensor],
        base_position_ids: Optional[torch.Tensor],
    ) -> EDRPreparedBatch:
        """Detached statistics phase of cross-row EDR."""
        batch_size = input_ids.shape[0]
        if self.edr_chunk_size < batch_size:
            raise ValueError(
                "EDR chunk size must be at least the number of input rows in flight"
            )
        horizons_by_row: list[list[EDRHorizon]] = [[] for _ in range(batch_size)]
        degenerate_horizons = 0
        total_horizon_tokens = 0
        # Copy each small mask to CPU once instead of reading it per horizon/chunk.
        validation_attention = validation_docs = None
        if attention_mask is not None:
            validation_attention = attention_mask.detach().to("cpu")
        if ctx_doc_ids is not None:
            validation_docs = ctx_doc_ids.detach().to("cpu")

        for horizon in horizons:
            row_index = horizon.batch_index
            if row_index < 0 or row_index >= batch_size:
                raise ValueError("EDR horizon batch index is out of bounds")
            if not 0 <= horizon.start <= horizon.boundary < input_ids.shape[1]:
                raise ValueError("EDR horizon token bounds are outside the input sequence")
            length = horizon.ordinary_length
            total_horizon_tokens += length
            if length == 0:
                degenerate_horizons += 1
                continue
            if horizon.start <= 0:
                raise ValueError(
                    "EDR horizon has no causal initialization anchor before its first "
                    f"supervised token (batch={row_index}, start={horizon.start})"
                )
            if validation_attention is not None and not bool(
                validation_attention[row_index, horizon.start - 1] > 0
            ):
                raise ValueError("EDR initialization anchor is masked as padding")
            if validation_docs is not None:
                anchor_document = int(validation_docs[row_index, horizon.start - 1].item())
                if anchor_document < 0 or anchor_document != horizon.document_id:
                    raise ValueError(
                        "EDR initialization anchor must belong to the horizon's packed document"
                    )
            horizons_by_row[row_index].append(horizon)

        flat_statistics = []
        target_distribution = statistics_context_cache = None
        active_horizons = [horizon for row in horizons_by_row for horizon in row]
        if active_horizons:
            statistics_context_cache = self._prepare_edr_statistics_context_cache(
                input_ids,
                hidden_states_list,
                base_position_ids,
            )
            with isolated_flex_attention_fallback():
                statistics_by_row, target_distribution = self._edr_all_row_statistics(
                    input_ids=input_ids,
                    hidden_states_list=hidden_states_list,
                    loss_mask=loss_mask,
                    lm_head_weight=lm_head_weight,
                    normalized_target_hidden=normalized_target_hidden,
                    horizons_by_row=horizons_by_row,
                    attention_mask=attention_mask,
                    ctx_doc_ids=ctx_doc_ids,
                    base_position_ids=base_position_ids,
                    draft_context_cache=statistics_context_cache,
                    _validated_horizons=True,
                )

            flat_statistics = [
                statistics for row_statistics in statistics_by_row for statistics in row_statistics
            ]
        return EDRPreparedBatch(
            inputs=dict(
                input_ids=input_ids, hidden_states_list=hidden_states_list,
                loss_mask=loss_mask, lm_head_weight=lm_head_weight,
                attention_mask=attention_mask, ctx_doc_ids=ctx_doc_ids,
                base_position_ids=base_position_ids,
            ),
            horizons=horizons, statistics=flat_statistics,
            target_distribution=target_distribution, context_cache=statistics_context_cache,
            degenerate_horizons=degenerate_horizons, total_horizon_tokens=total_horizon_tokens,
            owner=id(self),
        )

    def finish_edr_batch(
        self, prepared: EDRPreparedBatch,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Consume exactly one prepared group at the same optimizer parameter version."""
        if prepared.owner != id(self) or prepared.consumed:
            raise ValueError("EDR prepared batch belongs to another model or was already consumed")
        prepared.consumed = True
        inputs = prepared.inputs
        input_ids = inputs["input_ids"]
        device = input_ids.device
        batch_size = input_ids.shape[0]
        width = self.edr_proposal_width
        expected_passes_sum = torch.tensor(
            float(prepared.degenerate_horizons), device=device, dtype=torch.float32,
        )
        cost_sum_per_position = torch.zeros(width, device=device, dtype=torch.float32)
        acceptance_sum_per_position = torch.zeros_like(cost_sum_per_position)
        count_per_position = torch.zeros_like(cost_sum_per_position)
        surrogate_sum = (
            self._zero_edr_loss() if torch.is_grad_enabled() else expected_passes_sum * 0
        )
        flat_statistics = prepared.statistics
        target_distribution = prepared.target_distribution
        if flat_statistics:
            dynamic_programs = exact_edr_dynamic_programs(
                [
                    (statistics.costs, statistics.acceptance)
                    for statistics in flat_statistics
                ],
                num_proposals=width,
                max_workers=self.edr_dp_workers,
                return_on_cpu=torch.is_grad_enabled() and not self.edr_full_anchor_backprop,
            )
            queries_by_row = [[] for _ in range(batch_size)]
            for statistics, dynamic_program in zip(
                flat_statistics,
                dynamic_programs,
                strict=True,
            ):
                entry = statistics.entry
                costs = statistics.costs
                acceptance = statistics.acceptance
                expected_passes_sum = expected_passes_sum + dynamic_program.expected_passes.to(device)

                valid = dynamic_program.learned_mask.to(device)
                cost_sum_per_position += torch.where(
                    valid, costs.detach(), torch.zeros_like(costs)
                ).sum(dim=0)
                acceptance_sum_per_position += torch.where(
                    valid,
                    acceptance.detach(),
                    torch.zeros_like(acceptance),
                ).sum(dim=0)
                count_per_position += valid.sum(dim=0)

                if not torch.is_grad_enabled():
                    surrogate_sum = surrogate_sum + edr_surrogate_sum(
                        costs,
                        acceptance,
                        dynamic_program,
                    )
                    continue

                query = self._prepare_edr_gradient_query(entry, dynamic_program, device)
                if query is not None:
                    queries_by_row[entry.horizon.batch_index].append(query)

            if any(queries_by_row):
                if target_distribution is None:
                    raise RuntimeError("EDR gradient queries require a target distribution")
                gradient_context_cache = prepared.context_cache
                if not self.edr_reuse_context_cache:
                    gradient_context_cache = self._prepare_edr_context_cache(
                        input_ids,
                        inputs["hidden_states_list"],
                        inputs["base_position_ids"],
                    )
                surrogate_sum = surrogate_sum + self._edr_gradient_row_surrogate(
                    **inputs,
                    target_distribution=target_distribution,
                    queries_by_row=queries_by_row,
                    draft_context_cache=gradient_context_cache,
                    _validated_horizons=True,
                )

        horizon_count = len(prepared.horizons)
        safe_position_counts = count_per_position.clamp(min=1.0)
        loss_per_position = cost_sum_per_position / safe_position_counts
        acc_per_position = acceptance_sum_per_position / safe_position_counts
        total_learned_states = count_per_position.sum().clamp(min=1.0)
        accuracy = acceptance_sum_per_position.sum() / total_learned_states
        loss_components = {
            "edr_surrogate_loss": surrogate_sum.detach(),
            "edr_weighted_cost_sum": expected_passes_sum.detach(),
            "edr_num_horizons": torch.tensor(float(horizon_count), device=device),
            "edr_num_degenerate_horizons": torch.tensor(
                float(prepared.degenerate_horizons), device=device
            ),
            "edr_generated_tokens": torch.tensor(float(prepared.total_horizon_tokens), device=device),
        }
        return (
            surrogate_sum,
            accuracy.detach(),
            loss_per_position.detach(),
            acc_per_position.detach(),
            count_per_position.detach(),
            loss_components,
        )

    def _forward_edr(
        self,
        *,
        input_ids: torch.Tensor,
        hidden_states_list: List[torch.Tensor],
        loss_mask: torch.Tensor,
        lm_head_weight: torch.Tensor,
        last_hidden_states: Optional[torch.Tensor],
        target_norm: Optional[nn.Module],
        attention_mask: Optional[torch.Tensor],
        ctx_doc_ids: Optional[torch.Tensor],
        base_position_ids: Optional[torch.Tensor],
        return_draft: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Compute exact EDR metrics and the exact (Algorithm 1) or sampled (Eq. (29)) TD surrogate."""
        if return_draft:
            raise ValueError("EDR is a replacement objective and cannot be combined with OPD")
        if last_hidden_states is None:
            raise ValueError(
                "EDR requires target last_hidden_states; set "
                "inference.store_last_hidden_states=true"
            )
        if target_norm is None:
            raise ValueError("EDR requires the target model's final normalization layer")

        device = input_ids.device
        horizons = extract_edr_horizons(loss_mask, attention_mask, ctx_doc_ids)
        with torch.no_grad():
            normalized_target_hidden = target_norm(last_hidden_states.detach()).to(
                lm_head_weight.dtype
            )

        if input_ids.shape[0] > 1:
            return self._forward_edr_cross_row(
                input_ids=input_ids,
                hidden_states_list=hidden_states_list,
                loss_mask=loss_mask,
                lm_head_weight=lm_head_weight,
                normalized_target_hidden=normalized_target_hidden,
                horizons=horizons,
                attention_mask=attention_mask,
                ctx_doc_ids=ctx_doc_ids,
                base_position_ids=base_position_ids,
            )

        width = self.edr_proposal_width
        cost_sum_per_position = torch.zeros(width, device=device, dtype=torch.float32)
        acceptance_sum_per_position = torch.zeros_like(cost_sum_per_position)
        count_per_position = torch.zeros_like(cost_sum_per_position)
        expected_passes_sum = torch.zeros((), device=device, dtype=torch.float32)
        surrogate_sum = (
            self._zero_edr_loss() if torch.is_grad_enabled() else expected_passes_sum * 0
        )
        degenerate_horizons = 0
        total_horizon_tokens = 0
        horizon_cursor = 0
        while horizon_cursor < len(horizons):
            row_index = horizons[horizon_cursor].batch_index
            row_end = horizon_cursor + 1
            while row_end < len(horizons) and horizons[row_end].batch_index == row_index:
                row_end += 1
            row_horizons = horizons[horizon_cursor:row_end]
            horizon_cursor = row_end

            row = slice(row_index, row_index + 1)
            row_input_ids = input_ids[row]
            row_hidden_states = [hidden[row] for hidden in hidden_states_list]
            row_loss_mask = loss_mask[row]
            row_attention_mask = attention_mask[row] if attention_mask is not None else None
            row_target_hidden = normalized_target_hidden[row]
            row_doc_ids = ctx_doc_ids[row] if ctx_doc_ids is not None else None
            row_position_ids = base_position_ids[row] if base_position_ids is not None else None

            active_horizons: list[EDRHorizon] = []
            for horizon in row_horizons:
                length = horizon.ordinary_length
                total_horizon_tokens += length
                if length == 0:
                    degenerate_horizons += 1
                    expected_passes_sum = expected_passes_sum + 1.0
                    continue
                if horizon.start <= 0:
                    raise ValueError(
                        "EDR horizon has no causal initialization anchor before its first "
                        f"supervised token (batch={horizon.batch_index}, start={horizon.start})"
                    )
                if attention_mask is not None and not bool(
                    attention_mask[horizon.batch_index, horizon.start - 1].item()
                ):
                    raise ValueError("EDR initialization anchor is masked as padding")
                if ctx_doc_ids is not None:
                    anchor_document = int(
                        ctx_doc_ids[horizon.batch_index, horizon.start - 1].item()
                    )
                    if anchor_document < 0 or anchor_document != horizon.document_id:
                        raise ValueError(
                            "EDR initialization anchor must belong to the horizon's "
                            "packed document"
                        )
                active_horizons.append(horizon)

            if not active_horizons:
                continue

            has_learned_targets = any(horizon.ordinary_length > 0 for horizon in active_horizons)
            statistics_context_cache = None
            if has_learned_targets:
                statistics_context_cache = self._prepare_edr_statistics_context_cache(
                    row_input_ids,
                    row_hidden_states,
                    row_position_ids,
                )

            # Phase 1: concatenate this row's target positions and every
            # reachable round start. Chunk boundaries may cross horizons, but
            # each block is attention-isolated and only compact per-horizon
            # [L, W] statistics are kept.
            with isolated_flex_attention_fallback():
                horizon_statistics, target_distribution = self._edr_all_horizon_statistics(
                    input_ids=row_input_ids,
                    hidden_states_list=row_hidden_states,
                    loss_mask=row_loss_mask,
                    lm_head_weight=lm_head_weight,
                    normalized_target_hidden=row_target_hidden,
                    horizons=active_horizons,
                    attention_mask=row_attention_mask,
                    ctx_doc_ids=row_doc_ids,
                    base_position_ids=row_position_ids,
                    draft_context_cache=statistics_context_cache,
                )

            dynamic_programs = exact_edr_dynamic_programs(
                [
                    (statistics.costs, statistics.acceptance)
                    for statistics in horizon_statistics
                ],
                num_proposals=self.edr_proposal_width,
                max_workers=self.edr_dp_workers,
                return_on_cpu=torch.is_grad_enabled() and not self.edr_full_anchor_backprop,
            )

            gradient_queries: list[_EDRGradientQuery] = []
            for statistics, dynamic_program in zip(
                horizon_statistics,
                dynamic_programs,
                strict=True,
            ):
                entry = statistics.entry
                costs = statistics.costs
                acceptance = statistics.acceptance
                expected_passes_sum = expected_passes_sum + dynamic_program.expected_passes.to(device)

                valid = dynamic_program.learned_mask.to(device)
                metric_costs = costs.detach()
                metric_acceptance = acceptance.detach()
                cost_sum_per_position += torch.where(valid, metric_costs, 0.0).sum(dim=0)
                acceptance_sum_per_position += torch.where(
                    valid,
                    metric_acceptance,
                    0.0,
                ).sum(dim=0)
                count_per_position += valid.sum(dim=0)

                if not torch.is_grad_enabled():
                    surrogate_sum = surrogate_sum + edr_surrogate_sum(
                        costs,
                        acceptance,
                        dynamic_program,
                    )
                    continue
                if entry.target_probability_count == 0:
                    continue

                query = self._prepare_edr_gradient_query(entry, dynamic_program, device)
                if query is not None:
                    gradient_queries.append(query)

            if gradient_queries:
                if target_distribution is None:
                    raise RuntimeError("EDR gradient queries require a target distribution")
                gradient_context_cache = statistics_context_cache
                if not self.edr_reuse_context_cache:
                    gradient_context_cache = self._prepare_edr_context_cache(
                        row_input_ids,
                        row_hidden_states,
                        row_position_ids,
                    )
                surrogate_sum = surrogate_sum + self._edr_gradient_horizon_surrogate(
                    input_ids=row_input_ids,
                    hidden_states_list=row_hidden_states,
                    loss_mask=row_loss_mask,
                    lm_head_weight=lm_head_weight,
                    target_distribution=target_distribution,
                    queries=gradient_queries,
                    attention_mask=row_attention_mask,
                    ctx_doc_ids=row_doc_ids,
                    base_position_ids=row_position_ids,
                    draft_context_cache=gradient_context_cache,
                )

        horizon_count = len(horizons)
        # The surrogate is summed over horizons. The trainer normalizes by its
        # fixed number of rows/micro-batches, so a horizon's gradient is
        # independent of which horizons share its row.
        loss = surrogate_sum
        safe_position_counts = count_per_position.clamp(min=1.0)
        loss_per_position = cost_sum_per_position / safe_position_counts
        acc_per_position = acceptance_sum_per_position / safe_position_counts
        total_learned_states = count_per_position.sum().clamp(min=1.0)
        accuracy = acceptance_sum_per_position.sum() / total_learned_states

        loss_components = {
            "edr_surrogate_loss": loss.detach(),
            # These two quantities are additive across rows and data-parallel
            # ranks. The trainer forms corpus-style MAL only after summing both.
            "edr_weighted_cost_sum": expected_passes_sum.detach(),
            "edr_num_horizons": torch.tensor(float(horizon_count), device=device),
            "edr_num_degenerate_horizons": torch.tensor(float(degenerate_horizons), device=device),
            "edr_generated_tokens": torch.tensor(float(total_horizon_tokens), device=device),
        }
        return (
            loss,
            accuracy.detach(),
            loss_per_position.detach(),
            acc_per_position.detach(),
            count_per_position.detach(),
            loss_components,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states_list: List[torch.Tensor],
        loss_mask: torch.Tensor,
        lm_head_weight: torch.Tensor,
        last_hidden_states: Optional[torch.Tensor] = None,
        target_norm: Optional[nn.Module] = None,
        attention_mask: Optional[torch.Tensor] = None,
        ctx_doc_ids: Optional[torch.Tensor] = None,
        base_position_ids: Optional[torch.Tensor] = None,
        injected_anchors: Optional[torch.Tensor] = None,
        injected_keep_mask: Optional[torch.Tensor] = None,
        return_draft: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Full DFlash training forward pass.

        Args:
            input_ids: [B, seq_len] token IDs.
            hidden_states_list: per-target-layer [B, seq_len, D] hidden states.
            loss_mask: [B, seq_len] 1 for supervised positions.
            lm_head_weight: frozen target LM head weight.
            last_hidden_states: [B, seq_len, D] target pre-norm final hidden states
                (from mooncake); required for KL/LK/L1/e2e-TV distillation. None
                disables the distillation terms (base CE path).
            target_norm: target model's final RMSNorm module, applied to
                ``last_hidden_states`` before the LM head so teacher logits match
                inference-time output. None => teacher logits from raw hidden.
            ctx_doc_ids: [B, seq_len] long — per-token document id (padding=-1)
                for sequence packing. When None (default), the whole sequence is
                treated as one document (legacy pad-to-longest path). When given,
                anchor sampling, the block-causal mask, and RoPE positions all
                become doc-aware so packed segments never attend across doc
                boundaries.
            base_position_ids: [B, seq_len] long — doc-local context positions
                (reset to 0 at each doc boundary) used for RoPE when packing.
                Required to be paired with ``ctx_doc_ids``; ignored when
                ``ctx_doc_ids`` is None.
            injected_anchors: [B, num_anchors] — bypass random anchor sampling
                (parity tests only). See ``_sample_anchor_positions``.
            injected_keep_mask: [B, num_anchors] bool — validity for injected
                anchors.
            return_draft: half on-policy OPD — additionally return the OPD dict
                (grad-carrying draft hidden + detached proposals/anchors) as a
                trailing 7th element so the trainer can build the tree, score it,
                and add the OPD-KL term.

        Returns:
            loss: scalar training loss (objective-weighted, + optional distill)
            accuracy: scalar accuracy (binary mask, no decay)
            loss_per_position: [proposal_width] mean loss for learned proposals.
            acc_per_position: [proposal_width] mean accuracy at each learned position.
            count_per_position: [proposal_width] valid label count at each learned
                position before loss decay is applied
            loss_components: dict of per-component loss scalars for logging
                (``ce_loss``/``kl_loss``/``lk_loss``; subclasses add more).
            opd (only when return_draft=True): dict for OPD tree scoring.
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        if self.loss_objective == "edr":
            if injected_anchors is not None or injected_keep_mask is not None:
                raise ValueError(
                    "EDR enumerates round starts exactly; injected anchors are invalid"
                )
            return self._forward_edr(
                input_ids=input_ids,
                hidden_states_list=hidden_states_list,
                loss_mask=loss_mask,
                lm_head_weight=lm_head_weight,
                last_hidden_states=last_hidden_states,
                target_norm=target_norm,
                attention_mask=attention_mask,
                ctx_doc_ids=ctx_doc_ids,
                base_position_ids=base_position_ids,
                return_draft=return_draft,
            )

        # 1-6. Shared backbone → draft hidden states + anchor bookkeeping
        #      (doc-aware + injection args threaded through for packing/parity).
        draft_hidden, anchor_positions, block_keep_mask, n_blocks = self._draft_backbone(
            input_ids,
            hidden_states_list,
            loss_mask,
            attention_mask=attention_mask,
            ctx_doc_ids=ctx_doc_ids,
            base_position_ids=base_position_ids,
            injected_anchors=injected_anchors,
            injected_keep_mask=injected_keep_mask,
        )
        draft_hidden = self._select_learned_query_states(draft_hidden, n_blocks)
        proposal_width = self.proposal_width

        # 7. Build labels for the learned proposal slots only. An input-anchor
        #    query slot takes part in backbone attention but not in this layout.
        proposal_layout = build_dflash_proposal_layout(
            anchor_positions,
            sequence_length=seq_len,
            block_size=proposal_width,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            ctx_doc_ids=ctx_doc_ids,
        )
        safe_label_indices = proposal_layout.safe_label_indices
        valid_label_mask = proposal_layout.valid_mask
        target_ids = proposal_layout.gather_labels(input_ids)
        prev_token_ids = proposal_layout.gather_predecessor_tokens(input_ids)

        # Chunked draft-logit projection (full-vocab logits + their CE gradient
        # are ~half the training-step peak). Only the production path — decay
        # objective, no distillation, no subclass teacher head — is chunked.
        chunk = _dflash_loss_chunk()
        distill_active = (
            self.l1_loss_alpha > 0
            or (self.lk_loss_weight > 0.0 and last_hidden_states is not None)
            or (self.kl_loss_weight > 0.0 and last_hidden_states is not None)
            or (self.e2e_tv_loss_weight > 0.0 and last_hidden_states is not None)
            or self._extra_distill_needed()
        )
        if chunk > 0 and self.loss_objective == "decay" and not distill_active:
            return self._forward_chunked_decay(
                input_ids=input_ids,
                draft_hidden=draft_hidden,
                target_ids=target_ids,
                prev_token_ids=prev_token_ids,
                block_keep_mask=block_keep_mask,
                valid_label_mask=valid_label_mask,
                safe_label_indices=safe_label_indices,
                loss_mask=loss_mask,
                lm_head_weight=lm_head_weight,
                n_blocks=n_blocks,
                anchor_positions=anchor_positions,
                chunk=chunk,
                return_draft=return_draft,
            )

        # Establish validity before the expensive vocabulary projection. Keep
        # this same rectangular mask for every loss/metric denominator below.
        weight_mask = block_keep_mask.unsqueeze(-1).expand(-1, -1, proposal_width).float()
        weight_mask = weight_mask * valid_label_mask.float()

        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, n_blocks, -1),
            2,
            safe_label_indices,
        )
        weight_mask = weight_mask * original_loss_mask_gathered

        # Pure E2E/LK projects logits only at supervised positions. Auxiliary
        # losses and OPD use the full rectangular layout.
        compact_distillation = (
            last_hidden_states is not None
            and (self.lk_loss_weight > 0.0 or self.e2e_tv_loss_weight > 0.0)
            and self.ce_loss_alpha == 0.0 and self.l1_loss_alpha == 0.0
            and self.kl_loss_weight == 0.0 and not self._extra_distill_needed()
            and not return_draft
        )
        projection_indices = (
            self._distill_projection_indices(weight_mask) if compact_distillation else None
        )
        if projection_indices is None:
            logits = self._compute_draft_logits(
                draft_hidden, lm_head_weight, prev_token_ids, n_blocks,
            )
        else:
            logits = self._project_selected_distill_logits(
                draft_hidden, lm_head_weight, prev_token_ids, projection_indices,
            )
        if self.fp32_lm_head:
            logits = logits.float()

        def restore_positions(values):
            # Scatter per-position scalars back to the rectangular layout so
            # row/block identities and weighted-mean denominators match.
            if projection_indices is None:
                return values
            return values.new_zeros(weight_mask.numel()).index_copy(0, projection_indices, values)

        # Binary mask BEFORE objective weighting — accuracy measures "did we
        # predict correctly?" uniformly; weighting only shapes gradient.
        binary_eval_mask = weight_mask.view(-1)

        # 9a. Per-token loss: ce_loss_alpha*CE + l1_loss_alpha*L1.
        vocab_size = logits.size(-1)
        flat_logits = logits.view(-1, vocab_size)
        flat_targets = target_ids.view(-1)
        if projection_indices is not None:
            flat_targets = flat_targets.index_select(0, projection_indices)
        reuse_streamed_student_statistics = (
            self.ce_loss_alpha == 0.0
            and last_hidden_states is not None
            and (self.lk_loss_weight > 0.0 or self.e2e_tv_loss_weight > 0.0)
        )
        student_log_normalizers = None
        streamed_pred_ids = None
        student_logit_scale = 1.0 / self.distill_temperature
        if reuse_streamed_student_statistics or (
            self.distill_distribution_aware and self.loss_objective == "dpace"
        ):
            student_log_normalizers, streamed_pred_ids = (
                streaming_student_log_normalizers_and_argmax(
                    flat_logits,
                    self.edr_vocab_chunk_size,
                    logit_scale=student_logit_scale,
                )
            )
        confidence_nll = None
        if student_log_normalizers is not None:
            with torch.no_grad():
                realized_logits = torch.gather(
                    flat_logits.detach(), -1, flat_targets.unsqueeze(-1),
                ).squeeze(-1).float()
                confidence_nll = student_log_normalizers - realized_logits * student_logit_scale
        # With ce_loss_alpha == 0, CE is computed only as a detached metric (and
        # D-PACE confidence), so it adds no full-vocabulary autograd graph.
        if self.ce_loss_alpha > 0:
            ce_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
            loss_per_token = self.ce_loss_alpha * ce_per_token
        elif student_log_normalizers is not None:
            assert confidence_nll is not None
            ce_per_token = confidence_nll
            loss_per_token = torch.zeros_like(ce_per_token)
        else:
            with torch.no_grad():
                # Compute the detached CE metric in FP32. The FP32 copy is a
                # temporary, so the streamed loss keeps the logits' own dtype.
                ce_per_token = F.cross_entropy(
                    flat_logits.float(),
                    flat_targets,
                    reduction="none",
                )
            loss_per_token = torch.zeros_like(ce_per_token)
        ce_per_token = restore_positions(ce_per_token)
        loss_per_token = restore_positions(loss_per_token)
        l1_per_token = None
        if self.l1_loss_alpha > 0:
            if last_hidden_states is None:
                raise ValueError(
                    "DFlash L1 distillation (l1_loss_alpha > 0) requires target "
                    "last_hidden_states; set inference.store_last_hidden_states=true in the "
                    "run config."
                )
            tgt_idx = (safe_label_indices - 1).clamp(min=0)
            hdim = last_hidden_states.size(-1)
            gather_idx = tgt_idx.reshape(bsz, -1, 1).expand(-1, -1, hdim)
            aligned_hidden = torch.gather(last_hidden_states, 1, gather_idx)
            target_logits = F.linear(aligned_hidden, lm_head_weight).view(-1, vocab_size)
            l1_per_token = self._compute_l1_loss(flat_logits, target_logits)
            loss_per_token = loss_per_token + self.l1_loss_alpha * l1_per_token

        # Reported per-position loss: the observed-token NLL at the distillation
        # temperature when streamed student statistics replace CE, else CE at T=1.
        loss_per_token_by_position = ce_per_token.view(bsz, n_blocks, proposal_width)

        # 9b. Objective weighting: exp-decay or D-PACE continuation value.
        objective_weights = weight_mask
        if (
            self.loss_objective == "decay"
            and self.loss_decay_gamma is not None
            and self.loss_decay_gamma > 0
        ):
            k = torch.arange(proposal_width, device=device).view(1, 1, -1)
            decay_weights = torch.exp(-k.float() / self.loss_decay_gamma)
            objective_weights = weight_mask * decay_weights
        elif self.loss_objective == "dpace":
            with torch.no_grad():
                confidence_losses = (
                    restore_positions(confidence_nll).view(bsz, n_blocks, proposal_width)
                    if self.distill_distribution_aware and confidence_nll is not None
                    else loss_per_token_by_position
                )
                if projection_indices is not None:
                    confidence_losses = self._complete_dpace_tail_confidences(
                        confidence_losses, draft_hidden, lm_head_weight,
                        prev_token_ids, target_ids, weight_mask,
                    )
                target_confidences = torch.exp(-confidence_losses.float())
                dpace_weights = _dpace_position_weights(target_confidences, self.dpace_alpha).to(
                    dtype=weight_mask.dtype
                )
            objective_weights = weight_mask * dpace_weights

        flat_weights = objective_weights.view(-1)
        valid_token_count = flat_weights.sum().clamp(min=1e-6)
        loss = _weighted_loss_mean(
            loss_per_token,
            flat_weights,
            batch_size=bsz,
            mean_by_row=self.distill_mean_by_row,
        )

        # 9c. Optional KL / LK distillation vs the target's true last-layer logits.
        #     Convex-mix in [0,1] against the base loss; LK precedes KL. Teacher
        #     logits are also computed when a subclass head needs them
        #     (``_extra_distill_needed``) even if KL/LK are off.
        base_loss = loss
        kl_loss = torch.zeros((), device=device, dtype=base_loss.dtype)
        lk_loss = torch.zeros((), device=device, dtype=base_loss.dtype)
        e2e_tv_loss = torch.zeros((), device=device, dtype=base_loss.dtype)

        lk_active = self.lk_loss_weight > 0.0 and last_hidden_states is not None
        kl_active = (
            (not lk_active) and self.kl_loss_weight > 0.0 and last_hidden_states is not None
        )
        e2e_tv_active = self.e2e_tv_loss_weight > 0.0 and last_hidden_states is not None
        want_teacher = lk_active or kl_active or e2e_tv_active or self._extra_distill_needed()

        teacher_logits_flat = None
        teacher_logits_unique = None
        teacher_inverse_indices = None
        if want_teacher and last_hidden_states is not None:
            with torch.no_grad():
                # last_hidden_states is the pre-`norm` slot under vllm capture;
                # apply the target's final RMSNorm before lm_head so teacher
                # logits match what the target emits at inference time.
                lhs = last_hidden_states
                if target_norm is not None:
                    lhs = target_norm(lhs)
                lhs = lhs.to(lm_head_weight.dtype)

                # Teacher LM at position p emits next-token logits, so gather
                # teacher hidden at (anchor+k - 1) to match input_ids[anchor+k].
                teacher_label_indices = (safe_label_indices - 1).clamp(min=0)
                (
                    teacher_logits_unique,
                    teacher_inverse_indices,
                    _,
                ) = (
                    _project_unique_teacher_logits(
                        lhs,
                        teacher_label_indices,
                        lm_head_weight,
                        projection_indices=projection_indices,
                    )
                )
                if kl_active or self._extra_distill_needed():
                    teacher_logits_flat = teacher_logits_unique.index_select(
                        0, teacher_inverse_indices
                    ).detach()

        if lk_active:
            assert teacher_logits_unique is not None
            assert teacher_inverse_indices is not None
            lk_per_position = self._compute_lk_loss(
                student_logits=flat_logits,
                teacher_logits=teacher_logits_unique,
                loss_type=self.lk_loss_type,
                eta=self.lk_eta,
                teacher_row_indices=teacher_inverse_indices,
                student_log_normalizers=student_log_normalizers,
                vocab_chunk_size=self.edr_vocab_chunk_size,
                **self._distill_sampling_kwargs,
            )
            lk_loss = _weighted_loss_mean(
                restore_positions(lk_per_position),
                flat_weights.float(),
                batch_size=bsz,
                mean_by_row=self.distill_mean_by_row,
            ).to(base_loss.dtype)
            distill_w = max(0.0, min(1.0, self.lk_loss_weight))
            loss = (
                lk_loss
                if distill_w >= 1.0
                else distill_w * lk_loss + (1.0 - distill_w) * base_loss
            )
        elif kl_active:
            kl_per_position = self._compute_topk_kl_loss_variant_b(
                student_logits=flat_logits,
                teacher_logits=teacher_logits_flat,
                topk=self.kl_topk,
            )
            kl_loss = (
                (kl_per_position * flat_weights.float()).sum() / valid_token_count.float()
            ).to(base_loss.dtype)
            distill_w = max(0.0, min(1.0, self.kl_loss_weight))
            loss = (
                kl_loss
                if distill_w >= 1.0
                else distill_w * kl_loss + (1.0 - distill_w) * base_loss
            )

        # 9c'. Independent e2e multi-step TV term, added on top of the total
        #      (not mutually exclusive with KL/LK). Bypasses flat_weights/decay.
        if e2e_tv_active and teacher_logits_unique is not None:
            assert teacher_inverse_indices is not None
            vocab_size_e2e = flat_logits.size(-1)
            e2e_tv_loss, _accept_len = self._compute_e2e_tv_loss(
                student_logits_pb=(
                    flat_logits if projection_indices is not None else flat_logits.view(
                        bsz, n_blocks, proposal_width, vocab_size_e2e,
                    )
                ),
                teacher_logits=teacher_logits_unique,
                teacher_row_indices=teacher_inverse_indices,
                valid_mask_pb=weight_mask,
                student_log_normalizers=student_log_normalizers,
                vocab_chunk_size=self.edr_vocab_chunk_size,
                mean_by_row=self.distill_mean_by_row,
                projection_indices=projection_indices,
                **self._distill_sampling_kwargs,
            )
            e2e_tv_loss = e2e_tv_loss.to(base_loss.dtype)
            loss = loss + self.e2e_tv_loss_weight * e2e_tv_loss

        # 9d. Subclass extra-loss hook (DSpark confidence head; no-op for DFlash).
        loss, extra_components = self._compute_extra_loss(
            loss,
            flat_logits,
            teacher_logits_flat,
            flat_weights,
            valid_token_count,
            prev_token_ids,
            n_blocks,
        )

        # 9e. Optional gate-sparsity penalty (gated_sum layer-selection; default off).
        #     Added to the final total so it applies regardless of distillation.
        if self.gate_entropy_weight > 0 and hasattr(self.draft_model, "gate_entropy"):
            loss = loss + self.gate_entropy_weight * self.draft_model.gate_entropy()

        # 10. Accuracy (using binary mask without decay)
        with torch.no_grad():
            pred_ids = (
                streamed_pred_ids
                if streamed_pred_ids is not None
                else torch.argmax(flat_logits, dim=-1)
            )
            correct = restore_positions(pred_ids == flat_targets) & (binary_eval_mask > 0.5)
            actual_token_count = binary_eval_mask.sum().clamp(min=1e-6)
            accuracy = correct.sum().float() / actual_token_count

            # Per-position metrics cover learned proposal slots only.
            binary_weights = binary_eval_mask.view(bsz, n_blocks, proposal_width)
            count_per_position = binary_weights.sum(dim=(0, 1))
            count_per_pos = count_per_position.clamp(min=1.0)

            loss_per_position = (loss_per_token_by_position * binary_weights).sum(
                dim=(0, 1)
            ) / count_per_pos
            acc_per_position = (correct.view(bsz, n_blocks, proposal_width).float()).sum(
                dim=(0, 1)
            ) / count_per_pos

        # Single registration point for all loss terms (for logging). Components
        # are the PURE objective-weighted means of each term (``ce`` and ``l1``
        # separately, not the ``ce_α·ce + l1_α·l1`` combination), so they read as
        # an interpretable decomposition; ``kl``/``lk`` are the distill terms; the
        # extra-loss hook merges its own named components (e.g. confidence_loss).
        # Add a term here + list its key in the trainer's
        # ``_extra_loss_component_keys`` — never change the tuple arity.
        ce_component = _weighted_loss_mean(
            ce_per_token,
            flat_weights,
            batch_size=bsz,
            mean_by_row=self.distill_mean_by_row,
        )
        loss_components = {
            "ce_loss": ce_component.detach(),
            "kl_loss": kl_loss.detach(),
            "lk_loss": lk_loss.detach(),
            "e2e_tv_loss": e2e_tv_loss.detach(),
        }
        if l1_per_token is not None:
            loss_components["l1_loss"] = _weighted_loss_mean(
                l1_per_token,
                flat_weights.float(),
                batch_size=bsz,
                mean_by_row=self.distill_mean_by_row,
            ).detach()
        loss_components.update(extra_components)

        if return_draft:
            # Half on-policy OPD: hand back the grad-carrying draft hidden + the
            # (detached) proposal ids / anchors so the trainer can build the tree,
            # score it on the target, and add the OPD-KL term. An input-anchor
            # query slot is excluded from the proposals.
            opd = {
                "draft_hidden": draft_hidden,
                "proposals": self._greedy_proposals_from_hidden(
                    input_ids=input_ids,
                    draft_hidden=draft_hidden.detach(),
                    anchor_positions=anchor_positions,
                    lm_head_weight=lm_head_weight,
                ),
                "anchor_positions": anchor_positions,
                "block_keep_mask": block_keep_mask,
            }
            return (
                loss,
                accuracy,
                loss_per_position,
                acc_per_position,
                count_per_position,
                loss_components,
                opd,
            )

        return (
            loss,
            accuracy,
            loss_per_position,
            acc_per_position,
            count_per_position,
            loss_components,
        )

    def _forward_chunked_decay(
        self,
        *,
        input_ids: torch.Tensor,
        draft_hidden: torch.Tensor,
        target_ids: torch.Tensor,
        prev_token_ids: torch.Tensor,
        block_keep_mask: torch.Tensor,
        valid_label_mask: torch.Tensor,
        safe_label_indices: torch.Tensor,
        loss_mask: torch.Tensor,
        lm_head_weight: torch.Tensor,
        n_blocks: int,
        anchor_positions: torch.Tensor,
        chunk: int,
        return_draft: bool,
    ):
        """Memory-lean equivalent of ``forward``'s decay + no-distill tail.

        Projects draft logits one block-group at a time instead of materializing
        the full ``[B, n_blocks*block_size, V]`` tensor (and its CE gradient) at
        once. Numerically equals the full path up to summation order; gated on in
        ``forward`` only when the objective is decay and no distillation / subclass
        teacher head is active, so every value here mirrors the corresponding full
        path line exactly. Chunk unit is blocks, so per-block-position weighting
        (decay) and metrics stay intact.
        """
        bsz = draft_hidden.shape[0]
        device = draft_hidden.device
        bs = self.proposal_width
        D = draft_hidden.shape[-1]

        if not getattr(self, "_dflash_chunk_logged", False):
            logger.info(
                "DFlash chunked-projection loss active "
                f"(ANGELSPEC_DFLASH_LOSS_CHUNK={chunk} rows, {max(1, chunk // bs)} blocks/chunk)"
            )
            self._dflash_chunk_logged = True

        # weight_mask — identical to the full path in ``forward`` (block validity
        # × checked labels × supervision). Logit-free, so computed up front.
        weight_mask = block_keep_mask.unsqueeze(-1).expand(-1, -1, bs).float()
        weight_mask = weight_mask * valid_label_mask.float()
        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_label_indices
        )
        weight_mask = weight_mask * original_loss_mask_gathered  # [B, n_blocks, bs]

        # Binary (accuracy) mask is weight_mask before objective weighting.
        binary_eval_mask = weight_mask

        # Decay objective weights (logit-free); matches full path lines 799-801.
        objective_weights = weight_mask
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            k = torch.arange(bs, device=device).view(1, 1, -1)
            decay_weights = torch.exp(-k.float() / self.loss_decay_gamma)
            objective_weights = weight_mask * decay_weights

        valid_token_count = objective_weights.view(-1).sum().clamp(min=1e-6)
        actual_token_count = binary_eval_mask.view(-1).sum().clamp(min=1e-6)
        count_per_position = binary_eval_mask.sum(dim=(0, 1))
        count_per_pos = count_per_position.clamp(min=1.0)

        blocks_per_chunk = max(1, chunk // bs)
        dh = draft_hidden.view(bsz, n_blocks, bs, D)

        loss_num = draft_hidden.new_zeros(())
        ce_comp_num = draft_hidden.new_zeros(())
        correct_num = torch.zeros((), device=device)
        loss_pos_num = torch.zeros(bs, device=device)
        acc_pos_num = torch.zeros(bs, device=device)

        for start in range(0, n_blocks, blocks_per_chunk):
            end = min(start + blocks_per_chunk, n_blocks)
            nb = end - start
            dh_chunk = dh[:, start:end].reshape(bsz, nb * bs, D)
            prev_chunk = prev_token_ids[:, start:end].reshape(bsz, nb * bs)
            logits_chunk = self._compute_draft_logits(dh_chunk, lm_head_weight, prev_chunk, nb)
            if self.fp32_lm_head:
                logits_chunk = logits_chunk.float()
            flat = logits_chunk.view(-1, logits_chunk.size(-1))
            tgt = target_ids[:, start:end].reshape(-1)
            ce = F.cross_entropy(flat, tgt, reduction="none")

            w = objective_weights[:, start:end].reshape(-1)
            loss_num = loss_num + (self.ce_loss_alpha * ce * w).sum()
            ce_comp_num = ce_comp_num + (ce * w).sum()

            with torch.no_grad():
                b = binary_eval_mask[:, start:end].reshape(-1)
                pred = flat.argmax(dim=-1)
                correct = (pred == tgt) & (b > 0.5)
                correct_num = correct_num + correct.sum().float()
                loss_pos_num = loss_pos_num + (ce.view(bsz, nb, bs) * b.view(bsz, nb, bs)).sum(
                    dim=(0, 1)
                )
                acc_pos_num = acc_pos_num + correct.view(bsz, nb, bs).float().sum(dim=(0, 1))

        loss = loss_num / valid_token_count
        if self.gate_entropy_weight > 0 and hasattr(self.draft_model, "gate_entropy"):
            loss = loss + self.gate_entropy_weight * self.draft_model.gate_entropy()
        accuracy = correct_num / actual_token_count
        loss_per_position = loss_pos_num / count_per_pos
        acc_per_position = acc_pos_num / count_per_pos

        zero = torch.zeros((), device=device, dtype=loss.dtype)
        loss_components = {
            "ce_loss": (ce_comp_num / valid_token_count).detach(),
            "kl_loss": zero,
            "lk_loss": zero,
        }

        if return_draft:
            proposals = self._greedy_proposals_from_hidden(
                input_ids=input_ids,
                draft_hidden=draft_hidden.detach(),
                anchor_positions=anchor_positions,
                lm_head_weight=lm_head_weight,
            )
            opd = {
                "draft_hidden": draft_hidden,
                "proposals": proposals,
                "anchor_positions": anchor_positions,
                "block_keep_mask": block_keep_mask,
            }
            return (
                loss,
                accuracy,
                loss_per_position,
                acc_per_position,
                count_per_position,
                loss_components,
                opd,
            )
        return (
            loss,
            accuracy,
            loss_per_position,
            acc_per_position,
            count_per_position,
            loss_components,
        )
