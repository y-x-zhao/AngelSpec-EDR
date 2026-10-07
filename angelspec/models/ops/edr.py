"""Exact Expected Decoding Rounds (EDR) horizon and dynamic-program primitives."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Optional, Sequence

import numba
import numpy as np
import torch

from angelspec.utils.sampling import sampling_logits, validate_sampling_parameters

if TYPE_CHECKING:
    from angelspec.models.ops.edr_sparse_target import EDRSparseTargetDistribution


@dataclass(frozen=True)
class EDRHorizon:
    """A contiguous supervised trajectory with a positional terminal boundary."""

    batch_index: int
    start: int
    boundary: int
    document_id: Optional[int] = None

    @property
    def ordinary_length(self) -> int:
        """Number of optimized tokens before the boundary-only final token."""
        return self.boundary - self.start


@dataclass(frozen=True)
class EDRDynamicProgram:
    """Detached result of the exact pathwise EDR recurrences.

    Learned-state tensors use shape ``[L, B]``. Every column is an actual draft
    proposal: column ``h-1`` represents ``q[n,n+h]``. The input-anchor query
    slot is not a proposal and is excluded from this recurrence.
    ``conditional_survivals[n, t]`` is the within-round acceptance product
    ``omega[n,t] / omega[n,n+1]``.
    """

    expected_passes: torch.Tensor
    round_values: torch.Tensor
    learned_values: torch.Tensor
    continuation_advantages: torch.Tensor
    occupancies: torch.Tensor
    conditional_survivals: torch.Tensor
    round_start_probabilities: torch.Tensor
    learned_mask: torch.Tensor


@dataclass(frozen=True)
class EDRAnchorSample:
    """Random-start systematic sample of EDR round-start indices.

    ``inclusion_probabilities`` contains the marginal probability for every
    candidate round start. ``indices`` contains the distinct starts selected in
    this draw. Zero-occupancy starts may have zero inclusion probability and are
    omitted; callers can pad the selected tensors with masked slots when a fixed
    query length is required. ``inverse_pps_scale`` is ``1/lambda`` for the
    unsaturated marginals ``pi_n=lambda*omega[n,n+1]`` and is ``None`` when no
    unsaturated correction is required.
    """

    indices: torch.Tensor
    inclusion_probabilities: torch.Tensor
    inverse_pps_scale: Optional[torch.Tensor]

    @property
    def selected_inclusion_probabilities(self) -> torch.Tensor:
        return self.inclusion_probabilities[self.indices]


@dataclass(frozen=True)
class EDRSurrogateCoefficients:
    """Detached Eq. (29) coefficients for the selected round starts."""

    weights: torch.Tensor
    continuation_advantages: torch.Tensor
    learned_mask: torch.Tensor

    def to(self, device: torch.device | str) -> EDRSurrogateCoefficients:
        device = torch.device(device)
        if self.weights.device == device:
            return self
        # Pack the small selected fields into one transfer; the bool mask is
        # exact in FP32.
        packed = torch.stack((
            self.weights,
            self.continuation_advantages,
            self.learned_mask.float(),
        )).to(device=device)
        return EDRSurrogateCoefficients(packed[0], packed[1], packed[2].bool())


@dataclass(frozen=True)
class EDRTargetDistribution:
    """Detached target distribution constants used by exact EDR statistics."""

    logits: torch.Tensor
    log_normalizers: torch.Tensor
    stop_probabilities: torch.Tensor
    stopping_token_ids: torch.Tensor
    # Filtered logits retain their original BF16/FP16 storage. Apply temperature
    # only inside FP32 vocabulary tiles, avoiding a full FP32 logits allocation.
    logit_scale: float = 1.0

    @property
    def vocab_size(self) -> int:
        return self.logits.shape[-1]

    @property
    def num_positions(self) -> int:
        return self.logits.shape[0]

    @cached_property
    def non_stopping_vocab(self) -> torch.Tensor:
        """One read-only vocabulary mask shared by statistics and backward tiles.

        Cached on the distribution, so its device, stop set, and lifetime
        follow the target tensors. It is created outside inference mode because
        a gradient pass may save the mask built by a no-grad statistics pass.
        """
        with torch.inference_mode(False):
            mask = torch.ones(
                self.logits.shape[-1], device=self.logits.device, dtype=torch.bool,
            )
            if self.stopping_token_ids.numel():
                mask[self.stopping_token_ids] = False
        return mask


@dataclass(frozen=True)
class _EDRAnchorSamplingDesign:
    """Capped-PPS marginals and the reciprocal of their common scale."""

    inclusion_probabilities: torch.Tensor
    inverse_pps_scale: Optional[torch.Tensor]


def extract_edr_horizons(
    loss_mask: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    ctx_doc_ids: Optional[torch.Tensor] = None,
) -> list[EDRHorizon]:
    """Extract one EDR horizon per contiguous supervised span.

    The final token of every span is positional boundary context, not an EDR
    draft target. Horizons depend only on the masks, not on token IDs.
    """
    if loss_mask.ndim != 2:
        raise ValueError(f"loss_mask must be rank 2, got shape {tuple(loss_mask.shape)}")
    if attention_mask is not None and attention_mask.shape != loss_mask.shape:
        raise ValueError("attention_mask must have the same shape as loss_mask")
    if ctx_doc_ids is not None and ctx_doc_ids.shape != loss_mask.shape:
        raise ValueError("ctx_doc_ids must have the same shape as loss_mask")

    supervised = loss_mask > 0
    if attention_mask is not None:
        supervised = supervised & (attention_mask > 0)

    # Parsing is branch-heavy and Python-driven. Copy each small positional tensor
    # once instead of synchronizing one CUDA scalar per token in the loops below.
    supervised_rows = supervised.detach().to(device="cpu").tolist()
    attention_rows = (
        attention_mask.detach().to(device="cpu").tolist() if attention_mask is not None else None
    )
    document_rows = (
        ctx_doc_ids.detach().to(device="cpu").tolist() if ctx_doc_ids is not None else None
    )

    horizons: list[EDRHorizon] = []
    batch_size, seq_len = loss_mask.shape
    for batch_index in range(batch_size):
        pos = 0
        while pos < seq_len:
            if attention_rows is not None and not bool(attention_rows[batch_index][pos]):
                pos += 1
                continue

            document_id = None
            if document_rows is not None:
                raw_document_id = int(document_rows[batch_index][pos])
                if raw_document_id < 0:
                    pos += 1
                    continue
                document_id = raw_document_id

            segment_end = pos + 1
            while segment_end < seq_len:
                if attention_rows is not None and not bool(
                    attention_rows[batch_index][segment_end]
                ):
                    break
                if (
                    document_rows is not None
                    and int(document_rows[batch_index][segment_end]) != document_id
                ):
                    break
                segment_end += 1

            cursor = pos
            while cursor < segment_end:
                if not supervised_rows[batch_index][cursor]:
                    cursor += 1
                    continue
                start = cursor
                cursor += 1
                while cursor < segment_end and supervised_rows[batch_index][cursor]:
                    cursor += 1
                horizons.append(
                    EDRHorizon(
                        batch_index=batch_index,
                        start=start,
                        boundary=cursor - 1,
                        document_id=document_id,
                    )
                )
            pos = segment_end

    return horizons


def edr_distribution_statistics(
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    target_ids: torch.Tensor,
    stopping_token_ids: Sequence[int] = (),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return conditional non-stopping rejection cost and EDR acceptance.

    Target distributions are always detached. Probability operations run in
    FP32 so BF16/FP16 model execution does not round small acceptance ratios.
    """
    if draft_logits.shape != target_logits.shape:
        raise ValueError("draft_logits and target_logits must have the same shape")
    if target_ids.shape != draft_logits.shape[:-1]:
        raise ValueError("target_ids must match the non-vocabulary logit dimensions")

    target_probabilities = torch.softmax(target_logits.detach().float(), dim=-1)
    return edr_distribution_statistics_from_target_probabilities(
        draft_logits,
        target_probabilities,
        target_ids,
        stopping_token_ids,
    )


def edr_distribution_statistics_from_target_probabilities(
    draft_logits: torch.Tensor,
    target_probabilities: torch.Tensor,
    target_ids: torch.Tensor,
    stopping_token_ids: Sequence[int] = (),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return EDR statistics while reusing precomputed target distributions.

    The immediate cost is the rejection probability conditioned on the target
    token being non-stopping::

        sum_{a not in S} [p(a) - q(a)]_+ / (1 - sum_{s in S} p(s)).

    It is defined as zero when the conditioning event has zero probability.
    """
    if draft_logits.shape != target_probabilities.shape:
        raise ValueError("draft logits and target probabilities must have the same shape")
    if target_ids.shape != draft_logits.shape[:-1]:
        raise ValueError("target_ids must match the non-vocabulary logit dimensions")

    draft_probabilities = torch.softmax(draft_logits.float(), dim=-1)
    target_probabilities = target_probabilities.detach().float()
    stop_ids = _validated_stopping_token_ids(
        stopping_token_ids,
        draft_logits.shape[-1],
        draft_logits.device,
    )
    non_stopping = torch.ones(
        draft_logits.shape[-1],
        device=draft_logits.device,
        dtype=torch.bool,
    )
    if stop_ids.numel():
        non_stopping[stop_ids] = False
        stop_probability = target_probabilities.index_select(-1, stop_ids).sum(dim=-1)
    else:
        stop_probability = torch.zeros_like(target_probabilities[..., 0])
    non_stop_probability = (1.0 - stop_probability).clamp(min=0.0, max=1.0)
    rejection_numerator = (
        (target_probabilities - draft_probabilities).clamp_min(0.0)
        * non_stopping.to(dtype=target_probabilities.dtype)
    ).sum(dim=-1)
    safe_denominator = torch.where(
        non_stop_probability > 0,
        non_stop_probability,
        torch.ones_like(non_stop_probability),
    )
    costs = torch.where(
        non_stop_probability > 0,
        rejection_numerator / safe_denominator,
        torch.zeros_like(rejection_numerator),
    )

    gather_ids = target_ids.long().unsqueeze(-1)
    draft_realized = torch.gather(draft_probabilities, -1, gather_ids).squeeze(-1)
    target_realized = torch.gather(target_probabilities, -1, gather_ids).squeeze(-1)
    denominator = target_realized.clamp_min(torch.finfo(target_realized.dtype).tiny)
    acceptance = torch.minimum(draft_realized / denominator, torch.ones_like(draft_realized))
    return costs, acceptance


def _validated_stopping_token_ids(
    stopping_token_ids: Sequence[int] | torch.Tensor,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    # Validate on the CPU before the host-to-device copy, which avoids GPU
    # reductions and a synchronization to read their results.
    stop_ids = torch.as_tensor(
        stopping_token_ids,
        device="cpu",
        dtype=torch.long,
    ).reshape(-1)
    if stop_ids.numel():
        stop_ids = torch.unique(stop_ids, sorted=True)
        if bool((stop_ids < 0).any()) or bool((stop_ids >= vocab_size).any()):
            raise ValueError(
                f"stopping token IDs must be in [0, {vocab_size}), got {stop_ids.tolist()}"
            )
    return stop_ids.to(device=device)


def _edr_log_normalizer_tile(
    logits_chunk: torch.Tensor, logit_scale: float = 1.0,
) -> torch.Tensor:
    return torch.logsumexp(logits_chunk.float() * logit_scale, dim=-1)


def _edr_merge_log_normalizer_tile(
    normalizers: torch.Tensor,
    logits_chunk: torch.Tensor,
    logit_scale: float = 1.0,
) -> torch.Tensor:
    return torch.logaddexp(normalizers, _edr_log_normalizer_tile(logits_chunk, logit_scale))


def _edr_cost_tile(
    rejection_numerators: torch.Tensor,
    draft_logits_chunk: torch.Tensor,
    draft_log_normalizers: torch.Tensor,
    unique_target_probabilities: torch.Tensor,
    target_inverse_indices: torch.Tensor,
    non_stopping_vocab: torch.Tensor,
    draft_logit_scale: float = 1.0,
) -> torch.Tensor:
    draft_probabilities = torch.exp(
        draft_logits_chunk.float() * draft_logit_scale - draft_log_normalizers.unsqueeze(-1)
    )
    target_probabilities = unique_target_probabilities[target_inverse_indices]
    rejection = (target_probabilities - draft_probabilities).clamp_min(0.0)
    return rejection_numerators + (
        rejection * non_stopping_vocab.to(rejection.dtype)
    ).sum(dim=-1)


def _edr_cost_and_dot_tile(
    rejection_numerators: torch.Tensor,
    cost_probability_dot: torch.Tensor,
    draft_logits_chunk: torch.Tensor,
    draft_log_normalizers: torch.Tensor,
    unique_target_probabilities: torch.Tensor,
    target_inverse_indices: torch.Tensor,
    non_stopping_vocab: torch.Tensor,
    rejection_mask_output: Optional[torch.Tensor] = None,
    draft_logit_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    draft_probabilities = torch.exp(
        draft_logits_chunk.float() * draft_logit_scale - draft_log_normalizers.unsqueeze(-1)
    )
    target_probabilities = unique_target_probabilities[target_inverse_indices]
    active_rejection = (target_probabilities > draft_probabilities) & non_stopping_vocab
    if rejection_mask_output is not None:
        # A byte per vocabulary element avoids reconstructing/gathering the
        # teacher probabilities again in backward. The store is fused with
        # the forward tile; no FP32 probability cache is retained.
        rejection_mask_output.copy_(active_rejection)
    return (
        rejection_numerators
        + (
            (target_probabilities - draft_probabilities).clamp_min(0.0)
            * non_stopping_vocab.to(draft_probabilities.dtype)
        ).sum(dim=-1),
        cost_probability_dot
        - (active_rejection.to(draft_probabilities.dtype) * draft_probabilities).sum(dim=-1),
    )


def _edr_acceptance(
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    target_rows: torch.Tensor,
    target_ids: torch.Tensor,
    draft_log_normalizers: torch.Tensor,
    target_row_log_normalizers: torch.Tensor,
    draft_logit_scale: float = 1.0,
    target_logit_scale: float = 1.0,
) -> torch.Tensor:
    gather_ids = target_ids.unsqueeze(-1)
    draft_realized_logits = torch.gather(draft_logits, -1, gather_ids).squeeze(-1).float()
    draft_realized = torch.exp(draft_realized_logits * draft_logit_scale - draft_log_normalizers)
    target_realized_logits = target_logits[target_rows, target_ids].float()
    target_realized = torch.exp(target_realized_logits * target_logit_scale - target_row_log_normalizers)
    denominator = target_realized.clamp_min(torch.finfo(target_realized.dtype).tiny)
    ratio = draft_realized / denominator
    return torch.minimum(ratio, torch.ones_like(ratio))


def _edr_acceptance_and_scale(
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    target_rows: torch.Tensor,
    target_ids: torch.Tensor,
    draft_log_normalizers: torch.Tensor,
    target_row_log_normalizers: torch.Tensor,
    draft_logit_scale: float = 1.0,
    target_logit_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    gather_ids = target_ids.unsqueeze(-1)
    draft_realized_logits = torch.gather(draft_logits, -1, gather_ids).squeeze(-1).float()
    draft_realized = torch.exp(draft_realized_logits * draft_logit_scale - draft_log_normalizers)
    target_realized_logits = target_logits[target_rows, target_ids].float()
    target_realized = torch.exp(target_realized_logits * target_logit_scale - target_row_log_normalizers)
    denominator = target_realized.clamp_min(torch.finfo(target_realized.dtype).tiny)
    ratio = draft_realized / denominator
    acceptance = torch.minimum(ratio, torch.ones_like(ratio))
    # torch.minimum assigns half the derivative to each equal input. Only the
    # ratio input is differentiable here, so preserve that tie behavior.
    acceptance_slope = torch.where(
        ratio < 1.0,
        torch.ones_like(ratio),
        torch.where(
            ratio > 1.0,
            torch.zeros_like(ratio),
            torch.full_like(ratio, 0.5),
        ),
    )
    return acceptance, acceptance_slope * draft_realized / denominator


def _edr_backward_tile(
    draft_logits_chunk: torch.Tensor,
    draft_log_normalizers: torch.Tensor,
    unique_target_probabilities: Optional[torch.Tensor],
    target_inverse_indices: torch.Tensor,
    target_ids: torch.Tensor,
    vocab_ids: torch.Tensor,
    cost_probability_dot: torch.Tensor,
    non_stop_probabilities: torch.Tensor,
    non_stopping_vocab: torch.Tensor,
    acceptance_softmax_scale: torch.Tensor,
    grad_costs: torch.Tensor,
    grad_acceptance: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    active_rejection: Optional[torch.Tensor] = None,
    draft_logit_scale: float = 1.0,
) -> torch.Tensor:
    draft_probabilities = torch.exp(
        draft_logits_chunk.float() * draft_logit_scale - draft_log_normalizers.unsqueeze(-1)
    )
    if active_rejection is None:
        assert unique_target_probabilities is not None
        target_probabilities = unique_target_probabilities[target_inverse_indices]
        active_rejection = (target_probabilities > draft_probabilities) & non_stopping_vocab
    safe_non_stop = torch.where(
        non_stop_probabilities > 0,
        non_stop_probabilities,
        torch.ones_like(non_stop_probabilities),
    )
    cost_probability_gradient = (
        -active_rejection.to(draft_probabilities.dtype)
        / safe_non_stop.unsqueeze(-1)
    )
    cost_logit_gradient = draft_probabilities * (
        cost_probability_gradient - cost_probability_dot.unsqueeze(-1)
    )
    cost_logit_gradient = torch.where(
        (non_stop_probabilities > 0).unsqueeze(-1),
        cost_logit_gradient,
        torch.zeros_like(cost_logit_gradient),
    )
    realized_mask = target_ids.unsqueeze(-1) == vocab_ids
    acceptance_logit_gradient = acceptance_softmax_scale.unsqueeze(-1) * (
        realized_mask.to(draft_probabilities.dtype) - draft_probabilities
    )
    # Keep the analytical math in FP32, but cast inside the compiled tile so
    # Inductor can emit a draft-dtype output store instead of materializing a
    # full FP32 gradient tile that is immediately downcast by the caller.
    gradient = (draft_logit_scale * (
        grad_costs.unsqueeze(-1) * cost_logit_gradient
        + grad_acceptance.unsqueeze(-1) * acceptance_logit_gradient
    )).to(draft_logits_chunk.dtype)
    if output is None:
        return gradient
    # Store directly into the destination slice, avoiding a temporary BF16
    # tile followed by a second full-vocabulary copy.
    output.copy_(gradient)
    return output


_compiled_edr_log_normalizer_tile = torch.compile(
    _edr_log_normalizer_tile,
    dynamic=True,
    fullgraph=True,
)
_compiled_edr_merge_log_normalizer_tile = torch.compile(
    _edr_merge_log_normalizer_tile,
    dynamic=True,
    fullgraph=True,
)
_compiled_edr_cost_tile = torch.compile(_edr_cost_tile, dynamic=True, fullgraph=True)
_compiled_edr_cost_and_dot_tile = torch.compile(
    _edr_cost_and_dot_tile,
    dynamic=True,
    fullgraph=True,
)
_compiled_edr_acceptance = torch.compile(_edr_acceptance, dynamic=True, fullgraph=True)
_compiled_edr_acceptance_and_scale = torch.compile(
    _edr_acceptance_and_scale,
    dynamic=True,
    fullgraph=True,
)
_compiled_edr_backward_tile = torch.compile(_edr_backward_tile, dynamic=True, fullgraph=True)


def _use_compiled_edr_kernel(tensor: torch.Tensor) -> bool:
    return tensor.device.type == "cuda" and not torch.compiler.is_compiling()


def _unique_target_probability_tile_impl(
    target_logits_chunk: torch.Tensor,
    unique_target_rows: torch.Tensor,
    unique_target_log_normalizers: torch.Tensor,
    target_logit_scale: float = 1.0,
) -> torch.Tensor:
    return torch.exp(
        target_logits_chunk.index_select(0, unique_target_rows).float() * target_logit_scale
        - unique_target_log_normalizers.unsqueeze(-1)
    )


_compiled_unique_target_probability_tile = torch.compile(
    _unique_target_probability_tile_impl,
    dynamic=True,
    fullgraph=True,
)


@torch.library.custom_op("angelspec::edr_unique_target_probability_tile", mutates_args=())
def _materialize_unique_target_probability_tile(
    target_logits_chunk: torch.Tensor,
    unique_target_rows: torch.Tensor,
    unique_target_log_normalizers: torch.Tensor,
    target_logit_scale: float = 1.0,
) -> torch.Tensor:
    """Materialize one probability tile per unique detached target row."""
    if target_logits_chunk.device.type == "cuda":
        return _compiled_unique_target_probability_tile(
            target_logits_chunk,
            unique_target_rows,
            unique_target_log_normalizers,
            target_logit_scale,
        )
    return _unique_target_probability_tile_impl(
        target_logits_chunk,
        unique_target_rows,
        unique_target_log_normalizers,
        target_logit_scale,
    )


@_materialize_unique_target_probability_tile.register_fake
def _materialize_unique_target_probability_tile_fake(
    target_logits_chunk: torch.Tensor,
    unique_target_rows: torch.Tensor,
    unique_target_log_normalizers: torch.Tensor,
    target_logit_scale: float = 1.0,
) -> torch.Tensor:
    del unique_target_log_normalizers, target_logit_scale
    return target_logits_chunk.new_empty(
        (unique_target_rows.shape[0], target_logits_chunk.shape[1]),
        dtype=torch.float32,
    )


def _streaming_log_normalizers(
    logits: torch.Tensor,
    vocab_chunk_size: int,
    logit_scale: float = 1.0,
) -> torch.Tensor:
    """Compute FP32 logsumexp without a full-vocabulary FP32 temporary."""
    if logits.ndim < 1 or logits.shape[-1] < 1:
        raise ValueError("logits must have a non-empty vocabulary dimension")
    if vocab_chunk_size < 1:
        raise ValueError(f"vocab_chunk_size must be >= 1, got {vocab_chunk_size}")

    normalizers = None
    use_compiled = _use_compiled_edr_kernel(logits)
    for chunk_start in range(0, logits.shape[-1], vocab_chunk_size):
        chunk = logits[..., chunk_start : chunk_start + vocab_chunk_size]
        if normalizers is None:
            normalizers = (
                _compiled_edr_log_normalizer_tile(chunk, logit_scale)
                if use_compiled
                else _edr_log_normalizer_tile(chunk, logit_scale)
            )
        else:
            normalizers = (
                _compiled_edr_merge_log_normalizer_tile(normalizers, chunk, logit_scale)
                if use_compiled
                else _edr_merge_log_normalizer_tile(normalizers, chunk, logit_scale)
            )
    assert normalizers is not None
    return normalizers


def _target_stop_probabilities(
    target_logits: torch.Tensor,
    target_log_normalizers: torch.Tensor,
    stopping_token_ids: torch.Tensor,
    logit_scale: float = 1.0,
) -> torch.Tensor:
    if stopping_token_ids.numel() == 0:
        return torch.zeros_like(target_log_normalizers, dtype=torch.float32)
    stop_logits = target_logits.index_select(-1, stopping_token_ids).float() * logit_scale
    stop_log_mass = torch.logsumexp(stop_logits, dim=-1) - target_log_normalizers.float()
    return torch.exp(stop_log_mass).clamp(min=0.0, max=1.0)


def prepare_edr_target_distribution(
    target_logits: torch.Tensor,
    vocab_chunk_size: int,
    *,
    stopping_token_ids: Sequence[int] | torch.Tensor = (),
    temperature: float = 1.0,
    top_k: int = -1,
    top_p: float = 1.0,
    _legacy_arithmetic: bool = False,
) -> EDRTargetDistribution:
    """Prepare the actual target sampling distribution and stopping mass.

    Without top-k/top-p filtering and with T > 0, the logits are used without a copy.
    Filtering runs once per unique target position in bounded row tiles, not
    for every overlapping anchor or backward. Retained logits stay in the
    original dtype; temperature is applied by the streamed FP32 reductions.
    """
    validate_sampling_parameters(temperature, top_k, top_p)
    if target_logits.ndim != 2:
        raise ValueError("EDR target logits must have shape [positions, vocabulary]")
    if target_logits.shape[-1] < 1:
        raise ValueError("EDR target logits must have a nonempty vocabulary dimension")
    if vocab_chunk_size < 1:
        raise ValueError(f"vocab_chunk_size must be >= 1, got {vocab_chunk_size}")
    detached_logits = target_logits.detach()
    logit_scale = 1.0 if temperature == 0 else 1.0 / temperature
    if temperature == 0 or top_k != -1 or top_p != 1.0:
        filtered_logits = torch.empty_like(detached_logits)
        # Sorting top-p needs full-vocabulary rows, but never a full-position
        # FP32 probability/sort tensor. Bound transient memory for large vocabularies.
        row_chunk_size = max(1, min(128, 4_194_304 // detached_logits.shape[-1]))
        with torch.no_grad():
            for start in range(0, detached_logits.shape[0], row_chunk_size):
                source = detached_logits[start:start + row_chunk_size]
                processed = sampling_logits(
                    source, temperature=temperature, top_k=top_k, top_p=top_p,
                    _legacy_arithmetic=_legacy_arithmetic,
                )
                destination = filtered_logits[start:start + row_chunk_size]
                if temperature == 0:
                    destination.copy_(processed)
                else:
                    destination.copy_(source)
                    destination.masked_fill_(torch.isneginf(processed), -torch.inf)
        detached_logits = filtered_logits
    stop_ids = _validated_stopping_token_ids(
        stopping_token_ids,
        detached_logits.shape[-1],
        detached_logits.device,
    )
    log_normalizers = _streaming_log_normalizers(
        detached_logits,
        vocab_chunk_size,
        logit_scale,
    ).detach()
    stop_probabilities = _target_stop_probabilities(
        detached_logits,
        log_normalizers,
        stop_ids,
        logit_scale,
    ).detach()

    return EDRTargetDistribution(
        logits=detached_logits,
        log_normalizers=log_normalizers,
        stop_probabilities=stop_probabilities,
        stopping_token_ids=stop_ids,
        logit_scale=logit_scale,
    )


@torch.no_grad()
def greedy_edr_distribution_statistics(
    draft_logits: torch.Tensor,
    target_distribution: EDRTargetDistribution | EDRSparseTargetDistribution,
    target_probability_indices: torch.Tensor,
    target_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact EDR statistics for an argmax/one-hot draft distribution.

    Conditional rejection is one when the greedy token is stopping; otherwise
    it is ``1 - p(y) / p(non-stop)``. Realized-token acceptance is one exactly
    when the target-path token is ``y`` and zero otherwise. This avoids
    materializing a vocabulary-sized one-hot tensor and is the temperature-zero counterpart of
    :func:`streaming_edr_distribution_statistics`.
    """
    from angelspec.models.ops.edr_sparse_target import EDRSparseTargetDistribution

    if isinstance(target_distribution, EDRSparseTargetDistribution):
        from angelspec.models.ops.edr_sparse import greedy_sparse_edr_distribution_statistics

        return greedy_sparse_edr_distribution_statistics(
            draft_logits, target_distribution, target_probability_indices, target_ids,
        )
    if draft_logits.ndim < 2:
        raise ValueError("draft logits must include position and vocabulary dimensions")
    if target_distribution.logits.ndim != 2:
        raise ValueError("EDR target logits must have shape [positions, vocabulary]")
    if draft_logits.shape[-1] != target_distribution.logits.shape[-1]:
        raise ValueError("EDR target and draft vocabulary sizes must match")
    if target_probability_indices.shape != draft_logits.shape[:-1]:
        raise ValueError("target probability indices must match draft positions")
    if target_ids.shape != draft_logits.shape[:-1]:
        raise ValueError("target_ids must match draft positions")
    if target_distribution.log_normalizers.shape != target_distribution.logits.shape[:1]:
        raise ValueError("target log normalizers must have one value per target position")
    if target_distribution.stop_probabilities.shape != target_distribution.logits.shape[:1]:
        raise ValueError("target stop probabilities must have one value per target position")
    tensors = (
        target_distribution.logits,
        target_distribution.log_normalizers,
        target_distribution.stop_probabilities,
        target_distribution.stopping_token_ids,
        target_probability_indices,
        target_ids,
    )
    if any(tensor.device != draft_logits.device for tensor in tensors):
        raise ValueError("EDR logits, indices, and token IDs must be on the same device")

    probability_indices = target_probability_indices.long()
    if probability_indices.numel():
        target_row_count = target_distribution.logits.shape[0]
        if bool((probability_indices < 0).any()) or bool(
            (probability_indices >= target_row_count).any()
        ):
            raise ValueError("target probability indices are out of bounds")

    greedy_ids = draft_logits.argmax(dim=-1)
    flat_rows = probability_indices.reshape(-1)
    flat_greedy_ids = greedy_ids.reshape(-1)
    selected_target_logits = target_distribution.logits[
        flat_rows,
        flat_greedy_ids,
    ].float()
    selected_target_normalizers = target_distribution.log_normalizers.index_select(
        0,
        flat_rows,
    )
    target_probability = torch.exp(
        selected_target_logits * target_distribution.logit_scale - selected_target_normalizers
    ).view_as(
        greedy_ids
    )
    selected_stop_probability = target_distribution.stop_probabilities.index_select(
        0,
        flat_rows,
    ).view_as(greedy_ids)
    non_stop_probability = (1.0 - selected_stop_probability).clamp(min=0.0, max=1.0)
    if target_distribution.stopping_token_ids.numel():
        greedy_is_stopping = greedy_ids.unsqueeze(-1).eq(
            target_distribution.stopping_token_ids
        ).any(dim=-1)
    else:
        greedy_is_stopping = torch.zeros_like(greedy_ids, dtype=torch.bool)
    safe_non_stop = torch.where(
        non_stop_probability > 0,
        non_stop_probability,
        torch.ones_like(non_stop_probability),
    )
    non_stopping_cost = (1.0 - target_probability / safe_non_stop).clamp(
        min=0.0,
        max=1.0,
    )
    costs = torch.where(
        non_stop_probability > 0,
        torch.where(greedy_is_stopping, torch.ones_like(non_stopping_cost), non_stopping_cost),
        torch.zeros_like(non_stopping_cost),
    )
    acceptance = greedy_ids.eq(target_ids.long()).to(dtype=torch.float32)
    return costs, acceptance


@torch.no_grad()
def _streaming_edr_statistics_forward(
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    target_log_normalizers: torch.Tensor,
    target_stop_probabilities: torch.Tensor,
    non_stopping_vocab_mask: torch.Tensor,
    target_probability_indices: torch.Tensor,
    target_ids: torch.Tensor,
    vocab_chunk_size: int,
    *,
    compute_backward_stats: bool,
    cache_rejection_mask: bool = False,
    draft_logit_scale: float = 1.0,
    target_logit_scale: float = 1.0,
):
    """Compute streamed statistics and optionally the compact backward state."""
    draft_log_normalizers = _streaming_log_normalizers(
        draft_logits,
        vocab_chunk_size,
        draft_logit_scale,
    )
    unique_target_rows, target_inverse_indices = torch.unique(
        target_probability_indices.reshape(-1),
        sorted=True,
        return_inverse=True,
    )
    target_inverse_indices = target_inverse_indices.view_as(target_probability_indices)
    unique_target_log_normalizers = target_log_normalizers.index_select(
        0,
        unique_target_rows,
    )
    unique_target_stop_probabilities = target_stop_probabilities.index_select(
        0,
        unique_target_rows,
    )
    non_stop_probabilities = (
        1.0 - unique_target_stop_probabilities[target_inverse_indices]
    ).clamp(min=0.0, max=1.0)

    rejection_numerators = torch.zeros_like(draft_log_normalizers)
    cost_probability_dot = (
        torch.zeros_like(draft_log_normalizers) if compute_backward_stats else None
    )
    rejection_mask = (
        torch.empty_like(draft_logits, dtype=torch.bool)
        if compute_backward_stats and cache_rejection_mask else None
    )
    vocab_size = draft_logits.shape[-1]
    use_compiled = _use_compiled_edr_kernel(draft_logits)
    for chunk_start in range(0, vocab_size, vocab_chunk_size):
        chunk_end = min(chunk_start + vocab_chunk_size, vocab_size)
        draft_logits_chunk = draft_logits[..., chunk_start:chunk_end]
        unique_target_probabilities = _materialize_unique_target_probability_tile(
            target_logits[:, chunk_start:chunk_end],
            unique_target_rows,
            unique_target_log_normalizers,
            target_logit_scale,
        )
        non_stopping_vocab = non_stopping_vocab_mask[chunk_start:chunk_end]
        if compute_backward_stats:
            assert cost_probability_dot is not None
            cost_kernel = (
                _compiled_edr_cost_and_dot_tile if use_compiled else _edr_cost_and_dot_tile
            )
            rejection_numerators, cost_probability_dot = cost_kernel(
                rejection_numerators,
                cost_probability_dot,
                draft_logits_chunk,
                draft_log_normalizers,
                unique_target_probabilities,
                target_inverse_indices,
                non_stopping_vocab,
                (
                    rejection_mask[..., chunk_start:chunk_end]
                    if rejection_mask is not None else None
                ),
                draft_logit_scale,
            )
        else:
            cost_kernel = _compiled_edr_cost_tile if use_compiled else _edr_cost_tile
            rejection_numerators = cost_kernel(
                rejection_numerators,
                draft_logits_chunk,
                draft_log_normalizers,
                unique_target_probabilities,
                target_inverse_indices,
                non_stopping_vocab,
                draft_logit_scale,
            )

    positive_non_stop = non_stop_probabilities > 0
    safe_non_stop = torch.where(
        positive_non_stop,
        non_stop_probabilities,
        torch.ones_like(non_stop_probabilities),
    )
    costs = torch.where(
        positive_non_stop,
        rejection_numerators / safe_non_stop,
        torch.zeros_like(rejection_numerators),
    )
    if cost_probability_dot is not None:
        cost_probability_dot = torch.where(
            positive_non_stop,
            cost_probability_dot / safe_non_stop,
            torch.zeros_like(cost_probability_dot),
        )

    target_rows = unique_target_rows[target_inverse_indices]
    target_row_log_normalizers = unique_target_log_normalizers[target_inverse_indices]
    if compute_backward_stats:
        acceptance_kernel = (
            _compiled_edr_acceptance_and_scale if use_compiled else _edr_acceptance_and_scale
        )
        acceptance, acceptance_softmax_scale = acceptance_kernel(
            draft_logits,
            target_logits,
            target_rows,
            target_ids,
            draft_log_normalizers,
            target_row_log_normalizers,
            draft_logit_scale,
            target_logit_scale,
        )
    else:
        acceptance_kernel = _compiled_edr_acceptance if use_compiled else _edr_acceptance
        acceptance = acceptance_kernel(
            draft_logits,
            target_logits,
            target_rows,
            target_ids,
            draft_log_normalizers,
            target_row_log_normalizers,
            draft_logit_scale,
            target_logit_scale,
        )
        acceptance_softmax_scale = None

    return (
        costs,
        acceptance,
        draft_log_normalizers,
        unique_target_rows,
        target_inverse_indices,
        unique_target_log_normalizers,
        non_stop_probabilities,
        cost_probability_dot,
        acceptance_softmax_scale,
        rejection_mask,
    )


class _StreamingEDRStatistics(torch.autograd.Function):
    """Exact streamed conditional cost/acceptance with analytical backward."""

    @staticmethod
    def forward(
        ctx,
        draft_logits: torch.Tensor,
        target_logits: torch.Tensor,
        target_log_normalizers: torch.Tensor,
        target_stop_probabilities: torch.Tensor,
        non_stopping_vocab_mask: torch.Tensor,
        target_probability_indices: torch.Tensor,
        target_ids: torch.Tensor,
        vocab_chunk_size: int,
        cache_rejection_mask: bool,
        draft_logit_scale: float,
        target_logit_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (
            costs,
            acceptance,
            draft_log_normalizers,
            unique_target_rows,
            target_inverse_indices,
            unique_target_log_normalizers,
            non_stop_probabilities,
            cost_probability_dot,
            acceptance_softmax_scale,
            rejection_mask,
        ) = _streaming_edr_statistics_forward(
            draft_logits,
            target_logits,
            target_log_normalizers,
            target_stop_probabilities,
            non_stopping_vocab_mask,
            target_probability_indices,
            target_ids,
            vocab_chunk_size,
            compute_backward_stats=True,
            cache_rejection_mask=cache_rejection_mask,
            draft_logit_scale=draft_logit_scale,
            target_logit_scale=target_logit_scale,
        )
        assert cost_probability_dot is not None
        assert acceptance_softmax_scale is not None

        ctx.vocab_chunk_size = vocab_chunk_size
        ctx.draft_logit_scale = draft_logit_scale
        ctx.target_logit_scale = target_logit_scale
        ctx.save_for_backward(
            draft_logits,
            target_logits if rejection_mask is None else None,
            unique_target_rows if rejection_mask is None else None,
            target_inverse_indices,
            target_ids,
            non_stopping_vocab_mask,
            draft_log_normalizers,
            unique_target_log_normalizers if rejection_mask is None else None,
            non_stop_probabilities,
            cost_probability_dot,
            acceptance_softmax_scale,
            rejection_mask,
        )
        ctx.set_materialize_grads(False)
        return costs, acceptance

    @staticmethod
    def backward(ctx, grad_costs, grad_acceptance):
        if grad_costs is None and grad_acceptance is None:
            return (None,) * 11

        (
            draft_logits,
            target_logits,
            unique_target_rows,
            target_inverse_indices,
            target_ids,
            non_stopping_vocab_mask,
            draft_log_normalizers,
            unique_target_log_normalizers,
            non_stop_probabilities,
            cost_probability_dot,
            acceptance_softmax_scale,
            rejection_mask,
        ) = ctx.saved_tensors
        if grad_costs is None:
            grad_costs = torch.zeros_like(draft_log_normalizers)
        if grad_acceptance is None:
            grad_acceptance = torch.zeros_like(draft_log_normalizers)

        grad_logits = torch.empty_like(draft_logits)
        vocab_size = draft_logits.shape[-1]
        vocab_ids = torch.arange(vocab_size, device=target_ids.device)
        backward_kernel = (
            _compiled_edr_backward_tile
            if _use_compiled_edr_kernel(draft_logits)
            else _edr_backward_tile
        )
        for chunk_start in range(0, vocab_size, ctx.vocab_chunk_size):
            chunk_end = min(chunk_start + ctx.vocab_chunk_size, vocab_size)
            unique_target_probabilities = None
            if rejection_mask is None:
                unique_target_probabilities = _materialize_unique_target_probability_tile(
                    target_logits[:, chunk_start:chunk_end],
                    unique_target_rows,
                    unique_target_log_normalizers,
                    ctx.target_logit_scale,
                )
            non_stopping_vocab = non_stopping_vocab_mask[chunk_start:chunk_end]
            backward_kernel(
                draft_logits[..., chunk_start:chunk_end],
                draft_log_normalizers,
                unique_target_probabilities,
                target_inverse_indices,
                target_ids,
                vocab_ids[chunk_start:chunk_end],
                cost_probability_dot,
                non_stop_probabilities,
                non_stopping_vocab,
                acceptance_softmax_scale,
                grad_costs,
                grad_acceptance,
                grad_logits[..., chunk_start:chunk_end],
                (
                    rejection_mask[..., chunk_start:chunk_end]
                    if rejection_mask is not None else None
                ),
                ctx.draft_logit_scale,
            )

        return grad_logits, None, None, None, None, None, None, None, None, None, None


def streaming_edr_distribution_statistics(
    draft_logits: torch.Tensor,
    target_distribution: EDRTargetDistribution | EDRSparseTargetDistribution,
    target_probability_indices: torch.Tensor,
    target_ids: torch.Tensor,
    vocab_chunk_size: int,
    *,
    cache_rejection_mask: bool = False,
    draft_temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact EDR statistics without materializing full FP32 probabilities.

    ``target_probability_indices`` maps every draft state to one row of the
    detached target distribution. Repeated rows are exponentiated once per
    vocabulary tile. CUDA uses compiled tile reducers; the custom backward
    saves BF16/FP16 logits and compact reductions, then reconstructs one tile
    at a time. ``cache_rejection_mask`` optionally retains one boolean per
    draft logit so backward can skip the teacher probability reconstruction;
    it applies only when gradients are required. Compact target inputs reduce
    costs on their support, use sparse backward corrections, and ignore
    ``cache_rejection_mask``. Both paths normalize the draft over the complete
    vocabulary.
    """
    validate_sampling_parameters(draft_temperature, -1, 1.0)
    if draft_temperature == 0:
        if torch.is_grad_enabled() and draft_logits.requires_grad:
            raise ValueError("Greedy EDR statistics have no differentiable draft distribution")
        return greedy_edr_distribution_statistics(
            draft_logits, target_distribution, target_probability_indices, target_ids,
        )
    from angelspec.models.ops.edr_sparse_target import EDRSparseTargetDistribution

    if isinstance(target_distribution, EDRSparseTargetDistribution):
        from angelspec.models.ops.edr_sparse import sparse_edr_distribution_statistics

        # The sparse path stores O(N*K) derivative coefficients in place of a
        # dense rejection mask.
        return sparse_edr_distribution_statistics(
            draft_logits, target_distribution, target_probability_indices, target_ids,
            vocab_chunk_size, draft_temperature=draft_temperature,
        )
    draft_logit_scale = 1.0 / draft_temperature
    if draft_logits.ndim < 2:
        raise ValueError("draft logits must include position and vocabulary dimensions")
    if target_distribution.logits.ndim != 2:
        raise ValueError("EDR target logits must have shape [positions, vocabulary]")
    if draft_logits.shape[-1] != target_distribution.logits.shape[-1]:
        raise ValueError("EDR target and draft vocabulary sizes must match")
    if target_probability_indices.shape != draft_logits.shape[:-1]:
        raise ValueError("target probability indices must match draft positions")
    if target_ids.shape != draft_logits.shape[:-1]:
        raise ValueError("target_ids must match draft positions")
    if target_distribution.log_normalizers.shape != target_distribution.logits.shape[:1]:
        raise ValueError("target log normalizers must have one value per target position")
    if target_distribution.stop_probabilities.shape != target_distribution.logits.shape[:1]:
        raise ValueError("target stop probabilities must have one value per target position")
    if vocab_chunk_size < 1:
        raise ValueError(f"vocab_chunk_size must be >= 1, got {vocab_chunk_size}")
    tensors = (
        target_distribution.logits,
        target_distribution.log_normalizers,
        target_distribution.stop_probabilities,
        target_distribution.stopping_token_ids,
        target_probability_indices,
        target_ids,
    )
    if any(tensor.device != draft_logits.device for tensor in tensors):
        raise ValueError("EDR logits, indices, and token IDs must be on the same device")

    target_probability_indices = target_probability_indices.long()
    target_ids = target_ids.long()
    if not torch.is_grad_enabled() or not draft_logits.requires_grad:
        costs, acceptance, *_ = _streaming_edr_statistics_forward(
            draft_logits,
            target_distribution.logits,
            target_distribution.log_normalizers,
            target_distribution.stop_probabilities,
            target_distribution.non_stopping_vocab,
            target_probability_indices,
            target_ids,
            int(vocab_chunk_size),
            compute_backward_stats=False,
            draft_logit_scale=draft_logit_scale,
            target_logit_scale=target_distribution.logit_scale,
        )
        return costs, acceptance

    return _StreamingEDRStatistics.apply(
        draft_logits,
        target_distribution.logits,
        target_distribution.log_normalizers,
        target_distribution.stop_probabilities,
        target_distribution.non_stopping_vocab,
        target_probability_indices,
        target_ids,
        int(vocab_chunk_size),
        bool(cache_rejection_mask),
        draft_logit_scale,
        target_distribution.logit_scale,
    )


def _learned_state_mask(
    length: int,
    num_proposals: int,
    device: torch.device,
) -> torch.Tensor:
    remaining = length - torch.arange(length, device=device)
    columns = torch.arange(num_proposals, device=device)
    return columns.unsqueeze(0) < remaining.unsqueeze(1)


@numba.njit(cache=True, nogil=True, inline="always")
def _exact_edr_numpy_recurrence_into(
    costs: np.ndarray,
    acceptance: np.ndarray,
    num_proposals: int,
    round_values: np.ndarray,
    learned_values: np.ndarray,
    advantages: np.ndarray,
    occupancies: np.ndarray,
    conditional_survivals: np.ndarray,
    round_starts: np.ndarray,
) -> None:
    """Fill one horizon's preallocated Bellman and occupancy arrays."""
    length, learned_width = costs.shape
    one32 = np.float32(1.0)

    for prefix in range(length - 1, -1, -1):
        exhaustion_position = prefix + num_proposals + 1
        continuation = (
            one32 + round_values[exhaustion_position]
            if exhaustion_position <= length
            else np.float32(0.0)
        )

        max_learned = min(learned_width, length - prefix)
        for column in range(max_learned - 1, -1, -1):
            position = prefix + 1 + column
            rejection_value = round_values[position]
            advantages[prefix, column] = continuation - rejection_value
            accept = acceptance[prefix, column]
            state_value = (
                costs[prefix, column] + accept * continuation + (one32 - accept) * rejection_value
            )
            learned_values[prefix, column] = state_value
            continuation = state_value

        round_values[prefix] = continuation

    # Propagate probabilities in FP64; small round-start probabilities feed
    # the canceled PPS estimator.
    round_starts[0] = 1.0
    one64 = np.float64(1.0)
    for prefix in range(length):
        mass = round_starts[prefix]
        conditional_survival = one64
        max_learned = min(learned_width, length - prefix)
        for column in range(max_learned):
            position = prefix + 1 + column
            occupancies[prefix, column] = mass
            conditional_survivals[prefix, column] = conditional_survival
            accept = np.float64(acceptance[prefix, column])
            round_starts[position] += mass * (one64 - accept)
            mass *= accept
            conditional_survival *= accept

        exhaustion_position = prefix + num_proposals + 1
        if exhaustion_position <= length:
            round_starts[exhaustion_position] += mass


@numba.njit(cache=True, nogil=True)
def _exact_edr_numpy_recurrence(
    costs: np.ndarray,
    acceptance: np.ndarray,
    num_proposals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run one compact EDR recurrence in compiled, GIL-free native code."""
    length = costs.shape[0]
    round_values = np.zeros(length + 1, dtype=np.float32)
    learned_values = np.zeros_like(costs, dtype=np.float32)
    advantages = np.zeros_like(costs, dtype=np.float32)
    occupancies = np.zeros_like(costs, dtype=np.float64)
    conditional_survivals = np.zeros_like(costs, dtype=np.float64)
    round_starts = np.zeros(length + 1, dtype=np.float64)
    _exact_edr_numpy_recurrence_into(
        costs,
        acceptance,
        num_proposals,
        round_values,
        learned_values,
        advantages,
        occupancies,
        conditional_survivals,
        round_starts,
    )
    return (
        round_values,
        learned_values,
        advantages,
        occupancies,
        conditional_survivals,
        round_starts,
    )


@numba.njit(cache=True, nogil=True, parallel=True)
def _exact_edr_numpy_recurrence_batch(
    costs: np.ndarray,
    acceptance: np.ndarray,
    state_offsets: np.ndarray,
    num_proposals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run independent flattened horizons concurrently with native CPU threads."""
    num_horizons = state_offsets.shape[0] - 1
    num_states, learned_width = costs.shape
    round_value_count = num_states + num_horizons
    round_values = np.zeros(round_value_count, dtype=np.float32)
    learned_values = np.zeros_like(costs, dtype=np.float32)
    advantages = np.zeros_like(costs, dtype=np.float32)
    occupancies = np.zeros_like(costs, dtype=np.float64)
    conditional_survivals = np.zeros_like(costs, dtype=np.float64)
    round_starts = np.zeros(round_value_count, dtype=np.float64)

    for horizon_index in numba.prange(num_horizons):
        state_start = state_offsets[horizon_index]
        state_end = state_offsets[horizon_index + 1]
        round_start = state_start + horizon_index
        round_end = round_start + (state_end - state_start) + 1
        _exact_edr_numpy_recurrence_into(
            costs[state_start:state_end, :learned_width],
            acceptance[state_start:state_end, :learned_width],
            num_proposals,
            round_values[round_start:round_end],
            learned_values[state_start:state_end, :learned_width],
            advantages[state_start:state_end, :learned_width],
            occupancies[state_start:state_end, :learned_width],
            conditional_survivals[state_start:state_end, :learned_width],
            round_starts[round_start:round_end],
        )

    return (
        round_values,
        learned_values,
        advantages,
        occupancies,
        conditional_survivals,
        round_starts,
    )


def _validate_exact_edr_shapes(
    costs: torch.Tensor,
    acceptance: torch.Tensor,
    num_proposals: int,
) -> int:
    if costs.ndim != 2 or acceptance.shape != costs.shape:
        raise ValueError("costs and acceptance must be rank-2 tensors with equal shape")
    if costs.shape[1] != num_proposals:
        raise ValueError(
            "learned-state width must equal num_proposals "
            f"({num_proposals}), got {costs.shape[1]}"
        )
    return int(costs.shape[0])


def _validate_exact_edr_values(
    costs: torch.Tensor,
    acceptance: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    if not mask.numel():
        return
    if (
        costs.device.type != "cpu"
        or acceptance.device.type != "cpu"
        or mask.device.type != "cpu"
    ):
        raise ValueError("EDR dynamic-program validation requires CPU tensors")
    mask_array = mask.numpy()
    if not mask_array.any():
        return
    valid_costs = costs.numpy()[mask_array]
    valid_acceptance = acceptance.numpy()[mask_array]
    if not np.isfinite(valid_costs).all() or not np.isfinite(valid_acceptance).all():
        raise ValueError("valid EDR costs and acceptance probabilities must be finite")
    if (valid_acceptance < 0).any() or (valid_acceptance > 1).any():
        raise ValueError("EDR acceptance probabilities must lie in [0, 1]")
    if (valid_costs < 0).any():
        raise ValueError("EDR costs must be nonnegative")


def _edr_dynamic_program_from_numpy(
    recurrence: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
    *,
    learned_mask: torch.Tensor,
    result_device: torch.device,
) -> EDRDynamicProgram:
    (
        round_values,
        learned_values,
        advantages,
        occupancies,
        conditional_survivals,
        round_starts,
    ) = (torch.from_numpy(array) for array in recurrence)

    expected_passes = 1.0 + round_values[0]
    if not torch.isfinite(expected_passes) or bool(expected_passes < 1.0):
        raise RuntimeError("invalid EDR dynamic-program result")

    return EDRDynamicProgram(
        expected_passes=expected_passes.detach().to(result_device),
        round_values=round_values.detach().to(result_device),
        learned_values=learned_values.detach().to(result_device),
        continuation_advantages=advantages.detach().to(result_device),
        occupancies=occupancies.detach().to(result_device),
        conditional_survivals=conditional_survivals.detach().to(result_device),
        round_start_probabilities=round_starts.detach().to(result_device),
        learned_mask=learned_mask.to(result_device),
    )


@torch.no_grad()
def exact_edr_dynamic_program(
    costs: torch.Tensor,
    acceptance: torch.Tensor,
    *,
    num_proposals: int,
) -> EDRDynamicProgram:
    """Evaluate the exact conditional EDR Bellman and occupancy recurrences.

    ``costs`` and ``acceptance`` contain all ``B`` learned proposal positions,
    including ``q[n,n+1]``. The input anchor is already committed and is not
    part of these tensors. The analytical
    ``n+B+1`` transition is the target bonus after all ``B`` proposals survive.
    """
    result_device = costs.device
    if num_proposals < 1:
        raise ValueError(f"num_proposals must be >= 1, got {num_proposals}")
    _validate_exact_edr_shapes(costs, acceptance, num_proposals)
    # The recurrence is scalar, branch-heavy, and only O(L*B) in storage. Running
    # it on CPU avoids launching one tiny CUDA kernel per state. Numba compiles
    # the NumPy recurrence to native code and releases the GIL.
    # Full-vocabulary probabilities remain on GPU; only compact [L, B]
    # statistics cross devices.
    costs = costs.detach().float().to(device="cpu").contiguous()
    acceptance = acceptance.detach().float().to(device="cpu").contiguous()
    length = costs.shape[0]
    mask = _learned_state_mask(length, num_proposals, costs.device)
    _validate_exact_edr_values(costs, acceptance, mask)

    recurrence = _exact_edr_numpy_recurrence(
        costs.numpy(),
        acceptance.numpy(),
        num_proposals,
    )
    return _edr_dynamic_program_from_numpy(
        recurrence,
        learned_mask=mask,
        result_device=result_device,
    )


@torch.no_grad()
def exact_edr_dynamic_programs(
    statistics: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    num_proposals: int,
    max_workers: int = 1,
    return_on_cpu: bool = False,
) -> list[EDRDynamicProgram]:
    """Evaluate independent horizons with one CPU transfer and native parallelism.

    Results retain input order. ``max_workers=1`` keeps the compiled batch
    serial; larger values parallelize across horizons and are capped by the
    horizon count and Numba's configured thread limit. ``return_on_cpu`` avoids
    copying recurrence tensors back to the input GPU when the caller only needs
    host-side evaluation metrics.
    """
    if num_proposals < 1:
        raise ValueError(f"num_proposals must be >= 1, got {num_proposals}")
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}")
    statistics = tuple(statistics)
    if not statistics:
        return []

    statistics_device = statistics[0][0].device
    result_device = torch.device("cpu") if return_on_cpu else statistics_device
    lengths: list[int] = []
    costs_to_join: list[torch.Tensor] = []
    acceptance_to_join: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    cpu_device = torch.device("cpu")
    for costs, acceptance in statistics:
        length = _validate_exact_edr_shapes(costs, acceptance, num_proposals)
        if costs.device != statistics_device or acceptance.device != statistics_device:
            raise ValueError("batched EDR statistics must all be on the same device")
        lengths.append(length)
        costs_to_join.append(costs.detach().float())
        acceptance_to_join.append(acceptance.detach().float())
        masks.append(_learned_state_mask(length, num_proposals, cpu_device))

    # The tensors are compact [sum(L), B] statistics. Joining before the copy
    # reduces per-horizon CUDA synchronization and transfer overhead.
    flat_costs = torch.cat(costs_to_join, dim=0).to(device="cpu").contiguous()
    flat_acceptance = (
        torch.cat(acceptance_to_join, dim=0).to(device="cpu").contiguous()
    )
    flat_mask = torch.cat(masks, dim=0)
    _validate_exact_edr_values(flat_costs, flat_acceptance, flat_mask)

    state_offsets = np.empty(len(lengths) + 1, dtype=np.int64)
    state_offsets[0] = 0
    np.cumsum(np.asarray(lengths, dtype=np.int64), out=state_offsets[1:])

    worker_count = min(
        int(max_workers),
        len(statistics),
        int(numba.config.NUMBA_NUM_THREADS),
    )
    previous_thread_count = numba.get_num_threads()
    numba.set_num_threads(worker_count)
    try:
        flat_recurrence = _exact_edr_numpy_recurrence_batch(
            flat_costs.numpy(),
            flat_acceptance.numpy(),
            state_offsets,
            num_proposals,
        )
    finally:
        numba.set_num_threads(previous_thread_count)

    expected_passes = np.empty(len(lengths), dtype=np.float32)
    for horizon_index, length in enumerate(lengths):
        state_start = int(state_offsets[horizon_index])
        round_start = state_start + horizon_index
        expected_passes[horizon_index] = (
            np.float32(1.0) + flat_recurrence[0][round_start]
        )
    if not np.isfinite(expected_passes).all() or (expected_passes < 1.0).any():
        raise RuntimeError("invalid EDR dynamic-program result")

    # Materialize each flattened field once on the requested result device, then
    # return per-horizon views. Training receives compact GPU coefficients;
    # evaluation can retain the native NumPy results on CPU.
    device_recurrence = tuple(
        torch.from_numpy(array).to(result_device) for array in flat_recurrence
    )
    device_expected_passes = torch.from_numpy(expected_passes).to(result_device)
    device_mask = flat_mask.to(result_device)

    results: list[EDRDynamicProgram] = []
    for horizon_index, length in enumerate(lengths):
        state_start = int(state_offsets[horizon_index])
        state_end = int(state_offsets[horizon_index + 1])
        round_start = state_start + horizon_index
        round_end = round_start + length + 1
        results.append(
            EDRDynamicProgram(
                expected_passes=device_expected_passes[horizon_index],
                round_values=device_recurrence[0][round_start:round_end],
                learned_values=device_recurrence[1][state_start:state_end],
                continuation_advantages=device_recurrence[2][state_start:state_end],
                occupancies=device_recurrence[3][state_start:state_end],
                conditional_survivals=device_recurrence[4][state_start:state_end],
                round_start_probabilities=device_recurrence[5][round_start:round_end],
                learned_mask=device_mask[state_start:state_end],
            )
        )
    return results


def edr_surrogate_sum(
    costs: torch.Tensor,
    acceptance: torch.Tensor,
    dynamic_program: EDRDynamicProgram,
) -> torch.Tensor:
    """Build a scalar whose gradient is exactly the paper's EDR gradient, Eq. (14).

    Occupancy and continuation values are explicit stop-gradient coefficients.
    The scalar value itself is not the expected target-pass count.
    """
    if costs.shape != acceptance.shape or costs.shape != dynamic_program.occupancies.shape:
        raise ValueError("surrogate tensors and dynamic-program tensors must have equal shapes")
    weights = dynamic_program.occupancies.detach().float()
    advantages = dynamic_program.continuation_advantages.detach()
    mask = dynamic_program.learned_mask
    terms = weights * (costs.float() + acceptance.float() * advantages)
    return torch.where(mask, terms, torch.zeros_like(terms)).sum()


@torch.no_grad()
def _edr_anchor_sampling_design(
    round_start_probabilities: torch.Tensor,
    num_anchors: int,
) -> _EDRAnchorSamplingDesign:
    """Compute the paper's capped-proportional anchor inclusion probabilities, Eqs. (27)-(28).

    For ``rho=num_anchors < L``, this solves

    ``sum_n min(1, lambda * omega[n, n+1]) = rho``

    with ``omega[n, n+1]`` equal to the supplied round-start probability. If
    fewer than ``rho`` starts have nonzero occupancy, all positive-occupancy
    starts are included with probability one; the remaining fixed query slots
    can be masked because their contribution to Eq. (14) is exactly zero.
    """
    if round_start_probabilities.ndim != 1:
        raise ValueError("round_start_probabilities must be rank 1")
    if num_anchors < 1:
        raise ValueError(f"num_anchors must be >= 1, got {num_anchors}")
    if not round_start_probabilities.is_floating_point():
        raise TypeError("round_start_probabilities must use a floating-point dtype")

    result_device = round_start_probabilities.device
    # Keep the sampling design in float64. Casting capped probabilities back to
    # float32 can make cumulative intervals slightly wider than one, allowing
    # two adjacent systematic thresholds to select the same index.
    result_dtype = torch.float64
    weights = round_start_probabilities.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("round-start probabilities must be finite and nonnegative")

    length = weights.numel()
    probabilities = torch.zeros_like(weights)
    if length == 0:
        return _EDRAnchorSamplingDesign(
            inclusion_probabilities=probabilities.to(
                device=result_device,
                dtype=result_dtype,
            ),
            inverse_pps_scale=None,
        )
    if num_anchors >= length:
        return _EDRAnchorSamplingDesign(
            inclusion_probabilities=torch.ones_like(
                round_start_probabilities,
                device=result_device,
                dtype=result_dtype,
            ),
            inverse_pps_scale=None,
        )

    positive = weights > 0
    positive_count = int(positive.sum().item())
    if positive_count <= num_anchors:
        probabilities[positive] = 1.0
        return _EDRAnchorSamplingDesign(
            inclusion_probabilities=probabilities.to(
                device=result_device,
                dtype=result_dtype,
            ),
            inverse_pps_scale=None,
        )

    positive_weights = weights[positive]
    sorted_weights, _ = positive_weights.sort(descending=True)
    inverse_scale = None
    # Water filling: the first ``saturated_count`` largest weights have pi=1;
    # all remaining probabilities are lambda*w. At most rho-1 entries can be
    # saturated when there are more than rho positive candidates.
    for saturated_count in range(num_anchors):
        tail_sum = sorted_weights[saturated_count:].sum()
        candidate_inverse_scale = tail_sum / float(num_anchors - saturated_count)
        if bool(sorted_weights[saturated_count] <= candidate_inverse_scale):
            inverse_scale = candidate_inverse_scale
            break
    if inverse_scale is None:  # pragma: no cover - guarded by the water-filling invariant
        raise RuntimeError("failed to solve EDR anchor inclusion probabilities")

    probabilities = torch.minimum(torch.ones_like(weights), weights / inverse_scale)
    return _EDRAnchorSamplingDesign(
        inclusion_probabilities=probabilities.to(
            device=result_device,
            dtype=result_dtype,
        ),
        inverse_pps_scale=inverse_scale.to(device=result_device, dtype=result_dtype),
    )


@torch.no_grad()
def edr_anchor_inclusion_probabilities(
    round_start_probabilities: torch.Tensor,
    num_anchors: int,
) -> torch.Tensor:
    """Return the paper's capped-proportional anchor inclusion probabilities."""
    return _edr_anchor_sampling_design(
        round_start_probabilities,
        num_anchors,
    ).inclusion_probabilities


@torch.no_grad()
def sample_edr_round_starts(
    round_start_probabilities: torch.Tensor,
    num_anchors: int,
    *,
    random_start: Optional[float] = None,
) -> EDRAnchorSample:
    """Sample distinct round starts with the paper's systematic PPS design (Appendix C.2)."""
    design = _edr_anchor_sampling_design(
        round_start_probabilities,
        num_anchors,
    )
    inclusion = design.inclusion_probabilities
    result_device = inclusion.device
    inclusion_cpu = inclusion.detach().to(device="cpu", dtype=torch.float64)
    sample_size = int(round(float(inclusion_cpu.sum().item())))
    if sample_size == 0:
        indices = torch.empty(0, dtype=torch.long, device=result_device)
        return EDRAnchorSample(
            indices=indices,
            inclusion_probabilities=inclusion,
            inverse_pps_scale=design.inverse_pps_scale,
        )

    eligible = torch.nonzero(inclusion_cpu > 0.0, as_tuple=False).flatten()
    if sample_size == eligible.numel() and bool((inclusion_cpu[eligible] == 1.0).all()):
        return EDRAnchorSample(
            indices=eligible.to(device=result_device),
            inclusion_probabilities=inclusion,
            inverse_pps_scale=design.inverse_pps_scale,
        )

    if random_start is None:
        start = float(torch.rand((), dtype=torch.float64).item())
    else:
        start = float(random_start)
        if not 0.0 <= start < 1.0:
            raise ValueError(f"random_start must be in [0, 1), got {random_start}")

    cumulative = inclusion_cpu.cumsum(dim=0)
    # Guard only against a downward float64 summation error. Lowering a total
    # that rounded slightly above the integer could make the final cumulative
    # boundary non-monotonic when the last marginal is extremely small.
    if bool(cumulative[-1] < sample_size):
        cumulative[-1] = float(sample_size)
    threshold_offsets = torch.arange(sample_size, dtype=torch.float64)
    thresholds = start + threshold_offsets
    # Although start < 1, float64 addition can round start + j up to j + 1
    # when start is very close to one. Keep every systematic threshold inside
    # its mathematical half-open interval [j, j + 1) so the final threshold
    # cannot search past the end of the cumulative marginals.
    threshold_upper_bounds = threshold_offsets + 1.0
    thresholds = torch.minimum(
        thresholds,
        torch.nextafter(threshold_upper_bounds, threshold_offsets),
    )
    # right=True assigns exact cumulative-boundary hits to the next interval;
    # this also preserves distinctness for the measure-zero random_start=0 case.
    indices_cpu = torch.searchsorted(cumulative, thresholds, right=True)
    if (
        indices_cpu.numel() != sample_size
        or bool((indices_cpu >= inclusion_cpu.numel()).any())
        or torch.unique(indices_cpu).numel() != sample_size
    ):
        raise RuntimeError("systematic EDR sampling did not produce distinct anchors")

    return EDRAnchorSample(
        indices=indices_cpu.to(device=result_device),
        inclusion_probabilities=inclusion,
        inverse_pps_scale=design.inverse_pps_scale,
    )


def sampled_edr_surrogate_sum(
    costs: torch.Tensor,
    acceptance: torch.Tensor,
    dynamic_program: EDRDynamicProgram,
    round_prefixes: torch.Tensor,
    inclusion_probabilities: torch.Tensor,
    keep_mask: torch.Tensor,
    inverse_pps_scale: Optional[torch.Tensor],
) -> torch.Tensor:
    """Canceled Horvitz--Thompson sampled surrogate, Eq. (29).

    The three one-dimensional sampling tensors align with the first dimension
    of ``costs`` and may include fixed-query padding. Padded rows are excluded
    by ``keep_mask`` and therefore contribute neither value nor gradient. For
    an unsaturated capped-PPS anchor, ``pi_n=lambda*d_n`` and
    ``omega[n,t]=d_n*S[n,t]``, so its coefficient is evaluated as
    ``S[n,t]/lambda`` rather than the numerically fragile ``omega[n,t]/pi_n``.
    """
    if costs.ndim != 2 or acceptance.shape != costs.shape:
        raise ValueError("sampled EDR costs and acceptance must be equal rank-2 tensors")
    sample_size = costs.shape[0]
    expected_shape = (sample_size,)
    if round_prefixes.shape != expected_shape:
        raise ValueError(f"round_prefixes must have shape {expected_shape}")
    if inclusion_probabilities.shape != expected_shape:
        raise ValueError(f"inclusion_probabilities must have shape {expected_shape}")
    if keep_mask.shape != expected_shape:
        raise ValueError(f"keep_mask must have shape {expected_shape}")

    coefficients = prepare_sampled_edr_surrogate_coefficients(
        dynamic_program,
        round_prefixes,
        inclusion_probabilities,
        keep_mask,
        inverse_pps_scale,
        device=costs.device,
    )
    return edr_surrogate_sum_from_coefficients(costs, acceptance, coefficients)


@torch.no_grad()
def prepare_sampled_edr_surrogate_coefficients(
    dynamic_program: EDRDynamicProgram,
    round_prefixes: torch.Tensor,
    inclusion_probabilities: torch.Tensor,
    keep_mask: torch.Tensor,
    inverse_pps_scale: Optional[torch.Tensor] = None,
    *,
    device: Optional[torch.device | str] = None,
) -> EDRSurrogateCoefficients:
    """Validate and gather canceled HT coefficients where the DP already lives.

    Sampled training keeps the DP, FP64 PPS sampling, and this calculation on
    the CPU; only the selected FP32 coefficients are moved to the training GPU.
    ``device`` selects another destination explicitly.
    """
    if round_prefixes.ndim != 1:
        raise ValueError("round_prefixes must be rank 1")
    if inclusion_probabilities.shape != round_prefixes.shape:
        raise ValueError("inclusion_probabilities must have the same shape as round_prefixes")
    if keep_mask.shape != round_prefixes.shape:
        raise ValueError("keep_mask must have the same shape as round_prefixes")
    device = dynamic_program.occupancies.device if device is None else torch.device(device)
    keep = keep_mask.to(device=device, dtype=torch.bool)
    prefixes = round_prefixes.to(device=device, dtype=torch.long)
    kept_prefixes = prefixes[keep]
    horizon_length = dynamic_program.occupancies.shape[0]
    if kept_prefixes.numel() and bool(
        ((kept_prefixes < 0) | (kept_prefixes >= horizon_length)).any()
    ):
        raise ValueError("kept EDR round prefixes must be in dynamic-program bounds")

    probabilities = inclusion_probabilities.to(device=device, dtype=torch.float64)
    if bool((probabilities[keep] <= 0).any()):
        raise ValueError("kept EDR anchors must have positive inclusion probability")
    safe_prefixes = torch.where(keep, prefixes, torch.zeros_like(prefixes))

    weights = dynamic_program.occupancies.detach().to(device)[safe_prefixes]
    survivals = dynamic_program.conditional_survivals.detach().to(device)[safe_prefixes]
    advantages = dynamic_program.continuation_advantages.detach().to(device)[safe_prefixes]
    learned_mask = dynamic_program.learned_mask.to(device)[safe_prefixes]
    learned_mask = learned_mask & keep.unsqueeze(-1)
    round_starts = dynamic_program.round_start_probabilities.detach().to(device)[
        safe_prefixes
    ]
    if inverse_pps_scale is None:
        if bool((probabilities[keep] < 1.0).any()):
            raise ValueError("inverse_pps_scale is required for unsaturated EDR anchors")
        ht_weights = weights
    else:
        inverse_scale = torch.as_tensor(
            inverse_pps_scale,
            device=device,
            dtype=torch.float64,
        )
        if inverse_scale.numel() != 1 or not bool(torch.isfinite(inverse_scale)):
            raise ValueError("inverse_pps_scale must be one finite scalar")
        if bool(inverse_scale <= 0):
            raise ValueError("inverse_pps_scale must be positive")
        expected_probabilities = torch.minimum(
            torch.ones_like(round_starts),
            round_starts / inverse_scale,
        )
        if not torch.allclose(
            probabilities[keep],
            expected_probabilities[keep],
            rtol=1e-12,
            atol=0.0,
        ):
            raise ValueError(
                "EDR inclusion probabilities do not match the dynamic-program "
                "round starts and inverse_pps_scale"
            )
        saturated = round_starts >= inverse_scale
        canceled_weights = survivals * inverse_scale
        ht_weights = torch.where(saturated.unsqueeze(-1), weights, canceled_weights)
    return EDRSurrogateCoefficients(ht_weights.float(), advantages, learned_mask)


def edr_surrogate_sum_from_coefficients(
    costs: torch.Tensor,
    acceptance: torch.Tensor,
    coefficients: EDRSurrogateCoefficients,
) -> torch.Tensor:
    """Apply already-validated coefficients without GPU-backed host checks."""
    fields = (coefficients.weights, coefficients.continuation_advantages, coefficients.learned_mask)
    if costs.ndim != 2 or any(t.shape != costs.shape for t in (acceptance, *fields)):
        raise ValueError("EDR costs, acceptance, and prepared coefficients must have equal rank-2 shapes")
    if any(t.device != costs.device for t in (acceptance, *fields)):
        raise ValueError("prepared EDR coefficients and statistics must be on the same device")
    terms = coefficients.weights * (
        costs.float() + acceptance.float() * coefficients.continuation_advantages
    )
    return torch.where(coefficients.learned_mask, terms, torch.zeros_like(terms)).sum()
