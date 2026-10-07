"""DFlash proposal indexing.

The anchor is an input token. Query slot ``j`` predicts the token at
``anchor + 1 + j`` for every ``j`` in ``[0, block_size)``. All DFlash training
objectives share this helper for label and predecessor indexing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

DFLASH_PROPOSAL_PROTOCOL = "learned_first_b_proposals_v2"


@dataclass(frozen=True)
class DFlashProposalLayout:
    """Checked proposal indices and their per-slot validity mask."""

    anchor_indices: torch.Tensor
    label_indices: torch.Tensor
    safe_label_indices: torch.Tensor
    predecessor_indices: torch.Tensor
    valid_mask: torch.Tensor

    def _validate_source(self, tensor: torch.Tensor, name: str) -> None:
        if tensor.ndim != 2:
            raise ValueError(f"{name} must be rank 2, got shape {tuple(tensor.shape)}")
        if tensor.shape[0] != self.anchor_indices.shape[0]:
            raise ValueError(f"{name} batch size does not match proposal anchors")
        if tensor.shape[1] < 1:
            raise ValueError(f"{name} must have a non-empty sequence dimension")

    def gather_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Gather the teacher-forced output token for every query slot."""

        self._validate_source(input_ids, "input_ids")
        return torch.gather(
            input_ids.unsqueeze(1).expand(-1, self.anchor_indices.shape[1], -1),
            2,
            self.safe_label_indices,
        )

    def gather_predecessor_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Gather each slot's teacher-forced predecessor token (used by DFly and DSpark)."""

        self._validate_source(input_ids, "input_ids")
        return torch.gather(
            input_ids.unsqueeze(1).expand(-1, self.anchor_indices.shape[1], -1),
            2,
            self.predecessor_indices,
        )


def _validate_optional_mask(
    value: Optional[torch.Tensor],
    *,
    name: str,
    batch_size: int,
    sequence_length: int,
) -> None:
    if value is not None and value.shape != (batch_size, sequence_length):
        raise ValueError(
            f"{name} must have shape {(batch_size, sequence_length)}, got {tuple(value.shape)}"
        )


def build_dflash_proposal_layout(
    anchor_indices: torch.Tensor,
    *,
    sequence_length: int,
    block_size: int,
    attention_mask: Optional[torch.Tensor] = None,
    loss_mask: Optional[torch.Tensor] = None,
    ctx_doc_ids: Optional[torch.Tensor] = None,
) -> DFlashProposalLayout:
    """Map input anchors to ``B`` learned next-token proposal slots.

    ``anchor_indices`` has shape ``[batch, blocks]``.  Slot ``j`` predicts
    ``anchor + 1 + j`` and consumes the token at ``anchor + j`` as its
    teacher-forced predecessor.  Bounds, padding, supervision, and packed
    document constraints are combined in ``valid_mask``.
    """

    if anchor_indices.ndim != 2:
        raise ValueError(f"anchor_indices must be rank 2, got shape {tuple(anchor_indices.shape)}")
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if anchor_indices.dtype == torch.bool or anchor_indices.is_floating_point():
        raise TypeError("anchor_indices must use an integer dtype")

    batch_size, number_of_blocks = anchor_indices.shape
    _validate_optional_mask(
        attention_mask,
        name="attention_mask",
        batch_size=batch_size,
        sequence_length=sequence_length,
    )
    _validate_optional_mask(
        loss_mask,
        name="loss_mask",
        batch_size=batch_size,
        sequence_length=sequence_length,
    )
    _validate_optional_mask(
        ctx_doc_ids,
        name="ctx_doc_ids",
        batch_size=batch_size,
        sequence_length=sequence_length,
    )

    anchors = anchor_indices.to(dtype=torch.long)
    offsets = torch.arange(
        1,
        block_size + 1,
        device=anchors.device,
        dtype=torch.long,
    ).view(1, 1, -1)
    labels = anchors.unsqueeze(-1) + offsets
    anchor_in_bounds = (anchors >= 0) & (anchors < sequence_length)
    label_in_bounds = (labels >= 1) & (labels < sequence_length)
    safe_labels = labels.clamp(min=0, max=sequence_length - 1)
    predecessors = (safe_labels - 1).clamp(min=0)
    valid = anchor_in_bounds.unsqueeze(-1) & label_in_bounds

    safe_anchors = anchors.clamp(min=0, max=sequence_length - 1)
    expanded_attention = None
    if attention_mask is not None:
        expanded_attention = attention_mask.to(device=anchors.device)
        anchor_attention = torch.gather(expanded_attention, 1, safe_anchors) > 0
        label_attention = (
            torch.gather(
                expanded_attention.unsqueeze(1).expand(-1, number_of_blocks, -1),
                2,
                safe_labels,
            )
            > 0
        )
        predecessor_attention = (
            torch.gather(
                expanded_attention.unsqueeze(1).expand(-1, number_of_blocks, -1),
                2,
                predecessors,
            )
            > 0
        )
        valid = valid & anchor_attention.unsqueeze(-1) & predecessor_attention & label_attention

    if loss_mask is not None:
        supervised = (
            torch.gather(
                loss_mask.to(device=anchors.device).unsqueeze(1).expand(-1, number_of_blocks, -1),
                2,
                safe_labels,
            )
            > 0
        )
        valid = valid & supervised

    if ctx_doc_ids is not None:
        documents = ctx_doc_ids.to(device=anchors.device)
        anchor_documents = torch.gather(documents, 1, safe_anchors)
        label_documents = torch.gather(
            documents.unsqueeze(1).expand(-1, number_of_blocks, -1),
            2,
            safe_labels,
        )
        predecessor_documents = torch.gather(
            documents.unsqueeze(1).expand(-1, number_of_blocks, -1),
            2,
            predecessors,
        )
        same_document = (label_documents == anchor_documents.unsqueeze(-1)) & (
            predecessor_documents == anchor_documents.unsqueeze(-1)
        )
        valid = valid & (anchor_documents >= 0).unsqueeze(-1) & same_document

    # A proposal block is a causal chain: once a slot is invalid (out of bounds,
    # padding, unsupervised, or in another document), every later slot is too.
    valid = valid & valid.to(dtype=torch.int64).cumprod(dim=-1).bool()

    return DFlashProposalLayout(
        anchor_indices=anchors,
        label_indices=labels,
        safe_label_indices=safe_labels,
        predecessor_indices=predecessors,
        valid_mask=valid,
    )


__all__ = [
    "DFLASH_PROPOSAL_PROTOCOL",
    "DFlashProposalLayout",
    "build_dflash_proposal_layout",
]
