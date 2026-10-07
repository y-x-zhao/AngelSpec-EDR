from collections.abc import Sequence

import torch
import torch.nn.functional as F

from angelspec.models.dflash import DFlashModel
from angelspec.models.draft.dspark import VanillaMarkov


class DSparkModel(DFlashModel):
    """DSpark training wrapper (DFlash backbone + Markov / confidence heads)."""

    def __init__(
        self,
        draft_model,
        block_size: int = 7,
        num_anchors: int = 512,
        loss_decay_gamma: float = 7.0,
        loss_objective: str = "decay",
        dpace_alpha: float = 0.5,
        ce_loss_alpha: float = 0.1,
        l1_loss_alpha: float = 0.0,
        kl_loss_weight: float = 0.0,
        kl_topk: int = 10,
        lk_loss_weight: float = 0.0,
        lk_loss_type: str = "hybrid",
        lk_eta: float = 3.0,
        e2e_tv_loss_weight: float = 0.0,
        fp32_lm_head: bool = True,
        gate_entropy_weight: float = 0.0,
        confidence_head_alpha: float = 1.0,
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
        distill_distribution_aware: bool | None = None,
        distill_temperature: float = 1.0,
        distill_top_k: int = -1,
        distill_top_p: float = 1.0,
    ):
        # Forward the full DFlash loss config to the parent; ``confidence`` is
        # DSpark's only extra term (added in ``_compute_extra_loss``).
        super().__init__(
            draft_model=draft_model,
            block_size=block_size,
            num_anchors=num_anchors,
            loss_decay_gamma=loss_decay_gamma,
            fp32_lm_head=fp32_lm_head,
            gate_entropy_weight=gate_entropy_weight,
            loss_objective=loss_objective,
            dpace_alpha=dpace_alpha,
            ce_loss_alpha=ce_loss_alpha,
            l1_loss_alpha=l1_loss_alpha,
            kl_loss_weight=kl_loss_weight,
            kl_topk=kl_topk,
            lk_loss_weight=lk_loss_weight,
            lk_loss_type=lk_loss_type,
            lk_eta=lk_eta,
            e2e_tv_loss_weight=e2e_tv_loss_weight,
            edr_chunk_size=edr_chunk_size,
            edr_vocab_chunk_size=edr_vocab_chunk_size,
            edr_dp_workers=edr_dp_workers,
            edr_full_anchor_backprop=edr_full_anchor_backprop,
            query_includes_input_anchor=query_includes_input_anchor,
            edr_stop_token_ids=edr_stop_token_ids,
            edr_temperature=edr_temperature,
            edr_top_k=edr_top_k,
            edr_top_p=edr_top_p,
            distill_mean_by_row=distill_mean_by_row,
            edr_reuse_context_cache=edr_reuse_context_cache,
            edr_rejection_cache_max_mb=edr_rejection_cache_max_mb,
            distill_distribution_aware=distill_distribution_aware,
            distill_temperature=distill_temperature,
            distill_top_k=distill_top_k,
            distill_top_p=distill_top_p,
        )
        self.confidence_head_alpha = float(confidence_head_alpha)
        confidence_head = getattr(self.draft_model, "confidence_head", None)
        if confidence_head is not None and (
            self.loss_objective == "edr" or self.confidence_head_alpha <= 0
        ):
            # Keep the module and its checkpoint keys, but freeze it so DDP
            # skips its gradients when the confidence loss is off.
            confidence_head.requires_grad_(False)
        # Handoff buffer for corrected draft hidden states, between the
        # ``_compute_draft_logits`` and ``_compute_extra_loss`` hooks.
        self._dspark_hidden_4d = None
        # Runtime execution flags (not checkpoint state), set by the trainer:
        # fuse the VanillaMarkov bias into the logit projection.
        self.edr_fused_markov_projection = False
        self.distill_fused_markov_projection = False

    # ------------------------------------------------------------------
    # DFlash subclass hooks
    # ------------------------------------------------------------------

    def _compute_draft_logits(
        self,
        draft_hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        prev_token_ids: torch.Tensor,
        n_blocks: int,
    ) -> torch.Tensor:
        """Inject hidden-states correction + Markov bias, then project to logits.

        ``prev_token_ids`` is ``[B, n_blocks, projection_width]`` — the
        ground-truth token preceding each projected slot's target (aligned with
        ``draft_hidden``). With ``query_includes_input_anchor``, the input-anchor
        query slot is removed before this hook.
        """
        bsz = draft_hidden.size(0)
        if n_blocks < 1 or draft_hidden.shape[1] % n_blocks:
            raise ValueError("draft hidden length must be divisible by n_blocks")
        projection_width = draft_hidden.shape[1] // n_blocks
        if prev_token_ids.shape != (bsz, n_blocks, projection_width):
            raise ValueError("previous-token layout must match projected draft slots")

        # Cache the corrected hidden states for the confidence head.
        self._dspark_hidden_4d = (
            draft_hidden.view(bsz, n_blocks, projection_width, -1)
            if self._extra_distill_needed() else None
        )

        markov_head = getattr(self.draft_model, "markov_head", None)
        fuse_markov = (
            (
                (self.edr_fused_markov_projection and self.loss_objective == "edr")
                or (
                    self.distill_fused_markov_projection
                    and self.loss_objective in {"decay", "dpace"}
                )
            )
            and isinstance(markov_head, VanillaMarkov)
        )
        # Project directly to a 2D allocation for in-place addmm. Mutating a
        # view of a 3D output instead inserts CopySlices into autograd, which
        # copies the full vocabulary gradient during backward.
        projection_input = draft_hidden.reshape(-1, draft_hidden.shape[-1]) if fuse_markov else draft_hidden
        logits = F.linear(projection_input, lm_head_weight)

        # Markov-biased draft logits (teacher-forced previous token).
        if markov_head is not None:
            if (
                fuse_markov and logits.dtype == markov_head.markov_w2.weight.dtype
            ):
                previous_hidden = markov_head.get_prev_embeddings(prev_token_ids)
                # ``logits`` is a fresh F.linear output owned by this call, so
                # addmm_ adds the Markov bias in place inside the GEMM without
                # materializing an [N, V] bias tensor. The bias is accumulated
                # in the GEMM rather than rounded to the logits dtype first.
                logits.addmm_(
                    previous_hidden.reshape(-1, previous_hidden.shape[-1]),
                    markov_head.markov_w2.weight.t(),
                )
            else:
                logits_4d = markov_head.apply_block_logits(
                    logits.view(bsz, n_blocks, projection_width, -1),
                    token_ids=prev_token_ids,
                )
                logits = logits_4d.reshape(bsz, n_blocks * projection_width, -1)

        return logits.reshape(bsz, n_blocks * projection_width, logits.shape[-1]) if fuse_markov else logits

    def _compute_step_logits(
        self,
        draft_hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        previous_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the Markov correction to one sampled proposal position."""

        logits = F.linear(draft_hidden, lm_head_weight)
        markov_head = getattr(self.draft_model, "markov_head", None)
        if markov_head is not None:
            logits = logits + markov_head.compute_step_bias(previous_token_ids)
        return logits

    def _extra_distill_needed(self) -> bool:
        """Confidence head needs the teacher logits (for the accept-rate target)."""
        return (
            getattr(self.draft_model, "confidence_head", None) is not None
            and self.confidence_head_alpha > 0
        )

    def _compute_extra_loss(
        self,
        loss: torch.Tensor,
        flat_logits: torch.Tensor,
        teacher_logits_flat,
        flat_weights: torch.Tensor,
        valid_token_count: torch.Tensor,
        prev_token_ids: torch.Tensor,
        n_blocks: int,
    ):
        """Add the confidence-head BCE against the empirical accept rate.

        Uses the same objective-weighted mask (``flat_weights``) and weighted-mean
        reduction as the DFlash CE / distillation terms. Returns
        ``(loss, {"confidence_loss": ...})`` for shared logging.
        """
        confidence_loss = torch.zeros((), device=loss.device, dtype=loss.dtype)

        # Confidence BCE needs the teacher accept-rate target; skip when the head
        # is off or target last_hidden_states weren't delivered this step.
        if not self._extra_distill_needed() or teacher_logits_flat is None:
            return loss, {"confidence_loss": confidence_loss.detach()}

        # accept_rate = 1 - 0.5 * L1(draft, teacher)  in [0, 1], per position.
        l1_per_position = self._compute_l1_loss(flat_logits, teacher_logits_flat)  # [N]
        accept_rate = (1.0 - 0.5 * l1_per_position).clamp(0.0, 1.0)

        hidden_4d = self._dspark_hidden_4d
        if getattr(self.draft_model, "confidence_head_with_markov", False):
            prev_emb = self.draft_model.markov_head.get_prev_embeddings(prev_token_ids).to(
                hidden_4d.dtype
            )
            conf_features = torch.cat([hidden_4d, prev_emb], dim=-1)
        else:
            conf_features = hidden_4d

        confidence_pred = self.draft_model.confidence_head(conf_features).float().reshape(-1)
        conf_bce = F.binary_cross_entropy_with_logits(
            confidence_pred, accept_rate.detach(), reduction="none"
        )
        confidence_loss = ((conf_bce * flat_weights.float()).sum() / valid_token_count.float()).to(
            loss.dtype
        )

        loss = loss + self.confidence_head_alpha * confidence_loss
        return loss, {"confidence_loss": confidence_loss.detach()}
