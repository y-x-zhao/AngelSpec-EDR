"""DFly training wrapper — DFlash backbone + TreeFlash hidden-states correction.

Like :class:`DFlashModel`, but applies the optional TreeFlash correction
(formula (1)) to the drafter output before the LM head, conditioning the token
distribution on the previous token. The correction lives on
``draft_model.hidden_correction`` and is a no-op (``None``) when disabled.
"""

import torch
import torch.nn.functional as F

from angelspec.models.dflash import DFlashModel


class DFlyModel(DFlashModel):
    """DFly training wrapper (DFlash backbone + hidden-states correction)."""

    def _compute_draft_logits(
        self,
        draft_hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        prev_token_ids: torch.Tensor,
        n_blocks: int,
    ) -> torch.Tensor:
        """Apply hidden-states correction (if present), then project to logits.

        ``prev_token_ids`` is ``[B, n_blocks, projection_width]`` — the
        ground-truth token preceding each projected slot's target (aligned with
        ``draft_hidden``). With ``query_includes_input_anchor``, the input-anchor
        query slot is removed before this hook.
        """
        # Correction (TreeFlash formula (1)) BEFORE the LM head, conditioning the
        # token distribution on the previous token.
        if getattr(self.draft_model, "hidden_correction", None) is not None:
            bsz = draft_hidden.size(0)
            prev_embeds = self.draft_model.embed_tokens(prev_token_ids)
            prev_embeds = prev_embeds.view(bsz, -1, prev_embeds.size(-1))
            draft_hidden = self.draft_model.hidden_correction(draft_hidden, prev_embeds)

        if hasattr(self.draft_model, "lm_head"):
            return self.draft_model.lm_head(draft_hidden)
        return F.linear(draft_hidden, lm_head_weight)

    def _compute_step_logits(
        self,
        draft_hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        previous_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the DFly correction to one sampled proposal position."""

        correction = getattr(self.draft_model, "hidden_correction", None)
        if correction is not None:
            previous_embeddings = self.draft_model.embed_tokens(previous_token_ids)
            draft_hidden = correction(draft_hidden, previous_embeddings)
        if hasattr(self.draft_model, "lm_head"):
            return self.draft_model.lm_head(draft_hidden)
        return F.linear(draft_hidden, lm_head_weight)
