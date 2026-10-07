"""DSpark trainer — DFlash trainer + Markov / confidence heads.

Reuses DFlashTrainer's initialization, checkpoint and optimizer pipeline and
overrides model construction. DSpark reads its own dspark_* loss settings.
"""

from argparse import Namespace

from angelspec.config.distillation import resolve_distillation_sampling
from angelspec.models.draft.dspark import DSparkConfig, DSparkDraftModel
from angelspec.models.dspark import DSparkModel
from angelspec.training.dflash_trainer import DFlashTrainer


class DSparkTrainer(DFlashTrainer):
    """DSpark-specific trainer (DFlash backbone + EAGLE-style heads)."""

    _extra_loss_component_keys = DFlashTrainer._extra_loss_component_keys + ["confidence_loss"]

    def __init__(self, args: Namespace):
        super().__init__(args)
        # DSpark reads its own dspark_* settings and defaults.
        self.block_size = getattr(args, "dflash_block_size", 7)
        self.num_anchors = getattr(args, "dspark_num_anchors", 512)
        self.num_target_layers = getattr(args, "dspark_num_target_layers", 5)
        self.loss_decay_gamma = getattr(args, "dspark_loss_decay_gamma", 4.0)
        self.ce_loss_alpha = getattr(args, "dspark_ce_loss_alpha", 0.1)
        self.l1_loss_alpha = getattr(args, "dspark_l1_loss_alpha", 0.9)
        self.confidence_head_alpha = getattr(args, "dspark_confidence_head_alpha", 1.0)

    def init_model(self, draft_model_config, target_model_path: str, mooncake_config=None) -> int:
        # DFlash's shared initializer accepts subclasses of DFlashConfig. Parse
        # path/dict inputs as DSparkConfig so DSpark's head defaults apply.
        if isinstance(draft_model_config, DSparkConfig):
            config = draft_model_config
        elif isinstance(draft_model_config, str):
            config = DSparkConfig.from_pretrained(draft_model_config)
        elif isinstance(draft_model_config, dict):
            config = DSparkConfig(**draft_model_config)
        else:
            raise TypeError(
                f"Unsupported draft_model_config type: {type(draft_model_config).__name__}. "
                "Expected str, dict, or DSparkConfig."
            )
        return super().init_model(config, target_model_path, mooncake_config)

    def _build_draft_model(self, config):
        draft_model = DSparkDraftModel(config)
        if draft_model.confidence_head is not None and (
            self.loss_objective == "edr" or self.confidence_head_alpha <= 0
        ):
            # The head stays in the architecture and checkpoint, but receives no
            # gradient here; freezing it excludes it from DDP and the optimizer.
            draft_model.confidence_head.requires_grad_(False)
        return draft_model

    def _build_training_wrapper(self, draft_model):
        return DSparkModel(
            draft_model=draft_model,
            block_size=self.block_size,
            num_anchors=self.num_anchors,
            loss_decay_gamma=self.loss_decay_gamma,
            loss_objective=self.loss_objective,
            dpace_alpha=self.dpace_alpha,
            ce_loss_alpha=self.ce_loss_alpha,
            l1_loss_alpha=self.l1_loss_alpha,
            kl_loss_weight=self.kl_loss_weight,
            kl_topk=self.kl_topk,
            lk_loss_weight=self.lk_loss_weight,
            lk_loss_type=self.lk_loss_type,
            lk_eta=self.lk_eta,
            e2e_tv_loss_weight=self.e2e_tv_loss_weight,
            **resolve_distillation_sampling(self.args),
            fp32_lm_head=self.fp32_lm_head,
            gate_entropy_weight=getattr(self.args, "dflash_gate_entropy_weight", 0.0),
            confidence_head_alpha=self.confidence_head_alpha,
            edr_chunk_size=self.edr_chunk_size,
            edr_vocab_chunk_size=self.edr_vocab_chunk_size,
            edr_dp_workers=self.edr_dp_workers,
            edr_full_anchor_backprop=self.edr_full_anchor_backprop,
            query_includes_input_anchor=self.query_includes_input_anchor,
            edr_stop_token_ids=self.edr_stop_token_ids,
            edr_temperature=getattr(self.args, "dflash_edr_temperature", 1.0),
            edr_top_k=getattr(self.args, "dflash_edr_top_k", -1),
            edr_top_p=getattr(self.args, "dflash_edr_top_p", 1.0),
            distill_mean_by_row=self.distill_cross_row_batch_size > 1,
            edr_reuse_context_cache=getattr(self.args, "dflash_edr_reuse_context_cache", False),
            edr_rejection_cache_max_mb=getattr(self.args, "dflash_edr_rejection_cache_max_mb", 0),
        )
