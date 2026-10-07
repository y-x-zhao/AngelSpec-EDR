"""Lightweight configuration validation for the DFlash EDR objective."""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Iterable
from typing import Optional

from angelspec.utils.logging import logger
from angelspec.utils.sampling import validate_sampling_parameters


def _normalize_stop_token_ids(value) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, int):
        values: Iterable[int] = (value,)
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        values = value
    else:
        raise TypeError(
            "stopping token IDs must be an integer or an iterable of integers"
        )

    normalized = set()
    for token_id in values:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError(f"stopping token ID must be an integer, got {token_id!r}")
        if token_id < 0:
            raise ValueError(f"stopping token ID must be non-negative, got {token_id}")
        normalized.add(int(token_id))
    return normalized


def resolve_edr_stop_token_ids(
    target_model_path: str,
    *,
    explicit_stop_token_ids=None,
    decoder_stop_token_ids=None,
    trust_remote_code: bool = True,
    local_files_only: Optional[bool] = None,
) -> tuple[int, ...]:
    """Resolve every token ID that independently stops the deployed decoder.

    The target generation configuration is authoritative for EOS. Explicit EDR
    and decoder stop-token IDs are additive because those tokens also terminate
    decoding. Stop strings are intentionally excluded: a token inside a
    multi-token stop string is not independently terminal.
    """
    stop_ids = _normalize_stop_token_ids(explicit_stop_token_ids)
    stop_ids.update(_normalize_stop_token_ids(decoder_stop_token_ids))

    load_kwargs = {"trust_remote_code": trust_remote_code}
    if local_files_only is not None:
        load_kwargs["local_files_only"] = local_files_only

    generation_eos = None
    if target_model_path:
        try:
            from transformers import GenerationConfig

            generation_config = GenerationConfig.from_pretrained(
                target_model_path,
                **load_kwargs,
            )
            generation_eos = getattr(generation_config, "eos_token_id", None)
        except (OSError, ValueError):
            # Some checkpoints do not ship generation_config.json. Their model
            # config remains the same fallback used by Transformers generation.
            pass

        if generation_eos is None:
            from transformers import AutoConfig

            model_config = AutoConfig.from_pretrained(target_model_path, **load_kwargs)
            text_config = getattr(model_config, "text_config", model_config)
            generation_eos = getattr(text_config, "eos_token_id", None)
            if generation_eos is None:
                generation_eos = getattr(model_config, "eos_token_id", None)
        stop_ids.update(_normalize_stop_token_ids(generation_eos))

    if not stop_ids:
        raise ValueError(
            "EDR requires at least one independently stopping token ID. Set "
            "training.dflash_edr_stop_token_ids explicitly when the target "
            "generation config has no eos_token_id."
        )
    return tuple(sorted(stop_ids))


def configure_dflash_edr(args: Namespace) -> bool:
    """Validate EDR's second-stage inputs and neutralize auxiliary losses."""
    loss_objective = str(getattr(args, "dflash_loss_objective", "decay")).lower()
    if loss_objective != "edr":
        return False

    temperature = float(getattr(args, "dflash_edr_temperature", 1.0))
    top_k = getattr(args, "dflash_edr_top_k", -1)
    top_p = float(getattr(args, "dflash_edr_top_p", 1.0))
    # Reject T=0: a greedy draft distribution has zero gradient almost everywhere.
    validate_sampling_parameters(temperature, top_k, top_p, allow_greedy=False)

    if not getattr(args, "load_path", None):
        raise ValueError(
            "DFlash EDR is a second-stage replacement objective and requires "
            "training.load_path to an existing draft checkpoint."
        )
    edr_chunk_size = int(getattr(args, "dflash_edr_chunk_size", 64))
    if edr_chunk_size < 1:
        raise ValueError(f"training.dflash_edr_chunk_size must be >= 1, got {edr_chunk_size}")
    vocab_chunk_size = int(getattr(args, "dflash_edr_vocab_chunk_size", 16384))
    if vocab_chunk_size < 1:
        raise ValueError(
            "training.dflash_edr_vocab_chunk_size must be >= 1, "
            f"got {vocab_chunk_size}"
        )
    cross_row_batch_size = int(
        getattr(args, "dflash_edr_cross_row_batch_size", 1)
    )
    if cross_row_batch_size < 1:
        raise ValueError(
            "training.dflash_edr_cross_row_batch_size must be >= 1, "
            f"got {cross_row_batch_size}"
        )
    dp_workers = int(getattr(args, "dflash_edr_dp_workers", 1))
    if dp_workers < 1:
        raise ValueError(
            f"training.dflash_edr_dp_workers must be >= 1, got {dp_workers}"
        )
    accumulation_steps = int(getattr(args, "draft_accumulation_steps", 1))
    if accumulation_steps % cross_row_batch_size:
        raise ValueError(
            "training.draft_accumulation_steps must be divisible by "
            "training.dflash_edr_cross_row_batch_size"
        )
    if cross_row_batch_size > 1:
        if bool(getattr(args, "dflash_packing", False)):
            raise ValueError("cross-row EDR batching is incompatible with dflash_packing")
        if str(getattr(args, "attention_backend", "")).lower() == "usp":
            raise ValueError("cross-row EDR batching is incompatible with USP")
        micro_batch_size = int(getattr(args, "micro_batch_size", 1))
        rows_in_flight = cross_row_batch_size * micro_batch_size
        if edr_chunk_size < rows_in_flight:
            raise ValueError(
                "training.dflash_edr_chunk_size must be at least the number of EDR "
                f"rows in flight ({rows_in_flight})"
            )
    stop_token_ids = resolve_edr_stop_token_ids(
        getattr(args, "target_model_path", ""),
        explicit_stop_token_ids=getattr(args, "dflash_edr_stop_token_ids", None),
        decoder_stop_token_ids=getattr(args, "decode_stop_token_ids", None),
        trust_remote_code=bool(getattr(args, "trust_remote_code", True)),
    )
    # Resolve once before the training actors are created so every draft rank
    # conditions on the same deployed stopping-token set.
    args.dflash_edr_stop_token_ids = list(stop_token_ids)
    num_anchors = int(getattr(args, "dflash_num_anchors", 512))
    if num_anchors < 1:
        raise ValueError(f"training.dflash_num_anchors must be >= 1, got {num_anchors}")
    full_anchor_backprop = bool(
        getattr(args, "dflash_edr_full_anchor_backprop", False)
    )
    if not getattr(args, "store_last_hidden_states", False):
        raise ValueError(
            "DFlash EDR requires target last_hidden_states; set "
            "inference.store_last_hidden_states=true."
        )
    fsdp_strategy = str(getattr(args, "fsdp_strategy", "REPLICATE")).upper()
    if fsdp_strategy != "REPLICATE":
        raise ValueError(
            "DFlash EDR with token-length-only balancing requires "
            "training.fsdp_strategy=REPLICATE. FULL_SHARD cannot safely run "
            "the variable number of EDR horizon/statistics forwards issued by each "
            "data-parallel rank."
        )
    # dflash_block_size counts learned proposals; the input anchor adds one query slot.
    block_size = int(getattr(args, "dflash_block_size", 7))
    query_includes_anchor = bool(getattr(args, "dflash_query_includes_input_anchor", False))
    if block_size < 1:
        raise ValueError("DFlash EDR requires dflash_block_size >= 1.")

    replacement_weights = {
        "dflash_l1_loss_alpha": 0.0,
        "dflash_kl_loss_weight": 0.0,
        "dflash_lk_loss_weight": 0.0,
        "dflash_e2e_tv_loss_weight": 0.0,
        "dflash_gate_entropy_weight": 0.0,
        "dspark_l1_loss_alpha": 0.0,
        "dspark_confidence_head_alpha": 0.0,
    }
    disabled = []
    for name, zero in replacement_weights.items():
        value = getattr(args, name, zero)
        if value is not None and float(value) != zero:
            disabled.append(name)
        setattr(args, name, zero)
    if getattr(args, "dflash_opd_enabled", False):
        disabled.append("dflash_opd_enabled")
    args.dflash_opd_enabled = False

    gradient_anchor_mode = (
        "all round starts (importance sampling disabled)"
        if full_anchor_backprop
        else f"capped-PPS sampling (max {num_anchors})"
    )
    logger.info(
        "EDR distributions: target T=%s top_k=%s top_p=%s; "
        "draft T=%s without top-k/top-p; target-generated cache must match this policy",
        temperature, top_k, top_p, temperature,
    )
    logger.info(
        "DFlash EDR: exact second-stage replacement objective enabled "
        "(statistics_chunk_size=%d, vocab_chunk_size=%d, cross_row_batch_size=%d, "
        "dp_workers=%d, gradient_anchor_mode=%s, "
        "query_slots=%d, learned_proposals=%d, stopping_token_ids=%s); "
        "CE/D-PACE weights and auxiliary losses are inactive%s.",
        edr_chunk_size,
        vocab_chunk_size,
        cross_row_batch_size,
        dp_workers,
        gradient_anchor_mode,
        block_size + int(query_includes_anchor),
        block_size,
        list(stop_token_ids),
        f"; disabled {', '.join(disabled)}" if disabled else "",
    )
    return True
