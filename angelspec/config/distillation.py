"""Loss-distribution policy for the DFlash-family E2E/LK objectives.

E2E follows the cached target response policy by default; LK opts in explicitly.
The policy is resolved without importing a model, dataset loader, or inference
runtime. EDR uses its own sampling configuration.
"""

from dataclasses import asdict, is_dataclass

from angelspec.utils.sampling import validate_sampling_parameters


def _has_distillation_loss(args) -> bool:
    return (
        float(getattr(args, "dflash_e2e_tv_loss_weight", 0.0)) > 0.0
        or float(getattr(args, "dflash_lk_loss_weight", 0.0)) > 0.0
    )


def distillation_distribution_aware_enabled(
    enabled: bool | None,
    *,
    loss_objective: str,
    e2e_tv_loss_weight: float,
    lk_loss_weight: float,
) -> bool:
    """Resolve an omitted flag consistently for config and model callers.

    Only E2E without LK defaults on. Explicit booleans override that default.
    Returns False for EDR and for objectives without an E2E or LK term.
    """
    if str(loss_objective).lower() == "edr":
        return False
    e2e_active = float(e2e_tv_loss_weight) > 0.0
    lk_active = float(lk_loss_weight) > 0.0
    if not (e2e_active or lk_active):
        return False
    return (e2e_active and not lk_active) if enabled is None else bool(enabled)


def resolve_distillation_sampling(args) -> dict:
    """Return the effective loss policy.

    E2E is distribution-aware by default; LK uses T=1 over the full vocabulary
    by default. When enabled, the policy comes only from
    dataset.target_sampling (T=1 without filtering if it is unset); EDR
    sampling settings are ignored. Cache provenance is left unchanged.
    """
    enabled = distillation_distribution_aware_enabled(
        getattr(args, "dflash_distill_distribution_aware", None),
        loss_objective=getattr(args, "dflash_loss_objective", "decay"),
        e2e_tv_loss_weight=getattr(args, "dflash_e2e_tv_loss_weight", 0.0),
        lk_loss_weight=getattr(args, "dflash_lk_loss_weight", 0.0),
    )
    kwargs = {
        "distill_distribution_aware": enabled,
        "distill_temperature": 1.0,
        "distill_top_k": -1,
        "distill_top_p": 1.0,
    }
    if not enabled:
        return kwargs

    configured = getattr(args, "target_sampling", None)
    policy = (
        asdict(configured) if is_dataclass(configured)
        else dict(configured) if configured is not None else {}
    )
    temperature = policy.get("temperature", 1.0)
    top_k = policy.get("top_k", -1)
    top_p = policy.get("top_p", 1.0)
    validate_sampling_parameters(temperature, top_k, top_p)
    if temperature == 0:
        raise ValueError(
            "Distribution-aware E2E/LK training requires "
            "dataset.target_sampling.temperature > 0"
        )
    if policy.get("min_p", 0.0) != 0.0 or policy.get("enable_thinking", False) is not False:
        raise ValueError(
            "Target rollout caches currently require min_p=0 and enable_thinking=false"
        )
    kwargs.update(
        distill_temperature=float(temperature),
        distill_top_k=int(top_k),
        distill_top_p=float(top_p),
    )
    return kwargs


def configure_dflash_distillation(args) -> None:
    """Validate and log the active loss policy before model/dataset initialization."""
    if str(getattr(args, "dflash_loss_objective", "decay")).lower() == "edr" or not _has_distillation_loss(args):
        return
    policy = resolve_distillation_sampling(args)
    from angelspec.utils.logging import logger

    if policy["distill_distribution_aware"]:
        logger.info(
            "DFlash E2E/LK distribution-aware loss: target T=%s, top_k=%s, top_p=%s; "
            "draft T=%s, no top-k/top-p; TV/KL, LK mixture and D-PACE confidence "
            "use these probabilities",
            policy["distill_temperature"], policy["distill_top_k"], policy["distill_top_p"],
            policy["distill_temperature"],
        )
    else:
        logger.info(
            "DFlash E2E/LK distribution-aware loss disabled: legacy full-vocabulary "
            "target/draft T=1; configured rollout/cache policy is unchanged"
        )
