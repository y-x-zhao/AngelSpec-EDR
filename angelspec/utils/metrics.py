from __future__ import annotations

import torch


def edr_metric_totals(metrics: list[dict]) -> torch.Tensor:
    """Build EDR [surrogate sum, weighted cost, count, degenerate, tokens] totals.

    The backward surrogate is already summed over horizons within each row;
    weighted costs ``1 + U[0, 1]`` and generated-token counts are also
    additive. MAL is formed only after these totals have been combined globally.
    """
    if not metrics:
        raise ValueError("EDR metric reduction requires at least one metric dictionary")
    reference = metrics[0]["edr_num_horizons"]
    totals = torch.zeros(5, device=reference.device, dtype=torch.float32)
    for metric in metrics:
        horizon_count = metric["edr_num_horizons"].float()
        totals[0] += metric["edr_surrogate_loss"].float()
        totals[1] += metric["edr_weighted_cost_sum"].float()
        totals[2] += horizon_count
        totals[3] += metric["edr_num_degenerate_horizons"].float()
        totals[4] += metric["edr_generated_tokens"].float()
    return totals


def edr_metrics_from_totals(totals: torch.Tensor, prefix: str) -> dict[str, float]:
    """Convert globally summed EDR totals into user-facing metrics."""
    if totals.shape != (5,):
        raise ValueError(f"EDR totals must have shape (5,), got {tuple(totals.shape)}")
    horizon_count = totals[2]
    denominator = horizon_count.clamp(min=1.0)
    surrogate = totals[0] / denominator
    # Count each horizon's terminal EOS/stop/length-boundary token, matching the
    # offline evaluator's sampled-output length (ordinary tokens + 1).
    mal = (totals[4] + horizon_count) / totals[1].clamp_min(1.0)
    return {
        f"{prefix}edr_surrogate_loss": surrogate.item(),
        f"{prefix}edr_generated_tokens": totals[4].item(),
        f"{prefix}edr_weighted_cost": totals[1].item(),
        f"{prefix}edr_mal": mal.item(),
        f"{prefix}edr_num_horizons": horizon_count.item(),
        f"{prefix}edr_num_degenerate_horizons": totals[3].item(),
        f"{prefix}edr_mean_horizon_length": (totals[4] / denominator).item(),
    }


def token_metric_totals(
    metrics: list[dict],
    *,
    ce_key: str,
) -> torch.Tensor:
    """Build per-step [ce, kl, correct, correct_gt, count] token sums."""
    totals = []
    for item in metrics:
        count = item["acc_counts"].float()
        totals.append(
            torch.stack(
                [
                    item[ce_key].float() * count,
                    item.get("kl", torch.zeros_like(count)).float() * count,
                    item["acces"].float() * count,
                    item["acces_gt"].float() * count,
                    count,
                ],
                dim=-1,
            )
        )
    return torch.stack(totals, dim=0).sum(dim=0)


def means_from_token_totals(totals: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Convert per-step sums to means; zero-count steps remain finite zeros."""
    count = totals[:, 4]
    denom = count.clamp_min(1.0)
    return (
        totals[:, 0] / denom,
        totals[:, 1] / denom,
        totals[:, 2] / denom,
        totals[:, 3] / denom,
        count,
    )


def token_weighted_loss_scale(
    row_tokens: torch.Tensor,
    global_step_tokens: torch.Tensor,
    dp_world_size: int,
) -> torch.Tensor:
    """Scale a row so a later DP-AVG yields the global supervised-token mean."""
    if dp_world_size <= 0:
        raise ValueError(f"dp_world_size must be positive, got {dp_world_size}")
    return row_tokens.float() / global_step_tokens.float().clamp_min(1.0) * dp_world_size
