"""Sampling policy shared by EDR, distribution-aware E2E/LK, and evaluation.

Imports stay lightweight so cache lookup and CLI validation never initialize
CUDA. Reconstruction follows vLLM's FP32 PyTorch sampling path, not its
Triton/FlashInfer alternatives. Top-k precedes top-p.
"""

from __future__ import annotations

import hashlib
import math
import re
from numbers import Integral
from pathlib import Path


def validate_sampling_parameters(
    temperature: float, top_k: int, top_p: float, *, allow_greedy: bool = True,
) -> None:
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    if not allow_greedy and temperature == 0:
        raise ValueError(
            "EDR training requires temperature > 0: a greedy draft distribution "
            "has zero gradient almost everywhere. T=0 is supported for rollouts/evaluation."
        )
    if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k == 0 or top_k < -1:
        raise ValueError("top_k must be -1 (disabled) or a positive integer")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be finite and in (0, 1]")


def validate_vllm_sampling_parameters(params, *, temperature, top_k, top_p) -> None:
    """Raise if vLLM changed the requested sampling policy (e.g. by clamping T).

    Greedy vLLM requests normalize k/p to disabled values; those filters have
    no effect on an argmax distribution. Different releases use 0 or -1 for
    disabled top-k, so compare their meaning rather than that representation.
    """
    validate_sampling_parameters(temperature, top_k, top_p)
    if float(params.temperature) != float(temperature):
        raise ValueError(
            f"vLLM changed requested temperature {temperature} to {params.temperature}; "
            "use T=0 or a positive temperature supported without clamping by your "
            "vLLM build (commonly T>=0.01). EDR requires matching rollout/scoring distributions."
        )
    if temperature == 0:
        return
    actual_k = -1 if params.top_k in (-1, 0) else params.top_k
    if actual_k != top_k or float(params.top_p) != float(top_p):
        raise ValueError("vLLM changed the requested top-k/top-p sampling policy")


def _scale_sampling_logits(logits, temperature, *, _legacy_arithmetic=False):
    """Scale in FP32 using vLLM's tensor division (including on CUDA).

    ``_legacy_arithmetic`` divides by a Python scalar instead, which can
    select reciprocal multiplication. All callers use the default.
    """
    scaled = logits.float()
    if temperature != 1:
        divisor = temperature if _legacy_arithmetic else scaled.new_full((1,), temperature)
        scaled = scaled / divisor
    return scaled


def _sorted_sampling_logits(scaled, *, top_k, top_p, _legacy_arithmetic=False):
    """vLLM PyTorch's full-vocabulary sort/filter, without the final scatter.

    Dense reconstruction and compact target selection must share both the sort
    order of ties and the FP32 nucleus cutoff. Sorting only top-k candidates
    can choose different token IDs when top-p splits a group of tied logits.
    """
    import torch

    sorted_logits, indices = scaled.sort(dim=-1, descending=False)
    if 0 < top_k < scaled.shape[-1]:
        threshold = sorted_logits[..., -top_k].unsqueeze(-1)
        sorted_logits = sorted_logits.masked_fill(sorted_logits < threshold, -torch.inf)
    if top_p < 1:
        cumulative = sorted_logits.softmax(dim=-1, dtype=torch.float32).cumsum(dim=-1)
        # vLLM stores p in an FP32 tensor *before* subtracting. Rounding the
        # Python result 1-p to FP32 instead changes membership at the cutoff.
        cutoff = 1 - top_p if _legacy_arithmetic else 1 - scaled.new_full((1,), top_p)
        remove = cumulative <= cutoff
        remove[..., -1] = False
        sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
    return sorted_logits, indices


def sampling_logits(
    logits, *, temperature=1.0, top_k=-1, top_p=1.0, _legacy_arithmetic=False,
):
    """Return FP32 logits for the actual sampling distribution, without mutation.

    At T=0, encode the deterministic (first-argmax) point mass as 0/-inf.
    Positive temperatures scale before top-k, then nucleus filtering. Top-k
    keeps threshold ties, and nucleus filtering removes the ascending tail
    whose cumulative mass is <=1-p, always retaining at least one token.
    The default matches vLLM's FP32 PyTorch filter for identical input logits
    on the same device. ``_legacy_arithmetic`` selects Python-scalar
    arithmetic (see ``_scale_sampling_logits``).
    """
    import torch

    validate_sampling_parameters(temperature, top_k, top_p)
    if logits.ndim < 1 or logits.shape[-1] < 1:
        raise ValueError("logits must have a nonempty vocabulary dimension")
    if temperature == 0:
        greedy_ids = logits.argmax(dim=-1, keepdim=True)
        return torch.full_like(logits, -torch.inf, dtype=torch.float32).scatter_(
            -1, greedy_ids, 0.0,
        )
    scaled = _scale_sampling_logits(logits, temperature, _legacy_arithmetic=_legacy_arithmetic)
    if top_p == 1:
        if 0 < top_k < logits.shape[-1]:
            threshold = scaled.topk(top_k, dim=-1).values[..., -1:]
            return scaled.masked_fill(scaled < threshold, -torch.inf)
        return scaled

    sorted_logits, indices = _sorted_sampling_logits(
        scaled, top_k=top_k, top_p=top_p, _legacy_arithmetic=_legacy_arithmetic,
    )
    return torch.empty_like(scaled).scatter_(-1, indices, sorted_logits)


def sampling_log_probs(logits, *, temperature=1.0, top_k=-1, top_p=1.0):
    return sampling_logits(
        logits, temperature=temperature, top_k=top_k, top_p=top_p,
    ).log_softmax(dim=-1)


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._") or "model"


def target_model_cache_id(target_model: str | Path) -> str:
    """Return a cache ID from the model basename and a hash of its full path.

    The hash covers the resolved path (or the raw identifier when it is not
    an existing or absolute path), so
    models with a common basename get distinct IDs. Model weights are neither
    read nor downloaded; a model replaced in place needs a new path or
    explicit cache removal.
    """
    path = Path(target_model).expanduser()
    identity = str(path.resolve()) if path.exists() or path.is_absolute() else str(target_model)
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"{_safe_component(path.name)}--{digest}"


def sampling_cache_key(*, temperature: float, top_k: int, top_p: float) -> str:
    validate_sampling_parameters(temperature, top_k, top_p)

    def scalar(value):
        # repr preserves distinct float parameters; normalizing integral values
        # also makes T=1 and T=1.0 share the same cache.
        value = float(value)
        return (str(int(value)) if value.is_integer() else repr(value)).replace(".", "p")

    return f"T{scalar(temperature)}__topP{scalar(top_p)}__topK{'all' if top_k == -1 else top_k}"
