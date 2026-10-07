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

"""DFlash draft model: block-diffusion drafter with dual-source KV injection.

Architecture overview:
  - W_proj projects concatenated multi-layer target hidden states into context features
  - Each decoder layer uses dual-source KV: context KV (from target) + draft KV (from draft)
  - Bidirectional attention within each block; no inter-block attention
  - Shared embedding and LM head from target model (frozen)
"""

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import PretrainedConfig, PreTrainedModel

_FLASH_FLEX_KERNEL_OPTIONS = {"BACKEND": "FLASH"}
_SM120_FLEX_KERNEL_OPTIONS = {
    "BACKEND": "TRITON",
    "BLOCK_M": 64,
    "BLOCK_N": 64,
    "num_warps": 4,
    "num_stages": 2,
}


def _dflash_flex_kernel_options(
    device: torch.device,
) -> dict[str, object]:
    """Select a DFlash FlexAttention backend supported by the execution phase.

    FA4 does not fully support block-sparse training on SM120, so every DFlash
    sparse-attention phase uses constrained Triton there. Other architectures
    use the FLASH backend.
    """
    if device.type == "cuda":
        major, _minor = torch.cuda.get_device_capability(device)
        if major == 12:
            return dict(_SM120_FLEX_KERNEL_OPTIONS)
    return dict(_FLASH_FLEX_KERNEL_OPTIONS)


class DFlashConfig(PretrainedConfig):
    """Configuration for DFlash draft model."""

    model_type = "dflash"

    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_hidden_layers: int = 1,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        vocab_size: int = 152064,
        rms_norm_eps: float = 1e-6,
        max_position_embeddings: int = 32768,
        rope_theta: float = 10000.0,
        num_target_layers: int = 5,
        target_hidden_size: int = 4096,
        target_num_hidden_layers: int = 36,
        target_layer_ids: Optional[List[int]] = None,
        mask_token_id: int = 151669,
        tie_word_embeddings: bool = False,
        model_arch: str = "dflash",
        fusion_type: str = "concat_fc",
        gate_temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.num_target_layers = num_target_layers
        self.target_hidden_size = target_hidden_size
        self.target_num_hidden_layers = target_num_hidden_layers
        self.target_layer_ids = target_layer_ids
        self.mask_token_id = mask_token_id
        # Selects the draft model class within the shared DFlash training stack:
        # "dflash" -> DFlashDraftModel, "dflare" -> DFlareDraftModel. DFlare reuses
        # DFlashConfig wholesale (see dflare.py header), so the architecture choice
        # rides on this field rather than a separate config class.
        self.model_arch = model_arch
        # Context fusion mechanism: "concat_fc" = DFlash's concat -> Linear (default,
        # unchanged); "gated_sum" = per-candidate RMSNorm -> softmax gate -> weighted
        # sum for learnable layer selection (DFlashGatedDraftModel). gate_temperature
        # only applies to gated_sum.
        self.fusion_type = fusion_type
        self.gate_temperature = gate_temperature


class DFlashRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class DFlashRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 32768, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # +20 buffer avoids cache rebuild if sequences slightly exceed the configured limit.
        self._set_cos_sin_cache(max_position_embeddings + 20, self.inv_freq.device, torch.float32)

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len and seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# NOTE: Not called in DFlashAttention.forward() (RoPE is applied inline there),
# but kept as a standard utility.
def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


@dataclass(frozen=True)
class DFlashAttentionContextCache:
    """Anchor-independent context K/V after K-norm and context RoPE."""

    key: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class DFlashContextCache:
    """Per-layer context K/V reused by independent DFlash query blocks."""

    layer_caches: tuple[DFlashAttentionContextCache, ...]
    hidden_dtype: torch.dtype


class DFlashAttention(nn.Module):
    """Dual-source KV attention for DFlash.

    K/V come from two sources concatenated along the sequence dimension:
      1. Context KV: projected from target model's context features (via shared W_k/W_v)
      2. Draft KV: projected from draft model's own hidden states (via same W_k/W_v)

    Q comes only from the draft model's hidden states.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 32768)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Q-norm and K-norm (Qwen3 architecture requirement)
        self.q_norm = DFlashRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = DFlashRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rotary_emb = DFlashRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=getattr(config, "rope_theta", 10000.0),
        )

    def prepare_context_cache(
        self,
        context_hidden: torch.Tensor,
        context_position_ids: torch.Tensor,
        rope_sequence_length: Optional[int] = None,
    ) -> DFlashAttentionContextCache:
        """Project the context side once for reuse by independent draft blocks."""
        if context_hidden.ndim != 3:
            raise ValueError("context_hidden must have shape [batch, sequence, hidden]")
        if context_position_ids.shape != context_hidden.shape[:2]:
            raise ValueError("context_position_ids must match the context batch and sequence")
        if context_position_ids.device != context_hidden.device:
            raise ValueError("context positions and hidden states must be on the same device")
        if context_position_ids.dtype == torch.bool or context_position_ids.is_floating_point():
            raise TypeError("context_position_ids must use an integer dtype")
        if (
            rope_sequence_length is None
            and context_position_ids.numel()
            and bool((context_position_ids < 0).any())
        ):
            raise ValueError("context_position_ids must be non-negative")

        bsz, ctx_len, _ = context_hidden.shape
        key = self.k_proj(context_hidden)
        key = key.view(bsz, ctx_len, self.num_kv_heads, self.head_dim)
        key = self.k_norm(key).transpose(1, 2)
        value = self.v_proj(context_hidden)
        value = value.view(bsz, ctx_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # A context-cache append can contain only a suffix while retaining the
        # suffix's absolute positions.  Size the RoPE view for the largest
        # position rather than assuming every cache segment starts at zero.
        rope_len = ctx_len if rope_sequence_length is None else int(rope_sequence_length)
        if rope_len < ctx_len:
            raise ValueError("rope_sequence_length cannot be shorter than the context segment")
        if rope_sequence_length is None and context_position_ids.numel():
            rope_len = max(rope_len, int(context_position_ids.max().item()) + 1)
        cos, sin = self.rotary_emb(key, seq_len=rope_len)
        cos = cos.to(key.device)
        sin = sin.to(key.device)
        cos_key = cos.squeeze(1).squeeze(0)[context_position_ids].unsqueeze(1)
        sin_key = sin.squeeze(1).squeeze(0)[context_position_ids].unsqueeze(1)
        key = (key * cos_key) + (_rotate_half(key) * sin_key)
        return DFlashAttentionContextCache(key=key, value=value)

    def forward(
        self,
        draft_hidden: torch.Tensor,
        context_hidden: Optional[torch.Tensor],
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask=None,
        context_cache: Optional[DFlashAttentionContextCache] = None,
    ) -> torch.Tensor:
        """Forward pass with dual-source KV.

        Args:
            draft_hidden: [B, draft_len, D] — hidden states of draft tokens
            context_hidden: [B, ctx_len, D] — context features from target
            draft_position_ids: [B, draft_len] — position IDs for draft tokens
            context_position_ids: [B, ctx_len] — position IDs for context tokens
            block_mask: FlexAttention BlockMask for block-causal attention
        """
        bsz, draft_len, _ = draft_hidden.shape
        if context_cache is None:
            if context_hidden is None:
                raise ValueError("context_hidden is required when context_cache is not provided")
            ctx_len = context_hidden.shape[1]
        else:
            if context_hidden is not None:
                raise ValueError("provide either context_hidden or context_cache, not both")
            key_ctx, value_ctx = context_cache.key, context_cache.value
            if key_ctx.device != draft_hidden.device or value_ctx.device != draft_hidden.device:
                raise ValueError("cached context K/V and draft hidden states must share a device")
            if key_ctx.shape != value_ctx.shape:
                raise ValueError("cached context K/V must have equal shapes")
            if key_ctx.ndim != 4 or key_ctx.shape[0] != bsz:
                raise ValueError("cached context K/V must have shape [batch, heads, sequence, dim]")
            if key_ctx.shape[1] != self.num_kv_heads or key_ctx.shape[3] != self.head_dim:
                raise ValueError("cached context K/V has incompatible attention dimensions")
            ctx_len = key_ctx.shape[2]

        # Q only from draft
        q = self.q_proj(draft_hidden)
        q = q.view(bsz, draft_len, self.num_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)  # [B, num_heads, draft_len, head_dim]

        if context_cache is None:
            # Without a cache, concatenate context and draft K before the
            # tokenwise K-norm and RoPE.
            key_ctx = self.k_proj(context_hidden)
            value_ctx = self.v_proj(context_hidden)
            key_draft = self.k_proj(draft_hidden)
            value_draft = self.v_proj(draft_hidden)
            key = torch.cat([key_ctx, key_draft], dim=1)
            value = torch.cat([value_ctx, value_draft], dim=1)

            total_len = ctx_len + draft_len
            key = key.view(bsz, total_len, self.num_kv_heads, self.head_dim)
            key = self.k_norm(key).transpose(1, 2)
            value = value.view(bsz, total_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

            full_position_ids = torch.cat(
                [context_position_ids, draft_position_ids],
                dim=1,
            )
            cos, sin = self.rotary_emb(q, seq_len=total_len)
            cos = cos.to(q.device)
            sin = sin.to(q.device)
            cos_query = cos.squeeze(1).squeeze(0)[draft_position_ids].unsqueeze(1)
            sin_query = sin.squeeze(1).squeeze(0)[draft_position_ids].unsqueeze(1)
            q = (q * cos_query) + (_rotate_half(q) * sin_query)
            cos_key = cos.squeeze(1).squeeze(0)[full_position_ids].unsqueeze(1)
            sin_key = sin.squeeze(1).squeeze(0)[full_position_ids].unsqueeze(1)
            key = (key * cos_key) + (_rotate_half(key) * sin_key)
        else:
            key_draft = self.k_proj(draft_hidden)
            value_draft = self.v_proj(draft_hidden)
            key_draft = key_draft.view(
                bsz,
                draft_len,
                self.num_kv_heads,
                self.head_dim,
            )
            key_draft = self.k_norm(key_draft).transpose(1, 2)
            value_draft = value_draft.view(
                bsz,
                draft_len,
                self.num_kv_heads,
                self.head_dim,
            ).transpose(1, 2)

            total_len = ctx_len + draft_len
            cos, sin = self.rotary_emb(q, seq_len=total_len)
            cos = cos.to(q.device)
            sin = sin.to(q.device)
            cos_query = cos.squeeze(1).squeeze(0)[draft_position_ids].unsqueeze(1)
            sin_query = sin.squeeze(1).squeeze(0)[draft_position_ids].unsqueeze(1)
            q = (q * cos_query) + (_rotate_half(q) * sin_query)

            # K-norm and RoPE are tokenwise, so the cached context K (normalized
            # and rotated in advance) equals the uncached result.
            cos_draft = cos.squeeze(1).squeeze(0)[draft_position_ids].unsqueeze(1)
            sin_draft = sin.squeeze(1).squeeze(0)[draft_position_ids].unsqueeze(1)
            key_draft = (key_draft * cos_draft) + (_rotate_half(key_draft) * sin_draft)
            key = torch.cat([key_ctx, key_draft], dim=2)
            value = torch.cat([value_ctx, value_draft], dim=2)

        if block_mask is not None:
            from angelspec.models.ops.flex_attention import (
                compile_friendly_flex_attention,
            )

            # Use enable_gqa=True to let FlexAttention handle GQA internally
            # instead of materializing expanded KV via _repeat_kv
            attn_output = compile_friendly_flex_attention(
                query=q,
                key=key,
                value=value,
                block_mask=block_mask,
                enable_gqa=True,
                kernel_options=_dflash_flex_kernel_options(
                    q.device,
                ),
            )
        else:
            # Fallback: bidirectional attention (no mask) — expand KV for SDPA
            key = _repeat_kv(key, self.num_kv_groups)
            value = _repeat_kv(value, self.num_kv_groups)
            attn_output = F.scaled_dot_product_attention(
                q,
                key,
                value,
                is_causal=False,
                dropout_p=0.0,
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, draft_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


class DFlashMLP(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlashDecoderLayer(nn.Module):
    """Single transformer decoder layer for DFlash draft model."""

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.self_attn = DFlashAttention(config)
        self.mlp = DFlashMLP(config)
        self.input_layernorm = DFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        draft_hidden: torch.Tensor,
        context_hidden: Optional[torch.Tensor],
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask=None,
        context_cache: Optional[DFlashAttentionContextCache] = None,
    ) -> torch.Tensor:
        residual = draft_hidden
        draft_hidden = self.input_layernorm(draft_hidden)

        draft_hidden = self.self_attn(
            draft_hidden=draft_hidden,
            context_hidden=context_hidden,
            draft_position_ids=draft_position_ids,
            context_position_ids=context_position_ids,
            block_mask=block_mask,
            context_cache=context_cache,
        )
        draft_hidden = residual + draft_hidden

        residual = draft_hidden
        draft_hidden = self.post_attention_layernorm(draft_hidden)
        draft_hidden = self.mlp(draft_hidden)
        draft_hidden = residual + draft_hidden

        return draft_hidden


def build_target_layer_ids(num_target_layers: int, num_hidden_layers: int) -> List[int]:
    """Compute uniformly spaced layer IDs from the target model.

    Uses evenly-spaced layer IDs:
      start = 1, end = num_hidden_layers - 3, span = end - start
      For num_target_layers=5 and num_hidden_layers=36:
        start=1, end=33, span=32
        → [1, 9, 17, 25, 33]

    Note: num_target_layers here is the number of layers
    to capture. num_hidden_layers is the
    total number of target model decoder layers.
    """
    if num_target_layers == 1:
        return [num_hidden_layers // 2]
    start = 1
    end = num_hidden_layers - 3
    span = end - start
    return [
        int(round(start + (i * span) / (num_target_layers - 1))) for i in range(num_target_layers)
    ]


class DFlashDraftModel(PreTrainedModel):
    """DFlash draft model with dual-source KV injection.

    Trainable parameters:
      - W_proj: Linear(num_target_layers * target_hidden_size, hidden_size)
      - proj_norm: RMSNorm after projection
      - N decoder layers (each with attention + FFN)

    Frozen (from target):
      - embed_tokens: token embedding
      - LM head is external (loaded separately in trainer)
    """

    config_class = DFlashConfig
    supports_context_cache = True

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_hidden_layers

        target_hidden_size = getattr(config, "target_hidden_size", config.hidden_size)
        num_target_layers = getattr(config, "num_target_layers", 5)
        self.num_target_layers = num_target_layers
        self.mask_token_id = getattr(config, "mask_token_id", 151669)

        # Target layer IDs for hidden state extraction
        target_num_hidden = getattr(config, "target_num_hidden_layers", 36)
        self.target_layer_ids = getattr(config, "target_layer_ids", None)
        if self.target_layer_ids is None:
            self.target_layer_ids = build_target_layer_ids(num_target_layers, target_num_hidden)

        # Context feature projection: concat(multi-layer hidden) → hidden_size
        proj_input_dim = num_target_layers * target_hidden_size
        self.context_proj = nn.Linear(proj_input_dim, self.hidden_size, bias=False)
        self.context_norm = DFlashRMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        # Token embedding (shared from target, frozen)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList([DFlashDecoderLayer(config) for _ in range(self.num_layers)])

        self.final_norm = DFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def extract_context_feature(self, all_hidden_states: List[torch.Tensor]) -> torch.Tensor:
        """Extract and project context features from target hidden states.

        Args:
            all_hidden_states: list of [B, seq_len, D] tensors from target layers

        Returns:
            context_feature: [B, seq_len, hidden_size]
        """
        concatenated = torch.cat(all_hidden_states, dim=-1).to(self.context_proj.weight.dtype)
        projected = self.context_proj(concatenated)
        return self.context_norm(projected)

    def prepare_context_cache(
        self,
        context_feature: torch.Tensor,
        context_position_ids: torch.Tensor,
        rope_sequence_length: Optional[int] = None,
    ) -> DFlashContextCache:
        """Build anchor-independent context K/V once for repeated EDR queries."""
        if context_feature.ndim != 3:
            raise ValueError("DFlash context_feature must have shape [B, S, D]")
        if context_position_ids.shape != context_feature.shape[:2]:
            raise ValueError("context_position_ids must match the DFlash context shape")

        return DFlashContextCache(
            layer_caches=tuple(
                layer.self_attn.prepare_context_cache(
                    context_feature,
                    context_position_ids,
                    rope_sequence_length=rope_sequence_length,
                )
                for layer in self.layers
            ),
            hidden_dtype=context_feature.dtype,
        )

    def forward_with_context_cache(
        self,
        *,
        draft_input_ids: Optional[torch.Tensor],
        context_cache: DFlashContextCache,
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask=None,
        noise_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run anchor-dependent draft work against prepared context K/V."""
        if len(context_cache.layer_caches) != len(self.layers):
            raise ValueError("DFlash context cache does not match the number of draft layers")
        if noise_embedding is not None:
            draft_hidden = noise_embedding.to(context_cache.hidden_dtype)
        else:
            draft_hidden = self.embed_tokens(draft_input_ids).to(context_cache.hidden_dtype)

        for layer, layer_cache in zip(self.layers, context_cache.layer_caches):
            draft_hidden = layer(
                draft_hidden=draft_hidden,
                context_hidden=None,
                draft_position_ids=draft_position_ids,
                context_position_ids=context_position_ids,
                block_mask=block_mask,
                context_cache=layer_cache,
            )

        return self.final_norm(draft_hidden)

    def forward(
        self,
        draft_input_ids: Optional[torch.Tensor],
        context_feature: torch.Tensor,
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask=None,
        noise_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through draft model.

        Args:
            draft_input_ids: [B, draft_len] — token IDs (anchor + MASK tokens).
                Ignored if noise_embedding is provided.
            context_feature: [B, ctx_len, D] — projected context from target
            draft_position_ids: [B, draft_len]
            context_position_ids: [B, ctx_len]
            block_mask: FlexAttention BlockMask
            noise_embedding: [B, draft_len, D] — pre-computed embeddings (from training wrapper)

        Returns:
            hidden_states: [B, draft_len, D] — pre-norm hidden states
        """
        if noise_embedding is not None:
            draft_hidden = noise_embedding.to(context_feature.dtype)
        else:
            draft_hidden = self.embed_tokens(draft_input_ids).to(context_feature.dtype)

        for layer in self.layers:
            draft_hidden = layer(
                draft_hidden=draft_hidden,
                context_hidden=context_feature,
                draft_position_ids=draft_position_ids,
                context_position_ids=context_position_ids,
                block_mask=block_mask,
            )

        return self.final_norm(draft_hidden)

    def freeze_embedding(self) -> None:
        self.embed_tokens.weight.requires_grad = False

    @torch.no_grad()
    def load_embedding(
        self, model_path: str, embedding_key: str = "model.embed_tokens.weight"
    ) -> None:
        """Load embedding weights from target model checkpoint."""
        if os.path.exists(model_path):
            glob_path = os.path.join(model_path, "*.index.json")
            import glob as glob_mod

            index_json_path = glob_mod.glob(glob_path)

            if len(index_json_path) == 0:
                safetensors_path = os.path.join(model_path, "model.safetensors")
                if os.path.exists(safetensors_path):
                    with safe_open(safetensors_path, framework="pt") as f:
                        self.embed_tokens.weight.copy_(f.get_tensor(embedding_key))
                    return
                pytorch_model_path = os.path.join(model_path, "pytorch_model.bin")
                if os.path.exists(pytorch_model_path):
                    state_dict = torch.load(
                        pytorch_model_path, map_location="cpu", weights_only=True
                    )
                    self.embed_tokens.weight.copy_(state_dict[embedding_key])
                    return
                raise FileNotFoundError(
                    f"No index.json, model.safetensors or pytorch_model.bin found in {model_path}"
                )
            index_json_path = index_json_path[0]
            with open(index_json_path, "r") as f:
                index_json = json.load(f)
            ckpt_file = index_json["weight_map"][embedding_key]
            if ckpt_file.endswith(".safetensors"):
                with safe_open(os.path.join(model_path, ckpt_file), framework="pt") as f:
                    self.embed_tokens.weight.copy_(f.get_tensor(embedding_key))
            else:
                state_dict = torch.load(
                    os.path.join(model_path, ckpt_file), map_location="cpu", weights_only=True
                )
                self.embed_tokens.weight.copy_(state_dict[embedding_key])
        else:
            local_cache_path = snapshot_download(repo_id=model_path)
            self.load_embedding(local_cache_path, embedding_key)
