"""In-process frozen target features for single-GPU DFly and DSpark training."""

from __future__ import annotations

import operator
from collections.abc import Sequence

import torch
from torch import nn


class LocalTargetFeatures:
    """Extract the same Qwen3 residual streams as the inference connector.

    ``layer_ids`` identify zero-based decoder-layer *outputs*, not the entries
    of Hugging Face's ``hidden_states`` tuple. ``last_hidden_states`` is the
    residual immediately before the final RMSNorm. The frozen ``norm`` and
    ``lm_head`` are exposed so the draft trainer can share them.

    The target and draft run sequentially on the caller's device. Hooks capture
    only during this object's call, so draft-side calls to the shared norm store
    nothing here. Calls are not reentrant. Target logits are computed by the
    training loss from ``last_hidden_states``.
    """

    def __init__(
        self,
        target_model: nn.Module,
        layer_ids: Sequence[int],
    ) -> None:
        config = getattr(target_model, "config", None)
        if getattr(config, "model_type", None) != "qwen3":
            raise ValueError("Local target features support only a Hugging Face Qwen3 target")
        backbone = getattr(target_model, "model", None)
        layers = getattr(backbone, "layers", None)
        norm = getattr(backbone, "norm", None)
        lm_head = getattr(target_model, "lm_head", None)
        if (
            not isinstance(backbone, nn.Module)
            or not isinstance(layers, nn.ModuleList)
            or not isinstance(norm, nn.Module)
            or not isinstance(lm_head, nn.Linear)
            or not hasattr(backbone, "embed_tokens")
        ):
            raise ValueError(
                "Local target features require Qwen3ForCausalLM with model.layers, "
                "model.norm, model.embed_tokens and a dense lm_head"
            )
        try:
            layer_ids = tuple(operator.index(index) for index in layer_ids)
        except TypeError as exc:
            raise ValueError("Target layer IDs must be integers") from exc
        if not layer_ids or len(set(layer_ids)) != len(layer_ids):
            raise ValueError("Target layer IDs must be non-empty and unique")
        for layer_id in layer_ids:
            if not 0 <= layer_id < len(layers):
                raise ValueError(
                    f"Target layer {layer_id} is out of bounds for {len(layers)} layers"
                )
        if lm_head.in_features != config.hidden_size:
            raise ValueError("Target LM-head input dimension does not match hidden_size")

        self.target_model = target_model.eval().requires_grad_(False)
        self.norm = norm
        self.lm_head = lm_head
        self.layer_ids = layer_ids
        self._captured: dict[int, torch.Tensor] | None = None
        self._final_norm_input: torch.Tensor | None = None
        self._closed = False
        self._handles = []
        try:
            for layer_id in self.layer_ids:
                self._handles.append(
                    layers[layer_id].register_forward_hook(self._layer_hook(layer_id))
                )
            self._handles.append(norm.register_forward_pre_hook(self._norm_hook))
        except Exception:
            self.close()
            raise

    def _layer_hook(self, layer_id: int):
        def capture(_module, _inputs, output):
            if self._captured is not None:
                self._captured[layer_id] = output[0] if isinstance(output, tuple) else output

        return capture

    def _norm_hook(self, _module, inputs) -> None:
        if self._captured is not None:
            if not inputs:
                raise RuntimeError("Target final norm received no hidden-state input")
            self._final_norm_input = inputs[0]

    @torch.no_grad()
    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Do not use inference_mode: these tensors subsequently enter the
        # trainable draft's autograd graph and must be saveable for backward.
        if self._closed:
            raise RuntimeError("Local target features have been closed")
        if self._captured is not None:
            raise RuntimeError("Local target feature extraction is not reentrant")
        if input_ids.ndim != 2 or min(input_ids.shape) < 1:
            raise ValueError("input_ids must have non-empty shape [batch, sequence]")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must contain integer token IDs")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids [batch, sequence]")
        if attention_mask.device != input_ids.device:
            raise ValueError("attention_mask and input_ids must be on the same device")
        if (
            self.target_model.model.embed_tokens.weight.device != input_ids.device
            or self.lm_head.weight.device != input_ids.device
        ):
            raise ValueError("Target model and input_ids must be on the same device")

        self.target_model.eval()
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        self._captured = {}
        self._final_norm_input = None
        try:
            output = self.target_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=True,
            )
            missing = [index for index in self.layer_ids if index not in self._captured]
            if missing:
                raise RuntimeError(f"Target forward did not capture DFly layers: {missing}")
            if self._final_norm_input is None:
                raise RuntimeError("Target forward did not reach the final normalization layer")
            states = [self._captured[index] for index in self.layer_ids]
            last_hidden_states = self._final_norm_input
            normalized = output.last_hidden_state
            expected_shape = (*input_ids.shape, self.target_model.config.hidden_size)
            for state in (*states, last_hidden_states, normalized):
                if not isinstance(state, torch.Tensor) or tuple(state.shape) != expected_shape:
                    raise RuntimeError("Captured target hidden states do not match input shape")
                if state.device != input_ids.device:
                    raise RuntimeError("Captured target hidden states are not on the input device")

            return {
                "hidden_states": torch.cat(states, dim=-1),
                "last_hidden_states": last_hidden_states,
            }
        finally:
            self._captured = None
            self._final_norm_input = None

    def close(self) -> None:
        """Remove hooks and release captured references; safe to call twice."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._captured = None
        self._final_norm_input = None
        self._closed = True
