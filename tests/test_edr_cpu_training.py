"""Exercise CPU DP/PPS through both actual EDR training dispatch paths."""

from copy import deepcopy
from types import MethodType
from unittest import mock

import pytest
import torch
from torch import nn

import angelspec.models.dflash as dflash
from angelspec.models.ops import edr


class TinyDraft(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_token_id = 22
        self.embed_tokens = nn.Embedding(23, 8)
        self.proj = nn.Linear(8, 8, bias=False)
        self.lm_head = nn.Linear(8, 23, bias=False)

    def extract_context_feature(self, hidden_states_list):
        return hidden_states_list[0]

    def forward(self, *, noise_embedding, **kwargs):
        return self.proj(noise_embedding)


def legacy_query(self, entry, dynamic_program, device):
    """Reference query preparation that keeps the full DP on the training device."""
    if self.edr_full_anchor_backprop:
        indices = torch.arange(entry.horizon.ordinary_length, device=device)
        probabilities, inverse_scale = None, None
    else:
        sample = edr.sample_edr_round_starts(
            dynamic_program.round_start_probabilities[:entry.horizon.ordinary_length],
            self.num_anchors,
        )
        indices = sample.indices
        probabilities = sample.selected_inclusion_probabilities
        inverse_scale = sample.inverse_pps_scale
    if not indices.numel():
        return None
    return dflash._EDRGradientQuery(
        entry, dynamic_program, indices, probabilities,
        torch.ones_like(indices, dtype=torch.bool), inverse_scale,
    )


@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize("mode", ["sampled", "full", "eval"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cpu_sampling_matches_original_training_route(rows, mode, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(201)
    model = dflash.DFlashModel(
        TinyDraft(), block_size=7, num_anchors=2, loss_objective="edr",
        edr_chunk_size=8, edr_vocab_chunk_size=11,
        query_includes_input_anchor=True, edr_full_anchor_backprop=mode == "full",
        edr_stop_token_ids=[1, 21],
    ).to(device)
    reference = deepcopy(model)
    reference._prepare_edr_gradient_query = MethodType(legacy_query, reference)
    mask = torch.tensor([
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    ], dtype=torch.float32, device=device)[:rows]
    batch = dict(
        input_ids=torch.randint(0, 23, (rows, 14), device=device),
        hidden_states_list=[torch.randn(rows, 14, 8, device=device)],
        loss_mask=mask, attention_mask=torch.ones(rows, 14, device=device),
        lm_head_weight=torch.randn(23, 8, device=device),
        last_hidden_states=torch.randn(rows, 14, 8, device=device), target_norm=nn.Identity(),
    )
    sampler_inputs = []
    prepared_queries = []

    def sample(round_starts, *args, **kwargs):
        sampler_inputs.append(round_starts.device.type)
        return edr.sample_edr_round_starts(round_starts, *args, **kwargs)

    original_query = model._prepare_edr_gradient_query

    def query(*args, **kwargs):
        result = original_query(*args, **kwargs)
        prepared_queries.append(result)
        return result

    model._prepare_edr_gradient_query = query
    with torch.set_grad_enabled(mode != "eval"):
        torch.manual_seed(202)
        with (
            mock.patch.object(dflash, "exact_edr_dynamic_programs", wraps=edr.exact_edr_dynamic_programs) as dp,
            mock.patch.object(dflash, "sample_edr_round_starts", side_effect=sample),
        ):
            actual = model(**batch)
        assert dp.call_count > 0
        assert all(call.kwargs["return_on_cpu"] == (mode == "sampled") for call in dp.call_args_list)
        if mode == "sampled":
            assert sampler_inputs and set(sampler_inputs) == {"cpu"}
            for result in prepared_queries:
                assert result.dynamic_program.occupancies.device.type == "cpu"
                assert result.inclusion_probabilities.device.type == "cpu"
                assert result.coefficients.weights.device.type == device
                assert result.prefixes.device.type == device
        else:
            assert not sampler_inputs

        def old_dp(*args, **kwargs):
            kwargs["return_on_cpu"] = False
            return edr.exact_edr_dynamic_programs(*args, **kwargs)

        torch.manual_seed(202)
        with mock.patch.object(dflash, "exact_edr_dynamic_programs", side_effect=old_dp):
            expected = reference(**batch)
        for value, original in zip(actual[:5], expected[:5], strict=True):
            torch.testing.assert_close(value, original)
        for key in expected[5]:
            torch.testing.assert_close(actual[5][key], expected[5][key])
        if mode != "eval":
            actual[0].backward()
            expected[0].backward()
            for parameter, original in zip(model.parameters(), reference.parameters(), strict=True):
                if original.grad is None:
                    assert parameter.grad is None
                else:
                    torch.testing.assert_close(parameter.grad, original.grad)
