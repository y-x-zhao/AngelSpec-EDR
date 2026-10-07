"""CPU tests for the exact Expected Decoding Rounds objective primitives."""

import math
import unittest
from argparse import Namespace
from copy import deepcopy
from dataclasses import replace
from unittest import mock

import torch
import torch.nn as nn

import angelspec.models.ops.edr as edr_ops
from angelspec.models.dflash import DFlashModel, _edr_gradient_anchor_slots
from angelspec.models.dfly import DFlyModel
from angelspec.models.dspark import DSparkModel
from angelspec.models.ops.dflash_layout import build_dflash_proposal_layout
from angelspec.models.ops.edr import (
    edr_anchor_inclusion_probabilities,
    edr_distribution_statistics,
    edr_distribution_statistics_from_target_probabilities,
    edr_surrogate_sum,
    exact_edr_dynamic_program,
    exact_edr_dynamic_programs,
    extract_edr_horizons,
    greedy_edr_distribution_statistics,
    prepare_edr_target_distribution,
    sample_edr_round_starts,
    sampled_edr_surrogate_sum,
    streaming_edr_distribution_statistics,
)


class _TinyDraft(nn.Module):
    """Minimal DFlash-compatible draft used to exercise the EDR model path."""

    def __init__(self, hidden_size=8, vocab_size=16):
        super().__init__()
        self.mask_token_id = vocab_size - 1
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def extract_context_feature(self, hidden_states_list):
        return hidden_states_list[0]

    def forward(
        self,
        *,
        draft_input_ids,
        context_feature,
        draft_position_ids,
        context_position_ids,
        block_mask,
        noise_embedding,
    ):
        del (
            draft_input_ids,
            context_feature,
            draft_position_ids,
            context_position_ids,
            block_mask,
        )
        return self.proj(noise_embedding)


class _TinyCorrection(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden, previous_embedding):
        return hidden + self.proj(previous_embedding)


class _TinyMarkov(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def apply_block_logits(self, logits, token_ids):
        return logits + self.proj(self.embedding(token_ids))


def _edr_model_and_batch(
    chunk_size,
    *,
    query_includes_input_anchor=False,
    block_size=None,
    full_anchor_backprop=False,
):
    if block_size is None:
        # Three physical query slots either way; the input anchor uses one.
        block_size = 2 if query_includes_input_anchor else 3
    torch.manual_seed(10)
    hidden_size, vocab_size = 8, 16
    model = DFlashModel(
        _TinyDraft(hidden_size, vocab_size),
        block_size=block_size,
        num_anchors=2,
        loss_objective="edr",
        edr_chunk_size=chunk_size,
        edr_full_anchor_backprop=full_anchor_backprop,
        query_includes_input_anchor=query_includes_input_anchor,
    )
    generator = torch.Generator().manual_seed(11)
    input_ids = torch.randint(0, vocab_size, (1, 9), generator=generator)
    batch = {
        "input_ids": input_ids,
        "hidden_states_list": [torch.randn(1, 9, hidden_size, generator=generator)],
        "loss_mask": torch.tensor([[0, 1, 1, 1, 1, 1, 1, 0, 0]], dtype=torch.float32),
        "attention_mask": torch.ones(1, 9),
        "lm_head_weight": torch.randn(vocab_size, hidden_size, generator=generator),
        "last_hidden_states": torch.randn(1, 9, hidden_size, generator=generator),
        "target_norm": nn.Identity(),
    }
    return model, batch


def _differentiable_reference(costs, acceptance, block_size):
    """Small recursive Bellman reference used only to check Eq. (14)."""
    length = costs.shape[0]
    round_values = [None] * (length + 1)
    round_values[length] = costs.new_zeros(())
    for prefix in range(length - 1, -1, -1):
        exhaustion_position = prefix + block_size + 1
        if exhaustion_position <= length:
            continuation = 1.0 + round_values[exhaustion_position]
        else:
            continuation = costs.new_zeros(())
        for offset in range(min(block_size, length - prefix), 0, -1):
            column = offset - 1
            position = prefix + offset
            accept = acceptance[prefix, column]
            continuation = (
                costs[prefix, column]
                + accept * continuation
                + (1.0 - accept) * round_values[position]
            )
        round_values[prefix] = continuation
    return 1.0 + round_values[0]


def _exhaustive_path_reference(costs, acceptance, block_size):
    """Enumerate every reject/accept/exhaustion branch for a short horizon."""

    length = costs.shape[0]
    occupancies = torch.zeros_like(costs)
    round_starts = torch.zeros(length + 1, dtype=costs.dtype)
    additional_passes = costs.new_zeros(())

    def visit_round(prefix: int, probability: torch.Tensor) -> None:
        nonlocal additional_passes
        round_starts[prefix] += probability
        if prefix == length:
            return
        surviving = probability
        for offset in range(1, min(block_size, length - prefix) + 1):
            column = offset - 1
            position = prefix + offset
            occupancies[prefix, column] += surviving
            additional_passes += surviving * costs[prefix, column]
            visit_round(position, surviving * (1.0 - acceptance[prefix, column]))
            surviving = surviving * acceptance[prefix, column]
        exhaustion_position = prefix + block_size + 1
        if exhaustion_position <= length:
            additional_passes += surviving
            visit_round(exhaustion_position, surviving)

    visit_round(0, costs.new_ones(()))
    return 1.0 + additional_passes, occupancies, round_starts


class TestEDRHorizons(unittest.TestCase):
    def test_each_span_uses_last_token_as_boundary(self):
        mask = torch.tensor([[0, 1, 1, 1, 0, 1, 0]], dtype=torch.float32)
        horizons = extract_edr_horizons(mask)
        self.assertEqual([(h.start, h.boundary) for h in horizons], [(1, 3), (5, 5)])
        self.assertEqual([h.ordinary_length for h in horizons], [2, 0])

    def test_attention_and_document_boundaries_split_runs(self):
        mask = torch.ones(1, 8)
        attention = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])
        doc_ids = torch.tensor([[0, 0, 0, 1, 1, 1, -1, -1]])
        horizons = extract_edr_horizons(mask, attention, doc_ids)
        self.assertEqual(
            [(h.start, h.boundary, h.document_id) for h in horizons],
            [(0, 2, 0), (3, 5, 1)],
        )

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            extract_edr_horizons(torch.ones(2, 3), attention_mask=torch.ones(2, 2))


class TestLearnedFirstLayout(unittest.TestCase):
    def test_anchor_is_input_and_all_block_slots_are_next_token_labels(self):
        input_ids = torch.tensor([[10, 11, 12, 13, 14]])
        layout = build_dflash_proposal_layout(
            torch.tensor([[1]]),
            sequence_length=5,
            block_size=3,
            attention_mask=torch.ones(1, 5),
            loss_mask=torch.tensor([[0, 0, 1, 1, 1]]),
        )
        self.assertEqual(layout.label_indices.tolist(), [[[2, 3, 4]]])
        self.assertEqual(layout.gather_labels(input_ids).tolist(), [[[12, 13, 14]]])
        self.assertEqual(layout.gather_predecessor_tokens(input_ids).tolist(), [[[11, 12, 13]]])
        self.assertTrue(layout.valid_mask.all())

    def test_packed_document_boundary_masks_later_slots(self):
        layout = build_dflash_proposal_layout(
            torch.tensor([[1]]),
            sequence_length=5,
            block_size=3,
            loss_mask=torch.tensor([[0, 0, 1, 1, 1]]),
            ctx_doc_ids=torch.tensor([[0, 0, 0, 1, 1]]),
        )
        self.assertEqual(layout.valid_mask.tolist(), [[[True, False, False]]])

    def test_invalid_middle_slot_cannot_reenter_the_causal_chain(self):
        layout = build_dflash_proposal_layout(
            torch.tensor([[0]]),
            sequence_length=5,
            block_size=4,
            attention_mask=torch.tensor([[1, 1, 0, 1, 1]]),
            loss_mask=torch.tensor([[0, 1, 1, 1, 1]]),
        )
        self.assertEqual(layout.valid_mask.tolist(), [[[True, False, False, False]]])


class TestEDRStatistics(unittest.TestCase):
    def test_tv_and_realized_acceptance(self):
        draft = torch.log(torch.tensor([[0.2, 0.3, 0.5]], requires_grad=True))
        target = torch.log(torch.tensor([[0.4, 0.2, 0.4]]))
        cost, acceptance = edr_distribution_statistics(draft, target, torch.tensor([0]))
        self.assertTrue(torch.allclose(cost, torch.tensor([0.2]), atol=1e-6))
        self.assertTrue(torch.allclose(acceptance, torch.tensor([0.5]), atol=1e-6))
        self.assertTrue(cost.requires_grad)
        self.assertTrue(acceptance.requires_grad)

    def test_precomputed_target_probabilities_match_target_logits(self):
        draft = torch.randn(2, 3, requires_grad=True)
        target = torch.randn(2, 3)
        target_ids = torch.tensor([0, 2])
        expected = edr_distribution_statistics(draft, target, target_ids)
        actual = edr_distribution_statistics_from_target_probabilities(
            draft,
            torch.softmax(target.float(), dim=-1),
            target_ids,
        )
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])

    def test_conditional_non_stopping_rejection_cost(self):
        target_probabilities = torch.tensor([[0.30, 0.10, 0.20, 0.25, 0.15]])
        draft_probabilities = torch.tensor([[0.10, 0.30, 0.15, 0.10, 0.35]])
        draft_logits = draft_probabilities.log().requires_grad_()

        costs, _ = edr_distribution_statistics_from_target_probabilities(
            draft_logits,
            target_probabilities,
            torch.tensor([0]),
            stopping_token_ids=[1, 4],
        )

        # Non-stop numerator = (0.30-0.10) + (0.20-0.15) + (0.25-0.10).
        # Target non-stop probability = 1 - (0.10+0.15).
        torch.testing.assert_close(costs, torch.tensor([(0.20 + 0.05 + 0.15) / 0.75]))

    def test_streamed_conditional_cost_and_gradient_match_dense_reference(self):
        torch.manual_seed(36)
        target_logits = torch.randn(4, 11)
        target_indices = torch.tensor([[0, 1, 2], [1, 2, 3]])
        target_ids = torch.tensor([[0, 2, 4], [6, 8, 10]])
        stop_ids = [1, 9]
        draft_reference = torch.randn(2, 3, 11, requires_grad=True)
        draft_streamed = draft_reference.detach().clone().requires_grad_()
        expected = edr_distribution_statistics_from_target_probabilities(
            draft_reference,
            torch.softmax(target_logits.float(), dim=-1)[target_indices],
            target_ids,
            stopping_token_ids=stop_ids,
        )
        actual = streaming_edr_distribution_statistics(
            draft_streamed,
            prepare_edr_target_distribution(
                target_logits,
                4,
                stopping_token_ids=stop_ids,
            ),
            target_indices,
            target_ids,
            4,
        )
        cost_weights = torch.randn(2, 3)
        acceptance_weights = torch.randn(2, 3)
        expected_gradient = torch.autograd.grad(
            (expected[0] * cost_weights + expected[1] * acceptance_weights).sum(),
            draft_reference,
        )[0]
        actual_gradient = torch.autograd.grad(
            (actual[0] * cost_weights + actual[1] * acceptance_weights).sum(),
            draft_streamed,
        )[0]

        torch.testing.assert_close(actual[0], expected[0], atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(actual[1], expected[1], atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=3e-6, rtol=3e-6)

    def test_target_statistics_are_computed_locally(self):
        torch.manual_seed(38)
        target_logits = torch.randn(4, 13)
        with (
            mock.patch.object(
                edr_ops,
                "_streaming_log_normalizers",
                wraps=edr_ops._streaming_log_normalizers,
            ) as log_normalizers,
            mock.patch.object(
                edr_ops,
                "_target_stop_probabilities",
                wraps=edr_ops._target_stop_probabilities,
            ) as stop_probabilities,
        ):
            distribution = prepare_edr_target_distribution(
                target_logits,
                5,
                stopping_token_ids=[2, 12],
            )

        self.assertEqual(log_normalizers.call_count, 1)
        self.assertEqual(stop_probabilities.call_count, 1)
        expected_probabilities = torch.softmax(target_logits.float(), dim=-1)
        torch.testing.assert_close(
            distribution.log_normalizers,
            torch.logsumexp(target_logits.float(), dim=-1),
        )
        torch.testing.assert_close(
            distribution.stop_probabilities,
            expected_probabilities[:, [2, 12]].sum(dim=-1),
        )

    def test_zero_non_stopping_target_mass_defines_zero_cost(self):
        draft_logits = torch.randn(1, 4, requires_grad=True)
        target_probabilities = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        costs, _ = edr_distribution_statistics_from_target_probabilities(
            draft_logits,
            target_probabilities,
            torch.tensor([1]),
            stopping_token_ids=[1],
        )
        costs.sum().backward()

        torch.testing.assert_close(costs, torch.zeros_like(costs))
        torch.testing.assert_close(draft_logits.grad, torch.zeros_like(draft_logits))

    def test_acceptance_clips_at_one_and_target_is_detached(self):
        draft = torch.tensor([[0.0, 2.0]], requires_grad=True)
        target = torch.tensor([[0.0, -2.0]], requires_grad=True)
        _, acceptance = edr_distribution_statistics(draft, target, torch.tensor([1]))
        self.assertEqual(acceptance.item(), 1.0)
        acceptance.backward()
        self.assertIsNone(target.grad)

    def test_streamed_statistics_match_full_probabilities_and_gradient(self):
        torch.manual_seed(37)
        draft_reference = torch.randn(2, 2, 11, requires_grad=True)
        draft_streamed = draft_reference.detach().clone().requires_grad_()
        target_logits = torch.randn(4, 11)
        target_indices = torch.tensor([[0, 1], [1, 2]])
        target_ids = torch.tensor([[1, 7], [4, 10]])
        target_probabilities = torch.softmax(target_logits.float(), dim=-1)

        expected = edr_distribution_statistics_from_target_probabilities(
            draft_reference,
            target_probabilities[target_indices],
            target_ids,
        )
        target_distribution = prepare_edr_target_distribution(target_logits, 3)
        actual = streaming_edr_distribution_statistics(
            draft_streamed,
            target_distribution,
            target_indices,
            target_ids,
            3,
        )

        torch.testing.assert_close(actual[0], expected[0], atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(actual[1], expected[1], atol=2e-6, rtol=2e-6)
        cost_weights = torch.randn(2, 2)
        acceptance_weights = torch.randn(2, 2)
        expected_objective = (expected[0] * cost_weights + expected[1] * acceptance_weights).sum()
        actual_objective = (actual[0] * cost_weights + actual[1] * acceptance_weights).sum()
        expected_gradient = torch.autograd.grad(expected_objective, draft_reference)[0]
        actual_gradient = torch.autograd.grad(actual_objective, draft_streamed)[0]
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            atol=3e-6,
            rtol=3e-6,
        )

    def test_streamed_detached_path_skips_backward_statistics(self):
        torch.manual_seed(39)
        draft_logits = torch.randn(2, 2, 13, requires_grad=True)
        target_distribution = prepare_edr_target_distribution(torch.randn(4, 13), 5)
        target_indices = torch.tensor([[0, 2], [3, 1]])
        target_ids = torch.tensor([[1, 8], [5, 12]])
        backward_flags = []
        original_forward = edr_ops._streaming_edr_statistics_forward

        def record_forward(*args, **kwargs):
            backward_flags.append(kwargs["compute_backward_stats"])
            return original_forward(*args, **kwargs)

        with mock.patch.object(
            edr_ops,
            "_streaming_edr_statistics_forward",
            side_effect=record_forward,
        ):
            differentiable = streaming_edr_distribution_statistics(
                draft_logits,
                target_distribution,
                target_indices,
                target_ids,
                5,
            )
            with torch.no_grad():
                no_grad = streaming_edr_distribution_statistics(
                    draft_logits,
                    target_distribution,
                    target_indices,
                    target_ids,
                    5,
                )
            detached = streaming_edr_distribution_statistics(
                draft_logits.detach(),
                target_distribution,
                target_indices,
                target_ids,
                5,
            )

        self.assertEqual(backward_flags, [True, False, False])
        for differentiable_value, no_grad_value, detached_value in zip(
            differentiable,
            no_grad,
            detached,
        ):
            torch.testing.assert_close(no_grad_value, differentiable_value)
            torch.testing.assert_close(detached_value, differentiable_value)
            self.assertTrue(differentiable_value.requires_grad)
            self.assertFalse(no_grad_value.requires_grad)
            self.assertFalse(detached_value.requires_grad)

    def test_greedy_statistics_match_explicit_one_hot_distribution(self):
        torch.manual_seed(42)
        draft_logits = torch.randn(2, 3, 11, dtype=torch.bfloat16)
        target_logits = torch.randn(4, 11, dtype=torch.bfloat16)
        target_indices = torch.tensor([[0, 1, 2], [1, 2, 3]])
        target_ids = torch.tensor([[0, 2, 4], [6, 8, 10]])
        greedy_ids = draft_logits.argmax(dim=-1)
        greedy_probabilities = torch.nn.functional.one_hot(
            greedy_ids,
            num_classes=draft_logits.shape[-1],
        ).float()
        target_probabilities = torch.softmax(target_logits.float(), dim=-1)[target_indices]
        expected_costs = 0.5 * (target_probabilities - greedy_probabilities).abs().sum(dim=-1)
        expected_acceptance = greedy_ids.eq(target_ids).float()

        actual_costs, actual_acceptance = greedy_edr_distribution_statistics(
            draft_logits,
            prepare_edr_target_distribution(target_logits, 4),
            target_indices,
            target_ids,
        )

        torch.testing.assert_close(actual_costs, expected_costs, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(actual_acceptance, expected_acceptance)

    def test_greedy_conditional_cost_handles_stopping_proposals(self):
        target_probabilities = torch.tensor(
            [
                [0.20, 0.10, 0.30, 0.25, 0.15],
                [0.20, 0.10, 0.30, 0.25, 0.15],
            ]
        )
        target_logits = target_probabilities.log()
        # Row 0 greedily proposes stopping token 4. Row 1 proposes non-stop token 2.
        draft_logits = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 2.0], [0.0, 0.0, 2.0, 0.0, 0.0]]
        )
        costs, _ = greedy_edr_distribution_statistics(
            draft_logits,
            prepare_edr_target_distribution(
                target_logits,
                3,
                stopping_token_ids=[1, 4],
            ),
            torch.tensor([0, 1]),
            torch.tensor([0, 0]),
        )

        torch.testing.assert_close(costs, torch.tensor([1.0, 1.0 - 0.30 / 0.75]))

    def test_streamed_tie_gradient_matches_full_probabilities(self):
        torch.manual_seed(43)
        target_logits = torch.randn(2, 9)
        target_indices = torch.tensor([[0, 0], [1, 1]])
        target_ids = torch.tensor([[0, 3], [5, 8]])
        draft_reference = target_logits[target_indices].detach().clone().requires_grad_(True)
        draft_streamed = draft_reference.detach().clone().requires_grad_(True)
        target_probabilities = torch.softmax(target_logits.float(), dim=-1)
        expected = edr_distribution_statistics_from_target_probabilities(
            draft_reference,
            target_probabilities[target_indices],
            target_ids,
        )
        actual = streaming_edr_distribution_statistics(
            draft_streamed,
            prepare_edr_target_distribution(target_logits, 4),
            target_indices,
            target_ids,
            4,
        )
        weights = torch.randn(2, 2)
        expected_gradient = torch.autograd.grad(
            (expected[0] * weights + expected[1] * weights).sum(),
            draft_reference,
        )[0]
        actual_gradient = torch.autograd.grad(
            (actual[0] * weights + actual[1] * weights).sum(),
            draft_streamed,
        )[0]

        torch.testing.assert_close(actual[0], expected[0], atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(actual[1], expected[1], atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=3e-6, rtol=3e-6)

    def test_streamed_bfloat16_repeated_rows_match_full_gradient(self):
        torch.manual_seed(47)
        target_logits = torch.randn(3, 13, dtype=torch.bfloat16)
        target_indices = torch.tensor([[0, 1], [1, 2]])
        target_ids = torch.tensor([[1, 4], [7, 12]])
        draft_reference = torch.randn(
            2,
            2,
            13,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        draft_streamed = draft_reference.detach().clone().requires_grad_(True)
        expected = edr_distribution_statistics_from_target_probabilities(
            draft_reference,
            torch.softmax(target_logits.float(), dim=-1)[target_indices],
            target_ids,
        )
        actual = streaming_edr_distribution_statistics(
            draft_streamed,
            prepare_edr_target_distribution(target_logits, 5),
            target_indices,
            target_ids,
            5,
        )
        weights = torch.randn(2, 2)
        expected_gradient = torch.autograd.grad(
            (expected[0] * weights + expected[1] * weights).sum(),
            draft_reference,
        )[0]
        actual_gradient = torch.autograd.grad(
            (actual[0] * weights + actual[1] * weights).sum(),
            draft_streamed,
        )[0]

        torch.testing.assert_close(actual[0], expected[0], atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(actual[1], expected[1], atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(actual_gradient, expected_gradient)

    def test_backward_tile_returns_draft_dtype(self):
        torch.manual_seed(48)
        draft_logits = torch.randn(2, 3, 5, dtype=torch.bfloat16)
        target_probabilities = torch.softmax(torch.randn(4, 5), dim=-1)
        target_inverse_indices = torch.tensor([[0, 1, 2], [1, 2, 3]])

        gradient = edr_ops._edr_backward_tile(
            draft_logits,
            torch.logsumexp(draft_logits.float(), dim=-1),
            target_probabilities,
            target_inverse_indices,
            torch.tensor([[0, 1, 2], [2, 3, 4]]),
            torch.arange(5),
            torch.zeros(2, 3),
            torch.ones(2, 3),
            torch.ones(5, dtype=torch.bool),
            torch.rand(2, 3),
            torch.rand(2, 3),
            torch.rand(2, 3),
        )

        self.assertEqual(gradient.dtype, draft_logits.dtype)
        self.assertEqual(gradient.shape, draft_logits.shape)


class TestEDRDynamicProgram(unittest.TestCase):
    def test_batched_native_parallel_matches_serial_for_mixed_horizons(self):
        generator = torch.Generator().manual_seed(40)
        statistics = []
        for length in (1, 17, 64, 129):
            costs = torch.rand(length, 7, generator=generator)
            acceptance = torch.rand(length, 7, generator=generator)
            for prefix in range(length):
                valid_width = min(7, length - prefix)
                costs[prefix, valid_width:] = float("nan")
                acceptance[prefix, valid_width:] = float("nan")
            statistics.append((costs, acceptance))

        serial = [
            exact_edr_dynamic_program(costs, acceptance, num_proposals=7)
            for costs, acceptance in statistics
        ]
        parallel = exact_edr_dynamic_programs(
            statistics,
            num_proposals=7,
            max_workers=4,
            return_on_cpu=True,
        )

        self.assertEqual(len(parallel), len(serial))
        fields = (
            "expected_passes",
            "round_values",
            "learned_values",
            "continuation_advantages",
            "occupancies",
            "conditional_survivals",
            "round_start_probabilities",
            "learned_mask",
        )
        for expected, actual in zip(serial, parallel, strict=True):
            self.assertEqual(actual.expected_passes.device.type, "cpu")
            for field in fields:
                torch.testing.assert_close(
                    getattr(actual, field),
                    getattr(expected, field),
                    rtol=0,
                    atol=0,
                )

    def test_batched_dynamic_program_rejects_nonpositive_workers(self):
        with self.assertRaisesRegex(ValueError, "max_workers"):
            exact_edr_dynamic_programs([], num_proposals=7, max_workers=0)

    def test_matches_differentiable_reference_for_b_one_two_and_three(self):
        torch.manual_seed(41)
        for block_size in (1, 2, 3):
            with self.subTest(block_size=block_size):
                length = 5
                mask = torch.zeros(length, block_size, dtype=torch.bool)
                for prefix in range(length):
                    mask[prefix, : min(block_size, length - prefix)] = True
                costs = torch.rand(length, block_size) * mask
                acceptance = torch.rand(length, block_size) * mask
                expected = _differentiable_reference(costs, acceptance, block_size)
                exhaustive, occupancies, round_starts = _exhaustive_path_reference(
                    costs, acceptance, block_size
                )
                actual = exact_edr_dynamic_program(costs, acceptance, num_proposals=block_size)
                torch.testing.assert_close(actual.expected_passes, expected)
                torch.testing.assert_close(actual.expected_passes, exhaustive)
                torch.testing.assert_close(actual.occupancies, occupancies, check_dtype=False)
                torch.testing.assert_close(
                    actual.round_start_probabilities,
                    round_starts,
                    check_dtype=False,
                )
                expected_occupancies = (
                    actual.round_start_probabilities[:-1].unsqueeze(-1)
                    * actual.conditional_survivals
                )
                torch.testing.assert_close(
                    actual.occupancies[actual.learned_mask],
                    expected_occupancies[actual.learned_mask],
                )
                self.assertEqual(actual.learned_mask.shape, (length, block_size))
                self.assertTrue(actual.learned_mask[:, 0].all())

    def test_last_ordinary_token_cost_is_retained(self):
        # B=2: both x1 and x2 are learned; x3 is boundary-only.
        costs = torch.tensor([[0.0, 0.3], [0.0, 0.0]])
        acceptance = torch.tensor([[1.0, 0.4], [1.0, 0.0]])
        result = exact_edr_dynamic_program(costs, acceptance, num_proposals=2)
        self.assertAlmostEqual(result.expected_passes.item(), 1.3, places=6)

    def test_block_exhaustion_costs_one_pass(self):
        # x1 and x2 are accepted learned proposals, x3 is the target bonus, and
        # x4 is boundary EOS. A second target pass is therefore certain.
        costs = torch.zeros(3, 2)
        acceptance = torch.ones(3, 2)
        result = exact_edr_dynamic_program(costs, acceptance, num_proposals=2)
        self.assertAlmostEqual(result.expected_passes.item(), 2.0, places=6)
        self.assertAlmostEqual(result.round_start_probabilities[3].item(), 1.0, places=6)

    def test_first_proposal_can_reject_and_one_proposal_is_learned(self):
        costs = torch.tensor([[1.0], [0.0]])
        acceptance = torch.tensor([[0.0], [1.0]])
        result = exact_edr_dynamic_program(costs, acceptance, num_proposals=1)
        self.assertEqual(result.learned_mask.shape, (2, 1))
        self.assertAlmostEqual(result.round_start_probabilities[1].item(), 1.0)
        self.assertAlmostEqual(result.expected_passes.item(), 2.0)

    def test_equation_14_matches_direct_dp_gradient(self):
        torch.manual_seed(4)
        length, block_size = 5, 3
        mask = torch.zeros(length, block_size, dtype=torch.bool)
        for prefix in range(length):
            mask[prefix, : min(block_size, length - prefix)] = True

        base_costs = torch.rand(length, block_size) * mask
        base_acceptance = (0.1 + 0.8 * torch.rand(length, block_size)) * mask

        direct_costs = base_costs.clone().requires_grad_()
        direct_acceptance = base_acceptance.clone().requires_grad_()
        direct = _differentiable_reference(direct_costs, direct_acceptance, block_size)
        direct_grads = torch.autograd.grad(direct, (direct_costs, direct_acceptance))

        surrogate_costs = base_costs.clone().requires_grad_()
        surrogate_acceptance = base_acceptance.clone().requires_grad_()
        result = exact_edr_dynamic_program(
            surrogate_costs, surrogate_acceptance, num_proposals=block_size
        )
        surrogate = edr_surrogate_sum(surrogate_costs, surrogate_acceptance, result)
        surrogate_grads = torch.autograd.grad(surrogate, (surrogate_costs, surrogate_acceptance))

        self.assertTrue(torch.allclose(direct_grads[0], surrogate_grads[0], atol=1e-6))
        self.assertTrue(torch.allclose(direct_grads[1], surrogate_grads[1], atol=1e-6))
        self.assertGreater(surrogate_grads[0][:, 0].abs().sum().item(), 0.0)
        self.assertFalse(result.occupancies.requires_grad)
        self.assertFalse(result.round_values.requires_grad)
        self.assertFalse(result.continuation_advantages.requires_grad)

    def test_invalid_acceptance_rejected(self):
        with self.assertRaises(ValueError):
            exact_edr_dynamic_program(
                torch.zeros(2, 2),
                torch.tensor([[1.1, 0.0], [0.0, 0.0]]),
                num_proposals=2,
            )

    def test_invalid_padding_entries_do_not_affect_the_dynamic_program(self):
        costs = torch.tensor([[0.2, 0.3, float("nan")], [0.4, float("nan"), -1.0]])
        acceptance = torch.tensor([[0.5, 0.6, float("nan")], [0.7, 4.0, -2.0]])
        result = exact_edr_dynamic_program(costs, acceptance, num_proposals=3)
        self.assertTrue(torch.isfinite(result.expected_passes))

    def test_input_query_anchor_cannot_be_counted_as_an_eighth_proposal(self):
        with self.assertRaisesRegex(ValueError, "learned-state width"):
            exact_edr_dynamic_program(
                torch.zeros(3, 8),
                torch.ones(3, 8),
                num_proposals=7,
            )

    def test_seven_proposals_and_one_bonus_advance_at_most_eight_tokens(self):
        # The implementation-side input anchor is already committed. Seven
        # accepted proposals plus one target bonus move the next round from
        # prefix 0 to prefix 8, never to prefix 9.
        length = 16
        costs = torch.zeros(length, 7)
        acceptance = torch.ones(length, 7)
        result = exact_edr_dynamic_program(costs, acceptance, num_proposals=7)

        occupied_round_starts = torch.nonzero(
            result.round_start_probabilities > 0,
            as_tuple=False,
        ).flatten()
        self.assertEqual(occupied_round_starts.tolist(), [0, 8, 16])
        self.assertAlmostEqual(result.expected_passes.item(), 3.0)
        self.assertEqual(result.learned_mask.shape, (length, 7))
        torch.testing.assert_close(result.occupancies[0], torch.ones(7, dtype=torch.float64))
        self.assertAlmostEqual(result.continuation_advantages[0, 6].item(), 1.0)
        self.assertTrue(result.learned_mask[-1, 0])
        self.assertFalse(result.learned_mask[-1, 1:].any())


class TestEDRAnchorSampling(unittest.TestCase):
    def test_capped_proportional_inclusion_probabilities(self):
        round_starts = torch.tensor([0.5, 0.25, 0.125, 0.125])
        probabilities = edr_anchor_inclusion_probabilities(round_starts, num_anchors=2)
        torch.testing.assert_close(
            probabilities,
            torch.tensor([1.0, 0.5, 0.25, 0.25], dtype=torch.float64),
        )
        self.assertEqual(probabilities.sum().item(), 2.0)

    def test_systematic_sample_is_distinct_and_uses_fixed_marginals(self):
        round_starts = torch.tensor([0.5, 0.25, 0.125, 0.125])
        sample = sample_edr_round_starts(
            round_starts,
            num_anchors=2,
            random_start=0.1,
        )
        torch.testing.assert_close(
            sample.inclusion_probabilities,
            torch.tensor([1.0, 0.5, 0.25, 0.25], dtype=torch.float64),
        )
        torch.testing.assert_close(
            sample.inverse_pps_scale,
            torch.tensor(0.5, dtype=torch.float64),
        )
        self.assertEqual(sample.indices.tolist(), [0, 1])
        self.assertEqual(torch.unique(sample.indices).numel(), 2)

        # Four equally spaced systematic starts integrate these rational
        # marginals exactly: anchor 0 appears four times, then 2, 1, and 1.
        counts = torch.zeros(4, dtype=torch.long)
        for start in (0.125, 0.375, 0.625, 0.875):
            draw = sample_edr_round_starts(
                round_starts,
                num_anchors=2,
                random_start=start,
            )
            counts[draw.indices] += 1
        self.assertEqual(counts.tolist(), [4, 2, 1, 1])

    def test_systematic_sample_keeps_rounded_thresholds_in_bounds(self):
        sample = sample_edr_round_starts(
            torch.ones(4, dtype=torch.float64),
            num_anchors=2,
            random_start=math.nextafter(1.0, 0.0),
        )

        self.assertEqual(sample.indices.tolist(), [1, 3])
        self.assertLess(int(sample.indices.max()), 4)
        self.assertEqual(torch.unique(sample.indices).numel(), 2)

    def test_zero_occupancy_starts_are_omitted_for_fixed_query_padding(self):
        sample = sample_edr_round_starts(
            torch.tensor([1.0, 0.0, 0.0]),
            num_anchors=2,
        )
        self.assertEqual(sample.indices.tolist(), [0])
        torch.testing.assert_close(
            sample.inclusion_probabilities,
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        )
        self.assertIsNone(sample.inverse_pps_scale)

    def test_horvitz_thompson_surrogate_averages_to_full_surrogate(self):
        costs = torch.tensor([[0.2], [0.4], [0.6], [0.8]])
        acceptance = torch.tensor([[0.3], [0.5], [0.7], [0.9]])
        dynamic_program = exact_edr_dynamic_program(
            costs,
            acceptance,
            num_proposals=1,
        )
        full = edr_surrogate_sum(costs, acceptance, dynamic_program)

        sampling_weights = dynamic_program.round_start_probabilities[: costs.shape[0]]
        design = sample_edr_round_starts(
            sampling_weights,
            num_anchors=2,
            random_start=0.5,
        )
        cumulative = design.inclusion_probabilities.cumsum(dim=0)
        cumulative[-1] = round(float(cumulative[-1].item()))
        breakpoints = {0.0, 1.0}
        for value in cumulative.tolist():
            fractional = value % 1.0
            if 0.0 < fractional < 1.0:
                breakpoints.add(fractional)

        sampled_expectation = full.new_zeros(())
        ordered_breakpoints = sorted(breakpoints)
        for lower, upper in zip(ordered_breakpoints, ordered_breakpoints[1:]):
            interval_width = upper - lower
            draw = sample_edr_round_starts(
                sampling_weights,
                num_anchors=2,
                random_start=lower + interval_width / 2.0,
            )
            sampled_expectation = sampled_expectation + interval_width * (
                sampled_edr_surrogate_sum(
                    costs[draw.indices],
                    acceptance[draw.indices],
                    dynamic_program,
                    draw.indices,
                    draw.selected_inclusion_probabilities,
                    torch.ones(draw.indices.numel(), dtype=torch.bool),
                    draw.inverse_pps_scale,
                )
            )
        torch.testing.assert_close(sampled_expectation, full)

    def test_canceled_weights_preserve_tiny_conditional_survival(self):
        base = exact_edr_dynamic_program(
            torch.zeros(3, 2),
            torch.ones(3, 2),
            num_proposals=2,
        )
        round_starts = torch.tensor([1.0, 1e-300, 0.0, 0.0], dtype=torch.float64)
        occupancies = base.occupancies.clone()
        occupancies[1] = torch.tensor([1e-300, 0.0], dtype=torch.float64)
        conditional_survivals = base.conditional_survivals.clone()
        conditional_survivals[1] = torch.tensor([1.0, 1e-30], dtype=torch.float64)
        dynamic_program = replace(
            base,
            occupancies=occupancies,
            conditional_survivals=conditional_survivals,
            round_start_probabilities=round_starts,
        )
        design = sample_edr_round_starts(
            round_starts[:3],
            num_anchors=1,
            random_start=0.5,
        )
        self.assertIsNotNone(design.inverse_pps_scale)
        self.assertEqual(design.inclusion_probabilities[1].item(), 1e-300)

        sampled_costs = torch.zeros(1, 2, requires_grad=True)
        sampled = sampled_edr_surrogate_sum(
            sampled_costs,
            torch.zeros_like(sampled_costs),
            dynamic_program,
            torch.tensor([1]),
            design.inclusion_probabilities[1:2],
            torch.tensor([True]),
            design.inverse_pps_scale,
        )
        sampled.backward()
        self.assertTrue(torch.isfinite(sampled))
        torch.testing.assert_close(
            sampled_costs.grad,
            torch.tensor([[1.0, 1e-30]]),
        )

    def test_canceled_weights_match_direct_ht_for_capped_and_uncapped_anchors(self):
        length, block_size = 4, 3
        costs = torch.zeros(length, block_size)
        acceptance = torch.full_like(costs, 0.99)
        dynamic_program = exact_edr_dynamic_program(
            costs,
            acceptance,
            num_proposals=block_size,
        )
        draw = sample_edr_round_starts(
            dynamic_program.round_start_probabilities[:length],
            num_anchors=2,
            random_start=0.5,
        )
        selected_probabilities = draw.selected_inclusion_probabilities
        self.assertTrue(bool((selected_probabilities == 1.0).any()))
        self.assertTrue(bool((selected_probabilities < 1.0).any()))

        sampled_costs = torch.zeros(draw.indices.numel(), block_size, requires_grad=True)
        sampled = sampled_edr_surrogate_sum(
            sampled_costs,
            torch.zeros_like(sampled_costs),
            dynamic_program,
            draw.indices,
            selected_probabilities,
            torch.ones(draw.indices.numel(), dtype=torch.bool),
            draw.inverse_pps_scale,
        )
        sampled.backward()
        direct_ht_weights = (
            dynamic_program.occupancies[draw.indices] / selected_probabilities.unsqueeze(-1)
        ).float()
        expected_grad = torch.where(
            dynamic_program.learned_mask[draw.indices],
            direct_ht_weights,
            torch.zeros_like(direct_ht_weights),
        )
        torch.testing.assert_close(sampled_costs.grad, expected_grad)

    def test_sampled_surrogate_is_exact_when_all_anchors_fit(self):
        torch.manual_seed(29)
        length, block_size, query_slots = 5, 3, 8
        mask = torch.zeros(length, block_size, dtype=torch.bool)
        for prefix in range(length):
            mask[prefix, : min(block_size, length - prefix)] = True
        base_costs = torch.rand(length, block_size) * mask
        base_acceptance = (0.1 + 0.8 * torch.rand(length, block_size)) * mask
        dynamic_program = exact_edr_dynamic_program(
            base_costs,
            base_acceptance,
            num_proposals=block_size,
        )

        full_costs = base_costs.clone().requires_grad_()
        full_acceptance = base_acceptance.clone().requires_grad_()
        full = edr_surrogate_sum(full_costs, full_acceptance, dynamic_program)
        full_grads = torch.autograd.grad(full, (full_costs, full_acceptance))

        sampled_costs = torch.zeros(query_slots, block_size)
        sampled_acceptance = torch.zeros_like(sampled_costs)
        sampled_costs[:length] = base_costs
        sampled_acceptance[:length] = base_acceptance
        sampled_costs.requires_grad_()
        sampled_acceptance.requires_grad_()
        prefixes = torch.zeros(query_slots, dtype=torch.long)
        prefixes[:length] = torch.arange(length)
        inclusion = torch.ones(query_slots)
        keep = torch.arange(query_slots) < length
        sampled = sampled_edr_surrogate_sum(
            sampled_costs,
            sampled_acceptance,
            dynamic_program,
            prefixes,
            inclusion,
            keep,
            None,
        )
        sampled_grads = torch.autograd.grad(
            sampled,
            (sampled_costs, sampled_acceptance),
        )
        torch.testing.assert_close(sampled, full)
        torch.testing.assert_close(sampled_grads[0][:length], full_grads[0])
        torch.testing.assert_close(sampled_grads[1][:length], full_grads[1])
        self.assertEqual(sampled_grads[0][length:].abs().sum().item(), 0.0)
        self.assertEqual(sampled_grads[1][length:].abs().sum().item(), 0.0)


class TestDFlashEDRForward(unittest.TestCase):
    def test_full_anchor_backprop_requires_edr_objective(self):
        with self.assertRaisesRegex(ValueError, "only by the EDR objective"):
            DFlashModel(
                _TinyDraft(),
                loss_objective="decay",
                edr_full_anchor_backprop=True,
            )

    def test_gradient_anchor_bucket_boundaries(self):
        expected = {
            1: 64,
            64: 64,
            65: 128,
            128: 128,
            129: 256,
            256: 256,
            257: 512,
            512: 512,
        }
        for selected_count, slots in expected.items():
            with self.subTest(selected_count=selected_count):
                self.assertEqual(_edr_gradient_anchor_slots(selected_count, 512), slots)
        self.assertEqual(_edr_gradient_anchor_slots(17, 31), 31)
        self.assertEqual(_edr_gradient_anchor_slots(257, 300), 300)

    def test_all_row_statistics_forwards_requested_draft_temperature(self):
        model, batch = _edr_model_and_batch(chunk_size=16)
        horizons = extract_edr_horizons(
            batch["loss_mask"],
            batch["attention_mask"],
        )
        temperatures = []

        def capture_statistics(**kwargs):
            temperatures.append(kwargs["draft_temperature"])
            shape = (*kwargs["anchors"].shape, model.edr_proposal_width)
            return (
                torch.zeros(shape, dtype=torch.float32),
                torch.ones(shape, dtype=torch.float32),
                torch.ones(shape, dtype=torch.bool),
            )

        model._edr_chunk_statistics = capture_statistics
        statistics_by_row, _ = model._edr_all_row_statistics(
            input_ids=batch["input_ids"],
            hidden_states_list=batch["hidden_states_list"],
            loss_mask=batch["loss_mask"],
            lm_head_weight=batch["lm_head_weight"],
            normalized_target_hidden=batch["last_hidden_states"],
            horizons_by_row=[horizons],
            attention_mask=batch["attention_mask"],
            ctx_doc_ids=None,
            base_position_ids=None,
            draft_temperature=0.0,
        )

        self.assertEqual(temperatures, [0.0])
        self.assertEqual(statistics_by_row[0][0].costs.shape, (5, 3))

    def test_detached_statistics_sweep_then_one_fixed_query_gradient_pass(self):
        for includes_anchor in (False, True):
            with self.subTest(includes_anchor=includes_anchor):
                model, batch = _edr_model_and_batch(
                    chunk_size=2,
                    query_includes_input_anchor=includes_anchor,
                )
                calls = []
                original_statistics = model._edr_chunk_statistics

                def count_statistics(
                    *args,
                    _calls=calls,
                    _original_statistics=original_statistics,
                    **kwargs,
                ):
                    _calls.append(
                        (
                            torch.is_grad_enabled(),
                            kwargs["anchors"].numel(),
                            kwargs.get("block_keep_mask"),
                            kwargs["target_distribution"].logits.data_ptr(),
                            tuple(kwargs["target_distribution"].logits.shape),
                        )
                    )
                    return _original_statistics(*args, **kwargs)

                model._edr_chunk_statistics = count_statistics
                loss, *_ = model(**batch)

                expected_chunks = 3
                self.assertEqual(len(calls), expected_chunks + 1)
                self.assertTrue(all(not grad_enabled for grad_enabled, *_ in calls[:-1]))
                self.assertTrue(calls[-1][0])
                self.assertEqual(calls[-1][1], model.num_anchors)
                self.assertEqual(calls[-1][2].shape, (model.num_anchors,))
                self.assertEqual(len({call[3] for call in calls}), 1)
                expected_target_positions = 5
                self.assertTrue(all(call[4] == (expected_target_positions, 16) for call in calls))
                loss.backward()
                # No activation-checkpoint recomputation occurs in backward.
                self.assertEqual(len(calls), expected_chunks + 1)

    def test_full_anchor_backprop_skips_sampling_and_matches_exact_gradient(self):
        model, batch = _edr_model_and_batch(
            chunk_size=2,
            full_anchor_backprop=True,
        )
        batch = dict(
            batch,
            loss_mask=torch.tensor(
                [[0, 1, 1, 1, 0, 1, 1, 1, 1]],
                dtype=torch.float32,
            ),
        )
        sampled_all_reference = deepcopy(model)
        sampled_all_reference.edr_full_anchor_backprop = False
        sampled_all_reference.num_anchors = 128

        gradient_prefix_chunks = []
        original_statistics = model._edr_chunk_statistics

        def capture_full_statistics(*args, **kwargs):
            if torch.is_grad_enabled():
                gradient_prefix_chunks.append(kwargs["prefixes"].detach().clone())
            return original_statistics(*args, **kwargs)

        model._edr_chunk_statistics = capture_full_statistics
        with mock.patch(
            "angelspec.models.dflash.sample_edr_round_starts",
            side_effect=AssertionError("full-anchor mode must not sample"),
        ):
            full_loss, *_ = model(**batch)
        full_loss.backward()

        reference_loss, *_ = sampled_all_reference(**batch)
        reference_loss.backward()

        full_prefixes = torch.cat(gradient_prefix_chunks)
        torch.testing.assert_close(
            full_prefixes,
            torch.tensor([0, 1, 0, 1, 2], device=full_prefixes.device),
        )
        self.assertEqual([chunk.numel() for chunk in gradient_prefix_chunks], [2, 2, 1])
        torch.testing.assert_close(full_loss, reference_loss, rtol=1e-6, atol=1e-6)
        reference_parameters = dict(sampled_all_reference.draft_model.named_parameters())
        for name, parameter in model.draft_model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertIsNotNone(reference_parameters[name].grad)
                torch.testing.assert_close(
                    parameter.grad,
                    reference_parameters[name].grad,
                    rtol=1e-5,
                    atol=1e-6,
                )

    def test_cross_horizon_batches_share_detached_and_gradient_forwards(self):
        two_horizon_mask = torch.tensor(
            [[0, 1, 1, 1, 0, 1, 1, 1, 1]],
            dtype=torch.float32,
        )
        for includes_anchor in (False, True):
            with self.subTest(includes_anchor=includes_anchor):
                model, batch = _edr_model_and_batch(
                    chunk_size=8,
                    query_includes_input_anchor=includes_anchor,
                )
                batch = dict(batch, loss_mask=two_horizon_mask)
                calls = []
                original_statistics = model._edr_chunk_statistics

                def capture_statistics(
                    *args,
                    _calls=calls,
                    _original_statistics=original_statistics,
                    **kwargs,
                ):
                    _calls.append(
                        {
                            "grad": torch.is_grad_enabled(),
                            "anchors": kwargs["anchors"].detach().clone(),
                            "starts": torch.as_tensor(kwargs["target_probability_start"])
                            .detach()
                            .clone(),
                            "offsets": kwargs["target_distribution_offsets"].detach().clone(),
                            "counts": kwargs["target_probability_counts"].detach().clone(),
                            "target_shape": tuple(kwargs["target_distribution"].logits.shape),
                        }
                    )
                    return _original_statistics(*args, **kwargs)

                model._edr_chunk_statistics = capture_statistics
                loss, *_ = model(**batch)

                self.assertEqual(len(calls), 2)
                self.assertFalse(calls[0]["grad"])
                self.assertTrue(calls[1]["grad"])
                self.assertEqual(calls[0]["anchors"].numel(), 5)
                self.assertEqual(calls[1]["anchors"].numel(), 4)
                self.assertEqual(calls[0]["target_shape"], (5, 16))
                self.assertEqual(calls[1]["target_shape"], (5, 16))
                self.assertGreater(torch.unique(calls[0]["starts"]).numel(), 1)
                self.assertGreater(torch.unique(calls[0]["offsets"]).numel(), 1)
                self.assertGreater(torch.unique(calls[0]["counts"]).numel(), 1)

                loss.backward()
                self.assertEqual(len(calls), 2)

    def test_cross_horizon_gradient_padding_is_applied_once(self):
        two_horizon_mask = torch.tensor(
            [[0, 1, 1, 1, 0, 1, 1, 1, 1]],
            dtype=torch.float32,
        )
        model, batch = _edr_model_and_batch(chunk_size=128)
        model.num_anchors = 128
        padded_reference = deepcopy(model)
        padded_reference._edr_combined_gradient_anchor_slots = (
            lambda _selected, _capacity: padded_reference.num_anchors
        )
        batch = dict(batch, loss_mask=two_horizon_mask)

        optimized_call = None
        reference_call = None
        optimized_statistics = model._edr_chunk_statistics
        reference_statistics = padded_reference._edr_chunk_statistics

        def capture_optimized(*args, **kwargs):
            nonlocal optimized_call
            if torch.is_grad_enabled():
                optimized_call = {
                    "anchors": kwargs["anchors"].detach().clone(),
                    "keep": kwargs["block_keep_mask"].detach().clone(),
                }
            return optimized_statistics(*args, **kwargs)

        def capture_reference(*args, **kwargs):
            nonlocal reference_call
            if torch.is_grad_enabled():
                reference_call = {
                    "anchors": kwargs["anchors"].detach().clone(),
                    "keep": kwargs["block_keep_mask"].detach().clone(),
                }
            return reference_statistics(*args, **kwargs)

        model._edr_chunk_statistics = capture_optimized
        padded_reference._edr_chunk_statistics = capture_reference
        torch.manual_seed(127)
        optimized_loss, *_ = model(**batch)
        torch.manual_seed(127)
        reference_loss, *_ = padded_reference(**batch)
        optimized_loss.backward()
        reference_loss.backward()

        self.assertIsNotNone(optimized_call)
        self.assertIsNotNone(reference_call)
        self.assertEqual(optimized_call["anchors"].shape, (64,))
        self.assertEqual(reference_call["anchors"].shape, (128,))
        self.assertEqual(optimized_call["keep"].sum().item(), 5)
        self.assertTrue(optimized_call["keep"][:5].all())
        self.assertFalse(optimized_call["keep"][5:].any())
        self.assertEqual(optimized_call["anchors"][5:].tolist(), [0] * 59)
        torch.testing.assert_close(optimized_loss, reference_loss)
        torch.testing.assert_close(
            model.draft_model.proj.weight.grad,
            padded_reference.draft_model.proj.weight.grad,
        )

    def test_cross_row_batching_preserves_independent_objectives_and_gradients(self):
        model, batch = _edr_model_and_batch(chunk_size=16)
        separate = deepcopy(model)
        first_mask = torch.tensor(
            [[0, 1, 1, 1, 0, 0, 0, 0, 0]],
            dtype=torch.float32,
        )
        two_horizon_mask = torch.tensor(
            [[0, 1, 1, 1, 0, 1, 1, 1, 1]],
            dtype=torch.float32,
        )
        combined_batch = dict(
            batch,
            input_ids=batch["input_ids"].repeat(2, 1),
            hidden_states_list=[state.repeat(2, 1, 1) for state in batch["hidden_states_list"]],
            loss_mask=torch.cat((first_mask, two_horizon_mask)),
            attention_mask=batch["attention_mask"].repeat(2, 1),
            last_hidden_states=batch["last_hidden_states"].repeat(2, 1, 1),
        )
        calls = []
        original_statistics = model._edr_chunk_statistics

        def count_statistics(*args, **kwargs):
            calls.append(
                (torch.is_grad_enabled(), tuple(kwargs["anchors"].shape))
            )
            return original_statistics(*args, **kwargs)

        model._edr_chunk_statistics = count_statistics
        torch.manual_seed(107)
        combined_loss, *_ = model(**combined_batch)
        torch.manual_seed(107)
        first_loss, *_ = separate(**dict(batch, loss_mask=first_mask))
        second_loss, *_ = separate(**dict(batch, loss_mask=two_horizon_mask))

        combined_loss.backward()
        (first_loss + second_loss).backward()

        # Both rows share one detached and one gradient call. The leading batch
        # dimension keeps their context/proposal attention independent.
        self.assertEqual(calls, [(False, (2, 5)), (True, (2, 4))])
        torch.testing.assert_close(combined_loss, first_loss + second_loss)
        torch.testing.assert_close(
            model.draft_model.proj.weight.grad,
            separate.draft_model.proj.weight.grad,
        )

    def test_short_horizon_uses_e2e_style_fixed_query_padding(self):
        model, batch = _edr_model_and_batch(chunk_size=2)
        model.num_anchors = 8
        selected_call = None
        original_statistics = model._edr_chunk_statistics

        def capture_selected(*args, **kwargs):
            nonlocal selected_call
            if torch.is_grad_enabled():
                selected_call = {
                    "anchors": kwargs["anchors"].detach().clone(),
                    "keep": kwargs["block_keep_mask"].detach().clone(),
                }
            return original_statistics(*args, **kwargs)

        model._edr_chunk_statistics = capture_selected
        loss, *_ = model(**batch)
        self.assertIsNotNone(selected_call)
        self.assertEqual(selected_call["anchors"].shape, (8,))
        self.assertEqual(selected_call["keep"].tolist(), [True] * 5 + [False] * 3)
        self.assertEqual(selected_call["anchors"][5:].tolist(), [0, 0, 0])
        loss.backward()

    def test_short_horizon_bucket_matches_full_padding_gradient(self):
        model, batch = _edr_model_and_batch(chunk_size=2)
        model.num_anchors = 128
        full_padding = deepcopy(model)
        full_padding._edr_gradient_anchor_slots = lambda _selected: full_padding.num_anchors

        selected_call = None
        original_statistics = model._edr_chunk_statistics

        def capture_selected(*args, **kwargs):
            nonlocal selected_call
            if torch.is_grad_enabled():
                selected_call = {
                    "anchors": kwargs["anchors"].detach().clone(),
                    "keep": kwargs["block_keep_mask"].detach().clone(),
                }
            return original_statistics(*args, **kwargs)

        model._edr_chunk_statistics = capture_selected
        torch.manual_seed(97)
        bucketed_loss, *_ = model(**batch)
        torch.manual_seed(97)
        full_loss, *_ = full_padding(**batch)
        bucketed_loss.backward()
        full_loss.backward()

        self.assertIsNotNone(selected_call)
        self.assertEqual(selected_call["anchors"].shape, (64,))
        self.assertEqual(selected_call["keep"].sum().item(), 5)
        torch.testing.assert_close(bucketed_loss, full_loss)
        torch.testing.assert_close(
            model.draft_model.proj.weight.grad,
            full_padding.draft_model.proj.weight.grad,
        )

    def test_input_anchor_layout_scores_a_one_token_horizon(self):
        model, batch = _edr_model_and_batch(
            chunk_size=2,
            query_includes_input_anchor=True,
        )
        batch = dict(
            batch,
            loss_mask=torch.tensor([[0, 1, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.float32),
        )
        calls = []
        original_statistics = model._edr_chunk_statistics

        def count_statistics(*args, **kwargs):
            calls.append(
                {
                    "grad": torch.is_grad_enabled(),
                    "anchors": kwargs["anchors"].detach().clone(),
                    "keep": (
                        None
                        if kwargs.get("block_keep_mask") is None
                        else kwargs["block_keep_mask"].detach().clone()
                    ),
                }
            )
            return original_statistics(*args, **kwargs)

        model._edr_chunk_statistics = count_statistics
        loss, *_ = model(**batch)
        self.assertEqual(len(calls), 2)
        self.assertFalse(calls[0]["grad"])
        self.assertEqual(calls[0]["anchors"].tolist(), [0])
        self.assertTrue(calls[1]["grad"])
        self.assertEqual(calls[1]["anchors"].tolist(), [0, 0])
        self.assertEqual(calls[1]["keep"].tolist(), [True, False])
        self.assertTrue(loss.requires_grad)
        loss.backward()
        self.assertEqual(len(calls), 2)
        self.assertGreater(model.draft_model.proj.weight.grad.abs().sum().item(), 0.0)

    def test_forward_backward_and_components(self):
        model, batch = _edr_model_and_batch(chunk_size=2)
        loss, acceptance, tv_per_slot, acceptance_per_slot, counts, components = model(**batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(loss.requires_grad)
        self.assertTrue(torch.isfinite(acceptance))
        self.assertEqual(tv_per_slot.shape, (3,))
        self.assertEqual(acceptance_per_slot.shape, (3,))
        self.assertGreater(counts[0].item(), 0.0)
        self.assertEqual(
            set(components),
            {
                "edr_surrogate_loss",
                "edr_weighted_cost_sum",
                "edr_num_horizons",
                "edr_num_degenerate_horizons",
                "edr_generated_tokens",
            },
        )
        self.assertGreater(components["edr_weighted_cost_sum"].item(), 0.0)
        self.assertGreater(components["edr_generated_tokens"].item(), 0.0)
        loss.backward()
        self.assertIsNotNone(model.draft_model.proj.weight.grad)
        self.assertGreater(model.draft_model.proj.weight.grad.abs().sum().item(), 0.0)

    def test_learned_only_projection_matches_project_then_slice(self):
        def build_model(kind):
            draft = _TinyDraft()
            if kind == "dfly":
                draft.hidden_correction = _TinyCorrection(8)
                return DFlyModel(draft, block_size=3)
            if kind == "dspark":
                draft.markov_head = _TinyMarkov(8, 16)
                return DSparkModel(draft, block_size=3)
            return DFlashModel(draft, block_size=3)

        for kind in ("dflash", "dfly", "dspark"):
            with self.subTest(kind=kind):
                torch.manual_seed(211)
                legacy = build_model(kind)
                optimized = deepcopy(legacy)
                n_blocks, block_size, hidden_size, vocab_size = 2, 3, 8, 16
                legacy_hidden = torch.randn(
                    1,
                    n_blocks * block_size,
                    hidden_size,
                    requires_grad=True,
                )
                optimized_hidden = legacy_hidden.detach().clone().requires_grad_(True)
                legacy_lm_head = torch.randn(
                    vocab_size,
                    hidden_size,
                    requires_grad=True,
                )
                optimized_lm_head = legacy_lm_head.detach().clone().requires_grad_(True)
                previous_tokens = torch.randint(
                    0,
                    vocab_size,
                    (1, n_blocks, block_size),
                )
                objective_weights = torch.randn(1, n_blocks, block_size - 1, vocab_size)

                legacy_logits = legacy._compute_draft_logits(
                    legacy_hidden,
                    legacy_lm_head,
                    previous_tokens,
                    n_blocks,
                ).view(1, n_blocks, block_size, vocab_size)[:, :, 1:, :]
                learned_hidden = optimized_hidden.view(
                    1,
                    n_blocks,
                    block_size,
                    hidden_size,
                )[
                    :, :, 1:, :
                ].reshape(1, n_blocks * (block_size - 1), hidden_size)
                optimized_logits = optimized._compute_draft_logits(
                    learned_hidden,
                    optimized_lm_head,
                    previous_tokens[:, :, 1:],
                    n_blocks,
                ).view(1, n_blocks, block_size - 1, vocab_size)

                torch.testing.assert_close(optimized_logits, legacy_logits)
                (legacy_logits * objective_weights).sum().backward()
                (optimized_logits * objective_weights).sum().backward()
                torch.testing.assert_close(optimized_hidden.grad, legacy_hidden.grad)
                self.assertEqual(
                    legacy_hidden.grad.view(1, n_blocks, block_size, -1)[:, :, 0].count_nonzero(),
                    0,
                )

                if legacy_lm_head.grad is None:
                    self.assertIsNone(optimized_lm_head.grad)
                else:
                    torch.testing.assert_close(optimized_lm_head.grad, legacy_lm_head.grad)
                legacy_parameters = dict(legacy.named_parameters())
                optimized_parameters = dict(optimized.named_parameters())
                self.assertEqual(legacy_parameters.keys(), optimized_parameters.keys())
                for name, legacy_parameter in legacy_parameters.items():
                    with self.subTest(kind=kind, parameter=name):
                        optimized_gradient = optimized_parameters[name].grad
                        if legacy_parameter.grad is None:
                            self.assertIsNone(optimized_gradient)
                        else:
                            torch.testing.assert_close(
                                optimized_gradient,
                                legacy_parameter.grad,
                            )

    def test_chunk_size_does_not_change_metric_or_gradient(self):
        model, batch = _edr_model_and_batch(chunk_size=1)
        unchunked = deepcopy(model)
        unchunked.edr_chunk_size = 64

        torch.manual_seed(101)
        loss_a, *out_a = model(**batch)
        torch.manual_seed(101)
        loss_b, *out_b = unchunked(**batch)
        loss_a.backward()
        loss_b.backward()

        self.assertTrue(
            torch.allclose(
                out_a[-1]["edr_weighted_cost_sum"],
                out_b[-1]["edr_weighted_cost_sum"],
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                model.draft_model.proj.weight.grad,
                unchunked.draft_model.proj.weight.grad,
                atol=1e-6,
            )
        )

    def test_packing_horizons_does_not_reduce_their_gradient(self):
        # A chunk of eight fits both horizons, exercising the row-level batch;
        # separate calls remain the exact unbatched reference.
        first_mask = torch.tensor([[0, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.float32)
        second_mask = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.float32)
        for includes_anchor in (False, True):
            for chunk_size in (4, 8):
                with self.subTest(
                    includes_anchor=includes_anchor,
                    chunk_size=chunk_size,
                ):
                    model, batch = _edr_model_and_batch(
                        chunk_size=chunk_size,
                        query_includes_input_anchor=includes_anchor,
                    )
                    separate = deepcopy(model)
                    combined_batch = dict(batch, loss_mask=first_mask + second_mask)

                    torch.manual_seed(103)
                    combined_loss, *combined_output = model(**combined_batch)
                    torch.manual_seed(103)
                    first_loss, *first_output = separate(**dict(batch, loss_mask=first_mask))
                    second_loss, *second_output = separate(**dict(batch, loss_mask=second_mask))

                    combined_loss.backward()
                    (first_loss + second_loss).backward()

                    self.assertTrue(
                        torch.allclose(combined_loss, first_loss + second_loss, atol=1e-6)
                    )
                    expected_mal = (
                        first_output[-1]["edr_generated_tokens"]
                        + second_output[-1]["edr_generated_tokens"]
                    ) / (
                        first_output[-1]["edr_weighted_cost_sum"]
                        + second_output[-1]["edr_weighted_cost_sum"]
                    )
                    self.assertTrue(
                        torch.allclose(
                            combined_output[-1]["edr_generated_tokens"]
                            / combined_output[-1]["edr_weighted_cost_sum"],
                            expected_mal,
                            atol=1e-6,
                        )
                    )
                    self.assertTrue(
                        torch.allclose(
                            model.draft_model.proj.weight.grad,
                            separate.draft_model.proj.weight.grad,
                            atol=1e-6,
                        )
                    )

    def test_cross_horizon_batching_preserves_dfly_and_dspark_gradients(self):
        _, batch = _edr_model_and_batch(chunk_size=8)
        first_mask = torch.tensor([[0, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.float32)
        second_mask = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.float32)

        for kind in ("dfly", "dspark"):
            with self.subTest(kind=kind):
                draft = _TinyDraft()
                if kind == "dfly":
                    draft.hidden_correction = _TinyCorrection(8)
                    model = DFlyModel(
                        draft,
                        block_size=2,
                        num_anchors=2,
                        loss_objective="edr",
                        edr_chunk_size=8,
                        query_includes_input_anchor=True,
                    )
                else:
                    draft.markov_head = _TinyMarkov(8, 16)
                    model = DSparkModel(
                        draft,
                        block_size=2,
                        num_anchors=2,
                        loss_objective="edr",
                        edr_chunk_size=8,
                        query_includes_input_anchor=True,
                    )
                separate = deepcopy(model)

                torch.manual_seed(113)
                combined_loss, *_ = model(**dict(batch, loss_mask=first_mask + second_mask))
                torch.manual_seed(113)
                first_loss, *_ = separate(**dict(batch, loss_mask=first_mask))
                second_loss, *_ = separate(**dict(batch, loss_mask=second_mask))
                combined_loss.backward()
                (first_loss + second_loss).backward()

                torch.testing.assert_close(combined_loss, first_loss + second_loss)
                combined_parameters = dict(model.draft_model.named_parameters())
                separate_parameters = dict(separate.draft_model.named_parameters())
                self.assertEqual(combined_parameters.keys(), separate_parameters.keys())
                for name, parameter in combined_parameters.items():
                    with self.subTest(kind=kind, parameter=name):
                        expected_gradient = separate_parameters[name].grad
                        if expected_gradient is None:
                            self.assertIsNone(parameter.grad)
                        else:
                            torch.testing.assert_close(parameter.grad, expected_gradient)

    def test_eval_skips_gradient_graph(self):
        model, batch = _edr_model_and_batch(chunk_size=2)
        with torch.no_grad():
            loss, _, _, _, _, components = model(**batch)
        self.assertFalse(loss.requires_grad)
        self.assertGreater(components["edr_weighted_cost_sum"].item(), 0.0)

    def test_missing_target_state_or_norm_is_rejected(self):
        model, batch = _edr_model_and_batch(chunk_size=2)
        missing_hidden = dict(batch, last_hidden_states=None)
        with self.assertRaises(ValueError):
            model(**missing_hidden)
        missing_norm = dict(batch, target_norm=None)
        with self.assertRaises(ValueError):
            model(**missing_norm)

    def test_dfly_correction_receives_edr_gradient(self):
        _, batch = _edr_model_and_batch(chunk_size=2)
        draft = _TinyDraft()
        draft.hidden_correction = _TinyCorrection(8)
        model = DFlyModel(
            draft,
            block_size=2,
            loss_objective="edr",
            edr_chunk_size=2,
            query_includes_input_anchor=True,
        )
        loss, *_, components = model(**batch)
        self.assertNotIn("confidence_loss", components)
        loss.backward()
        self.assertGreater(draft.hidden_correction.proj.weight.grad.abs().sum().item(), 0.0)

    def test_dspark_markov_receives_edr_gradient_but_confidence_is_bypassed(self):
        _, batch = _edr_model_and_batch(chunk_size=2)
        draft = _TinyDraft()
        draft.markov_head = _TinyMarkov(8, 16)
        draft.confidence_head = nn.Linear(8, 1)
        draft.confidence_head_with_markov = False
        model = DSparkModel(
            draft,
            block_size=2,
            loss_objective="edr",
            confidence_head_alpha=1.0,
            edr_chunk_size=2,
            query_includes_input_anchor=True,
        )
        loss, *_, components = model(**batch)
        self.assertNotIn("confidence_loss", components)
        loss.backward()
        self.assertGreater(draft.markov_head.proj.weight.grad.abs().sum().item(), 0.0)
        self.assertIsNone(draft.confidence_head.weight.grad)


def _validation_args(**overrides):
    values = {
        "inference_engine_type": "hf",
        "defer_tokenization": False,
        "attention_backend": "sdpa",
        "dflash_block_size": 2,
        "min_loss_tokens": 4,
        "aux_hidden_states_layers": None,
        "dflash_loss_objective": "edr",
        "dflash_num_anchors": 512,
        "dflash_edr_chunk_size": 8,
        "dflash_edr_stop_token_ids": [2],
        "load_path": "/checkpoint/root",
        "store_last_hidden_states": True,
        "fsdp_strategy": "REPLICATE",
        "dflash_l1_loss_alpha": 0.4,
        "dflash_kl_loss_weight": 0.2,
        "dflash_lk_loss_weight": 0.0,
        "dflash_e2e_tv_loss_weight": 1.0,
        "dflash_gate_entropy_weight": 0.1,
        "dflash_opd_enabled": True,
        "dspark_l1_loss_alpha": 0.9,
        "dspark_confidence_head_alpha": 1.0,
    }
    values.update(overrides)
    return Namespace(**values)


class TestEDRConfigurationAndMetrics(unittest.TestCase):
    def test_training_config_exposes_chunk_size(self):
        from angelspec.config.train_config import TrainingConfig

        self.assertEqual(TrainingConfig().dflash_edr_chunk_size, 64)
        self.assertEqual(TrainingConfig().dflash_edr_vocab_chunk_size, 16384)
        self.assertFalse(TrainingConfig().dflash_edr_full_anchor_backprop)
        self.assertEqual(TrainingConfig().dflash_edr_cross_row_batch_size, 1)
        self.assertEqual(TrainingConfig().dflash_edr_dp_workers, 1)

    def test_validation_requires_second_stage_checkpoint(self):
        from angelspec.config.edr import configure_dflash_edr

        with self.assertRaisesRegex(ValueError, "second-stage"):
            configure_dflash_edr(_validation_args(load_path=None))

    def test_validation_requires_target_last_hidden_states(self):
        from angelspec.config.edr import configure_dflash_edr

        with self.assertRaisesRegex(ValueError, "store_last_hidden_states"):
            configure_dflash_edr(_validation_args(store_last_hidden_states=False))

    def test_validation_rejects_invalid_chunk_size(self):
        from angelspec.config.edr import configure_dflash_edr

        with self.assertRaisesRegex(ValueError, "chunk_size"):
            configure_dflash_edr(_validation_args(dflash_edr_chunk_size=0))

    def test_validation_rejects_invalid_vocab_chunk_size(self):
        from angelspec.config.edr import configure_dflash_edr

        with self.assertRaisesRegex(ValueError, "vocab_chunk_size"):
            configure_dflash_edr(_validation_args(dflash_edr_vocab_chunk_size=0))

    def test_validation_rejects_inexact_cross_row_accumulation_groups(self):
        from angelspec.config.edr import configure_dflash_edr

        with self.assertRaisesRegex(ValueError, "draft_accumulation_steps"):
            configure_dflash_edr(
                _validation_args(
                    dflash_edr_cross_row_batch_size=3,
                    draft_accumulation_steps=8,
                )
            )

    def test_validation_rejects_invalid_dp_worker_count(self):
        from angelspec.config.edr import configure_dflash_edr

        with self.assertRaisesRegex(ValueError, "dp_workers"):
            configure_dflash_edr(_validation_args(dflash_edr_dp_workers=0))

    def test_validation_accepts_bounded_cross_row_edr_batching(self):
        from angelspec.config.edr import configure_dflash_edr

        args = _validation_args(
            dflash_edr_cross_row_batch_size=2,
            draft_accumulation_steps=8,
            micro_batch_size=1,
        )
        self.assertTrue(configure_dflash_edr(args))

    def test_validation_rejects_invalid_anchor_count(self):
        from angelspec.config.edr import configure_dflash_edr

        with self.assertRaisesRegex(ValueError, "num_anchors"):
            configure_dflash_edr(_validation_args(dflash_num_anchors=0))

    def test_validation_rejects_full_shard(self):
        from angelspec.config.edr import configure_dflash_edr

        with self.assertRaisesRegex(ValueError, "fsdp_strategy=REPLICATE"):
            configure_dflash_edr(_validation_args(fsdp_strategy="FULL_SHARD"))

    def test_validation_neutralizes_every_auxiliary_loss(self):
        from angelspec.config.edr import configure_dflash_edr

        args = _validation_args()
        configure_dflash_edr(args)
        for name in (
            "dflash_l1_loss_alpha",
            "dflash_kl_loss_weight",
            "dflash_lk_loss_weight",
            "dflash_e2e_tv_loss_weight",
            "dflash_gate_entropy_weight",
            "dspark_l1_loss_alpha",
            "dspark_confidence_head_alpha",
        ):
            self.assertEqual(getattr(args, name), 0.0)
        self.assertFalse(args.dflash_opd_enabled)

    def test_metric_reduction_forms_global_edr_batch_mal(self):
        from angelspec.utils.metrics import edr_metric_totals, edr_metrics_from_totals

        metrics = [
            {
                "edr_surrogate_loss": torch.tensor(5.0),
                "edr_weighted_cost_sum": torch.tensor(2.0),
                "edr_num_horizons": torch.tensor(1.0),
                "edr_num_degenerate_horizons": torch.tensor(0.0),
                "edr_generated_tokens": torch.tensor(4.0),
            },
            {
                "edr_surrogate_loss": torch.tensor(3.0),
                "edr_weighted_cost_sum": torch.tensor(12.0),
                "edr_num_horizons": torch.tensor(3.0),
                "edr_num_degenerate_horizons": torch.tensor(1.0),
                "edr_generated_tokens": torch.tensor(15.0),
            },
        ]
        reduced = edr_metrics_from_totals(edr_metric_totals(metrics), "train/")
        self.assertAlmostEqual(reduced["train/edr_surrogate_loss"], 2.0)
        self.assertEqual(reduced["train/edr_generated_tokens"], 19.0)
        self.assertEqual(reduced["train/edr_weighted_cost"], 14.0)
        self.assertAlmostEqual(reduced["train/edr_mal"], (19.0 + 4.0) / 14.0, places=6)
        self.assertEqual(reduced["train/edr_num_horizons"], 4.0)
        self.assertEqual(reduced["train/edr_num_degenerate_horizons"], 1.0)
        self.assertEqual(reduced["train/edr_mean_horizon_length"], 4.75)


if __name__ == "__main__":
    unittest.main()
