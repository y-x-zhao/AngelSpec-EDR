"""Grouping changes execution, not the local runner's 96-sequence gradient mean."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from angelspec.data.utils import pack_loss_mask
from angelspec.models.dfly import DFlyModel
from angelspec.models.draft.dfly import DFlyConfig, DFlyDraftModel
from angelspec.train_single_gpu import LocalTrainingBatches


@pytest.mark.parametrize("group", [4, 6, 8])
def test_local_groups_preserve_global_row_mean(group):
    args = SimpleNamespace(
        dflash_distill_cross_row_batch_size=group,
        draft_accumulation_steps=96,
        max_seq_length=4096,
        min_loss_tokens=4,
        length_balance_optimizer_step=True,
    )
    records = []
    for index in range(96):
        length = 12 + index % 7
        records.append({
            "input_ids": torch.arange(1, length + 1),
            "packed_loss_mask": pack_loss_mask(torch.tensor([0] * 4 + [1] * (length - 4))),
        })
    target_calls = []

    def target_features(ids, mask):
        target_calls.append(ids.shape[0])
        assert ids.shape == mask.shape
        return {}

    batches = LocalTrainingBatches(args, target_features, device="cpu")
    batches.set_step(records)
    weight = torch.tensor(0.3, requires_grad=True)
    scales = []
    for batch in batches:
        mask = batch["loss_mask"].float()
        row_losses = ((weight * batch["input_ids"]).square() * mask).sum(1) / mask.sum(1)
        scales.append(batch["_loss_scale"])
        (row_losses.mean() * batch["_loss_scale"]).backward()
    reference = torch.tensor(0.3, requires_grad=True)
    torch.stack([
        (reference * row["input_ids"][4:]).square().mean() for row in records
    ]).mean().backward()
    assert target_calls == [group] * (96 // group)
    assert scales == [group / 96] * (96 // group)
    assert sum(scales) == pytest.approx(1.0)
    torch.testing.assert_close(weight.grad, reference.grad)


@pytest.mark.parametrize("objective, weights", [
    ("decay", dict(ce_loss_alpha=1.0, l1_loss_alpha=0.5)),
    ("decay", dict(ce_loss_alpha=0.0, e2e_tv_loss_weight=1.0, lk_loss_weight=0.5)),
    ("dpace", dict(ce_loss_alpha=0.0, lk_loss_weight=1.0)),
])
def test_grouped_row_mean_loss_matches_mean_of_single_row_calls(objective, weights):
    """A cross-row group with mean_by_row equals the mean of one-row model calls."""
    torch.manual_seed(31)
    draft = DFlyDraftModel(DFlyConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, vocab_size=61, max_position_embeddings=256,
        num_target_layers=2, target_hidden_size=32, target_num_hidden_layers=3,
        target_layer_ids=[0, 2], mask_token_id=60, block_size=4,
        enable_hidden_correction=True, attention_dropout=0.0,
    )).float()

    def make(mean_by_row):
        torch.manual_seed(32)
        return DFlyModel(
            deepcopy(draft), block_size=3, num_anchors=3, query_includes_input_anchor=True,
            loss_objective=objective, loss_decay_gamma=4.0, kl_loss_weight=0.0,
            distill_mean_by_row=mean_by_row, **weights,
        ).float()

    rows, length = 4, 24
    generator = torch.Generator().manual_seed(33)
    loss_mask = torch.ones(rows, length)
    for row in range(rows):
        loss_mask[row, : 4 + 2 * row] = 0  # unequal supervised lengths
    batch = dict(
        input_ids=torch.randint(3, 60, (rows, length), generator=generator),
        hidden_states_list=[torch.randn(rows, length, 32, generator=generator) for _ in range(2)],
        loss_mask=loss_mask,
        lm_head_weight=torch.randn(61, 32, generator=generator),
        last_hidden_states=torch.randn(rows, length, 32, generator=generator),
        target_norm=torch.nn.Identity(),
        attention_mask=torch.ones(rows, length),
        injected_anchors=torch.tensor([[8, 12, 16]] * rows),
        injected_keep_mask=torch.ones(rows, 3, dtype=torch.bool),
    )
    grouped = make(True)
    grouped_loss = grouped(**batch)[0]
    grouped_loss.backward()

    separate = make(False)
    row_losses = []
    for row in range(rows):
        row_batch = {
            key: value[row : row + 1] if isinstance(value, torch.Tensor)
            and value.shape[:1] == (rows,) else value
            for key, value in batch.items()
        }
        row_batch["hidden_states_list"] = [state[row : row + 1] for state in batch["hidden_states_list"]]
        row_losses.append(separate(**row_batch)[0])
    reference = torch.stack(row_losses).mean()
    reference.backward()

    torch.testing.assert_close(grouped_loss, reference, atol=1e-6, rtol=1e-5)
    for (name, a), (_, b) in zip(grouped.named_parameters(), separate.named_parameters(), strict=True):
        if a.grad is None:
            assert b.grad is None, name
            continue
        torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-4, msg=name)
