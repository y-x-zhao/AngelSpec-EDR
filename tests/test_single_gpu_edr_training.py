"""Single-GPU EDR recipe, grouping and true-surrogate gradient regressions."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from angelspec.config.edr import configure_dflash_edr
from angelspec.config.train_config import config_to_flat_args, load_config
from angelspec.data.utils import pack_loss_mask
from angelspec.models.dflash import DFlashModel
from angelspec.models.draft.auto import AutoDraftModelConfig
from angelspec.train_single_gpu import (
    LocalDFlyTrainer,
    LocalTrainingBatches,
    _progress_metrics,
    validate_single_gpu_args,
)
from angelspec.training.schedule import auto_calculate_training_steps

ROOT = Path(__file__).resolve().parents[1]


def _edr_args():
    config = load_config(
        str(ROOT / "configs/vllm_qwen3_8b_dfly_edr.yaml"),
        cli_args=[
            "model.target_model_backend=hf", "training.training_num_nodes=1",
            "training.training_num_gpus_per_node=1", "training.micro_batch_size=1",
            "training.draft_accumulation_steps=96", "training.prefetch_depth=0",
        ],
        save_snapshot=False,
    )
    return config_to_flat_args(config)


@pytest.mark.parametrize("full_anchors", [False, True])
def test_edr_keeps_distributed_recipe_and_resolves_deployed_stops(tmp_path, full_anchors):
    args = _edr_args()
    args.dflash_edr_full_anchor_backprop = full_anchors
    args.target_model_path = str(tmp_path)
    (tmp_path / "generation_config.json").write_text('{"eos_token_id": [2, 21]}')
    args.decode_stop_token_ids = [20]
    draft = AutoDraftModelConfig.from_file(str(ROOT / "angelspec/config/dfly_qwen3_8b_draft_config.json"))
    validate_single_gpu_args(args, draft)
    assert configure_dflash_edr(args)
    distributed = config_to_flat_args(load_config(
        str(ROOT / "configs/vllm_qwen3_8b_dfly_edr.yaml"), save_snapshot=False,
    ))
    assert args.global_batch_size == (
        distributed.micro_batch_size * distributed.draft_accumulation_steps
        * distributed.training_num_gpus_per_node
    ) == 96
    for key in (
        "learning_rate", "min_lr", "warmup_ratio", "weight_decay", "seed", "max_grad_norm",
        "max_seq_length", "dflash_num_anchors", "dflash_block_size", "min_loss_tokens",
        "dflash_loss_objective", "dflash_query_includes_input_anchor", "dflash_edr_dp_workers",
    ):
        assert getattr(args, key) == getattr(distributed, key), key
    assert args.dflash_edr_stop_token_ids == [2, 20, 21]
    assert args.dflash_edr_full_anchor_backprop is full_anchors
    assert args.dflash_e2e_tv_loss_weight == args.dflash_lk_loss_weight == 0
    assert args.aux_hidden_states_layers == draft.target_layer_ids
    auto_calculate_training_steps(args, 1_343_616)
    assert args.steps_per_epoch == 13_996
    assert args.num_epochs == 1
    assert args.num_train_steps == args.lr_total_steps == 13_996


@pytest.mark.parametrize("group", [1, 2, 4])
def test_edr_local_groups_are_not_grouped_twice(group):
    trainer = LocalDFlyTrainer.__new__(LocalDFlyTrainer)
    trainer.loss_objective = "edr"
    trainer.edr_cross_row_batch_size = group
    trainer.data_fetcher = SimpleNamespace(microbatches_per_item=group)
    batches = iter([object()])
    actual, count = trainer._prepare_training_batches(batches, 96)
    assert actual is batches and count == 96 // group
    trainer.edr_cross_row_batch_size = group + 1
    with pytest.raises(ValueError, match="local batches do not match"):
        trainer._prepare_training_batches(batches, 96)


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


def _features(ids, attention):
    hidden = (ids.unsqueeze(-1).float() + torch.arange(8)).sin()
    return {"hidden_states": hidden, "last_hidden_states": hidden.cos()}


@pytest.mark.parametrize("group", [2, 4])
def test_actual_edr_grouped_gradient_equals_sequential_row_mean(group):
    torch.manual_seed(51)
    model = DFlashModel(
        TinyDraft(), block_size=7, num_anchors=64, loss_objective="edr",
        edr_chunk_size=8, edr_vocab_chunk_size=11, query_includes_input_anchor=True,
        edr_full_anchor_backprop=True, edr_stop_token_ids=[2, 21],
    )
    reference = deepcopy(model)
    weight = torch.randn(23, 8)
    records = []
    for i in range(4):
        ids = torch.tensor([3, 4] + [5 + i] * (10 + i) + [2])
        mask = torch.tensor([0, 0] + [1] * (ids.numel() - 2))
        records.append({"input_ids": ids, "packed_loss_mask": pack_loss_mask(mask)})

    def run(wrapper, row_group):
        args = SimpleNamespace(
            dflash_loss_objective="edr", dflash_edr_cross_row_batch_size=row_group,
            dflash_distill_cross_row_batch_size=6, draft_accumulation_steps=4,
            max_seq_length=128, min_loss_tokens=1, length_balance_optimizer_step=True,
        )
        loader = LocalTrainingBatches(args, _features, device="cpu")
        loader.set_step(records)
        totals = {}
        for batch in loader:
            assert batch["_loss_scale"] == 1 / 4
            outputs = wrapper(
                input_ids=batch["input_ids"], hidden_states_list=[batch["hidden_states"]],
                loss_mask=batch["loss_mask"], attention_mask=batch["attention_mask"],
                last_hidden_states=batch["last_hidden_states"], lm_head_weight=weight,
                target_norm=nn.Identity(),
            )
            (outputs[0] * batch["_loss_scale"]).backward()
            for key in ("edr_weighted_cost_sum", "edr_generated_tokens", "edr_num_horizons"):
                totals[key] = totals.get(key, 0) + outputs[5][key]
        return totals

    actual, expected = run(model, group), run(reference, 1)
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key])
    for parameter, original in zip(model.parameters(), reference.parameters(), strict=True):
        if original.grad is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, original.grad)


def test_edr_progress_reports_additive_tokens_and_cost():
    assert _progress_metrics({
        "train/edr_generated_tokens": 4800, "train/edr_weighted_cost": 1200,
        "train/edr_mal": 4.0,
    }) == {"tokens": "4800", "cost": "1200.00", "MAL": "4.000"}
    assert _progress_metrics({"train/avg_acc": 0.5, "train/simulated_acc_len": 2}) == {
        "acc": "0.500", "acc_len": "2.00",
    }
