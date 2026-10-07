"""Single-GPU routing/math/lifecycle regressions; no checkpoint downloads or GPUs."""

import copy
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from angelspec.config.train_config import config_to_flat_args, load_config
from angelspec.data.utils import pack_loss_mask
from angelspec.models.draft.auto import AutoDraftModelConfig
from angelspec.train_single_gpu import (
    LocalTrainingBatches,
    run_local_training,
    validate_local_checkpoint,
    validate_single_gpu_args,
)
from angelspec.training.schedule import auto_calculate_training_steps

ROOT = Path(__file__).resolve().parents[1]


def _baseline(objective="e2e", **overrides):
    dotlist = [
        "model.target_model_backend=hf", "training.training_num_nodes=1",
        "training.training_num_gpus_per_node=1", "training.micro_batch_size=1",
        "training.draft_accumulation_steps=96", "training.prefetch_depth=0",
    ]
    config = load_config(str(ROOT / f"configs/vllm_qwen3_8b_dfly_{objective}.yaml"), cli_args=dotlist)
    args = config_to_flat_args(config)
    for key, value in overrides.items():
        setattr(args, key, value)
    draft = AutoDraftModelConfig.from_file(str(ROOT / "angelspec/config/dfly_qwen3_8b_draft_config.json"))
    return args, draft


@pytest.mark.parametrize("objective", ["e2e"])
def test_single_gpu_preserves_baseline_global_batch_and_recipe(objective):
    args, draft = _baseline(objective)
    validate_single_gpu_args(args, draft)
    distributed = config_to_flat_args(load_config(str(ROOT / f"configs/vllm_qwen3_8b_dfly_{objective}.yaml")))
    assert args.global_batch_size == (
        distributed.micro_batch_size * distributed.draft_accumulation_steps
        * distributed.training_num_gpus_per_node
    ) == 96
    for key in (
        "learning_rate", "min_lr", "warmup_ratio", "weight_decay", "seed", "max_grad_norm",
        "max_seq_length", "dflash_num_anchors", "dflash_block_size", "min_loss_tokens",
        "dflash_loss_objective", "dflash_lk_loss_weight", "dflash_e2e_tv_loss_weight",
        "dflash_dpace_alpha", "dflash_distill_cross_row_batch_size", "dflash_fp32_lm_head",
    ):
        assert getattr(args, key) == getattr(distributed, key), key
    assert args.last_hidden_states_prenorm
    assert args.aux_hidden_states_layers == draft.target_layer_ids
    auto_calculate_training_steps(args, 1_343_616)
    assert args.steps_per_epoch == 13_996
    assert args.num_epochs == 1
    assert args.num_train_steps == args.lr_total_steps == 13_996 * args.num_epochs


@pytest.mark.parametrize("overrides", [
    {"training_num_gpus_per_node": 6}, {"target_model_backend": "vllm"},
    {"fsdp_strategy": "FULL_SHARD"}, {"dflash_loss_objective": "unknown"},
    {"dflash_edr_full_anchor_backprop": True}, {"defer_tokenization": True},
    {"train_with_decode": True}, {"micro_batch_size": 4}, {"prefetch_depth": 16},
    {"dflash_packing": True},
    {"draft_accumulation_steps": 95}, {"dflash_kl_loss_weight": 1.0},
    {"dflash_num_anchors": 0},
    {"last_hidden_states_prenorm": False},
    {"eval_data_path": "eval.jsonl"}, {"online_eval_enabled": True},
])
def test_unsupported_modes_fail_before_startup(overrides):
    args, draft = _baseline(**overrides)
    with pytest.raises(ValueError):
        validate_single_gpu_args(args, draft)


@pytest.mark.parametrize("overrides", [
    {"dflash_ce_loss_alpha": 1.0, "dflash_l1_loss_alpha": 0.5},
    {"dflash_lk_loss_weight": 1.0},
    {"dflash_loss_objective": "dpace", "dflash_lk_loss_weight": 1.0},
    {"dflash_kl_loss_weight": 1.0, "dflash_distill_cross_row_batch_size": 1},
])
def test_row_mean_objectives_and_ungrouped_token_mean_terms_are_accepted(overrides):
    args, draft = _baseline(**overrides)
    validate_single_gpu_args(args, draft)


def test_chunked_ce_is_rejected_with_cross_row_groups(monkeypatch):
    monkeypatch.setenv("ANGELSPEC_DFLASH_LOSS_CHUNK", "1024")
    args, draft = _baseline()
    with pytest.raises(ValueError, match="ANGELSPEC_DFLASH_LOSS_CHUNK"):
        validate_single_gpu_args(args, draft)
    args, draft = _baseline(dflash_distill_cross_row_batch_size=1)
    validate_single_gpu_args(args, draft)


def _records(count=96):
    records = []
    for i in range(count):
        length = 12 + i % 7
        mask = torch.tensor([0] * 4 + [1] * (length - 4))
        records.append({
            "data_id": i,
            "input_ids": torch.arange(1, length + 1, dtype=torch.int32),
            "packed_loss_mask": pack_loss_mask(mask),
        })
    return records


def _batch_args(**overrides):
    args = SimpleNamespace(
        dflash_distill_cross_row_batch_size=4, draft_accumulation_steps=96,
        max_seq_length=4096, min_loss_tokens=4, length_balance_optimizer_step=True,
    )
    vars(args).update(overrides)
    return args


def test_features_are_lazy_grouped_masked_and_preserve_row_weights():
    calls = []

    def features(ids, mask):
        calls.append(ids.shape)
        return {"hidden_states": ids.float().unsqueeze(-1)}

    records = _records()
    original = copy.deepcopy(records)
    loader = LocalTrainingBatches(_batch_args(), features, device="cpu")
    loader.set_step(records)
    assert calls == []
    total_rows = total_supervised = 0
    scales = []
    for batch in loader:
        assert batch["input_ids"].shape == (4, 128)
        assert batch["loss_mask"][:, :4].count_nonzero() == 0
        assert (batch["loss_mask"] <= batch["attention_mask"]).all()
        total_rows += batch["input_ids"].shape[0]
        total_supervised += batch["_token_counts"][1]
        scales.append(batch["_loss_scale"])
    assert total_rows == 96 and len(calls) == 24
    assert sum(scales) == pytest.approx(1.0)
    assert total_supervised == sum(len(row["input_ids"]) - 4 for row in records)
    for before, after in zip(original, records):
        torch.testing.assert_close(before["input_ids"], after["input_ids"])
        assert before["packed_loss_mask"] == after["packed_loss_mask"]
        assert "loss_mask" not in after
    with pytest.raises(RuntimeError, match="set_step"):
        next(iter(loader))


def test_96_rows_accumulate_the_same_gradient_as_global_sample_mean():
    records = _records()
    loader = LocalTrainingBatches(_batch_args(), lambda *args: {}, device="cpu")
    loader.set_step(records)
    weight = torch.tensor(0.3, requires_grad=True)
    for batch in loader:
        mask = batch["loss_mask"].float()
        losses = ((weight * batch["input_ids"]).square() * mask).sum(1) / mask.sum(1)
        (losses.mean() * batch["_loss_scale"]).backward()
    reference = torch.tensor(0.3, requires_grad=True)
    torch.stack([(reference * row["input_ids"][4:]).square().mean() for row in records]).mean().backward()
    torch.testing.assert_close(weight.grad, reference.grad)


def test_zero_or_short_response_retains_zero_weighted_row():
    records = _records(4)
    records[0]["packed_loss_mask"] = [len(records[0]["input_ids"])]
    records[1]["packed_loss_mask"] = [len(records[1]["input_ids"]) - 1, 1]
    loader = LocalTrainingBatches(_batch_args(draft_accumulation_steps=4), lambda *args: {}, device="cpu")
    loader.set_step(records)
    batch = next(iter(loader))
    assert batch["loss_mask"][:2].count_nonzero() == 0
    assert batch["input_ids"].shape[0] == 4 and batch["_loss_scale"] == 1.0


@pytest.mark.parametrize("mutation", ["missing_mask", "empty", "overlength", "partial_batch", "multimodal"])
def test_invalid_cached_data_fails_before_any_target_forward(mutation):
    records = _records()
    if mutation == "missing_mask":
        records[-1].pop("packed_loss_mask")
    elif mutation == "empty":
        records[-1]["input_ids"] = []
    elif mutation == "overlength":
        records[-1]["input_ids"] = torch.ones(4096)
    elif mutation == "partial_batch":
        records.pop()
    else:
        records[-1]["multimodal_inputs"] = {"image": "unsupported"}
    loader = LocalTrainingBatches(_batch_args(), lambda *args: pytest.fail("target called"), device="cpu")
    with pytest.raises(ValueError):
        loader.set_step(records)


class _FakeTrainer:
    def __init__(self, *, step=0, failure_at=None):
        self.global_step = step
        self.failure_at = failure_at
        self.data_fetcher = self
        self.events = []

    def set_step(self, rows):
        self.events.append(("rows", tuple(rows)))

    def train_from_queue(self, *, step, num_batches):
        for i in range(num_batches):
            self.events.append(("micro", step, i))
            if i == self.failure_at:
                raise RuntimeError("synthetic failure")
        self.events.append(("optimizer", step))
        self.global_step += 1
        return {}

    def save_model(self, step, *, force_sync):
        assert force_sync
        self.events.append(("save", step))

    def save_draft_model_for_serving(self, path):
        self.events.append(("export", path))


def _loop_args(tmp_path, **overrides):
    args = SimpleNamespace(
        global_batch_size=96, draft_accumulation_steps=96, num_train_steps=2,
        steps_per_epoch=2, shuffle_dataset=True, seed=42, save_interval=1,
        save_per_epoch=True, num_epochs=1, use_wandb=False, max_checkpoints=0,
        output_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    vars(args).update(overrides)
    return args


def test_partial_step_failure_never_saves(tmp_path):
    trainer = _FakeTrainer(failure_at=2)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        run_local_training(_loop_args(tmp_path), range(192), trainer)
    assert not any(event[0] in ("save", "optimizer", "export") for event in trainer.events)


def test_resume_uses_epoch_cursor_and_saves_before_final_export(tmp_path):
    trainer = _FakeTrainer(step=1)
    args = _loop_args(tmp_path)
    run_local_training(args, list(range(199)), trainer)
    order = list(range(199))
    random.Random(42).shuffle(order)
    assert trainer.events[0] == ("rows", tuple(order[96:192]))
    assert trainer.events[-3:] == [("optimizer", 1), ("save", 2), ("export", str(tmp_path / "hf_final"))]


def _checkpoint_root(tmp_path):
    root = tmp_path / "checkpoints"
    directory = root / "iter_0006001"
    for part in ("model", "optimizer", "lr_scheduler"):
        (directory / part).mkdir(parents=True)
        (directory / part / ".metadata").touch()
    (directory / "rng.pt").touch()
    (directory / "meta.json").write_text(json.dumps({"global_step": 6000}))
    (root / "latest_checkpointed_iteration.txt").write_text("6001")
    return root


def test_checkpoint_preflight_matches_normal_iteration_convention(tmp_path):
    root = _checkpoint_root(tmp_path)
    args = SimpleNamespace(load_path=str(root), continual_training=False)
    assert validate_local_checkpoint(args) == 6000
    args.continual_training = True
    assert validate_local_checkpoint(args) == 0


@pytest.mark.parametrize("missing", ["optimizer/.metadata", "rng.pt", "meta.json", "model/.metadata"])
def test_incomplete_resume_fails_preflight(tmp_path, missing):
    root = _checkpoint_root(tmp_path)
    (root / "iter_0006001" / missing).unlink()
    with pytest.raises(FileNotFoundError, match="Incomplete"):
        validate_local_checkpoint(SimpleNamespace(load_path=str(root), continual_training=False))


def test_wrong_existing_resume_directory_cannot_silently_start_fresh(tmp_path):
    with pytest.raises(ValueError, match="ROOT"):
        validate_local_checkpoint(SimpleNamespace(load_path=str(tmp_path), continual_training=False))
    (tmp_path / "pytorch_model.bin").touch()
    assert validate_local_checkpoint(SimpleNamespace(load_path=str(tmp_path), continual_training=True)) == 0


def test_import_and_help_need_no_transport_or_vllm_libraries():
    script = """
import importlib.abc
import sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mooncake', 'vllm'} or fullname.startswith('angelspec.controller') or fullname.startswith('angelspec.transfer.mooncake'):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Guard())
from angelspec.train_single_gpu import main
import ray
assert not ray.is_initialized()
main(['--help'])
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, text=True, capture_output=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stderr
    assert "Single-GPU EDR and DFlash-family baseline training" in result.stdout
