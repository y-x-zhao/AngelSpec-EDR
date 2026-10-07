"""Opt-in real CUDA EDR/e2e/LK train/save/resume using tiny local random models.

Run with ``ANGELSPEC_RUN_SINGLE_GPU_CUDA_TESTS=1 python -m pytest -q
tests/test_single_gpu_baseline_cuda.py``. This starts no Ray/Mooncake services,
downloads nothing, and exercises the real training/loss/optimizer/checkpoint
path. Each process uses 96 cached sequences per optimizer step. EDR also
exercises the shared context cache and coalesced local teacher prefills.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    os.environ.get("ANGELSPEC_RUN_SINGLE_GPU_CUDA_TESTS") != "1",
    reason="opt-in synthetic GPU training; set ANGELSPEC_RUN_SINGLE_GPU_CUDA_TESTS=1",
)


def _write_assets(directory: Path) -> dict[str, Path]:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    from angelspec.data.utils import pack_loss_mask
    from angelspec.models.draft.dfly import DFlyConfig, DFlyDraftModel

    torch.manual_seed(730)
    target_dir = directory / "target"
    target_dir.mkdir()
    target_config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=2048,
        attention_dropout=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    Qwen3ForCausalLM(target_config).save_pretrained(target_dir)
    vocabulary = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    vocabulary.update({f"token{index}": index for index in range(4, 128)})
    tokenizer_backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        pad_token="<pad>", bos_token="<bos>", eos_token="<eos>", unk_token="<unk>",
    )
    tokenizer.save_pretrained(target_dir)

    draft_dir = directory / "draft"
    draft_dir.mkdir()
    draft_config = DFlyConfig(
        architectures=["Qwen3DFlyModel"],
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        max_position_embeddings=2048,
        num_target_layers=2,
        target_hidden_size=64,
        target_num_hidden_layers=3,
        target_layer_ids=[0, 2],
        block_size=8,
        mask_token_id=127,
        enable_hidden_correction=True,
        attention_dropout=0.0,
    )
    draft_config.save_pretrained(draft_dir)
    torch.save(DFlyDraftModel(draft_config).state_dict(), draft_dir / "pytorch_model.bin")

    cache_dir = directory / "cache"
    # The recipes read dataset.epoch_cache_dirs: ["${cache_dir}/epoch1"].
    cache_root = cache_dir / "epoch1" / "tokenized_dataset"
    cache_root.mkdir(parents=True)
    records = []
    for row in range(192):  # two optimizer steps of 96 rows
        length = 48 + row % 8
        ids = torch.tensor(
            [1] + [4 + (row * 7 + index) % 120 for index in range(length - 2)] + [2],
            dtype=torch.int32,
        )
        loss_mask = torch.tensor([0] * 8 + [1] * (length - 8), dtype=torch.long)
        records.append({
            "data_id": f"synthetic-{row}", "input_ids": ids,
            "packed_loss_mask": pack_loss_mask(loss_mask),
        })
    artifact = cache_root / "synthetic-target-T1.pt"
    torch.save(records, artifact)
    # Provenance sidecar matching the recipes' T=1 unfiltered target policy.
    artifact.with_suffix(".pt.json").write_text(json.dumps({
        "status": "complete", "artifact_name": artifact.name,
        "target_model": str(target_dir), "cached_samples": len(records),
        "sampling": {
            "temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0,
            "enable_thinking": False, "n": 1, "max_total_tokens": 128,
        },
    }))
    source = directory / "unused-source.jsonl"
    # The sole cache must be used; this deliberately has no source responses.
    source.write_text("{}\n")
    return {"target": target_dir, "draft": draft_dir, "cache": cache_dir, "source": source}


def _run_training(
    directory: Path, assets: dict[str, Path], objective: str, *, total_steps: int, resume: bool,
    num_anchors: int = 64,
) -> str:
    output = directory / "outputs"
    overrides = [
        f"model.target_model_path={assets['target']}",
        "model.target_model_backend=hf", "model.trust_remote_code=false",
        f"model.draft_model_config={assets['draft'] / 'config.json'}",
        f"dataset.train_data_path={assets['source']}",
        f"cache_dir={assets['cache']}", f"output_dir={output}",
        "training.training_num_nodes=1", "training.training_num_gpus_per_node=1",
        "training.micro_batch_size=1", "training.draft_accumulation_steps=96",
        "training.dflash_distill_cross_row_batch_size=4",
        "training.prefetch_depth=0", "training.compile_model=false",
        "training.max_seq_length=128", f"training.dflash_num_anchors={num_anchors}",
        "training.dflash_num_target_layers=2", "training.learning_rate=0.001",
        "training.min_lr=0.0001", "training.lr_total_steps=4", "training.warmup_ratio=0",
        "training.dflash_edr_vocab_chunk_size=64",
        "training.save_interval=1000",
        "training.distributed_timeout_minutes=2", "logging.report_to=none",
        f"training.num_train_steps={total_steps}",
        f"training.continual_training={'false' if resume else 'true'}",
        f"training.load_path={output / 'checkpoints' if resume else assets['draft']}",
    ]
    if objective == "lk":
        # Pure LK runs through the E2E config with only the LK loss enabled.
        overrides.extend([
            "training.dflash_e2e_tv_loss_weight=0", "training.dflash_lk_loss_weight=1.0",
            "training.dflash_distill_distribution_aware=false",
        ])
    if objective == "edr":
        overrides.extend([
            "training.dflash_edr_cross_row_batch_size=2",
            "training.dflash_edr_chunk_size=64",
            "training.dflash_edr_reuse_context_cache=true",
            "training.dflash_edr_rejection_cache_max_mb=2",
            "training.single_gpu_target_batch_size=6",
            "training.single_gpu_target_max_tokens=1024",
        ])
    env = os.environ.copy()
    env.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(directory / "hf-cache"),
        "TORCHINDUCTOR_CACHE_DIR": str(directory / "inductor-cache"),
        "TRITON_CACHE_DIR": str(directory / "triton-cache"),
        "CUDA_CACHE_PATH": str(directory / "cuda-cache"),
        "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "2",
    })
    log_path = directory / ("resume.log" if resume else "initial.log")
    with log_path.open("w") as log:
        result = subprocess.run(
            [sys.executable, "-m", "angelspec.train_single_gpu", "--config",
             str(ROOT / f"configs/vllm_qwen3_8b_dfly_{'edr' if objective == 'edr' else 'e2e'}.yaml"),
             *overrides],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=360,
        )
    text = log_path.read_text()
    assert result.returncode == 0, f"Training failed; full log at {log_path}\n{text[-16000:]}"
    return text


def _read_component(checkpoint: Path, component: str) -> dict:
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    destination = checkpoint.parent / f"{checkpoint.name}-{component}-inspection.pt"
    dcp_to_torch_save(checkpoint / component, destination)
    return torch.load(destination, map_location="cpu", weights_only=False)


def _check_saved_state(output: Path, step: int) -> dict:
    root = output / "checkpoints"
    # Normal checkpoint numbering is completed global_step + 1.
    iteration = step + 1
    assert (root / "latest_checkpointed_iteration.txt").read_text().strip() == str(iteration)
    checkpoint = root / f"iter_{iteration:07d}"
    metadata = json.loads((checkpoint / "meta.json").read_text())
    assert metadata["global_step"] == step
    assert metadata["iteration"] == iteration and metadata["world_size"] == 1
    for component in ("model", "optimizer", "lr_scheduler"):
        assert (checkpoint / component / ".metadata").is_file()
        assert list((checkpoint / component).glob("*.distcp"))
    rng = torch.load(checkpoint / "rng.pt", map_location="cpu", weights_only=True)
    assert rng["torch"].dtype == torch.uint8
    assert len(rng["cuda"]) == 1 and rng["cuda"][0].dtype == torch.uint8

    model = _read_component(checkpoint, "model")["model_state"]["model"]
    assert model and all(key.startswith("draft_model.") for key in model)
    assert all(torch.isfinite(value).all() for value in model.values())
    optimizer = _read_component(checkpoint, "optimizer")["optim_state"]
    states = optimizer["optim"]["state"]
    assert states, "Adam moments must not be silently omitted during resume"
    assert all(int(state["step"]) == step for state in states.values())
    assert all({"exp_avg", "exp_avg_sq", "step"}.issubset(state) for state in states.values())
    assert any(state["exp_avg"].count_nonzero() > 0 for state in states.values())
    assert all(torch.isfinite(state["exp_avg"]).all() for state in states.values())
    assert all(torch.isfinite(state["exp_avg_sq"]).all() for state in states.values())
    masters = optimizer["fp32_params"]
    assert masters and all(value.dtype == torch.float32 for value in masters.values())
    scheduler = _read_component(checkpoint, "lr_scheduler")["lr_scheduler_state"]["lr_scheduler"]
    assert scheduler["last_epoch"] == step
    assert scheduler["total_steps"] == 4
    export = output / "hf_final"
    assert (export / "config.json").is_file()
    exported = torch.load(export / "pytorch_model.bin", map_location="cpu", weights_only=True)
    assert exported.keys() == {key.removeprefix("draft_model.") for key in model}
    for key, value in exported.items():
        torch.testing.assert_close(value, model[f"draft_model.{key}"], atol=0, rtol=0)
    return {"model": model, "optimizer": optimizer, "scheduler": scheduler}


@pytest.mark.parametrize("objective", ["e2e", "lk", "edr"])
def test_real_single_gpu_baseline_train_save_and_resume(tmp_path, objective):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable; synthetic integration requires a local GPU")
    if torch.cuda.device_count() != 1:
        pytest.skip("Expose exactly one GPU with CUDA_VISIBLE_DEVICES for this integration test")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("GPU does not support the production BF16 baseline precision")
    assets = _write_assets(tmp_path)
    initial_log = _run_training(
        tmp_path, assets, objective, total_steps=1, resume=False,
    )
    expected_group = 2 if objective == "edr" else 4
    assert f"Global batch=96; cross-row group={expected_group}" in initial_log
    assert "Single-GPU training ready at completed step 0" in initial_log
    initial = _check_saved_state(tmp_path / "outputs", 1)

    resume_log = _run_training(
        tmp_path, assets, objective, total_steps=2, resume=True,
    )
    assert "Single-GPU training ready at completed step 1" in resume_log
    assert "Restored fp32 master params" in resume_log
    resumed = _check_saved_state(tmp_path / "outputs", 2)
    assert resumed["optimizer"]["optim"]["state"].keys() == initial["optimizer"]["optim"]["state"].keys()
    assert any(
        not torch.equal(value, initial["model"][key])
        for key, value in resumed["model"].items()
    ), "The resumed optimizer step must actually update the draft"


def test_edr_resume_matches_uninterrupted_optimizer_update(tmp_path):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        pytest.skip("requires exactly one visible CUDA device")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("requires production BF16 precision")
    assets = _write_assets(tmp_path)
    reference_dir, resumed_dir = tmp_path / "reference", tmp_path / "resumed"
    reference_dir.mkdir()
    resumed_dir.mkdir()
    _run_training(reference_dir, assets, "edr", total_steps=2, resume=False, num_anchors=16)
    expected = _check_saved_state(reference_dir / "outputs", 2)

    _run_training(resumed_dir, assets, "edr", total_steps=1, resume=False, num_anchors=16)
    _run_training(resumed_dir, assets, "edr", total_steps=2, resume=True, num_anchors=16)
    actual = _check_saved_state(resumed_dir / "outputs", 2)
    assert expected["scheduler"] == actual["scheduler"]
    for component in ("model", "optimizer"):
        torch.testing.assert_close(actual[component], expected[component], atol=2e-6, rtol=2e-5)
    for directory in (reference_dir, resumed_dir):
        rng_path = directory / "outputs/checkpoints/iter_0000003/rng.pt"
        rng = torch.load(rng_path, map_location="cpu", weights_only=True)["torch"]
        if directory == reference_dir:
            expected_rng = rng
        else:
            torch.testing.assert_close(rng, expected_rng, atol=0, rtol=0)
