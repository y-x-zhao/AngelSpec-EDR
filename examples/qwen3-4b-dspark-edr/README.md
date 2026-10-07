# Qwen3-4B DSpark — EDR

Uses `configs/vllm_qwen3_4b_dspark_edr.yaml`. This README also covers the
setup shared with the [E2E TV recipe](../qwen3-4b-dspark-e2e/). Both recipes
use the five-layer Qwen3-4B DSpark architecture (Markov rank 256), seven
predicted tokens per block, 512 anchors, and a global optimizer batch of 96
source sequences. CE, hidden-state L1 and confidence-head auxiliary
losses are disabled; the unused confidence parameters remain in the checkpoint
but are frozen.

| Objective | Launcher | Config |
| --- | --- | --- |
| EDR | `examples/qwen3-4b-dspark-edr/run.sh` | `configs/vllm_qwen3_4b_dspark_edr.yaml` |
| E2E TV | `examples/qwen3-4b-dspark-e2e/run.sh` | `configs/vllm_qwen3_4b_dspark_e2e.yaml` |

Both recipes train for five epochs. They use 4% of **epoch 1's steps** as linear
warmup to LR `3e-5`, then a constant LR (`min_lr` is inactive with this
schedule), weight decay `0.01`, seed 42, and a fresh deterministic shuffle per
epoch (`42 + zero-based epoch`). Each epoch uses freshly regenerated target
responses; warmup runs only once. `training.override_lr_scheduler=true` applies
the configured LR schedule at the restored step on resume while keeping Adam
state and the completed-step count.

## Import the DSpark initialization on a CPU node

The checkpoint conversion does not need a GPU. Run the following from the
repository root. In a dedicated CPU environment, the required packages are:

```bash
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install safetensors huggingface_hub
```

The conversion tool needs neither the GPU training dependencies nor vLLM, Ray,
Mooncake or FlashAttention. Download the target and published DSpark weights,
then convert:

```bash
hf download Qwen/Qwen3-4B \
  --local-dir ./target_models/Qwen3-4B

hf download deepseek-ai/dspark_qwen3_4b_block7 \
  --revision 3457dff1417cb84927f6098a5fcb7cee85c934b7 \
  --local-dir ./draft_checkpoints/dspark_qwen3_4b_block7

python3 tools/import_dspark_checkpoint.py \
  --source ./draft_checkpoints/dspark_qwen3_4b_block7 \
  --target-model ./target_models/Qwen3-4B \
  --output-dir ./draft_checkpoints/dspark_qwen3_4b_block7-angelspec
```

The [HF download CLI](https://huggingface.co/docs/huggingface_hub/guides/cli#download-to-a-local-folder)
reuses completed downloads; rerun the same commands after an interruption.

`import_dspark_checkpoint.py` renames `fc`, `hidden_norm`, and `norm` to the
corresponding AngelSpec parameter names and validates every tensor name and
shape. It checks that the published frozen embedding **and** LM head exactly
equal the target weights (including Qwen3-4B's tied head), then omits the
duplicate target LM head. The output directory contains `pytorch_model.bin`,
`config.json`, and `import_manifest.json`. A completed identical conversion is
verified and reused; an interrupted conversion can be retried and leaves the
original weights unmodified.

When moving the assets to the GPU node, keep these repository-relative
locations: `target_models/Qwen3-4B` and
`draft_checkpoints/dspark_qwen3_4b_block7-angelspec`.

## Generate the target responses

Every epoch's training data are fresh Qwen3-4B responses to the Open-PerfectBlend
prompts. Generate all five epochs **before training**, from the training YAML so
they use exactly its `dataset.target_sampling` (non-thinking, `T=0.7`,
`top_p=0.8`, `top_k=20`, `min_p=0`); see
[generate_training_data](../generate_training_data/):

```bash
CUDA_VISIBLE_DEVICES=0 ./examples/generate_training_data/run.sh \
  --training-config configs/vllm_qwen3_4b_dspark_edr.yaml
```

Both recipes read the same five caches. Training verifies each `.pt.json`
provenance sidecar and rejects wrong-target, wrong-sampling and missing caches.

## Train or resume

Set `DRAFT_CKPT` to the converted initialization and launch one recipe at a time
on a single GPU:

```bash
export DRAFT_CKPT="$PWD/draft_checkpoints/dspark_qwen3_4b_block7-angelspec"
CUDA_VISIBLE_DEVICES=0 bash examples/qwen3-4b-dspark-edr/run.sh
# Run the other objective separately on the same GPU:
# CUDA_VISIBLE_DEVICES=0 bash examples/qwen3-4b-dspark-e2e/run.sh
```

`DRAFT_CKPT` must be an AngelSpec HF export containing `pytorch_model.bin` or an
AngelSpec DCP checkpoint root; DFly weights are not interchangeable. Training
does not require the source JSONL (`TRAIN_DATASET`): `dataset.epoch_cache_dirs`
selects the cache for each epoch, and all five caches must be complete before
starting.

Outputs and checkpoints go to `outputs/qwen3-4b-dspark-edr` and
`outputs/qwen3-4b-dspark-e2e` (or `OUTPUT_DIR`). A completed checkpoint tracker
under the output directory takes precedence over `DRAFT_CKPT` and restores
model, optimizer, scheduler, RNG state and the completed-step count. Epoch
lengths are `floor(cached_samples / 96)`; checkpoint progress selects the
epoch and shuffle offset. Keep every earlier epoch cache and sidecar when
resuming, because their sample counts determine the epoch boundaries, and keep
any nondefault loss or sampling overrides used for the checkpoint.

To train fewer epochs, override both fields, for example:

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/qwen3-4b-dspark-edr/run.sh \
  training.num_epochs=2 \
  'dataset.epoch_cache_dirs=["${cache_dir}/epoch1","${cache_dir}/epoch2"]'
```

## Hardware presets

Both recipes run on exactly one GPU (the paper used one H200; one B200 or RTX
PRO 6000 also works); the launchers reject any other visible GPU count. A local
HF target and the draft share the GPU, with no Ray, Mooncake or vLLM process
during training. Each `run.sh` selects these execution sizes; each source row
keeps its own loss normalization, so grouping does not change the optimizer
batch.

| GPU | EDR draft rows/group | EDR anchor chunk (total) | E2E draft rows/group | Target prefill rows / token cap |
| --- | ---: | ---: | ---: | ---: |
| H200 (SM90, ≥128 GiB) | 4 | 2048 | 8 | 8 / 32768 |
| Smaller SM90 | 2 | 2048 | 4 | 8 / 32768 |
| B200 (SM100) | 4 | 2048 | 4 | EDR 8 / 32768, E2E 4 / 16384 |
| RTX PRO 6000 (SM120) | 2 | 1024 | 4 | 4 / 16384 |

The vocabulary tile is 65536 on every GPU. Target prefills combine only
consecutive groups with the same padded width, so no padding is added. On SM90
and SM100, EDR fuses the DSpark Markov projection into the LM-head output
(E2E does so on SM90), avoiding separate full-vocabulary bias and sum tensors;
the draft distribution and gradients cover the full vocabulary.

Command-line overrides take precedence over these presets, for example:

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/qwen3-4b-dspark-edr/run.sh \
  training.dflash_edr_chunk_size=1024 \
  training.single_gpu_target_batch_size=4 \
  training.single_gpu_target_max_tokens=16384
```

CUDA caches default to `outputs/compiled_kernels-<arch>` and
`outputs/cuda-compute-cache-<arch>` (`sm90`, `sm100` or `sm120`);
`TORCHINDUCTOR_CACHE_DIR` and `CUDA_CACHE_PATH` override them.

## Sampling and loss distributions

The sampling settings above define the cached responses for both objectives.
EDR uses the filtered, renormalized target distribution in its DP statistics and
the full-vocabulary DSpark distribution at `T=0.7` (no draft top-k/top-p); its
sampling fields reference `dataset.target_sampling` directly.

E2E sets `training.dflash_distill_distribution_aware=true`: TV uses the target
distribution after temperature, top-k and top-p from `dataset.target_sampling`
(followed by renormalization, matching vLLM's PyTorch sampling path) and the
draft at the same temperature without top-k/top-p. Positive temperature is
required. Small target top-k (1–128, including 20 here) uses compact teacher
support; draft normalization and the analytical draft gradient still cover the
entire vocabulary. Setting the option to `false` uses full-vocabulary
target/draft `T=1`; it does not affect response data or cache validation.
