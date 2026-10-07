# Qwen3-8B DFly — CPT with E2E TV

Pure end-to-end TV (E2E) baseline for the released
[AngelSlim/Qwen3-8B-DFly-Block8](https://huggingface.co/AngelSlim/Qwen3-8B-DFly-Block8)
draft, using `configs/vllm_qwen3_8b_dfly_e2e.yaml`. It matches the
[EDR recipe](../qwen3-8b-dfly-cpt-edr/README.md) in initialization, data, seed,
seven learned proposals, 512 anchors, optimizer schedule and global batch.

The recipe trains on a single GPU (the paper used one NVIDIA H200). The
launcher requires exactly one visible H200 (SM90), B200 (SM100) or RTX PRO 6000
Blackwell (SM120) GPU. Explicit `CUDA_VISIBLE_DEVICES` masks, including GPU
UUIDs, are respected; when unset, the launcher inspects the GPUs visible to
PyTorch.

## Data and schedule

E2E trains on the same regenerated target responses as EDR: generate them once,
before training, with
[generate_training_data](../generate_training_data/) from the training config,
so they use its sampling configuration (see the
[EDR recipe](../qwen3-8b-dfly-cpt-edr/README.md#regenerate-target-trajectories-on-a-gpu-node)).
Both configs set the same `dataset.target_sampling` (T=1, no top-p/top-k) and
`dataset.epoch_cache_dirs: ["${cache_dir}/epoch1"]`, so the cache is selected by
its provenance sidecar. The regenerated `.pt` file and its `.pt.json` sidecar
must be present in
`dataset/Open-PerfectBlend/cache/qwen3-8b/epoch1/tokenized_dataset/`. A missing
cache is an error; the runner does not use the original Open-PerfectBlend
answers or resample responses.

The schedule follows the paper's Sec. 5.1 and Table 3: one epoch, AdamW at `3e-5` with 4%
linear warmup and then a constant rate, gradient clipping at 1.0, and a global
batch of 96.

## Run

Select one GPU on a larger host, or omit the environment variable on a host
with exactly one visible GPU:

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/qwen3-8b-dfly-cpt-e2e/run.sh
```

The launcher runs `python3 -m angelspec.train_single_gpu` with the E2E YAML. A
frozen Hugging Face target and the draft live in one process; target features
are computed from the cached conversation tokens and passed directly as GPU
tensors. No Ray, Mooncake or vLLM process is started, and no evaluation runs
during training. H200 and B200 require `flash_attn.cute` for the draft's
FLASH/FA4 FlexAttention backend; SM120 uses the Triton FlexAttention backend.

`DRAFT_CKPT`, `TRAIN_DATASET`, `OUTPUT_DIR` and trailing dot-list config
overrides work as in the EDR recipe; trailing overrides take precedence over
the launcher defaults.

## Execution sizes

The YAML and launcher set microbatch size 1 and accumulation 96, giving the
96-sample global optimizer batch. Rows are grouped into model calls; each row
keeps its own loss normalization, so grouping changes execution only.

| Setting | H200 | B200 | RTX PRO 6000 |
| --- | ---: | ---: | ---: |
| Rows per call (`dflash_distill_cross_row_batch_size`) | 6 | 8 | 4 |
| Vocabulary tile (`dflash_edr_vocab_chunk_size`) | 131072 | 131072 | 65536 |
| Target prefill rows / padded tokens (`single_gpu_target_*`) | one call per group | 8 / 32768 | one call per group |

On B200, consecutive groups with the same padded width share one target
prefill up to the row and token limits. The target token budget does not truncate sequences or split an individual
loss group. Prefetch is disabled (`prefetch_depth=0`) because the direct runner
manages its input groups. To use smaller groups, tiles and target prefills on a
B200:

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/qwen3-8b-dfly-cpt-e2e/run.sh \
  training.dflash_distill_cross_row_batch_size=4 \
  training.dflash_edr_vocab_chunk_size=65536 \
  training.single_gpu_target_batch_size=4 \
  training.single_gpu_target_max_tokens=16384
```

The first two overrides also select smaller groups and tiles on H200.

The default TorchInductor/CUDA cache suffix is `sm90` for H200, `sm100` for
B200 and `sm120` for PRO 6000. `TORCHINDUCTOR_CACHE_DIR` and `CUDA_CACHE_PATH`
override these paths; keep custom paths separated by architecture.

## Checkpoints

Checkpoints include optimizer, scheduler and RNG state and are written to
`outputs/qwen3-8b-dfly-cpt-e2e/checkpoints/`, or `$OUTPUT_DIR/checkpoints/`
when `OUTPUT_DIR` is set. Run the same launcher again to resume from the
`latest_checkpointed_iteration.txt` tracker. Checkpoints are saved at
`save_interval`, at each epoch end (`save_per_epoch`) and at the final step;
evaluate them offline with [examples/eval](../eval/).

Resuming on a different GPU type can change data grouping, random anchors and
floating-point results.

## Distribution-aware E2E loss

`training.dflash_distill_distribution_aware` defaults to `true` for E2E (this
recipe sets it explicitly). When enabled, TV uses the target distribution of
`dataset.target_sampling` (temperature, then top-k and top-p, following vLLM's
PyTorch sampling path) and an unfiltered draft at the same positive
temperature. These 8B recipes use `T=1` without top-p/top-k, so the TV target
is the unfiltered `T=1` distribution. Setting it to `false` uses full-vocabulary
target/draft `T=1`. The option does not affect rollout-cache selection, batch
size or checkpoint format.

The same option applies to the LK distillation loss (`dflash_lk_loss_weight`),
where it defaults to `false`.
