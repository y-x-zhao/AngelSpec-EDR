# Qwen3-8B DFly — CPT with EDR

Train the released
[AngelSlim/Qwen3-8B-DFly-Block8](https://huggingface.co/AngelSlim/Qwen3-8B-DFly-Block8)
draft on
[mlabonne/open-perfectblend](https://huggingface.co/datasets/mlabonne/open-perfectblend)
with the exact Expected Decoding Rounds (EDR) replacement objective. The frozen target is
[Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), loaded directly with
Hugging Face on the same single GPU as the draft. The paper's experiments used
one NVIDIA H200; one B200 or RTX PRO 6000 is also supported.

The recipe uses only repository-local assets during GPU training:

```text
AngelSpec/
├── angelspec/
├── target_models/
│   └── Qwen3-8B/
├── draft_checkpoints/
│   └── Qwen3-8B-DFly-Block8/
├── dataset/
│   └── Open-PerfectBlend/
│       ├── open-perfectblend.jsonl
│       └── cache/
└── outputs/
    └── qwen3-8b-dfly-cpt-edr/
        ├── checkpoints/
        └── training.log
```

## Hardware layout

| Visible GPUs | Target features and draft training | FlexAttention |
| --- | --- | --- |
| 1 NVIDIA H200 (SM90) | One process, shared GPU, Hugging Face target | FLASH/FA4 |
| 1 NVIDIA B200 (SM100) | One process, shared GPU, Hugging Face target | FLASH/FA4 |
| 1 NVIDIA RTX PRO 6000 Blackwell (SM120) | One process, shared GPU, Hugging Face target | TRITON |

The launcher runs `angelspec.train_single_gpu`, which keeps the frozen target
and draft together without starting Ray, Mooncake, or vLLM. Training parameters
are replicated (`fsdp_strategy: REPLICATE`). The effective global batch is 96
source conversations: microbatch 1 x 96 accumulation rows.

The launcher respects an explicit `CUDA_VISIBLE_DEVICES` mask (including GPU
UUIDs) and otherwise counts the GPUs visible to PyTorch. Before allocating
models, it rejects any visible GPU count other than one, a mask that disagrees
with PyTorch's device count, and unsupported architectures.

The launcher selects these execution sizes per architecture. They change only
how the 96 rows are grouped and how temporaries are bounded; the loss
normalization, 512 sampled anchors per horizon, LR schedule and checkpoint
format are the same on every GPU.

| Single-GPU execution setting | H200 | B200 | RTX PRO 6000 |
| --- | ---: | ---: | ---: |
| EDR rows per gradient group (`dflash_edr_cross_row_batch_size`) | 2 | 4 | 2 |
| Target prefill row limit (`single_gpu_target_batch_size`) | 6 | 8 | 4 |
| Target prefill padded-token limit (`single_gpu_target_max_tokens`) | 16384 | 32768 | 16384 |
| Total detached anchor blocks per call (`dflash_edr_chunk_size`) | 2048 | 2048 | 1024 |
| Vocabulary tile (`dflash_edr_vocab_chunk_size`) | 65536 | 65536 | 65536 |
| Boolean rejection-cache budget (`dflash_edr_rejection_cache_max_mb`) | 2048 MiB | 4096 MiB | 0 (disabled) |

With the rejection cache enabled, selected-anchor forward saves the boolean
`(p > q) & non_stopping_vocab` mask and backward reuses it instead of
recomputing teacher probabilities. Two rows of `512 x 7 x 151936` entries need
about 1.014 GiB (four rows about 2.03 GiB). The budget bounds the total across
all gradient chunks of one model call; chunks that do not fit recompute the
mask. The budget is not a cap on total GPU memory.

Any execution setting can be overridden on the command line, for example to use
the H200 sizes on a B200:

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/qwen3-8b-dfly-cpt-edr/run.sh \
  training.dflash_edr_cross_row_batch_size=2 \
  training.single_gpu_target_batch_size=6 \
  training.single_gpu_target_max_tokens=16384 \
  training.dflash_edr_rejection_cache_max_mb=2048
```

## Run

### Prepare assets on a CPU node

Use a CPU node with internet access and a filesystem shared with the GPU node.
From the repository root, first download both model repositories into the local
tree:

```bash
hf download Qwen/Qwen3-8B --local-dir ./target_models/Qwen3-8B
hf download AngelSlim/Qwen3-8B-DFly-Block8 \
  --local-dir ./draft_checkpoints/Qwen3-8B-DFly-Block8
```

Then download and normalize Open-PerfectBlend (prompts only; the original
assistant answers are not used for training):

```bash
python3 tools/prepare_perfectblend.py --num-proc 32 --skip-tokenization
```

This writes `dataset/Open-PerfectBlend/open-perfectblend.jsonl`. Re-running the
command reuses it; use `--force` only when it should be rebuilt.

### Regenerate target trajectories on a GPU node

Following the paper (Sec. 5.1), training uses a fresh Qwen3-8B response for
each prompt. Generate them **before training**, from the training config so they
use exactly its `dataset.target_sampling` (T=1, no top-p/top-k, non-thinking),
which is also the evaluation policy; see
[generate_training_data](../generate_training_data/):

```bash
CUDA_VISIBLE_DEVICES=0 ./examples/generate_training_data/run.sh \
  --training-config configs/vllm_qwen3_8b_dfly_edr.yaml
```

This writes the cache and its `.pt.json` provenance sidecar to
`dataset/Open-PerfectBlend/cache/qwen3-8b/epoch1/tokenized_dataset/`, the single
entry of `dataset.epoch_cache_dirs`. The EDR and E2E configs share the target,
sampling policy, seed and cache directory, so both recipes train on the same
regenerated responses. Training selects the cache by its sidecar and raises an
error if it is missing or was sampled differently; it does not use the original
Open-PerfectBlend answers. See
[tools/regenerate_perfectblend.md](../../tools/regenerate_perfectblend.md) for
resume and throughput options.

If the CPU and GPU nodes do not share storage, copy `target_models/`,
`draft_checkpoints/`, and `dataset/Open-PerfectBlend/` to the same paths in the
GPU-node checkout.

### Train on the GPU node

H200 and B200 use PyTorch FlexAttention's `FLASH` backend; install the
FA4/CuTeDSL dependencies:

```bash
python3 -m pip install -e ".[fa]"
```

The RTX PRO 6000 path uses the base package and the SM120 TRITON backend,
without FA4 or vLLM. The launcher checks for `flash_attn.cute` on SM90/SM100
before starting training. On B200, use a CUDA-enabled PyTorch build that
supports SM100 and an FA4 build that supports its block-sparse forward/backward
kernels; for CUDA 13 the FA4 maintainers recommend the `cu13` extra
([FA4 installation notes](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/README.md)).

TorchInductor kernels go to `outputs/compiled_kernels-<arch>/` and CUDA driver
JIT artifacts to `outputs/cuda-compute-cache-<arch>/`, where `<arch>` is `sm90`,
`sm100` or `sm120`; compiled artifacts are specific to one architecture.
`TORCHINDUCTOR_CACHE_DIR` and `CUDA_CACHE_PATH` override these paths. The
launcher sets the CUDA JIT cache limit to 4 GiB (`CUDA_CACHE_MAXSIZE=4294967296`)
when the caller has not set it.

From the repository root:

```bash
./examples/qwen3-8b-dfly-cpt-edr/run.sh
```

On a multi-GPU host, select one H200, B200 or RTX PRO 6000 explicitly:

```bash
CUDA_VISIBLE_DEVICES=0 ./examples/qwen3-8b-dfly-cpt-edr/run.sh
```

The training launcher uses the prepared local assets and does not need Hub
access. Combined training stdout and stderr are displayed in the terminal and
appended to `outputs/qwen3-8b-dfly-cpt-edr/training.log`; each invocation adds a
timestamped run header. To write a run to a separate file, set `TRAIN_LOG_FILE`:

```bash
TRAIN_LOG_FILE="$PWD/outputs/qwen3-8b-dfly-cpt-edr/training-$(date +%Y%m%d-%H%M%S).log" \
./examples/qwen3-8b-dfly-cpt-edr/run.sh
```

To isolate a run, set `OUTPUT_DIR=/path/to/run`. This moves the log and
auto-resume checkpoint root together; by default the launcher uses
`outputs/qwen3-8b-dfly-cpt-edr`, the config's `output_dir`. Relative paths are
resolved against the current directory.

The launcher automatically resumes the complete checkpoint referenced by
`outputs/qwen3-8b-dfly-cpt-edr/checkpoints/latest_checkpointed_iteration.txt`.
Resume restores the model, optimizer, learning-rate scheduler, RNG state, and
optimizer-step position. If there is no checkpoint tracker and no `iter_*`
checkpoint directory, training starts from the released DFly weights. A
malformed tracker, an incomplete tracked checkpoint, or `iter_*` directories
without a tracker stop the launcher with an error.

Resuming with a different global batch recomputes the dataset skip from the
restored step and the current global batch, so the mid-epoch data position is
approximate in that case.

### Schedule

The EDR and E2E configs follow the paper's Sec. 5.1 and Table 3: one epoch over the
regenerated cache, AdamW at learning rate `3e-5` with 4% linear warmup and then
a constant rate, gradient clipping at 1.0, and a global batch of 96.
`num_train_steps=null` lets the code compute the step count from the cache size.

The checkpoint and dataset can be overridden without editing the config:

```bash
DRAFT_CKPT=/path/to/checkpoint \
TRAIN_DATASET=/path/to/conversations.jsonl \
./examples/qwen3-8b-dfly-cpt-edr/run.sh
```

`DRAFT_CKPT` accepts a local Hugging Face export containing
`pytorch_model.bin`, an AngelSpec distributed-checkpoint root, or a Hub ID when
the GPU node has network access.
Additional dot-list overrides can be appended to the command; for example:

```bash
./examples/qwen3-8b-dfly-cpt-edr/run.sh training.learning_rate=5e-5
```

## Configuration

The launcher uses `configs/vllm_qwen3_8b_dfly_edr.yaml`. Its essential
single-GPU defaults (the H200 execution sizes) are:

```yaml
model:
  target_model_path: ./target_models/Qwen3-8B
  target_model_backend: hf
  draft_model_config: angelspec/config/dfly_qwen3_8b_draft_config.json

dataset:
  epoch_cache_dirs: ["${cache_dir}/epoch1"]
  train_data_path: ../dataset/Open-PerfectBlend/open-perfectblend.jsonl
  prompt_key: conversations
  allow_full_length_cached_sequences: true
  target_sampling: {temperature: 1.0, top_p: 1.0, top_k: -1, min_p: 0.0, enable_thinking: false}

training:
  load_path: ./draft_checkpoints/Qwen3-8B-DFly-Block8
  continual_training: true
  num_epochs: 1
  learning_rate: 3e-5
  lr_decay_style: constant
  warmup_ratio: 0.04
  micro_batch_size: 1
  draft_accumulation_steps: 96
  length_balance_optimizer_step: true
  training_num_gpus_per_node: 1
  fsdp_strategy: REPLICATE
  dflash_block_size: 7  # learned proposals; Block8 runs 7 + 1 input-anchor query slots
  dflash_query_includes_input_anchor: true
  dflash_loss_objective: edr
  dflash_num_anchors: 512
  dflash_edr_chunk_size: 2048
  dflash_edr_cross_row_batch_size: 2
  dflash_edr_vocab_chunk_size: 65536
  dflash_edr_reuse_context_cache: true
  dflash_edr_rejection_cache_max_mb: 2048
  single_gpu_target_batch_size: 6
  single_gpu_target_max_tokens: 16384
  dflash_packing: false

inference:
  store_last_hidden_states: true
  last_hidden_states_prenorm: true

output_dir: ./outputs/qwen3-8b-dfly-cpt-edr
cache_dir: ./dataset/Open-PerfectBlend/cache/qwen3-8b
```

The Block8 settings match the released checkpoint and vLLM layout:
all eight query slots run through bidirectional block attention, while slot 0
is an already committed input anchor (not an accepted draft proposal). Slots
1–7 are `q[n,n+1]` through `q[n,n+7]`; EDR applies hidden correction and the
vocabulary head only to those seven states. The all-accepted target bonus moves
the next round to `n+8`, so maximum advancement is eight tokens.

The draft is constructed from the local DFly architecture config. The launcher
imports only the released tensor weights and requires an exact key/shape match.

The launcher also passes the target backend `hf`, one training node and GPU,
accumulation 96, `prefetch_depth=0`, and the per-architecture execution sizes
above as dot-list overrides. Extra dot-list overrides come last and take
precedence over these launcher defaults.

### Target prefill coalescing

Single-GPU EDR batches the frozen teacher prefill independently of the draft
group: up to `single_gpu_target_batch_size` rows, bounded by
`single_gpu_target_max_tokens` padded tokens. Only consecutive draft groups in
the same 128-token length bucket are combined, so no additional padding tokens
are forwarded. Features stay on the GPU and are sliced back into the original
draft groups. On H200 this reduces teacher forward calls from 48 to as few as
16 per optimizer step; target-model FLOPs are the same. The feature block is
released as training advances and is never kept across optimizer updates.

The target row limit must be a multiple of the draft group size. The token cap
limits only additional coalescing; an original draft group is always admitted
whole. To disable coalescing, or reduce its memory budget:

```bash
CUDA_VISIBLE_DEVICES=0 ./examples/qwen3-8b-dfly-cpt-edr/run.sh \
  training.single_gpu_target_batch_size=0

CUDA_VISIBLE_DEVICES=0 ./examples/qwen3-8b-dfly-cpt-edr/run.sh \
  training.single_gpu_target_batch_size=4 \
  training.single_gpu_target_max_tokens=8192
```

### Batching and padding

Packing is disabled: the EDR implementation evaluates every supervised horizon,
so packing short documents into a 4096-token context adds repeated context
work, and the epoch step count is based on source rows.

`length_balance_optimizer_step: true` sorts the 96 rows of each optimizer step
by length so similar lengths share an EDR call. This changes order only inside
the optimizer step; every conversation is still used once.

The training `fill` metric is the ratio of real attention-mask tokens to padded
model slots. Each local batch is padded to the next 128-token bucket to avoid a
FlexAttention recompile for nearly every sequence length. `dflash_edr_chunk_size`
divides the reachable anchor starts inside that padded sequence and does not
affect sequence padding, so a short sequence can report low fill for any chunk
size.

### EDR estimator

EDR first evaluates all reachable round starts in a detached statistics sweep,
runs the exact CPU dynamic program, then selects distinct starts with the paper's
random-start systematic sampler. The selected contributions use their exact
Horvitz–Thompson weights. Each horizon samples at most 512 gradient anchors.
For one horizon, short samples use bounded 64/128/256/512 anchor shapes.
For multiple same-row horizons, selected anchors are compacted before
concatenation: full combined chunks contain no dummy blocks and only the final
tail pads to a 64-anchor shape boundary.

Horizons from the same input row concatenate their target rows and anchor blocks
before the vocabulary projection. The detached starts stream through calls of at
most `dflash_edr_chunk_size` total blocks, shared by the rows of a group (with two
rows, 512 blocks per row on PRO 6000 and 1024 on H200). Sampled gradient buckets
combine up to the same bound. Every Bellman recurrence, capped-PPS draw, and
Horvitz–Thompson term remains horizon-local.

`training.dflash_edr_reuse_context_cache=true` builds each layer's context K/V
once per group and reuses it for both the no-grad statistics sweep and the
gradient pass. The statistics stay detached; trainable context projections
still receive their gradients, and the cache is discarded before each optimizer
update. This saves one context projection pass per group but keeps its autograd
activations live during the statistics sweep. Set it to `false` to use separate
detached and gradient caches.

For an exact, non-sampled gradient pass, set
`dflash_edr_full_anchor_backprop: true`. This evaluates every round start,
ignores `dflash_num_anchors` for EDR gradient selection, and applies the exact
Bellman occupancy weights directly without Horvitz–Thompson correction.
`dflash_edr_chunk_size` still bounds each gradient forward. The option defaults
to `false` because full-anchor backward can be much more expensive on long
horizons.

`dflash_edr_vocab_chunk_size` streams the exact conditional non-stopping
rejection cost and realized-token acceptance over vocabulary tiles. The
estimator and its analytical gradient stay exact without full FP32 draft and
target probability tensors. The frozen target supplies final hidden states
directly; the training process reconstructs the target logits for its selected
rows and computes the FP32 log-normalizer and total stopping-token probability
locally. The detached all-start sweep uses a forward-only path. Repeated teacher
rows are exponentiated once per vocabulary tile, and on CUDA the cost,
acceptance, and analytical-backward reducers run as compiled fused kernels.

### Small target top-k

When EDR uses a positive temperature and `1 <= training.dflash_edr_top_k <= 128`
(with k smaller than the vocabulary), the trainer uses compact target support.
The rollout cache must have been generated with the same target and sampling
policy, for example:

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/qwen3-8b-dfly-cpt-edr/run.sh \
  training.dflash_edr_temperature=1 \
  training.dflash_edr_top_k=20 \
  training.dflash_edr_top_p=1
```

The teacher projects bounded position tiles and retains only supported IDs and
FP32 probabilities. Top-k=20 reserves 128 slots per target position to preserve
cutoff ties (common in BF16). Excessively large tied support uses the dense
implementation. Rejection statistics use the compact support; the analytical
backward uses one full-vocabulary softmax term plus sparse support and
realized-token corrections. The draft distribution is always the full
vocabulary: its LM-head projection, normalizer and gradients cover every token.

### Dense-mask guard

The launcher sets `ANGELSPEC_DENSE_FALLBACK_MAX_MASK_ELEMENTS=67108864`
(64 Mi elements) unless the caller overrides it. At the 4096-token context
limit, a full 4096-anchor detached or cross-horizon gradient batch exceeds this
guard and therefore runs as sparse FlexAttention (FLASH/FA4 on SM90/SM100,
TRITON on SM120). DFlash builds sparse BlockMask metadata with FlexAttention's
compiled builder.

### Smoke test

An opt-in CUDA test uses tiny locally initialized models (no downloads) to check
EDR training, checkpoint saving, and optimizer/scheduler resume:

```bash
CUDA_VISIBLE_DEVICES=0 ANGELSPEC_RUN_SINGLE_GPU_CUDA_TESTS=1 OMP_NUM_THREADS=2 \
  python -m pytest -q tests/test_single_gpu_baseline_cuda.py -k edr
```

## Outputs

Training checkpoints are written under `outputs/qwen3-8b-dfly-cpt-edr/checkpoints/`
by default; the resolved configuration and final Hugging Face export are written
alongside that directory under `outputs/qwen3-8b-dfly-cpt-edr/`. `OUTPUT_DIR`
moves all of these run artifacts.
