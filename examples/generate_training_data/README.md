# Generate training data

The training recipes train on fresh target-model responses to the
Open-PerfectBlend (OPB) prompts, never on OPB's original assistant answers.
`run.sh` samples these target rollouts with
[`tools/regenerate_perfectblend.py`](../../tools/regenerate_perfectblend.md) for
every epoch a training config lists and writes the caches that config reads.

**Sample the target rollouts before training, from the same training config,
so they use exactly its sampling configuration** (`dataset.target_sampling`).
Training verifies each cache's `.pt.json` provenance sidecar and rejects a cache
sampled from a different target model or with a different temperature, top-p or
top-k.

## Prepare the inputs

Download the target model and the OPB source data from the repository root:

```bash
hf download Qwen/Qwen3-4B --local-dir ./target_models/Qwen3-4B   # DSpark recipes
hf download Qwen/Qwen3-8B --local-dir ./target_models/Qwen3-8B   # DFly recipes
python3 tools/prepare_perfectblend.py --num-proc 32 --skip-tokenization
```

The last command writes `dataset/Open-PerfectBlend/open-perfectblend.jsonl`.

## Generate the rollouts

Run on one GPU in an environment with vLLM:

```bash
./examples/generate_training_data/run.sh \
  --training-config CONFIG [--epoch N] [REGENERATE_ARGS...]
```

Everything is read from the training config, so the rollouts always match what
training expects:

| Config field | Used for |
| --- | --- |
| `model.target_model_path` | Target model that samples the responses |
| `dataset.target_sampling` | Sampling configuration: temperature, top-p, top-k (`min_p` must be 0, thinking off) |
| `dataset.epoch_cache_dirs` | One destination cache per epoch |
| `dataset.train_data_path` | Source OPB JSONL |
| `training.max_seq_length` | Token limit for prompt plus reply |

Without `--epoch`, every epoch in `dataset.epoch_cache_dirs` is generated in
order; `--epoch N` generates only epoch N (for example, to run epochs on
different GPUs). Each epoch has its own base seed (42, 749, 1456, 2163, 2870
for epochs 1–5). The script rejects `--target-model`, `--temperature`, `--top-p`
and `--top-k`: to change the target or sampling configuration, edit the
training config, then generate and train with that same config.

Relative paths are resolved from the repository root. Source rows that cannot
form a valid generation prompt (for example, a truncated or malformed user
turn) are skipped by default and listed in `.rejected.json` sidecars next to the
prompt shards; pass `--invalid-source error` to stop on the first one instead.
Any other argument is passed to `tools/regenerate_perfectblend.py`; see
[its documentation](../../tools/regenerate_perfectblend.md) for resume,
throughput and turn-policy options. Rerun the same command after an
interruption to resume from the durable rollout shards.

Each conversation receives one new non-thinking reply to its first user
message. Leading system instructions are kept; the source assistant answers and
later turns are discarded. A response that stops normally ends with its
sampled EOS token, not the newline the chat template appends after it.

## Paper settings

```bash
# Qwen3-4B DSpark: five epochs, T=0.7, top-p 0.8, top-k 20
CUDA_VISIBLE_DEVICES=0 ./examples/generate_training_data/run.sh \
  --training-config configs/vllm_qwen3_4b_dspark_edr.yaml

# Qwen3-8B DFly: one epoch, T=1, no top-p/top-k
CUDA_VISIBLE_DEVICES=0 ./examples/generate_training_data/run.sh \
  --training-config configs/vllm_qwen3_8b_dfly_edr.yaml
```

The EDR and E2E configs of each drafter share the target, sampling
configuration and cache directory, so either config can be passed. The 4B
caches are written to:

```text
dataset/Open-PerfectBlend/cache/Qwen3-4B__nonthink__T0.7__topP0.8__topK20__minP0/
  epoch1/tokenized_dataset/   # .pt and .pt.json
  epoch1/target_rollout_work/ # resume shards
  ...
  epoch5/tokenized_dataset/
```

and the 8B cache to `dataset/Open-PerfectBlend/cache/qwen3-8b/epoch1/`. Copy
both the `.pt` and its `.pt.json` sidecar when moving a completed cache, and
keep only one matching cache per `tokenized_dataset` directory.
