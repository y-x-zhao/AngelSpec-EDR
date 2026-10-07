# Regenerate Open-PerfectBlend responses

`regenerate_perfectblend.py` generates **one new assistant reply to the first user
message of each conversation**, by default for every epoch. Leading system
instructions are retained; every old assistant answer and all later turns are
discarded before tokenization and length checks.

Defaults come from `configs/vllm_qwen3_4b_dspark_edr.yaml`: local Qwen3-4B,
non-thinking, `T=0.7`, `top_p=0.8`, `top_k=20`, `min_p=0`, one response per
conversation. vLLM's native sampler produces the tokens; generated responses are never
decoded and re-tokenized. The complete conversation is at most **4096 tokens**.
Pass `--training-config` to use another recipe's target and sampling policy, e.g.
`configs/vllm_qwen3_8b_dfly_edr.yaml`.

## Commands

From the repository root, with the target model already downloaded:

```bash
# Epoch 2: generate one first-user reply per original OPB conversation.
CUDA_VISIBLE_DEVICES=0 python3 tools/regenerate_perfectblend.py --epoch 2

# Epoch 3: new responses, with base seed 1456 and a separate epoch3 cache.
CUDA_VISIBLE_DEVICES=0 python3 tools/regenerate_perfectblend.py --epoch 3

# Epoch 1: tokenize user/system messages from the OPB JSONL with 24 CPU workers.
CUDA_VISIBLE_DEVICES=0 python3 tools/regenerate_perfectblend.py --epoch 1 --num-proc 24
```

`--epoch` accepts `1` through `5` and defaults to `5`; pass it explicitly.
All epochs read the original OPB JSONL specified by
`dataset.train_data_path`; all existing assistant replies are discarded. Generation
does not require the epoch-1 cache. The selected epoch writes to its corresponding
`dataset.epoch_cache_dirs` entry (or `<cache_dir>/epochN` when no entry exists).
All epochs use the **same generation policy**, with fresh base seeds: 42 for
epoch 1, 749 for epoch 2, 1456 for epoch 3, 2163 for epoch 4 and 2870 for
epoch 5. `--seed` overrides these defaults.
Each request's seed is
`base_seed + zero-based source conversation index`. For JSONL input, the index
is the original zero-based line number,
preserved even when invalid rows are explicitly skipped.

Explicit source/destination overrides are available:

```bash
OPB_CACHE_ROOT="$PWD/dataset/Open-PerfectBlend/cache/Qwen3-4B__nonthink__T0.7__topP0.8__topK20__minP0"
CUDA_VISIBLE_DEVICES=0 python3 tools/regenerate_perfectblend.py \
  --epoch 2 \
  --source "$PWD/dataset/Open-PerfectBlend/open-perfectblend.jsonl" \
  --cache-dir "$OPB_CACHE_ROOT/epoch2" \
  --turn-policy first \
  --shard-size 4096 \
  --seed 749 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 32768
```

As an explicit opt-in alternative, `--source-cache` accepts an epoch root,
a `tokenized_dataset` directory containing exactly one `.pt`, or the `.pt` file
itself. Its complete `.pt.json` sidecar is
required and must match the configured target/sampling. Cached user/system token
IDs are preserved without re-tokenization; seeds then use the cache ordinal.
The default turn policy still selects only the first user message and leading
system context. `--source` and `--source-cache` are mutually exclusive.
Explicit sampling flags override config defaults; keep the training config in
sync. Nonzero min-p and thinking mode are unsupported. `T=0` generation is
supported, but gradient training requires a positive temperature.

### Optional multi-turn generation

Pass `--turn-policy all` to regenerate every assistant reply instead. Later replies
then see the newly generated earlier replies and retained user/system messages,
never old answers. Each turn in a conversation uses the same conversation seed.
Changing the policy requires fresh work shards.

## Attention backend and batching

The default shard size is **4096 conversations**. This controls durable prompt
and rollout groups, not GPU concurrency, which `--max-num-seqs` (default 128)
and `--max-num-batched-tokens` (default 32768) bound. Larger shards supply more
requests to vLLM's scheduler, but an interruption can replay more conversations.

`--attention-backend auto` is the default and lets vLLM select a supported
backend; the engine startup log reports the selected one. Pass
`--attention-backend FLASH_ATTN` or `--attention-backend TRITON_ATTN` to choose
explicitly; an unsupported explicit choice raises an error.

Backend and engine batch-limit changes can reuse the same durable shards;
shard-size changes require a new work manifest. Batching/backend changes can
affect floating-point rounding and sampled token identity.

## Token boundaries and masks

- A naturally stopped response includes its **actual sampled EOS**, with no
  template newline after the final reply. Genuine answer whitespace is retained.
- A length-capped reply ends its loss span at its last sampled token. No fake
  supervised EOS is appended.
- The default reply can use all remaining capacity after its formatted prompt;
  no space is reserved for discarded later turns.
- User/system messages, role headers and the deterministic empty-think prefix
  have mask zero. Only newly sampled response tokens, including EOS, are supervised.
- Incomplete user messages, malformed masks and insufficient conversation capacity
  stop preparation with the source index. To skip and audit such rows explicitly,
  use `--invalid-source skip`; rejected rows appear in `.rejected.json` sidecars.
  `--stop-after-preprocessing` validates/extracts prompts without starting vLLM.

In optional `all` mode, earlier replies reserve room for subsequent user/header
tokens and at least one sampled token per later reply. A capped intermediate
reply gets an unsupervised structural `<|im_end|>` for the next user turn.
Later prompts extend exact earlier token IDs without re-rendering them; template
separators are unsupervised. With `--source-cache`, complete empty-think
prefixes keep their source masks; newly inserted prefixes are unsupervised.

## Resume and artifacts

**Rerun the same command after interruption.** Prompt shards and rollout states
are atomically persisted. Completed shards are reused; only unfinished work is
replayed, at most one shard.
Optional `all` mode also persists each completed turn wave. A lock prevents two
generators from writing the same work directory.

```text
<sampling-cache-root>/
  epoch1/tokenized_dataset/<artifact>.pt
  epoch1/tokenized_dataset/<artifact>.pt.json
  epoch2/tokenized_dataset/<artifact>.pt
  epoch2/tokenized_dataset/<artifact>.pt.json
  epoch2/target_rollout_work/<artifact-stem>/
    manifest.json
    prompts/part-*.pt
    rollouts/part-*.pt
  epoch3/tokenized_dataset/<artifact>.pt
  epoch3/tokenized_dataset/<artifact>.pt.json
  epoch3/target_rollout_work/<artifact-stem>/
    manifest.json
    prompts/part-*.pt
    rollouts/part-*.pt
  epoch4/...  epoch5/...   # same layout
```

Artifact names include target identity, temperature/top-p/top-k/min-p, reply policy
(`firstAssistant` by default, or `allAssistant`), total length and seed.
The `.pt` uses standard AngelSpec rows (`input_ids`
and `packed_loss_mask`). Final assembly and sidecar publication are atomic and
resumable; keep the durable shards until completion. Copy both final files for
training. A completed matching output is reused without loading the model.

Source, target/tokenizer metadata, sampling, seed, reply policy, shard size and
invalid-row policy must match the work manifest; a different source requires a
new `--work-dir`. GPU type, CPU worker count and engine batch limits can change,
but different hardware or library versions may produce different samples.
The model identity does not hash all weight bytes: use a new model path if
replacing weights in place.

Training selects the cache for each epoch from `dataset.epoch_cache_dirs` (see
the [Qwen3-4B DSpark recipe](../examples/qwen3-4b-dspark-edr/README.md)). Keep all
earlier epoch caches and sidecars when resuming a later epoch: their sample
counts determine the completed-step boundaries.
