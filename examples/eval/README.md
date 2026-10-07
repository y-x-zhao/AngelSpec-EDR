# Offline MAL evaluation

This folder implements the paper's offline estimator of the mean accepted length
(MAL, Sec. 3.4). For each benchmark a target-only vLLM instance samples at least
1,500 trajectories; every draft checkpoint is then scored on the same cached
trajectories with the exact EDR dynamic program, giving paired comparisons. No
online speculative decoding is run and answer correctness is not graded.

- `eval_dp.sh` / `evaluate_dp.py`: target trajectory caching and DP scoring.
- `dp_resume.py`: durable per-dataset completion records for resuming.
- `common.py`: checkpoint resolution, draft loading and target-feature helpers.

## Prompt sets (DeepSpec / DSpark)

The nine JSONLs below contain the exact first-turn user prompts selected
by [DeepSpec's default evaluator](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/eval.py)
at revision `005e03b81cec38b7da6399833d609ee89a2587f2`:

| AngelSpec dataset | Prompts |
| --- | ---: |
| gsm8k | 500 |
| humaneval | 164 |
| mbpp | 256 |
| math500 | 500 |
| mtbench (DeepSpec `mt-bench`) | 80 |
| livecodebench | 500 |
| aime25 | 30 |
| alpaca | 500 |
| arena-hard-v2 | 500 |

Selection follows DeepSpec's seed `980406`: shuffle only datasets exceeding
the task limit, then take that many rows. Each row is converted from
`{"turns": [...]}` to `{"prompt": turns[0]}` without changing prompt text.
Second MT-Bench turns are excluded, as in DeepSpec. Sampling seeds and rollout
counts are set by this evaluator, so generated outputs can differ from DeepSpec's.

Source URLs, counts, and SHA-256 hashes are in
`angelspec/data/eval_prompts/deepspec_manifest.json`. Upstream attribution is in
`angelspec/data/eval_prompts/DeepSpec-NOTICE.md`.

## Offline EDR dynamic-program evaluation

`examples/eval/eval_dp.sh` evaluates the two pathwise estimators used by EDR. On the first run for a dataset, it
submits every prompt from each missing dataset to a target-only
vLLM instance and samples `ceil(1500 / prompt_count)` trajectories per prompt
using `--temperature`, `--top-k` and `--top-p`.

The target model (`--target-model`, or `TARGET_MODEL`), the draft architecture
config (`--draft-config`) and the target sampling configuration
(`--temperature`, `--top-p`, `--top-k`) are required; the evaluator has no
model-specific defaults. **The sampling configuration must be consistent with the
offline target rollouts:** it defines how the cached rollouts are sampled and the
target distribution that the DP scores against, so it must be identical for every
checkpoint compared on those rollouts. Each sampling configuration has its own
cache namespace, so a different configuration samples new rollouts instead of
reusing the old ones. To evaluate a recipe under its training distribution, use
the recipe's `dataset.target_sampling`. The paper's settings are:

| Target | Draft | `--draft-config` | `--temperature` | `--top-p` | `--top-k` |
| --- | --- | --- | --- | --- | --- |
| Qwen3-4B | DSpark Block7 | `angelspec/config/dspark_qwen3_4b_draft_config.json` | 0.7 | 0.8 | 20 |
| Qwen3-8B | DFly Block8 | `angelspec/config/dfly_qwen3_8b_draft_config.json` | 1 | 1 (disabled) | -1 (disabled) |

Both use non-thinking mode (the default; `--enable-thinking` turns it on) and an
unfiltered draft distribution.
`--seed` (target sampling) defaults to 1 and `--dataset-seed` (prompt
subsampling) to 42. Reused caches keep the seed recorded in their manifest.
The default selection is all **nine DeepSpec datasets**: `gsm8k`, `math500`,
`aime25`, `humaneval`, `mbpp`, `livecodebench`, `mtbench`, `alpaca`, and
`arena-hard-v2`. Their prompts match the pinned official DeepSpec selection;
see [prompt provenance](../../angelspec/data/eval_prompts/DeepSpec-NOTICE.md).
`--datasets` selects a subset; `AIME25`, `LCB`, `Alpaca`, and `Arena-Hard` are
also accepted as aliases.
`--max-new-tokens` defaults to 2048, plus one sampled terminal boundary (up to
**2049 actual new tokens**, including that boundary). EOS or the total context
limit (`--max-model-len`, default 16384) can stop generation earlier. Reused
caches keep the length limit they were created with; use a fresh
`EVAL_DP_CACHE_ROOT` to regenerate trajectories with a different limit.
Every prompt within a dataset gets the same number of samples,
and every newly created dataset cache has at least 1,500 sequences. Counts use
the prompts remaining after `--sample-size` and `--limit-per-dataset`, so even
a one-prompt selection generates 1,500 sequences. With the bundled datasets:

| Dataset | Prompts | Samples/prompt | Sequences |
| --- | ---: | ---: | ---: |
| gsm8k / math500 / livecodebench / alpaca / arena-hard-v2 (each) | 500 | 3 | 1500 |
| aime25 | 30 | 50 | 1500 |
| humaneval | 164 | 10 | 1640 |
| mbpp | 256 | 6 | 1536 |
| mtbench | 80 | 19 | 1520 |

It then writes one independent cache shard per dataset containing the five target
residual streams consumed by the draft and the final normalized hidden rows used to
reconstruct target logits. Later draft checkpoints reuse existing dataset
shards; only absent datasets are sampled and extracted. Scoring loads only the
target LM head, reconstructs logits, computes exact full-vocabulary EDR
statistics, and runs the NumPy Bellman recurrence on CPU. On cache
hits, trajectories are length-bucketed into bounded GPU batches; a background
thread loads and pins the next feature batch while the current batch runs. The
batcher also keeps every row's complete proposal horizon inside one EDR call,
avoiding repeated context-K/V scans for long trajectories. Proposal-length
bands then keep both the padded proposal and padded context rectangles dense;
the effective row count therefore decreases automatically for long outputs.

```bash
CUDA_VISIBLE_DEVICES=0 \
./examples/eval/eval_dp.sh ./outputs/qwen3-4b-dspark-edr/checkpoints/iter_0000500 \
  --target-model ./target_models/Qwen3-4B \
  --draft-config angelspec/config/dspark_qwen3_4b_draft_config.json \
  --temperature 0.7 --top-p 0.8 --top-k 20
```

The 96 GB single-GPU defaults score up to 32 trajectories together, cap the
padded context at 65,536 tokens and target LM-head projection at 8,192 rows,
use 2,048 EDR starts per detached call, and run up to eight independent NumPy
recurrences in parallel. Override these independently with
`--score-batch-size`, `--score-max-batch-tokens`,
`--score-max-target-tokens`, `--edr-chunk-size`, and `--score-dp-workers`.
Use `--no-score-prefetch` only when pinned host memory is constrained.

Target logits are temperature-scaled, filtered by target top-k/top-p, and
renormalized before calculating DP costs, acceptance probabilities and stop
mass. The draft uses the **same temperature**, without top-k/top-p filtering.
`--draft-temperature`, if given, must equal `--temperature`; mismatches are
rejected. `--temperature 0` uses exact one-hot
argmax distributions for both models, with target filters ignored. Since greedy
decoding is deterministic, it samples once and reuses identical copies for the
required per-prompt trajectory count; these are not independent stochastic draws.
D-Cut is not part of this evaluator: all seven learned proposal positions are included.
DSpark Block7 uses all seven slots; DFly Block8 slot 0 is the already-committed
input anchor and slots 1–7 are proposals. Both represent
`q[n,n+1]` through `q[n,n+7]`. If all seven survive, the target bonus advances
the next round to `n+8`.

The evaluator reports two corpus-level MAL values:

- `total generated tokens / Σ_sequences Σ_{n=0}^{L} ω_{n,n+1}`;
- `total generated tokens / Σ_sequences (1 + U_{0,1})`.

Here `ω_{0,1}=1`, so the first denominator does not add a second leading one.
The terminal EOS, stopping token, or one-token length lookahead is boundary
context and is not counted in the DP horizon length `L`. **The MAL numerator
does count this sampled token:** it uses `L + 1` actual new target tokens,
including EOS. Immediate EOS has MAL 1, not 0. Length-capped sampling still
requests up to `--max-new-tokens + 1` tokens for the DP boundary, and this actual
sampled lookahead is counted too.
Target trajectories and the distributions scored by the recurrence use the
same sampling transformation. If installed vLLM silently changes a requested
positive temperature (for example by clamping a tiny value), evaluation rejects
that setting rather than score a different distribution.

The immutable target cache is shared across checkpoint evaluations at
`eval_outputs/eval_dp_cache/<target-model-id>/<sampling-key>/<dataset>/`; set
`EVAL_DP_CACHE_ROOT` or pass `--target-cache-root` to move the root. The default
cache root stays under the repository even when the report output root changes.
The sampling key separates temperature, top-k and top-p; the target-model ID separates target
models.
Within a namespace, a run loads present dataset folders first, then samples
only absent datasets in one target-model lifecycle. Move or remove the exact
dataset folder when you intentionally want fresh trajectories.
`--sample-size`, `--limit-per-dataset`, dataset seed, and prompt-file changes
therefore affect only creation of an absent dataset folder, not reuse.
The 1,500-sequence minimum applies only to newly populated caches. Prompt limits,
seed, thinking mode and length limits do not replace an existing cache inside
the same target/temperature/top-k/top-p namespace; its manifest records the
original rollout settings.

DP results default to model/objective groups inferred from the checkpoint path:

```text
eval_outputs/qwen3-4b-spark-e2e/iter_0013956/
eval_outputs/qwen3-4b-spark-edr/iter_0013956/
eval_outputs/qwen3-8b-dfly-e2e/iter_0013997/
eval_outputs/qwen3-8b-dfly-edr/iter_0013997/
```

Training recipe directory names (`qwen3-4b-dspark`, `qwen3-4b-spark` or
`qwen3-8b-dfly`, optionally with `-cpt`, followed by `-e2e` or `-edr`) are
recognized; the 4B report directory uses `spark`. Each checkpoint keeps its own
subdirectory: the `iter_*` directory name for DCP checkpoints, or the export's
directory or file name for HF exports.
This grouping applies to both `eval_dp.sh` and direct `evaluate_dp.py` calls.

`EVAL_OUTPUT_ROOT` (or `EVAL_OUTPUT_DIR`) or `--output-root` overrides
the root exactly, producing `<override>/<checkpoint>/`; `CHECKPOINT_NAME` or
`--checkpoint-name` overrides the final subdirectory. CLI overrides take
precedence over environment variables. Unknown/renamed checkpoint paths fall
back to `eval_outputs/<checkpoint>/` rather than guessing the model or training loss.
For example, a generic `outputs/checkpoints` location needs
`EVAL_OUTPUT_ROOT=./eval_outputs/qwen3-8b-dfly-edr` to be grouped as 8B EDR.
Cache subdirectories include the target-model identity and sampling parameters,
so 4B and 8B never share target trajectories, even with identical sampling.

```bash
# Replace checkpoint paths as appropriate; use -edr instead of -e2e for EDR reports.
CUDA_VISIBLE_DEVICES=0 bash examples/eval/eval_dp.sh \
  ./outputs/qwen3-4b-dspark-e2e/checkpoints \
  --target-model ./target_models/Qwen3-4B \
  --draft-config angelspec/config/dspark_qwen3_4b_draft_config.json \
  --temperature 0.7 --top-p 0.8 --top-k 20 \
  --output-root ./eval_outputs/qwen3-4b-spark-e2e

CUDA_VISIBLE_DEVICES=0 bash examples/eval/eval_dp.sh \
  ./outputs/qwen3-8b-dfly-cpt-e2e/checkpoints \
  --target-model ./target_models/Qwen3-8B \
  --draft-config angelspec/config/dfly_qwen3_8b_draft_config.json \
  --temperature 1 --top-p 1 --top-k -1 \
  --output-root ./eval_outputs/qwen3-8b-dfly-e2e
```

Each checkpoint directory contains:

```text
dp_results.jsonl      # per-sequence expected-round estimates
dp_completed/*.json   # atomic completion record for each scored dataset
metrics.json          # dataset and global corpus sums/MAL
report_dp.md          # human-readable offline-DP report
eval_dp.log
```

**Evaluation is incremental by default.** Run the same command again to score
only unfinished datasets. Completed datasets stay in the combined report even
when `--datasets` selects only a new subset. Each dataset is saved as soon as all
of its cached sequences have been scored; an interruption can require rescoring
an unfinished dataset, but not previously completed ones. An entirely completed
rerun does not initialize the target/draft models or require a GPU.

Reuse checks the resolved draft checkpoint (file sizes/mtimes), draft architecture,
target/sampling settings, and a hash of the actual target trajectories/manifest.
Scoring batch and chunk sizes are excluded from this check.
A completed `metrics.json` + `dp_results.jsonl` pair without `dp_completed/`
records is imported when its recorded evaluation identity, cache paths,
completion time, and complete sequence records match. Unknown or changed
inputs stop the run before any result is overwritten. To reevaluate or compare
different settings, use a fresh `--checkpoint-name` (or `CHECKPOINT_NAME`).

Each dataset cache shard contains:

```text
manifest.json         # creation metadata, timings, and completeness marker
target_outputs.jsonl  # equal samples/prompt, at least 1500 per newly populated dataset
features/*.pt         # BF16 target residual streams and LM-head inputs
```

The offline report is written to `report_dp.md` and the per-checkpoint metrics to
`metrics.json`; the DP run writes its log to `eval_dp.log`.
