#!/usr/bin/env bash
# Batched-target, offline exact-EDR DP evaluation on one GPU.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
EVALUATOR="$SCRIPT_DIR/evaluate_dp.py"

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'USAGE'
Usage: ./examples/eval/eval_dp.sh DRAFT_CHECKPOINT [EVALUATOR_ARGS...]

Required evaluator arguments:
  --target-model PATH    Target model (or set TARGET_MODEL)
  --draft-config PATH    Draft architecture config JSON
  --temperature T --top-p P --top-k K
                         Target sampling configuration. It must match the
                         configuration of the target rollouts being scored
                         (for the paper recipes, dataset.target_sampling).

Examples:
  ./examples/eval/eval_dp.sh ./draft_checkpoints/dspark_qwen3_4b_block7-angelspec \
    --target-model ./target_models/Qwen3-4B \
    --draft-config angelspec/config/dspark_qwen3_4b_draft_config.json \
    --temperature 0.7 --top-p 0.8 --top-k 20
  ./examples/eval/eval_dp.sh ./outputs/qwen3-8b-dfly-cpt-edr/checkpoints \
    --target-model ./target_models/Qwen3-8B \
    --draft-config angelspec/config/dfly_qwen3_8b_draft_config.json \
    --temperature 1 --top-p 1 --top-k -1 --datasets gsm8k math500

Thinking is disabled unless --enable-thinking is given.
The context limit defaults to 16384 tokens; override with --max-model-len.
New caches default to --max-new-tokens 2048 plus one sampled DP boundary.
Existing caches keep their original lengths; use a fresh EVAL_DP_CACHE_ROOT
to regenerate with a different limit.
By default, all nine DeepSpec datasets are selected: gsm8k, math500, aime25,
humaneval, mbpp, livecodebench, mtbench, alpaca, and arena-hard-v2.
Completed datasets for the same checkpoint/settings/cache are skipped on reruns.
Each completed dataset is saved immediately; the combined report retains earlier
results. Use a new CHECKPOINT_NAME to intentionally score everything again.
The draft uses the same temperature with no top-k/top-p filtering. Each target
model/sampling configuration has an independent cache per dataset, containing
ceil(1500 / prompt_count) trajectories per
prompt (at least 1500 sequences) and their target features. MAL counts generated
tokens including the sampled EOS/stop/length boundary. Existing dataset caches
are reused; all missing datasets
are populated together before length-bucketed GPU scoring and parallel NumPy DP.

Environment:
  CUDA_VISIBLE_DEVICES   Single GPU id (default: 0)
  TARGET_MODEL           Target model path (used when --target-model is absent)
  EVAL_OUTPUT_ROOT       Report root override (default: ./eval_outputs/<model-objective>)
  EVAL_DP_CACHE_ROOT     Shared target cache root (default: ./eval_outputs/eval_dp_cache)
  CHECKPOINT_NAME        Optional output subdirectory override

Known training recipe paths are grouped, for example:
  eval_outputs/qwen3-4b-spark-e2e/iter_0013956/
  eval_outputs/qwen3-8b-dfly-edr/iter_0013997/
Unrecognized checkpoint paths use eval_outputs/<checkpoint>/;
set EVAL_OUTPUT_ROOT explicitly to name the objective for generic locations.
USAGE
    exit 0
fi

DRAFT_CHECKPOINT="$1"
shift
EVALUATOR_ARGS=("$@")
OUTPUT_ROOT_OVERRIDE="${EVAL_OUTPUT_ROOT:-${EVAL_OUTPUT_DIR:-}}"

REPORT_FILENAME="report_dp.md"
for ((arg_index = 0; arg_index < ${#EVALUATOR_ARGS[@]}; arg_index++)); do
    case "${EVALUATOR_ARGS[$arg_index]}" in
        --output-root|--checkpoint-name)
            if ((arg_index + 1 >= ${#EVALUATOR_ARGS[@]})); then
                echo "Missing value for ${EVALUATOR_ARGS[$arg_index]}" >&2
                exit 2
            fi
            if [[ "${EVALUATOR_ARGS[$arg_index]}" == "--output-root" ]]; then
                OUTPUT_ROOT_OVERRIDE="${EVALUATOR_ARGS[$((arg_index + 1))]}"
            else
                CHECKPOINT_NAME="${EVALUATOR_ARGS[$((arg_index + 1))]}"
            fi
            ;;
        --output-root=*) OUTPUT_ROOT_OVERRIDE="${EVALUATOR_ARGS[$arg_index]#*=}" ;;
        --checkpoint-name=*) CHECKPOINT_NAME="${EVALUATOR_ARGS[$arg_index]#*=}" ;;
    esac
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROOT_DIR"
if [[ -n "$OUTPUT_ROOT_OVERRIDE" ]]; then
    EVAL_OUTPUT_ROOT="$OUTPUT_ROOT_OVERRIDE"
else
    EVAL_OUTPUT_ROOT="$(python3 "$EVALUATOR" "${EVALUATOR_ARGS[@]}" --resolve-output-root "$DRAFT_CHECKPOINT")"
fi
if [[ "$EVAL_OUTPUT_ROOT" != /* ]]; then
    EVAL_OUTPUT_ROOT="$ROOT_DIR/$EVAL_OUTPUT_ROOT"
fi

if [[ -z "${CHECKPOINT_NAME:-}" ]]; then
    CHECKPOINT_NAME="$(python3 "$EVALUATOR" \
        --resolve-checkpoint-name "$DRAFT_CHECKPOINT")"
fi
if [[ ! "$CHECKPOINT_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || \
    [[ "$CHECKPOINT_NAME" == "." || "$CHECKPOINT_NAME" == ".." ]]; then
    echo "Invalid CHECKPOINT_NAME: $CHECKPOINT_NAME" >&2
    exit 2
fi
RUN_DIR="$EVAL_OUTPUT_ROOT/$CHECKPOINT_NAME"
mkdir -p "$RUN_DIR"

{
    echo "Batched-target offline EDR DP evaluation"
    echo "  checkpoint: $DRAFT_CHECKPOINT"
    echo "  target:     ${TARGET_MODEL:-from --target-model}; CLI overrides apply"
    echo "  sampling:   from --temperature/--top-p/--top-k"
    echo "  GPU:        $CUDA_VISIBLE_DEVICES"
    echo "  report:     $RUN_DIR/$REPORT_FILENAME"
    echo "  resume:     skip completed matching datasets; preserve combined results"

    python3 "$EVALUATOR" \
        --draft-checkpoint "$DRAFT_CHECKPOINT" \
        "${EVALUATOR_ARGS[@]}" \
        --output-root "$EVAL_OUTPUT_ROOT" \
        --checkpoint-name "$CHECKPOINT_NAME"
} 2>&1 | tee -a "$RUN_DIR/eval_dp.log"
