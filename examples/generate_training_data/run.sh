#!/bin/bash
# Sample target rollouts on Open-PerfectBlend prompts for a training config.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

usage() {
    cat <<'USAGE'
Usage: ./examples/generate_training_data/run.sh \
    --training-config CONFIG [--epoch N] [REGENERATE_ARGS...]

  --training-config CONFIG  Training YAML (required). The target model
                            (model.target_model_path), sampling configuration
                            (dataset.target_sampling), epoch caches
                            (dataset.epoch_cache_dirs), source JSONL and token
                            limit are all read from it.
  --epoch N                 Generate only epoch N (default: every epoch in
                            dataset.epoch_cache_dirs, in order)

To change the target model or sampling configuration, edit the training
config, then generate and train with that same config.
Source rows that cannot form a valid generation prompt are skipped and listed
in .rejected.json sidecars; pass --invalid-source error to stop on them instead.
Other arguments are passed to tools/regenerate_perfectblend.py.
USAGE
}

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

TRAINING_CONFIG=""
EPOCH=""
PASSTHROUGH_ARGS=()
while (($#)); do
    case "$1" in
        --training-config|--epoch)
            if (($# < 2)); then
                echo "Missing value for $1" >&2
                exit 2
            fi
            if [[ "$1" == --training-config ]]; then TRAINING_CONFIG="$2"; else EPOCH="$2"; fi
            shift 2
            ;;
        --training-config=*) TRAINING_CONFIG="${1#*=}"; shift ;;
        --epoch=*) EPOCH="${1#*=}"; shift ;;
        --target-model|--target-model=*|--temperature|--temperature=*|--top-p|--top-p=*|--top-k|--top-k=*)
            echo "${1%%=*} is read from the training config; edit the config instead." >&2
            exit 2
            ;;
        *) PASSTHROUGH_ARGS+=("$1"); shift ;;
    esac
done
if [[ -z "$TRAINING_CONFIG" ]]; then
    echo "Missing required argument: --training-config" >&2
    usage >&2
    exit 2
fi

if [[ -n "$EPOCH" ]]; then
    EPOCHS=("$EPOCH")
else
    # Relative config paths resolve from the repository root, as in the tool.
    NUM_EPOCHS="$(python3 - "$ROOT_DIR" "$TRAINING_CONFIG" <<'PY'
import sys
from pathlib import Path

from omegaconf import OmegaConf

root, config = Path(sys.argv[1]), Path(sys.argv[2]).expanduser()
config = config if config.is_absolute() else root / config
epochs = OmegaConf.load(config).dataset.get("epoch_cache_dirs")
if not epochs:
    raise SystemExit(f"{config} does not set dataset.epoch_cache_dirs; pass --epoch")
print(len(epochs))
PY
)"
    EPOCHS=($(seq 1 "$NUM_EPOCHS"))
fi

for epoch in "${EPOCHS[@]}"; do
    echo "Generating epoch $epoch from $TRAINING_CONFIG"
    # Caller arguments come last, so an explicit --invalid-source wins.
    python3 "$ROOT_DIR/tools/regenerate_perfectblend.py" \
        --training-config "$TRAINING_CONFIG" \
        --epoch "$epoch" \
        --invalid-source skip \
        "${PASSTHROUGH_ARGS[@]}"
done
