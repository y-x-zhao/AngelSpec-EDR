#!/bin/bash
# Train or resume Qwen3-8B DFly with pure e2e-TV loss.
# Uses exactly one H200, B200 or RTX PRO 6000 GPU shared by the HF target and draft.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
set -x

# Leave an unset CUDA_VISIBLE_DEVICES unset. Explicit masks (including UUIDs)
# stay unchanged.
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-4294967296}"
export ANGELSPEC_LOG_LEVEL=${ANGELSPEC_LOG_LEVEL:-INFO}
export ANGELSPEC_DENSE_FALLBACK_MAX_MASK_ELEMENTS=${ANGELSPEC_DENSE_FALLBACK_MAX_MASK_ELEMENTS:-67108864}

CONFIG_FILE="$ROOT_DIR/configs/vllm_qwen3_8b_dfly_e2e.yaml"
DRAFT_CKPT="${DRAFT_CKPT:-$ROOT_DIR/draft_checkpoints/Qwen3-8B-DFly-Block8}"
TRAIN_DATASET="${TRAIN_DATASET:-$ROOT_DIR/dataset/Open-PerfectBlend/open-perfectblend.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/qwen3-8b-dfly-cpt-e2e}"
TRAIN_LOG_FILE="${TRAIN_LOG_FILE:-$OUTPUT_DIR/training.log}"
CHECKPOINT_ROOT="$OUTPUT_DIR/checkpoints"
CHECKPOINT_TRACKER="$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt"

LOAD_PATH="$DRAFT_CKPT"
CONTINUAL_TRAINING=true
START_MODE="new e2e run from released DFly weights"

if [[ -e "$CHECKPOINT_TRACKER" ]]; then
    if [[ ! -s "$CHECKPOINT_TRACKER" ]]; then
        echo "Checkpoint tracker is empty: $CHECKPOINT_TRACKER" >&2
        exit 1
    fi

    LATEST_CHECKPOINT_TEXT="$(<"$CHECKPOINT_TRACKER")"
    if [[ ! "$LATEST_CHECKPOINT_TEXT" =~ ^[[:space:]]*([0-9]+)[[:space:]]*$ ]]; then
        echo "Checkpoint tracker does not contain an integer: $CHECKPOINT_TRACKER" >&2
        exit 1
    fi
    LATEST_CHECKPOINT_STEP=$((10#${BASH_REMATCH[1]}))
    printf -v LATEST_CHECKPOINT_DIR '%s/iter_%07d' \
        "$CHECKPOINT_ROOT" "$LATEST_CHECKPOINT_STEP"

    REQUIRED_CHECKPOINT_PATHS=(
        "$LATEST_CHECKPOINT_DIR/model/.metadata"
        "$LATEST_CHECKPOINT_DIR/optimizer/.metadata"
        "$LATEST_CHECKPOINT_DIR/lr_scheduler/.metadata"
        "$LATEST_CHECKPOINT_DIR/rng.pt"
        "$LATEST_CHECKPOINT_DIR/meta.json"
    )
    for checkpoint_path in "${REQUIRED_CHECKPOINT_PATHS[@]}"; do
        if [[ ! -e "$checkpoint_path" ]]; then
            echo "Latest checkpoint is incomplete; missing: $checkpoint_path" >&2
            exit 1
        fi
    done

    LOAD_PATH="$CHECKPOINT_ROOT"
    CONTINUAL_TRAINING=false
    START_MODE="resume checkpoint iter_$(printf '%07d' "$LATEST_CHECKPOINT_STEP")"
elif compgen -G "$CHECKPOINT_ROOT/iter_*" > /dev/null; then
    echo "Checkpoint directories exist but the latest-checkpoint tracker is missing:" >&2
    echo "  $CHECKPOINT_ROOT" >&2
    exit 1
elif [[ ! -d "$DRAFT_CKPT" ]]; then
    echo "DFly initialization checkpoint not found: $DRAFT_CKPT" >&2
    exit 1
fi

if [[ -v CUDA_VISIBLE_DEVICES ]]; then
    IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
    TOTAL_GPUS=${#GPU_ARRAY[@]}
else
    TOTAL_GPUS="$(python3 - <<'PY'
import torch

print(torch.cuda.device_count())
PY
)"
fi
if [[ "$TOTAL_GPUS" != "1" ]]; then
    echo "Expected exactly 1 CUDA device, got $TOTAL_GPUS: ${CUDA_VISIBLE_DEVICES:-unmasked}" >&2
    exit 1
fi
if [[ ! -f "$TRAIN_DATASET" ]]; then
    echo "Training dataset not found: $TRAIN_DATASET" >&2
    exit 1
fi

GPU_ARCH="$(python3 - <<'PY'
import importlib.util
import os
import sys
import torch

gpu_count = torch.cuda.device_count()
if not torch.cuda.is_available() or gpu_count != 1:
    raise SystemExit(f"Expected one visible GPU, found {gpu_count}")
visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
if visible_devices is not None and len(visible_devices.split(",")) != gpu_count:
    raise SystemExit(
        f"CUDA_VISIBLE_DEVICES selects {len(visible_devices.split(','))} devices, "
        f"but PyTorch sees {gpu_count}"
    )
capability = torch.cuda.get_device_capability(0)
if capability not in {(9, 0), (10, 0), (12, 0)}:
    raise SystemExit(
        f"Expected an SM90 H200, SM100 B200 or SM120 RTX PRO 6000 GPU; got {capability}"
    )
if capability in {(9, 0), (10, 0)}:
    gpu_name = "H200" if capability == (9, 0) else "B200"
    gpu_arch = "sm90" if capability == (9, 0) else "sm100"
    try:
        flash_spec = importlib.util.find_spec("flash_attn.cute")
    except (ImportError, ModuleNotFoundError):
        flash_spec = None
    if flash_spec is None:
        raise SystemExit(f"{gpu_name} FlexAttention requires flash_attn.cute")
    print(f"{gpu_name} preflight passed: {gpu_count} x {gpu_arch.upper()}; FlexAttention FLASH/FA4 available", file=sys.stderr)
    print(gpu_arch)
else:
    # The draft model selects its TRITON FlexAttention backend on SM120.
    print(f"Blackwell preflight passed: {gpu_count} x SM120; FlexAttention TRITON selected", file=sys.stderr)
    print("sm120")
PY
)"

GPU_LAYOUT="1 GPU: single-process HF target features + draft training (no Ray/Mooncake/vLLM)"
TRAIN_TOPOLOGY_ARGS=(
    model.target_model_backend=hf
    training.micro_batch_size=1
    training.draft_accumulation_steps=96
    training.training_num_nodes=1
    training.prefetch_depth=0
)
if [[ "$GPU_ARCH" == "sm90" ]]; then
    # Group six rows on H200; the optimizer batch remains 96 original rows.
    TRAIN_TOPOLOGY_ARGS+=(
        training.dflash_distill_cross_row_batch_size=6
        # Two vocabulary tiles for Qwen3; PRO 6000 uses the YAML's 65536.
        training.dflash_edr_vocab_chunk_size=131072
    )
elif [[ "$GPU_ARCH" == "sm100" ]]; then
    # B200 execution preset: 12 eight-row groups per 96-row optimizer step.
    TRAIN_TOPOLOGY_ARGS+=(
        training.dflash_distill_cross_row_batch_size=8
        training.dflash_edr_vocab_chunk_size=131072
        training.single_gpu_target_batch_size=8
        training.single_gpu_target_max_tokens=32768
    )
fi

export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ROOT_DIR/outputs/compiled_kernels-$GPU_ARCH}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$ROOT_DIR/outputs/cuda-compute-cache-$GPU_ARCH}"
mkdir -p "$OUTPUT_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH"

{
    echo "=============================================="
    echo "Run started: $(date --iso-8601=seconds)"
    echo "DFly CPT comparison — pure e2e-TV"
    echo "Start mode: $START_MODE"
    echo "Load path: $LOAD_PATH"
    echo "Dataset: $TRAIN_DATASET"
    echo "Global batch: 96 (microbatch 1 x 1 GPU x accumulation 96)"
    echo "GPUs: $GPU_LAYOUT"
    echo "Output: $OUTPUT_DIR"
    echo "Config: $CONFIG_FILE"
    echo "GPU architecture: $GPU_ARCH"
    echo "TorchInductor cache: $TORCHINDUCTOR_CACHE_DIR"
    echo "CUDA cache: $CUDA_CACHE_PATH"
    echo "Extra args: $*"
    echo "=============================================="
} 2>&1 | tee -a "$TRAIN_LOG_FILE"

cd "$ROOT_DIR"

python3 -m angelspec.train_single_gpu \
    --config "$CONFIG_FILE" \
    dataset.train_data_path="$TRAIN_DATASET" \
    training.load_path="$LOAD_PATH" \
    training.continual_training="$CONTINUAL_TRAINING" \
    training.training_num_gpus_per_node=1 \
    output_dir="$OUTPUT_DIR" \
    "${TRAIN_TOPOLOGY_ARGS[@]}" \
    "$@" 2>&1 | tee -a "$TRAIN_LOG_FILE"
