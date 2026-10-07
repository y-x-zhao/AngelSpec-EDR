#!/bin/bash
# Qwen3-4B DSpark with pure E2E-TV loss. Runs on exactly one H200, B200 or RTX PRO 6000.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
DSPARK_OBJECTIVE=e2e

export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-4294967296}"
export ANGELSPEC_LOG_LEVEL=${ANGELSPEC_LOG_LEVEL:-INFO}
export ANGELSPEC_DENSE_FALLBACK_MAX_MASK_ELEMENTS=${ANGELSPEC_DENSE_FALLBACK_MAX_MASK_ELEMENTS:-67108864}

CONFIG_FILE="$ROOT_DIR/configs/vllm_qwen3_4b_dspark_${DSPARK_OBJECTIVE}.yaml"
TRAIN_DATASET="${TRAIN_DATASET:-$ROOT_DIR/dataset/Open-PerfectBlend/open-perfectblend.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/$(basename "$SCRIPT_DIR")}"
TRAIN_LOG_FILE="${TRAIN_LOG_FILE:-$OUTPUT_DIR/training.log}"
# Resolve caller-relative environment paths before the later cd to the repo.
for path_name in TRAIN_DATASET OUTPUT_DIR TRAIN_LOG_FILE; do
    if [[ "${!path_name}" != /* ]]; then
        printf -v "$path_name" '%s/%s' "$PWD" "${!path_name}"
    fi
done
CHECKPOINT_ROOT="$OUTPUT_DIR/checkpoints"
CHECKPOINT_TRACKER="$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt"
LOAD_PATH="${DRAFT_CKPT:-}"
CONTINUAL_TRAINING=true
START_MODE="new $DSPARK_OBJECTIVE run from supplied DSpark weights"

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
    printf -v LATEST_CHECKPOINT_DIR '%s/iter_%07d' "$CHECKPOINT_ROOT" "$LATEST_CHECKPOINT_STEP"
    for checkpoint_path in \
        "$LATEST_CHECKPOINT_DIR/model/.metadata" \
        "$LATEST_CHECKPOINT_DIR/optimizer/.metadata" \
        "$LATEST_CHECKPOINT_DIR/lr_scheduler/.metadata" \
        "$LATEST_CHECKPOINT_DIR/rng.pt" \
        "$LATEST_CHECKPOINT_DIR/meta.json"; do
        if [[ ! -e "$checkpoint_path" ]]; then
            echo "Latest checkpoint is incomplete; missing: $checkpoint_path" >&2
            exit 1
        fi
    done
    LOAD_PATH="$CHECKPOINT_ROOT"
    CONTINUAL_TRAINING=false
    START_MODE="resume checkpoint iter_$(printf '%07d' "$LATEST_CHECKPOINT_STEP")"
elif compgen -G "$CHECKPOINT_ROOT/iter_*" > /dev/null; then
    echo "Checkpoint directories exist but the latest-checkpoint tracker is missing: $CHECKPOINT_ROOT" >&2
    exit 1
elif [[ -z "$LOAD_PATH" || ! -d "$LOAD_PATH" ]]; then
    echo "Set DRAFT_CKPT to a local Qwen3-4B-compatible DSpark initialization checkpoint." >&2
    echo "No default released DSpark checkpoint is assumed; existing output checkpoints resume automatically." >&2
    exit 1
elif [[ "$LOAD_PATH" != /* ]]; then
    LOAD_PATH="$PWD/$LOAD_PATH"
fi
# The recipes train from dataset.epoch_cache_dirs, so the source JSONL may be
# absent; the driver validates every configured cache sidecar before startup.

# Respect caller masks, including GPU UUIDs, and validate against PyTorch's
# actual visible devices.
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
GPU_INFO="$(python3 - <<'PY'
import importlib.util
import os
import sys
import torch

count = torch.cuda.device_count()
if not torch.cuda.is_available() or count != 1:
    raise SystemExit(f"Expected one visible GPU, found {count}")
mask = os.environ.get("CUDA_VISIBLE_DEVICES")
if mask is not None and len(mask.split(",")) != count:
    raise SystemExit(f"CUDA_VISIBLE_DEVICES selects {len(mask.split(','))} GPUs, but PyTorch sees {count}")
cap = torch.cuda.get_device_capability(0)
if cap not in {(9, 0), (10, 0), (12, 0)}:
    raise SystemExit(f"Expected an SM90 H200, SM100 B200 or SM120 RTX PRO 6000 GPU; got {cap}")
arch = {(9, 0): "sm90", (10, 0): "sm100", (12, 0): "sm120"}[cap]
if arch in {"sm90", "sm100"}:
    try:
        flash = importlib.util.find_spec("flash_attn.cute")
    except (ImportError, ModuleNotFoundError):
        flash = None
    if flash is None:
        raise SystemExit(f"{arch.upper()} FlexAttention requires flash_attn.cute")
print(f"GPU preflight passed: {count} x {arch.upper()}", file=sys.stderr)
print(arch, torch.cuda.get_device_properties(0).total_memory // (1024**2))
PY
)"
read -r GPU_ARCH GPU_MEMORY_MIB <<< "$GPU_INFO"

GPU_LAYOUT="1 GPU: single-process HF target features + DSpark training (no Ray/Mooncake/vLLM)"
EDR_CROSS_ROW_SIZE=2
DISTILL_CROSS_ROW_SIZE=4
EDR_CHUNK_SIZE=1024
TARGET_PREFILL_ROWS=4
TARGET_PREFILL_MAX_TOKENS=16384
if [[ "$DSPARK_OBJECTIVE" == edr && "$GPU_ARCH" == sm100 ]]; then
    # B200: 24 four-row draft groups per optimizer step. The 96-row batch
    # and 512-anchor cap per horizon are the same on every GPU.
    EDR_CROSS_ROW_SIZE=4
    EDR_CHUNK_SIZE=2048
    TARGET_PREFILL_ROWS=8
    TARGET_PREFILL_MAX_TOKENS=32768
elif [[ "$DSPARK_OBJECTIVE" == edr && "$GPU_ARCH" == sm90 ]]; then
    # SM90 with >= 128 GiB (H200): 24 four-row draft groups per step.
    # Smaller SM90 cards use 48 two-row groups.
    if [[ "$GPU_MEMORY_MIB" -ge 131072 ]]; then
        EDR_CROSS_ROW_SIZE=4
    fi
    EDR_CHUNK_SIZE=2048
    TARGET_PREFILL_ROWS=8
    TARGET_PREFILL_MAX_TOKENS=32768
elif [[ "$GPU_ARCH" == sm90 ]]; then
    # E2E on SM90 with >= 128 GiB (H200): 12 eight-row draft calls per step.
    # Smaller SM90 cards use four-row groups.
    TARGET_PREFILL_ROWS=8
    TARGET_PREFILL_MAX_TOKENS=32768
    if [[ "$GPU_MEMORY_MIB" -ge 131072 ]]; then
        DISTILL_CROSS_ROW_SIZE=8
    fi
fi
TRAIN_TOPOLOGY_ARGS=(
    model.target_model_backend=hf
    training.training_num_nodes=1
    training.fsdp_strategy=REPLICATE
    training.prefetch_depth=0
    training.dflash_edr_vocab_chunk_size=65536
    training.single_gpu_target_batch_size="$TARGET_PREFILL_ROWS"
    training.single_gpu_target_max_tokens="$TARGET_PREFILL_MAX_TOKENS"
)
if [[ "$DSPARK_OBJECTIVE" == edr ]]; then
    # Command-line overrides are appended later and take precedence.
    TRAIN_TOPOLOGY_ARGS+=(
        training.dflash_edr_cross_row_batch_size="$EDR_CROSS_ROW_SIZE"
        training.dflash_edr_chunk_size="$EDR_CHUNK_SIZE"
        training.dflash_edr_reuse_context_cache=true
        training.dflash_edr_rejection_cache_max_mb=0
    )
else
    TRAIN_TOPOLOGY_ARGS+=(training.dflash_distill_cross_row_batch_size="$DISTILL_CROSS_ROW_SIZE")
fi

# Reuse kernels only within their GPU architecture. Explicit cache paths win.
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ROOT_DIR/outputs/compiled_kernels-$GPU_ARCH}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$ROOT_DIR/outputs/cuda-compute-cache-$GPU_ARCH}"
for path_name in TORCHINDUCTOR_CACHE_DIR CUDA_CACHE_PATH; do
    if [[ "${!path_name}" != /* ]]; then
        printf -v "$path_name" '%s/%s' "$PWD" "${!path_name}"
    fi
done
mkdir -p "$OUTPUT_DIR" "$(dirname -- "$TRAIN_LOG_FILE")" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH"
{
    echo "Run started: $(date --iso-8601=seconds)"
    echo "Qwen3-4B DSpark — $DSPARK_OBJECTIVE"
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
} 2>&1 | tee -a "$TRAIN_LOG_FILE"

cd "$ROOT_DIR"

python3 -m angelspec.train_single_gpu \
    --config "$CONFIG_FILE" \
    dataset.train_data_path="$TRAIN_DATASET" \
    training.load_path="$LOAD_PATH" \
    training.continual_training="$CONTINUAL_TRAINING" \
    training.training_num_gpus_per_node=1 \
    training.micro_batch_size=1 \
    training.draft_accumulation_steps=96 \
    output_dir="$OUTPUT_DIR" \
    "${TRAIN_TOPOLOGY_ARGS[@]}" \
    "$@" 2>&1 | tee -a "$TRAIN_LOG_FILE"
