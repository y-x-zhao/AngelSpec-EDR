#!/bin/bash
# Train the released Qwen3-8B DFly draft with the exact Expected Decoding Rounds objective.
#
# Uses exactly one H200, B200 or RTX PRO 6000 GPU shared by the HF target and draft.
#
# Usage:
#   ./examples/qwen3-8b-dfly-cpt-edr/run.sh [EXTRA_ARGS...]

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
set -x

# Explicit caller masks (including UUIDs) stay unchanged.
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-4294967296}"
export ANGELSPEC_LOG_LEVEL=${ANGELSPEC_LOG_LEVEL:-INFO}
# Keep production EDR shapes on sparse FlexAttention. At the 4096-token limit,
# a full 4096-anchor detached or cross-horizon gradient batch exceeds this
# 64-Mi-element dense-mask safety guard.
export ANGELSPEC_DENSE_FALLBACK_MAX_MASK_ELEMENTS=${ANGELSPEC_DENSE_FALLBACK_MAX_MASK_ELEMENTS:-67108864}

CONFIG_FILE="$ROOT_DIR/configs/vllm_qwen3_8b_dfly_edr.yaml"
DRAFT_CKPT="${DRAFT_CKPT:-$ROOT_DIR/draft_checkpoints/Qwen3-8B-DFly-Block8}"
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

LOAD_PATH="$DRAFT_CKPT"
CONTINUAL_TRAINING=true
START_MODE="new EDR run from released DFly weights"

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

# Check the actual CUDA view before allocating model weights.
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
    arch = "sm90" if capability == (9, 0) else "sm100"
    try:
        flash_spec = importlib.util.find_spec("flash_attn.cute")
    except (ImportError, ModuleNotFoundError):
        flash_spec = None
    if flash_spec is None:
        raise SystemExit(f"{gpu_name} FlexAttention requires flash_attn.cute")
    print(f"{gpu_name} preflight passed: {gpu_count} x {arch.upper()}; FlexAttention FLASH/FA4 available", file=sys.stderr)
    print(arch)
else:
    # DFlash selects its TRITON FlexAttention backend on SM120.
    print("Blackwell preflight passed: 1 x SM120; FlexAttention TRITON selected", file=sys.stderr)
    print("sm120")
PY
)"

GPU_LAYOUT="1 GPU: single-process HF target features + draft training (no Ray/Mooncake/vLLM)"
EDR_CROSS_ROW_SIZE=2
TARGET_PREFILL_ROWS=4
TARGET_PREFILL_MAX_TOKENS=16384
EDR_CHUNK_SIZE=1024
EDR_REJECTION_CACHE_MB=0
if [[ "$GPU_ARCH" == "sm90" ]]; then
    TARGET_PREFILL_ROWS=6
    # 2048-block detached chunks. With two rows x 512 sampled anchors, a
    # gradient call has at most 1024 blocks.
    EDR_CHUNK_SIZE=2048
    # ~1 GiB for 2 x 512 x 7 x 151936 boolean entries; 2 GiB bounds the total
    # across all gradient chunks of a model call.
    EDR_REJECTION_CACHE_MB=2048
elif [[ "$GPU_ARCH" == "sm100" ]]; then
    # B200 (180 GB): four-row groups, i.e. 24 draft groups per 96-row
    # optimizer step. Detached anchor and vocabulary tiles stay bounded
    # instead of scaling with the row count.
    EDR_CROSS_ROW_SIZE=4
    TARGET_PREFILL_ROWS=8
    TARGET_PREFILL_MAX_TOKENS=32768
    EDR_CHUNK_SIZE=2048
    # Four rows x 512 sampled anchors need about 2 GiB of boolean masks.
    EDR_REJECTION_CACHE_MB=4096
fi
TRAIN_TOPOLOGY_ARGS=(
    model.target_model_backend=hf
    training.micro_batch_size=1
    training.draft_accumulation_steps=96
    training.training_num_nodes=1
    training.fsdp_strategy=REPLICATE
    training.prefetch_depth=0
    # Group execution only; keep 96 independently normalized source rows.
    training.dflash_edr_cross_row_batch_size="$EDR_CROSS_ROW_SIZE"
    # Bound temporaries while target and draft share the GPU.
    training.dflash_edr_chunk_size="$EDR_CHUNK_SIZE"
    training.dflash_edr_vocab_chunk_size=65536
    training.dflash_edr_reuse_context_cache=true
    training.dflash_edr_rejection_cache_max_mb="$EDR_REJECTION_CACHE_MB"
    # Coalesce only equal-width teacher groups; no additional padding work.
    training.single_gpu_target_batch_size="$TARGET_PREFILL_ROWS"
    training.single_gpu_target_max_tokens="$TARGET_PREFILL_MAX_TOKENS"
)

# Keep Hopper, data-center Blackwell and SM120 artifacts separate.
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ROOT_DIR/outputs/compiled_kernels-$GPU_ARCH}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$ROOT_DIR/outputs/cuda-compute-cache-$GPU_ARCH}"
mkdir -p "$OUTPUT_DIR" "$(dirname -- "$TRAIN_LOG_FILE")" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH"
if [[ "$GPU_ARCH" == "sm90" ]]; then
    FLEX_BACKEND="FLASH/FA4 on SM90"
elif [[ "$GPU_ARCH" == "sm100" ]]; then
    FLEX_BACKEND="FLASH/FA4 on SM100"
else
    FLEX_BACKEND="TRITON on SM120"
fi

{
    echo "=============================================="
    echo "Run started: $(date --iso-8601=seconds)"
    echo "DFly CPT-EDR — Qwen3-8B"
    echo "Start mode: $START_MODE"
    echo "Load path: $LOAD_PATH"
    echo "Dataset: $TRAIN_DATASET (epochs from config/overrides)"
    echo "Global batch: 96 (microbatch 1 x 1 GPU x accumulation 96)"
    echo "GPUs: $GPU_LAYOUT"
    echo "Output: $OUTPUT_DIR"
    echo "=============================================="
    echo "Config: $CONFIG_FILE"
    echo "Training log: $TRAIN_LOG_FILE"
    echo "Dense SDPA mask limit: $ANGELSPEC_DENSE_FALLBACK_MAX_MASK_ELEMENTS elements"
    echo "TorchInductor cache: $TORCHINDUCTOR_CACHE_DIR"
    echo "CUDA JIT cache: $CUDA_CACHE_PATH"
    echo "CUDA JIT cache limit: $CUDA_CACHE_MAXSIZE bytes"
    echo "FlexAttention backend: $FLEX_BACKEND"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unmasked}"
    echo "Extra args: $*"
    echo "=============================================="
} 2>&1 | tee -a "$TRAIN_LOG_FILE"

cd "$ROOT_DIR"

python3 -m angelspec.train_single_gpu \
    --config "$CONFIG_FILE" \
    dataset.train_data_path="$TRAIN_DATASET" \
    training.training_num_gpus_per_node=1 \
    training.load_path="$LOAD_PATH" \
    training.continual_training="$CONTINUAL_TRAINING" \
    output_dir="$OUTPUT_DIR" \
    "${TRAIN_TOPOLOGY_ARGS[@]}" \
    "$@" 2>&1 | tee -a "$TRAIN_LOG_FILE"
