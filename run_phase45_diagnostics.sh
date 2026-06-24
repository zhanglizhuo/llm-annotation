#!/bin/bash
# Run Step 4 (selective annotation) then Step 5 (retention curve)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

resolve_python_bin() {
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi
  return 1
}

resolve_latest_dir() {
  local prefix="$1"
  find . -maxdepth 1 -mindepth 1 -type d -name "${prefix}*" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -n1 \
    | cut -d' ' -f2-
}

resolve_latest_analysis_dir() {
  local structured_base="./results/phase2_filtering"
  local latest=""
  if [[ -d "$structured_base" ]]; then
    latest=$(find "$structured_base" -maxdepth 1 -mindepth 1 -type d -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr \
      | head -n1 \
      | cut -d' ' -f2-)
    if [[ -n "$latest" ]]; then
      echo "$latest"
      return
    fi
  fi
  resolve_latest_dir "analysis_results"
}

export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_URL=${HF_HUB_URL:-$HF_ENDPOINT}

PYTHON_BIN=${PYTHON_BIN:-$(resolve_python_bin)}
if [[ -z "$PYTHON_BIN" ]]; then
  echo "[ERROR] Could not find python3 or python in PATH. Set PYTHON_BIN explicitly."
  exit 1
fi

ANALYSIS_DIR=${ANALYSIS_DIR:-$(resolve_latest_analysis_dir)}
if [[ -z "$ANALYSIS_DIR" ]]; then
  echo "[ERROR] No phase2 filtering directory found. Run run_phase123_full_pipeline.sh first or set ANALYSIS_DIR."
  exit 1
fi

STEP4_OUT_DIR=${STEP4_OUT_DIR:-./results/phase4_selective_annotation/default}
STEP5_OUT_DIR=${STEP5_OUT_DIR:-./results/phase5_retention_curve/default}
STEP4_EPOCHS=${STEP4_EPOCHS:-20}
STEP5_EPOCHS=${STEP5_EPOCHS:-20}
STEP4_BATCH_SIZE_LINEAR=${STEP4_BATCH_SIZE_LINEAR:-128}
STEP4_BATCH_SIZE_LORA=${STEP4_BATCH_SIZE_LORA:-64}
STEP5_BATCH_SIZE=${STEP5_BATCH_SIZE:-128}
NUM_WORKERS=${NUM_WORKERS:-4}
SKIP_STEP4_LORA=${SKIP_STEP4_LORA:-0}
CUDA_VISIBLE_DEVICES_SET=${CUDA_VISIBLE_DEVICES_SET:-0,1}

# Fallback settings used when a step fails (e.g., worker crash / unstable loader)
RETRY_NUM_WORKERS=${RETRY_NUM_WORKERS:-0}
STEP4_RETRY_BATCH_SIZE_LINEAR=${STEP4_RETRY_BATCH_SIZE_LINEAR:-64}
STEP4_RETRY_BATCH_SIZE_LORA=${STEP4_RETRY_BATCH_SIZE_LORA:-16}
STEP5_RETRY_BATCH_SIZE=${STEP5_RETRY_BATCH_SIZE:-64}

if [[ "$CUDA_VISIBLE_DEVICES_SET" != *,* ]]; then
  echo "[ERROR] CUDA_VISIBLE_DEVICES_SET must contain two GPU ids, e.g. 0,1"
  exit 1
fi

VISIBLE_GPU_COUNT=$(CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_SET" "$PYTHON_BIN" - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)

if [[ "$VISIBLE_GPU_COUNT" -lt 2 ]]; then
  echo "[ERROR] Visible GPU count is $VISIBLE_GPU_COUNT under CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_SET; require at least 2 GPUs."
  exit 1
fi

run_step4() {
  local mode="$1"
  local batch_size="$2"
  local retry_batch_size="$3"

  echo ">>> Step 4 (${mode}) attempt 1 | batch_size=${batch_size} num_workers=${NUM_WORKERS}"
  if ! CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_SET" "$PYTHON_BIN" step4_selective_annotation.py \
      --mode "$mode" \
      --epochs "$STEP4_EPOCHS" \
      --batch_size "$batch_size" \
      --analysis_dir "$ANALYSIS_DIR" \
      --out_dir "$STEP4_OUT_DIR" \
      --num_workers "$NUM_WORKERS"; then
    echo "[WARN] Step 4 (${mode}) attempt 1 failed. Retrying with safer settings..."
    echo ">>> Step 4 (${mode}) attempt 2 | batch_size=${retry_batch_size} num_workers=${RETRY_NUM_WORKERS}"
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_SET" "$PYTHON_BIN" step4_selective_annotation.py \
      --mode "$mode" \
      --epochs "$STEP4_EPOCHS" \
      --batch_size "$retry_batch_size" \
      --analysis_dir "$ANALYSIS_DIR" \
      --out_dir "$STEP4_OUT_DIR" \
      --num_workers "$RETRY_NUM_WORKERS"
  fi
}

run_step5() {
  local batch_size="$1"
  local retry_batch_size="$2"

  echo ">>> Step 5 attempt 1 | batch_size=${batch_size} num_workers=${NUM_WORKERS}"
  if ! CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_SET" "$PYTHON_BIN" step5_retention_curve.py \
      --all \
      --epochs "$STEP5_EPOCHS" \
      --batch_size "$batch_size" \
      --analysis_dir "$ANALYSIS_DIR" \
      --out_dir "$STEP5_OUT_DIR" \
      --num_workers "$NUM_WORKERS"; then
    echo "[WARN] Step 5 attempt 1 failed. Retrying with safer settings..."
    echo ">>> Step 5 attempt 2 | batch_size=${retry_batch_size} num_workers=${RETRY_NUM_WORKERS}"
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_SET" "$PYTHON_BIN" step5_retention_curve.py \
      --all \
      --epochs "$STEP5_EPOCHS" \
      --batch_size "$retry_batch_size" \
      --analysis_dir "$ANALYSIS_DIR" \
      --out_dir "$STEP5_OUT_DIR" \
      --num_workers "$RETRY_NUM_WORKERS"
  fi
}

echo "========================================="
echo "Step4/5 run settings"
echo "HF_ENDPOINT: $HF_ENDPOINT"
echo "ANALYSIS_DIR: $ANALYSIS_DIR"
echo "PYTHON_BIN: $PYTHON_BIN"
echo "STEP4_OUT_DIR: $STEP4_OUT_DIR"
echo "STEP5_OUT_DIR: $STEP5_OUT_DIR"
echo "SKIP_STEP4_LORA: $SKIP_STEP4_LORA"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES_SET"
echo "Visible GPUs in process: $VISIBLE_GPU_COUNT"
echo "========================================="

echo ">>> Step 4.1: selective + linear"
run_step4 linear "$STEP4_BATCH_SIZE_LINEAR" "$STEP4_RETRY_BATCH_SIZE_LINEAR"

if [[ "$SKIP_STEP4_LORA" == "1" ]]; then
  echo ">>> Step 4.2: selective + lora (skipped)"
else
  echo ">>> Step 4.2: selective + lora"
  run_step4 lora "$STEP4_BATCH_SIZE_LORA" "$STEP4_RETRY_BATCH_SIZE_LORA"
fi

echo ">>> Step 5: retention curve intermediate points"
run_step5 "$STEP5_BATCH_SIZE" "$STEP5_RETRY_BATCH_SIZE"

echo "========================================="
echo "Step 4 + Step 5 completed"
echo "Step4 results: $STEP4_OUT_DIR"
echo "Step5 results: $STEP5_OUT_DIR"
echo "========================================="

