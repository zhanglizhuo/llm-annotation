#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs/cross_model_validation

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

PYTHON_BIN=${PYTHON_BIN:-$(resolve_python_bin)}
if [[ -z "$PYTHON_BIN" ]]; then
  echo "[ERROR] Could not find python3 or python in PATH. Set PYTHON_BIN explicitly."
  exit 1
fi

HF_ENDPOINT_VALUE=${HF_ENDPOINT_VALUE:-https://hf-mirror.com}
MODELS=${MODELS:-qwen25}
OUT_DIR=${OUT_DIR:-./results/cross_model_validation/default}
STAMP=$(date +%Y%m%d_%H%M%S)

TRAIN_LOG="logs/cross_model_validation/qwen25_teacher_train_${STAMP}.log"
TRAIN_PIDFILE="logs/cross_model_validation/qwen25_teacher_train_${STAMP}.pid"
VAL_LOG="logs/cross_model_validation/qwen25_teacher_val_${STAMP}.log"
VAL_PIDFILE="logs/cross_model_validation/qwen25_teacher_val_${STAMP}.pid"

HF_ENDPOINT="$HF_ENDPOINT_VALUE" \
HF_HUB_URL="$HF_ENDPOINT_VALUE" \
HUGGINGFACE_HUB_URL="$HF_ENDPOINT_VALUE" \
HUGGINGFACE_CO_RESOLVE_ENDPOINT="$HF_ENDPOINT_VALUE" \
CUDA_VISIBLE_DEVICES=0 \
nohup "$PYTHON_BIN" cross_model_annotate.py \
  --dataset TeacherBehavior \
  --split train \
  --models "$MODELS" \
  --out_dir "$OUT_DIR" > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$TRAIN_PIDFILE"

HF_ENDPOINT="$HF_ENDPOINT_VALUE" \
HF_HUB_URL="$HF_ENDPOINT_VALUE" \
HUGGINGFACE_HUB_URL="$HF_ENDPOINT_VALUE" \
HUGGINGFACE_CO_RESOLVE_ENDPOINT="$HF_ENDPOINT_VALUE" \
CUDA_VISIBLE_DEVICES=1 \
nohup "$PYTHON_BIN" cross_model_annotate.py \
  --dataset TeacherBehavior \
  --split val \
  --models "$MODELS" \
  --out_dir "$OUT_DIR" > "$VAL_LOG" 2>&1 &
VAL_PID=$!
echo "$VAL_PID" > "$VAL_PIDFILE"

echo "started cross-model TeacherBehavior jobs"
echo "python_bin: $PYTHON_BIN"
echo "models: $MODELS"
echo "out_dir: $OUT_DIR"
echo "train_pid: $TRAIN_PID"
echo "train_log: $TRAIN_LOG"
echo "train_pidfile: $TRAIN_PIDFILE"
echo "val_pid: $VAL_PID"
echo "val_log: $VAL_LOG"
echo "val_pidfile: $VAL_PIDFILE"
echo "monitor: tail -f $TRAIN_LOG"
echo "monitor: tail -f $VAL_LOG"