#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SEED="${SEED:-42}"

ANALYSIS_DIR="${ANALYSIS_DIR:-$ROOT_DIR/analysis_results}"
PHASE3_OUT_DIR="${PHASE3_OUT_DIR:-$ROOT_DIR/results/phase3_finetune/qwen35_27b_none}"
PHASE4_OUT_DIR="${PHASE4_OUT_DIR:-$ROOT_DIR/results/phase4_selective_annotation/qwen35_27b_none}"

mkdir -p "$ANALYSIS_DIR" "$PHASE3_OUT_DIR" "$PHASE4_OUT_DIR" "$ROOT_DIR/logs/phase34"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$ROOT_DIR/logs/phase34/phase34_qwen35_27b_none_${TS}.log"

echo "[$(date '+%F %T')] Starting phase3+phase4 run" | tee -a "$LOG_FILE"
echo "PYTHON_BIN=$PYTHON_BIN" | tee -a "$LOG_FILE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" | tee -a "$LOG_FILE"
echo "ANALYSIS_DIR=$ANALYSIS_DIR" | tee -a "$LOG_FILE"
echo "PHASE3_OUT_DIR=$PHASE3_OUT_DIR" | tee -a "$LOG_FILE"
echo "PHASE4_OUT_DIR=$PHASE4_OUT_DIR" | tee -a "$LOG_FILE"

run_cmd() {
  local label="$1"
  shift
  echo "[$(date '+%F %T')] >>> $label" | tee -a "$LOG_FILE"
  "$@" 2>&1 | tee -a "$LOG_FILE"
  echo "[$(date '+%F %T')] <<< $label" | tee -a "$LOG_FILE"
}

export CUDA_VISIBLE_DEVICES

run_cmd "step3 linear" \
  "$PYTHON_BIN" step3_clip_finetune.py \
    --dataset TeacherBehavior \
    --pseudo_strategy none \
    --mode linear \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr 5e-4 \
    --seed "$SEED" \
    --num_workers "$NUM_WORKERS" \
    --analysis_dir "$ANALYSIS_DIR" \
    --out_dir "$PHASE3_OUT_DIR"

run_cmd "step3 lora" \
  "$PYTHON_BIN" step3_clip_finetune.py \
    --dataset TeacherBehavior \
    --pseudo_strategy none \
    --mode lora \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr 1e-4 \
    --seed "$SEED" \
    --num_workers "$NUM_WORKERS" \
    --analysis_dir "$ANALYSIS_DIR" \
    --out_dir "$PHASE3_OUT_DIR"

run_cmd "step4 selective linear" \
  "$PYTHON_BIN" step4_selective_annotation.py \
    --mode linear \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr 5e-4 \
    --seed "$SEED" \
    --num_workers "$NUM_WORKERS" \
    --analysis_dir "$ANALYSIS_DIR" \
    --out_dir "$PHASE4_OUT_DIR"

run_cmd "step4 selective lora" \
  "$PYTHON_BIN" step4_selective_annotation.py \
    --mode lora \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr 1e-4 \
    --seed "$SEED" \
    --num_workers "$NUM_WORKERS" \
    --analysis_dir "$ANALYSIS_DIR" \
    --out_dir "$PHASE4_OUT_DIR"

echo "[$(date '+%F %T')] All runs completed" | tee -a "$LOG_FILE"
echo "Log: $LOG_FILE"
