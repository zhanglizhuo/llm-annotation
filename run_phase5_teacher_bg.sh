#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs/phase5

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

PYTHON_BIN=${PYTHON_BIN:-$(resolve_python_bin)}
if [[ -z "$PYTHON_BIN" ]]; then
  echo "[ERROR] Could not find python3 or python in PATH. Set PYTHON_BIN explicitly."
  exit 1
fi

CUDA_VISIBLE_DEVICES_SET=${CUDA_VISIBLE_DEVICES_SET:-0,1}
ANALYSIS_DIR=${ANALYSIS_DIR:-$(resolve_latest_analysis_dir)}
if [[ -z "$ANALYSIS_DIR" ]]; then
  echo "[ERROR] No phase2 filtering directory found. Run run_phase123_full_pipeline.sh first or set ANALYSIS_DIR."
  exit 1
fi

OUT_DIR=${OUT_DIR:-./results/phase5_retention_curve/default}
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-128}
NUM_WORKERS=${NUM_WORKERS:-4}

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/phase5/phase5_teacher_${STAMP}.log"
PIDFILE="logs/phase5/phase5_teacher_${STAMP}.pid"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_SET" nohup "$PYTHON_BIN" step5_retention_curve.py \
  --dataset TeacherBehavior \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --analysis_dir "$ANALYSIS_DIR" \
  --out_dir "$OUT_DIR" > "$LOG" 2>&1 &

echo $! > "$PIDFILE"
echo "started TeacherBehavior retention curve"
echo "pid: $(cat "$PIDFILE")"
echo "log: $LOG"
echo "pidfile: $PIDFILE"
echo "monitor: tail -f $LOG"