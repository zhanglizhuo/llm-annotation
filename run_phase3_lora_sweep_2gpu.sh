#!/usr/bin/env bash
# Sequential LoRA sweep using 2 GPUs (DataParallel)
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

CUDA_VISIBLE_DEVICES_SET=${CUDA_VISIBLE_DEVICES_SET:-0,1}
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-128}
NUM_WORKERS=${NUM_WORKERS:-0}

OUT_BASE=${OUT_BASE:-./results/phase3_finetune/lora_sweep/$(date +%Y%m%d_%H%M%S)}
mkdir -p logs/phase3
mkdir -p "$OUT_BASE"

DATASETS=(BowTurnHead HandriseReadWrite TeacherBehavior)
PSEUDOS=(none agreement gt)

LOG=${LOG:-logs/phase3/phase3_lora_sweep_2gpu.log}
echo "Starting LoRA sweep: out=$OUT_BASE" | tee -a "$LOG"
echo "Python: $PYTHON_BIN" | tee -a "$LOG"
echo "Analysis dir: $ANALYSIS_DIR" | tee -a "$LOG"

for ds in "${DATASETS[@]}"; do
  for ps in "${PSEUDOS[@]}"; do
    echo "---- Run: $ds / $ps ----" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_SET" "$PYTHON_BIN" step3_clip_finetune.py \
      --dataset "$ds" \
      --pseudo_strategy "$ps" \
      --mode lora \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --num_workers "$NUM_WORKERS" \
      --analysis_dir "$ANALYSIS_DIR" \
      --out_dir "$OUT_BASE" 2>&1 | tee -a "$LOG"

    # short pause between runs
    sleep 3
  done
done

echo "LoRA sweep finished: out=$OUT_BASE" | tee -a "$LOG"
