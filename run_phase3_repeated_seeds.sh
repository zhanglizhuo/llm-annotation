#!/bin/bash

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
export HUGGINGFACE_HUB_URL=${HUGGINGFACE_HUB_URL:-$HF_ENDPOINT}
export HF_HUB_URL=${HF_HUB_URL:-$HF_ENDPOINT}

PYTHON_BIN=${PYTHON_BIN:-$(resolve_python_bin)}
if [[ -z "$PYTHON_BIN" ]]; then
  echo "[ERROR] Could not find python3 or python in PATH. Set PYTHON_BIN explicitly."
  exit 1
fi

RUN_TAG=${RUN_TAG:-repeated_seeds_$(date +%Y%m%d_%H%M%S)}
ANALYSIS_DIR=${ANALYSIS_DIR:-$(resolve_latest_analysis_dir)}
if [[ -z "$ANALYSIS_DIR" ]]; then
  echo "[ERROR] No phase2 filtering directory found. Run run_phase123_full_pipeline.sh first or set ANALYSIS_DIR."
  exit 1
fi

OUT_ROOT=${OUT_ROOT:-./results/phase3_finetune/repeated_seeds/$RUN_TAG}
SEEDS=${SEEDS:-43 44 45 46 47}
DATASETS=${DATASETS:-BowTurnHead HandriseReadWrite TeacherBehavior}
STRATEGIES=${STRATEGIES:-none agreement gt}
MODES=${MODES:-linear}
EPOCHS=${EPOCHS:-20}
LINEAR_BATCH_SIZE=${LINEAR_BATCH_SIZE:-128}
LORA_BATCH_SIZE=${LORA_BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-4}
RETRY_NUM_WORKERS=${RETRY_NUM_WORKERS:-0}
LINEAR_RETRY_BATCH_SIZE=${LINEAR_RETRY_BATCH_SIZE:-64}
LORA_RETRY_BATCH_SIZE=${LORA_RETRY_BATCH_SIZE:-16}
CUDA_VISIBLE_DEVICES_LINEAR=${CUDA_VISIBLE_DEVICES_LINEAR:-0,1}
CUDA_VISIBLE_DEVICES_LORA=${CUDA_VISIBLE_DEVICES_LORA:-0,1}
LOG_DIR=${LOG_DIR:-./logs/phase3}

mkdir -p "$OUT_ROOT" "$LOG_DIR"
LOG_FILE="$LOG_DIR/phase3_repeated_${RUN_TAG}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================="
echo "Repeated-seed run tag: $RUN_TAG"
echo "Analysis dir: $ANALYSIS_DIR"
echo "Out root: $OUT_ROOT"
echo "Seeds: $SEEDS"
echo "Datasets: $DATASETS"
echo "Strategies: $STRATEGIES"
echo "Modes: $MODES"
echo "Epochs: $EPOCHS"
echo "Linear batch size: $LINEAR_BATCH_SIZE"
echo "LoRA batch size: $LORA_BATCH_SIZE"
echo "Num workers: $NUM_WORKERS"
echo "Retry num workers: $RETRY_NUM_WORKERS"
echo "Linear retry batch size: $LINEAR_RETRY_BATCH_SIZE"
echo "LoRA retry batch size: $LORA_RETRY_BATCH_SIZE"
echo "Log file: $LOG_FILE"
echo "========================================="

run_step3_job() {
  local dataset="$1"
  local strategy="$2"
  local mode="$3"
  local seed="$4"
  local visible_devices="$5"
  local batch_size="$6"
  local retry_batch_size="$7"

  echo ">>> seed=$seed | dataset=$dataset | pseudo=$strategy | mode=$mode | attempt=1 | batch_size=$batch_size | num_workers=$NUM_WORKERS"
  if ! CUDA_VISIBLE_DEVICES="$visible_devices" "$PYTHON_BIN" step3_clip_finetune.py \
      --dataset "$dataset" \
      --pseudo_strategy "$strategy" \
      --mode "$mode" \
      --epochs "$EPOCHS" \
      --batch_size "$batch_size" \
      --seed "$seed" \
      --num_workers "$NUM_WORKERS" \
      --analysis_dir "$ANALYSIS_DIR" \
      --out_dir "$seed_out_dir"; then
    echo "[WARN] step3 failed for seed=$seed dataset=$dataset pseudo=$strategy mode=$mode. Retrying with safer settings..."
    echo ">>> seed=$seed | dataset=$dataset | pseudo=$strategy | mode=$mode | attempt=2 | batch_size=$retry_batch_size | num_workers=$RETRY_NUM_WORKERS"
    CUDA_VISIBLE_DEVICES="$visible_devices" "$PYTHON_BIN" step3_clip_finetune.py \
      --dataset "$dataset" \
      --pseudo_strategy "$strategy" \
      --mode "$mode" \
      --epochs "$EPOCHS" \
      --batch_size "$retry_batch_size" \
      --seed "$seed" \
      --num_workers "$RETRY_NUM_WORKERS" \
      --analysis_dir "$ANALYSIS_DIR" \
      --out_dir "$seed_out_dir"
  fi
}

read -r -a seed_array <<< "$SEEDS"
read -r -a dataset_array <<< "$DATASETS"
read -r -a strategy_array <<< "$STRATEGIES"
read -r -a mode_array <<< "$MODES"

for seed in "${seed_array[@]}"; do
  seed_out_dir="$OUT_ROOT/seed_${seed}"
  mkdir -p "$seed_out_dir"

  for dataset in "${dataset_array[@]}"; do
    for strategy in "${strategy_array[@]}"; do
      for mode in "${mode_array[@]}"; do
        batch_size="$LINEAR_BATCH_SIZE"
        visible_devices="$CUDA_VISIBLE_DEVICES_LINEAR"
        retry_batch_size="$LINEAR_RETRY_BATCH_SIZE"
        if [[ "$mode" == "lora" ]]; then
          batch_size="$LORA_BATCH_SIZE"
          visible_devices="$CUDA_VISIBLE_DEVICES_LORA"
          retry_batch_size="$LORA_RETRY_BATCH_SIZE"
        fi

        result_json="$seed_out_dir/${dataset}_${mode}_${strategy}_result.json"
        if [[ -f "$result_json" ]]; then
          echo ">>> Skip existing result: $result_json"
          continue
        fi

        run_step3_job "$dataset" "$strategy" "$mode" "$seed" "$visible_devices" "$batch_size" "$retry_batch_size"
      done
    done
  done
done

summary_csv="$OUT_ROOT/phase3_repeated_seed_summary.csv"
echo ">>> Aggregating repeated-seed summary -> $summary_csv"
"$PYTHON_BIN" summarize_repeated_seeds.py \
  --results_root "$OUT_ROOT" \
  --out_csv "$summary_csv"

echo "========================================="
echo "Repeated-seed experiments completed"
echo "Summary: $summary_csv"
echo "========================================="