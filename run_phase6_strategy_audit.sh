#!/usr/bin/env bash
# Run phase-6 strategy-audit analyses for auxiliary baselines and consistency checks.

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

resolve_latest_analysis_dir() {
  local base="./results/phase2_filtering"
  if [[ -d "$base" ]]; then
    local latest
    latest=$(find "$base" -maxdepth 1 -mindepth 1 -type d -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr | head -n1 | cut -d' ' -f2-)
    if [[ -n "$latest" ]]; then
      echo "$latest"
      return
    fi
  fi
  echo "./results/phase2_filtering/default"
}

export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_URL=${HF_HUB_URL:-$HF_ENDPOINT}
export HUGGINGFACE_HUB_URL=${HUGGINGFACE_HUB_URL:-$HF_ENDPOINT}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

PYTHON_BIN=${PYTHON_BIN:-$(resolve_python_bin)}
if [[ -z "$PYTHON_BIN" ]]; then
  echo "[ERROR] Could not find python3 or python in PATH. Set PYTHON_BIN explicitly."
  exit 1
fi

RUN_TAG=${RUN_TAG:-strategy_audit_$(date +%Y%m%d_%H%M%S)}
OUT_ROOT=${OUT_ROOT:-./results/phase6_strategy_audit/${RUN_TAG}}
LOG_DIR=${LOG_DIR:-./logs/phase6}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/phase6_${RUN_TAG}.log}
mkdir -p "$LOG_DIR" "$OUT_ROOT"
exec > >(tee -a "$LOG_FILE") 2>&1

DATASETS=(BowTurnHead HandriseReadWrite TeacherBehavior)
ANALYSIS_DIR=${ANALYSIS_DIR:-$(resolve_latest_analysis_dir)}
MAIN_ANN_DIR=${MAIN_ANN_DIR:-./results/phase1_annotations/full_20260418_0001}
CROSSMODEL_DIR=${CROSSMODEL_DIR:-./results/cross_model_validation/default}
CLIP_ZS_JSON=${CLIP_ZS_JSON:-./results/phase0_zero_shot/canonical_20260425_231347/phase0_zero_shot_results.json}

CUDA_VISIBLE_DEVICES_SET=${CUDA_VISIBLE_DEVICES_SET:-0,1}
SEEDS=${SEEDS:-"43 44 45 46 47"}
THRESHOLDS=${THRESHOLDS:-"0.3 0.5 0.6 0.7 0.8 0.9"}
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-128}
HEAD_BATCH_SIZE=${HEAD_BATCH_SIZE:-128}
NUM_WORKERS=${NUM_WORKERS:-0}
TEACHER_LABEL_FRACTION=${TEACHER_LABEL_FRACTION:-0.1}
MIN_LABELED_PER_CLASS=${MIN_LABELED_PER_CLASS:-5}
TEACHER_CONF_THRESHOLD=${TEACHER_CONF_THRESHOLD:-0.0}

RUN_CROSS_MODEL_CONSISTENCY=${RUN_CROSS_MODEL_CONSISTENCY:-1}
RUN_CONFIDENCE_FILTERING=${RUN_CONFIDENCE_FILTERING:-1}
RUN_TEACHER_STUDENT=${RUN_TEACHER_STUDENT:-1}

echo "========================================="
echo "Phase 6 strategy audit"
echo "run_tag: $RUN_TAG"
echo "python: $PYTHON_BIN"
echo "out_root: $OUT_ROOT"
echo "log_file: $LOG_FILE"
echo "analysis_dir: $ANALYSIS_DIR"
echo "main_ann_dir: $MAIN_ANN_DIR"
echo "crossmodel_dir: $CROSSMODEL_DIR"
echo "clip_zs_json: $CLIP_ZS_JSON"
echo "cuda_visible_devices: $CUDA_VISIBLE_DEVICES_SET"
echo "hf_hub_offline: $HF_HUB_OFFLINE"
echo "seeds: $SEEDS"
echo "thresholds: $THRESHOLDS"
echo "teacher_label_fraction: $TEACHER_LABEL_FRACTION"
echo "========================================="

if [[ "$RUN_CROSS_MODEL_CONSISTENCY" == "1" ]]; then
  echo ">>> [Cross-model consistency] Independent anchoring check"
  "$PYTHON_BIN" cross_model_consistency.py \
    --dataset "${DATASETS[@]}" \
    --main_ann_dir "$MAIN_ANN_DIR" \
    --crossmodel_dir "$CROSSMODEL_DIR" \
    --clip_zs_json "$CLIP_ZS_JSON" \
    --output_dir "$OUT_ROOT/cross_model_consistency"
fi

if [[ "$RUN_CONFIDENCE_FILTERING" == "1" ]]; then
  echo ">>> [Confidence filtering] CLIP-assisted pseudo-label filtering"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_SET" "$PYTHON_BIN" confidence_filtering.py \
    --dataset "${DATASETS[@]}" \
    --analysis_dir "$ANALYSIS_DIR" \
    --output_dir "$OUT_ROOT/confidence_filtering" \
    --thresholds $THRESHOLDS \
    --seeds $SEEDS \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --head_batch_size "$HEAD_BATCH_SIZE" \
    --num_workers "$NUM_WORKERS"
fi

if [[ "$RUN_TEACHER_STUDENT" == "1" ]]; then
  echo ">>> [Teacher-student] Train-split self-training"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_SET" "$PYTHON_BIN" teacher_student_self_training.py \
    --dataset "${DATASETS[@]}" \
    --output_dir "$OUT_ROOT/teacher_student" \
    --seeds $SEEDS \
    --teacher_label_fraction "$TEACHER_LABEL_FRACTION" \
    --min_labeled_per_class "$MIN_LABELED_PER_CLASS" \
    --teacher_conf_threshold "$TEACHER_CONF_THRESHOLD" \
    --teacher_epochs "$EPOCHS" \
    --student_epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --head_batch_size "$HEAD_BATCH_SIZE" \
    --num_workers "$NUM_WORKERS"
fi

echo "========================================="
  echo "Phase 6 strategy audit completed"
echo "Results: $OUT_ROOT"
echo "Log: $LOG_FILE"
echo "========================================="