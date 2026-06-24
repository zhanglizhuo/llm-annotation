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
OUT_DIR=${OUT_DIR:-./results/cross_model_validation/default}
SMOKE_BASE_DIR=${SMOKE_BASE_DIR:-./results/cross_model_validation/smoke}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}

RUNS=(
  "qwen2532 BowTurnHead val"
  "qwen2532 HandriseReadWrite val"
  "qwen2532 TeacherBehavior val"
  "qwen2532 TeacherBehavior train"
  "gemma4 BowTurnHead val"
  "gemma4 HandriseReadWrite val"
  "gemma4 TeacherBehavior val"
  "gemma4 TeacherBehavior train"
)

run_step() {
  local model=$1
  local dataset=$2
  local split=$3

  echo "[$(date '+%F %T')] START model=${model} dataset=${dataset} split=${split}"
  HF_ENDPOINT="$HF_ENDPOINT_VALUE" \
  HF_HUB_URL="$HF_ENDPOINT_VALUE" \
  HUGGINGFACE_HUB_URL="$HF_ENDPOINT_VALUE" \
  HUGGINGFACE_CO_RESOLVE_ENDPOINT="$HF_ENDPOINT_VALUE" \
  CUDA_VISIBLE_DEVICES=0,1 \
  PYTHONUNBUFFERED=1 \
  "$PYTHON_BIN" cross_model_annotate.py \
    --dataset "$dataset" \
    --split "$split" \
    --models "$model" \
    --out_dir "$OUT_DIR"
  echo "[$(date '+%F %T')] DONE  model=${model} dataset=${dataset} split=${split}"
}

run_gemma4_smoke() {
  local smoke_dir="$SMOKE_BASE_DIR/gemma4_${STAMP}"
  mkdir -p "$smoke_dir"
  echo "[$(date '+%F %T')] START gemma4 smoke out_dir=${smoke_dir}"
  HF_ENDPOINT="$HF_ENDPOINT_VALUE" \
  HF_HUB_URL="$HF_ENDPOINT_VALUE" \
  HUGGINGFACE_HUB_URL="$HF_ENDPOINT_VALUE" \
  HUGGINGFACE_CO_RESOLVE_ENDPOINT="$HF_ENDPOINT_VALUE" \
  CUDA_VISIBLE_DEVICES=0,1 \
  PYTHONUNBUFFERED=1 \
  "$PYTHON_BIN" cross_model_annotate.py \
    --dataset BowTurnHead \
    --split val \
    --models gemma4 \
    --max_images 1 \
    --out_dir "$smoke_dir"
  echo "[$(date '+%F %T')] DONE  gemma4 smoke out_dir=${smoke_dir}"
}

echo "[$(date '+%F %T')] Pipeline start"
echo "python_bin=$PYTHON_BIN"
echo "out_dir=$OUT_DIR"
echo "smoke_base_dir=$SMOKE_BASE_DIR"
echo "hf_endpoint=$HF_ENDPOINT_VALUE"

for spec in "${RUNS[@]}"; do
  read -r model dataset split <<<"$spec"
  if [[ "$model" == "gemma4" && "$dataset" == "BowTurnHead" && "$split" == "val" ]]; then
    run_gemma4_smoke
  fi
  run_step "$model" "$dataset" "$split"
done

echo "[$(date '+%F %T')] Pipeline finished"