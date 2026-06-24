#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "$0")"
cd "$SCRIPT_DIR"
mkdir -p logs/cross_model_validation

resolve_python_bin() {
  local preferred="${QWEN36_PYTHON_BIN:-/root/anaconda3/envs/higentec_llava/bin/python}"
  if [[ -x "$preferred" ]]; then
    echo "$preferred"
    return
  fi
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

MODEL_KEY=${MODEL_KEY:-qwen36_35b_fp8}
HF_ENDPOINT_VALUE=${HF_ENDPOINT_VALUE:-https://hf-mirror.com}
CUDA_VISIBLE_DEVICES_SET=${CUDA_VISIBLE_DEVICES_SET:-0,1}
OUT_DIR=${OUT_DIR:-./results/cross_model_validation/${MODEL_KEY}_v2}
HF_MAX_MEMORY=${HF_MAX_MEMORY:-0:34GiB,1:34GiB,cpu:512GiB}
HF_OFFLOAD_FOLDER=${HF_OFFLOAD_FOLDER:-${OUT_DIR}/hf_offload}
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
LOG=${LOG:-logs/cross_model_validation/${MODEL_KEY}_${STAMP}.log}
PIDFILE=${PIDFILE:-logs/cross_model_validation/${MODEL_KEY}_${STAMP}.pid}

DATASETS=${DATASETS:-"BowTurnHead HandriseReadWrite TeacherBehavior"}
SPLITS=${SPLITS:-"val train"}
MAX_IMAGES=${MAX_IMAGES:-}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32}
NO_ANALYZE=${NO_ANALYZE:-0}

run_worker() {
  echo "[$(date '+%F %T')] qwen3.6:35b cross-model run start"
  echo "python_bin=$PYTHON_BIN"
  echo "model_key=$MODEL_KEY"
  echo "hf_endpoint=$HF_ENDPOINT"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "out_dir=$OUT_DIR"
  echo "hf_max_memory=$HF_MAX_MEMORY"
  echo "hf_offload_folder=$HF_OFFLOAD_FOLDER"
  echo "pytorch_cuda_alloc_conf=$PYTORCH_CUDA_ALLOC_CONF"
  echo "datasets=$DATASETS"
  echo "splits=$SPLITS"
  echo "max_images=${MAX_IMAGES:-<all>}"

  for dataset in $DATASETS; do
    for split in $SPLITS; do
      echo "[$(date '+%F %T')] START dataset=${dataset} split=${split} model=${MODEL_KEY}"
      args=(
        --dataset "$dataset"
        --split "$split"
        --models "$MODEL_KEY"
        --out_dir "$OUT_DIR"
        --max_new_tokens "$MAX_NEW_TOKENS"
      )
      if [[ -n "${MAX_IMAGES:-}" ]]; then
        args+=(--max_images "$MAX_IMAGES")
      fi
      if [[ "$NO_ANALYZE" == "1" ]]; then
        args+=(--no_analyze)
      fi
      "$PYTHON_BIN" cross_model_annotate.py "${args[@]}"
      echo "[$(date '+%F %T')] DONE  dataset=${dataset} split=${split} model=${MODEL_KEY}"
    done
  done
  echo "[$(date '+%F %T')] qwen3.6:35b cross-model run finished"
}

if [[ "${1:-}" == "--worker" ]]; then
  run_worker
  exit 0
fi

mkdir -p "$OUT_DIR"

nohup env \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_SET" \
  HF_ENDPOINT="$HF_ENDPOINT_VALUE" \
  HF_HUB_URL="$HF_ENDPOINT_VALUE" \
  HUGGINGFACE_HUB_URL="$HF_ENDPOINT_VALUE" \
  HUGGINGFACE_CO_RESOLVE_ENDPOINT="$HF_ENDPOINT_VALUE" \
  HF_MAX_MEMORY="$HF_MAX_MEMORY" \
  HF_OFFLOAD_FOLDER="$HF_OFFLOAD_FOLDER" \
  PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
  PYTHONUNBUFFERED=1 \
  PYTHON_BIN="$PYTHON_BIN" \
  MODEL_KEY="$MODEL_KEY" \
  OUT_DIR="$OUT_DIR" \
  DATASETS="$DATASETS" \
  SPLITS="$SPLITS" \
  MAX_IMAGES="$MAX_IMAGES" \
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  NO_ANALYZE="$NO_ANALYZE" \
  bash "$SCRIPT_PATH" --worker > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PIDFILE"

echo "started Qwen3.6-35B HF cross-model background run"
echo "pid: $PID"
echo "pidfile: $PIDFILE"
echo "log: $LOG"
echo "out_dir: $OUT_DIR"
echo "monitor: tail -f $LOG"
