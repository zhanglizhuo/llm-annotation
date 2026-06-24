#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "$0")"
cd "$SCRIPT_DIR"
mkdir -p logs/cross_model_validation

resolve_python_bin() {
  local preferred="${GEMMA4_PYTHON_BIN:-/root/anaconda3/envs/higentec_llava/bin/python}"
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

HF_ENDPOINT_VALUE=${HF_ENDPOINT_VALUE:-https://hf-mirror.com}
CUDA_VISIBLE_DEVICES_SET=${CUDA_VISIBLE_DEVICES_SET:-0,1}
HF_MAX_MEMORY=${HF_MAX_MEMORY:-0:34GiB,1:34GiB,cpu:512GiB}
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
LOG=${LOG:-logs/cross_model_validation/gemma4_dual_${STAMP}.log}
PIDFILE=${PIDFILE:-logs/cross_model_validation/gemma4_dual_${STAMP}.pid}

MODEL_SEQUENCE=${MODEL_SEQUENCE:-"gemma4_26b_a4b gemma4_31b"}
DATASETS=${DATASETS:-"BowTurnHead HandriseReadWrite TeacherBehavior"}
SPLITS=${SPLITS:-"val train"}
MAX_IMAGES=${MAX_IMAGES:-}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32}
NO_ANALYZE=${NO_ANALYZE:-0}
OUT_BASE_DIR=${OUT_BASE_DIR:-./results/cross_model_validation}

run_worker() {
  echo "[$(date '+%F %T')] Gemma-4 dual run start"
  echo "python_bin=$PYTHON_BIN"
  echo "hf_endpoint=$HF_ENDPOINT"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "hf_max_memory=$HF_MAX_MEMORY"
  echo "pytorch_cuda_alloc_conf=$PYTORCH_CUDA_ALLOC_CONF"
  echo "model_sequence=$MODEL_SEQUENCE"
  echo "datasets=$DATASETS"
  echo "splits=$SPLITS"
  echo "max_images=${MAX_IMAGES:-<all>}"
  echo "max_new_tokens=$MAX_NEW_TOKENS"
  echo "out_base_dir=$OUT_BASE_DIR"

  for model_key in $MODEL_SEQUENCE; do
    local_out_dir="${OUT_BASE_DIR}/${model_key}_v1"
    local_offload_dir="${local_out_dir}/hf_offload"
    mkdir -p "$local_out_dir"

    echo "[$(date '+%F %T')] MODEL_START model=${model_key} out_dir=${local_out_dir}"

    for dataset in $DATASETS; do
      for split in $SPLITS; do
        echo "[$(date '+%F %T')] START model=${model_key} dataset=${dataset} split=${split}"
        args=(
          --dataset "$dataset"
          --split "$split"
          --models "$model_key"
          --out_dir "$local_out_dir"
          --max_new_tokens "$MAX_NEW_TOKENS"
        )
        if [[ -n "${MAX_IMAGES:-}" ]]; then
          args+=(--max_images "$MAX_IMAGES")
        fi
        if [[ "$NO_ANALYZE" == "1" ]]; then
          args+=(--no_analyze)
        fi

        HF_MAX_MEMORY="$HF_MAX_MEMORY" \
        HF_OFFLOAD_FOLDER="$local_offload_dir" \
        "$PYTHON_BIN" cross_model_annotate.py "${args[@]}"

        echo "[$(date '+%F %T')] DONE  model=${model_key} dataset=${dataset} split=${split}"
      done
    done

    echo "[$(date '+%F %T')] MODEL_DONE  model=${model_key} out_dir=${local_out_dir}"
  done

  echo "[$(date '+%F %T')] Gemma-4 dual run finished"
}

if [[ "${1:-}" == "--worker" ]]; then
  run_worker
  exit 0
fi

nohup env \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_SET" \
  HF_ENDPOINT="$HF_ENDPOINT_VALUE" \
  HF_HUB_URL="$HF_ENDPOINT_VALUE" \
  HUGGINGFACE_HUB_URL="$HF_ENDPOINT_VALUE" \
  HUGGINGFACE_CO_RESOLVE_ENDPOINT="$HF_ENDPOINT_VALUE" \
  PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
  PYTHONUNBUFFERED=1 \
  PYTHON_BIN="$PYTHON_BIN" \
  HF_MAX_MEMORY="$HF_MAX_MEMORY" \
  MODEL_SEQUENCE="$MODEL_SEQUENCE" \
  DATASETS="$DATASETS" \
  SPLITS="$SPLITS" \
  MAX_IMAGES="$MAX_IMAGES" \
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  NO_ANALYZE="$NO_ANALYZE" \
  OUT_BASE_DIR="$OUT_BASE_DIR" \
  bash "$SCRIPT_PATH" --worker > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PIDFILE"

echo "started Gemma-4 dual-model HF background run"
echo "pid: $PID"
echo "pidfile: $PIDFILE"
echo "log: $LOG"
echo "monitor: tail -f $LOG"