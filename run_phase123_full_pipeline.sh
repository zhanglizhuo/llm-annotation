#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# 快速启动脚本 — SCB LLM自动标注实验，请后台运行！！！
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
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

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HUGGINGFACE_HUB_URL=${HUGGINGFACE_HUB_URL:-$HF_ENDPOINT}
export HF_HUB_URL=${HF_HUB_URL:-$HF_ENDPOINT}
PYTHON_BIN=${PYTHON_BIN:-$(resolve_python_bin)}
if [[ -z "$PYTHON_BIN" ]]; then
    echo "[ERROR] Could not find python3 or python in PATH. Set PYTHON_BIN explicitly."
    exit 1
fi
INSTALL_DEPS=${INSTALL_DEPS:-1}
RUN_TAG=${RUN_TAG:-full_$(date +%Y%m%d_%H%M%S)}
STEP3_NUM_WORKERS=${STEP3_NUM_WORKERS:-0}
SKIP_PHASE12=${SKIP_PHASE12:-0}
STEP3_LINEAR_BATCH_SIZE=${STEP3_LINEAR_BATCH_SIZE:-64}
STEP3_LORA_BATCH_SIZE=${STEP3_LORA_BATCH_SIZE:-16}
STEP3_LINEAR_CUDA_VISIBLE_DEVICES=${STEP3_LINEAR_CUDA_VISIBLE_DEVICES:-0,1}
STEP3_LORA_CUDA_VISIBLE_DEVICES=${STEP3_LORA_CUDA_VISIBLE_DEVICES:-0,1}
RESULTS_ROOT=${RESULTS_ROOT:-./results}

DATASETS=("BowTurnHead" "HandriseReadWrite" "TeacherBehavior")
ANN_DIR=${ANN_DIR:-${RESULTS_ROOT}/phase1_annotations/${RUN_TAG}}
ANALYSIS_DIR=${ANALYSIS_DIR:-${RESULTS_ROOT}/phase2_filtering/${RUN_TAG}}
FINETUNE_DIR=${FINETUNE_DIR:-${RESULTS_ROOT}/phase3_finetune/full_pipeline/${RUN_TAG}}
LOG_DIR=${LOG_DIR:-./logs/pipeline}
LOG_FILE="${LOG_DIR}/phase123_${RUN_TAG}.log"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================="
echo "Run tag: $RUN_TAG"
echo "HF mirror: $HF_ENDPOINT"
echo "Python: $PYTHON_BIN"
echo "Install deps: $INSTALL_DEPS"
echo "Step3 num_workers: $STEP3_NUM_WORKERS"
echo "Step3 linear batch_size: $STEP3_LINEAR_BATCH_SIZE"
echo "Step3 lora batch_size: $STEP3_LORA_BATCH_SIZE"
echo "Step3 linear CUDA_VISIBLE_DEVICES: $STEP3_LINEAR_CUDA_VISIBLE_DEVICES"
echo "Step3 lora CUDA_VISIBLE_DEVICES: $STEP3_LORA_CUDA_VISIBLE_DEVICES"
echo "Skip Phase1/2: $SKIP_PHASE12"
echo "Annotations: $ANN_DIR"
echo "Analysis: $ANALYSIS_DIR"
echo "Finetune: $FINETUNE_DIR"
echo "Log: $LOG_FILE"
echo "========================================="

if [[ "$INSTALL_DEPS" == "1" ]]; then
    "$PYTHON_BIN" -m pip install -q -r "$REPO_ROOT/requirements.txt"
fi

if [[ "$SKIP_PHASE12" != "1" ]]; then
    for ds in "${DATASETS[@]}"; do
        echo ">>> [Phase 1] Annotate $ds val"
        CUDA_VISIBLE_DEVICES=0,1 "$PYTHON_BIN" step1_llm_annotate.py \
            --dataset "$ds" \
            --split val \
            --out_dir "$ANN_DIR" \
            --overwrite

        echo ">>> [Phase 1] Annotate $ds train"
        CUDA_VISIBLE_DEVICES=0,1 "$PYTHON_BIN" step1_llm_annotate.py \
            --dataset "$ds" \
            --split train \
            --out_dir "$ANN_DIR" \
            --overwrite
    done

    for ds in "${DATASETS[@]}"; do
        echo ">>> [Phase 2] Analyze $ds val"
        "$PYTHON_BIN" step2_filter_analysis.py \
            --dataset "$ds" \
            --split val \
            --ann_dir "$ANN_DIR" \
            --out_dir "$ANALYSIS_DIR"

        echo ">>> [Phase 2] Analyze $ds train"
        "$PYTHON_BIN" step2_filter_analysis.py \
            --dataset "$ds" \
            --split train \
            --ann_dir "$ANN_DIR" \
            --out_dir "$ANALYSIS_DIR"
    done
fi

for ds in "${DATASETS[@]}"; do
    for strategy in "none" "agreement" "gt"; do
        for mode in "linear" "lora"; do
            result_json="${FINETUNE_DIR}/${ds}_${mode}_${strategy}_result.json"
            if [[ -f "$result_json" ]]; then
                echo ">>> [Phase 3] Skip existing result: $result_json"
                continue
            fi
            batch_size="$STEP3_LINEAR_BATCH_SIZE"
            cuda_visible_devices="$STEP3_LINEAR_CUDA_VISIBLE_DEVICES"
            if [[ "$mode" == "lora" ]]; then
                batch_size="$STEP3_LORA_BATCH_SIZE"
                cuda_visible_devices="$STEP3_LORA_CUDA_VISIBLE_DEVICES"
            fi
            echo ">>> [Phase 3] $ds | pseudo=$strategy | mode=$mode"
            CUDA_VISIBLE_DEVICES="$cuda_visible_devices" "$PYTHON_BIN" step3_clip_finetune.py \
                --dataset "$ds" \
                --pseudo_strategy "$strategy" \
                --mode "$mode" \
                --epochs 20 \
                --batch_size "$batch_size" \
                --num_workers "$STEP3_NUM_WORKERS" \
                --analysis_dir "$ANALYSIS_DIR" \
                --out_dir "$FINETUNE_DIR"
        done
    done
done

echo "========================================="
echo "Full experiment completed successfully"
echo "Results: $FINETUNE_DIR"
echo "========================================="