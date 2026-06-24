#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# 早期全量实验脚本（保留作参考；推荐使用 run_phase123_full_pipeline.sh）
# 当前环境：2x A100 40G
# 建议分阶段运行：先完成LLM标注，再释放GPU跑CLIP微调
# ══════════════════════════════════════════════════════════════════════════════

set -e  # 遇到错误即停止

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
PYTHON_BIN=${PYTHON_BIN:-$(resolve_python_bin)}
if [[ -z "$PYTHON_BIN" ]]; then
    echo "[ERROR] Could not find python3 or python in PATH. Set PYTHON_BIN explicitly."
    exit 1
fi

RUN_TAG=${RUN_TAG:-legacy_full_$(date +%Y%m%d_%H%M%S)}
RESULTS_ROOT=${RESULTS_ROOT:-./results}
ANN_DIR=${ANN_DIR:-${RESULTS_ROOT}/phase1_annotations/${RUN_TAG}}
ANALYSIS_DIR=${ANALYSIS_DIR:-${RESULTS_ROOT}/phase2_filtering/${RUN_TAG}}
FINETUNE_DIR=${FINETUNE_DIR:-${RESULTS_ROOT}/phase3_finetune/full_pipeline/${RUN_TAG}}
LOG_DIR=${LOG_DIR:-./logs/pipeline}
LOG_FILE="${LOG_DIR}/phase123_${RUN_TAG}.log"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

DATASETS=("BowTurnHead" "HandriseReadWrite" "TeacherBehavior")

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: LLM标注（val集用于质量评估，train集用于微调）
# ══════════════════════════════════════════════════════════════════════════════
echo "========================================="
echo "Phase 1: LLM自动标注"
echo "========================================="
echo "Using HF mirror: $HF_ENDPOINT"

for ds in "${DATASETS[@]}"; do
    echo ">>> 标注 $ds val..."
    CUDA_VISIBLE_DEVICES=0,1 "$PYTHON_BIN" step1_llm_annotate.py --dataset "$ds" --split val --out_dir "$ANN_DIR"

    echo ">>> 标注 $ds train..."
    CUDA_VISIBLE_DEVICES=0,1 "$PYTHON_BIN" step1_llm_annotate.py --dataset "$ds" --split train --out_dir "$ANN_DIR"
done

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: 质量分析 + 生成伪标签文件
# ══════════════════════════════════════════════════════════════════════════════
echo "========================================="
echo "Phase 2: 标注质量分析"
echo "========================================="

for ds in "${DATASETS[@]}"; do
    echo ">>> 分析 $ds val（用于计算LLM标注准确率）..."
    "$PYTHON_BIN" step2_filter_analysis.py --dataset "$ds" --split val --ann_dir "$ANN_DIR" --out_dir "$ANALYSIS_DIR"

    echo ">>> 分析 $ds train（生成微调用伪标签）..."
    "$PYTHON_BIN" step2_filter_analysis.py --dataset "$ds" --split train --ann_dir "$ANN_DIR" --out_dir "$ANALYSIS_DIR"
done

# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: CLIP微调对比实验（GPU0+GPU1）
# ══════════════════════════════════════════════════════════════════════════════
echo "========================================="
echo "Phase 3: CLIP微调实验"
echo "========================================="

for ds in "${DATASETS[@]}"; do
    for strategy in "none" "agreement" "gt"; do
        for mode in "linear" "lora"; do
            echo ">>> $ds | pseudo=$strategy | mode=$mode"
            CUDA_VISIBLE_DEVICES=0,1 "$PYTHON_BIN" step3_clip_finetune.py \
                --dataset "$ds" \
                --pseudo_strategy "$strategy" \
                --mode "$mode" \
                --epochs 20 \
                --batch_size 64 \
                --analysis_dir "$ANALYSIS_DIR" \
                --out_dir "$FINETUNE_DIR"
        done
    done
done

echo "========================================="
echo "全部实验完成！结果已写入 ${RESULTS_ROOT}"
echo "========================================="

# ══════════════════════════════════════════════════════════════════════════════
# 实验矩阵（论文表格结构）：
#
# 数据集        | 标签来源     | 微调方式      | Val Acc
# --------------|-------------|--------------|--------
# TeacherBehavior | 零样本CLIP  | 无           | 见 finetune_summary.csv
# TeacherBehavior | 全部伪标签  | Linear Probe | ?
# TeacherBehavior | 全部伪标签  | LoRA         | ?
# TeacherBehavior | 一致性伪标签| Linear Probe | ?
# TeacherBehavior | 一致性伪标签| LoRA         | ?
# TeacherBehavior | 真实标签    | LoRA (上界)  | ?
# (同上，HandriseReadWrite 和 BowTurnHead)
#
# 核心研究问题：
# 1. LLM伪标签准确率 vs 任务难度（三个数据集对比）
# 2. 两模型一致性过滤能提升多少准确率？（代价：减少多少数据量）
# 3. 伪标签微调后能超过零样本CLIP多少？
# 4. Qwen vs Llava：哪个对中国课堂场景更友好？
# ══════════════════════════════════════════════════════════════════════════════
