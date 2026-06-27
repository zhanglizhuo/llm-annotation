"""
Step 2: 标注质量分析 + 四种过滤策略对比
输入：step1输出的 .jsonl 文件
输出：各策略的标注统计表 + 过滤后的伪标签文件（供step3使用）
用法：
  python step2_filter_analysis.py --dataset TeacherBehavior
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

FILTER_STRATEGIES = ["none", "confidence_qwen", "confidence_llava", "agreement"]
# none             ：全部使用（不过滤）
# confidence_qwen  ：只用Qwen有输出的（非None）
# confidence_llava ：只用Llava有输出的（非None）
# agreement        ：只用两模型预测一致的


def load_jsonl(path: Path) -> List[Dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


def apply_filter(records: List[Dict], strategy: str) -> List[Dict]:
    if strategy == "none":
        # 没有输出的用Qwen结果兜底，仍为None则丢弃
        return [r for r in records if r.get("pred_qwen") is not None]
    elif strategy == "confidence_qwen":
        return [r for r in records if r.get("pred_qwen") is not None]
    elif strategy == "confidence_llava":
        return [r for r in records if r.get("pred_llava") is not None]
    elif strategy == "agreement":
        return [r for r in records if r.get("agreement") is True]
    return records


def evaluate(records: List[Dict], model_key: str) -> Dict[str, object]:
    """计算某模型在过滤后记录上的准确率"""
    gts, preds = [], []
    for r in records:
        p = r.get(f"pred_{model_key}")
        if p is not None:
            gts.append(r["gt"])
            preds.append(p)
    if not gts:
        return {"acc": 0.0, "n": 0}
    return {
        "acc": round(accuracy_score(gts, preds) * 100, 2),
        "n": len(gts),
    }


def analyze(dataset_name: str, ann_dir: Path, split: str, out_dir: Path):
    ann_file = ann_dir / f"{dataset_name}_{split}_annotations.jsonl"
    if not ann_file.exists():
        print(f"[ERROR] 找不到标注文件: {ann_file}")
        return

    records = load_jsonl(ann_file)
    total = len(records)
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} | Split: {split} | Total bboxes: {total}")
    print(f"{'='*60}")

    # ── 基础统计 ──────────────────────────────────────────────────────────────
    qwen_valid = sum(1 for r in records if r.get("pred_qwen") is not None)
    llava_valid = sum(1 for r in records if r.get("pred_llava") is not None)
    agree = sum(1 for r in records if r.get("agreement") is True)

    print(f"\n[基础统计]")
    print(f"  Qwen有效输出: {qwen_valid}/{total} ({100*qwen_valid/total:.1f}%)")
    print(f"  Llava有效输出: {llava_valid}/{total} ({100*llava_valid/total:.1f}%)")
    print(f"  两模型一致: {agree}/{total} ({100*agree/total:.1f}%)")

    avg_qwen_time = np.mean([r["time_qwen"] for r in records if "time_qwen" in r])
    avg_llava_time = np.mean([r["time_llava"] for r in records if "time_llava" in r])
    print(f"  Qwen平均耗时: {avg_qwen_time:.2f}s/bbox")
    print(f"  Llava平均耗时: {avg_llava_time:.2f}s/bbox")

    # ── 四种过滤策略对比 ────────────────────────────────────────────────────
    print(f"\n[过滤策略对比]")
    rows = []
    for strategy in FILTER_STRATEGIES:
        filtered = apply_filter(records, strategy)
        n = len(filtered)
        retention = 100 * n / total if total > 0 else 0

        qwen_eval = evaluate(filtered, "qwen")
        llava_eval = evaluate(filtered, "llava")

        # 集成预测：两个都有则取一致，否则用有输出的那个
        ensemble_correct = 0
        ensemble_total = 0
        for r in filtered:
            qp = r.get("pred_qwen")
            lp = r.get("pred_llava")
            if qp is None and lp is None:
                continue
            pred = qp if lp is None else (lp if qp is None else (qp if qp == lp else qp))
            ensemble_correct += int(pred == r["gt"])
            ensemble_total += 1
        ens_acc = round(100 * ensemble_correct / ensemble_total, 2) if ensemble_total else 0

        row = {
            "策略": strategy,
            "保留bbox数": n,
            "保留率(%)": round(retention, 1),
            "Qwen准确率(%)": qwen_eval["acc"],
            "Llava准确率(%)": llava_eval["acc"],
            "集成准确率(%)": ens_acc,
        }
        rows.append(row)
        print(f"  {strategy:20s} | 保留{n:5d}({retention:5.1f}%) | "
              f"Qwen={qwen_eval['acc']:5.1f}% Llava={llava_eval['acc']:5.1f}% "
              f"Ensemble={ens_acc:5.1f}%")

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{dataset_name}_{split}_filter_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  [保存] {csv_path}")

    # ── 逐类别准确率（agreement策略，用于误差分析） ────────────────────────
    print(f"\n[逐类别准确率 — agreement策略]")
    agreed = apply_filter(records, "agreement")
    class_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in agreed:
        gt_name = r["gt_name"]
        pred = r.get("pred_qwen")  # agreement时两个相同，取qwen
        class_stats[gt_name]["total"] += 1
        if pred == r["gt"]:
            class_stats[gt_name]["correct"] += 1

    for cls, stat in sorted(class_stats.items()):
        acc = 100 * stat["correct"] / stat["total"] if stat["total"] else 0
        print(f"  {cls:25s}: {acc:5.1f}%  (n={stat['total']})")

    # ── 输出最佳过滤策略的伪标签文件（供step3使用） ─────────────────────────
    # 默认输出agreement策略的伪标签（最干净），也输出none策略（最大量）
    for strategy in ["none", "agreement"]:
        filtered = apply_filter(records, strategy)
        pseudo_path = out_dir / f"{dataset_name}_{split}_pseudo_{strategy}.jsonl"
        with open(pseudo_path, "w") as f:
            for r in filtered:
                # 选择伪标签：agreement用qwen，none也用qwen
                pseudo_label = r.get("pred_qwen")
                if pseudo_label is None:
                    continue
                out_rec = {
                    "image": r["image"],
                    "bbox_idx": r["bbox_idx"],
                    "gt": r["gt"],
                    "pseudo_label": pseudo_label,
                    "pseudo_label_name": r.get("pred_qwen_name"),
                    "cx": r["cx"], "cy": r["cy"],
                    "w": r["w"], "h": r["h"],
                    "agreement": r.get("agreement", False),
                }
                f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
        print(f"\n  [伪标签] {strategy}: {len(filtered)} bboxes → {pseudo_path}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        choices=["TeacherBehavior", "HandriseReadWrite", "BowTurnHead"],
                        required=True)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--ann_dir", default=None,
                        help='Annotation input directory (defaults to llm_annotations in repo root)')
    parser.add_argument("--out_dir", default=None,
                        help='Analysis output directory (defaults to analysis_results in repo root)')
    args = parser.parse_args()

    # Resolve sensible defaults relative to the repository root (parent of this file's parent)
    repo_root = Path(__file__).resolve().parents[1]
    ann_dir = Path(args.ann_dir) if args.ann_dir else repo_root / "llm_annotations"
    out_dir = Path(args.out_dir) if args.out_dir else repo_root / "analysis_results"

    analyze(args.dataset, ann_dir, args.split, out_dir)

# ══════════════════════════════════════════════════════════════════════════════
# 运行示例：
#   python step2_filter_analysis.py --dataset TeacherBehavior --split val
#   python step2_filter_analysis.py --dataset HandriseReadWrite --split val
#   python step2_filter_analysis.py --dataset BowTurnHead --split val
# ══════════════════════════════════════════════════════════════════════════════
