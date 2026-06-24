"""
Step 2b: Direct category-distribution shift analysis for agreement filtering.

Compares the original train distribution against the agreement-retained
distribution, with the none-retained subset included as a near-identity
reference. Outputs per-category counts/proportions plus divergence summaries.

Usage:
    python step2b_distribution_bias.py --dataset BowTurnHead
    python step2b_distribution_bias.py --dataset HandriseReadWrite
    python step2b_distribution_bias.py --all
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_latest_dir(prefix: str) -> Path:
    candidates = sorted(
        (path for path in SCRIPT_DIR.glob(f"{prefix}*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    return SCRIPT_DIR / prefix


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def proportions(counter: Counter, ordered_keys: Iterable[str]) -> dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        return {key: 0.0 for key in ordered_keys}
    return {key: counter.get(key, 0) / total for key in ordered_keys}


def kl_divergence(p: dict[str, float], q: dict[str, float], eps: float = 1e-12) -> float:
    value = 0.0
    for key, p_i in p.items():
        if p_i <= 0:
            continue
        q_i = max(q.get(key, 0.0), eps)
        value += p_i * math.log(p_i / q_i)
    return value


def total_variation(p: dict[str, float], q: dict[str, float]) -> float:
    return 0.5 * sum(abs(p.get(key, 0.0) - q.get(key, 0.0)) for key in p)


def analyze_dataset(dataset: str, ann_dir: Path, out_dir: Path) -> tuple[Path, Path]:
    ann_path = ann_dir / f"{dataset}_train_annotations.jsonl"
    if not ann_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {ann_path}")

    records = load_jsonl(ann_path)

    all_counter = Counter(r["gt_name"] for r in records)
    none_counter = Counter(r["gt_name"] for r in records if r.get("pred_qwen") is not None)
    agreement_counter = Counter(r["gt_name"] for r in records if r.get("agreement") is True)

    classes = sorted(all_counter)
    p_all = proportions(all_counter, classes)
    p_none = proportions(none_counter, classes)
    p_agreement = proportions(agreement_counter, classes)

    rows = []
    for class_name in classes:
        all_count = all_counter[class_name]
        none_count = none_counter[class_name]
        agreement_count = agreement_counter[class_name]
        all_pct = p_all[class_name] * 100
        none_pct = p_none[class_name] * 100
        agreement_pct = p_agreement[class_name] * 100
        rows.append(
            {
                "dataset": dataset,
                "class_name": class_name,
                "all_count": all_count,
                "all_pct": round(all_pct, 4),
                "none_count": none_count,
                "none_pct": round(none_pct, 4),
                "agreement_count": agreement_count,
                "agreement_pct": round(agreement_pct, 4),
                "agreement_minus_all_pct": round(agreement_pct - all_pct, 4),
                "none_minus_all_pct": round(none_pct - all_pct, 4),
            }
        )

    summary = {
        "dataset": dataset,
        "n_all": sum(all_counter.values()),
        "n_none": sum(none_counter.values()),
        "n_agreement": sum(agreement_counter.values()),
        "retention_none_pct": round(100 * sum(none_counter.values()) / sum(all_counter.values()), 4),
        "retention_agreement_pct": round(100 * sum(agreement_counter.values()) / sum(all_counter.values()), 4),
        "kl_all_to_none": round(kl_divergence(p_all, p_none), 8),
        "kl_all_to_agreement": round(kl_divergence(p_all, p_agreement), 8),
        "tv_all_to_none": round(total_variation(p_all, p_none), 8),
        "tv_all_to_agreement": round(total_variation(p_all, p_agreement), 8),
        "largest_positive_shift": max(rows, key=lambda row: row["agreement_minus_all_pct"]),
        "largest_negative_shift": min(rows, key=lambda row: row["agreement_minus_all_pct"]),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{dataset}_train_distribution_shift.csv"
    json_path = out_dir / f"{dataset}_train_distribution_shift_summary.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"[{dataset}] summary -> {json_path}")
    print(
        f"[{dataset}] retention none={summary['retention_none_pct']:.2f}% "
        f"agreement={summary['retention_agreement_pct']:.2f}% | "
        f"TV(all, agreement)={summary['tv_all_to_agreement']:.4f} | "
        f"KL(all||agreement)={summary['kl_all_to_agreement']:.4f}"
    )
    print(
        f"[{dataset}] max positive shift: {summary['largest_positive_shift']['class_name']} "
        f"({summary['largest_positive_shift']['agreement_minus_all_pct']:+.2f} pp)"
    )
    print(
        f"[{dataset}] max negative shift: {summary['largest_negative_shift']['class_name']} "
        f"({summary['largest_negative_shift']['agreement_minus_all_pct']:+.2f} pp)"
    )

    return csv_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze class-distribution shift induced by agreement filtering")
    parser.add_argument(
        "--dataset",
        choices=["BowTurnHead", "HandriseReadWrite", "TeacherBehavior"],
        default=None,
    )
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--ann_dir",
        default=str(resolve_latest_dir("llm_annotations")),
        help="Directory containing *_train_annotations.jsonl files",
    )
    parser.add_argument(
        "--out_dir",
        default=str(resolve_latest_dir("analysis_results")),
        help="Directory to receive CSV/JSON outputs",
    )
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset and not args.all else ["BowTurnHead", "HandriseReadWrite", "TeacherBehavior"]
    ann_dir = Path(args.ann_dir)
    out_dir = Path(args.out_dir)

    for dataset in datasets:
        analyze_dataset(dataset, ann_dir, out_dir)


if __name__ == "__main__":
    main()