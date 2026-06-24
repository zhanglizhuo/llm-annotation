from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent

DATASET_CFG = {
    "TeacherBehavior": {
        "classes": [
            "guide",
            "answer",
            "on-stage interaction",
            "blackboard-writing",
            "teacher",
            "stand",
            "screen",
            "blackboard",
        ],
    },
    "HandriseReadWrite": {"classes": ["hand-raise", "read", "write"]},
    "BowTurnHead": {"classes": ["bow-head", "turn-head"]},
}


MODEL_SPECS = [
    ("qwen2", "Qwen2-VL-7B", "main", ["pred_qwen"]),
    ("llava", "LLaVA-1.5-7B", "main", ["pred_llava"]),
    ("qwen25", "Qwen2.5-VL-7B", "cross", ["pred_qwen25"]),
    ("qwen2532", "Qwen2.5-VL-32B", "cross", ["pred_qwen2532", "pred_qwen36"]),
    ("qwen36_35b", "Qwen3.6-35B", "cross", ["pred_qwen36_35b"]),
    ("qwen36_35b_fp8", "Qwen3.6-35B-FP8", "cross", ["pred_qwen36_35b_fp8"]),
    ("gemma327", "Gemma-3-27B", "cross", ["pred_gemma327", "pred_gemma4"]),
]


def read_jsonl(path: Path) -> list:
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def record_id(record: dict) -> tuple[str, int]:
    return str(record["image"]), int(record["bbox_idx"])


def first_present(record: dict, fields: list[str]):
    for field in fields:
        if field in record:
            return record.get(field)
    return None


def load_clip_zs_from_json(path: Path) -> pd.DataFrame:
    data = json.load(open(path, encoding="utf-8"))
    rows = []
    for item in data:
        dataset = item["dataset"]
        classes = DATASET_CFG[dataset]["classes"]
        per_class = item.get("per_class_acc", {})
        for class_idx, class_name in enumerate(classes):
            rows.append(
                {
                    "dataset": dataset,
                    "class_idx": class_idx,
                    "class_name": class_name,
                    "zs_acc": float(per_class.get(class_name, np.nan)),
                }
            )
    return pd.DataFrame(rows)


def load_clip_zs(args) -> pd.DataFrame | None:
    if args.clip_zs_acc_csv:
        path = Path(args.clip_zs_acc_csv)
        if path.exists():
            return pd.read_csv(path)
        logger.warning("clip_zs_acc_csv not found: %s", path)
    if args.clip_zs_json:
        path = Path(args.clip_zs_json)
        if path.exists():
            return load_clip_zs_from_json(path)
        logger.warning("clip_zs_json not found: %s", path)
    return None


def load_joined_records(dataset: str, split: str, main_ann_dir: Path, crossmodel_dir: Path) -> list[dict]:
    main_file = main_ann_dir / f"{dataset}_{split}_annotations.jsonl"
    cross_file = crossmodel_dir / f"{dataset}_{split}_crossmodel_annotations.jsonl"
    if not main_file.exists():
        raise FileNotFoundError(f"Missing main annotation file: {main_file}")
    if not cross_file.exists():
        raise FileNotFoundError(f"Missing cross-model annotation file: {cross_file}")

    main_records = {record_id(record): record for record in read_jsonl(main_file)}
    cross_records = {record_id(record): record for record in read_jsonl(cross_file)}
    common_ids = sorted(set(main_records) & set(cross_records))
    logger.info("%s/%s common records: %d", dataset, split, len(common_ids))

    joined = []
    for rid in common_ids:
        main_record = main_records[rid]
        cross_record = cross_records[rid]
        out = {
            "image": main_record["image"],
            "bbox_idx": int(main_record["bbox_idx"]),
            "gt": int(main_record["gt"]),
            "gt_name": main_record.get("gt_name"),
        }
        for model_key, _, source, fields in MODEL_SPECS:
            source_record = main_record if source == "main" else cross_record
            pred = first_present(source_record, fields)
            out[model_key] = None if pred is None else int(pred)
        joined.append(out)
    return joined


def compute_consistency(dataset: str, records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    classes = DATASET_CFG[dataset]["classes"]
    model_keys = [spec[0] for spec in MODEL_SPECS]
    model_labels = {spec[0]: spec[1] for spec in MODEL_SPECS}
    pairs = list(itertools.combinations(model_keys, 2))
    rows = []
    pair_rows = []
    for class_idx, class_name in enumerate(classes):
        class_records = [record for record in records if int(record["gt"]) == class_idx]
        pair_rates = []
        weighted_correct = 0
        weighted_total = 0
        for left, right in pairs:
            valid = [
                record
                for record in class_records
                if record.get(left) is not None and record.get(right) is not None
            ]
            if not valid:
                continue
            agree = sum(1 for record in valid if record[left] == record[right])
            rate = agree / len(valid)
            pair_rates.append(rate)
            weighted_correct += agree
            weighted_total += len(valid)
            pair_rows.append(
                {
                    "dataset": dataset,
                    "class_idx": class_idx,
                    "class_name": class_name,
                    "model_a": model_labels[left],
                    "model_b": model_labels[right],
                    "n_pair_valid": len(valid),
                    "pair_agreement": round(100.0 * rate, 4),
                }
            )
        rows.append(
            {
                "dataset": dataset,
                "class_idx": class_idx,
                "class_name": class_name,
                "n_gt_samples": len(class_records),
                "n_model_pairs": len(pair_rates),
                "cross_model_consistency": round(100.0 * float(np.mean(pair_rates)), 4)
                if pair_rates
                else np.nan,
                "cross_model_consistency_weighted": round(100.0 * weighted_correct / weighted_total, 4)
                if weighted_total
                else np.nan,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


def correlation_row(dataset: str, merged: pd.DataFrame) -> dict:
    clean = merged.dropna(subset=["cross_model_consistency", "zs_acc"])
    result = {
        "dataset": dataset,
        "n_classes": int(len(clean)),
        "pearson_r": np.nan,
        "pearson_p": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
    }
    if len(clean) < 3:
        return result
    x = clean["cross_model_consistency"].astype(float).values
    y = clean["zs_acc"].astype(float).values
    if scipy_stats is not None:
        pearson = scipy_stats.pearsonr(x, y)
        spearman = scipy_stats.spearmanr(x, y)
        pearson_r = pearson.statistic if hasattr(pearson, "statistic") else pearson[0]
        pearson_p = pearson.pvalue if hasattr(pearson, "pvalue") else pearson[1]
        spearman_r = spearman.statistic if hasattr(spearman, "statistic") else spearman[0]
        spearman_p = spearman.pvalue if hasattr(spearman, "pvalue") else spearman[1]
        result.update(
            {
                "pearson_r": round(float(pearson_r), 6),
                "pearson_p": round(float(pearson_p), 6),
                "spearman_rho": round(float(spearman_r), 6),
                "spearman_p": round(float(spearman_p), 6),
            }
        )
    else:
        result["pearson_r"] = round(float(np.corrcoef(x, y)[0, 1]), 6)
        result["spearman_rho"] = round(float(pd.Series(x).corr(pd.Series(y), method="spearman")), 6)
    return result


def maybe_plot(dataset: str, merged: pd.DataFrame, corr: dict, output_dir: Path) -> None:
    if plt is None:
        logger.warning("matplotlib not available; skipping proxy scatter plot for %s", dataset)
        return
    clean = merged.dropna(subset=["cross_model_consistency", "zs_acc"])
    if clean.empty:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.scatter(clean["cross_model_consistency"], clean["zs_acc"], s=64, color="#4C78A8", edgecolors="white")
    for _, row in clean.iterrows():
        ax.annotate(
            row["class_name"],
            (row["cross_model_consistency"], row["zs_acc"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    if len(clean) >= 2:
        x = clean["cross_model_consistency"].astype(float).values
        y = clean["zs_acc"].astype(float).values
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(float(x.min()), float(x.max()), 100)
        ax.plot(x_line, slope * x_line + intercept, color="#E45756", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Cross-model consistency (%)")
    ax.set_ylabel("CLIP zero-shot per-class accuracy (%)")
    ax.set_title(f"{dataset}: r={corr.get('pearson_r', np.nan):.2f}, rho={corr.get('spearman_rho', np.nan):.2f}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"anchor_proxy_comparison_{dataset}.png", dpi=200)
    plt.close(fig)


def run(args) -> None:
    main_ann_dir = Path(args.main_ann_dir)
    crossmodel_dir = Path(args.crossmodel_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_zs_df = load_clip_zs(args)
    if clip_zs_df is None:
        logger.info("No CLIP zero-shot per-class file provided; writing CMC only.")

    consistency_frames = []
    pair_frames = []
    comparison_frames = []
    corr_rows = []
    for dataset in args.dataset:
        records = load_joined_records(dataset, args.split, main_ann_dir, crossmodel_dir)
        consistency_df, pair_df = compute_consistency(dataset, records)
        consistency_frames.append(consistency_df)
        pair_frames.append(pair_df)
        if clip_zs_df is not None:
            zs_sub = clip_zs_df[clip_zs_df["dataset"] == dataset][["class_idx", "zs_acc"]]
            merged = consistency_df.merge(zs_sub, on="class_idx", how="left")
            corr = correlation_row(dataset, merged)
            comparison_frames.append(merged)
            corr_rows.append(corr)
            maybe_plot(dataset, merged, corr, output_dir)
            logger.info("%s CMC-vs-ZS correlation: %s", dataset, corr)

    pd.concat(consistency_frames, ignore_index=True).to_csv(output_dir / "cross_model_consistency.csv", index=False)
    pd.concat(pair_frames, ignore_index=True).to_csv(output_dir / "cross_model_pairwise_agreement.csv", index=False)
    if comparison_frames:
        pd.concat(comparison_frames, ignore_index=True).to_csv(output_dir / "anchor_proxy_comparison.csv", index=False)
        pd.DataFrame(corr_rows).to_csv(output_dir / "anchor_proxy_correlations.csv", index=False)
        with open(output_dir / "anchor_proxy_correlations.json", "w", encoding="utf-8") as handle:
            json.dump(corr_rows, handle, indent=2, ensure_ascii=False)
    logger.info("Saved anchor proxy outputs to %s", output_dir)


def main() -> None:
    default_main = SCRIPT_DIR / "results" / "phase1_annotations" / "full_20260418_0001"
    default_cross = SCRIPT_DIR / "results" / "cross_model_validation" / "default"
    default_zs = SCRIPT_DIR / "results" / "phase0_zero_shot" / "canonical_20260425_231347" / "phase0_zero_shot_results.json"
    default_out = SCRIPT_DIR / "results" / "phase6_strategy_audit" / "cross_model_consistency" / "default"
    parser = argparse.ArgumentParser(description="Compute cross-model consistency as an independent anchoring check.")
    parser.add_argument("--dataset", nargs="+", choices=list(DATASET_CFG), default=list(DATASET_CFG))
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--main_ann_dir", default=str(default_main))
    parser.add_argument("--crossmodel_dir", default=str(default_cross))
    parser.add_argument("--clip_zs_json", default=str(default_zs))
    parser.add_argument("--clip_zs_acc_csv", default=None)
    parser.add_argument("--output_dir", default=str(default_out))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()