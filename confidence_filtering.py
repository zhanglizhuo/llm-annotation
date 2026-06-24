from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from strategy_audit_utils import (
    DATASET_CFG,
    SCRIPT_DIR,
    build_zeroshot_classifier,
    clip_probabilities,
    encode_records,
    load_clip,
    load_gt_records,
    load_pseudo_records,
    maybe_limit_records,
    resolve_latest_analysis_dir,
    set_seed,
    train_linear_head,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_thresholds(values: list[str]) -> list[float]:
    return [float(value) for value in values]


def summarize(df: pd.DataFrame, thresholds: list[float]) -> dict:
    summary: dict[str, dict] = {}
    for dataset, dataset_df in df.groupby("dataset"):
        threshold_rows = []
        for threshold in thresholds:
            sub = dataset_df[dataset_df["threshold"] == threshold].dropna(subset=["best_val_acc"])
            if sub.empty:
                continue
            threshold_rows.append(
                {
                    "threshold": threshold,
                    "mean_acc": float(sub["best_val_acc"].mean()),
                    "std_acc": float(sub["best_val_acc"].std(ddof=1)) if len(sub) > 1 else 0.0,
                    "mean_retention_pct": float(sub["retention_pct"].mean()),
                    "n_seeds": int(len(sub)),
                }
            )
        if not threshold_rows:
            continue
        best = max(threshold_rows, key=lambda row: row["mean_acc"])
        summary[dataset] = {
            "best_threshold": best["threshold"],
            "best_mean_acc": round(best["mean_acc"], 4),
            "best_std_acc": round(best["std_acc"], 4),
            "best_mean_retention_pct": round(best["mean_retention_pct"], 4),
            "thresholds": [
                {
                    "threshold": row["threshold"],
                    "mean_acc": round(row["mean_acc"], 4),
                    "std_acc": round(row["std_acc"], 4),
                    "mean_retention_pct": round(row["mean_retention_pct"], 4),
                    "n_seeds": row["n_seeds"],
                }
                for row in threshold_rows
            ],
        }
    return summary


def run_dataset(args, dataset: str, device: torch.device) -> list[dict]:
    num_classes = DATASET_CFG[dataset]["num_classes"]
    classes = DATASET_CFG[dataset]["classes"]
    analysis_dir = Path(args.analysis_dir)

    pseudo_records = load_pseudo_records(dataset, analysis_dir, strategy=args.pseudo_strategy)
    val_records = load_gt_records(dataset, "val")
    pseudo_records = maybe_limit_records(pseudo_records, args.max_train_samples, seed=0)
    val_records = maybe_limit_records(val_records, args.max_val_samples, seed=0)

    logger.info("%s: pseudo train=%d | val=%d", dataset, len(pseudo_records), len(val_records))
    model, visual, preprocess = load_clip(device)
    train_features, qwen_labels = encode_records(
        dataset,
        "train",
        pseudo_records,
        preprocess,
        visual,
        device,
        args.batch_size,
        args.num_workers,
        label_key="label",
    )
    val_features, val_labels = encode_records(
        dataset,
        "val",
        val_records,
        preprocess,
        visual,
        device,
        args.batch_size,
        args.num_workers,
        label_key="label",
    )

    classifier = build_zeroshot_classifier(model, classes, device)
    probs = clip_probabilities(train_features, classifier, device)
    clip_max_prob, clip_pred = probs.max(dim=1)
    qwen_label_indices = qwen_labels.long().clamp(min=0, max=num_classes - 1)
    clip_prob_at_qwen = probs[torch.arange(len(qwen_label_indices)), qwen_label_indices]

    if args.label_source == "qwen":
        train_labels_source = qwen_labels
    else:
        train_labels_source = clip_pred.long()

    if args.confidence_score == "pseudo_prob":
        confidence = clip_prob_at_qwen if args.label_source == "qwen" else clip_max_prob
    else:
        confidence = clip_max_prob

    agreement_mask = clip_pred.cpu().long() == qwen_labels.cpu().long()
    rows: list[dict] = []
    for threshold in args.thresholds:
        base_mask = confidence.cpu() >= threshold
        if args.require_clip_agreement:
            base_mask = base_mask & agreement_mask
        selected = torch.nonzero(base_mask, as_tuple=False).flatten()
        n_selected = int(selected.numel())
        retention_pct = 100.0 * n_selected / len(train_features) if len(train_features) else 0.0
        class_coverage = int(torch.unique(train_labels_source[selected]).numel()) if n_selected else 0
        logger.info(
            "%s threshold=%.3f selected=%d/%d (%.2f%%) class_coverage=%d/%d",
            dataset,
            threshold,
            n_selected,
            len(train_features),
            retention_pct,
            class_coverage,
            num_classes,
        )
        for seed in args.seeds:
            set_seed(seed)
            if n_selected == 0 or class_coverage < 1:
                rows.append(
                    {
                        "dataset": dataset,
                        "threshold": threshold,
                        "seed": seed,
                        "n_train": n_selected,
                        "retention_pct": round(retention_pct, 4),
                        "class_coverage": class_coverage,
                        "best_val_acc": None,
                        "best_epoch": None,
                        "label_source": args.label_source,
                        "confidence_score": args.confidence_score,
                        "require_clip_agreement": args.require_clip_agreement,
                    }
                )
                continue
            result = train_linear_head(
                train_features[selected],
                train_labels_source[selected].long(),
                val_features,
                val_labels,
                num_classes=num_classes,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.head_batch_size,
                lr=args.lr,
                device=device,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "threshold": threshold,
                    "seed": seed,
                    "n_train": n_selected,
                    "retention_pct": round(retention_pct, 4),
                    "class_coverage": class_coverage,
                    "best_val_acc": None if result["best_val_acc"] is None else round(float(result["best_val_acc"]), 4),
                    "best_epoch": result["best_epoch"],
                    "label_source": args.label_source,
                    "confidence_score": args.confidence_score,
                    "require_clip_agreement": args.require_clip_agreement,
                }
            )
            logger.info(
                "%s threshold=%.3f seed=%d best_val_acc=%s",
                dataset,
                threshold,
                seed,
                rows[-1]["best_val_acc"],
            )
    return rows


def run(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | GPUs: %d", device, torch.cuda.device_count())
    all_rows: list[dict] = []
    for dataset in args.dataset:
        all_rows.extend(run_dataset(args, dataset, device))
    df = pd.DataFrame(all_rows)
    df.to_csv(output_dir / "confidence_filtering_results.csv", index=False)
    summary = summarize(df, args.thresholds)
    with open(output_dir / "confidence_filtering_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    pd.DataFrame(
        [
            {"dataset": dataset, **values}
            for dataset, values in summary.items()
        ]
    ).to_csv(output_dir / "confidence_filtering_summary.csv", index=False)
    logger.info("Saved confidence-filtering outputs to %s", output_dir)


def main() -> None:
    default_out = SCRIPT_DIR / "results" / "phase6_strategy_audit" / "confidence_filtering" / "default"
    parser = argparse.ArgumentParser(description="CLIP-assisted confidence filtering for Qwen pseudo-labels.")
    parser.add_argument("--dataset", nargs="+", choices=list(DATASET_CFG), default=list(DATASET_CFG))
    parser.add_argument("--analysis_dir", default=str(resolve_latest_analysis_dir()))
    parser.add_argument("--output_dir", default=str(default_out))
    parser.add_argument("--pseudo_strategy", choices=["none", "agreement"], default="none")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.3, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--seeds", nargs="+", type=int, default=[43, 44, 45, 46, 47])
    parser.add_argument("--label_source", choices=["qwen", "clip"], default="qwen")
    parser.add_argument("--confidence_score", choices=["pseudo_prob", "max_prob"], default="pseudo_prob")
    parser.add_argument("--require_clip_agreement", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128, help="Image-encoding batch size.")
    parser.add_argument("--head_batch_size", type=int, default=128, help="Cached-feature LP batch size.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_train_samples", type=int, default=None, help="Optional smoke-test cap.")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Optional smoke-test cap.")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()