from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch

from strategy_audit_utils import (
    DATASET_CFG,
    SCRIPT_DIR,
    encode_records,
    load_clip,
    load_gt_records,
    maybe_limit_records,
    set_seed,
    stratified_labeled_indices,
    subset_tensor,
    train_linear_head,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def predict_with_linear_head(features: torch.Tensor, head_state: dict, num_classes: int, device: torch.device):
    head = torch.nn.Linear(features.shape[1], num_classes).to(device)
    head.load_state_dict(head_state)
    head.eval()
    logits_chunks = []
    chunk_size = 8192
    for start in range(0, len(features), chunk_size):
        logits_chunks.append(head(features[start : start + chunk_size].to(device)).cpu())
    logits = torch.cat(logits_chunks, dim=0) if logits_chunks else torch.empty(0, num_classes)
    probs = logits.softmax(dim=1)
    max_prob, pred = probs.max(dim=1)
    return pred.long(), max_prob.float()


def train_teacher_return_state(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    num_classes: int,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> tuple[dict, dict]:
    # Small local trainer because the shared helper intentionally returns only metrics.
    set_seed(seed)
    head = torch.nn.Linear(train_features.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_features.float(), train_labels.long()),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_features.float(), val_labels.long()),
        batch_size=min(max(batch_size * 16, 4096), max(1, len(val_features))),
        shuffle=False,
    )
    best_acc = 0.0
    best_epoch = 0
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        head.train()
        correct = total = 0
        total_loss = 0.0
        for feats, labels in train_loader:
            feats = feats.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = head(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += len(labels)
        scheduler.step()
        train_acc = 100.0 * correct / total if total else 0.0
        head.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats = feats.to(device)
                labels = labels.to(device)
                logits = head(feats)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += len(labels)
        val_acc = 100.0 * val_correct / val_total if val_total else 0.0
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / total if total else 0.0,
                "train_acc": train_acc,
                "val_acc": val_acc,
            }
        )
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    return best_state, {"best_val_acc": best_acc, "best_epoch": best_epoch, "history": history}


def run_dataset(args, dataset: str, device: torch.device) -> list[dict]:
    num_classes = DATASET_CFG[dataset]["num_classes"]
    train_records = load_gt_records(dataset, "train")
    val_records = load_gt_records(dataset, "val")
    train_records = maybe_limit_records(train_records, args.max_train_samples, seed=0)
    val_records = maybe_limit_records(val_records, args.max_val_samples, seed=0)
    logger.info("%s: GT train=%d | val=%d", dataset, len(train_records), len(val_records))

    _, visual, preprocess = load_clip(device)
    train_features, train_gt = encode_records(
        dataset,
        "train",
        train_records,
        preprocess,
        visual,
        device,
        args.batch_size,
        args.num_workers,
        label_key="label",
    )
    val_features, val_gt = encode_records(
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

    rows: list[dict] = []
    for seed in args.seeds:
        labeled_idx, unlabeled_idx = stratified_labeled_indices(
            train_gt,
            fraction=args.teacher_label_fraction,
            min_per_class=args.min_labeled_per_class,
            seed=seed,
        )
        logger.info(
            "%s seed=%d labeled=%d unlabeled=%d fraction=%.3f",
            dataset,
            seed,
            len(labeled_idx),
            len(unlabeled_idx),
            args.teacher_label_fraction,
        )
        teacher_state, teacher_metrics = train_teacher_return_state(
            subset_tensor(train_features, labeled_idx),
            subset_tensor(train_gt, labeled_idx),
            val_features,
            val_gt,
            num_classes=num_classes,
            seed=seed,
            epochs=args.teacher_epochs,
            batch_size=args.head_batch_size,
            lr=args.lr,
            device=device,
        )
        unlabeled_features = subset_tensor(train_features, unlabeled_idx)
        pseudo_pred, pseudo_conf = predict_with_linear_head(unlabeled_features, teacher_state, num_classes, device)
        keep_mask = pseudo_conf >= args.teacher_conf_threshold
        kept_unlabeled_features = unlabeled_features[keep_mask]
        kept_pseudo_labels = pseudo_pred[keep_mask]
        retention_pct = 100.0 * int(keep_mask.sum().item()) / len(unlabeled_idx) if unlabeled_idx else 0.0

        if args.student_include_labeled_gt:
            student_features = torch.cat([subset_tensor(train_features, labeled_idx), kept_unlabeled_features], dim=0)
            student_labels = torch.cat([subset_tensor(train_gt, labeled_idx), kept_pseudo_labels], dim=0)
        else:
            student_features = kept_unlabeled_features
            student_labels = kept_pseudo_labels

        student_metrics = train_linear_head(
            student_features,
            student_labels,
            val_features,
            val_gt,
            num_classes=num_classes,
            seed=seed,
            epochs=args.student_epochs,
            batch_size=args.head_batch_size,
            lr=args.lr,
            device=device,
        )
        row = {
            "dataset": dataset,
            "seed": seed,
            "teacher_label_fraction": args.teacher_label_fraction,
            "n_labeled_teacher": len(labeled_idx),
            "n_unlabeled_pool": len(unlabeled_idx),
            "n_pseudo_kept": int(keep_mask.sum().item()),
            "pseudo_retention_pct": round(retention_pct, 4),
            "teacher_conf_threshold": args.teacher_conf_threshold,
            "teacher_best_val_acc": round(float(teacher_metrics["best_val_acc"]), 4),
            "teacher_best_epoch": teacher_metrics["best_epoch"],
            "student_n_train": int(len(student_features)),
            "student_include_labeled_gt": args.student_include_labeled_gt,
            "student_best_val_acc": None
            if student_metrics["best_val_acc"] is None
            else round(float(student_metrics["best_val_acc"]), 4),
            "student_best_epoch": student_metrics["best_epoch"],
        }
        rows.append(row)
        logger.info(
            "%s seed=%d teacher=%.2f student=%s kept=%d",
            dataset,
            seed,
            row["teacher_best_val_acc"],
            row["student_best_val_acc"],
            row["n_pseudo_kept"],
        )
    return rows


def summarize(df: pd.DataFrame) -> dict:
    summary: dict[str, dict] = {}
    for dataset, sub in df.groupby("dataset"):
        clean = sub.dropna(subset=["student_best_val_acc"])
        if clean.empty:
            continue
        summary[dataset] = {
            "student_val_acc_mean": round(float(clean["student_best_val_acc"].mean()), 4),
            "student_val_acc_std": round(float(clean["student_best_val_acc"].std(ddof=1)), 4)
            if len(clean) > 1
            else 0.0,
            "teacher_val_acc_mean": round(float(clean["teacher_best_val_acc"].mean()), 4),
            "teacher_val_acc_std": round(float(clean["teacher_best_val_acc"].std(ddof=1)), 4)
            if len(clean) > 1
            else 0.0,
            "mean_pseudo_retention_pct": round(float(clean["pseudo_retention_pct"].mean()), 4),
            "mean_student_n_train": round(float(clean["student_n_train"].mean()), 2),
            "n_seeds": int(len(clean)),
        }
    return summary


def run(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | GPUs: %d", device, torch.cuda.device_count())
    all_rows: list[dict] = []
    for dataset in args.dataset:
        all_rows.extend(run_dataset(args, dataset, device))
    df = pd.DataFrame(all_rows)
    df.to_csv(output_dir / "teacher_student_results.csv", index=False)
    summary = summarize(df)
    with open(output_dir / "teacher_student_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    pd.DataFrame([{"dataset": dataset, **values} for dataset, values in summary.items()]).to_csv(
        output_dir / "teacher_student_summary.csv",
        index=False,
    )
    logger.info("Saved teacher-student outputs to %s", output_dir)


def main() -> None:
    default_out = SCRIPT_DIR / "results" / "phase6_strategy_audit" / "teacher_student" / "default"
    parser = argparse.ArgumentParser(description="Train-split teacher-student self-training.")
    parser.add_argument("--dataset", nargs="+", choices=list(DATASET_CFG), default=list(DATASET_CFG))
    parser.add_argument("--output_dir", default=str(default_out))
    parser.add_argument("--seeds", nargs="+", type=int, default=[43, 44, 45, 46, 47])
    parser.add_argument("--teacher_label_fraction", type=float, default=0.1)
    parser.add_argument("--min_labeled_per_class", type=int, default=5)
    parser.add_argument("--teacher_conf_threshold", type=float, default=0.0)
    parser.set_defaults(student_include_labeled_gt=True)
    parser.add_argument("--student_include_labeled_gt", dest="student_include_labeled_gt", action="store_true")
    parser.add_argument("--student_exclude_labeled_gt", dest="student_include_labeled_gt", action="store_false")
    parser.add_argument("--teacher_epochs", type=int, default=20)
    parser.add_argument("--student_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128, help="Image-encoding batch size.")
    parser.add_argument("--head_batch_size", type=int, default=128, help="Cached-feature LP batch size.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_train_samples", type=int, default=None, help="Optional smoke-test cap.")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Optional smoke-test cap.")
    args = parser.parse_args()
    if not 0.0 < args.teacher_label_fraction <= 1.0:
        raise ValueError("teacher_label_fraction must be in (0, 1].")
    run(args)


if __name__ == "__main__":
    main()