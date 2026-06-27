"""
Step 5: Retention Curve — intermediate points experiment
For BowTurnHead, HandriseReadWrite, and TeacherBehavior using none pseudo-labels,
run Linear Probe on sampled train subsets and build a retention curve.

This script adds 25% / 50% / 75% points to complement main-table endpoints.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python step5_retention_curve.py --dataset BowTurnHead
    CUDA_VISIBLE_DEVICES=0,1 python step5_retention_curve.py --dataset HandriseReadWrite
    CUDA_VISIBLE_DEVICES=0,1 python step5_retention_curve.py --dataset TeacherBehavior
    CUDA_VISIBLE_DEVICES=0,1 python step5_retention_curve.py --all
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# HF mirror
HF_DEFAULT_ENDPOINT = "https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", HF_DEFAULT_ENDPOINT)
os.environ["HF_HUB_URL"] = os.environ.get("HF_HUB_URL", HF_DEFAULT_ENDPOINT)

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parent


def resolve_dataset_root() -> Path:
    candidates = []
    env_root = os.environ.get("SCB_DATASET_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.append(REPO_ROOT / "datasets_scb")
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Cannot find datasets_scb. Set SCB_DATASET_ROOT or place datasets_scb at the repository root. Checked: "
        + ", ".join(str(p) for p in candidates)
    )


def resolve_default_analysis_dir() -> Path:
    env_dir = os.environ.get("ANALYSIS_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    structured_root = SCRIPT_DIR / "results" / "phase2_filtering"
    if structured_root.is_dir():
        structured_candidates = sorted(
            (path for path in structured_root.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if structured_candidates:
            return structured_candidates[0]
    candidates = sorted(
        (path for path in SCRIPT_DIR.glob("analysis_results*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    return structured_root / "default"


DATASET_ROOT = resolve_dataset_root()
DEFAULT_ANALYSIS_DIR = resolve_default_analysis_dir()

DATASET_CFG = {
    "BowTurnHead": {
        "path": DATASET_ROOT / "SCB_BowTurnHead",
        "num_classes": 2,
    },
    "HandriseReadWrite": {
        "path": DATASET_ROOT / "SCB5_HandriseReadWrite",
        "num_classes": 3,
    },
    "TeacherBehavior": {
        "path": DATASET_ROOT / "SCB5_TeacherBehavior",
        "num_classes": 8,
    },
}

RETENTION_RATIOS = [0.25, 0.50, 0.75]

CLIP_MODEL = "ViT-L-14"
CLIP_PRETRAIN = "openai"
FEAT_DIM = 768
BATCH_LOG_INTERVAL = 10


def resolve_split_dirs(dataset_name: str, split: str) -> tuple[Path, Path]:
    base = DATASET_CFG[dataset_name]["path"]
    img_dir = base / "images" / split
    lbl_dir = base / "labels" / split
    if img_dir.is_dir() and lbl_dir.is_dir():
        return img_dir, lbl_dir
    imgs = sorted(p for p in base.glob(f"**/images/{split}") if p.is_dir())
    lbls = sorted(p for p in base.glob(f"**/labels/{split}") if p.is_dir())
    if imgs and lbls:
        return imgs[0], lbls[0]
    raise FileNotFoundError(f"Cannot find {dataset_name}/{split}")


class PseudoLabelDataset(Dataset):
    """Load training samples from none pseudo-label JSONL."""

    def __init__(
        self,
        dataset_name: str,
        pseudo_jsonl: Path,
        transform,
        margin: float = 0.05,
        min_px: int = 32,
    ):
        self.transform = transform
        self.margin = margin
        self.min_px = min_px
        self.num_classes = DATASET_CFG[dataset_name]["num_classes"]
        self.samples: list[tuple] = []

        img_dir, _ = resolve_split_dirs(dataset_name, "train")

        with open(pseudo_jsonl) as f:
            for line in f:
                r = json.loads(line)
                label = r.get("pseudo_label")
                if label is None or label >= self.num_classes:
                    continue
                img_path = img_dir / r["image"]
                if not img_path.exists():
                    continue
                self.samples.append((img_path, r["cx"], r["cy"], r["w"], r["h"], label))

        logger.info("%s pseudo train: %d crops", dataset_name, len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, cx, cy, w, h, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            return torch.zeros(3, 224, 224), label

        width, height = img.size
        x1 = max(0, (cx - w / 2 - self.margin) * width)
        y1 = max(0, (cy - h / 2 - self.margin) * height)
        x2 = min(width, (cx + w / 2 + self.margin) * width)
        y2 = min(height, (cy + h / 2 + self.margin) * height)
        if (x2 - x1) < self.min_px or (y2 - y1) < self.min_px:
            return torch.zeros(3, 224, 224), label

        return self.transform(img.crop((x1, y1, x2, y2))), label


class GTValDataset(Dataset):
    """Load val samples with GT labels from YOLO txt labels."""

    def __init__(self, dataset_name: str, transform, margin: float = 0.05, min_px: int = 32):
        self.transform = transform
        self.margin = margin
        self.min_px = min_px
        self.num_classes = DATASET_CFG[dataset_name]["num_classes"]
        self.samples: list[tuple] = []

        img_dir, lbl_dir = resolve_split_dirs(dataset_name, "val")

        for lbl_file in sorted(lbl_dir.glob("*.txt")):
            img_path = img_dir / (lbl_file.stem + ".jpg")
            if not img_path.exists():
                img_path = img_dir / (lbl_file.stem + ".png")
            if not img_path.exists():
                continue
            for line in lbl_file.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                gt = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                self.samples.append((img_path, cx, cy, w, h, gt))

        logger.info("%s GT val: %d crops", dataset_name, len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, cx, cy, w, h, gt = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            return torch.zeros(3, 224, 224), gt

        width, height = img.size
        x1 = max(0, (cx - w / 2 - self.margin) * width)
        y1 = max(0, (cy - h / 2 - self.margin) * height)
        x2 = min(width, (cx + w / 2 + self.margin) * width)
        y2 = min(height, (cy + h / 2 + self.margin) * height)
        if (x2 - x1) < self.min_px or (y2 - y1) < self.min_px:
            return torch.zeros(3, 224, 224), gt

        return self.transform(img.crop((x1, y1, x2, y2))), gt


def train_linear_probe(
    visual: nn.Module,
    num_classes: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    log_prefix: str,
) -> tuple[float, int]:
    """Train linear probe and return (best_val_acc, best_epoch)."""
    head = nn.Linear(FEAT_DIM, num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        visual.eval()
        head.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        num_batches = len(train_loader)
        for batch_idx, (imgs, labels) in enumerate(train_loader, start=1):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                feats = visual(imgs).float()
            logits = head(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size_now = len(labels)
            train_loss_sum += loss.item() * batch_size_now
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += batch_size_now

            if batch_idx % BATCH_LOG_INTERVAL == 0 or batch_idx == num_batches:
                running_loss = train_loss_sum / train_total if train_total else 0.0
                running_acc = 100.0 * train_correct / train_total if train_total else 0.0
                logger.info(
                    "%s epoch %d/%d batch %d/%d | running_loss=%.4f running_acc=%.2f%%",
                    log_prefix,
                    epoch,
                    epochs,
                    batch_idx,
                    num_batches,
                    running_loss,
                    running_acc,
                )
        scheduler.step()

        visual.eval()
        head.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                feats = visual(imgs).float()
                logits = head(feats)
                preds = logits.argmax(1).cpu()
                correct += (preds == labels).sum().item()
                total += len(labels)

        val_acc = 100.0 * correct / total if total else 0.0
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch

        train_loss = train_loss_sum / train_total if train_total else 0.0
        train_acc = 100.0 * train_correct / train_total if train_total else 0.0
        logger.info(
            "%s epoch %d/%d | train_loss=%.4f train_acc=%.2f%% val_acc=%.2f%% best=%.2f%%@%d",
            log_prefix,
            epoch,
            epochs,
            train_loss,
            train_acc,
            val_acc,
            best_acc,
            best_epoch,
        )

    return best_acc, best_epoch


def run_dataset(
    dataset_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    analysis_dir: Path,
    num_workers: int,
    device: torch.device,
    visual: nn.Module,
    preprocess,
) -> list[dict]:
    """Run all retention ratios for one dataset and return result rows."""
    pseudo_jsonl = analysis_dir / f"{dataset_name}_train_pseudo_none.jsonl"
    if not pseudo_jsonl.exists():
        raise FileNotFoundError(f"Pseudo-label file not found: {pseudo_jsonl}")

    num_classes = DATASET_CFG[dataset_name]["num_classes"]

    full_train_ds = PseudoLabelDataset(dataset_name, pseudo_jsonl, preprocess)
    val_ds = GTValDataset(dataset_name, preprocess)

    if len(full_train_ds) == 0:
        raise RuntimeError(f"No pseudo-labeled samples found for {dataset_name}.")
    if len(val_ds) == 0:
        raise RuntimeError(f"No val samples found for {dataset_name}.")

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["multiprocessing_context"] = "spawn"

    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    n_total = len(full_train_ds)
    results = []

    for ratio in RETENTION_RATIOS:
        rng = random.Random(seed)
        n_keep = max(1, int(n_total * ratio))
        indices = rng.sample(range(n_total), n_keep)
        subset = Subset(full_train_ds, indices)

        train_loader = DataLoader(subset, shuffle=True, **loader_kwargs)

        logger.info(
            "[%s] ratio=%.0f%% | train=%d / %d",
            dataset_name,
            ratio * 100,
            n_keep,
            n_total,
        )

        best_acc, best_epoch = train_linear_probe(
            visual,
            num_classes,
            train_loader,
            val_loader,
            epochs,
            lr,
            device,
            log_prefix=f"[{dataset_name}][{int(ratio * 100)}%]",
        )

        logger.info(
            "[%s] ratio=%.0f%% -> best_val_acc=%.2f%% @ epoch %d",
            dataset_name,
            ratio * 100,
            best_acc,
            best_epoch,
        )

        results.append(
            {
                "dataset": dataset_name,
                "retention_pct": round(ratio * 100, 1),
                "n_train": n_keep,
                "n_total": n_total,
                "best_val_acc": round(best_acc, 4),
                "best_epoch": best_epoch,
                "epochs": epochs,
                "lr": lr,
                "seed": seed,
            }
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 5: Retention curve with intermediate data ratios"
    )
    parser.add_argument("--dataset", choices=list(DATASET_CFG), default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_dir", default="./results/phase5_retention_curve/default")
    parser.add_argument("--analysis_dir", default=str(DEFAULT_ANALYSIS_DIR))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("Device: %s | GPUs: %d", device, n_gpu)

    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL,
        pretrained=CLIP_PRETRAIN,
    )
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(device)

    visual: nn.Module
    if n_gpu > 1:
        visual = nn.DataParallel(model.visual)
    else:
        visual = model.visual
    visual = visual.to(device)

    datasets = list(DATASET_CFG) if (args.all or not args.dataset) else [args.dataset]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for ds in datasets:
        results = run_dataset(
            dataset_name=ds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            analysis_dir=Path(args.analysis_dir),
            num_workers=args.num_workers,
            device=device,
            visual=visual,
            preprocess=preprocess,
        )
        all_results.extend(results)

    out_file = out_dir / "phase5_retention_curve_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info("All results saved -> %s", out_file)

    print("\n=== Retention Curve Summary ===")
    print(f"{'Dataset':<22} {'Retention%':>10} {'n_train':>8} {'val_acc%':>10}")
    print("-" * 54)

    endpoints = {
        "BowTurnHead": [
            (100.0, 88.33, "<-main table none+LP"),
            (20.0, 14.39, "<-main table agree+LP"),
        ],
        "HandriseReadWrite": [
            (100.0, 75.61, "<-main table none+LP"),
            (55.9, 53.98, "<-main table agree+LP"),
        ],
        "TeacherBehavior": [
            (100.0, 42.27, "<-main table none+LP"),
            (66.54, 42.97, "<-main table agree+LP"),
        ],
    }

    for ds in datasets:
        ds_rows = [r for r in all_results if r["dataset"] == ds]
        if ds in endpoints:
            ep_start = endpoints[ds][0]
            print(
                f"{ds:<22} {ep_start[0]:>10.1f} {'(main)':>8} "
                f"{ep_start[1]:>10.2f}  {ep_start[2]}"
            )

        for r in ds_rows:
            print(
                f"{'':22} {r['retention_pct']:>10.1f} "
                f"{r['n_train']:>8} {r['best_val_acc']:>10.2f}"
            )

        if ds in endpoints:
            ep_end = endpoints[ds][1]
            print(
                f"{'':22} {ep_end[0]:>10.1f} {'(main)':>8} "
                f"{ep_end[1]:>10.2f}  {ep_end[2]}"
            )


if __name__ == "__main__":
    main()

# ══════════════════════════════════════════════════════════════════════════════
# 运行：
#
# 两个数据集全跑：
#   CUDA_VISIBLE_DEVICES=0,1 python step5_retention_curve.py --all
#
# 单独跑一个：
#   CUDA_VISIBLE_DEVICES=0,1 python step5_retention_curve.py --dataset BowTurnHead
#   CUDA_VISIBLE_DEVICES=0,1 python step5_retention_curve.py --dataset TeacherBehavior
#
# 预期输出结构（含主表端点参照）：
#   BowTurnHead       100.0%  (main)    88.33%  ← none+LP
#   BowTurnHead        75.0%   9273     ??.??%
#   BowTurnHead        50.0%   6183     ??.??%
#   BowTurnHead        25.0%   3092     ??.??%
#   BowTurnHead        20.0%  (main)    14.39%  ← agree+LP
#
#   HandriseReadWrite 100.0%  (main)    75.61%  ← none+LP
#   HandriseReadWrite  75.0%  25893     ??.??%
#   HandriseReadWrite  50.0%  17262     ??.??%
#   HandriseReadWrite  25.0%   8631     ??.??%
#   HandriseReadWrite  55.9%  (main)    53.98%  ← agree+LP
#
# 结果用于替换论文Figure 3，从两点连线变成完整曲线。
# ══════════════════════════════════════════════════════════════════════════════