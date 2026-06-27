"""
Step 0: 零样本评估 - 用 bbox 裁剪图像 + CLIP ViT-L/14 + 相同提示词在 val 集上跑分类

Usage examples:
    CUDA_VISIBLE_DEVICES=0 python step0_zeroshot_eval.py --dataset BowTurnHead
    CUDA_VISIBLE_DEVICES=0 python step0_zeroshot_eval.py --dataset HandriseReadWrite
    CUDA_VISIBLE_DEVICES=0 python step0_zeroshot_eval.py --dataset TeacherBehavior
    CUDA_VISIBLE_DEVICES=0 python step0_zeroshot_eval.py --all

This is a copy of step4_zeroshot_eval.py but with automatic DataParallel wrapping
when multiple GPUs are available so it can leverage two GPUs when run with
`CUDA_VISIBLE_DEVICES=0,1`.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import open_clip
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── HF镜像 ──
HF_DEFAULT_ENDPOINT = "https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", HF_DEFAULT_ENDPOINT)

# ── 路径配置 ──
REPO_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = REPO_ROOT / "datasets_scb"

DATASET_CFG = {
    "TeacherBehavior": {
        "path": DATASET_ROOT / "SCB5_TeacherBehavior",
        "classes": [
            "guide", "answer", "on-stage interaction", "blackboard-writing",
            "teacher", "stand", "screen", "blackboard",
        ],
        "num_classes": 8,
    },
    "HandriseReadWrite": {
        "path": DATASET_ROOT / "SCB5_HandriseReadWrite",
        "classes": ["hand-raise", "read", "write"],
        "num_classes": 3,
    },
    "BowTurnHead": {
        "path": DATASET_ROOT / "SCB_BowTurnHead",
        "classes": ["bow-head", "turn-head"],
        "num_classes": 2,
    },
}


def resolve_split_dirs(dataset_name: str, split: str):
    base = DATASET_CFG[dataset_name]["path"]
    img_dir = base / "images" / split
    lbl_dir = base / "labels" / split
    if img_dir.is_dir() and lbl_dir.is_dir():
        return img_dir, lbl_dir
    imgs = sorted(p for p in base.glob(f"**/images/{split}") if p.is_dir())
    lbls = sorted(p for p in base.glob(f"**/labels/{split}") if p.is_dir())
    if imgs and lbls:
        return imgs[0], lbls[0]
    raise FileNotFoundError(
        f"Cannot find images/labels for {dataset_name}/{split} under {base}"
    )


class ZeroShotDataset(torch.utils.data.Dataset):
    """从YOLO标签文件加载val集样本，用于零样本评估。"""
    def __init__(
        self,
        dataset_name: str,
        split: str,
        transform,
        margin: float = 0.05,
        min_px: int = 32,
    ):
        self.cfg = DATASET_CFG[dataset_name]
        self.img_dir, self.lbl_dir = resolve_split_dirs(dataset_name, split)
        self.transform = transform
        self.margin = margin
        self.min_px = min_px
        self.samples = []

        for lbl_file in sorted(self.lbl_dir.glob("*.txt")):
            img_path = self.img_dir / (lbl_file.stem + ".jpg")
            if not img_path.exists():
                img_path = self.img_dir / (lbl_file.stem + ".png")
            if not img_path.exists():
                continue
            for line in lbl_file.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                gt = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                self.samples.append((img_path, cx, cy, w, h, gt))

        logger.info("ZeroShotDataset %s/%s: %d samples", dataset_name, split, len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cx, cy, w, h, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            return torch.zeros(3, 224, 224), label

        W, H = img.size
        x1 = max(0, (cx - w / 2 - self.margin) * W)
        y1 = max(0, (cy - h / 2 - self.margin) * H)
        x2 = min(W, (cx + w / 2 + self.margin) * W)
        y2 = min(H, (cy + h / 2 + self.margin) * H)
        if (x2 - x1) < self.min_px or (y2 - y1) < self.min_px:
            return torch.zeros(3, 224, 224), label

        return self.transform(img.crop((x1, y1, x2, y2))), label


def build_text_prompts(classes):
    """构建与微调实验一致的文本提示词。"""
    prompts = []
    for cls in classes:
        prompts.append(f"a photo of a {cls}")
    return prompts


@torch.no_grad()
def zeroshot_classifier(model, text_tokenized, device):
    """构建零样本分类器。"""
    base_model = model.module if hasattr(model, "module") else model
    text_features = base_model.encode_text(text_tokenized)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.T  # (num_classes, num_features)


def run_zeroshot(
    dataset_name: str,
    model,
    preprocess,
    text_tokenized,
    device,
):
    """运行零样本评估并返回结果。"""
    cfg = DATASET_CFG[dataset_name]
    classes = cfg["classes"]

    val_ds = ZeroShotDataset(dataset_name, "val", preprocess)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    # 构建分类器
    classifier = zeroshot_classifier(model, text_tokenized, device)

    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    per_class_correct = {cls: 0 for cls in classes}
    per_class_total = {cls: 0 for cls in classes}

    for imgs, labels in tqdm(val_loader, desc=f"zeroshot {dataset_name}", leave=False):
        imgs = imgs.to(device)
        base_model = model.module if hasattr(model, "module") else model
        image_features = base_model.encode_image(imgs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = (image_features @ classifier)
        preds = logits.argmax(1)

        correct += (preds == labels.to(device)).sum().item()
        total += len(labels)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        for pred, lbl in zip(preds.cpu().tolist(), labels.cpu().tolist()):
            cls_name = classes[lbl]
            per_class_total[cls_name] += 1
            if pred == lbl:
                per_class_correct[cls_name] += 1

    acc = 100.0 * correct / total if total > 0 else 0.0

    # 打印每类结果
    logger.info(f"\n{'='*60}")
    logger.info(f"Zero-shot CLIP ViT-L/14 on {dataset_name} (val set)")
    logger.info(f"{'='*60}")
    logger.info(f"Total samples: {total}")
    logger.info(f"Overall accuracy: {acc:.2f}%")
    logger.info(f"\nPer-class results:")
    logger.info(f"{'Class':<30} {'Samples':>8} {'Correct':>8} {'Accuracy':>10}")
    logger.info(f"{'-'*58}")
    for cls in classes:
        n = per_class_total[cls]
        c = per_class_correct[cls]
        a = 100.0 * c / n if n > 0 else 0.0
        logger.info(f"{cls:<30} {n:>8} {c:>8} {a:>9.2f}%")
    logger.info(f"{'='*60}\n")

    return {
        "dataset": dataset_name,
        "model": "CLIP_ViT-L-14_openai",
        "method": "zero_shot",
        "total_samples": total,
        "overall_acc": acc,
        "per_class_acc": {cls: 100.0 * per_class_correct[cls] / per_class_total[cls] if per_class_total[cls] > 0 else 0.0 for cls in classes},
        "per_class_samples": per_class_total,
    }


def main():
    parser = argparse.ArgumentParser(description="Zero-shot CLIP evaluation on SCB val sets")
    parser.add_argument("--dataset", choices=list(DATASET_CFG), default=None)
    parser.add_argument("--all", action="store_true", help="Run on all datasets")
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "results" / "phase0_zero_shot" / "manual" / "phase0_zero_shot_results.json"),
        help="Output JSON file",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # 加载CLIP模型
    logger.info("Loading CLIP ViT-L/14...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    model = model.to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and 'cuda' in str(device):
        logger.info("Wrapping model with DataParallel across %d devices", torch.cuda.device_count())
        model = nn.DataParallel(model)
    model.eval()

    # 构建文本特征
    all_results = []

    datasets_to_run = list(DATASET_CFG.keys()) if args.all or not args.dataset else [args.dataset]

    for dataset_name in datasets_to_run:
        cfg = DATASET_CFG[dataset_name]
        classes = cfg["classes"]

        # 构建文本提示词
        text_prompts = build_text_prompts(classes)
        logger.info(f"Text prompts for {dataset_name}: {text_prompts}")

        #  tokenize
        text_tokenized = open_clip.tokenize(text_prompts).to(device)

        # 运行零样本评估
        result = run_zeroshot(dataset_name, model, preprocess, text_tokenized, device)
        all_results.append(result)

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
