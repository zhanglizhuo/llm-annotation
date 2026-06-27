"""
Step 4: Selective Annotation Experiment (方案B)
- 高锚点类别（blackboard-writing, teacher, screen, blackboard）：使用伪标签微调CLIP分类头
- 低锚点类别（answer, guide, on-stage interaction, stand）：退回零样本CLIP预测
- 合并两部分预测，计算整体val准确率，与主表的42.27%(none+LP)直接对比

核心逻辑：
  1. 从none伪标签里只保留高锚点4类的训练样本
  2. 训练一个4类分类头（只对高锚点类别）
  3. Val评估时：
     - 如果GT是高锚点类别 -> 用微调分类头预测
     - 如果GT是低锚点类别 -> 用零样本CLIP预测（CAPE ensemble）
  4. 合并计算整体8类准确率

运行：
  CUDA_VISIBLE_DEVICES=0,1 python step4_selective_annotation.py --mode linear
  CUDA_VISIBLE_DEVICES=0,1 python step4_selective_annotation.py --mode lora
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import open_clip
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from huggingface_hub import hf_hub_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# HF mirror
HF_DEFAULT_ENDPOINT = "https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", HF_DEFAULT_ENDPOINT)
os.environ["HF_HUB_URL"] = os.environ.get("HF_HUB_URL", HF_DEFAULT_ENDPOINT)
os.environ["HUGGINGFACE_HUB_URL"] = os.environ.get("HUGGINGFACE_HUB_URL", HF_DEFAULT_ENDPOINT)
os.environ["HUGGINGFACE_CO_RESOLVE_ENDPOINT"] = os.environ.get(
    "HUGGINGFACE_CO_RESOLVE_ENDPOINT", HF_DEFAULT_ENDPOINT
)

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

# TeacherBehavior class config
ALL_CLASSES = [
    "guide",                  # 0 low-anchor
    "answer",                 # 1 low-anchor
    "on-stage interaction",   # 2 low-anchor
    "blackboard-writing",     # 3 high-anchor
    "teacher",                # 4 high-anchor
    "stand",                  # 5 low-anchor
    "screen",                 # 6 high-anchor
    "blackboard",             # 7 high-anchor
]
NUM_ALL_CLASSES = 8

# High-anchor class original indices (in ALL_CLASSES)
HIGH_ANCHOR_ORIG_IDX = [3, 4, 6, 7]
HIGH_ANCHOR_NAMES = ["blackboard-writing", "teacher", "screen", "blackboard"]

# Low-anchor class original indices
LOW_ANCHOR_ORIG_IDX = [0, 1, 2, 5]
LOW_ANCHOR_NAMES = ["guide", "answer", "on-stage interaction", "stand"]

# Mapping between original idx and local 4-way head idx
ORIG_TO_LOCAL = {orig: local for local, orig in enumerate(HIGH_ANCHOR_ORIG_IDX)}
LOCAL_TO_ORIG = {local: orig for local, orig in enumerate(HIGH_ANCHOR_ORIG_IDX)}

NUM_HIGH_ANCHOR = len(HIGH_ANCHOR_ORIG_IDX)

CLIP_MODEL = "ViT-L-14"
CLIP_PRETRAIN = "openai"
FEAT_DIM = 768
OPEN_CLIP_HF_REPO = "timm/vit_large_patch14_clip_224.openai"
OPEN_CLIP_FILENAMES = ["open_clip_model.safetensors", "open_clip_pytorch_model.bin"]


def resolve_open_clip_cache_dir() -> Path:
    return Path(
        os.environ.get("OPEN_CLIP_CACHE_DIR", str(REPO_ROOT / ".cache" / "open_clip"))
    ).expanduser()


def resolve_open_clip_checkpoint() -> Path:
    cache_dir = resolve_open_clip_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    for filename in OPEN_CLIP_FILENAMES:
        matches = sorted(cache_dir.rglob(filename))
        if matches:
            logger.info("Using cached open_clip checkpoint: %s", matches[0])
            return matches[0]

    endpoint = os.environ.get("HF_ENDPOINT", HF_DEFAULT_ENDPOINT)
    download_errors = []
    for filename in OPEN_CLIP_FILENAMES:
        try:
            local_path = hf_hub_download(
                repo_id=OPEN_CLIP_HF_REPO,
                filename=filename,
                cache_dir=cache_dir,
                endpoint=endpoint,
            )
            logger.info("Downloaded open_clip checkpoint via %s: %s", endpoint, local_path)
            return Path(local_path)
        except Exception as exc:
            download_errors.append(f"{filename}: {exc}")

    raise RuntimeError(
        "Failed to resolve open_clip checkpoint from local cache or hf-mirror. "
        + " | ".join(download_errors)
    )


def create_clip_model_and_transforms():
    checkpoint_path = resolve_open_clip_checkpoint()
    return open_clip.create_model_and_transforms(
        CLIP_MODEL,
        pretrained=str(checkpoint_path),
        pretrained_hf=False,
        cache_dir=str(resolve_open_clip_cache_dir()),
    )

# CAPE prompts (Set A/B/C, TeacherBehavior only)
CAPE_PROMPTS = {
    "guide": [
        "a teacher guiding a student one-on-one",
        "a teacher helping a student at their desk",
        "a teacher walking among students and offering guidance",
        "a teacher leaning over a student's desk to help",
        "individual tutoring in a classroom setting",
        "a teacher assisting a single student with their work",
        "a person providing guidance in an indoor setting",
        "someone showing directions or instructions to another person",
        "a mentor giving advice to a learner",
    ],
    "answer": [
        "a teacher answering a student's question",
        "a teacher responding to a raised hand in class",
        "a student asking a question and the teacher replying",
        "a teacher explaining something to a student who asked a question",
        "a dialogue between teacher and student in a classroom",
        "a teacher addressing a student's raised hand",
        "a person giving an answer or reply",
        "someone responding verbally to a question",
        "a conversation where one person explains something",
    ],
    "on-stage interaction": [
        "a teacher interacting with students at the front of the classroom",
        "a teacher engaging with students on the podium",
        "a teacher and students having a discussion in front of the class",
        "a lively discussion between a teacher and students at the podium",
        "a teacher calling on students from the front of the room",
        "interactive teaching at the front of a classroom",
        "people interacting on a stage or platform",
        "a speaker engaging with an audience from a raised platform",
        "a person standing on stage communicating with others",
    ],
    "blackboard-writing": [
        "a teacher writing on a blackboard with chalk",
        "a hand writing equations on a chalkboard",
        "a teacher's back while writing on the blackboard",
        "chalk writing on a green or black chalkboard",
        "a teacher facing the blackboard writing notes",
        "handwritten text being written on a classroom board",
        "someone writing text on a large dark board",
        "handwriting being produced on a wall-mounted board",
        "a person using chalk to write on a board surface",
    ],
    "teacher": [
        "a teacher standing and explaining a concept",
        "a teacher giving a lecture at the podium",
        "a teacher talking to the class while standing",
        "a teacher delivering a lecture to students",
        "a teacher speaking in front of the classroom",
        "a professor explaining a topic while standing",
        "a person in the role of a teacher or instructor",
        "an educator standing in front of learners",
        "a professional conducting a lesson",
    ],
    "stand": [
        "a person standing still in a classroom",
        "a teacher standing at the front without interacting",
        "a teacher standing idle near the podium",
        "a teacher standing motionless in a classroom",
        "a person standing at the front of a classroom doing nothing",
        "an idle teacher standing near a desk",
        "a person standing upright in a room",
        "someone in a standing posture without movement",
        "a figure standing still indoors",
    ],
    "screen": [
        "a teacher pointing at a projection screen",
        "a teacher presenting slides on a screen",
        "a screen displaying a presentation in a classroom",
        "a projection screen showing slides in a classroom",
        "a teacher using a projector for a presentation",
        "a digital display showing educational content",
        "an electronic display or projection screen",
        "a monitor or screen showing digital content",
        "a flat display surface mounted in a room",
    ],
    "blackboard": [
        "a teacher pointing at the blackboard",
        "a teacher referring to content on the blackboard",
        "a blackboard with writing visible in a classroom",
        "a chalkboard with written content visible",
        "a teacher gesturing toward a blackboard",
        "educational content displayed on a classroom blackboard",
        "a dark-colored board mounted on a wall",
        "a chalkboard visible in a room",
        "a traditional writing board with content on it",
    ],
}


def resolve_split_dirs(split: str) -> tuple[Path, Path]:
    base = DATASET_ROOT / "SCB5_TeacherBehavior"
    img_dir = base / "images" / split
    lbl_dir = base / "labels" / split
    if img_dir.is_dir() and lbl_dir.is_dir():
        return img_dir, lbl_dir
    imgs = sorted(p for p in base.glob(f"**/images/{split}") if p.is_dir())
    lbls = sorted(p for p in base.glob(f"**/labels/{split}") if p.is_dir())
    if imgs and lbls:
        return imgs[0], lbls[0]
    raise FileNotFoundError(f"Cannot find TeacherBehavior/{split} under {base}")


class HighAnchorTrainDataset(Dataset):
    """Load only high-anchor pseudo-labeled samples for training."""

    def __init__(
        self,
        pseudo_jsonl: Path,
        transform,
        margin: float = 0.05,
        min_px: int = 32,
    ):
        self.transform = transform
        self.margin = margin
        self.min_px = min_px
        self.samples: list[tuple] = []

        img_dir, _ = resolve_split_dirs("train")

        with open(pseudo_jsonl) as f:
            for line in f:
                r = json.loads(line)
                pseudo_label = r.get("pseudo_label")
                if pseudo_label is None:
                    continue
                if pseudo_label not in HIGH_ANCHOR_ORIG_IDX:
                    continue
                img_path = img_dir / r["image"]
                if not img_path.exists():
                    continue
                local_label = ORIG_TO_LOCAL[pseudo_label]
                self.samples.append(
                    (img_path, r["cx"], r["cy"], r["w"], r["h"], local_label)
                )

        logger.info(
            "HighAnchorTrainDataset: %d crops from high-anchor classes (%s)",
            len(self.samples),
            ", ".join(
                f"{n}(orig={i})" for i, n in zip(HIGH_ANCHOR_ORIG_IDX, HIGH_ANCHOR_NAMES)
            ),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, cx, cy, w, h, local_label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            return torch.zeros(3, 224, 224), local_label

        width, height = img.size
        x1 = max(0, (cx - w / 2 - self.margin) * width)
        y1 = max(0, (cy - h / 2 - self.margin) * height)
        x2 = min(width, (cx + w / 2 + self.margin) * width)
        y2 = min(height, (cy + h / 2 + self.margin) * height)
        if (x2 - x1) < self.min_px or (y2 - y1) < self.min_px:
            return torch.zeros(3, 224, 224), local_label

        return self.transform(img.crop((x1, y1, x2, y2))), local_label


class FullValDataset(Dataset):
    """Load all TeacherBehavior val samples with original 8-way GT labels."""

    def __init__(self, transform, margin: float = 0.05, min_px: int = 32):
        self.transform = transform
        self.margin = margin
        self.min_px = min_px
        self.samples: list[tuple] = []

        img_dir, lbl_dir = resolve_split_dirs("val")

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

        logger.info("FullValDataset: %d crops (all 8 classes)", len(self.samples))

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


LORA_TARGET_KEYWORDS = [".mlp.c_fc", ".mlp.c_proj"]


class LoRALayer(nn.Module):
    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original = original
        self.scale = alpha / rank
        d_in, d_out = original.in_features, original.out_features
        self.lora_A = nn.Parameter(torch.empty(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        for p in self.original.parameters():
            p.requires_grad_(False)

    @property
    def weight(self) -> torch.Tensor:
        return self.original.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.original.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale


def inject_lora(visual: nn.Module, rank: int = 8, alpha: float = 16.0) -> list[LoRALayer]:
    injected: list[LoRALayer] = []
    targets = []
    for name, module in visual.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(kw in name for kw in LORA_TARGET_KEYWORDS):
            continue
        parts = name.split(".")
        parent = visual
        for p in parts[:-1]:
            parent = getattr(parent, p)
        targets.append((parent, parts[-1], module))

    for parent, attr, original in targets:
        ll = LoRALayer(original, rank=rank, alpha=alpha)
        setattr(parent, attr, ll)
        injected.append(ll)

    return injected


def collect_lora_params(lora_layers: list[LoRALayer]) -> list[nn.Parameter]:
    return [p for ll in lora_layers for p in [ll.lora_A, ll.lora_B]]


@torch.no_grad()
def build_zeroshot_classifier(model: nn.Module, device: torch.device) -> torch.Tensor:
    """Build CAPE ensemble text classifier for all 8 classes."""
    base = model.module if hasattr(model, "module") else model
    class_embeddings = []

    for cls in ALL_CLASSES:
        prompts = CAPE_PROMPTS.get(cls, [f"a photo of a {cls}"])
        tokens = open_clip.tokenize(prompts).to(device)
        feats = base.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        cls_emb = feats.mean(dim=0)
        cls_emb = cls_emb / cls_emb.norm()
        class_embeddings.append(cls_emb)

    classifier = torch.stack(class_embeddings, dim=1)
    logger.info("Zero-shot classifier built for all %d classes.", NUM_ALL_CLASSES)
    return classifier


def train_epoch(
    visual: nn.Module,
    head: nn.Linear,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mode: str,
) -> tuple[float, float]:
    visual.train() if mode == "lora" else visual.eval()
    head.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in tqdm(loader, desc="train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.set_grad_enabled(mode == "lora"):
            feats = visual(imgs)
        feats = feats.float()
        logits = head(feats)
        loss = criterion(logits, labels)
        loss.backward()

        if mode == "lora":
            nn.utils.clip_grad_norm_(
                [p for p in visual.parameters() if p.requires_grad] + list(head.parameters()),
                max_norm=1.0,
            )

        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate_hybrid(
    visual: nn.Module,
    head: nn.Linear,
    zs_classifier: torch.Tensor,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Hybrid eval on val set:
    - If GT is high-anchor class: use fine-tuned 4-way head
    - If GT is low-anchor class: use 8-way zero-shot CLIP
    """
    visual.eval()
    head.eval()

    stats = {
        "overall": {"correct": 0, "total": 0},
        "high_anchor": {"correct": 0, "total": 0},
        "low_anchor": {"correct": 0, "total": 0},
    }
    per_class = {cls: {"correct": 0, "total": 0} for cls in ALL_CLASSES}

    for imgs, gts in tqdm(loader, desc="eval_hybrid", leave=False):
        imgs = imgs.to(device)
        gts_list = gts.tolist()

        feats = visual(imgs).float()
        feats_norm = feats / feats.norm(dim=-1, keepdim=True)

        ft_logits = head(feats)
        ft_preds_local = ft_logits.argmax(1).cpu().tolist()

        zs_logits = feats_norm @ zs_classifier.float()
        zs_preds = zs_logits.argmax(1).cpu().tolist()

        for i, gt in enumerate(gts_list):
            cls_name = ALL_CLASSES[gt]
            if gt in HIGH_ANCHOR_ORIG_IDX:
                pred = LOCAL_TO_ORIG[ft_preds_local[i]]
                group = "high_anchor"
            else:
                pred = zs_preds[i]
                group = "low_anchor"

            is_correct = int(pred == gt)
            stats["overall"]["correct"] += is_correct
            stats["overall"]["total"] += 1
            stats[group]["correct"] += is_correct
            stats[group]["total"] += 1
            per_class[cls_name]["correct"] += is_correct
            per_class[cls_name]["total"] += 1

    def safe_acc(d: dict[str, int]) -> float:
        return 100.0 * d["correct"] / d["total"] if d["total"] else 0.0

    overall_acc = safe_acc(stats["overall"])
    high_anchor_acc = safe_acc(stats["high_anchor"])
    low_anchor_acc = safe_acc(stats["low_anchor"])

    logger.info("=" * 60)
    logger.info("Hybrid Selective Evaluation")
    logger.info(
        "  Overall  (8-class): %.2f%%  (%d/%d)",
        overall_acc,
        stats["overall"]["correct"],
        stats["overall"]["total"],
    )
    logger.info(
        "  High-anchor (4cls): %.2f%%  (%d/%d)",
        high_anchor_acc,
        stats["high_anchor"]["correct"],
        stats["high_anchor"]["total"],
    )
    logger.info(
        "  Low-anchor  (4cls): %.2f%%  (%d/%d)  [zero-shot]",
        low_anchor_acc,
        stats["low_anchor"]["correct"],
        stats["low_anchor"]["total"],
    )
    logger.info("-" * 60)
    logger.info("%-28s %8s %8s %10s", "Class", "n", "correct", "acc(%)")
    for cls in ALL_CLASSES:
        d = per_class[cls]
        tag = "[FT]" if ALL_CLASSES.index(cls) in HIGH_ANCHOR_ORIG_IDX else "[ZS]"
        logger.info(
            "%-28s %8d %8d %9.2f%%  %s",
            cls,
            d["total"],
            d["correct"],
            safe_acc(d),
            tag,
        )
    logger.info("=" * 60)

    return {
        "overall_acc": round(overall_acc, 4),
        "high_anchor_acc": round(high_anchor_acc, 4),
        "low_anchor_acc": round(low_anchor_acc, 4),
        "per_class_acc": {cls: round(safe_acc(per_class[cls]), 2) for cls in ALL_CLASSES},
        "per_class_n": {cls: per_class[cls]["total"] for cls in ALL_CLASSES},
    }


def run(
    mode: str,
    epochs: int,
    batch_size: int,
    lr: float,
    lora_rank: int,
    lora_alpha: float,
    seed: int,
    out_dir: Path,
    analysis_dir: Path,
    num_workers: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("Device: %s | GPUs: %d | mode: %s", device, n_gpu, mode)

    model, _, preprocess = create_clip_model_and_transforms()

    lora_layers: list[LoRALayer] = []
    if mode == "lora":
        lora_layers = inject_lora(model.visual, rank=lora_rank, alpha=lora_alpha)
        if len(lora_layers) == 0:
            raise RuntimeError("LoRA injection failed: no compatible layers found.")
        n_params = sum(p.numel() for ll in lora_layers for p in [ll.lora_A, ll.lora_B])
        logger.info("LoRA: %d layers | trainable params: %d", len(lora_layers), n_params)

        for p in model.parameters():
            p.requires_grad_(False)
        for ll in lora_layers:
            ll.lora_A.requires_grad_(True)
            ll.lora_B.requires_grad_(True)
    else:
        for p in model.parameters():
            p.requires_grad_(False)

    model = model.to(device)

    zs_classifier = build_zeroshot_classifier(model, device).to(device)

    visual: nn.Module
    if n_gpu > 1:
        visual = nn.DataParallel(model.visual)
    else:
        visual = model.visual
    visual = visual.to(device)

    head = nn.Linear(FEAT_DIM, NUM_HIGH_ANCHOR).to(device)

    if mode == "lora":
        trainable = collect_lora_params(lora_layers) + list(head.parameters())
    else:
        trainable = list(head.parameters())

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    logger.info("Optimizer trainable params: %d", sum(p.numel() for p in trainable))

    pseudo_jsonl = analysis_dir / "TeacherBehavior_train_pseudo_none.jsonl"
    if not pseudo_jsonl.exists():
        raise FileNotFoundError(f"Pseudo-label file not found: {pseudo_jsonl}")

    train_ds = HighAnchorTrainDataset(pseudo_jsonl, preprocess)
    val_ds = FullValDataset(preprocess)

    if len(train_ds) == 0:
        raise RuntimeError("No high-anchor pseudo-labeled samples found for training.")
    if len(val_ds) == 0:
        raise RuntimeError("No validation samples found for TeacherBehavior.")

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["multiprocessing_context"] = "spawn"

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    out_dir.mkdir(parents=True, exist_ok=True)
    best_overall_acc = 0.0
    best_epoch = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            visual,
            head,
            train_loader,
            optimizer,
            device,
            mode,
        )
        eval_result = evaluate_hybrid(visual, head, zs_classifier, val_loader, device)
        scheduler.step()

        overall_acc = eval_result["overall_acc"]
        logger.info(
            "Epoch %3d/%d | loss=%.4f train_acc=%.1f%% val_overall=%.2f%% high=%.2f%% low=%.2f%%",
            epoch,
            epochs,
            train_loss,
            train_acc,
            overall_acc,
            eval_result["high_anchor_acc"],
            eval_result["low_anchor_acc"],
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "overall_acc": overall_acc,
                "high_anchor_acc": eval_result["high_anchor_acc"],
                "low_anchor_acc": eval_result["low_anchor_acc"],
            }
        )

        if overall_acc > best_overall_acc:
            best_overall_acc = overall_acc
            best_epoch = epoch
            ckpt = {
                "head": head.state_dict(),
                "epoch": epoch,
            }
            if mode == "lora":
                ckpt["visual"] = visual.state_dict()
            torch.save(ckpt, out_dir / f"TeacherBehavior_selective_{mode}_best.pt")

    logger.info("Best overall val_acc: %.4f%% @ epoch %d", best_overall_acc, best_epoch)

    result = {
        "dataset": "TeacherBehavior",
        "mode": mode,
        "strategy": "selective_none",
        "high_anchor_classes": HIGH_ANCHOR_NAMES,
        "low_anchor_classes": LOW_ANCHOR_NAMES,
        "best_overall_acc": best_overall_acc,
        "best_epoch": best_epoch,
        "epochs": epochs,
        "lr": lr,
        "lora_rank": lora_rank if mode == "lora" else None,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "history": history,
    }
    result_path = out_dir / f"TeacherBehavior_selective_{mode}_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Result saved: %s", result_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Step 4: Selective annotation with hybrid prediction "
            "(high-anchor=fine-tuned, low-anchor=zero-shot)"
        )
    )
    parser.add_argument("--mode", choices=["linear", "lora"], default="linear")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_dir", default="./results/phase4_selective_annotation/default")
    parser.add_argument("--analysis_dir", default=str(DEFAULT_ANALYSIS_DIR))
    args = parser.parse_args()

    run(
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        seed=args.seed,
        out_dir=Path(args.out_dir),
        analysis_dir=Path(args.analysis_dir),
        num_workers=args.num_workers,
    )

# ══════════════════════════════════════════════════════════════════════════════
# 运行：
#
# Linear probe:
#   CUDA_VISIBLE_DEVICES=0,1 python step4_selective_annotation.py --mode linear
#
# LoRA:
#   CUDA_VISIBLE_DEVICES=0,1 python step4_selective_annotation.py --mode lora
#
# 预期结果对比（主表数字）：
#   主表 none+LP (全类别伪标签):  42.27%
#   主表 none+LoRA (全类别):      44.34%
#   主表 zero-shot CAPE:          53.43%  ← selective策略的对比基准
#
#   如果 selective+LP 或 selective+LoRA 的 overall_acc > 53.43%：
#   → 证明selective策略优于纯零样本，visual anchoring是可操作的标注准则
#
#   如果 high_anchor_acc 显著高于主表中对应类别在none策略下的准确率：
#   → 进一步证明去掉低质量伪标签对高锚点类别有帮助
# ══════════════════════════════════════════════════════════════════════════════