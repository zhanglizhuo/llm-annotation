from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm


logger = logging.getLogger(__name__)

try:
    import open_clip
except Exception:
    open_clip = None

HF_DEFAULT_ENDPOINT = "https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", HF_DEFAULT_ENDPOINT)
os.environ["HF_HUB_URL"] = os.environ.get("HF_HUB_URL", os.environ["HF_ENDPOINT"])
os.environ["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parent


def resolve_dataset_root() -> Path:
    candidates: list[Path] = []
    env_root = os.environ.get("SCB_DATASET_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.append(REPO_ROOT / "datasets_scb")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Cannot find datasets_scb. Set SCB_DATASET_ROOT or place datasets_scb "
        "at the repository root. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


DATASET_ROOT = resolve_dataset_root()

DATASET_CFG = {
    "TeacherBehavior": {
        "path": DATASET_ROOT / "SCB5_TeacherBehavior",
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

CLIP_MODEL = "ViT-L-14"
CLIP_PRETRAIN = "openai"
FEAT_DIM = 768
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 0.01


def iter_progress(iterable, desc: str):
    return tqdm(iterable, desc=desc, leave=False, disable=not os.isatty(2))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def resolve_split_dirs(dataset_name: str, split: str) -> tuple[Path, Path]:
    base = DATASET_CFG[dataset_name]["path"]
    img_dir = base / "images" / split
    lbl_dir = base / "labels" / split
    if img_dir.is_dir() and lbl_dir.is_dir():
        return img_dir, lbl_dir

    img_candidates = sorted(path for path in base.glob(f"**/images/{split}") if path.is_dir())
    lbl_candidates = sorted(path for path in base.glob(f"**/labels/{split}") if path.is_dir())
    if img_candidates and lbl_candidates:
        return img_candidates[0], lbl_candidates[0]

    raise FileNotFoundError(f"Cannot find images/labels for {dataset_name}/{split} under {base}")


def resolve_latest_analysis_dir() -> Path:
    env_dir = os.environ.get("ANALYSIS_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    base = SCRIPT_DIR / "results" / "phase2_filtering"
    if base.is_dir():
        candidates = sorted(
            (path for path in base.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
    return base / "default"


def find_image_path(img_dir: Path, image_name: str) -> Optional[Path]:
    direct = img_dir / image_name
    if direct.exists():
        return direct
    stem = Path(image_name).stem
    for suffix in (".jpg", ".jpeg", ".png", ".bmp"):
        candidate = img_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def load_gt_records(dataset_name: str, split: str) -> list[dict]:
    img_dir, lbl_dir = resolve_split_dirs(dataset_name, split)
    records: list[dict] = []
    for lbl_file in sorted(lbl_dir.glob("*.txt")):
        image_path = find_image_path(img_dir, lbl_file.name.replace(".txt", ".jpg"))
        if image_path is None:
            continue
        lines = lbl_file.read_text(encoding="utf-8").strip().splitlines()
        for bbox_idx, line in enumerate(lines):
            parts = line.split()
            if len(parts) < 5:
                continue
            gt = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            records.append(
                {
                    "image": image_path.name,
                    "bbox_idx": bbox_idx,
                    "cx": cx,
                    "cy": cy,
                    "w": w,
                    "h": h,
                    "gt": gt,
                    "label": gt,
                }
            )
    return records


def load_pseudo_records(dataset_name: str, analysis_dir: Path, strategy: str = "none") -> list[dict]:
    pseudo_path = analysis_dir / f"{dataset_name}_train_pseudo_{strategy}.jsonl"
    if not pseudo_path.exists():
        raise FileNotFoundError(f"Pseudo-label file not found: {pseudo_path}")
    num_classes = DATASET_CFG[dataset_name]["num_classes"]
    records: list[dict] = []
    for record in read_jsonl(pseudo_path):
        label = record.get("pseudo_label")
        if label is None or int(label) >= num_classes:
            continue
        item = dict(record)
        item["label"] = int(label)
        records.append(item)
    return records


def maybe_limit_records(records: list[dict], max_records: Optional[int], seed: int) -> list[dict]:
    if max_records is None or max_records <= 0 or len(records) <= max_records:
        return records
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    keep = sorted(indices[:max_records])
    return [records[index] for index in keep]


def crop_bbox(img: Image.Image, cx: float, cy: float, w: float, h: float, margin: float, min_px: int):
    width, height = img.size
    x1 = max(0, (cx - w / 2 - margin) * width)
    y1 = max(0, (cy - h / 2 - margin) * height)
    x2 = min(width, (cx + w / 2 + margin) * width)
    y2 = min(height, (cy + h / 2 + margin) * height)
    if (x2 - x1) < min_px or (y2 - y1) < min_px:
        return None
    return img.crop((x1, y1, x2, y2))


class RecordCropDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        split: str,
        records: list[dict],
        transform,
        label_key: str = "label",
        margin: float = 0.05,
        min_px: int = 32,
    ):
        self.img_dir, _ = resolve_split_dirs(dataset_name, split)
        self.records = records
        self.transform = transform
        self.label_key = label_key
        self.margin = margin
        self.min_px = min_px

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        img_path = find_image_path(self.img_dir, record["image"])
        label = int(record[self.label_key])
        if img_path is None:
            return torch.zeros(3, 224, 224), label, index
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            return torch.zeros(3, 224, 224), label, index
        crop = crop_bbox(
            img,
            float(record["cx"]),
            float(record["cy"]),
            float(record["w"]),
            float(record["h"]),
            self.margin,
            self.min_px,
        )
        if crop is None:
            return torch.zeros(3, 224, 224), label, index
        return self.transform(crop), label, index


def load_clip(device: torch.device) -> tuple[nn.Module, nn.Module, object]:
    if open_clip is None:
        raise ImportError(
            "open_clip_torch is required for CLIP feature extraction. "
            "Install the repository requirements before running training baselines."
        )
    logger.info("Loading CLIP %s/%s", CLIP_MODEL, CLIP_PRETRAIN)
    logger.info("HF_HUB_OFFLINE=%s", os.environ.get("HF_HUB_OFFLINE"))
    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_PRETRAIN)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model = model.to(device)
    visual: nn.Module = model.visual
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        logger.info("Wrapping visual encoder with DataParallel across %d GPUs", torch.cuda.device_count())
        visual = nn.DataParallel(visual)
    visual = visual.to(device)
    visual.eval()
    model.eval()
    return model, visual, preprocess


def loader_kwargs(batch_size: int, num_workers: int, device: torch.device) -> dict:
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        kwargs["multiprocessing_context"] = "spawn"
    return kwargs


@torch.no_grad()
def encode_records(
    dataset_name: str,
    split: str,
    records: list[dict],
    transform,
    visual: nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    label_key: str = "label",
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = RecordCropDataset(dataset_name, split, records, transform, label_key=label_key)
    loader = DataLoader(dataset, shuffle=False, **loader_kwargs(batch_size, num_workers, device))
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    visual.eval()
    for imgs, batch_labels, _ in iter_progress(loader, desc=f"encode {dataset_name}/{split}"):
        imgs = imgs.to(device)
        feats = visual(imgs).float().cpu()
        features.append(feats)
        labels.append(batch_labels.clone().long())
    if not features:
        return torch.empty(0, FEAT_DIM), torch.empty(0, dtype=torch.long)
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


@torch.no_grad()
def build_zeroshot_classifier(model: nn.Module, classes: list[str], device: torch.device) -> torch.Tensor:
    if open_clip is None:
        raise ImportError("open_clip_torch is required to build the zero-shot classifier.")
    prompts = [f"a photo of a {class_name}" for class_name in classes]
    tokens = open_clip.tokenize(prompts).to(device)
    text_features = model.encode_text(tokens).float()
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.T


@torch.no_grad()
def clip_probabilities(features: torch.Tensor, classifier: torch.Tensor, device: torch.device) -> torch.Tensor:
    if features.numel() == 0:
        return torch.empty(0, classifier.shape[1])
    feats = features.to(device).float()
    feats = feats / feats.norm(dim=-1, keepdim=True)
    probs: list[torch.Tensor] = []
    chunk_size = 8192
    for start in range(0, len(feats), chunk_size):
        logits = feats[start : start + chunk_size] @ classifier
        probs.append(logits.softmax(dim=-1).cpu())
    return torch.cat(probs, dim=0)


def train_linear_head(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    num_classes: int,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    device: Optional[torch.device] = None,
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(train_features) == 0:
        return {"best_val_acc": None, "best_epoch": None, "history": []}
    set_seed(seed)
    head = nn.Linear(FEAT_DIM, num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    train_ds = TensorDataset(train_features.float(), train_labels.long())
    val_ds = TensorDataset(val_features.float(), val_labels.long())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=min(max(batch_size * 16, 4096), max(1, len(val_ds))), shuffle=False)
    best_acc = 0.0
    best_epoch = 0
    history = []
    for epoch in range(1, epochs + 1):
        head.train()
        total_loss = 0.0
        correct = 0
        total = 0
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
        train_loss = total_loss / total if total else 0.0

        head.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats = feats.to(device)
                labels = labels.to(device)
                logits = head(feats)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += len(labels)
        val_acc = 100.0 * val_correct / val_total if val_total else 0.0
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc})
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
    return {"best_val_acc": best_acc, "best_epoch": best_epoch, "history": history}


def stratified_labeled_indices(
    labels: torch.Tensor,
    fraction: float,
    min_per_class: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    labels_np = labels.cpu().numpy()
    labeled: list[int] = []
    unlabeled: list[int] = []
    for class_idx in sorted(set(int(value) for value in labels_np.tolist())):
        class_indices = [idx for idx, value in enumerate(labels_np) if int(value) == class_idx]
        rng.shuffle(class_indices)
        n_labeled = int(round(len(class_indices) * fraction))
        n_labeled = max(min_per_class, n_labeled)
        n_labeled = min(len(class_indices), n_labeled)
        labeled.extend(class_indices[:n_labeled])
        unlabeled.extend(class_indices[n_labeled:])
    labeled.sort()
    unlabeled.sort()
    return labeled, unlabeled


def subset_tensor(tensor: torch.Tensor, indices: Iterable[int]) -> torch.Tensor:
    index_list = list(indices)
    if not index_list:
        return tensor[:0]
    return tensor[torch.tensor(index_list, dtype=torch.long)]