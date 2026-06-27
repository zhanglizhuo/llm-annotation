"""
Step 3: 用伪标签微调CLIP（LoRA / Linear Probe）
修复：LoRA注入到MLP层，参数收集方式修正

运行示例：
  # Linear Probe
  CUDA_VISIBLE_DEVICES=0,1 python step3_clip_finetune.py \
      --dataset TeacherBehavior --pseudo_strategy none --mode linear

  # LoRA
  CUDA_VISIBLE_DEVICES=0,1 python step3_clip_finetune.py \
      --dataset TeacherBehavior --pseudo_strategy none --mode lora

  # GT上界
  CUDA_VISIBLE_DEVICES=0,1 python step3_clip_finetune.py \
      --dataset TeacherBehavior --pseudo_strategy gt --mode lora
"""

from __future__ import annotations

import json
import logging
import os
import random
import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image
from tqdm import tqdm
import open_clip
from huggingface_hub import hf_hub_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent


def iter_progress(iterable, desc: str):
    """Only render tqdm bars on an interactive terminal.

    Background runs pipe stdout/stderr through tee, and high-frequency tqdm
    updates can flood the pipe and stall long LoRA jobs.
    """
    return tqdm(iterable, desc=desc, leave=False, disable=not sys.stderr.isatty())

# ── HF镜像（A100机器同样需要） ────────────────────────────────────────────────
HF_DEFAULT_ENDPOINT = "https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", HF_DEFAULT_ENDPOINT)
os.environ["HF_HUB_URL"] = os.environ.get("HF_HUB_URL", HF_DEFAULT_ENDPOINT)
os.environ["HUGGINGFACE_HUB_URL"] = os.environ.get("HUGGINGFACE_HUB_URL", HF_DEFAULT_ENDPOINT)
os.environ["HUGGINGFACE_CO_RESOLVE_ENDPOINT"] = os.environ.get(
    "HUGGINGFACE_CO_RESOLVE_ENDPOINT", HF_DEFAULT_ENDPOINT
)

# ── 路径配置 ──────────────────────────────────────────────────────────────────
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


DATASET_ROOT = resolve_dataset_root()


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


DEFAULT_ANALYSIS_DIR = resolve_default_analysis_dir()

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

CLIP_MODEL   = "ViT-L-14"
CLIP_PRETRAIN = "openai"
FEAT_DIM     = 768   # ViT-L/14 image embedding dim
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


# ══════════════════════════════════════════════════════════════════════════════
# 路径工具
# ══════════════════════════════════════════════════════════════════════════════

def resolve_split_dirs(dataset_name: str, split: str) -> tuple[Path, Path]:
    base = DATASET_CFG[dataset_name]["path"]
    img_dir = base / "images" / split
    lbl_dir = base / "labels" / split
    if img_dir.is_dir() and lbl_dir.is_dir():
        return img_dir, lbl_dir
    # 嵌套目录兜底
    imgs = sorted(p for p in base.glob(f"**/images/{split}") if p.is_dir())
    lbls = sorted(p for p in base.glob(f"**/labels/{split}") if p.is_dir())
    if imgs and lbls:
        return imgs[0], lbls[0]
    raise FileNotFoundError(
        f"Cannot find images/labels for {dataset_name}/{split} under {base}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class SCBCropDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        split: str,
        pseudo_jsonl: Optional[Path],
        transform,
        use_gt: bool = False,
        margin: float = 0.05,
        min_px: int = 32,
    ):
        self.cfg      = DATASET_CFG[dataset_name]
        self.img_dir, self.lbl_dir = resolve_split_dirs(dataset_name, split)
        self.transform = transform
        self.margin    = margin
        self.min_px    = min_px
        self.samples: list[tuple] = []

        if pseudo_jsonl is not None and not use_gt:
            self._load_from_jsonl(pseudo_jsonl)
        else:
            self._load_from_yolo()

        logger.info(
            "Dataset %s/%s: %d crops (%s)",
            dataset_name, split, len(self.samples),
            "GT" if use_gt else "pseudo",
        )

    def _load_from_jsonl(self, path: Path) -> None:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                img_path = self.img_dir / r["image"]
                if not img_path.exists():
                    continue
                label = r["pseudo_label"]
                if label is None or label >= self.cfg["num_classes"]:
                    continue
                self.samples.append(
                    (img_path, r["cx"], r["cy"], r["w"], r["h"], label)
                )

    def _load_from_yolo(self) -> None:
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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
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


# ══════════════════════════════════════════════════════════════════════════════
# LoRA — 修复版
# 注入范围：MLP (c_fc/c_proj)
# 参数收集：从LoRALayer实例直接收集，不依赖requires_grad扫描
# ══════════════════════════════════════════════════════════════════════════════

# ViT-L/14 transformer层中需要注入LoRA的子模块名关键词
# open_clip的ViT命名：resblocks.N.mlp.{c_fc/c_proj}
LORA_TARGET_KEYWORDS = [
    # 只注入MLP层，避免替换MultiheadAttention的out_proj（某些实现直接访问out_proj.weight/bias）
    ".mlp.c_fc",        # MLP第一层
    ".mlp.c_proj",      # MLP第二层
]


class LoRALayer(nn.Module):
    """
    将原始Linear层替换为 W + BA·scale 的形式。
    原始权重完全冻结，只有lora_A和lora_B参与梯度更新。
    """
    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original = original
        self.rank     = rank
        self.scale    = alpha / rank

        d_in  = original.in_features
        d_out = original.out_features

        # lora_A: 下投影，lora_B: 上投影
        # 初始化：A用高斯，B用零 → 训练开始时LoRA增量为0
        self.lora_A = nn.Parameter(torch.empty(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

        # 冻结原始权重
        for p in self.original.parameters():
            p.requires_grad_(False)

    @property
    def weight(self):
        return self.original.weight

    @property
    def bias(self):
        return self.original.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base  = self.original(x)
        delta = (x @ self.lora_A.T @ self.lora_B.T) * self.scale
        return base + delta


def inject_lora(visual: nn.Module, rank: int = 8, alpha: float = 16.0) -> list[LoRALayer]:
    """
    在visual encoder中注入LoRA层，返回所有注入的LoRALayer实例列表。
    使用列表而非依赖requires_grad，避免DataParallel包装后参数状态不一致。
    """
    injected: list[LoRALayer] = []

    # 收集需要替换的 (parent_module, attr_name, original_layer)
    targets = []
    for name, module in visual.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(kw in name for kw in LORA_TARGET_KEYWORDS):
            continue
        # 找到父模块
        parts = name.split(".")
        parent = visual
        for part in parts[:-1]:
            parent = getattr(parent, part)
        targets.append((parent, parts[-1], module))

    for parent, attr, original in targets:
        lora_layer = LoRALayer(original, rank=rank, alpha=alpha)
        setattr(parent, attr, lora_layer)
        injected.append(lora_layer)

    return injected


def collect_lora_params(lora_layers: list[LoRALayer]) -> list[nn.Parameter]:
    """从LoRALayer列表直接收集可训练参数，不依赖requires_grad扫描。"""
    params = []
    for layer in lora_layers:
        params.extend([layer.lora_A, layer.lora_B])
    return params


def serialize_lora_layers(lora_layers: list[LoRALayer]) -> dict[str, dict[str, torch.Tensor]]:
    return {
        f"layer_{i}": {
            "lora_A": layer.lora_A.detach().cpu(),
            "lora_B": layer.lora_B.detach().cpu(),
        }
        for i, layer in enumerate(lora_layers)
    }


def load_lora_layers(
    lora_layers: list[LoRALayer],
    state: dict[str, dict[str, torch.Tensor]],
) -> None:
    for i, layer in enumerate(lora_layers):
        layer_key = f"layer_{i}"
        if layer_key not in state:
            raise KeyError(f"Missing LoRA state for {layer_key}")
        layer_state = state[layer_key]
        layer.lora_A.data.copy_(layer_state["lora_A"].to(layer.lora_A.device))
        layer.lora_B.data.copy_(layer_state["lora_B"].to(layer.lora_B.device))


# ══════════════════════════════════════════════════════════════════════════════
# 训练 / 评估
# ══════════════════════════════════════════════════════════════════════════════

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
    total_loss = correct = total = 0

    for imgs, labels in iter_progress(loader, desc="train"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.set_grad_enabled(mode == "lora"):
            feats = visual(imgs)
        feats  = feats.float()
        logits = head(feats)
        loss   = criterion(logits, labels)
        loss.backward()

        if mode == "lora":
            # 梯度裁剪，防止LoRA层早期震荡
            nn.utils.clip_grad_norm_(
                [p for p in visual.parameters() if p.requires_grad]
                + list(head.parameters()),
                max_norm=1.0,
            )

        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(labels)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(
    visual: nn.Module,
    head: nn.Linear,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, list, list]:
    visual.eval()
    head.eval()
    correct = total = 0
    all_preds, all_labels = [], []

    for imgs, labels in iter_progress(loader, desc="eval"):
        imgs, labels = imgs.to(device), labels.to(device)
        feats  = visual(imgs).float()
        logits = head(feats)
        preds  = logits.argmax(1)
        correct    += (preds == labels).sum().item()
        total      += len(labels)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return 100.0 * correct / total, all_preds, all_labels


def build_feature_loader(
    visual: nn.Module,
    loader: DataLoader,
    device: torch.device,
 ) -> TensorDataset:
    features = []
    labels = []

    for imgs, batch_labels in iter_progress(loader, desc="cache"):
        imgs = imgs.to(device)
        batch_features = visual(imgs).float().cpu()
        features.append(batch_features)
        labels.append(batch_labels.clone())

    feature_tensor = torch.cat(features, dim=0)
    label_tensor = torch.cat(labels, dim=0)
    return TensorDataset(feature_tensor, label_tensor)


def train_head_epoch(
    head: nn.Linear,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    head.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = correct = total = 0

    for feats, labels in iter_progress(loader, desc="train_head"):
        feats = feats.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        logits = head(feats)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate_head(
    head: nn.Linear,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, list, list]:
    head.eval()
    correct = total = 0
    all_preds, all_labels = [], []

    for feats, labels in iter_progress(loader, desc="eval_head"):
        feats = feats.to(device)
        labels = labels.to(device)
        logits = head(feats)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += len(labels)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return 100.0 * correct / total, all_preds, all_labels


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def run(
    dataset_name: str,
    pseudo_strategy: str,
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

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu   = torch.cuda.device_count()
    logger.info("Device: %s | GPUs: %d", device, n_gpu)

    # ── 加载CLIP ──────────────────────────────────────────────────────────────
    model, _, preprocess = create_clip_model_and_transforms()

    lora_layers: list[LoRALayer] = []

    if mode == "lora":
        # 注入LoRA到visual encoder
        lora_layers = inject_lora(model.visual, rank=lora_rank, alpha=lora_alpha)
        n_injected  = len(lora_layers)
        if n_injected == 0:
            raise RuntimeError(
                "LoRA injection failed: no compatible layers found. "
                "Check LORA_TARGET_KEYWORDS against model architecture."
            )
        n_params = sum(p.numel() for ll in lora_layers for p in [ll.lora_A, ll.lora_B])
        logger.info(
            "LoRA injected into %d layers | rank=%d alpha=%.1f | trainable params: %d",
            n_injected, lora_rank, lora_alpha, n_params,
        )
        # 冻结CLIP其余参数（LoRALayer内部已冻结original，这里确保其他层也冻结）
        for name, p in model.named_parameters():
            if not any(
                isinstance(m, LoRALayer) and (p is m.lora_A or p is m.lora_B)
                for m in model.modules()
            ):
                p.requires_grad_(False)
    else:
        # Linear probe：冻结全部CLIP参数
        for p in model.parameters():
            p.requires_grad_(False)

    model = model.to(device)

    # visual encoder（多卡用DataParallel）
    visual: nn.Module
    if n_gpu > 1:
        visual = nn.DataParallel(model.visual)
    else:
        visual = model.visual
    visual = visual.to(device)

    # 分类头
    num_classes = DATASET_CFG[dataset_name]["num_classes"]
    head = nn.Linear(FEAT_DIM, num_classes).to(device)

    # 优化器：LoRA参数从lora_layers直接收集，不依赖requires_grad扫描
    if mode == "lora":
        trainable_params = collect_lora_params(lora_layers) + list(head.parameters())
        # LoRA通常需要比linear probe更小的学习率
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    logger.info(
        "Optimizer params: %d | lr=%.2e",
        sum(p.numel() for p in optimizer.param_groups[0]["params"]),
        lr,
    )

    # ── 数据集 ───────────────────────────────────────────────────────────────
    pseudo_jsonl: Optional[Path] = None
    if pseudo_strategy != "gt":
        pseudo_jsonl = analysis_dir / f"{dataset_name}_train_pseudo_{pseudo_strategy}.jsonl"
        if not pseudo_jsonl.exists():
            raise FileNotFoundError(f"伪标签文件不存在: {pseudo_jsonl}")

    train_ds = SCBCropDataset(
        dataset_name, "train", pseudo_jsonl, preprocess,
        use_gt=(pseudo_strategy == "gt"),
    )
    val_ds = SCBCropDataset(
        dataset_name, "val", None, preprocess, use_gt=True,
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    if num_workers > 0:
        loader_kwargs["multiprocessing_context"] = "spawn"

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    feature_train_loader = None
    feature_val_loader = None
    if mode == "linear":
        logger.info(
            "Caching frozen CLIP features once for linear probe | train_batch_size=%d",
            batch_size,
        )
        feature_train_ds = build_feature_loader(
            visual=visual,
            loader=DataLoader(train_ds, shuffle=False, **loader_kwargs),
            device=device,
        )
        feature_val_ds = build_feature_loader(
            visual=visual,
            loader=DataLoader(val_ds, shuffle=False, **loader_kwargs),
            device=device,
        )
        eval_batch_size = min(max(batch_size * 16, 4096), len(feature_val_ds))
        feature_train_loader = DataLoader(
            feature_train_ds,
            batch_size=batch_size,
            shuffle=True,
        )
        feature_val_loader = DataLoader(
            feature_val_ds,
            batch_size=eval_batch_size,
            shuffle=False,
        )

    # ── 训练循环 ─────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / f"{dataset_name}_{mode}_{pseudo_strategy}_best.pt"
    resume_path = out_dir / f"{dataset_name}_{mode}_{pseudo_strategy}_resume.pt"
    result_path = out_dir / f"{dataset_name}_{mode}_{pseudo_strategy}_result.json"
    best_acc   = 0.0
    best_epoch = 0
    history    = []
    start_epoch = 1

    if resume_path.exists() and not result_path.exists():
        logger.info("Resume checkpoint found: %s", resume_path)
        resume_ckpt = torch.load(resume_path, map_location=device)
        head.load_state_dict(resume_ckpt["head"])
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        scheduler.load_state_dict(resume_ckpt["scheduler"])
        best_acc = float(resume_ckpt.get("best_acc", 0.0))
        best_epoch = int(resume_ckpt.get("best_epoch", 0))
        history = list(resume_ckpt.get("history", []))
        start_epoch = int(resume_ckpt.get("completed_epochs", 0)) + 1
        if mode == "lora":
            load_lora_layers(lora_layers, resume_ckpt["lora"])
        logger.info(
            "Resuming %s/%s/%s seed=%d from epoch %d/%d",
            dataset_name,
            pseudo_strategy,
            mode,
            seed,
            start_epoch,
            epochs,
        )

    if start_epoch > epochs:
        logger.info(
            "Resume checkpoint already covers all %d epochs; writing final result only.",
            epochs,
        )

    for epoch in range(start_epoch, epochs + 1):
        if mode == "linear":
            train_loss, train_acc = train_head_epoch(
                head=head,
                loader=feature_train_loader,
                optimizer=optimizer,
                device=device,
            )
            val_acc, preds, labels_list = evaluate_head(
                head=head,
                loader=feature_val_loader,
                device=device,
            )
        else:
            train_loss, train_acc = train_epoch(
                visual, head, train_loader, optimizer, device, mode
            )
            val_acc, preds, labels_list = evaluate(visual, head, val_loader, device)
        scheduler.step()

        logger.info(
            "Epoch %3d/%d | loss=%.4f train_acc=%.1f%% val_acc=%.1f%%",
            epoch, epochs, train_loss, train_acc, val_acc,
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc":  train_acc,
            "val_acc":    val_acc,
        })

        if val_acc > best_acc:
            best_acc   = val_acc
            best_epoch = epoch
            ckpt = {
                "head":  head.state_dict(),
                "epoch": epoch,
            }
            if mode == "lora":
                # 只保存LoRA增量，不保存完整CLIP权重（节省磁盘）
                ckpt["lora"] = serialize_lora_layers(lora_layers)
            torch.save(ckpt, best_path)

        resume_ckpt = {
            "dataset": dataset_name,
            "pseudo_strategy": pseudo_strategy,
            "mode": mode,
            "seed": seed,
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "completed_epochs": epoch,
            "best_acc": best_acc,
            "best_epoch": best_epoch,
            "history": history,
        }
        if mode == "lora":
            resume_ckpt["lora"] = serialize_lora_layers(lora_layers)
        torch.save(resume_ckpt, resume_path)

    logger.info("Best val_acc: %.4f%% @ epoch %d", best_acc, best_epoch)

    # ── 保存结果JSON ─────────────────────────────────────────────────────────
    result = {
        "dataset":        dataset_name,
        "mode":           mode,
        "pseudo_strategy": pseudo_strategy,
        "seed":           seed,
        "best_val_acc":   best_acc,
        "best_epoch":     best_epoch,
        "epochs":         epochs,
        "lora_rank":      lora_rank if mode == "lora" else None,
        "lora_alpha":     lora_alpha if mode == "lora" else None,
        "lr":             lr,
        "history":        history,
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Result saved: %s", result_path)
    if resume_path.exists():
        resume_path.unlink()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune CLIP on SCB pseudo-labels (Linear Probe or LoRA)"
    )
    parser.add_argument("--dataset",
                        choices=list(DATASET_CFG), required=True)
    parser.add_argument("--pseudo_strategy",
                        choices=["none", "agreement", "gt"], default="none")
    parser.add_argument("--mode",
                        choices=["linear", "lora"], default="linear")
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch_size",  type=int,   default=128)
    parser.add_argument("--lr",          type=float, default=1e-4,
                        help="Learning rate. LoRA默认建议1e-4，linear可用5e-4")
    parser.add_argument("--lora_rank",   type=int,   default=8)
    parser.add_argument("--lora_alpha",  type=float, default=16.0)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--out_dir",     default=None,
                        help='Output directory for finetune results (defaults to results/phase3_finetune/manual)')
    parser.add_argument("--analysis_dir", default=None,
                        help='Analysis results directory (defaults to ANALYSIS_DIR or the newest results/phase2_filtering/* directory)')
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else SCRIPT_DIR / "results" / "phase3_finetune" / "manual"
    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else DEFAULT_ANALYSIS_DIR

    run(
        dataset_name     = args.dataset,
        pseudo_strategy  = args.pseudo_strategy,
        mode             = args.mode,
        epochs           = args.epochs,
        batch_size       = args.batch_size,
        lr               = args.lr,
        lora_rank        = args.lora_rank,
        lora_alpha       = args.lora_alpha,
        seed             = args.seed,
        out_dir          = out_dir,
        analysis_dir     = analysis_dir,
        num_workers      = args.num_workers,
    )
