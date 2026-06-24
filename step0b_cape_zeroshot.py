"""
Step 0b: CAPE零样本评估
用Set A / Set B / Set C的ensemble提示词在bbox裁剪图像上评估CLIP ViT-L/14
结果可直接与18组微调实验对比（相同评估协议：bbox crop + val集）

Usage:
  CUDA_VISIBLE_DEVICES=0 python step0b_cape_zeroshot.py --all
  CUDA_VISIBLE_DEVICES=0 python step0b_cape_zeroshot.py --dataset BowTurnHead
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import open_clip
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

# ── 路径 ──────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "datasets_scb"

DATASET_CFG = {
    "TeacherBehavior": {
        "path": DATASET_ROOT / "SCB5_TeacherBehavior",
        "classes": [
            "guide", "answer", "On-stage interaction", "blackboard-writing",
            "teacher", "stand", "screen", "blackBoard",
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

# ── CAPE提示词（Set A / B / C） ───────────────────────────────────────────────
CAPE_A = {
    "guide": [
        "a teacher guiding a student one-on-one",
        "a teacher helping a student at their desk",
        "a teacher walking among students and offering guidance",
    ],
    "answer": [
        "a teacher answering a student's question",
        "a teacher responding to a raised hand in class",
        "a student asking a question and the teacher replying",
    ],
    "On-stage interaction": [
        "a teacher interacting with students at the front of the classroom",
        "a teacher engaging with students on the podium",
        "a teacher and students having a discussion in front of the class",
    ],
    "blackboard-writing": [
        "a teacher writing on a blackboard with chalk",
        "a hand writing equations on a chalkboard",
        "a teacher's back while writing on the blackboard",
    ],
    "teacher": [
        "a teacher standing and explaining a concept",
        "a teacher giving a lecture at the podium",
        "a teacher talking to the class while standing",
    ],
    "stand": [
        "a person standing still in a classroom",
        "a teacher standing at the front without interacting",
        "a teacher standing idle near the podium",
    ],
    "screen": [
        "a teacher pointing at a projection screen",
        "a teacher presenting slides on a screen",
        "a screen displaying a presentation in a classroom",
    ],
    "blackBoard": [
        "a teacher pointing at the blackboard",
        "a teacher referring to content on the blackboard",
        "a blackboard with writing visible in a classroom",
    ],
    "hand-raise": [
        "a student raising their hand in a classroom",
        "a student with arm raised to ask a question",
        "a student raising hand to participate in class",
    ],
    "read": [
        "a student reading a textbook at their desk",
        "a student looking down at a book while reading",
        "a student engaged in reading study materials",
    ],
    "write": [
        "a student writing in a notebook",
        "a student taking notes with a pen",
        "a student writing at their desk in class",
    ],
    "bow-head": [
        "a student with their head bowed down",
        "a student looking down at their phone or desk",
        "a student with lowered head not paying attention",
    ],
    "turn-head": [
        "a student turning their head to look sideways",
        "a student looking away from the teacher",
        "a student turning around in their seat",
    ],
}

CAPE_B = {
    "guide": [
        "a teacher leaning over a student's desk to help",
        "individual tutoring in a classroom setting",
        "a teacher assisting a single student with their work",
    ],
    "answer": [
        "a teacher explaining something to a student who asked a question",
        "a dialogue between teacher and student in a classroom",
        "a teacher addressing a student's raised hand",
    ],
    "On-stage interaction": [
        "a lively discussion between a teacher and students at the podium",
        "a teacher calling on students from the front of the room",
        "interactive teaching at the front of a classroom",
    ],
    "blackboard-writing": [
        "chalk writing on a green or black chalkboard",
        "a teacher facing the blackboard writing notes",
        "handwritten text being written on a classroom board",
    ],
    "teacher": [
        "a teacher delivering a lecture to students",
        "a teacher speaking in front of the classroom",
        "a professor explaining a topic while standing",
    ],
    "stand": [
        "a teacher standing motionless in a classroom",
        "a person standing at the front of a classroom doing nothing",
        "an idle teacher standing near a desk",
    ],
    "screen": [
        "a projection screen showing slides in a classroom",
        "a teacher using a projector for a presentation",
        "a digital display showing educational content",
    ],
    "blackBoard": [
        "a chalkboard with written content visible",
        "a teacher gesturing toward a blackboard",
        "educational content displayed on a classroom blackboard",
    ],
    "hand-raise": [
        "a child with their arm stretched high in a classroom",
        "a student eagerly putting up their hand to answer",
        "students with hands up during class discussion",
    ],
    "read": [
        "a student silently reading at their desk",
        "a student's eyes focused on a textbook page",
        "students reading books in a quiet classroom",
    ],
    "write": [
        "a student's hand holding a pen writing on paper",
        "a student copying notes from the board",
        "a student doing written exercises in class",
    ],
    "bow-head": [
        "a student slouching with head down on desk",
        "a student looking at something under their desk",
        "a student with drooped head during class",
    ],
    "turn-head": [
        "a student whose face is turned to the side",
        "a student gazing sideways instead of forward",
        "a student looking at another student during class",
    ],
}

CAPE_C = {
    "guide": [
        "a person providing guidance in an indoor setting",
        "someone showing directions or instructions to another person",
        "a mentor giving advice to a learner",
    ],
    "answer": [
        "a person giving an answer or reply",
        "someone responding verbally to a question",
        "a conversation where one person explains something",
    ],
    "On-stage interaction": [
        "people interacting on a stage or platform",
        "a speaker engaging with an audience from a raised platform",
        "a person standing on stage communicating with others",
    ],
    "blackboard-writing": [
        "someone writing text on a large dark board",
        "handwriting being produced on a wall-mounted board",
        "a person using chalk to write on a board surface",
    ],
    "teacher": [
        "a person in the role of a teacher or instructor",
        "an educator standing in front of learners",
        "a professional conducting a lesson",
    ],
    "stand": [
        "a person standing upright in a room",
        "someone in a standing posture without movement",
        "a figure standing still indoors",
    ],
    "screen": [
        "an electronic display or projection screen",
        "a monitor or screen showing digital content",
        "a flat display surface mounted in a room",
    ],
    "blackBoard": [
        "a dark-colored board mounted on a wall",
        "a chalkboard visible in a room",
        "a traditional writing board with content on it",
    ],
    "hand-raise": [
        "a person raising one hand above their head",
        "someone with an arm lifted upward",
        "a person with their hand raised in the air",
    ],
    "read": [
        "a person reading printed material",
        "someone looking down at a book or document",
        "a person focused on reading text",
    ],
    "write": [
        "a person writing with a pen or pencil",
        "someone moving a writing instrument on paper",
        "a person engaged in handwriting",
    ],
    "bow-head": [
        "a person with their head tilted downward",
        "someone bowing their head forward",
        "a person looking down with lowered head",
    ],
    "turn-head": [
        "a person with their head turned to one side",
        "someone looking sideways",
        "a person whose head faces a different direction than their body",
    ],
}

ALL_SETS = {"A": CAPE_A, "B": CAPE_B, "C": CAPE_C}


# ── 数据集工具 ────────────────────────────────────────────────────────────────

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
    raise FileNotFoundError(f"Cannot find {dataset_name}/{split}")


class BBoxCropDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, split, transform,
                 margin=0.05, min_px=32):
        self.transform = transform
        img_dir, lbl_dir = resolve_split_dirs(dataset_name, split)
        self.samples = []
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
                self.samples.append((img_path, cx, cy, w, h, gt,
                                     margin, min_px))
        logger.info("%s/%s: %d bbox crops", dataset_name, split,
                    len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cx, cy, w, h, label, margin, min_px = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            return torch.zeros(3, 224, 224), label
        W, H = img.size
        x1 = max(0, (cx - w / 2 - margin) * W)
        y1 = max(0, (cy - h / 2 - margin) * H)
        x2 = min(W, (cx + w / 2 + margin) * W)
        y2 = min(H, (cy + h / 2 + margin) * H)
        if (x2 - x1) < min_px or (y2 - y1) < min_px:
            return torch.zeros(3, 224, 224), label
        return self.transform(img.crop((x1, y1, x2, y2))), label


# ── CAPE文本编码器 ────────────────────────────────────────────────────────────

@torch.no_grad()
def build_cape_classifier(model, classes, prompt_sets, device):
    """
    对每个类别，收集所有prompt_set里的提示词，
    编码后取平均作为该类别的文本嵌入（CAPE ensemble）。
    返回归一化后的分类矩阵 (feat_dim, num_classes)。
    """
    base = model.module if hasattr(model, "module") else model
    class_embeddings = []

    for cls in classes:
        prompts = []
        for pset in prompt_sets.values():
            if cls in pset:
                prompts.extend(pset[cls])
        if not prompts:
            # 兜底：用简单模板
            prompts = [f"a photo of a {cls}"]
            logger.warning("No CAPE prompts found for class '%s', using default.", cls)

        tokens = open_clip.tokenize(prompts).to(device)
        feats = base.encode_text(tokens)          # (n_prompts, dim)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        cls_emb = feats.mean(dim=0)               # (dim,)
        cls_emb = cls_emb / cls_emb.norm()
        class_embeddings.append(cls_emb)

    # (dim, num_classes)
    return torch.stack(class_embeddings, dim=1)


# ── 评估 ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_cape(dataset_name, model, classifier, preprocess, device,
                  batch_size=256, num_workers=4):
    val_ds = BBoxCropDataset(dataset_name, "val", preprocess)
    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    base = model.module if hasattr(model, "module") else model
    classes = DATASET_CFG[dataset_name]["classes"]

    correct = total = 0
    per_cls_correct = [0] * len(classes)
    per_cls_total   = [0] * len(classes)

    for imgs, labels in tqdm(loader, desc=dataset_name, leave=False):
        imgs = imgs.to(device)
        feats = base.encode_image(imgs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        logits = feats.float() @ classifier.float()   # (B, num_classes)
        preds = logits.argmax(1).cpu()

        correct += (preds == labels).sum().item()
        total   += len(labels)
        for p, l in zip(preds.tolist(), labels.tolist()):
            per_cls_total[l]   += 1
            per_cls_correct[l] += int(p == l)

    acc = 100.0 * correct / total if total else 0.0

    logger.info("=" * 60)
    logger.info("CAPE Zero-Shot | %s | Overall: %.2f%%", dataset_name, acc)
    logger.info("%-30s %8s %8s %10s", "Class", "n", "correct", "acc(%)")
    logger.info("-" * 58)
    per_cls_acc = {}
    for i, cls in enumerate(classes):
        n = per_cls_total[i]
        c = per_cls_correct[i]
        a = 100.0 * c / n if n else 0.0
        per_cls_acc[cls] = round(a, 2)
        logger.info("%-30s %8d %8d %9.2f%%", cls, n, c, a)
    logger.info("=" * 60)

    return {
        "dataset":        dataset_name,
        "model":          "CLIP_ViT-L-14_openai",
        "method":         "CAPE_zeroshot_ABC",
        "overall_acc":    round(acc, 4),
        "per_class_acc":  per_cls_acc,
        "total_samples":  total,
    }


# ── 主程序 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASET_CFG))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--prompt_sets", nargs="+",
                        choices=["A", "B", "C"], default=["A", "B", "C"],
                        help="Which prompt sets to ensemble (default: all three)")
    parser.add_argument("--batch_size",  type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_dir", default=None,
                        help='Output directory for CAPE zero-shot results (defaults to Annotation/results/phase0_zero_shot/cape_aux in repo root)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu  = torch.cuda.device_count()
    logger.info("Device: %s | GPUs: %d", device, n_gpu)

    # 加载CLIP
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    model = model.to(device)
    if n_gpu > 1:
        model = nn.DataParallel(model)
    model.eval()

    datasets = list(DATASET_CFG) if (args.all or not args.dataset) \
               else [args.dataset]
    prompt_sets = {k: ALL_SETS[k] for k in args.prompt_sets}

    all_results = []
    for ds in datasets:
        classes    = DATASET_CFG[ds]["classes"]
        classifier = build_cape_classifier(model, classes, prompt_sets, device)
        result     = evaluate_cape(ds, model, classifier, preprocess, device,
                                   args.batch_size, args.num_workers)
        all_results.append(result)

    # Resolve default location relative to repo root
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir) if args.out_dir else repo_root / "Annotation" / "results" / "phase0_zero_shot" / "cape_aux"
    out_dir.mkdir(parents=True, exist_ok=True)
    sets_tag = "".join(sorted(args.prompt_sets))
    out_file = out_dir / f"phase0_cape_zero_shot_set{sets_tag}_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info("Saved → %s", out_file)

    # 汇总打印
    print("\n=== CAPE Zero-Shot Summary ===")
    print(f"{'Dataset':<25} {'Overall Acc (%)':>16}")
    print("-" * 43)
    for r in all_results:
        print(f"{r['dataset']:<25} {r['overall_acc']:>16.2f}")


if __name__ == "__main__":
    main()

# ══════════════════════════════════════════════════════════════════════════════
# 运行：
#   CUDA_VISIBLE_DEVICES=0 python step0b_cape_zeroshot.py --all
#
# 只用Set A+B（排除blind prompts）：
#   CUDA_VISIBLE_DEVICES=0 python step0b_cape_zeroshot.py --all --prompt_sets A B
#
# 单独测试一个数据集：
#   CUDA_VISIBLE_DEVICES=0 python step0b_cape_zeroshot.py --dataset BowTurnHead
# ══════════════════════════════════════════════════════════════════════════════