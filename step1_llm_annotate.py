"""
Step 1: Local HF MLLM Auto-Annotation for SCB Dataset.
用本地 Hugging Face 多模态模型对裁剪后的 bbox 图像进行分类标注。

默认模型：
    - qwen: Qwen/Qwen2-VL-7B-Instruct
    - llava: llava-hf/llava-1.5-7b-hf

用法：
  HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=0,1 \
  python step1_llm_annotate.py --dataset TeacherBehavior --split val
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor

try:
    from transformers import AutoModelForImageTextToText as AutoVLM
except ImportError:
    from transformers import AutoModelForVision2Seq as AutoVLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Ensure a default HF endpoint is set so scripts work without explicit env var
HF_DEFAULT_ENDPOINT = "https://hf-mirror.com"
# Set multiple hub-related env vars to ensure huggingface_hub and transformers use the mirror
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", HF_DEFAULT_ENDPOINT)
os.environ["HUGGINGFACE_HUB_URL"] = os.environ.get("HUGGINGFACE_HUB_URL", HF_DEFAULT_ENDPOINT)
os.environ["HF_HUB_URL"] = os.environ.get("HF_HUB_URL", HF_DEFAULT_ENDPOINT)
logger.info("HF_ENDPOINT=%s", os.environ.get("HF_ENDPOINT"))
logger.info("HUGGINGFACE_HUB_URL=%s", os.environ.get("HUGGINGFACE_HUB_URL"))

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_dataset_root() -> Path:
    candidates = []
    env_root = os.environ.get("SCB_DATASET_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.append(REPO_ROOT / "datasets_scb")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Cannot find datasets_scb. Set SCB_DATASET_ROOT or place datasets_scb at the repository root. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


DATASET_ROOT = resolve_dataset_root()

DATASETS = {
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
    },
    "HandriseReadWrite": {
        "path": DATASET_ROOT / "SCB5_HandriseReadWrite",
        "classes": ["hand-raise", "read", "write"],
    },
    "BowTurnHead": {
        "path": DATASET_ROOT / "SCB_BowTurnHead",
        "classes": ["bow-head", "turn-head"],
    },
}


def resolve_split_dirs(dataset_name: str, split: str) -> Tuple[Path, Path]:
    cfg = DATASETS[dataset_name]
    base_path = cfg["path"]
    direct_img_dir = base_path / "images" / split
    direct_lbl_dir = base_path / "labels" / split
    if direct_img_dir.is_dir() and direct_lbl_dir.is_dir():
        return direct_img_dir, direct_lbl_dir

    nested_img_dirs = sorted(
        path for path in base_path.glob(f"**/images/{split}") if path.is_dir()
    )
    nested_lbl_dirs = sorted(
        path for path in base_path.glob(f"**/labels/{split}") if path.is_dir()
    )
    if nested_img_dirs and nested_lbl_dirs:
        return nested_img_dirs[0], nested_lbl_dirs[0]

    zip_candidates = sorted(base_path.glob("*.zip"))
    zip_hint = ""
    if zip_candidates:
        zip_hint = (
            " Found archive(s) but no extracted dataset layout: "
            + ", ".join(str(path) for path in zip_candidates)
        )

    raise FileNotFoundError(
        f"Dataset {dataset_name} split {split} does not have readable images/labels directories under {base_path}."
        f" Expected {direct_img_dir} and {direct_lbl_dir}.{zip_hint}"
    )

MODEL_CONFIGS = {
    "qwen": {
        "repo_id": "Qwen/Qwen2-VL-7B-Instruct",
        "device": "cuda:0",
    },
    "llava": {
        "repo_id": "llava-hf/llava-1.5-7b-hf",
        "device": "cuda:1",
    },
}

CROP_MARGIN = 0.05
MIN_CROP_PX = 32


def build_prompt(classes: List[str]) -> str:
    class_list = "\n".join(f"  {index}: {name}" for index, name in enumerate(classes))
    return (
        "You are a classroom behavior recognition expert. "
        "Given one cropped classroom image, choose the single best matching class. "
        "Reply with only the class index number. No explanation.\n\n"
        f"Classes:\n{class_list}\n\n"
        "Answer with index only:"
    )


def crop_bbox(
    img: Image.Image,
    cx: float,
    cy: float,
    w: float,
    h: float,
    margin: float = CROP_MARGIN,
) -> Optional[Image.Image]:
    width, height = img.size
    x1 = max(0, (cx - w / 2 - margin) * width)
    y1 = max(0, (cy - h / 2 - margin) * height)
    x2 = min(width, (cx + w / 2 + margin) * width)
    y2 = min(height, (cy + h / 2 + margin) * height)
    if (x2 - x1) < MIN_CROP_PX or (y2 - y1) < MIN_CROP_PX:
        return None
    return img.crop((x1, y1, x2, y2))


class HFVisionAnnotator:
    def __init__(
        self,
        key: str,
        repo_id: str,
        device: str,
        max_new_tokens: int = 8,
        trust_remote_code: bool = True,
    ):
        self.key = key
        self.repo_id = repo_id
        self.max_new_tokens = max_new_tokens
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype()

        logger.info("Loading %s model from %s on %s", key, repo_id, self.device)
        self.processor = AutoProcessor.from_pretrained(
            repo_id,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoVLM.from_pretrained(
            repo_id,
            trust_remote_code=trust_remote_code,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self._configure_processor()

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device.startswith("cuda") and torch.cuda.is_available():
            index = int(device.split(":", 1)[1]) if ":" in device else 0
            if index < torch.cuda.device_count():
                return torch.device(device)
        return torch.device("cpu")

    def _resolve_dtype(self) -> torch.dtype:
        if self.device.type != "cuda":
            return torch.float32
        if self.key == "llava":
            return torch.float16
        return torch.bfloat16

    def _configure_processor(self) -> None:
        if self.key != "llava":
            return

        patch_size = getattr(self.model.config.vision_config, "patch_size", None)
        strategy = getattr(self.model.config, "vision_feature_select_strategy", None)
        for target in (self.processor, getattr(self.processor, "image_processor", None)):
            if target is None:
                continue
            if patch_size is not None:
                setattr(target, "patch_size", patch_size)
            if strategy is not None:
                setattr(target, "vision_feature_select_strategy", strategy)

    def _build_messages(self, prompt: str) -> List[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def _build_inputs(self, prompt: str, image: Image.Image) -> Dict[str, torch.Tensor]:
        messages = self._build_messages(prompt)
        if hasattr(self.processor, "apply_chat_template"):
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = prompt

        return self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )

    def predict(self, prompt: str, image: Image.Image) -> Tuple[Optional[int], float, str]:
        start = time.time()
        try:
            with torch.inference_mode():
                inputs = self._build_inputs(prompt, image)
                inputs = {
                    key: value.to(self.device) if hasattr(value, "to") else value
                    for key, value in inputs.items()
                }
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
                if "input_ids" in inputs:
                    generated_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
                else:
                    generated_ids = output_ids
                text = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()
            match = re.search(r"\d+", text)
            pred = int(match.group()) if match else None
            return pred, time.time() - start, text
        except Exception as exc:
            logger.warning("%s inference failed: %s", self.key, exc)
            return None, time.time() - start, ""


def load_models(args: argparse.Namespace) -> Dict[str, HFVisionAnnotator]:
    model_settings = {
        "qwen": {"repo_id": args.qwen_repo, "device": args.qwen_device},
        "llava": {"repo_id": args.llava_repo, "device": args.llava_device},
    }
    models: Dict[str, HFVisionAnnotator] = {}
    for key, config in model_settings.items():
        models[key] = HFVisionAnnotator(
            key=key,
            repo_id=config["repo_id"],
            device=config["device"],
            max_new_tokens=args.max_new_tokens,
            trust_remote_code=not args.disable_trust_remote_code,
        )
    return models


def annotate_image(
    img_path: Path,
    label_path: Path,
    classes: List[str],
    prompt: str,
    models: Dict[str, HFVisionAnnotator],
) -> List[dict]:
    if not label_path.exists():
        return []

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as exc:
        logger.warning("Cannot open %s: %s", img_path, exc)
        return []

    results: List[dict] = []
    for line_idx, line in enumerate(label_path.read_text().strip().splitlines()):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        gt_cls = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])

        crop = crop_bbox(img, cx, cy, w, h)
        if crop is None:
            continue

        record = {
            "image": img_path.name,
            "bbox_idx": line_idx,
            "gt": gt_cls,
            "gt_name": classes[gt_cls] if gt_cls < len(classes) else "unknown",
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
        }

        with ThreadPoolExecutor(max_workers=len(models)) as pool:
            futures = {
                pool.submit(model.predict, prompt, crop.copy()): key
                for key, model in models.items()
            }
            for future in as_completed(futures):
                model_key = futures[future]
                pred, elapsed, raw_text = future.result()
                if pred is not None and pred >= len(classes):
                    pred = None
                record[f"pred_{model_key}"] = pred
                record[f"pred_{model_key}_name"] = (
                    classes[pred] if pred is not None else None
                )
                record[f"time_{model_key}"] = round(elapsed, 3)
                record[f"raw_{model_key}"] = raw_text

        record["agreement"] = (
            record.get("pred_qwen") is not None
            and record.get("pred_llava") is not None
            and record["pred_qwen"] == record["pred_llava"]
        )
        results.append(record)

    return results


def run(
    dataset_name: str,
    split: str,
    out_dir: Path,
    models: Dict[str, HFVisionAnnotator],
    max_images: Optional[int] = None,
    overwrite: bool = False,
):
    cfg = DATASETS[dataset_name]
    img_dir, lbl_dir = resolve_split_dirs(dataset_name, split)
    classes = cfg["classes"]
    prompt = build_prompt(classes)

    img_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if max_images is not None:
        img_paths = img_paths[:max_images]

    logger.info(
        "Dataset: %s | Split: %s | Images: %d | img_dir=%s | lbl_dir=%s",
        dataset_name,
        split,
        len(img_paths),
        img_dir,
        lbl_dir,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{dataset_name}_{split}_annotations.jsonl"

    done_images = set()
    if out_file.exists():
        if overwrite:
            logger.info("Overwrite enabled - removing existing output: %s", out_file)
            out_file.unlink()
        else:
            required_keys = {"pred_qwen", "pred_llava"}
            with open(out_file) as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    missing = required_keys.difference(record)
                    if missing:
                        raise ValueError(
                            f"Existing output {out_file} is incompatible with current schema; missing keys: {sorted(missing)}. "
                            "Use --overwrite or a different --out_dir to regenerate annotations."
                        )
                    break

    if out_file.exists():
        with open(out_file) as handle:
            for line in handle:
                record = json.loads(line)
                done_images.add(record["image"])
        logger.info("Resuming - %d images already done", len(done_images))

    with open(out_file, "a") as handle:
        for index, img_path in enumerate(img_paths, start=1):
            if img_path.name in done_images:
                continue
            label_path = lbl_dir / f"{img_path.stem}.txt"
            records = annotate_image(img_path, label_path, classes, prompt, models)
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 10 == 0:
                logger.info("Progress: %d/%d", index, len(img_paths))

    logger.info("Done. Output: %s", out_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--out_dir", default=None,
                        help='Output directory for annotations (defaults to Annotation/llm_annotations in repo root)')
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--qwen_repo", default=MODEL_CONFIGS["qwen"]["repo_id"])
    parser.add_argument("--llava_repo", default=MODEL_CONFIGS["llava"]["repo_id"])
    parser.add_argument("--qwen_device", default=MODEL_CONFIGS["qwen"]["device"])
    parser.add_argument("--llava_device", default=MODEL_CONFIGS["llava"]["device"])
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--disable_trust_remote_code", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info("HF_ENDPOINT=%s", os.environ.get("HF_ENDPOINT", "<default>"))
    resolve_split_dirs(args.dataset, args.split)
    models = load_models(args)

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir) if args.out_dir else repo_root / "Annotation" / "llm_annotations"

    run(args.dataset, args.split, out_dir, models, args.max_images, args.overwrite)
