"""
Cross-model annotation validation.
This branch extends the main annotation protocol to additional Hugging Face
multimodal models for cross-model validation experiments.

用法示例：
                try:
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                except TypeError:
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True)
    CUDA_VISIBLE_DEVICES=0,1 python cross_model_annotate.py \
      --dataset TeacherBehavior --split val \
      --models qwen25

  # 只跑 Qwen2.5-VL-32B（HF后端，多卡自动切分）
    python cross_model_annotate.py \
      --dataset TeacherBehavior --split val \
      --models qwen2532

  # 只跑 Gemma-3-27B-IT（HF后端，多卡自动切分）
    python cross_model_annotate.py \
      --dataset TeacherBehavior --split val \
      --models gemma327

  # 同时跑多个模型
    CUDA_VISIBLE_DEVICES=0 python cross_model_annotate.py \
      --dataset TeacherBehavior --split val \
      --models qwen25 qwen2532 gemma327

  # 全量标注（train + val，所有三个数据集）
  for ds in BowTurnHead HandriseReadWrite TeacherBehavior; do
    for split in val train; do
                        python cross_model_annotate.py --dataset $ds --split $split --models qwen25 qwen2532 gemma327
    done
  done

输出：
  每个模型独立输出一个 JSONL 文件：
    {out_dir}/{dataset}_{split}_{model_key}_annotations.jsonl
  每行包含 gt, pred_{model_key}, pred_{model_key}_name, time_{model_key}, raw_{model_key}

  最终也输出一个合并文件（如果多个模型同时跑）：
    {out_dir}/{dataset}_{split}_crossmodel_annotations.jsonl
  每行包含所有模型的预测，以及逐对 agreement 字段。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HF_DEFAULT_ENDPOINT = "https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", HF_DEFAULT_ENDPOINT)
os.environ["HUGGINGFACE_HUB_URL"] = os.environ.get("HUGGINGFACE_HUB_URL", HF_DEFAULT_ENDPOINT)
os.environ["HF_HUB_URL"] = os.environ.get("HF_HUB_URL", HF_DEFAULT_ENDPOINT)
os.environ["HUGGINGFACE_CO_RESOLVE_ENDPOINT"] = os.environ.get(
    "HUGGINGFACE_CO_RESOLVE_ENDPOINT", HF_DEFAULT_ENDPOINT
)

"""
https://huggingface.co/google/gemma-4-26B-A4B-it
https://huggingface.co/google/gemma-4-31B-it

https://huggingface.co/Qwen/Qwen3.5-27B
https://huggingface.co/Qwen/Qwen3.5-35B-A3B

https://huggingface.co/Qwen/Qwen3.6-27B
https://huggingface.co/Qwen/Qwen3.6-35B-A3B
"""

import torch
from PIL import Image
from transformers import AutoProcessor

try:
    from transformers import AutoModelForImageTextToText as AutoVLM
except ImportError:
    from transformers import AutoModelForVision2Seq as AutoVLM

import requests
import base64
import io

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
logger.info("HF_ENDPOINT=%s", os.environ.get("HF_ENDPOINT"))
logger.info("HF_HUB_URL=%s", os.environ.get("HF_HUB_URL"))
logger.info("HUGGINGFACE_CO_RESOLVE_ENDPOINT=%s", os.environ.get("HUGGINGFACE_CO_RESOLVE_ENDPOINT"))

REPO_ROOT = Path(__file__).resolve().parent


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
            "guide", "answer", "on-stage interaction", "blackboard-writing",
            "teacher", "stand", "screen", "blackboard",
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

# ── 模型配置 ──────────────────────────────────────────────────────────────────
# backend: "hf" = HuggingFace本地加载
QWEN36_MAX_MEMORY = {0: "34GiB", 1: "34GiB", "cpu": "512GiB"}
GEMMA4_MAX_MEMORY = {0: "34GiB", 1: "34GiB", "cpu": "512GiB"}

MODEL_REGISTRY = {
    "qwen2":   {"backend": "hf", "repo_id": "Qwen/Qwen2-VL-7B-Instruct",     "device": "cuda:0"},
    "qwen25":  {"backend": "hf", "repo_id": "Qwen/Qwen2.5-VL-7B-Instruct",   "device": "cuda:0"},
    "llava":   {"backend": "hf", "repo_id": "llava-hf/llava-1.5-7b-hf",      "device": "cuda:1"},
    "qwen2532": {"backend": "hf", "repo_id": "Qwen/Qwen2.5-VL-32B-Instruct",  "device_map": "auto", "input_device": "cuda:0"},
    "qwen35_27b": {"backend": "hf", "repo_id": "Qwen/Qwen3.5-27B", "device_map": "auto", "input_device": "cuda:0", "max_memory": QWEN36_MAX_MEMORY},
    "qwen35_35b_a3b": {"backend": "hf", "repo_id": "Qwen/Qwen3.5-35B-A3B", "device_map": "auto", "input_device": "cuda:0", "max_memory": QWEN36_MAX_MEMORY},
    "qwen36_27b": {"backend": "hf", "repo_id": "Qwen/Qwen3.6-27B", "device_map": "auto", "input_device": "cuda:0", "max_memory": QWEN36_MAX_MEMORY},
    "qwen36_35b": {"backend": "hf", "repo_id": "Qwen/Qwen3.6-35B-A3B", "device_map": "auto", "input_device": "cuda:0", "max_memory": QWEN36_MAX_MEMORY},
    "qwen36_35b_fp8": {"backend": "hf", "repo_id": "Qwen/Qwen3.6-35B-A3B-FP8", "device_map": "auto", "input_device": "cuda:0", "max_memory": QWEN36_MAX_MEMORY},
    # Canonical key for the Gemma-3-27B-IT run used in the paper. The
    # backend checkpoint is the quantized unsloth mirror of the same model.
    "gemma327": {"backend": "hf", "repo_id": "unsloth/gemma-3-27b-it-bnb-4bit", "device_map": "auto", "input_device": "cuda:0"},
    "gemma4_26b_a4b": {"backend": "hf", "repo_id": "google/gemma-4-26B-A4B-it", "device_map": "auto", "input_device": "cuda:0", "max_memory": GEMMA4_MAX_MEMORY},
    "gemma4_31b": {"backend": "hf", "repo_id": "google/gemma-4-31B-it", "device_map": "auto", "input_device": "cuda:0", "max_memory": GEMMA4_MAX_MEMORY},
}

# Keep old misleading spellings as compatibility aliases because older
# logs and launchers used them for the Qwen2.5-VL-32B and Gemma-3-27B runs.
LEGACY_MODEL_ALIASES = {
    "qwen36": "qwen2532",
    "qwen3.5:27b": "qwen35_27b",
    "qwen3.5:35b-a3b": "qwen35_35b_a3b",
    "qwen3.6:27b": "qwen36_27b",
    "qwen3.6:35b": "qwen36_35b",
    "qwen3.6:35b-fp8": "qwen36_35b_fp8",
    "gemma4": "gemma4_31b",
}

# Old result files used misleading key names. Keep a dedicated mapping for
# analysis/reporting so historical fields are displayed with canonical model keys.
HISTORICAL_RESULT_MODEL_ALIASES = {
    "qwen36": "qwen2532",
    "gemma4": "gemma327",
}

CROP_MARGIN = 0.05
MIN_CROP_PX = 32


def parse_max_memory(value: str) -> Dict[object, str]:
    max_memory: Dict[object, str] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        name, limit = item.split(":", 1)
        name = name.strip()
        key: object = int(name) if name.isdigit() else name
        max_memory[key] = limit.strip()
    return max_memory


def record_key(image_name: str, bbox_idx: int) -> Tuple[str, int]:
    return image_name, int(bbox_idx)


def record_sort_key(record: Dict) -> Tuple[str, int]:
    return record.get("image", ""), int(record.get("bbox_idx", 0))


def load_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def choose_compatibility_pair(
    model_keys: List[str],
    compat_qwen_model: Optional[str] = None,
    compat_llava_model: Optional[str] = None,
) -> Optional[Tuple[str, str]]:
    if len(model_keys) < 2:
        return None

    qwen_key = compat_qwen_model
    if qwen_key is None:
        for candidate in ["qwen25", "qwen2", "qwen"]:
            if candidate in model_keys:
                qwen_key = candidate
                break
        if qwen_key is None:
            qwen_key = model_keys[0]

    llava_key = compat_llava_model
    if llava_key is None:
        if "llava" in model_keys and "llava" != qwen_key:
            llava_key = "llava"
        else:
            llava_key = next((key for key in model_keys if key != qwen_key), None)

    if qwen_key is None or llava_key is None or qwen_key == llava_key:
        return None
    return qwen_key, llava_key


def add_compatibility_aliases(record: Dict, compat_pair: Optional[Tuple[str, str]]) -> None:
    if compat_pair is None:
        return

    qwen_key, llava_key = compat_pair
    for suffix in ["", "_name", "_time", "_raw"]:
        qwen_src = f"pred_{qwen_key}{suffix}" if suffix else f"pred_{qwen_key}"
        llava_src = f"pred_{llava_key}{suffix}" if suffix else f"pred_{llava_key}"
        qwen_dst = f"pred_qwen{suffix}" if suffix else "pred_qwen"
        llava_dst = f"pred_llava{suffix}" if suffix else "pred_llava"
        if qwen_src in record:
            record[qwen_dst] = record[qwen_src]
        if llava_src in record:
            record[llava_dst] = record[llava_src]

    qwen_pred = record.get("pred_qwen")
    llava_pred = record.get("pred_llava")
    record["agreement"] = (
        qwen_pred is not None and llava_pred is not None and qwen_pred == llava_pred
    )
    record["compat_qwen_source"] = qwen_key
    record["compat_llava_source"] = llava_key


def normalize_model_key(model_key: str) -> str:
    if model_key in {"qwen36", "gemma4"}:
        logger.warning(
            "Model alias '%s' is ambiguous historically. It is normalized to '%s' for execution.",
            model_key,
            LEGACY_MODEL_ALIASES[model_key],
        )
    return LEGACY_MODEL_ALIASES.get(model_key, model_key)


def normalize_model_keys(model_keys: List[str]) -> List[str]:
    normalized: List[str] = []
    for key in model_keys:
        canonical = normalize_model_key(key)
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized


def discover_existing_model_keys(dataset_name: str, split: str, out_dir: Path) -> List[str]:
    prefix = f"{dataset_name}_{split}_"
    suffix = "_annotations.jsonl"
    discovered: List[str] = []
    for path in sorted(out_dir.glob(f"{prefix}*{suffix}")):
        if path.name == f"{dataset_name}_{split}_crossmodel_annotations.jsonl":
            continue
        model_key = path.name[len(prefix):-len(suffix)]
        if model_key and model_key not in discovered:
            discovered.append(model_key)

    ordered: List[str] = []
    for key in MODEL_REGISTRY:
        if key in discovered:
            ordered.append(key)
    for key in discovered:
        if key not in ordered:
            ordered.append(key)
    return ordered


def rebuild_merged_annotations(
    dataset_name: str,
    split: str,
    model_keys: Optional[List[str]],
    out_dir: Path,
    compat_pair: Optional[Tuple[str, str]] = None,
) -> Path:
    merged_file = out_dir / f"{dataset_name}_{split}_crossmodel_annotations.jsonl"
    merged_records: Dict[Tuple[str, int], Dict] = {}
    effective_model_keys = model_keys or discover_existing_model_keys(dataset_name, split, out_dir)

    for key in effective_model_keys:
        model_file = out_dir / f"{dataset_name}_{split}_{key}_annotations.jsonl"
        if not model_file.exists():
            continue

        for rec in load_jsonl(model_file):
            rid = record_key(rec["image"], rec["bbox_idx"])
            merged = merged_records.setdefault(
                rid,
                {
                    "image": rec["image"],
                    "bbox_idx": rec["bbox_idx"],
                    "gt": rec["gt"],
                    "gt_name": rec.get("gt_name"),
                    "cx": rec["cx"],
                    "cy": rec["cy"],
                    "w": rec["w"],
                    "h": rec["h"],
                },
            )
            for field in [
                f"pred_{key}",
                f"pred_{key}_name",
                f"time_{key}",
                f"raw_{key}",
            ]:
                if field in rec:
                    merged[field] = rec[field]

    compat_pair = compat_pair or choose_compatibility_pair(effective_model_keys)

    for merged in merged_records.values():
        for i, ka in enumerate(effective_model_keys):
            for kb in effective_model_keys[i + 1:]:
                pa = merged.get(f"pred_{ka}")
                pb = merged.get(f"pred_{kb}")
                merged[f"agree_{ka}_{kb}"] = (
                    pa is not None and pb is not None and pa == pb
                )
        add_compatibility_aliases(merged, compat_pair)

    with open(merged_file, "w") as handle:
        for record in sorted(merged_records.values(), key=record_sort_key):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Rebuilt merged annotations: %s (%d records)", merged_file, len(merged_records))
    if compat_pair is not None:
        logger.info(
            "Compatibility aliases: pred_qwen -> %s, pred_llava -> %s",
            compat_pair[0], compat_pair[1]
        )
    return merged_file


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════

def resolve_split_dirs(dataset_name: str, split: str) -> Tuple[Path, Path]:
    cfg = DATASETS[dataset_name]
    base_path = cfg["path"]
    direct_img_dir = base_path / "images" / split
    direct_lbl_dir = base_path / "labels" / split
    if direct_img_dir.is_dir() and direct_lbl_dir.is_dir():
        return direct_img_dir, direct_lbl_dir
    nested_img_dirs = sorted(p for p in base_path.glob(f"**/images/{split}") if p.is_dir())
    nested_lbl_dirs = sorted(p for p in base_path.glob(f"**/labels/{split}") if p.is_dir())
    if nested_img_dirs and nested_lbl_dirs:
        return nested_img_dirs[0], nested_lbl_dirs[0]
    raise FileNotFoundError(
        f"Cannot find {dataset_name}/{split} under {base_path}"
    )


def build_prompt(classes: List[str]) -> str:
    class_list = "\n".join(f"  {i}: {name}" for i, name in enumerate(classes))
    return (
        "You are a classroom behavior recognition expert. "
        "Given one cropped classroom image, choose the single best matching class. "
        "Reply with only the class index number. No explanation.\n\n"
        f"Classes:\n{class_list}\n\n"
        "Answer with index only:"
    )


def extract_first_int(raw: str) -> Optional[int]:
    m = re.search(r"-?\d+", raw)
    return int(m.group()) if m else None


def infer_pred_from_text(raw: str, classes: List[str]) -> Optional[int]:
    text = raw.lower().strip()
    if not text:
        return None
    for idx, cls_name in enumerate(classes):
        tokens = [re.escape(tok) for tok in re.split(r"[-\s]+", cls_name.lower()) if tok]
        if not tokens:
            continue
        pattern = r"\b" + r"[-\s]+".join(tokens) + r"\b"
        if re.search(pattern, text):
            return idx
    return None


def crop_bbox(img: Image.Image, cx: float, cy: float,
              w: float, h: float, margin: float = CROP_MARGIN) -> Optional[Image.Image]:
    W, H = img.size
    x1 = max(0, (cx - w / 2 - margin) * W)
    y1 = max(0, (cy - h / 2 - margin) * H)
    x2 = min(W, (cx + w / 2 + margin) * W)
    y2 = min(H, (cy + h / 2 + margin) * H)
    if (x2 - x1) < MIN_CROP_PX or (y2 - y1) < MIN_CROP_PX:
        return None
    return img.crop((x1, y1, x2, y2))


def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


# ══════════════════════════════════════════════════════════════════════════════
# HuggingFace 本地模型标注器（复用 step1 的逻辑）
# ══════════════════════════════════════════════════════════════════════════════

class HFAnnotator:
    def __init__(
        self,
        key: str,
        repo_id: str,
        device: str = "cuda:0",
        max_new_tokens: int = 8,
        device_map: Optional[str] = None,
        input_device: Optional[str] = None,
        max_memory: Optional[Dict[object, str]] = None,
        offload_folder: Optional[str] = None,
    ):
        self.key = key
        self.repo_id = repo_id
        self.device_map = device_map
        self.device = self._resolve_device(device)
        self.input_device = self._resolve_device(input_device or device)
        self.dtype = torch.bfloat16 if self.input_device.type == "cuda" and key != "llava" \
                     else (torch.float16 if key == "llava" else torch.float32)

        logger.info(
            "Loading HF model %s (%s) device=%s device_map=%s input_device=%s max_memory=%s offload_folder=%s",
            key, repo_id, self.device, self.device_map, self.input_device, max_memory, offload_folder,
        )
        self.processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": True,
        }
        if self.device_map is not None:
            model_kwargs["device_map"] = self.device_map
        if max_memory:
            model_kwargs["max_memory"] = max_memory
        if offload_folder:
            Path(offload_folder).mkdir(parents=True, exist_ok=True)
            model_kwargs["offload_folder"] = offload_folder
            model_kwargs["offload_state_dict"] = True
        self.model = AutoVLM.from_pretrained(repo_id, **model_kwargs)
        if self.device_map is None:
            self.model = self.model.to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

        # LLaVA processor patch
        if key == "llava":
            patch_size = getattr(self.model.config.vision_config, "patch_size", None)
            strategy   = getattr(self.model.config, "vision_feature_select_strategy", None)
            for target in (self.processor, getattr(self.processor, "image_processor", None)):
                if target is None:
                    continue
                if patch_size is not None:
                    setattr(target, "patch_size", patch_size)
                if strategy is not None:
                    setattr(target, "vision_feature_select_strategy", strategy)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device.startswith("cuda") and torch.cuda.is_available():
            idx = int(device.split(":", 1)[1]) if ":" in device else 0
            if idx < torch.cuda.device_count():
                return torch.device(device)
        return torch.device("cpu")

    def predict(self, prompt: str, image: Image.Image) -> Tuple[Optional[int], float, str]:
        t0 = time.time()
        try:
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": prompt}
            ]}]
            if hasattr(self.processor, "apply_chat_template"):
                try:
                    text = self.processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
            else:
                text = prompt
            inputs = self.processor(text=[text], images=[image],
                                    return_tensors="pt", padding=True)
            inputs = {k: v.to(self.input_device) if hasattr(v, "to") else v
                      for k, v in inputs.items()}
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
                if "input_ids" in inputs:
                    generated_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
                else:
                    generated_ids = output_ids
                raw = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False)[0].strip()
            pred = extract_first_int(raw)
            return pred, time.time() - t0, raw
        except Exception as e:
            logger.warning("HF %s inference failed: %s", self.key, e)
            return None, time.time() - t0, ""


# ══════════════════════════════════════════════════════════════════════════════
# Ollama 标注器（调用本地 Ollama API）
# ══════════════════════════════════════════════════════════════════════════════

class OllamaAnnotator:
    def __init__(self, key: str, ollama_model: str,
                 ollama_url: str = "http://localhost:11434/api/generate",
                 timeout: int = 60):
        self.key          = key
        self.ollama_model = ollama_model
        self.ollama_url   = ollama_url
        self.timeout      = timeout
        logger.info("Ollama annotator: %s → %s @ %s", key, ollama_model, ollama_url)
        self._check_connection()

    def _check_connection(self) -> None:
        try:
            resp = requests.get(
                self.ollama_url.replace("/api/generate", "/api/tags"),
                timeout=5
            )
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            if not any(self.ollama_model.lower() in m.lower() for m in models):
                raise RuntimeError(
                    "Model '%s' not found in Ollama. Available: %s"
                    % (self.ollama_model, models)
                )
            logger.info("Ollama model '%s' found.", self.ollama_model)
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.ollama_url}: {e}"
            ) from e

    def predict(self, prompt: str, image: Image.Image) -> Tuple[Optional[int], float, str]:
        t0 = time.time()
        try:
            img_b64 = image_to_base64(image)
            payload = {
                "model":  self.ollama_model,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False,
                "options": {"temperature": 0.0, "seed": 42},
            }
            resp = requests.post(self.ollama_url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()
            m   = re.search(r"\d+", raw)
            pred = int(m.group()) if m else None
            return pred, time.time() - t0, raw
        except Exception as e:
            logger.warning("Ollama %s inference failed: %s", self.key, e)
            return None, time.time() - t0, ""


# ══════════════════════════════════════════════════════════════════════════════
# 标注器工厂
# ══════════════════════════════════════════════════════════════════════════════

def build_annotators(model_keys: List[str],
                     max_new_tokens: int = 8) -> Dict[str, HFAnnotator | OllamaAnnotator]:
    annotators = {}
    for key in normalize_model_keys(model_keys):
        if key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model key '{key}'. Choose from: {list(MODEL_REGISTRY)}")
        cfg = MODEL_REGISTRY[key]
        backend = cfg.get("backend", "hf")
        if backend != "hf":
            raise ValueError(f"Unsupported backend '{backend}' for model key '{key}'")
        repo_id = os.environ.get(f"{key.upper()}_REPO_ID", cfg["repo_id"])
        max_memory = cfg.get("max_memory")
        max_memory_env = os.environ.get(f"{key.upper()}_MAX_MEMORY") or os.environ.get("HF_MAX_MEMORY")
        if max_memory_env:
            max_memory = parse_max_memory(max_memory_env)
        offload_folder = (
            os.environ.get(f"{key.upper()}_OFFLOAD_FOLDER")
            or os.environ.get("HF_OFFLOAD_FOLDER")
        )
        effective_max_new_tokens = max_new_tokens
        if (key.startswith("qwen35_") or key.startswith("qwen36_")) and max_new_tokens < 32:
            logger.info(
                "Bumping max_new_tokens from %d to 32 for %s to avoid truncated non-answer preambles.",
                max_new_tokens,
                key,
            )
            effective_max_new_tokens = 32
        annotators[key] = HFAnnotator(
            key=key,
            repo_id=repo_id,
            device=cfg.get("device", "cuda:0"),
            device_map=cfg.get("device_map"),
            input_device=cfg.get("input_device"),
            max_memory=max_memory,
            offload_folder=offload_folder,
            max_new_tokens=effective_max_new_tokens,
        )
    return annotators


# ══════════════════════════════════════════════════════════════════════════════
# 主标注流程
# ══════════════════════════════════════════════════════════════════════════════

def annotate_dataset(
    dataset_name: str,
    split: str,
    annotators: Dict[str, HFAnnotator | OllamaAnnotator],
    out_dir: Path,
    max_images: Optional[int] = None,
    overwrite: bool = False,
    compat_qwen_model: Optional[str] = None,
    compat_llava_model: Optional[str] = None,
) -> None:
    cfg     = DATASETS[dataset_name]
    classes = cfg["classes"]
    prompt  = build_prompt(classes)
    img_dir, lbl_dir = resolve_split_dirs(dataset_name, split)

    img_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if max_images is not None:
        img_paths = img_paths[:max_images]

    logger.info("Dataset=%s split=%s images=%d models=%s",
                dataset_name, split, len(img_paths), list(annotators))

    out_dir.mkdir(parents=True, exist_ok=True)

    # 每个模型独立输出文件（支持断点续跑）
    model_files: Dict[str, Path] = {
        key: out_dir / f"{dataset_name}_{split}_{key}_annotations.jsonl"
        for key in annotators
    }

    compat_pair = choose_compatibility_pair(
        list(annotators),
        compat_qwen_model=compat_qwen_model,
        compat_llava_model=compat_llava_model,
    )

    # 找出各模型已完成的 bbox
    done_per_model: Dict[str, set] = {key: set() for key in annotators}
    for key, fpath in model_files.items():
        if fpath.exists() and not overwrite:
            with open(fpath) as f:
                for line in f:
                    rec = json.loads(line)
                    done_per_model[key].add(record_key(rec["image"], rec["bbox_idx"]))
            logger.info("[%s] resuming: %d bbox records done", key, len(done_per_model[key]))
        elif fpath.exists() and overwrite:
            fpath.unlink()

    merged_file = out_dir / f"{dataset_name}_{split}_crossmodel_annotations.jsonl"
    if merged_file.exists() and overwrite:
        merged_file.unlink()

    # 打开各模型输出文件（追加模式）
    handles = {key: open(model_files[key], "a") for key in annotators}

    try:
        for img_idx, img_path in enumerate(img_paths):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                logger.warning("Cannot open %s: %s", img_path, e)
                continue

            lines = lbl_path.read_text().strip().splitlines()

            for bbox_idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                gt_cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])

                crop = crop_bbox(img, cx, cy, w, h)
                if crop is None:
                    continue

                base_record = {
                    "image":    img_path.name,
                    "bbox_idx": bbox_idx,
                    "gt":       gt_cls,
                    "gt_name":  classes[gt_cls] if gt_cls < len(classes) else "unknown",
                    "cx": cx, "cy": cy, "w": w, "h": h,
                }
                rid = record_key(img_path.name, bbox_idx)

                for key, annotator in annotators.items():
                    if rid in done_per_model[key]:
                        continue

                    pred, elapsed, raw = annotator.predict(prompt, crop.copy())
                    if pred is None:
                        pred = infer_pred_from_text(raw, classes)
                    if pred is not None and pred >= len(classes):
                        pred = None

                    rec = dict(base_record)
                    rec[f"pred_{key}"]      = pred
                    rec[f"pred_{key}_name"] = classes[pred] if pred is not None else None
                    rec[f"time_{key}"]      = round(elapsed, 3)
                    rec[f"raw_{key}"]       = raw

                    handles[key].write(json.dumps(rec, ensure_ascii=False) + "\n")
                    handles[key].flush()
                    done_per_model[key].add(rid)

            if (img_idx + 1) % 20 == 0:
                logger.info("Progress: %d / %d images", img_idx + 1, len(img_paths))

    finally:
        for h in handles.values():
            h.close()

    rebuild_merged_annotations(
        dataset_name=dataset_name,
        split=split,
        model_keys=None,
        out_dir=out_dir,
        compat_pair=compat_pair,
    )

    logger.info("Done. Output dir: %s", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
# 标注质量分析（直接在标注完成后打印，不需要单独跑 step2）
# ══════════════════════════════════════════════════════════════════════════════

def analyze_annotations(dataset_name: str, split: str,
                         model_keys: List[str], out_dir: Path) -> None:
    classes = DATASETS[dataset_name]["classes"]
    summary_json_path = out_dir / f"{dataset_name}_{split}_crossmodel_summary.json"
    summary_csv_path = out_dir / f"{dataset_name}_{split}_crossmodel_summary.csv"
    merged_file = out_dir / f"{dataset_name}_{split}_crossmodel_annotations.jsonl"
    compat_pair = choose_compatibility_pair(model_keys)
    summary_rows: List[Dict[str, object]] = []
    used_summary_keys: set[str] = set()
    summary_payload: Dict[str, object] = {
        "dataset": dataset_name,
        "split": split,
        "models": model_keys,
        "compatibility_pair": compat_pair,
        "per_model": {},
        "pairwise_agreement": {},
    }

    print(f"\n{'='*70}")
    print(f"Cross-model annotation quality: {dataset_name} / {split}")
    print(f"{'='*70}")

    def resolve_summary_model_key(model_key: str) -> Tuple[str, Optional[str]]:
        summary_key = HISTORICAL_RESULT_MODEL_ALIASES.get(model_key, model_key)
        source_key = model_key if summary_key != model_key else None
        if summary_key in used_summary_keys:
            summary_key = model_key
            source_key = None
        used_summary_keys.add(summary_key)
        return summary_key, source_key

    for key in model_keys:
        fpath = out_dir / f"{dataset_name}_{split}_{key}_annotations.jsonl"
        if not fpath.exists():
            print(f"  [{key}] No output file found.")
            continue

        summary_key, source_key = resolve_summary_model_key(key)

        records = load_jsonl(fpath)
        total   = len(records)
        pred_key = f"pred_{key}"

        valid   = [r for r in records if r.get(pred_key) is not None]
        correct = [r for r in valid if r[pred_key] == r["gt"]]
        acc     = 100.0 * len(correct) / len(valid) if valid else 0.0
        per_cls_summary: Dict[str, Dict[str, float]] = {}

        label = summary_key if source_key is None else f"{summary_key} <- legacy {source_key}"
        print(f"\n  [{label}] total={total} valid={len(valid)} "
              f"overall_acc={acc:.2f}%")

        # 逐类别
        per_cls: Dict[str, Dict] = {c: {"correct": 0, "total": 0} for c in classes}
        for r in valid:
            cls_name = r.get("gt_name", classes[r["gt"]] if r["gt"] < len(classes) else "?")
            per_cls[cls_name]["total"]   += 1
            per_cls[cls_name]["correct"] += int(r[pred_key] == r["gt"])

        print(f"  {'Class':<28} {'n':>6} {'acc(%)':>8}")
        print(f"  {'-'*44}")
        for cls in classes:
            d = per_cls[cls]
            a = 100.0 * d["correct"] / d["total"] if d["total"] else 0.0
            per_cls_summary[cls] = {
                "n": d["total"],
                "acc": round(a, 2),
            }
            print(f"  {cls:<28} {d['total']:>6} {a:>7.1f}%")

        summary_rows.append({
            "model": summary_key,
            "source_model_key": source_key,
            "total": total,
            "valid": len(valid),
            "overall_acc": round(acc, 2),
        })
        summary_payload["per_model"][summary_key] = {
            "source_model_key": source_key,
            "total": total,
            "valid": len(valid),
            "overall_acc": round(acc, 2),
            "per_class": per_cls_summary,
        }

    if merged_file.exists():
        merged_records = load_jsonl(merged_file)
        for i, ka in enumerate(model_keys):
            for kb in model_keys[i + 1:]:
                comparable = [
                    r for r in merged_records
                    if r.get(f"pred_{ka}") is not None and r.get(f"pred_{kb}") is not None
                ]
                if not comparable:
                    continue
                agree_count = sum(
                    1 for r in comparable if r.get(f"pred_{ka}") == r.get(f"pred_{kb}")
                )
                pair_key = f"{ka}__{kb}"
                summary_payload["pairwise_agreement"][pair_key] = {
                    "comparable": len(comparable),
                    "agree": agree_count,
                    "agreement_rate": round(100.0 * agree_count / len(comparable), 2),
                }

    with open(summary_json_path, "w") as handle:
        json.dump(summary_payload, handle, ensure_ascii=False, indent=2)

    with open(summary_csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "source_model_key", "total", "valid", "overall_acc"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    logger.info("Saved cross-model summaries: %s, %s", summary_json_path, summary_csv_path)

    print(f"\n{'='*70}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 1b: Cross-model annotation for validation experiment"
    )
    available_model_keys = list(MODEL_REGISTRY) + list(LEGACY_MODEL_ALIASES)
    parser.add_argument("--dataset",  choices=list(DATASETS), required=True)
    parser.add_argument("--split",    choices=["val", "train"], default="val")
    parser.add_argument("--models",   nargs="+",
                        choices=available_model_keys,
                        default=["qwen25"],
                        help=f"Models to run. Available: {available_model_keys}")
    parser.add_argument("--out_dir",  default=None)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--overwrite",  action="store_true")
    parser.add_argument("--compat_qwen_model", choices=available_model_keys, default=None,
                        help="Model key to expose as pred_qwen in merged compatibility output")
    parser.add_argument("--compat_llava_model", choices=available_model_keys, default=None,
                        help="Model key to expose as pred_llava in merged compatibility output")
    parser.add_argument("--no_analyze", action="store_true",
                        help="Skip quality analysis after annotation")
    args = parser.parse_args()

    args.models = normalize_model_keys(args.models)
    args.compat_qwen_model = (
        normalize_model_key(args.compat_qwen_model) if args.compat_qwen_model else None
    )
    args.compat_llava_model = (
        normalize_model_key(args.compat_llava_model) if args.compat_llava_model else None
    )

    for compat_key in [args.compat_qwen_model, args.compat_llava_model]:
        if compat_key is not None and compat_key not in args.models:
            raise ValueError(
                f"Compatibility alias model '{compat_key}' must also be listed in --models"
            )

    repo_root = Path(__file__).resolve().parent
    out_dir = (Path(args.out_dir) if args.out_dir
               else repo_root / "results" / "cross_model_validation" / "default")

    annotators = build_annotators(args.models, max_new_tokens=args.max_new_tokens)

    annotate_dataset(
        dataset_name = args.dataset,
        split        = args.split,
        annotators   = annotators,
        out_dir      = out_dir,
        max_images   = args.max_images,
        overwrite    = args.overwrite,
        compat_qwen_model = args.compat_qwen_model,
        compat_llava_model = args.compat_llava_model,
    )

    if not args.no_analyze:
        analyze_annotations(args.dataset, args.split, args.models, out_dir)


if __name__ == "__main__":
    main()


# ══════════════════════════════════════════════════════════════════════════════
# 快速开始
# ══════════════════════════════════════════════════════════════════════════════
#
# 1. 先用 50 张图冒烟测试，确认 Qwen2.5-VL-7B 可以正常跑：
#    CUDA_VISIBLE_DEVICES=0 python cross_model_annotate.py \
#        --dataset BowTurnHead --split val --models qwen25 --max_images 50
#
# 2. 测试 Qwen2.5-VL-32B-Instruct：
#    python cross_model_annotate.py \
#        --dataset BowTurnHead --split val --models qwen2532 --max_images 50
#
# 3. 全量 val 集跑三个新模型：
#    CUDA_VISIBLE_DEVICES=0 python cross_model_annotate.py \
#        --dataset TeacherBehavior --split val --models qwen25
#    python cross_model_annotate.py \
#        --dataset TeacherBehavior --split val --models qwen2532
#    python cross_model_annotate.py \
#        --dataset TeacherBehavior --split val --models gemma327
#
# 4. 关键数字：看 TeacherBehavior val 集上 answer/guide/on-stage interaction
#    三个低锚点类别的准确率，和原来的 Qwen2-VL-7B（0%）对比。
#    如果 Qwen2.5 和大模型仍然是 0%，说明这是任务本身的限制，结论更强。
#    如果有显著提升（>20%），说明模型能力是瓶颈，临界点在这里。
# ══════════════════════════════════════════════════════════════════════════════