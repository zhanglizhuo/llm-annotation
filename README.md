# LLM-Assisted Annotation for Classroom Behavior Recognition

**Can multimodal LLMs replace manual annotation for fine-grained classroom behavior recognition — and does it actually work when you try to train on those labels?**

This repository provides the complete experiment pipeline, results, and analysis code for our study. Every number in the paper can be traced back to a specific script, log file, and result JSON.

---

## What this research is about

Classroom behavior recognition matters: it enables automated teaching quality assessment, real-time feedback, and large-scale pedagogical research. But the bottleneck has always been **annotation cost** — frame-by-frame labeling of classroom videos requires domain experts and is prohibitively expensive at scale.

We test a simple but underexplored idea:

> **Use multimodal LLMs (Qwen2-VL, LLaVA) to generate pseudo-labels for bbox-cropped classroom images, then fine-tune CLIP on those labels.**

The question is not whether LLM labels are perfect (they aren't). The question is whether they are **good enough to train a downstream model that beats zero-shot CLIP**.

### The pipeline

```
Classroom video frames
    → YOLO bbox detection
    → Multimodal LLM annotation (Qwen2-VL + LLaVA)
    → Pseudo-label filtering (none / dual-model agreement)
    → CLIP ViT-L/14 fine-tuning (linear probe / LoRA)
    → Evaluation on 3 classroom behavior datasets
```

### Three datasets, three difficulty levels

| Dataset | Classes | Zero-shot CLIP | Best pseudo-label result | GT upper bound |
|---------|--------:|:-------------:|:------------------------:|:--------------:|
| **BowTurnHead** (2-class) | 2 | 42.37% | **88.33%** (+46pp) | 97.92% |
| **HandriseReadWrite** (5-class) | 5 | 56.88% | **76.69%** (+20pp) | 87.06% |
| **TeacherBehavior** (7-class) | 7 | 37.13% | **45.18%** (+8pp) | 74.76% |

### Key findings at a glance

1. **LLM pseudo-labels work — but the gain varies by task complexity.** On the 2-class and 5-class datasets, pseudo-label training roughly doubles zero-shot accuracy. On the 7-class TeacherBehavior dataset, gains are modest (+8pp), revealing a genuine difficulty boundary.

2. **Dual-model agreement filtering is a double-edged sword.** It improves label purity but sharply reduces sample count. On BowTurnHead, agreement filtering retains only 20% of samples and *hurts* performance (19.69% vs 88.33% with unfiltered pseudo-labels). On TeacherBehavior, it helps slightly (45.18% vs 44.34%).

3. **The gap to GT upper bound tells you where the problem lives.** For BowTurnHead, pseudo-label training nearly saturates the GT upper bound (88.33% vs 93.79% for linear probe). For TeacherBehavior, a ~30pp gap remains. The bottleneck shifts from "are the labels good enough" to "is the task intrinsically harder for CLIP," and our selective-routing and retention-curve diagnostics explore exactly this question.

4. **LoRA and linear probe perform similarly with pseudo-labels.** Unlike the GT setting where LoRA has a clear edge, pseudo-label training shows negligible difference between the two adaptation methods — suggesting label noise, not model capacity, is the binding constraint.

---

## Repository structure

| Directory / File | Purpose |
|---|---|
| `download_dataset.py` | Download SCB dataset from Hugging Face |
| `requirements.txt` / `environment.yml` | Python dependencies |
| `*.py` | All experiment scripts (annotation, filtering, training, diagnostics) |
| `*.sh` | Shell launchers for each experiment phase |
| `results/` | **Tracked** — all experimental outputs (JSON, JSONL, CSV) |
| `logs/` | **Tracked** — execution traces for every run |
| `finetune_summary.csv` | **The final result table** — 21 conditions across 3 datasets |
| `docs/research_plan.md` | Study design, hypotheses, module dependencies |
| `docs/experiment_file_map.md` | Which script produced which file → which table in the paper |
| `paper/` | Manuscript source (LaTeX + figures). Optional; not needed for reproduction. |

**Why results are tracked:** You can inspect every number in the paper without re-running a single experiment. Each result file links back to a specific script and log.

---

## Quick start (reproduction)

### 1. Install dependencies

```bash
# conda (recommended)
conda env create -f environment.yml
conda activate llm-annotation

# or pip
pip install -r requirements.txt
```

### 2. Download the dataset

```bash
python download_dataset.py
```

This downloads ~5.4 GB from Hugging Face (HF-Mirror for mainland China, with resume support) into `datasets_scb/`.

If you already have the data elsewhere:
```bash
export SCB_DATASET_ROOT=/path/to/datasets_scb
```

### 3. Run the pipeline

```bash
# Full pipeline: annotation → filtering → CLIP fine-tuning
bash run_phase123_full_pipeline.sh

# LoRA hyperparameter sweep
bash run_phase3_lora_sweep_2gpu.sh

# Mechanism diagnostics (selective routing + retention curves)
bash run_phase45_diagnostics.sh

# Strategy audit (cross-model consistency, confidence filtering, teacher-student)
bash run_phase6_strategy_audit.sh
```

**Skip dependency re-installation** in an already-prepared environment:
```bash
INSTALL_DEPS=0 bash run_phase123_full_pipeline.sh
```

### 4. Check the results

```bash
# The final result matrix
cat finetune_summary.csv

# Per-condition details (accuracy curves, epoch history)
cat results/phase3_finetune/full_pipeline/full_20260418_0001/BowTurnHead_linear_none_result.json
```

---

## Expected outputs

Every run writes structured outputs under `results/`:

| Phase | Output | Format |
|-------|--------|--------|
| 0 — Zero-shot baseline | `results/phase0_zero_shot/canonical_*/phase0_zero_shot_results.json` | JSON |
| 1 — LLM annotation | `results/phase1_annotations/*/*_annotations.jsonl` | JSONL (one record per bbox) |
| 2 — Filtering analysis | `results/phase2_filtering/*/*_filter_comparison.csv` | CSV |
| 3 — CLIP fine-tuning | `results/phase3_finetune/**/*_result.json` | JSON (accuracy + epoch history) |
| 4 — Selective routing | `results/phase4_selective_annotation/default/` | JSON |
| 5 — Retention curves | `results/phase5_retention_curve/default/` | JSON |
| 6 — Strategy audit | `results/phase6_strategy_audit/*/` | CSV + JSON + PNG |

---

## Models

### Primary annotators (dual-model pseudo-labeling)
| Model | Hugging Face ID | Role |
|---|---|---|
| **Qwen2-VL-7B-Instruct** | `Qwen/Qwen2-VL-7B-Instruct` | Primary pseudo-label generator |
| **LLaVA-1.5-7B** | `llava-hf/llava-1.5-7b-hf` | Secondary annotator (dual-model agreement) |

### Cross-model validation (annotator robustness)
| Model | Hugging Face ID | Size / Variant |
|---|---|---|
| **Qwen2.5-VL-7B-Instruct** | `Qwen/Qwen2.5-VL-7B-Instruct` | 7B |
| **Qwen2.5-VL-32B-Instruct** | `Qwen/Qwen2.5-VL-32B-Instruct` | 32B |
| **Qwen3.5-27B** | `Qwen/Qwen3.5-27B` | 27B |
| **Qwen3.5-35B-A3B** | `Qwen/Qwen3.5-35B-A3B` | 35B MoE (3B active) |
| **Qwen3.6-27B** | `Qwen/Qwen3.6-27B` | 27B |
| **Qwen3.6-35B-A3B** | `Qwen/Qwen3.6-35B-A3B` | 35B MoE (3B active) |
| **Qwen3.6-35B-A3B-FP8** | `Qwen/Qwen3.6-35B-A3B-FP8` | 35B MoE (FP8 quantized) |
| **Gemma-3-27B-IT** | `unsloth/gemma-3-27b-it-bnb-4bit` | 27B (4-bit) |
| **Gemma-4-26B-A4B-it** | `google/gemma-4-26B-A4B-it` | 26B MoE (4B active) |
| **Gemma-4-31B-it** | `google/gemma-4-31B-it` | 31B |

### Fine-tuned vision model
| Model | Backbone | Library |
|---|---|---|
| **CLIP ViT-L/14** | Vision Transformer Large (OpenAI pretrained) | `open_clip_torch` |

Training methods: **linear probe** and **LoRA** (rank=4).

## Environment & hardware

- **Tested on:** Ubuntu 18.04, Python 3.8, 2× NVIDIA A100 40G
- **GPU memory note:** Annotation phase needs ~20GB per 7B LLM; larger models (27B–35B) use `device_map="auto"` across both GPUs. With smaller GPUs, use `CUDA_VISIBLE_DEVICES` to run annotators sequentially.

Models are downloaded automatically by `transformers` on first use. For Chinese mainland users:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## Reproducibility guarantees

- **Deterministic results:** All training scripts accept a `--seed` flag. The canonical results use fixed seeds.
- **File provenance:** `docs/experiment_file_map.md` traces every paper figure/table back to its source file.
- **No silent fallbacks:** Training scripts explicitly error if pseudo-label files are missing (no silent fallback to GT).
- **Tracked outputs:** All result JSON/CSV/JSONL files are checked into this repo. Model weights (`.pt`) are excluded due to size, but can be regenerated by re-running the training scripts.

---

## Paper & citation

The manuscript source is in `paper/`. If you use this code or data, please cite:

```bibtex
@software{ma_zhang_llm_annotation,
  title        = {llm-annotation: LLM-Assisted Annotation for Classroom Behavior Recognition},
  author       = {Ma, Yan and Zhang, Lizhuo},
  year         = {2025},
  url          = {https://github.com/zhanglizhuo/llm-annotation},
  organization = {Hunan Agricultural University},
}
```

See `CITATION.cff` for full metadata. A Zenodo DOI will be added upon release.

---

## Reading order for reviewers

1. This README — overview and key results
2. `finetune_summary.csv` — the main result table (30 seconds)
3. `docs/research_plan.md` — study design and hypotheses (5 minutes)
4. `docs/experiment_file_map.md` — which file is which (5 minutes)
5. `results/phase3_finetune/` — drill into specific conditions
6. `paper/llm_annotation_paper_plos.tex` — full manuscript

---

## License

This repository is made available for research reproducibility. A formal license will be added before the first release.
