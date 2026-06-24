# llm-annotation

Code and experiment metadata for reproducing the classroom-behavior study on multimodal LLM pseudo-labeling and CLIP adaptation.

This repository is maintained as a reproducibility repo. Optional manuscript source is kept under `paper/`, but it is not required for rerunning the pipeline.

## Tested environment

- Linux (Ubuntu 18.04+)
- Python 3.8
- CUDA-enabled PyTorch stack (tested on 2× A100 40G)
- Tested package set is captured in `requirements.txt` and `environment.yml`

## Repository layout

- `download_dataset.py`: downloads the public SCB dataset files from Hugging Face into `datasets_scb/`.
- `datasets_scb/`: dataset downloads used by the experiments. This directory is **not** tracked by git — you must run `download_dataset.py` first.
- `docs/`: planning and experiment-governance documents.
  - `docs/research_plan.md`: study design, analysis modules, and recommended execution order.
  - `docs/experiment_file_map.md`: source-of-truth map from scripts to canonical output files used by the paper.
- `results/`: phase-organized experimental outputs (JSON, JSONL, CSV).
- `logs/`: phase-organized launcher and audit logs.
- `finetune_summary.csv`: final paper-level result matrix (21 conditions × 3 datasets).
- `paper/`: optional manuscript source (LaTeX + figures). Not required for reproduction.
- `EXPERIMENTS_SUMMARY.md`: quick overview of all scripts with one-line descriptions.
- `CITATION.cff` / `.zenodo.json`: citation metadata and Zenodo archival configuration.

## Quick start

### 1. Environment setup

```bash
# Option A: conda
conda env create -f environment.yml
conda activate llm-annotation

# Option B: pip only
python -m pip install -r requirements.txt
```

All shell runners also accept `PYTHON_BIN=/path/to/python` if you want to use a non-default interpreter.

### 2. Download the dataset

The SCB (Student Classroom Behavior) dataset is publicly hosted on Hugging Face. Download it with:

```bash
python download_dataset.py
```

This creates `datasets_scb/` at the repository root with four subsets:
- `SCB_BowTurnHead/` (~615 MB)
- `SCB5_HandriseReadWrite/` (~1.5 GB)
- `SCB5_TeacherBehavior/` (~3.2 GB)
- `SCB5_Discuss/` (~116 MB)

The script uses [HF-Mirror](https://hf-mirror.com) for users in mainland China and supports resume-on-failure.

If you already have the dataset elsewhere, set the environment variable:

```bash
export SCB_DATASET_ROOT=/path/to/your/datasets_scb
```

The Python scripts resolve data in this order:
1. `SCB_DATASET_ROOT` environment variable
2. `./datasets_scb` at the repository root

### 3. Reproduce the main experiments

```bash
# Full pipeline: annotation → filtering → fine-tuning
bash run_phase123_full_pipeline.sh

# LoRA sweep
bash run_phase3_lora_sweep_2gpu.sh

# Diagnostics (selective routing + retention curve)
bash run_phase45_diagnostics.sh

# Strategy audit (cross-model consistency, confidence filtering, teacher-student)
bash run_phase6_strategy_audit.sh
```

To skip dependency installation in an already-prepared environment:

```bash
INSTALL_DEPS=0 bash run_phase123_full_pipeline.sh
```

## LLM model setup for annotation

The annotation scripts (`step1_llm_annotate.py`, `cross_model_annotate.py`) use Hugging Face models:
- **Qwen2-VL-7B** — primary annotator
- **LLaVA-1.5-7B** — secondary annotator (agreement filtering)
- **Qwen2.5-7B / 32B, Gemma-3-27B** — cross-model validation

These models are downloaded automatically by `transformers` on first use. For Chinese mainland users, set:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## Reproducing later-stage diagnostics

After phase 1–3 outputs exist, the later runners auto-discover the newest filtering directory. You can also set it explicitly:

```bash
ANALYSIS_DIR=./results/phase2_filtering/<your_run_tag> bash run_phase45_diagnostics.sh
```

Use `docs/experiment_file_map.md` to identify the canonical JSONL, CSV, and JSON outputs referenced by the manuscript.

## Hardware requirements

The experiments were run on 2× NVIDIA A100 40G GPUs. The recommended execution policy is documented in `docs/research_plan.md` (§2). Key points:
- Phase A (annotation): GPU 0 runs Qwen, GPU 1 runs LLaVA
- Phase B: stop LLM services to free memory
- Phase C (fine-tuning): both GPUs used for CLIP training via DataParallel

With smaller GPUs, you may need to reduce `batch_size` from the default 64.

## Archival DOI via Zenodo

This repository includes `CITATION.cff` and `.zenodo.json`. To archive a release:

1. Push the desired commit to GitHub.
2. In Zenodo, connect the `zhanglizhuo/llm-annotation` repository and enable archiving.
3. Create a GitHub release with a version tag (e.g., `v1.0.0`).
4. Zenodo will mint both a version-specific DOI and a concept DOI.

## Notes

- Model weights (`.pt`, `.pth`, `.bin`, `.safetensors`, `.ckpt`) are **not** tracked in git — they are too large for GitHub. Re-run the training scripts to regenerate them.
- The dataset (`datasets_scb/`) is **not** tracked — download it with `download_dataset.py`.
- If you only want the paper source, see `paper/`. If you only want to reproduce experiments, start from the root scripts — the paper directory is optional.
- Generated outputs under `results/` and `logs/` are **tracked** so reviewers can inspect exact experimental outcomes without re-running.
