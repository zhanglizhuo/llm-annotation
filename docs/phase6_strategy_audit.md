# Phase-6 Strategy Audit Analyses

This note records the phase-6 strategy-audit analyses. The scripts use the
existing SCB YOLO layout, phase-based result directories, and canonical JSONL
pseudo-label files produced by the earlier stages.

## Scripts

| Script | Analysis focus | Main inputs | Outputs |
|---|---|---|---|
| `cross_model_consistency.py` | Cross-model anchoring check | Main Qwen/LLaVA JSONL plus cross-model validation JSONL | `cross_model_consistency.csv`, pairwise agreement, optional CMC-vs-ZS correlations and plots |
| `confidence_filtering.py` | CLIP-assisted confidence filtering | Phase-2 `*_train_pseudo_none.jsonl` plus SCB train/val YOLO labels | threshold x seed result CSV and best-threshold summary |
| `teacher_student_self_training.py` | Train-split teacher-student self-training | SCB train/val YOLO labels | seed-level self-training result CSV and summary |

## Protocol Notes

- Cross-model consistency is grouped by true validation class (`gt`) from the
  existing annotation JSONL records. It does not use model majority vote to
  approximate class membership.
- The teacher-student baseline does not train on validation labels. The default
  protocol trains the teacher on a stratified 10% labeled subset of the training
  split, labels the remaining training pool, trains a student, and evaluates on
  the validation split.
- The confidence-filtering baseline uses Qwen pseudo-labels from the canonical
  phase-2 JSONL files. Because the index-only MLLM prompt does not expose
  calibrated confidence, the implemented confidence signal is CLIP zero-shot
  probability assigned to the Qwen pseudo-label (`--confidence_score pseudo_prob`,
  default). The manuscript therefore treats it as CLIP-assisted confidence
  filtering, not as calibrated MLLM logit confidence.
- The phase-6 launcher defaults to `HF_HUB_OFFLINE=1` because the required CLIP
  weights are already cached locally in the experiment environment. Set
  `HF_HUB_OFFLINE=0` only if a fresh cache download is intentionally needed.

## Recommended Run

Foreground:

```bash
cd /school_Agri/Annotation
bash run_phase6_strategy_audit.sh
```

Background:

```bash
cd /school_Agri/Annotation
bash run_phase6_strategy_audit_bg.sh
tail -f logs/phase6/phase6_strategy_audit_*.log
```

Useful overrides:

```bash
RUN_CROSS_MODEL_CONSISTENCY=1 RUN_CONFIDENCE_FILTERING=0 RUN_TEACHER_STUDENT=0 bash run_phase6_strategy_audit.sh
RUN_CROSS_MODEL_CONSISTENCY=0 RUN_CONFIDENCE_FILTERING=1 RUN_TEACHER_STUDENT=0 BATCH_SIZE=64 bash run_phase6_strategy_audit.sh
TEACHER_LABEL_FRACTION=0.1 TEACHER_CONF_THRESHOLD=0.0 bash run_phase6_strategy_audit.sh
```

Default outputs are written under:

```text
Annotation/results/phase6_strategy_audit/<RUN_TAG>/
Annotation/logs/phase6/
```

## Manuscript Trace

For the auxiliary baseline rows, use the summary CSV/JSON files:

- `confidence_filtering/confidence_filtering_summary.csv`
- `teacher_student/teacher_student_summary.csv`

For the independent anchor proxy paragraph, use:

- `cross_model_consistency/cross_model_consistency.csv`
- `cross_model_consistency/anchor_proxy_correlations.csv` when CLIP per-class zero-shot
  results are available.

These analyses are auxiliary comparators within the audit protocol. They bound
the main finding rather than claiming that one training strategy uniformly
beats all semi-supervised alternatives.