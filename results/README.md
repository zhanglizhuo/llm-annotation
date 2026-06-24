# Results Layout

This directory is the single home for local experiment outputs.

Nothing here should be edited by hand after generation. Reorganization is done only by moving whole files or directories without changing result contents.

## Naming scheme

- `phase0_zero_shot/`: protocol-matched zero-shot baselines.
- `phase1_annotations/`: raw MLLM bbox-crop annotations.
- `cross_model_validation/`: additional cross-model validation outputs.
- `phase2_filtering/`: filtering tables and pseudo-label JSONL files.
- `phase3_finetune/`: linear-probe, LoRA, and repeated-seed fine-tuning outputs.
- `phase4_selective_annotation/`: selective-routing diagnostics.
- `phase5_retention_curve/`: retention-ratio diagnostics.
- `phase6_strategy_audit/`: phase-6 auxiliary baselines and consistency analyses.

## Rule

- Treat this directory as generated local state.
- Use `docs/experiment_file_map.md` to identify which subpaths are canonical paper evidence.