# Logs Layout

This directory is the single home for launcher logs, nohup traces, and auxiliary monitoring files.

## Naming scheme

- `pipeline/`: combined multi-phase launcher logs.
- `phase0/`: zero-shot evaluation logs.
- `cross_model_validation/`: cross-model validation logs.
- `phase3/`: fine-tuning, LoRA sweep, and repeated-seed logs.
- `phase45/`: selective-annotation plus retention diagnostic retries.
- `phase5/`: standalone phase-5 teacher runs.
- `phase6/`: phase-6 strategy-audit launchers and background traces.
- `aux/`: non-canonical monitoring or ad-hoc background traces.

## Rule

- Log names should describe phase, task, and run tag.
- Failed and historical logs may be retained, but they should stay in the appropriate phase bucket.