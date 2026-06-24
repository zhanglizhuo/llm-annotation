# Annotation

This directory is the executable core of the repository.

If your goal is to reproduce the experiments, start here rather than in `paper/`.

The structure is intentionally split into five stable areas:

- `docs/`: research planning and experiment governance.
- repository-root files in `Annotation/`: runnable launchers, analysis scripts, and the final summary CSV.
- `results/`: generated experiment outputs organized by artifact family.
- `logs/`: launcher, retry, and audit logs organized by artifact family.
- `paper/`: optional manuscript source.

## What belongs here

- Pipeline scripts such as `run_phase123_full_pipeline.sh`, `run_phase45_diagnostics.sh`, and the analysis scripts they call.
- Reproduction metadata such as `finetune_summary.csv`.
- Planning and experiment-governance documents under `docs/`.
- Generated outputs under `results/`.
- Execution traces under `logs/`.
- Small helper scripts that summarize or launch experiments.

## What does not belong here

- Local dataset downloads.
- Ad-hoc local artifacts written outside the canonical `results/` and `logs/` tree.
- Submission templates, cover letters, checklists, or local drafting files.

## Research docs

- `docs/research_plan.md`: study design, analysis modules, and the recommended execution policy.
- `docs/experiment_file_map.md`: source-of-truth registry for canonical evidence files and historical traces.
- Update `docs/research_plan.md` when the experiment design or recommended launcher policy changes.
- Update `docs/experiment_file_map.md` when canonical results or log paths change.
- Keep generated outputs out of `docs/`.

## Recommended reading order

1. `run_phase123_full_pipeline.sh` for the canonical end-to-end run.
2. `docs/research_plan.md` for the analysis modules, dependency logic, and recommended reproduction order.
3. `docs/experiment_file_map.md` to map scripts to the paper-facing outputs.
4. `results/README.md` and `logs/README.md` for the artifact naming scheme.
5. `run_phase45_diagnostics.sh` and `run_phase6_strategy_audit.sh` for planned diagnostic and strategy-audit analyses.

## Manuscript source

The optional manuscript source lives in `paper/` and is intentionally secondary to the runnable pipeline. Readers who only want to reproduce the experiments can ignore that subtree.