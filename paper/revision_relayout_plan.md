# Main-Text and Supplementary Relayout Draft

This draft proposes a conservative relayout for Electronics revision that improves readability without removing evidence.

## Goal

- Reduce main-text density for engineering readers.
- Keep all core claims auditable.
- Move highly detailed diagnostics to supplementary material.

## Keep in Main Text

- Table 1 equivalent main result matrix (`tab:main_results`): keep.
- Main retention curve figure (`fig:retention_curve`): keep.
- Core annotation-quality table (`tab:annotation_quality`): keep.
- Selective routing summary (`tab:selective_results`): keep.
- Key methodological protocol table (`tab:seed_policy`): keep.

## Move to Supplementary

- BowTurnHead confusion-count table (`tab:bow_confusion`) -> Supplementary Table S1.
- Full class-balance expansion (`tab:class_balance`) -> Supplementary Table S2.
- Extended intent-class cross-model numeric breakdown (currently discussed in text) -> Supplementary Table S3.
- Any per-model pairwise agreement detail grids -> Supplementary Table S4.

## Text Changes Required in Main

- Replace full confusion table discussion with a short pointer:
  - "Detailed confusion counts are reported in Supplementary Table S1."
- Replace extended class-balance narrative with a short pointer:
  - "Full class-count breakdown is listed in Supplementary Table S2."
- Keep only decision-relevant statistics in Discussion and move dense supporting numbers to supplementary tables.

## Expected Benefit

- Main narrative becomes shorter and more decision-focused.
- Core claims remain fully reproducible via supplementary materials.
- Better alignment with Electronics reviewer preference for engineering readability.

## Implementation Order

1. Create supplementary file with S1-S4 tables.
2. Move table environments from main to supplementary.
3. Update in-text references in main.
4. Recompile and verify all cross-references.
