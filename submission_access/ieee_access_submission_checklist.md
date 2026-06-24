# IEEE Access Submission Checklist

## Ready Files

- Main manuscript source: `ACCESS_latex_template_20240429/llm_annotation_paper_access.tex`
- Main manuscript PDF: `ACCESS_latex_template_20240429/llm_annotation_paper_access.pdf`
- Cover letter source: `cover_letter_access.tex`
- Cover letter PDF: `cover_letter_access.pdf`
- IEEE Access class/template files: include the full `ACCESS_latex_template_20240429/` source support files when uploading LaTeX source.

## Required Source Assets

- `ieeeaccess.cls`
- `IEEEtran.cls`
- `IEEEtran.bst`
- `spotcolor.sty`
- all `.fd`, `.tfm`, `.pfb`, and `.map` font files in `ACCESS_latex_template_20240429/`
- figures referenced by the manuscript:
  - `fig_pipeline_access.pdf`
  - `fig_visual_anchoring_access.pdf`
  - `fig_visual_anchoring_examples_access.pdf`
  - `fig_all_results_overview_access.pdf`
  - `fig_distribution_shift_access.pdf`
  - `fig_retention_curve_access.pdf`

## Verified On 2026-05-16

- Main manuscript compiles successfully with `pdflatex`.
- Abstract length: 211 words.
- Keywords: 4.
- Missing graphics: none detected.
- Missing bibliography keys: none detected.
- Undefined references/citations/control sequences: none detected.
- Forced `[H]` floats: 0.
- Privacy-sensitive visual examples are pixelated.

## Manual Checks Before Upload

- Confirm the final PDF pages show figures/tables in acceptable positions after `[!t]` float placement.
- Confirm IEEE Author Portal does not require author biographies at initial submission. If required, add `IEEEbiographynophoto` entries.
- `\history{}` and `\doi{}` are intentionally empty for initial submission; fill placeholder text only if the portal/template checker requires it.
- Ensure the GitHub repository in the Data Availability statement is public and includes environment/dependency instructions.
- Confirm all authors approve the submitted manuscript and cover letter.

## Recommended Cover Letter Framing

Use the manuscript as a reliability-audit paper rather than a single-dataset failure report:

> This manuscript presents a reproducible reliability-audit protocol for assessing when MLLM-generated pseudo-labels provide useful supervision for fine-grained classroom behavior recognition, and shows that cross-model agreement is not a portable denoising rule under single-frame classroom evidence.
