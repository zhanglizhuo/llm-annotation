# Response to Reviewers

**Manuscript ID:** Access-2026-23347
**Title:** When Agreement Fails: Visual Anchoring and Multimodal Pseudo-Label Reliability in Classroom Behavior Recognition
**Journal:** IEEE Access

---

We thank the Associate Editor and both reviewers for their careful reading and constructive feedback. The revised manuscript addresses all seven concerns raised by Reviewer 2. Reviewer 1 raised no technical concerns. Below we respond to each comment in turn, describe the action taken, and indicate where the change appears in the revised manuscript.

All line numbers refer to the revised manuscript with line numbering enabled. Substantive changes are highlighted in the submitted Highlighted PDF.

---

## REVIEWER 1

Reviewer 1 confirmed that the study is well designed and executed, contributions to the body of knowledge are present, the paper is technically sound, and the subject matter is presented in a comprehensive manner. No technical revisions were requested.

**Action:** No changes required in response to Reviewer 1.

---

## REVIEWER 2

### Comment 2.1
> "The abstract is informative but too dense. It should be shortened and re-structured to explicitly outline the problem, the solution, the main findings and the implications for practice."

**Response:** We agree. The original Abstract mixed protocol details, mechanism descriptions, and numerical results in a way that made the central argument hard to follow. We have rewritten the Abstract following a four-part structure: (1) problem and motivation, (2) study design, (3) main findings, and (4) practical implication. The revised Abstract is approximately 30% shorter than the original and leads with the central claim rather than with methodological caveats.

Additionally, the revised manuscript includes two new experiments completed after the original submission:
- (a) A eleven-model cross-model validation (expanded from five models) that tests annotation reliability across Qwen2, Qwen2.5, Qwen3.5, Qwen3.6, LLaVA, Gemma3, and Gemma4 families (7B--35B, dense and MoE).
- (b) A quality-threshold experiment using Qwen3.5-27B pseudo-labels (annotation accuracy 50.3%) for CLIP fine-tuning on TeacherBehavior, identifying the annotation quality range at which pseudo-label fine-tuning begins to surpass the zero-shot baseline.

**Action:** Abstract fully rewritten. New experimental results integrated.

---

### Comment 2.2
> "The explanation of the motivation of using agreement filtering should be more direct. Before highlighting the points of limitations of this filtering approach, the authors should explain the rationale for the filtering strategy in terms of how it is expected to enhance pseudo-label quality."

**Response:** We agree that the original Introduction moved too quickly to the failure modes without first establishing why agreement filtering is an appealing heuristic. We have added a paragraph to the Introduction that explains the intuitive rationale: when two independently queried annotators assign the same label, the probability of both being correct is generally higher than for a single annotator, which is why agreement has become a widely used proxy for pseudo-label reliability in crowdsourcing and semi-supervised learning. This positive motivation is stated before the paper identifies when and why the heuristic breaks down.

**Action:** New paragraph explaining the rationale for agreement filtering added to the Introduction.

---

### Comment 2.3
> "Some methodological details are scattered across different sections in the manuscript. Summary of the annotation pipeline, filtering strategies and fine-tuning of setup would be useful if summarized more compactly."

**Response:** We agree. The revised manuscript adds a Pipeline Overview subsection at the beginning of the Methods section. This subsection presents a concise three-stage summary: (1) bbox cropping and MLLM annotation, (2) filtering strategy comparison, and (3) downstream CLIP fine-tuning evaluation. The individual subsections retain their detail, but a reader can now obtain the full pipeline overview from a single location.

**Action:** New Pipeline Overview subsection added to Methods.

---

### Comment 2.4
> "The concept of 'visual anchoring' is interesting but should be explained more clearly and defined prior to the discussion, and the authors explain that this is not a causal measure but rather used as an operational diagnostic."

**Response:** We agree. The Introduction now contains a dedicated paragraph that formally defines visual anchoring as "the degree to which a category label can be recovered from a single cropped frame without recourse to discourse context, temporal information, or speaker intent." The Introduction explicitly states that the operational proxy (per-category zero-shot CLIP accuracy) is a task-specific diagnostic instrument rather than a causal measurement. The Discussion retains the detailed treatment, which now builds on the definition already established in the Introduction.

**Action:** Formal definition of visual anchoring added to the Introduction. Non-causal framing stated in both Introduction and Discussion.

---

### Comment 2.5
> "There are a number of tables and figures that are helpful but over informative. The authors need to make the text easier to read by emphasizing the most significant comparisons and avoiding repetition."

**Response:** We agree that the original manuscript contained redundant presentation:
- (a) Table 2 (class balance counts) merged into Table 1 as a footnote summary.
- (b) Cross-model tables (original Tables 6 and 7) merged into a single comprehensive table. Per-class breakdowns moved to Supplementary.
- (c) Results section now opens with a one-paragraph summary of the three main quantitative conclusions.
- (d) Numerical values reported once in Results, referenced by table number in Discussion rather than repeated in full.

**Action:** Tables merged; per-class cross-model details moved to Supplementary; Results opens with summary paragraph; Discussion removes repeated values.

---

### Comment 2.6
> "The study is primarily based on internal evaluation using sub-datasets of the SCB. The lack of broader external validation limits the generalizability of the conclusions."

**Response:** We acknowledge this limitation. We address it in three ways in the revised manuscript:
- (a) The expanded eleven-model cross-model validation provides annotator-side external validity. The key finding---that intent-dependent categories remain near-zero across all annotator families while visually anchored categories remain reliable---replicates across Qwen2, Qwen2.5, Qwen3.5, Qwen3.6, LLaVA, and Gemma4 families.
- (b) The new Qwen3.5-27B fine-tuning experiment provides partial external evidence for the quality-threshold claim, confirming that downstream behavior tracks annotator quality rather than being fixed by dataset properties.
- (c) An explicit paragraph in the Limitations section acknowledges the single-benchmark constraint and distinguishes dataset-dependent findings (exact retention-ratio thresholds) from those more likely to generalize (visual anchoring pattern, agreement filtering failure modes).

**Action:** Limitations updated. Cross-model section reframed as annotator-side external validity. Quality-threshold experiment added.

---

### Comment 2.7
> "It is recommended to carefully polish the language in the manuscript, particularly in long paragraphs containing the main argument, which becomes difficult to understand."

**Response:** We have undertaken a systematic language revision:
- (a) Long paragraphs in Introduction, Discussion (Finding 1, Finding 2), and Limitations broken into shorter units with clear topic sentences.
- (b) Abstract rewritten with shorter sentences and active constructions.
- (c) Hedged compound sentences replaced by direct statements followed by separate qualifications.
- (d) Technical terms (agreement filtering, visual anchoring, retention ratio, anchor proxy) defined at first use.
- (e) Conclusions tightened to three focused paragraphs.

**Action:** Language polished throughout.

---

## ADDITIONAL CHANGES NOT REQUESTED BY REVIEWERS

1. **Ten-model cross-model validation** (expanded from five models). Key findings: (a) generational boundary on *guide* (pre-2024 models all <25%, new models 39--83%); (b) *answer* remains <23% universally; (c) no single annotator is uniformly best.

2. **Quality-threshold experiment** with Qwen3.5-27B pseudo-labels. Identifies the critical annotation accuracy range (41--50%) at which pseudo-label fine-tuning crosses the zero-shot baseline on TeacherBehavior.

3. **GitHub repository publicly accessible** at https://github.com/zhanglizhuo/llm-annotation with environment specs, dependency instructions, and one-command run examples.

---

## SUMMARY OF ALL CHANGES

| Section | Change |
|---------|--------|
| Abstract | Fully rewritten; 4-part structure; 30% shorter; new results integrated |
| Introduction | New paragraph: agreement filtering rationale; formal visual anchoring definition |
| Methods | New Pipeline Overview subsection; cross-model expanded to 10 models |
| Results | Opening summary paragraph; tables merged; 10-model cross-model table; Qwen3.5-27B quality-threshold subsection |
| Discussion | Finding 3 updated with quality-threshold evidence; Finding 5 updated; language polished |
| Limitations | Updated with annotator-side external validity and quality-threshold evidence |
| Conclusions | Critical threshold, generational boundary, and practical annotator selection guidance added |
| Cross-model | Validation expanded from 5 to 11 models; new tables and analysis added |
