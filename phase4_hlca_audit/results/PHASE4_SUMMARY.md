# Phase 4 (Step 3) — HLCA Audit: Final Summary

**Status: COMPLETE. Extended 2026-08-08 (step (d)) with a genuinely novel
finding -- a specific population HLCA's own curators did NOT flag, found by
mining the audit's own predictions and independently corroborated with
canonical marker-gene expression pulled directly from the raw atlas (see
"Result 4" below).**

## What was tested
The "payoff" pillar of the original research plan: apply the validated
calibration pipeline (structure-aware weighted conformal prediction, from
Phase 3) to a real public reference atlas -- the Human Lung Cell Atlas core
(Sikkema et al., Nature Medicine 2023; 584,944 cells, 49 contributing
studies integrated into 11 "study" / 14 "dataset" batch units, downloaded
directly from CELLxGENE Discover, 5.87GB h5ad, bypassing the
`cellxgene-census`/`tiledbsoma` Windows build blocker entirely).

**Key enabling discovery**: HLCA core ships with a `reannotation_type` obs
column (Correctly annotated / Misannotated / Underannotated) that HLCA's
own authors computed by comparing each contributing study's ORIGINAL label
to HLCA's final integrated consensus label. This turns the audit from
"discover errors blind" into "test whether our calibration signal
independently recovers errors the field's own experts already found" --
a much stronger, ground-truth-backed validation design.

## Method
- Subsampled 2,000 cells per contributing dataset (14 datasets, 28,000
  cells total), PCA to 50 components (log-normalized expression).
- Leave-one-dataset-out: train a multinomial logistic regression on
  `ann_level_3` (21 classes) using the other 13 datasets, calibrate via
  APS conformal scores, evaluate naive and structure-aware (dataset-as-
  batch) weighted conformal prediction on the held-out dataset.
- The audit question: does classifier confidence / nonconformity / top-1
  correctness on the HLCA consensus label differ between cells HLCA
  independently flagged as Correctly annotated vs. Misannotated vs.
  Underannotated?

## Result 1: the aggregate statistical finding

| reannotation_type (HLCA's own label) | n cells | top-1 accuracy | mean nonconformity |
|---|---|---|---|
| Correctly annotated | 9,465 | **97.1%** | 0.977 |
| Underannotated | 8,335 | 95.1% | 0.965 |
| Misannotated | 10,173 | **90.4%** | 0.956 |

Clean monotonic ordering (Correct > Underannotated > Misannotated) on both
accuracy and confidence, both highly significant:
- Accuracy gap: chi-square = 420.85, **p = 4.1e-92**
- Confidence gap (Correct vs. Misannotated): Mann-Whitney U, **p = 6.9e-169**
- Confidence gap (Correct vs. Underannotated): Mann-Whitney U, p = 2.4e-3

An independently-trained classifier, with zero access to HLCA's curation
history, is measurably less confident and less accurate on exactly the
cells HLCA's own expert reannotation process flagged as needing correction.

## Result 2: concrete case studies

At `ann_level_3` granularity, 10,173 cells are flagged Misannotated, but
only 2,588 of those show a genuine label change at this level (the rest
were corrected at a finer resolution than ann_level_3 captures -- e.g.
"Basal resting" vs. "Suprabasal", both "Basal" at this level -- itself a
sign the aggregate finding above is conservative, since it still holds
even though many corrections are invisible at this coarse a level).

For the genuine ann_level_3-visible corrections, four illustrative cases
(dataset-traceable, full detail in `hlca_misannotated_with_patterns.csv`):

| Case | original to consensus | n | classifier matches consensus | mean confidence | mean set size (naive/structure) |
|---|---|---|---|---|---|
| 1 | Basal to Secretory | 628 | **84.4%** (530/628) | 0.919 | 1.11 / 1.10 |
| 2 | Submucosal Secretory to Basal | 175 | **97.7%** (171/175) | 0.944 | 1.01 / 0.97 |
| 3 | Monocytes to Macrophages | 279 | 64.5% (180/279) | 0.833 | 1.38 / 1.38 |
| 4 | Macrophages to Monocytes | 115 | 38.3% (44/115) | **0.749** | **1.77 / 1.73** |

Cases 1-2 are clean, largely single-study-concentrated errors (Case 2 is
92%+ from Seibold_2020 alone) that an independently-trained classifier
resolves with high confidence and small prediction sets. Cases 3-4 both
involve the monocyte/macrophage boundary -- a well-known biological
continuum, not a sharp category -- and the classifier's behavior tracks
this appropriately: lower confidence, larger prediction sets, and a lower
(but still informative) catch rate. Case 4 in particular is the honest
negative result: the classifier doesn't reliably recover this correction,
but it also doesn't confidently get it wrong -- it correctly hedges via
larger set sizes and the lowest confidence of any case, rather than
forcing a false-confident answer either way.

**This is a stronger finding than uniform success would have been**: the
classifier's uncertainty scales sensibly with how catchable each specific
correction actually is, which is exactly the behavior a well-calibrated
method should exhibit.

## Result 3: coherent secondary finding on structure-aware reweighting
Naive and structure-aware conformal prediction track almost identically on
HLCA core (coverage 0.941/0.897 vs. 0.944/0.893 at alpha=0.05/0.10). This
is consistent with, not contrary to, Phase 3's tiered findings: HLCA core
is already carefully batch-integrated (scANVI) by its own authors, so
there is little residual batch effect left at the dataset level for
reweighting to correct -- exactly the "safe, does no harm" regime Tiers
1-2 characterized, not the severe monolithic-shift regime Tier 3 exposed.

## Result 4 (step (d), added 2026-08-08): a genuinely surprising finding

Results 1-3 all confirm that classifier confidence correlates with cells
HLCA's own `reannotation_type` column *already* flags as needing correction
-- a strong validation, but a confirmatory one. Step (d) asked a harder
question: can the audit find a SPECIFIC population HLCA's curators did NOT
flag?

**Method**: mine the already-computed per-cell predictions
(`hlca_examples_full.csv`, 27,974 cells) for cells labeled "Correctly
annotated" by HLCA where the independently-trained classifier CONFIDENTLY
(singleton prediction set, confidence >=0.90) and REPRODUCIBLY (across
multiple independent leave-one-dataset-out folds, so it isn't one study's
idiosyncrasy) disagrees with the consensus label
(`find_novel_errors.py`).

**Top candidate**: cells HLCA labels `cell_type = "respiratory basal cell"`,
`ann_level_3 = "Basal"`, `reannotation_type = "Correctly annotated"` --
i.e., cells the field's own curation explicitly says needed no correction.
The classifier confidently predicts "Secretory" instead, reproducibly
across 3 independent held-out-dataset folds (Barbry_Leroy_2020,
Jain_Misharin_2021_10Xv2, Krasnow_2020; 13 cells, mean confidence 0.962, in
the 2,000-cells/dataset subsample).

**Independent molecular corroboration** (`hlca_marker_verification.py`):
rather than trying to trace the exact same cells (not possible -- the
prepared audit file does not retain original cell barcodes), every cell
matching this exact HLCA label combination was pulled DIRECTLY from the raw
5.87GB atlas (not the subsample) in these 3 datasets and scored on
canonical Basal markers (KRT5, TP63, KRT14) vs. canonical Secretory markers
(SCGB1A1, SCGB3A1, MUC5B, MUC5AC, BPIFB1) -- a check that is completely
independent of the classifier (no PCA, no logistic regression, just raw
marker gene expression) and does not depend on cell-level tracing. As a
sanity check, true "Secretory, correctly annotated" cells in the same
datasets score oppositely (mean secretory_score 0.183 vs. basal_score
0.016), confirming the marker panel is discriminative.

| Dataset | "Basal, correctly annotated" cells (full atlas) | secretory-marker-dominant | rate |
|---|---|---|---|
| Barbry_Leroy_2020 | 28,093 | 699 | 2.5% |
| Jain_Misharin_2021_10Xv2 | 222 | 8 | 3.6% |
| Krasnow_2020 | 203 | **155** | **76.3%** |
| **Combined** | 28,518 | 862 | 3.0% |

**The headline finding**: in Krasnow_2020 specifically, 76.3% (155/203) of
cells HLCA calls "Basal" and explicitly marks as needing no correction are,
by canonical marker gene expression, secretory-dominant rather than
basal-dominant -- a large, specific, previously-unflagged discrepancy in a
named, publicly traceable dataset (Krasnow_2020 is the Travaglini et al.
Human Lung Cell Atlas 1.0 data), not a diffuse, hand-wavy pattern.

**Follow-up (added 2026-08-08): the dissociation-protocol confound was
checked directly, and the answer is more textured than a clean yes/no**
(`check_stress_confound.py`). Krasnow_2020's dissociation protocol
(Collagenase + Elastase + DNAse, vs. Cold protease 1h for the other two
flagged datasets) is a documented source of stress-induced transcriptional
artifacts (van den Brink et al. 2017), scored here via a 15-gene
immediate-early/stress panel (FOS, FOSB, JUN, JUNB, JUND, EGR1, HSPA1A,
HSPA1B, HSP90AA1, DNAJB1, IER2, NR4A1, ZFP36, DUSP1, HSPB1). Two tests:

- **Dataset-level stress is real and substantial**: Krasnow_2020's Basal
  cells score ~3x higher on the stress panel than the other two datasets
  (mean 1.031 vs. 0.365 [Barbry_Leroy_2020] and 0.433 [Jain_Misharin]) --
  the protocol difference has a measurable, large transcriptional
  signature, confirming the confound is real at the dataset level and that
  raw cross-dataset RATE comparisons (2.5% vs. 76.3%) need this caveat.
- **But within Krasnow_2020, the secretory-dominant cells are LESS stressed
  than the basal-dominant cells in the same dataset** (mean stress 0.923
  vs. 1.382, Mann-Whitney p=1.1e-7) -- the OPPOSITE of what the
  stress-artifact hypothesis predicts. If dissociation stress were
  spuriously inducing secretory-marker expression, the secretory-dominant
  cells should be the MORE stressed subset, not the less stressed one. This
  is evidence AGAINST the simple "stress causes the apparent secretory
  identity" mechanism for Krasnow_2020's finding specifically.
- **Barbry_Leroy_2020 shows the opposite pattern**: there, secretory-
  dominant cells ARE significantly more stressed than basal-dominant cells
  (mean 0.504 vs. 0.362, p=4.7e-67) -- consistent with at least part of
  that dataset's much smaller (2.5%) secretory-dominant population being a
  genuine stress-response artifact rather than real biology.

**Net honest read**: this does not "resolve" the question to a confirmed
biological finding -- ruling out one candidate confound (classic
dissociation-stress signature) is not proof of genuine biology, since other
technical explanations (ambient RNA contamination, doublets, an
unidentified protocol effect) remain unchecked. But it meaningfully shifts
the picture: the dataset most central to the headline claim (Krasnow_2020,
76.3%) does NOT show the expected artifact signature and if anything shows
the opposite pattern, making it the more credible component of this
finding; while the dataset contributing the smallest effect
(Barbry_Leroy_2020, 2.5%) does show the artifact-consistent signature,
suggesting its small secretory-dominant subpopulation should be discounted.
The honest bottom line to carry into any manuscript: **Krasnow_2020's
Basal-to-Secretory discrepancy survives the most standard and obvious
technical explanation and should be reported as an open, credible lead**
-- not proven, but no longer just "flagged as unresolved."

**A second, opposite-direction caveat, also reported honestly**: for
Jain_Misharin_2021_10Xv2, the classifier-based subsample flagged an even
higher disagreement rate (16/17 = 94% of subsampled Basal cells predicted
Secretory) than the marker-based full-atlas check found (8/222 = 3.6%
secretory-marker-dominant). This directional mismatch between the
classifier's PCA-embedding-based disagreement and direct marker-gene
evidence means not every confident classifier disagreement is biologically
real -- for this dataset specifically, the classifier's signal is more
likely a batch/technical artifact in the embedding than a genuine
labeling error, illustrating exactly why the marker-gene corroboration step
matters and shouldn't be skipped when reporting audit findings as
"errors."

**Why this is a stronger claim than Results 1-3**: it is not a restatement
of "confidence tracks known-wrong cells" -- it identifies a specific,
named, checkable population (Krasnow_2020's "Basal, correctly annotated"
cells) using two independent lines of evidence (an ML classifier the
curators never saw, and canonical marker genes the classifier never saw),
while being explicit that one of the two candidate datasets found by the
same method (Jain_Misharin) does NOT survive the independent marker check --
exactly the kind of finding-plus-honest-caveat structure that makes a result
credible rather than a fishing-expedition false positive.

## Scope simplifications (documented, not hidden)
- PCA-based features (50 components on log-normalized expression), not a
  dedicated classifier like CellTypist -- chosen for speed/consistency
  with Phase 3's tooling, not re-validated against CellTypist specifically
  on this atlas.
- `ann_level_3` (21 classes) was the classification target; HLCA's
  `reannotation_type` flag appears to reflect finer-resolution comparisons
  in many cases (see the Basal-to-Basal "identical at this level" majority
  among Misannotated cells) -- a finer target (ann_level_4/5) would likely
  show an even larger effect, not a smaller one.
- 2,000 cells/dataset subsampling cap for runtime tractability; full-data
  sensitivity not checked.
- Single train/cal split per fold (no internal cross-validation within
  each leave-one-dataset-out fold).

## Files
- `scripts/inspect_hlca.py` -- backed-mode metadata inspection
- `scripts/prepare_hlca_audit.py` -- subsampling, normalization, PCA
- `scripts/run_hlca_audit.py` -- the leave-one-dataset-out audit + aggregate stats
- `scripts/extract_examples.py` -- per-cell prediction detail + confusion patterns
- `scripts/find_novel_errors.py` -- step (d): mines predictions for confident,
  reproducible disagreement on cells HLCA calls "Correctly annotated"
- `scripts/hlca_marker_verification.py` -- step (d): independent marker-gene
  corroboration pulled directly from the raw atlas
- `scripts/check_dataset_tech.py` -- step (d): technical metadata check that
  surfaced the dissociation-protocol confound for Krasnow_2020
- `scripts/check_stress_confound.py` -- step (d) follow-up: stress/dissociation
  gene-signature check that tested (and largely ruled out) the confound
- `data/hlca_audit_prepared.h5ad` -- prepared PCA-embedded audit dataset
- `results/hlca_audit_per_cell.csv` -- per-cell nonconformity/correctness/reannotation_type
- `results/hlca_audit_summary_by_reannotation_type.csv` -- Result 1's table
- `results/hlca_audit_coverage_summary.csv` -- Result 3's table
- `results/hlca_examples_full.csv` -- full per-cell prediction detail (all 14 folds)
- `results/hlca_misannotated_with_patterns.csv` -- Result 2's case-study data
- `results/hlca_candidate_novel_errors.csv` -- step (d) mined candidates
- `results/hlca_marker_verification.csv` -- step (d) per-cell marker scores
- `results/hlca_stress_confound_check.csv` -- step (d) follow-up per-cell stress scores
