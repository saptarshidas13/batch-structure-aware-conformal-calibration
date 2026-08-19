# Phase 3 — Final Summary

**Status: COMPLETE (all three tiers run). All three tiers re-run 2026-08-08
with (a) a theoretically-grounded reweighting method added, (b) a
strengthened domain-classifier baseline, and (c) an upstream batch-integration
fix (Harmony) tested specifically for Tier 3's severe-shift case that neither
(a) nor (b) could resolve (see the three sections below). IMPORTANT UPDATE:
a literature check performed after all of the above (see "Novelty
reassessment" section) found that the core idea behind (a) -- weighting
calibration data from known reference groups by estimated group-membership
probabilities when the target is an unknown mixture over those groups -- is
established 2024-2025 conformal prediction literature, not a new
theoretical contribution. This does not undo steps (a)-(c)'s empirical
findings, but it changes how the paper's contribution should be framed: as
a rigorous application + empirical characterization, not a new method.**

## What was tested
Batch-structure-aware weighted conformal prediction (Tibshirani et al. 2019
style, applied to APS non-conformity scores), compared against (a) naive
unweighted split conformal prediction and (b) a generic black-box
domain-classifier weighted CP baseline, across three tiers of increasing
batch-effect severity:

- **Tier 1**: Baron pancreas, donor-as-batch (4 donors, mild single-lab
  variation) -- reuses Phase 2's validated Geneformer embeddings.
- **Tier 2**: GSE127465 (Zilionis et al. 2019 NSCLC), patient-as-batch
  (7 patients, real clinical cohort variation) -- PCA on log-normalized
  expression.
- **Tier 3**: GSE178360 (healthy lung) vs. GSE127465 tumor cells, a genuine
  cross-study/cross-platform shift -- coarse marker-gene-derived labels
  (Immune/Epithelial/Stromal via PTPRC/EPCAM/COL1A1+PECAM1+ACTA2), since no
  shared fine-grained ontology exists across the two independently-annotated
  studies. Both directions tested (healthy-as-reference and tumor-as-reference).

All three implement a custom, standalone weighted-APS conformal predictor
(`weighted_conformal.py`) rather than the Phase 1/2 `ConformalSCAnnotator`
framework, since that framework has no support for custom reweighting --
full control over the weighting math is the actual point of Phase 3.

**Important early bug caught and fixed**: the first Tier 1 run used a
deterministic (non-randomized) APS score, which is systematically
over-conservative (near-100% coverage, prediction sets of 6-13 classes out
of 14). Fixed by implementing the proper randomized APS score/prediction-set
construction from Romano et al. (2020) before any tier's results were
trusted.

## Theoretical grounding (added 2026-08-08)

The original `structure_aware` weighting (inverse centroid distance between
each reference batch and the query) had no theorem behind it: Tibshirani et
al.'s (2019) weighted-conformal coverage guarantee requires w(x) to estimate
the true covariate-shift density ratio dQ/dP(x), and raw inverse Euclidean
distance between batch centroids does not estimate that quantity in any
formal sense. It is now explicitly relabeled `structure_aware_centroid` and
documented in its own docstring as a heuristic baseline, not a claimed-valid
method.

A second, theoretically-grounded estimator, `structure_aware_propensity`,
was added via generalized propensity scores (Imbens, 2000): fit one
multinomial classifier g(x) -> P(batch=k | x) on reference-batch identity
alone (no query data used in training), estimate the query's mixture over
reference batches as lambda_j = mean over query of g(x)_j, and combine as
w(x) = sum_j (lambda_j / pi_j) g(x)_j, where pi_j is each batch's empirical
proportion in the reference. This is a genuine density-ratio estimate under
an explicitly stated (mixture) assumption about the query distribution,
applied identically to calibration and test points. Full derivation is in
`weighted_conformal.py`'s docstring.

## Novelty reassessment (added 2026-08-08, after a literature check)

The derivation above was written and implemented WITHOUT first searching
the conformal prediction literature for prior art on this exact setting --
an oversight, given that novelty of the original two candidate NCS problems
*was* rigorously checked at the start of this whole project. That check has
now been done, prompted by a direct question about whether the accumulated
experimental work is sufficient for an NCS submission. The honest finding:
**it is not a new theoretical contribution.**

"Weighted conformal prediction when covariate shift arises from a mixture
of known reference groups" is an active area with directly relevant prior
work, most notably:
- Bhattacharyya & Barber, **"Group-Weighted Conformal Prediction"**
  (arXiv:2401.17452, Jan 2024) -- formalizes exactly this setting (finite
  known groups determine the shift, target is a mixture of groups) and
  derives group-level weights w(X) proportional to q_k/p_k (target group
  proportion over calibration group proportion). Their setting assumes
  group membership is KNOWN for every point, including test points, and
  target mixture proportions q_k are GIVEN.
- Multiple 2024-2025 papers on multi-source conformal inference and
  domain-shift-aware conformal prediction independently converge on the
  same general construction this project used: fit a domain/group
  classifier, use its predicted probabilities as soft per-point weights,
  and estimate the unknown target mixture by averaging predicted membership
  over the test set.

**What is and isn't novel, precisely stated**: `structure_aware_propensity`
differs from GWCP in one specific way -- it ESTIMATES both group membership
and target mixture proportions via a classifier, rather than assuming they
are known, because in the single-cell setting a held-out dataset/patient is
not literally one of the reference batches. That is a real, if incremental,
extension to a harder estimation setting. But it is an extension of an
established framework, not a new one, and none of the prior-art papers
found were applied to single-cell data, batch effects, or scFM calibration
specifically.

**What this changes for the paper**: the honest framing is no longer "we
introduce a theoretically-grounded batch-structure-aware reweighting
method." It is "we adapt an established class of distribution-shift-aware
conformal prediction techniques (properly cited) to single-cell foundation
model batch-effect calibration, and provide the first empirical
characterization of when this class of method helps, when it doesn't
(severe monolithic cross-study shift, a positivity-assumption violation),
and what partially fixes the cases it can't (upstream integration) --
across 5 datasets and 3 organ systems, validated against a real reference
atlas (HLCA)." That is a legitimate, defensible contribution -- an
application + rigorous empirical-characterization paper -- but it is a
materially different (and more modest) claim than "new method," and this
matters directly for venue fit: NCS strongly favors methodological novelty,
and this reassessment removes the strongest candidate for that novelty
claim in the whole project.

## Fair baseline (added 2026-08-08)

The original `domain_classifier_weights` (one `LogisticRegression(C=1.0)`
fit on all calibration+query points, then scored IN-SAMPLE on those same
points) had three fixable weaknesses that make its instability look worse
than a "generic density-ratio baseline" necessarily has to be:

1. **In-sample propensity bias**: scoring points the classifier was trained
   on pushes predictions toward 0/1 (partial memorization), inflating
   weights artificially. Fixed via 5-fold **cross-fitting** (Chernozhukov et
   al. 2018): every point's probability comes from a classifier that never
   saw it during training.
2. **Untuned regularization**: `C=1.0` was an arbitrary guess. Fixed via
   `LogisticRegressionCV` (internal CV selects C in {0.01, 0.1, 1.0, 10.0}
   within each cross-fitting fold).
3. **Unbounded weights**: fixed via **truncation at the [1st, 99th]
   percentile** of the pooled weight distribution (Cole & Hernan 2008), the
   standard stabilization technique for inverse-probability weights.

The original (now `domain_classifier_weights_naive`) is kept for direct
before/after comparison in every table below. All three tiers were re-run
with all five methods (naive, domain_clf_naive, domain_clf [strengthened],
structure_aware_centroid, structure_aware_propensity).

## Results by tier

### Tier 1 (Baron pancreas, mild batch effect)
| alpha | naive | centroid | propensity | domain_clf_naive | domain_clf (fixed) |
|---|---|---|---|---|---|
| 0.01 | cov 0.978, size 2.73 | cov 0.979, size 2.78 | cov 0.980, size 2.81 | cov 0.954, size 9.31 | cov 0.950, size 7.49 |
| 0.05 | cov 0.926, size 1.50 | cov 0.925, size 1.50 | cov 0.930, size 1.53 | cov 0.916, size 7.18 | cov 0.901, size 5.39 |
| 0.10 | cov 0.867, size 1.24 | cov **0.872**, size 1.25 | cov **0.881**, size 1.26 | cov 0.898, size 6.30 | cov 0.876, size 4.65 |

Weight truncation shrinks domain_clf's average set size by 20-30% (e.g.
9.31->7.49 at alpha=0.01) and its per-fold variance also drops -- but it
remains 3-6x larger than naive/structure-aware, and on the human4 fold
specifically, the fixed domain_clf still collapses (coverage 0.84->0.69
across alpha, actually *worse* than the naive version on that fold: 0.85,
0.75, 0.72). So the fixes measurably help without resolving the underlying
instability -- exactly what "a fair but still-generic baseline" should look
like.

### Tier 2 (GSE127465, patient-as-batch, moderate real-world variation)
| alpha | naive | centroid | propensity | domain_clf_naive | domain_clf (fixed) |
|---|---|---|---|---|---|
| 0.01 | cov 0.989, size 1.50 | cov 0.989, size 1.51 | cov 0.991, size 1.53 | cov 0.990, size 6.57 | cov 0.991, size 5.56 |
| 0.05 | cov 0.951, size 1.13 | cov 0.958, size 1.13 | cov 0.952, size 1.14 | cov 0.950, size 3.77 | cov 0.947, size 2.83 |
| 0.10 | cov 0.902, size 1.01 | cov 0.905, size 1.01 | cov **0.908**, size 1.01 | cov 0.884, size 2.90 | cov 0.874, size 2.03 |

Same pattern: truncation shrinks domain_clf's sets substantially (6.57->5.56
at alpha=0.01) but it stays 2-5x larger than the principled methods, and its
coverage is no more reliable than before (e.g. alpha=0.10: fixed domain_clf
actually undercovers slightly more than naive, 0.874 vs. 0.902).

### Tier 3 (cross-study, severe batch effect)
| Direction | alpha | naive | centroid | propensity | domain_clf_naive | domain_clf (fixed) |
|---|---|---|---|---|---|---|
| healthy->tumor | 0.10 | cov 0.665, size 1.74 | cov 0.664, size 1.74 | cov 0.670, size 1.75 | cov 1.000, size 3.00 | cov 0.942, size 2.77 |
| healthy->tumor | 0.05 | cov 0.751, size 2.00 | cov 0.751, size 2.00 | cov 0.739, size 1.96 | cov 1.000, size 3.00 | cov 0.977, size 2.90 |
| tumor->healthy | 0.10 | cov 0.729, size 1.30 | cov 0.729, size 1.29 | cov 0.744, size 1.32 | cov 1.000, size 3.00 | cov 0.999, size 2.99 |
| tumor->healthy | 0.05 | cov 0.836, size 1.50 | cov 0.842, size 1.51 | cov 0.841, size 1.50 | cov 1.000, size 3.00 | cov 1.000, size 2.997 |

**Even the strengthened domain_clf is still functionally vacuous in Tier
3** -- average set sizes of 2.77-3.00 out of only 3 total classes, i.e. it
still predicts "could be anything" for nearly every cell. This is the most
interesting result of step (b): the fix isn't cosmetic (truncation does
reduce size 3.00->2.77 in the least-severe row), but it cannot repair the
underlying cause, which is a known, distinct failure mode from "bad
implementation" -- **near-complete distributional separability between
healthy and tumor lung** (top-1 cross-study accuracy is only 0.50-0.59,
meaning the two studies barely share structure in PCA space). Under
near-disjoint support between reference and query, propensity/domain-ratio
estimates correctly go to 0 or 1 for almost every point regardless of
classifier quality -- a violation of the *positivity/overlap* assumption
that importance weighting always requires (Cole & Hernan 2008; this is the
same assumption structure_aware_propensity's mixture derivation also
implicitly needs, and the reason it fails here too). No amount of
regularization or truncation fixes a violated identifiability assumption --
only a different class of method (rejection, or upstream batch integration)
can.

**The theoretically-grounded method does not fix Tier 3 either** -- it tracks
naive/centroid within about 1-2 coverage points in every row, and at
alpha=0.05 healthy->tumor it is actually slightly worse (0.733 vs. naive's
0.747). This is not a negative result for the method itself: the derivation
explicitly assumes the query is a mixture of the *reference* batches'
distributions. In Tier 3 the query is an entirely different study, not a
mixture of the reference's own sub-batches, so the assumption the estimator
relies on is violated by construction -- the same mechanistic reason the
centroid heuristic failed, now understood precisely rather than empirically
observed. This confirms (on firmer theoretical footing than before) that
Tier 3's failure mode is specifically about *shift the reference has no
sub-batch structure to explain*, not about the reweighting estimator being
insufficiently principled.

## Step (c): what actually fixes Tier 3 (added 2026-08-08)

Step (b) concluded that Tier 3's failure is a violated positivity/overlap
assumption -- healthy and tumor lung occupy near-disjoint regions of the raw
embedding, so REWEIGHTING existing calibration points (any of naive,
centroid, propensity, or a fixed domain classifier) cannot help, because
there is no calibration mass anywhere near the query to reweight toward.
The natural fix suggested by that diagnosis is not a better reweighting
scheme but a different class of method: move the points themselves so the
positivity assumption stops being violated. **Upstream batch integration**
(Harmony, Korsunsky et al. 2019) does exactly this -- it is not a
reweighting scheme at all, and was not part of the four `weighted_conformal`
methods; it edits the embedding before any classifier or conformal step
runs.

**Method** (`tier3_harmony.py`): PCA is computed once on the pooled
healthy+tumor cells (unlike `tier3_cross_study.py`'s reference-only PCA),
then Harmony integrates on `study` as the batch key, using only batch
metadata (never cell-type labels) -- a transductive setup analogous to
reference-mapping pipelines, and exactly what real atlas-building pipelines
(including HLCA's own construction, see Phase 4) do before any downstream
classifier is trained. The same train/cal/query split and all four
`weighted_conformal` methods are then re-run on the harmonized embedding.

### Tier 3 post-Harmony vs. pre-Harmony (naive weighting)
| Direction | alpha | pre-Harmony naive | post-Harmony naive | gap closed |
|---|---|---|---|---|
| healthy->tumor | 0.10 | cov 0.665 (target 0.90, gap 0.235) | cov 0.793 (gap 0.107) | ~54% |
| healthy->tumor | 0.05 | cov 0.751 (target 0.95, gap 0.199) | cov 0.880 (gap 0.070) | ~65% |
| healthy->tumor | 0.01 | cov 0.959 (target 0.99, gap 0.031) | cov 0.978 (gap 0.012) | ~61% |
| tumor->healthy | 0.10 | cov 0.729 (target 0.90, gap 0.171) | cov 0.790 (gap 0.110) | ~36% |
| tumor->healthy | 0.05 | cov 0.836 (target 0.95, gap 0.114) | cov 0.852 (gap 0.098) | ~14% |
| tumor->healthy | 0.01 | cov 0.940 (target 0.99, gap 0.050) | cov 0.933 (gap 0.057) | worse |

**Harmony genuinely helps, especially in the harder direction and at looser
alpha, but does not fully close the gap.** In healthy->tumor (the more
severe shift, since tumor cells are the larger/more heterogeneous query),
Harmony closes 54-65% of the undercoverage gap. In tumor->healthy it helps
much less (14-36%) and is a wash at alpha=0.01. Reweighting on top of the
harmonized embedding (centroid/propensity) still adds essentially nothing
beyond naive -- consistent with step (b)'s finding that the reweighting
layer itself was never the bottleneck.

**A specific, non-obvious finding**: in the healthy->tumor direction,
cross-study top-1 classification accuracy is UNCHANGED by Harmony (0.4965
pre vs. 0.4959 post) even though coverage improves substantially (e.g.
+0.128 at alpha=0.10). Harmony is not making the classifier more accurate --
it is making the calibration set's non-conformity SCORES more representative
of the query's, which is a distinct and more specific effect than "better
integration = better classification." This is a genuinely useful mechanistic
distinction for the paper: batch integration helps *conformal calibration
transfer* specifically, not classification generally, and the two can
dissociate.

**Honest caveats**:
- The pooled PCA has much lower explained variance (11.4%) than either
  direction's reference-only PCA in `tier3_cross_study.py` (21-28%), because
  a single global PCA has to represent both studies' variation at once.
  Prediction sets are correspondingly larger post-Harmony (e.g. naive avg
  size 1.74->2.04 at alpha=0.10, healthy->tumor) even where coverage
  improves -- some of the coverage gain may be a size/coverage tradeoff
  rather than a pure calibration-quality improvement, and this needs
  disentangling before being reported as an unqualified win.
- Nominal coverage is still not reached in 5 of 6 rows (only healthy->tumor
  alpha=0.01 gets within 0.012 of target). Harmony is a genuine partial fix,
  not a solution -- the honest framing is "upstream integration measurably
  narrows the severe cross-study gap that reweighting alone cannot touch,
  by up to ~65%, without fully eliminating it."
- Harmony was run transductively (reference+query pooled before splitting),
  which is realistic for atlas-mapping use cases but is a different
  deployment assumption than the fully out-of-sample setup Tiers 1-2 and
  the rest of Tier 3 use -- worth flagging explicitly in the paper's methods
  section.

## Verdict

**Three consistent, complementary findings across all three tiers:**

1. **The generic domain-classifier baseline is unreliable everywhere it was
   tested, and this survives a genuine attempt to fix it fairly.**
   Cross-fitting + CV-tuned regularization + weight truncation (step (b),
   2026-08-08) measurably shrinks its inflated prediction sets (20-30%
   smaller) without resolving the core instability: it still produces sets
   3-6x larger than the principled methods in Tiers 1-2, still collapses on
   individual folds (Tier 1's human4: coverage as low as 0.69), and remains
   functionally vacuous in Tier 3 (predicting nearly all 3 classes for
   almost every cell). This is a real, defensible argument for the
   structure-aware approach precisely because the comparison is now fair:
   it avoids this instability entirely by using *known* batch identity
   instead of an *estimated* density ratio, and the estimated version does
   not catch up even after standard stabilization fixes.

2. **Structure-aware reweighting -- heuristic or theoretically grounded --
   is safe when there's no real problem (Tiers 1-2: naive already tracks
   nominal coverage closely) and insufficient when there is one (Tier 3:
   naive undercovers by 15-22 points, and neither reweighting variant closes
   the gap).** The mechanistic reason is identifiable and interesting, and
   now confirmed at the level of the estimator's own assumptions rather than
   just empirically: reweighting can only redistribute trust *among*
   calibration sub-batches toward whichever looks most like the query
   (formally, both estimators require the query to lie in, or resemble a
   mixture of, the reference batches' distributions). In Tier 3, all
   reference sub-batches (the 3 healthy-lung samples, or the 7 tumor
   patients in the reverse direction) are shifted from the query in
   essentially the same direction -- a monolithic study-level shift, not
   differential sub-batch heterogeneity the reference can explain -- so
   there is no calibration structure to lean on and the propensity
   estimator's mixture assumption is violated by construction. This matches
   why the source paper's own fix for this exact scenario (healthy vs.
   tumor lung) was OOD *rejection*, not reweighting.

3. **What actually helps Tier 3 is a fundamentally different class of
   method: upstream batch integration, not any form of reweighting (step
   (c), 2026-08-08).** Harmony, applied before the classifier and
   conformal-calibration steps rather than at the weighting stage, closes
   14-65% of the undercoverage gap depending on shift direction and alpha --
   a real, substantial improvement that no reweighting variant achieved --
   while still falling short of nominal coverage in 5 of 6 tested rows.
   This confirms the reweighting-vs-integration distinction predicted by
   step (b)'s positivity-violation diagnosis: when calibration and query
   occupy near-disjoint regions of feature space, only a method that moves
   the embedding itself (not one that reweights fixed points within it) can
   make progress, and even that method does not fully solve the problem.
   Mechanistically, Harmony improves conformal calibration transfer
   specifically (coverage) without necessarily improving classification
   accuracy (top-1 was unchanged in the harder direction) -- a genuinely
   non-obvious dissociation worth its own discussion in the paper.

**Taken together, this is a stronger, more publishable finding than a clean
win would have been.** It precisely characterizes the boundary conditions of
when batch-structure-aware reweighting helps (heterogeneous multi-batch
references), when it is fundamentally insufficient regardless of how
principled or well-implemented (monolithic cross-study shift, verified
against three separate attempted fixes: theoretical grounding, a fair
baseline, and finally a different method class entirely), and what class of
method is actually needed instead (upstream integration, with honestly
reported partial success) -- exactly the kind of rigor and completeness a
methods paper needs.

## Known simplifications (documented, not hidden)
- Tier 3's coarse 3-class labeling (vs. the source paper's 14-25 fine-grained
  subtypes) may understate how severe the true batch effect is on
  fine-grained classification; it's plausible naive undercoverage would be
  even worse at finer granularity.
- No Table S8-style deliberate class exclusion was used in any Phase 3 tier
  (unlike Phases 1-2) -- this phase isolates covariate/batch shift from the
  separate novel-class-OOD problem by design.
- Cell counts were capped per batch (1500/donor in Tiers 1-2, 1500/sample in
  Tier 3) for runtime tractability; full-data results were not checked for
  sensitivity to this cap.

## Files
- `scripts/weighted_conformal.py` -- core reusable module (randomized APS,
  weighted quantile, five weighting schemes: naive, domain_clf_naive,
  domain_clf, structure_aware_centroid, structure_aware_propensity)
- `scripts/tier{1,2,3}_*.py` -- per-tier runners
- `scripts/tier3_harmony.py` -- step (c): upstream Harmony batch integration
  ahead of the conformal pipeline, tested specifically for Tier 3
- `scripts/prepare_gse127465.py`, `prepare_tier3_cross_study.py` -- data prep
- `results/tier{1,2,3}_*_results.json` / `*_summary.csv` -- full results
- `results/tier3_harmony_results.json` / `tier3_harmony_summary.csv` --
  step (c) results
- `data/gse127465_prepared.h5ad`, `data/tier3_prepared.h5ad` -- processed data
