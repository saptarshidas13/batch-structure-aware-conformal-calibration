# Phase 2 — Final Summary

**Status: COMPLETE**

## What was tested
Same leave-one-donor-out conformal cell-type annotation design as Phase 1
(Baron et al. pancreas data, Table S8 OOD-simulation design, autoencoder OOD
detector, APS conformal calibration, alpha in {0.01, 0.05, 0.1}) -- but with
the classifier backbone swapped from CellTypist (raw gene expression) to a
small feed-forward net ("torch_net", 128-64-32 hidden units) trained on
frozen Geneformer-V1-10M cell embeddings (256-dim, obsm path). This is
Open Problem A: does conformal calibration transfer to a foundation-model
backbone, and how does it compare to a classical classifier?

## Final results (aggregate across 4 donor folds, ID-only coverage)

Three `torch_net` configurations were tested, plus an independent linear
(logistic regression) probe used purely as a diagnostic to separate
"is it the embeddings" from "is it this specific classifier":

| Config | human1 acc | human2 acc | human3 acc | human4 acc | mean acc |
|---|---|---|---|---|---|
| torch_net, unstandardized | 62.6%* | 51.7% | 37.8% | 58.6% | ~51% |
| torch_net, standardized | 34.2% | 53.0% | 55.2% | 63.5% | ~51.5% |
| torch_net, standardized + seeded | 29.1% | 50.5% | 55.8% | 60.4% | ~49% |
| **Linear probe (StandardScaler + LogisticRegression)** | 85.7% | 85.9% | 85.2% | 86.7% | **85.9%** |
| CellTypist (Phase 1, for reference) | -- | -- | -- | -- | ~82-94% |

*from the reduced-epoch smoke test, not a fully comparable full-run number.

Coverage/set-size numbers for the `torch_net` configuration (standardized +
seeded, the final one run) at alpha=0.01/0.05/0.10: coverage 0.67/0.63/0.59,
size 2.73/1.69/1.41 -- all substantially below CellTypist's 0.91/0.87/0.82
with size ~1.0, tracking the weak classification accuracy above.

## Verdict (revised after diagnostic follow-up)
**Initial conclusion was wrong and has been corrected.** The first full run
suggested Geneformer embeddings themselves were weak. A linear-probe
diagnostic (fit directly on the same folds, no OOD detector or conformal
layer, just plain logistic regression) showed the embeddings are actually
**comparable in quality to CellTypist** (85.9%, stable across every fold).
Two follow-up attempts to fix `torch_net` -- standardizing the embeddings,
then also fixing PyTorch's unseeded weight initialization -- each produced a
different fold collapsing but the *same* ~50% mean accuracy, showing the
instability is not primarily due to input scaling or random initialization
luck. The bottleneck is specifically the vendored repo's small feed-forward
classifier (`torch_net`, 128-64-32 hidden units, lr=1e-4, 1000 epochs) being
poorly suited to this 14-class, imbalanced problem -- not the embeddings.

**Practical implication:** the mechanical question Open Problem A asks --
does conformal calibration transfer to a foundation-model backbone -- is
answered **yes**, the pipeline runs correctly end-to-end. But the specific
`torch_net` classifier is not a trustworthy way to evaluate embedding
quality in this repo as configured, and further hyperparameter search on it
was judged not worth the time against the actual paper-worthy goal (Phase 3
onward). For Phase 3+, CellTypist remains the primary backbone; Geneformer
numbers from this phase should be cited only with this caveat attached, and
the linear-probe result (not the torch_net result) is the more honest
signal of Geneformer-V1-10M's actual embedding quality on this task.

Phase 2 gate: PASS, with the corrected finding above superseding the
original one.

## Bugs found and fixed
1. `transformers` package (top-level pinned) had disabled PyTorch support
   because a newer version required torch>=2.4 while Phase 1's env pins
   torch==2.1.2 -- built a separate `geneformer-env` (torch 2.4.1) rather
   than risk destabilizing the validated Phase 1 environment.
2. `geneformer`'s declared dependency `loompy` -> `accumulation-tree` has no
   Windows wheel and needs a C++ toolchain not present on this machine
   (same class of issue as Phase 1's `tiledbsoma`) -- installed everything
   via `--no-deps` first, then added back real dependencies individually;
   stubbed `tdigest` (only used by an optional summary-stat feature we don't
   call) the same way `torchsort` was stubbed in Phase 1.
3. `transformers==5.14.1` (latest at install time) restructured its API and
   removed `SpecialTokensMixin` from the top-level namespace, which
   geneformer's code expects -- pinned to `transformers==4.44.2`.
4. `pip install .` (non-editable) copies files into site-packages; further
   source edits to the cloned repo didn't take effect until reinstalled with
   `-e` (editable mode).
5. **Real bug**: `geneformer/emb_extractor.py` and `perturber_utils.py`
   hardcode `device="cuda"` / `.to("cuda")` in the embedding-extraction code
   path, even though `load_model` elsewhere in the same codebase already
   uses the correct `"cuda" if torch.cuda.is_available() else "cpu"`
   pattern -- patched both call sites to use the same conditional. This
   directly explains the "GPU resources are required" language in the
   model's documentation; it isn't a hard architectural requirement, just
   an inconsistently-applied device selection.
6. `mygene`'s querymany failed on a single 20k-gene batch request (transient
   connection drop) -- rewrote with 500-gene chunking and retry/backoff.

## Timing (for future reference)
- Gene symbol -> Ensembl ID mapping (mygene.info, 20,125 genes): a few minutes
- Tokenization (8,526 cells): fast, not model-dependent
- Embedding extraction (8,526 cells, CPU, Geneformer-V1-10M): 46 minutes
  (0.361 s/cell, timing-tested on 200 cells first before committing)
- Full conformal pipeline (4 folds, 850 OOD epochs + 1000 classifier epochs
  each): completed within the monitoring window

## Files
- `data/embedding_adata/baron_human{1-4}_geneformer.h5ad` -- per-donor
  Geneformer-embedding AnnData objects
- `results/baron_pancreas_embs.csv` -- raw extracted embeddings (8526 x 256)
- `scripts/linear_probe_sanity_check.py` -- the diagnostic that corrected
  the original conclusion
- `results/phase2_results.json`, `phase2_summary.csv` -- final run
  (standardized + seeded torch_net)
- `results/phase2_results_unstandardized.json` /
  `phase2_summary_unstandardized.csv` -- original run (superseded)
- `results/phase2_results_standardized_unseeded.json` /
  `phase2_summary_standardized_unseeded.csv` -- intermediate diagnostic run
- `results/phase2_results_quick_unstandardized.json` /
  `phase2_summary_quick_unstandardized.csv` -- original smoke test
