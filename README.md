# Batch-structure-aware conformal calibration for single-cell annotation

Code to reproduce all results reported in the manuscript *"Batch-structure-aware
conformal calibration for single-cell annotation reveals when reweighting works,
fails, and can be rescued"* and its Supplementary Information.

This repository contains **code only**. Raw and processed single-cell datasets are
**not included** (multi-gigabyte files, several with their own redistribution
terms) — Section 3 below gives the exact public URL and target path for every
file each script expects. Small numeric results (CSV/JSON/log files a few KB to a
few MB, directly underlying the manuscript's tables and figures) **are** included
so that figure/statistics regeneration is possible without first re-running the
full pipeline.

## 1. Repository layout

```
environment.yml                     conda environment (single env for all phases)
phase1_baseline/                    Phase 1: CellTypist + conformal-prediction baseline reproduction
  scripts/                          prepare_baron_pancreas.py, run_phase1_baseline.py
  results/                          small result JSON/CSV/summary
  vendor_patch/                     patch for the third-party conformal annotator (see §4)
phase2_geneformer/                  Phase 2: Geneformer embeddings + linear-probe validation
  scripts/
  results/
phase3_batch_reweighting/           Phase 3: Tiers 1-3, PCA ablation, Harmony rescue, bootstrap CIs
  scripts/
  results/
phase4_hlca_audit/                  Phase 4: HLCA audit, candidate-error mining, bootstrap CIs
  scripts/
  results/
phase5_breadth_tabula_blood/        Phase 5: Tabula Sapiens Blood breadth check
  scripts/
  results/
manuscript_figures/                 Figure-generation scripts (read the results/ CSVs above)
  make_workflow_figure_v2.py        -> Figure 1
  make_figures_v2.py                -> Figures 2-5
```

Empty `data/` subfolders are **not** pre-created; each phase's scripts expect a
`data/` folder alongside its `scripts/` folder (i.e. `phase1_baseline/data/`,
`phase2_geneformer/data/`, etc.) populated as described in §3. Create these
folders yourself as you download each dataset.

Each phase corresponds directly to a part of the manuscript:

| Folder | Manuscript section |
|---|---|
| `phase1_baseline` | Results: "validated the underlying classification pipeline against the source paper's own reported benchmarks" (CellTypist reproduction); Supplementary Note 1 |
| `phase2_geneformer` | Results: Geneformer backbone validation, linear-probe check; Methods "Embeddings and classifiers"; Supplementary Note 1 |
| `phase3_batch_reweighting` | Results: Tiers 1-3, Table 1, Figures 2-3, the PCA ablation, the Harmony rescue, Tier 3 bootstrap CIs (Supplementary Tables S1, S2, S5, S6, S7) |
| `phase4_hlca_audit` | Results: HLCA audit, Table 2, Figure 4, the Krasnow_2020 discrepancy, HLCA bootstrap CIs (Supplementary Note 2, Tables S3, S8) |
| `phase5_breadth_tabula_blood` | Results: "The pattern generalizes across tissues", Figure 5, Supplementary Table S4 |

## 2. Environment setup

```bash
conda env create -f environment.yml
conda activate annotator-env
```

This pins the exact package versions used to produce the reported results
(Python 3.11, scikit-learn 1.5.0, scanpy 1.10.1, PyTorch 2.1.2 CPU build,
torchcp 1.0.2, celltypist 1.6.3, harmonypy 0.0.10). `cellxgene-census` is
deliberately **not** included — on Windows its `tiledbsoma` dependency has no
prebuilt wheel and needs a C/C++ build toolchain. It isn't required: HLCA core
and Tabula Sapiens Blood are downloaded directly as `.h5ad` files instead (§3).

Two additional components are installed separately, not via this environment
file:

- **Geneformer** (the embedding model itself) — see §4, Phase 2.
- **The vendored CellTypist conformal annotator** — see §4, Phase 1.

## 3. External data downloads

Nothing here is redistributed in this repository. Download each file from its
public source and place it at the exact path shown (paths are relative to
each phase's folder). All raw files are used as downloaded — no manual
renaming beyond what's listed.

### Baron et al. 2016 pancreas (Phase 1, Phase 3 Tier 1)

GEO accession [GSE84133](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84133).
Download the four **human** per-donor supplementary CSV files and place them,
still gzipped, at:

```
phase1_baseline/data/raw/extracted/GSM2230757_human1_umifm_counts.csv.gz
phase1_baseline/data/raw/extracted/GSM2230758_human2_umifm_counts.csv.gz
phase1_baseline/data/raw/extracted/GSM2230759_human3_umifm_counts.csv.gz
phase1_baseline/data/raw/extracted/GSM2230760_human4_umifm_counts.csv.gz
```

(The mouse donor files on the same GEO page are not used.)

### Zilionis et al. 2019 NSCLC (Phase 3 Tiers 2 and 3)

GEO accession [GSE127465](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE127465).
Download the **human** supplementary files and place them at:

```
phase3_batch_reweighting/data/gse127465/human_cell_metadata.tsv        (decompressed)
phase3_batch_reweighting/data/gse127465/gene_names_human.tsv           (decompressed)
phase3_batch_reweighting/data/gse127465/human_counts_normalized.mtx.gz (leave gzipped)
```

### Kadur Lakshminarasimha Murthy et al. 2022 healthy lung (Phase 3 Tier 3)

GEO accession [GSE178360](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE178360).
Download the 10x Genomics filtered feature-barcode matrices for the three
samples used and place them at:

```
phase3_batch_reweighting/data/gse178360/extracted/GSM5388411_DD046Q_filtered_feature_bc_matrix.h5
phase3_batch_reweighting/data/gse178360/extracted/GSM5388412_DD047Q_filtered_feature_bc_matrix.h5
phase3_batch_reweighting/data/gse178360/extracted/GSM5388413_DD073Rfiltered_feature_bc_matrix.h5
```

(The third filename has no underscore before "filtered" — that's the actual
filename GEO serves; reproduce it exactly, the script expects it verbatim.)

### Human Lung Cell Atlas (HLCA) core (Phase 4)

CZ CELLxGENE Discover collection: *An integrated cell atlas of the lung in
health and disease* —
[https://cellxgene.cziscience.com/collections/6f6d381a-7701-4781-935c-db10d30de293](https://cellxgene.cziscience.com/collections/6f6d381a-7701-4781-935c-db10d30de293).
Open the **HLCA core** dataset (584,944 cells, 14 contributing datasets — not
the larger "extended" atlas) and download its `.h5ad` via the "Download" button,
or via the CELLxGENE Discover REST API
(`api.cellxgene.cziscience.com`, `/curation/v1/collections/{collection_id}`
returns each dataset's current asset/download URL). Place the file at:

```
phase4_hlca_audit/data/hlca_core.h5ad
```

This file is ~5.5 GB.

### Tabula Sapiens (Phase 5)

CZ CELLxGENE Discover collection: *Tabula Sapiens* —
[https://cellxgene.cziscience.com/collections/e5f58829-1a66-40b5-a624-9046778e74f5](https://cellxgene.cziscience.com/collections/e5f58829-1a66-40b5-a624-9046778e74f5).
Download the **Blood** tissue dataset (85,233 cells, 9 donors) and place it at:

```
phase4_hlca_audit/data_tabula_blood/tabula_blood.h5ad
```

(Yes, under the Phase 4 folder — that's the path the Phase 5 scripts read
from, preserved as-run; see §5 for why.)

## 4. Third-party / vendored code

### Geneformer (Phase 2)

Clone the model repository and install it into the `annotator-env` environment:

```bash
cd phase2_geneformer
git clone https://huggingface.co/ctheodoris/Geneformer models/Geneformer
cd models/Geneformer
git checkout 04c2b2e84da7c0f385c3f9ad8f3ec24bab6650e5
pip install .
```

This installs the `geneformer` package and its own dependencies (`datasets`,
`loompy`, etc.) automatically. Scripts expect the 256-dimensional
**Geneformer-V1-10M** checkpoint at
`phase2_geneformer/models/Geneformer/Geneformer-V1-10M/` — this comes with the
cloned repository.

### Conformal single-cell annotator (Phase 1)

The Phase 1 CellTypist baseline reproduction uses a third-party conformal
annotation package, not something original to this work:

```bash
cd phase1_baseline
git clone https://github.com/digital-medicine-research-group-UNAV/conformalized_single_cell_annotator.git repo
cd repo
git checkout 526bad858b120137d01267bde7ccb5dda5431308
git apply ../vendor_patch/conformalSC_annotator.patch
```

The patch fixes an uninitialized-attribute bug in `ConformalSCAnnotator`
(`self.obsm` / `self.layer` were read before being set on one code path) that
surfaced under the "raw counts in `adata.X`" configuration used here — see
Supplementary Note 1 for the full list of issues identified in this package
during validation. One further compatibility issue described there (a
scikit-learn keyword argument the package's internal logistic-regression call
depends on) is resolved simply by using the pinned `scikit-learn==1.5.0` from
`environment.yml`, not by a code patch. A third issue (the package's
out-of-distribution detector calling a tensor-conversion function directly on
sparse matrices) is worked around in *our* calling code, not the vendored
package: `run_phase1_baseline.py` densifies `adata.X` before invoking the
annotator (see the `scipy.sparse.issparse` check near the top of `main()`).

## 5. Reproduction pipeline

Run phases in order — later phases consume earlier phases' outputs. All
scripts are run from their own `scripts/` folder (`cd phaseN_.../scripts`) and
write into the sibling `data/` and `results/` folders via paths resolved from
`__file__`, so the overall folder layout in §1 must be preserved. All fitting
uses a fixed random seed (`random_state=42` / `numpy.random.default_rng(42)`
throughout), so re-running should reproduce the reported numbers exactly,
modulo floating-point nondeterminism across library versions/platforms.

**A note on three scripts**: `tier1_baron_pancreas.py`,
`tier1_baron_pancreas_pca_ablation.py`, and `inspect_blood.py` /
`prepare_blood.py` originally hardcoded the absolute path of the development
machine (`D:/Research/NCS/...`). The copies in this repository have been
edited to resolve these paths relative to the repository root instead
(`Path(__file__).resolve().parents[2] / ...`) so the pipeline runs unmodified
on any machine with this folder layout. No computation changed — only how the
input path is spelled.

### Phase 1 — CellTypist + conformal-prediction baseline

```bash
cd phase1_baseline/scripts
python prepare_baron_pancreas.py     # -> ../data/processed/baron_human{1..4}.h5ad
python run_phase1_baseline.py        # -> ../results/phase1_results.json, phase1_summary.csv
```
Reproduces the leave-one-donor-out CellTypist + APS coverage numbers
(0.908 / 0.871 / 0.822 at α = 0.01 / 0.05 / 0.10) reported in Results and
Methods, validating the reproduction pipeline before it is extended.

### Phase 2 — Geneformer embeddings and backbone validation

```bash
cd phase2_geneformer/scripts
python build_gene_mapping.py         # symbol -> Ensembl ID via mygene.info (needs internet; see caveat below)
python prepare_for_tokenizer.py      # -> ../data/for_tokenizer/*.h5ad
python tokenize_data.py              # -> ../data/tokenized/baron_pancreas.dataset/
python extract_embeddings.py         # -> ../results/baron_pancreas_embs.csv (Geneformer-V1-10M, 256-d)
python build_embedding_adata.py      # -> ../data/embedding_adata/baron_{donor}_geneformer.h5ad
python linear_probe_sanity_check.py  # backbone-quality check: 85.9% mean accuracy (Supplementary Note 1)
python run_phase2_geneformer.py      # vendored feed-forward classifier comparison (~50% mean accuracy, unstable)
```
`build_gene_mapping.py` queries the public `mygene.info` API and caches the
result; because that API's underlying gene annotation can be updated over
time, an exact-byte-identical remap is not guaranteed years later, though the
effect on downstream results should be negligible (a handful of gene symbols
at most). `timing_test_embeddings.py` is a standalone runtime-benchmarking
script, not required for any reported number.

### Phase 3 — Batch-structure-aware reweighting (Tiers 1-3)

```bash
cd phase3_batch_reweighting/scripts

# Tier 1 (Baron pancreas, donor-as-batch) — needs Phase 2 output
python tier1_baron_pancreas.py             # -> ../results/tier1_baron_pancreas_{results.json,summary.csv}  (Table 1 row 1, Supplementary Table S5)
python tier1_baron_pancreas_pca_ablation.py # -> ../results/tier1_pca_ablation_*  (Supplementary Table S6) — needs Phase 1 output only, not Phase 2

# Tier 2 (GSE127465, patient-as-batch)
python prepare_gse127465.py                # -> ../data/gse127465_prepared.h5ad
python tier2_gse127465.py                  # -> ../results/tier2_gse127465_*  (Table 1 row 2)

# Tier 3 (cross-study: GSE178360 healthy vs. GSE127465 tumor) — needs gse127465 raw files present too
python prepare_tier3_cross_study.py        # -> ../data/tier3_prepared.h5ad
python tier3_cross_study.py                # -> ../results/tier3_cross_study_*  (Table 1 row 3, Supplementary Table S1)
python tier3_coverage_ci.py                # -> ../results/tier3_coverage_ci.*  (Supplementary Table S7 — bootstrap CIs)
python tier3_harmony.py                    # -> ../results/tier3_harmony_*  (Table 1 row 4, Figure 3, Supplementary Table S2)
python tier3_hybrid.py                     # exploratory reject-option follow-up, not reported as a headline result
```
`weighted_conformal.py` is the shared core module (APS scoring, the four
weighting schemes, coverage evaluation, and the bootstrap helper
`evaluate_coverage_percell`) imported by every Tier 1-3 script — it has no
`data/` dependency of its own. `hybrid_reject.py` backs `tier3_hybrid.py` only.
`inspect_baron_raw.py` is a throwaway inspection script, not required.

### Phase 4 — HLCA audit

```bash
cd phase4_hlca_audit/scripts
python prepare_hlca_audit.py               # -> ../data/hlca_audit_prepared.h5ad  (needs data/hlca_core.h5ad, §3)
python run_hlca_audit.py                   # -> ../results/hlca_audit_*, hlca_audit_per_cell.csv  (Table 2, Figure 4a)
python hlca_accuracy_ci.py                 # -> ../results/hlca_accuracy_ci*  (Supplementary Table S8 — bootstrap CIs)
python extract_examples.py                 # -> ../results/hlca_examples_full.csv  (the 4 case studies, Supplementary Table S3)
python find_novel_errors.py                # -> ../results/hlca_candidate_novel_errors.csv  (candidate discrepancy mining)
python hlca_marker_verification.py         # -> ../results/hlca_marker_verification.csv  (Figure 4b — needs data/hlca_core.h5ad again, raw re-query)
python check_basal_secretory_per_dataset.py
python check_stress_confound.py            # -> ../results/hlca_stress_confound_check.csv  (Figure 4c, the dissociation-stress confound check)
python check_dataset_tech.py               # dissociation-protocol metadata check referenced in Results
```
`inspect_hlca.py` and `inspect_markers.py` are exploratory scripts used to
choose the annotation/marker columns during development; not required to
reproduce a reported number.

### Phase 5 — Cross-tissue breadth check (Tabula Sapiens Blood)

```bash
cd phase5_breadth_tabula_blood/scripts
python prepare_blood.py       # -> ../data/blood_prepared.h5ad  (needs phase4_hlca_audit/data_tabula_blood/tabula_blood.h5ad, §3)
python run_blood_breadth.py   # -> ../results/blood_breadth_*  (Figure 5, Supplementary Table S4)
```
`inspect_blood.py` is exploratory, not required.

## 6. Regenerating the manuscript figures

Once the relevant `results/` CSVs above exist (already present in this
repository from the original run — see §1), figures can be regenerated
without re-running any analysis:

```bash
cd manuscript_figures
python make_workflow_figure_v2.py   # -> figures_out/Figure1_workflow_overview.{png,tif,pdf}
python make_figures_v2.py           # -> figures_out/Figure{2,3,4,5}_*.{png,tif}
```

## 7. Results-to-manuscript cross-reference

| Result file | Manuscript location |
|---|---|
| `phase1_baseline/results/phase1_summary.csv` | Results §"A controlled gradient of batch-effect severity", Methods |
| `phase2_geneformer/results/phase2_summary*.csv` | Results (Geneformer/linear-probe validation), Supplementary Note 1 |
| `phase3_batch_reweighting/results/tier1_baron_pancreas_summary.csv` | Table 1 row 1, Figure 2, Supplementary Table S5 |
| `phase3_batch_reweighting/results/tier1_pca_ablation_summary.csv` | Results (PCA ablation paragraph), Supplementary Table S6 |
| `phase3_batch_reweighting/results/tier2_gse127465_summary.csv` | Table 1 row 2, Figure 2 |
| `phase3_batch_reweighting/results/tier3_cross_study_summary.csv` | Table 1 row 3, Figure 2, Supplementary Table S1 |
| `phase3_batch_reweighting/results/tier3_coverage_ci.csv` | Results (Tier 3 bootstrap CIs), Supplementary Table S7 |
| `phase3_batch_reweighting/results/tier3_harmony_summary.csv` | Table 1 row 4, Figure 3, Supplementary Table S2 |
| `phase4_hlca_audit/results/hlca_audit_summary_by_reannotation_type.csv` | Table 2, Figure 4a |
| `phase4_hlca_audit/results/hlca_accuracy_ci.csv` | Results (HLCA bootstrap CIs), Supplementary Table S8 |
| `phase4_hlca_audit/results/hlca_examples_full.csv` | Supplementary Table S3 (four case studies) |
| `phase4_hlca_audit/results/hlca_marker_verification.csv` | Figure 4b, Results §"A specific, previously unflagged discrepancy" |
| `phase4_hlca_audit/results/hlca_stress_confound_check.csv` | Figure 4c, Results (stress-artifact confound check) |
| `phase5_breadth_tabula_blood/results/blood_breadth_summary.csv` | Figure 5, Supplementary Table S4 |

## 8. Reproducibility notes

- All model fitting uses `random_state=42`; APS randomization uses a
  `numpy.random.Generator` seeded per fold. Bootstrap confidence intervals
  (Phase 3's `tier3_coverage_ci.py`, Phase 4's `hlca_accuracy_ci.py`) use a
  separately seeded generator (`seed + 1000`) so they don't perturb the
  point-estimate computation they wrap.
- The point estimates recomputed for the bootstrap-CI scripts can differ from
  the original Table 1 / Table 2 values in the last reported digit (e.g.
  0.941 vs. 0.942) because the CI script evaluates coverage per-cell in a
  loop rather than the vectorized batch computation `evaluate_coverage` uses,
  which consumes the shared random generator in a different order for APS
  tie-breaking. This does not reflect a different underlying model fit —
  both are valid estimates of the same quantity, well within each other's
  bootstrap CI.
- `git apply` fails silently if run from the wrong working directory —
  run it from inside the cloned `repo/` folder as shown in §4, with the
  patch path relative to that location.

## 9. License note

The vendored conformal annotator (§4) and the Geneformer model (§4) are
third-party code and models with their own licenses, not covered by this
repository's license. Consult their respective repositories before reuse.
