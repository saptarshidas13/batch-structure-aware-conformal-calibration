"""
Prepare Tier 3: a genuine cross-study batch comparison -- GSE178360 (healthy
lung, Kadur et al. 2022, 3 samples) vs. the tumor-lung cells already
extracted from GSE127465 (Zilionis et al. 2019, 7 patients). This is the
"severe batch effect" case: different studies, different years, different
protocols -- exactly the scenario the source paper found breaks naive
conformal coverage (their Table S9 / Figures S8-S11).

No shared fine-grained cell-type ontology exists across these two
independently-annotated studies, and GSE127465's own labels are hand-curated
while GSE178360 ships with no labels at all (raw 10x output). Rather than
attempt full label harmonization (the heavy lift Tier 3 was scoped to avoid
-- see conversation), both datasets are labeled into the SAME small set of
broad, marker-gene-derived categories using canonical markers:
  Immune     : PTPRC
  Epithelial : EPCAM
  Stromal    : COL1A1, PECAM1, ACTA2
Per-sample / per-patient batch identity is preserved (not collapsed to a
single "healthy" / "tumor" label) so structure-aware weighting has real
sub-batch structure to work with in both directions.
"""
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io
import scipy.sparse as sp

BASE_DIR = Path(__file__).resolve().parents[1]
HEALTHY_DIR = BASE_DIR / "data" / "gse178360" / "extracted"
TUMOR_META = BASE_DIR / "data" / "gse127465" / "human_cell_metadata.tsv"
TUMOR_MTX = BASE_DIR / "data" / "gse127465" / "human_counts_normalized.mtx.gz"
TUMOR_GENES = BASE_DIR / "data" / "gse127465" / "gene_names_human.tsv"
OUT_PATH = BASE_DIR / "data" / "tier3_prepared.h5ad"

HEALTHY_SAMPLES = {
    "GSM5388411": "GSM5388411_DD046Q_filtered_feature_bc_matrix.h5",
    "GSM5388412": "GSM5388412_DD047Q_filtered_feature_bc_matrix.h5",
    "GSM5388413": "GSM5388413_DD073Rfiltered_feature_bc_matrix.h5",
}

MARKERS = {
    "Immune": ["PTPRC"],
    "Epithelial": ["EPCAM"],
    "Stromal": ["COL1A1", "PECAM1", "ACTA2"],
}

MAX_CELLS_PER_SAMPLE = 1500
RANDOM_STATE = 42


def label_by_markers(adata: ad.AnnData) -> np.ndarray:
    present_markers = {k: [g for g in v if g in adata.var_names] for k, v in MARKERS.items()}
    for k, v in present_markers.items():
        print(f"    markers present for {k}: {v}")
    for name, genes in present_markers.items():
        if genes:
            sc.tl.score_genes(adata, genes, score_name=f"score_{name}")
        else:
            adata.obs[f"score_{name}"] = -np.inf
    score_cols = [f"score_{k}" for k in MARKERS]
    scores = adata.obs[score_cols].values
    labels = np.array(list(MARKERS.keys()))[np.argmax(scores, axis=1)]
    return labels


def load_healthy() -> ad.AnnData:
    parts = []
    rng = np.random.default_rng(RANDOM_STATE)
    for sample_id, fname in HEALTHY_SAMPLES.items():
        print(f"  loading {sample_id}...")
        a = sc.read_10x_h5(HEALTHY_DIR / fname)
        a.var_names_make_unique()
        sc.pp.filter_cells(a, min_genes=200)
        sc.pp.filter_genes(a, min_cells=3)
        if a.n_obs > MAX_CELLS_PER_SAMPLE:
            idx = rng.choice(a.n_obs, size=MAX_CELLS_PER_SAMPLE, replace=False)
            a = a[idx].copy()
        a.obs["donor"] = sample_id
        a.obs["study"] = "healthy_GSE178360"
        parts.append(a)
    adata = ad.concat(parts, join="inner")
    adata.var_names_make_unique()
    print(f"  healthy total: {adata.shape}")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def load_tumor() -> ad.AnnData:
    print("  loading GSE127465 tumor cells...")
    meta = pd.read_csv(TUMOR_META, sep="\t")
    genes = pd.read_csv(TUMOR_GENES, header=None)[0].values
    X = scipy.io.mmread(TUMOR_MTX)
    X = sp.csr_matrix(X)
    if X.shape[0] == len(genes) and X.shape[1] == meta.shape[0]:
        X = X.T.tocsr()

    tumor_mask = (meta["Tissue"] == "tumor").values
    X = X[tumor_mask]
    meta = meta[tumor_mask].reset_index(drop=True)

    rng = np.random.default_rng(RANDOM_STATE)
    keep_idx = []
    for patient, group in meta.groupby("Patient"):
        idx = group.index.values
        if len(idx) > MAX_CELLS_PER_SAMPLE:
            idx = rng.choice(idx, size=MAX_CELLS_PER_SAMPLE, replace=False)
        keep_idx.append(idx)
    keep_idx = np.sort(np.concatenate(keep_idx))
    X = X[keep_idx]
    meta = meta.iloc[keep_idx].reset_index(drop=True)

    adata = ad.AnnData(X=X)
    adata.var_names = genes
    adata.var_names_make_unique()
    adata.obs["donor"] = meta["Patient"].values
    adata.obs["study"] = "tumor_GSE127465"
    print(f"  tumor total: {adata.shape}")

    is_log_scale = adata.X.max() < 30
    if not is_log_scale:
        adata.X = adata.X.copy()
        adata.X.data = np.log1p(adata.X.data)
    return adata


def main():
    print("Loading healthy lung (GSE178360)...")
    healthy = load_healthy()
    print("\nLoading tumor lung (GSE127465)...")
    tumor = load_tumor()

    common_genes = sorted(set(healthy.var_names) & set(tumor.var_names))
    print(f"\nCommon genes between studies: {len(common_genes)}")
    healthy = healthy[:, common_genes].copy()
    tumor = tumor[:, common_genes].copy()

    print("\nLabeling healthy lung cells by marker score...")
    healthy.obs["broad_cell_type"] = label_by_markers(healthy)
    print(pd.Series(healthy.obs["broad_cell_type"]).value_counts())

    print("\nLabeling tumor lung cells by marker score...")
    tumor.obs["broad_cell_type"] = label_by_markers(tumor)
    print(pd.Series(tumor.obs["broad_cell_type"]).value_counts())

    combined = ad.concat([healthy, tumor], join="inner")
    combined.var["features"] = combined.var_names
    combined.write_h5ad(OUT_PATH)
    print(f"\nSaved -> {OUT_PATH} shape={combined.shape}")
    print(combined.obs.groupby(["study", "broad_cell_type"]).size())


if __name__ == "__main__":
    main()
