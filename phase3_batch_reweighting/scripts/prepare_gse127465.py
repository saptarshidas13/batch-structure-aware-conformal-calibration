"""
Prepare GSE127465 (Zilionis et al. 2019, NSCLC) for Tier 2: patient-as-batch
weighted conformal prediction. Filters to tumor tissue, drops the
"PatientX-specific" artifact/QC categories (not real generalizable cell
types), subsamples per patient to keep runtime tractable, and computes a
PCA embedding as the shared feature space for classification + batch
centroids (analogous role to the Geneformer embeddings in Tier 1).
"""
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
from sklearn.decomposition import PCA

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "gse127465"
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "gse127465_prepared.h5ad"

MAX_CELLS_PER_PATIENT = 1500
N_PCS = 50
RANDOM_STATE = 42

EXCLUDE_LABEL_SUBSTRINGS = ["-specific"]  # e.g. "Patient1-specific" QC/doublet artifacts


def main():
    print("Loading metadata...")
    meta = pd.read_csv(DATA_DIR / "human_cell_metadata.tsv", sep="\t")
    gene_names = pd.read_csv(DATA_DIR / "gene_names_human.tsv", header=None)[0].values
    print(f"  metadata: {meta.shape}, genes: {len(gene_names)}")

    print("Loading count matrix (this may take a bit, ~500MB compressed)...")
    X = scipy.io.mmread(DATA_DIR / "human_counts_normalized.mtx.gz")
    X = sp.csr_matrix(X)
    print(f"  matrix shape: {X.shape}")

    if X.shape[0] == len(gene_names) and X.shape[1] == meta.shape[0]:
        X = X.T.tocsr()
        print("  transposed to cells x genes")
    assert X.shape[0] == meta.shape[0], f"row mismatch: {X.shape[0]} vs {meta.shape[0]}"
    assert X.shape[1] == len(gene_names), f"col mismatch: {X.shape[1]} vs {len(gene_names)}"

    print("Value range check:", X.data[:10] if X.nnz else "empty", "max:", X.max())

    tumor_mask = (meta["Tissue"] == "tumor").values
    valid_label_mask = ~meta["Major cell type"].astype(str).str.contains(
        "|".join(EXCLUDE_LABEL_SUBSTRINGS), na=False
    ).values
    keep = tumor_mask & valid_label_mask
    print(f"Tumor + valid-label cells: {keep.sum()} / {len(keep)}")

    X = X[keep]
    meta = meta[keep].reset_index(drop=True)

    rng = np.random.default_rng(RANDOM_STATE)
    keep_idx = []
    for patient, group in meta.groupby("Patient"):
        idx = group.index.values
        if len(idx) > MAX_CELLS_PER_PATIENT:
            idx = rng.choice(idx, size=MAX_CELLS_PER_PATIENT, replace=False)
        keep_idx.append(idx)
    keep_idx = np.sort(np.concatenate(keep_idx))
    X = X[keep_idx]
    meta = meta.iloc[keep_idx].reset_index(drop=True)
    print(f"After per-patient subsampling (cap={MAX_CELLS_PER_PATIENT}): {X.shape[0]} cells")
    print(meta["Patient"].value_counts())
    print(meta["Major cell type"].value_counts())

    # data is already "normalized" per filename; log1p if it looks linear-scale
    is_log_scale = X.max() < 30
    if not is_log_scale:
        print("Applying log1p (data appears to be on a linear scale)...")
        X = X.copy()
        X.data = np.log1p(X.data)
    else:
        print("Data appears already log-scale, skipping log1p.")

    print(f"Computing PCA (n_components={N_PCS})...")
    pca = PCA(n_components=N_PCS, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X.toarray())
    print(f"  explained variance ratio (sum): {pca.explained_variance_ratio_.sum():.3f}")

    adata = ad.AnnData(X=X_pca.astype(np.float32))
    adata.obs["cell_type"] = meta["Major cell type"].values
    adata.obs["donor"] = meta["Patient"].values  # "donor" name kept for script reuse
    adata.obs["tissue"] = meta["Tissue"].values
    adata.var_names = [f"pc_{i}" for i in range(N_PCS)]
    adata.var["features"] = adata.var_names

    adata.write_h5ad(OUT_PATH)
    print(f"Saved -> {OUT_PATH} shape={adata.shape}")


if __name__ == "__main__":
    main()
