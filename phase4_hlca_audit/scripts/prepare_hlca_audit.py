"""
Prepare HLCA core for the audit: subsample per contributing dataset (finer
than "study" -- distinguishes chemistry version too, e.g. Jain_Misharin
_2021_10Xv1 vs _10Xv2, a genuine batch axis), keep the columns needed for
the audit's actual point -- HLCA's own `reannotation_type` column
(Correctly annotated / Misannotated / Underannotated), which independently
flags cells where each contributing study's ORIGINAL label disagreed with
HLCA's integrated consensus label. This lets the audit be validated against
real, pre-existing expert curation instead of discovering "errors" blind.
"""
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "hlca_core.h5ad"
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "hlca_audit_prepared.h5ad"

MAX_CELLS_PER_DATASET = 2000
N_PCS = 50
RANDOM_STATE = 42

KEEP_OBS_COLS = [
    "dataset",
    "study",
    "ann_level_3",
    "cell_type",
    "reannotation_type",
    "original_ann_level_3",
    "sequencing_platform",
    "assay",
    "fresh_or_frozen",
    "tissue_dissociation_protocol",
]


def main():
    print("Opening HLCA core in backed mode...")
    adata = ad.read_h5ad(DATA_PATH, backed="r")
    print(f"Full shape: {adata.shape}")

    # drop cells with no ann_level_3 / reannotation_type info
    valid = (
        adata.obs["ann_level_3"].astype(str).ne("None")
        & adata.obs["reannotation_type"].notna()
    )
    print(f"Cells with valid ann_level_3 + reannotation_type: {valid.sum()} / {len(valid)}")

    rng = np.random.default_rng(RANDOM_STATE)
    keep_idx = []
    obs_valid = adata.obs[valid]
    for dataset, group in obs_valid.groupby("dataset", observed=True):
        idx = group.index
        idx_pos = adata.obs.index.get_indexer(idx)
        if len(idx_pos) > MAX_CELLS_PER_DATASET:
            idx_pos = rng.choice(idx_pos, size=MAX_CELLS_PER_DATASET, replace=False)
        keep_idx.append(idx_pos)
    keep_idx = np.sort(np.concatenate(keep_idx))
    print(f"After per-dataset subsampling (cap={MAX_CELLS_PER_DATASET}): {len(keep_idx)} cells")

    print("Loading subset into memory (this reads only the selected rows)...")
    sub = adata[keep_idx].to_memory()
    print(f"Subset shape: {sub.shape}")
    print("X value range check -- max:", sub.X.max(), " min:", sub.X.min())

    sub = sub[:, sub.var["feature_is_filtered"] == False].copy()  # noqa: E712
    print(f"After dropping filtered features: {sub.shape}")

    is_log_scale = sub.X.max() < 30
    if not is_log_scale:
        print("Applying normalize_total + log1p (data appears to be raw counts)...")
        sc.pp.normalize_total(sub, target_sum=1e4)
        sc.pp.log1p(sub)
    else:
        print("Data appears already log-scale, skipping normalization.")

    print(f"Computing PCA (n_components={N_PCS})...")
    sc.pp.pca(sub, n_comps=N_PCS, random_state=RANDOM_STATE)
    print(f"  explained variance ratio (sum): {sub.uns['pca']['variance_ratio'].sum():.3f}")

    out = ad.AnnData(X=sub.obsm["X_pca"].astype(np.float32))
    for col in KEEP_OBS_COLS:
        out.obs[col] = sub.obs[col].astype(str).values
    out.var_names = [f"pc_{i}" for i in range(N_PCS)]
    out.var["features"] = out.var_names

    out.write_h5ad(OUT_PATH)
    print(f"\nSaved -> {OUT_PATH} shape={out.shape}")
    print("\nreannotation_type distribution:")
    print(out.obs["reannotation_type"].value_counts())
    print("\ndataset distribution:")
    print(out.obs["dataset"].value_counts())
    print("\nann_level_3 distribution:")
    print(out.obs["ann_level_3"].value_counts())


if __name__ == "__main__":
    main()
