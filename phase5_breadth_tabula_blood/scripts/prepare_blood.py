"""
Prepare Tabula Sapiens Blood for the breadth test: subsample per
donor_assay (donor + chemistry combo -- genuine technical batch unit,
mirroring HLCA's "dataset" granularity), restricted to 10x assays only
(drops the small Smart-seq2 fraction to avoid conflating platform types
in one PCA/classifier space), PCA to a shared embedding.
"""
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc

DATA_PATH = Path(__file__).resolve().parents[2] / "phase4_hlca_audit" / "data_tabula_blood" / "tabula_blood.h5ad"
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "blood_prepared.h5ad"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

MAX_CELLS_PER_BATCH = 2000
N_PCS = 50
RANDOM_STATE = 42


def main():
    print("Opening Tabula Sapiens Blood in backed mode...")
    adata = ad.read_h5ad(DATA_PATH, backed="r")
    print(f"Full shape: {adata.shape}")

    valid = adata.obs["assay"].isin(["10x 3' v3", "10x 5' v2"]) & adata.obs["cell_type"].notna()
    print(f"10x-assay cells with valid cell_type: {valid.sum()} / {len(valid)}")

    rng = np.random.default_rng(RANDOM_STATE)
    keep_idx = []
    obs_valid = adata.obs[valid]
    for batch_id, group in obs_valid.groupby("donor_assay", observed=True):
        idx_pos = adata.obs.index.get_indexer(group.index)
        if len(idx_pos) > MAX_CELLS_PER_BATCH:
            idx_pos = rng.choice(idx_pos, size=MAX_CELLS_PER_BATCH, replace=False)
        keep_idx.append(idx_pos)
    keep_idx = np.sort(np.concatenate(keep_idx))
    print(f"After per-donor_assay subsampling (cap={MAX_CELLS_PER_BATCH}): {len(keep_idx)} cells")

    print("Loading subset into memory...")
    sub = adata[keep_idx].to_memory()
    print(f"Subset shape: {sub.shape}")
    print("X value range -- max:", sub.X.max(), " min:", sub.X.min())

    sub = sub[:, sub.var["feature_is_filtered"] == False].copy()  # noqa: E712
    print(f"After dropping filtered features: {sub.shape}")

    is_log_scale = sub.X.max() < 30
    if not is_log_scale:
        print("Applying normalize_total + log1p...")
        sc.pp.normalize_total(sub, target_sum=1e4)
        sc.pp.log1p(sub)
    else:
        print("Data appears already log-scale, skipping normalization.")

    print(f"Computing PCA (n_components={N_PCS})...")
    sc.pp.pca(sub, n_comps=N_PCS, random_state=RANDOM_STATE)
    print(f"  explained variance ratio (sum): {sub.uns['pca']['variance_ratio'].sum():.3f}")

    out = ad.AnnData(X=sub.obsm["X_pca"].astype(np.float32))
    out.obs["cell_type"] = sub.obs["cell_type"].astype(str).values
    out.obs["donor"] = sub.obs["donor_assay"].astype(str).values  # batch unit for structure_aware
    out.obs["donor_id"] = sub.obs["donor_id"].astype(str).values
    out.var_names = [f"pc_{i}" for i in range(N_PCS)]
    out.var["features"] = out.var_names

    out.write_h5ad(OUT_PATH)
    print(f"\nSaved -> {OUT_PATH} shape={out.shape}")
    print("\ndonor_assay (batch) distribution:")
    print(out.obs["donor"].value_counts())
    print("\ncell_type distribution:")
    print(out.obs["cell_type"].value_counts())


if __name__ == "__main__":
    main()
