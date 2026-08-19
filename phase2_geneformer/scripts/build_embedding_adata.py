"""
Convert the Geneformer embeddings CSV into per-donor AnnData objects,
matching the structure Phase 1 used (obs["cell_type"], obs["donor"]) so the
same leave-one-donor-out conformal pipeline can be reused with the
underlying_model="torch_net" + obsm path instead of CellTypist.
"""
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
EMB_CSV = BASE_DIR / "results" / "baron_pancreas_embs.csv"
OUT_DIR = BASE_DIR / "data" / "embedding_adata"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(EMB_CSV, index_col=0)
    emb_cols = [c for c in df.columns if c not in ("cell_type", "donor")]
    print(f"Loaded {df.shape[0]} cells, {len(emb_cols)} embedding dims")

    for donor in sorted(df["donor"].unique()):
        sub = df[df["donor"] == donor].reset_index(drop=True)
        X = sub[emb_cols].values.astype(np.float32)

        adata = ad.AnnData(X=X)
        adata.obs_names = [f"{donor}_geneformer_{i}" for i in range(len(sub))]
        adata.obs["cell_type"] = sub["cell_type"].values
        adata.obs["donor"] = sub["donor"].values
        adata.var_names = [f"gf_{i}" for i in range(len(emb_cols))]
        adata.var["features"] = adata.var_names
        adata.obsm["X_geneformer"] = X

        out_path = OUT_DIR / f"baron_{donor}_geneformer.h5ad"
        adata.write_h5ad(out_path)
        print(f"  {donor}: {adata.shape} -> {out_path}")


if __name__ == "__main__":
    main()
