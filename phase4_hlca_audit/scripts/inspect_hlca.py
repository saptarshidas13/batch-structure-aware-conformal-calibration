"""
Inspect HLCA core structure in backed mode (doesn't load the full 5.87GB
expression matrix into memory) -- just obs/var metadata, to understand
available cell-type annotation levels and study/donor batch columns before
deciding on subsampling and classification target.
"""
from pathlib import Path

import anndata as ad
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "hlca_core.h5ad"

pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 160)


def main():
    print("Opening in backed mode (metadata only)...")
    adata = ad.read_h5ad(DATA_PATH, backed="r")
    print(f"Shape: {adata.shape}")
    print(f"\nobs columns:\n{list(adata.obs.columns)}")
    print(f"\nvar columns:\n{list(adata.var.columns)}")
    print(f"\nvar_names sample: {adata.var_names[:5].tolist()}")
    print(f"\nX dtype/type: {type(adata.X)}")
    if adata.raw is not None:
        print(f"raw.X present: shape {adata.raw.X.shape}")

    # look for likely annotation-level and batch columns
    for col in adata.obs.columns:
        nunique = adata.obs[col].nunique()
        if 2 <= nunique <= 60:
            print(f"\n--- obs['{col}'] ({nunique} unique) ---")
            print(adata.obs[col].value_counts().head(20))


if __name__ == "__main__":
    main()
