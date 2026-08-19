"""Inspect Tabula Sapiens Blood dataset structure in backed mode."""
from pathlib import Path

import anndata as ad
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[2] / "phase4_hlca_audit" / "data_tabula_blood" / "tabula_blood.h5ad"

pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 160)


def main():
    print("Opening in backed mode...")
    adata = ad.read_h5ad(DATA_PATH, backed="r")
    print(f"Shape: {adata.shape}")
    print(f"\nobs columns:\n{list(adata.obs.columns)}")
    print(f"\nvar columns:\n{list(adata.var.columns)}")
    print(f"\nvar_names sample: {adata.var_names[:5].tolist()}")

    for col in adata.obs.columns:
        nunique = adata.obs[col].nunique()
        if 2 <= nunique <= 60:
            print(f"\n--- obs['{col}'] ({nunique} unique) ---")
            print(adata.obs[col].value_counts().head(30))


if __name__ == "__main__":
    main()
