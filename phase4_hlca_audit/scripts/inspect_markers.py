import anndata as ad
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "hlca_core.h5ad"

adata = ad.read_h5ad(DATA_PATH, backed="r")
print("var columns:", adata.var.columns.tolist())
print(adata.var.head())
print("\nobs columns:", adata.obs.columns.tolist())

markers = ["KRT5", "TP63", "KRT14", "FOXJ1", "TUBB4B", "PIFO", "SCGB1A1", "SCGB3A1", "MUC5B", "MUC5AC", "BPIFB1"]
for m in markers:
    hit_symbol = m in adata.var_names
    hit_featname = (adata.var.get("feature_name", None) is not None) and (adata.var["feature_name"].astype(str) == m).any()
    print(f"{m}: in var_names={hit_symbol}, in feature_name col={hit_featname}")
