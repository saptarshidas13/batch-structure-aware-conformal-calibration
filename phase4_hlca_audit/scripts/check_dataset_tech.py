import anndata as ad
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "hlca_core.h5ad"
adata = ad.read_h5ad(DATA_PATH, backed="r")
obs = adata.obs

for ds in ["Barbry_Leroy_2020", "Jain_Misharin_2021_10Xv2", "Krasnow_2020"]:
    sub = obs[obs["dataset"].astype(str) == ds]
    print(f"\n=== {ds} (n={len(sub)}) ===")
    print("assay:", sub["assay"].value_counts().to_dict())
    print("sequencing_platform:", sub["sequencing_platform"].value_counts().to_dict())
    print("fresh_or_frozen:", sub["fresh_or_frozen"].value_counts().to_dict())
    print("tissue_dissociation_protocol:", sub["tissue_dissociation_protocol"].value_counts().to_dict())
