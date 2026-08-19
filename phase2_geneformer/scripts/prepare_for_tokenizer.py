"""
Add Geneformer's required fields to the (QC'd) Baron donor h5ad files:
  - var["ensembl_id"]: mapped gene symbol -> Ensembl ID, unmapped genes dropped
  - obs["n_counts"]: total raw UMI counts per cell (rank-value encoding needs this)

Output: a directory of per-donor h5ad files ready for TranscriptomeTokenizer,
which expects a directory of h5ad/loom files, not a single file.
"""
import json
from pathlib import Path

import anndata as ad
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[2] / "phase1_baseline" / "data" / "processed"
MAPPING_PATH = Path(__file__).resolve().parents[1] / "data" / "symbol_to_ensembl.json"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "for_tokenizer"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    with open(MAPPING_PATH) as f:
        mapping = json.load(f)

    for i in range(1, 5):
        donor = f"human{i}"
        adata = ad.read_h5ad(SRC_DIR / f"baron_{donor}.h5ad")

        adata.obs["n_counts"] = np.asarray(adata.X.sum(axis=1)).flatten()

        adata.var["ensembl_id"] = [mapping.get(g) for g in adata.var_names]
        n_before = adata.n_vars
        adata = adata[:, adata.var["ensembl_id"].notna()].copy()
        print(
            f"{donor}: {n_before} -> {adata.n_vars} genes with Ensembl mapping, "
            f"{adata.n_obs} cells"
        )

        out_path = OUT_DIR / f"baron_{donor}.h5ad"
        adata.write_h5ad(out_path)
        print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
