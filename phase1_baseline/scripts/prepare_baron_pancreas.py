"""
Phase 1 data prep: parse the four Baron et al. 2016 human pancreas donor
CSVs (GSE84133) into per-donor AnnData objects with raw UMI counts, applying
the QC steps described in the paper's Supplementary S4.2 (the three
tractable ones -- doublet removal, mito/ribo/hemoglobin filtering, and
low-detection gene removal; the pancreas-specific ADM ductal-doublet
subcluster removal is deliberately out of scope here, see conversation).

Output: data/processed/baron_human{1,2,3,4}.h5ad
  - X: raw counts (float32, dense -> stored sparse)
  - obs["cell_type"]: assigned_cluster label from the original paper
  - obs["donor"]: human1..human4
  - var["features"]: gene symbol (matches gene_column_name expected by
    the ConformalSCAnnotator repo)
"""
import gzip
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "extracted"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DONORS = {
    "human1": "GSM2230757_human1_umifm_counts.csv.gz",
    "human2": "GSM2230758_human2_umifm_counts.csv.gz",
    "human3": "GSM2230759_human3_umifm_counts.csv.gz",
    "human4": "GSM2230760_human4_umifm_counts.csv.gz",
}


def load_donor(donor_name: str, gz_filename: str) -> ad.AnnData:
    csv_path = RAW_DIR / gz_filename.replace(".gz", "")
    if not csv_path.exists():
        gz_path = RAW_DIR / gz_filename
        with gzip.open(gz_path, "rb") as f_in, open(csv_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    df = pd.read_csv(csv_path, index_col=0)
    # columns: barcode, assigned_cluster, <genes...>
    cell_type = df["assigned_cluster"].astype(str).values
    barcode = df["barcode"].astype(str).values
    gene_names = df.columns[2:]
    X = df[gene_names].values.astype(np.float32)

    adata = ad.AnnData(X=csr_matrix(X))
    adata.obs_names = [f"{donor_name}_{b}" for b in barcode]
    adata.obs["cell_type"] = cell_type
    adata.obs["donor"] = donor_name
    adata.var["features"] = gene_names.values
    adata.var_names = gene_names.values
    adata.var_names_make_unique()

    adata = apply_qc(adata)

    return adata


def apply_qc(adata: ad.AnnData) -> ad.AnnData:
    """Paper's S4.2 QC pipeline (tractable subset): mito/ribo/hgb ratio
    filtering, Scrublet doublet removal, low-detection gene removal."""

    # --- mito / ribo / hemoglobin ratio filtering ---
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    adata.var["hb"] = adata.var_names.str.contains(r"^HB[^(P)]", regex=True)
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt", "ribo", "hb"], percent_top=None, log1p=False, inplace=True
    )
    n_before = adata.n_obs
    adata = adata[
        (adata.obs["pct_counts_mt"] <= 10)
        & (adata.obs["pct_counts_ribo"] <= 60)
        & (adata.obs["pct_counts_hb"] <= 60)
    ].copy()
    print(f"    QC (mito<=10%, ribo<=60%, hgb<=60%): {n_before} -> {adata.n_obs} cells")

    # --- doublet detection (Scrublet) ---
    n_before = adata.n_obs
    sc.pp.scrublet(adata, verbose=False)
    keep = (~adata.obs["predicted_doublet"].values) & np.isfinite(
        adata.obs["doublet_score"].values
    )
    adata = adata[keep].copy()
    print(f"    Doublet filter (Scrublet): {n_before} -> {adata.n_obs} cells")

    # NOTE: the paper's "genes detected in <3 cells removed" step is applied
    # later, on the assembled reference for each fold (run_phase1_baseline.py)
    # -- not per-donor here. Filtering genes independently per donor would
    # leave each donor with a different gene panel, and reference/query
    # would no longer share the same feature space once folds are built.

    return adata


def main():
    for donor_name, gz_filename in DONORS.items():
        print(f"Processing {donor_name} ...")
        adata = load_donor(donor_name, gz_filename)
        print(f"  shape: {adata.shape}, cell types: {adata.obs['cell_type'].nunique()}")
        out_path = OUT_DIR / f"baron_{donor_name}.h5ad"
        adata.write_h5ad(out_path)
        print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
