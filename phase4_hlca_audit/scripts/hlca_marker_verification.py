"""
Phase 4, step (d) follow-up: independent molecular corroboration for the top
candidate novel error found by find_novel_errors.py -- cells HLCA calls
"Basal" (reannotation_type = Correctly annotated, i.e. NOT flagged by HLCA's
own curators) that our classifier confidently and reproducibly (3 datasets)
calls "Secretory" instead.

Exact cell-level tracing back to hlca_core.h5ad is not possible: the prepared
audit file (hlca_audit_prepared.h5ad) stores only PCA coordinates and does
not preserve original cell barcodes. Instead, this performs a DISTRIBUTIONAL
check that does not require cell-level correspondence: pull every cell HLCA
calls "Basal" + "Correctly annotated" (respiratory basal cell) in the three
flagged datasets directly from the raw atlas, score each cell on canonical
Basal markers (KRT5, TP63, KRT14) vs. canonical Secretory markers (SCGB1A1,
SCGB3A1, MUC5B, MUC5AC, BPIFB1), and ask: is there a real, sizeable
secretory-marker-high subpopulation within the nominally "Basal, correctly
annotated" cells, comparable in scale to what the classifier flagged
(13/~limited subsample in 3 datasets)? True "Secretory" and true "Basal"
consensus cells in the same datasets are pulled as reference distributions.
"""
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "hlca_core.h5ad"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

DATASETS = ["Barbry_Leroy_2020", "Jain_Misharin_2021_10Xv2", "Krasnow_2020"]
BASAL_MARKERS = ["KRT5", "TP63", "KRT14"]
SECRETORY_MARKERS = ["SCGB1A1", "SCGB3A1", "MUC5B", "MUC5AC", "BPIFB1"]


def main():
    print("Opening HLCA core in backed mode...")
    adata = ad.read_h5ad(DATA_PATH, backed="r")

    fname = adata.var["feature_name"].astype(str)
    marker_ens = {}
    for m in BASAL_MARKERS + SECRETORY_MARKERS:
        hits = adata.var_names[fname == m]
        if len(hits) > 0:
            marker_ens[m] = hits[0]
    print(f"Resolved {len(marker_ens)}/{len(BASAL_MARKERS) + len(SECRETORY_MARKERS)} marker genes:")
    print(marker_ens)

    obs = adata.obs
    in_datasets = obs["dataset"].astype(str).isin(DATASETS)

    basal_mask = (
        in_datasets
        & (obs["cell_type"].astype(str) == "respiratory basal cell")
        & (obs["ann_level_3"].astype(str) == "Basal")
        & (obs["reannotation_type"].astype(str) == "Correctly annotated")
    )
    secretory_mask = (
        in_datasets
        & (obs["ann_level_3"].astype(str) == "Secretory")
        & (obs["reannotation_type"].astype(str) == "Correctly annotated")
    )
    print(f"\n'Basal, correctly annotated, respiratory basal cell' in {DATASETS}: {basal_mask.sum()} cells")
    print(f"'Secretory, correctly annotated' in {DATASETS}: {secretory_mask.sum()} cells")

    gene_cols = list(marker_ens.values())
    gene_names = list(marker_ens.keys())

    def load_scores(mask, label):
        idx = np.where(mask.values)[0]
        print(f"  loading {len(idx)} cells for '{label}'...")
        sub = adata[idx, gene_cols].to_memory()
        X = sub.X.toarray() if hasattr(sub.X, "toarray") else np.asarray(sub.X)
        is_log = X.max() < 30
        if not is_log:
            tmp = ad.AnnData(X=sub.X.copy())
            sc.pp.normalize_total(tmp, target_sum=1e4)
            sc.pp.log1p(tmp)
            X = tmp.X.toarray() if hasattr(tmp.X, "toarray") else np.asarray(tmp.X)
        df = pd.DataFrame(X, columns=gene_names)
        df["group"] = label
        df["dataset"] = sub.obs["dataset"].astype(str).values
        return df

    basal_df = load_scores(basal_mask, "Basal_correctly_annotated")
    secretory_df = load_scores(secretory_mask, "Secretory_correctly_annotated")

    all_df = pd.concat([basal_df, secretory_df], ignore_index=True)
    basal_present = [g for g in BASAL_MARKERS if g in gene_names]
    secretory_present = [g for g in SECRETORY_MARKERS if g in gene_names]
    all_df["basal_score"] = all_df[basal_present].mean(axis=1)
    all_df["secretory_score"] = all_df[secretory_present].mean(axis=1)
    all_df["secretory_minus_basal"] = all_df["secretory_score"] - all_df["basal_score"]

    print("\n" + "=" * 90)
    print("MARKER SCORE SUMMARY BY GROUP")
    print("=" * 90)
    print(all_df.groupby("group")[["basal_score", "secretory_score", "secretory_minus_basal"]].describe().T)

    basal_only = all_df[all_df["group"] == "Basal_correctly_annotated"]
    secretory_high = basal_only[basal_only["secretory_score"] > basal_only["basal_score"]]
    print(f"\nOf {len(basal_only)} 'Basal, correctly annotated' cells in these 3 datasets, "
          f"{len(secretory_high)} ({100 * len(secretory_high) / len(basal_only):.1f}%) "
          f"have secretory_score > basal_score (secretory-marker-dominant despite Basal consensus label).")
    print("\nPer-dataset breakdown:")
    print(
        basal_only.assign(secretory_dominant=basal_only["secretory_score"] > basal_only["basal_score"])
        .groupby("dataset")["secretory_dominant"]
        .agg(["sum", "count", "mean"])
    )

    out_path = RESULTS_DIR / "hlca_marker_verification.csv"
    all_df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
