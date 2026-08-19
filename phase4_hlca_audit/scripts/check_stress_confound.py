"""
Follow-up to hlca_marker_verification.py: distinguish the two live
explanations for Krasnow_2020's extreme (76.3%) rate of secretory-marker-
dominant cells among nominally "Basal, correctly annotated" cells --
(i) a genuine transitional basal->secretory biological population, or
(ii) a technical artifact of Krasnow_2020's distinct dissociation protocol
(Collagenase + Elastase + DNAse, vs. Cold protease 1h for the other two
flagged datasets).

Standard check (van den Brink et al. 2017, Nat Methods): enzymatic
dissociation induces a well-characterized immediate-early / stress-response
transcriptional signature (FOS, FOSB, JUN, JUNB, JUND, EGR1, HSPA1A, HSPA1B,
HSP90AA1, DNAJB1, IER2, NR4A1, ZFP36, DUSP1, HSPB1) largely independent of
true cell identity. Two tests:

  1. Dataset-level: is Krasnow_2020's overall stress score (across ALL
     Basal cells, not just the disputed ones) elevated relative to the
     other two datasets? A protocol-driven artifact should show up broadly,
     not just in the disputed subpopulation.
  2. Within-dataset: for Krasnow_2020 specifically, do the secretory-marker-
     dominant "Basal" cells show HIGHER stress scores than the
     basal-marker-dominant "Basal" cells in the SAME dataset (same protocol,
     controlling for the dataset-level confound)? If dissociation stress is
     driving the secretory-like signature, the secretory-dominant subset
     should be the more stressed subset. If stress scores are similar
     between the two subsets, the stress-artifact explanation is not
     supported and the finding is more likely to reflect real biology (or
     at least a different, unidentified technical cause).
"""
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "hlca_core.h5ad"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

DATASETS = ["Barbry_Leroy_2020", "Jain_Misharin_2021_10Xv2", "Krasnow_2020"]
STRESS_GENES = [
    "FOS", "FOSB", "JUN", "JUNB", "JUND", "EGR1", "HSPA1A", "HSPA1B",
    "HSP90AA1", "DNAJB1", "IER2", "NR4A1", "ZFP36", "DUSP1", "HSPB1",
]
BASAL_MARKERS = ["KRT5", "TP63", "KRT14"]
SECRETORY_MARKERS = ["SCGB1A1", "SCGB3A1", "MUC5B", "MUC5AC", "BPIFB1"]


def main():
    print("Opening HLCA core in backed mode...")
    adata = ad.read_h5ad(DATA_PATH, backed="r")
    fname = adata.var["feature_name"].astype(str)

    all_markers = STRESS_GENES + BASAL_MARKERS + SECRETORY_MARKERS
    marker_ens = {}
    for m in all_markers:
        hits = adata.var_names[fname == m]
        if len(hits) > 0:
            marker_ens[m] = hits[0]
    missing = set(all_markers) - set(marker_ens)
    print(f"Resolved {len(marker_ens)}/{len(all_markers)} genes. Missing: {missing}")

    obs = adata.obs
    in_datasets = obs["dataset"].astype(str).isin(DATASETS)
    basal_mask = (
        in_datasets
        & (obs["cell_type"].astype(str) == "respiratory basal cell")
        & (obs["ann_level_3"].astype(str) == "Basal")
        & (obs["reannotation_type"].astype(str) == "Correctly annotated")
    )
    print(f"\n'Basal, correctly annotated, respiratory basal cell' in {DATASETS}: {basal_mask.sum()} cells")

    gene_cols = list(marker_ens.values())
    gene_names = list(marker_ens.keys())

    idx = np.where(basal_mask.values)[0]
    print(f"Loading {len(idx)} cells...")
    sub = adata[idx, gene_cols].to_memory()
    X = sub.X.toarray() if hasattr(sub.X, "toarray") else np.asarray(sub.X)
    is_log = X.max() < 30
    if not is_log:
        tmp = ad.AnnData(X=sub.X.copy())
        sc.pp.normalize_total(tmp, target_sum=1e4)
        sc.pp.log1p(tmp)
        X = tmp.X.toarray() if hasattr(tmp.X, "toarray") else np.asarray(tmp.X)

    df = pd.DataFrame(X, columns=gene_names)
    df["dataset"] = sub.obs["dataset"].astype(str).values

    stress_present = [g for g in STRESS_GENES if g in gene_names]
    basal_present = [g for g in BASAL_MARKERS if g in gene_names]
    secretory_present = [g for g in SECRETORY_MARKERS if g in gene_names]
    df["stress_score"] = df[stress_present].mean(axis=1)
    df["basal_score"] = df[basal_present].mean(axis=1)
    df["secretory_score"] = df[secretory_present].mean(axis=1)
    df["secretory_dominant"] = df["secretory_score"] > df["basal_score"]

    print("\n" + "=" * 90)
    print("TEST 1: dataset-level stress score (ALL Basal cells per dataset)")
    print("=" * 90)
    print(df.groupby("dataset")["stress_score"].describe())

    print("\n" + "=" * 90)
    print("TEST 2: within-dataset, secretory-dominant vs. basal-dominant stress score")
    print("=" * 90)
    for ds in DATASETS:
        sub_ds = df[df["dataset"] == ds]
        if sub_ds["secretory_dominant"].sum() < 5:
            print(f"\n{ds}: too few secretory-dominant cells ({sub_ds['secretory_dominant'].sum()}) for a stable comparison")
            continue
        g = sub_ds.groupby("secretory_dominant")["stress_score"]
        print(f"\n{ds} (n={len(sub_ds)}):")
        print(g.describe())
        from scipy.stats import mannwhitneyu
        a = sub_ds.loc[sub_ds["secretory_dominant"], "stress_score"]
        b = sub_ds.loc[~sub_ds["secretory_dominant"], "stress_score"]
        stat, p = mannwhitneyu(a, b, alternative="two-sided")
        print(f"  Mann-Whitney U (secretory-dominant vs. basal-dominant stress score): p={p:.4g}")

    out_path = RESULTS_DIR / "hlca_stress_confound_check.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
