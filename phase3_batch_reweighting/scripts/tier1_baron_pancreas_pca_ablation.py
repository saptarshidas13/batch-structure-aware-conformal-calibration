"""
Tier 1 PCA ablation (peer-review response): does the "safe reweighting"
pattern found in Tier 1 depend on using Geneformer embeddings specifically,
or does it hold with the same PCA-based feature space used in Tiers 2-5?

Tier 1's main-text results use Geneformer-V1-10M embeddings (256-dim,
validated via linear probe), while Tiers 2-5 use PCA (50 components) on
log-normalized expression. A reviewer correctly noted this is a genuine
confound: it is not possible from the main-text results alone to tell
whether Tier 3's severe failure is purely about batch/study shift, or is
partly exacerbated by PCA's linear compression compared to a richer
embedding. This script closes that gap for Tier 1 by rerunning the exact
same leave-one-donor-out weighted-conformal comparison on PCA features
instead of Geneformer features, using the same QC'd Baron pancreas data,
the same four weighting schemes, and the same evaluation code
(weighted_conformal.py) as tier1_baron_pancreas.py. Only the feature space
changes.
"""
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from weighted_conformal import (  # noqa: E402
    aps_scores_calibration,
    domain_classifier_weights,
    domain_classifier_weights_naive,
    evaluate_coverage,
    naive_weights,
    structure_aware_centroid_weights,
    structure_aware_propensity_weights,
)

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "phase1_baseline" / "data" / "processed"
DONORS = ["human1", "human2", "human3", "human4"]
ALPHAS = [0.01, 0.05, 0.10]
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
N_PCS = 50
RANDOM_STATE = 42


def build_pca_embedding():
    """Load the 4 QC'd donor files, restrict to shared genes, normalize,
    log1p, and fit one PCA (50 components) on the pooled data -- matching
    the exact preprocessing convention used for Tiers 2 and 3."""
    parts = []
    for d in DONORS:
        a = ad.read_h5ad(RAW_DATA_DIR / f"baron_{d}.h5ad")
        a.obs_names_make_unique()
        parts.append(a)

    common_genes = set(parts[0].var_names)
    for a in parts[1:]:
        common_genes &= set(a.var_names)
    common_genes = sorted(common_genes)
    print(f"Shared genes across all 4 donors: {len(common_genes)}")

    parts = [a[:, common_genes].copy() for a in parts]
    combined = ad.concat(parts, join="inner")
    print(f"Pooled shape: {combined.shape}")

    is_log_scale = combined.X.max() < 30
    if not is_log_scale:
        sc.pp.normalize_total(combined, target_sum=1e4)
        sc.pp.log1p(combined)
    else:
        print("Data already log-scale, skipping normalization.")

    X_dense = combined.X.toarray() if hasattr(combined.X, "toarray") else np.asarray(combined.X)
    pca = PCA(n_components=N_PCS, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_dense)
    print(f"Pooled PCA explained variance ratio (sum): {pca.explained_variance_ratio_.sum():.3f}")

    out = ad.AnnData(X=X_pca.astype(np.float32))
    out.obs["cell_type"] = combined.obs["cell_type"].astype(str).values
    out.obs["donor"] = combined.obs["donor"].astype(str).values
    return out


def run_fold(held_out: str, adata: ad.AnnData, seed: int = 42) -> dict:
    print(f"\n=== FOLD: held-out donor = {held_out} ===")
    rng = np.random.default_rng(seed)

    ref_mask = (adata.obs["donor"] != held_out).values
    X_ref = adata.X[ref_mask]
    y_ref = adata.obs["cell_type"].astype(str).values[ref_mask]
    batch_ref = adata.obs["donor"].astype(str).values[ref_mask]

    query_mask = (adata.obs["donor"] == held_out).values
    X_query = adata.X[query_mask]
    y_query = adata.obs["cell_type"].astype(str).values[query_mask]
    batch_query = adata.obs["donor"].astype(str).values[query_mask]

    X_train, X_cal, y_train, y_cal, _, batch_cal = train_test_split(
        X_ref, y_ref, batch_ref, test_size=0.3, random_state=42, stratify=y_ref
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_cal_s = scaler.transform(X_cal)
    X_query_s = scaler.transform(X_query)

    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X_train_s, y_train)
    class_to_idx = {c: i for i, c in enumerate(clf.classes_)}

    cal_true_idx = np.array([class_to_idx[y] for y in y_cal])
    query_true_idx = np.array([class_to_idx.get(y, -1) for y in y_query])
    valid = query_true_idx >= 0
    print(f"  train={len(X_train_s)} cal={len(X_cal_s)} query={len(X_query_s)} "
          f"(valid query labels: {valid.sum()}/{len(valid)})")

    probs_cal = clf.predict_proba(X_cal_s)
    probs_query = clf.predict_proba(X_query_s)[valid]
    query_true_idx = query_true_idx[valid]
    X_query_s_valid = X_query_s[valid]
    batch_query_valid = batch_query[valid]

    cal_scores = aps_scores_calibration(probs_cal, cal_true_idx, rng)
    top1_acc = float(np.mean(clf.predict(X_query_s) == y_query))
    print(f"  top-1 accuracy: {top1_acc:.4f}")

    results = {"held_out": held_out, "top1_accuracy": top1_acc, "per_alpha": {}}

    weights_by_method = {
        "naive": naive_weights(len(X_cal_s), len(X_query_s_valid)),
        "domain_clf_naive": domain_classifier_weights_naive(X_cal_s, X_query_s_valid),
        "domain_clf": domain_classifier_weights(X_cal_s, X_query_s_valid),
        "structure_aware_centroid": structure_aware_centroid_weights(
            X_cal_s, batch_cal, X_query_s_valid, batch_query_valid
        ),
        "structure_aware_propensity": structure_aware_propensity_weights(
            X_cal_s, batch_cal, X_query_s_valid, batch_query_valid
        ),
    }

    for alpha in ALPHAS:
        row = {"target_coverage": 1 - alpha}
        for method, (w_cal, w_test) in weights_by_method.items():
            cov, size = evaluate_coverage(cal_scores, w_cal, probs_query, w_test, query_true_idx, alpha, rng)
            row[method] = {"coverage": cov, "avg_size": size}
        print(
            f"  alpha={alpha:<5} target={1 - alpha:.2f} | "
            f"naive cov={row['naive']['coverage']:.3f} size={row['naive']['avg_size']:.2f} | "
            f"domain_clf cov={row['domain_clf']['coverage']:.3f} size={row['domain_clf']['avg_size']:.2f} | "
            f"propensity cov={row['structure_aware_propensity']['coverage']:.3f} size={row['structure_aware_propensity']['avg_size']:.2f}"
        )
        results["per_alpha"][str(alpha)] = row

    return results


def main():
    adata = build_pca_embedding()
    all_results = [run_fold(d, adata) for d in DONORS]

    out_path = RESULTS_DIR / "tier1_pca_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {out_path}")

    rows = []
    for r in all_results:
        for alpha, methods in r["per_alpha"].items():
            for method in [
                "naive", "domain_clf_naive", "domain_clf",
                "structure_aware_centroid", "structure_aware_propensity",
            ]:
                rows.append({
                    "held_out": r["held_out"], "alpha": alpha, "method": method,
                    "coverage": methods[method]["coverage"],
                    "avg_size": methods[method]["avg_size"],
                    "target_coverage": methods["target_coverage"],
                })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "tier1_pca_ablation_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("AGGREGATE (mean across folds) -- PCA feature space")
    print("=" * 70)
    agg = df.groupby(["alpha", "method"])[["coverage", "avg_size", "target_coverage"]].mean()
    print(agg.to_string())

    print("\n" + "=" * 70)
    print("COMPARISON TO GENEFORMER (main text) -- top-1 accuracy per fold")
    print("=" * 70)
    for r in all_results:
        print(f"  {r['held_out']}: PCA top-1 = {r['top1_accuracy']:.4f}")


if __name__ == "__main__":
    main()
