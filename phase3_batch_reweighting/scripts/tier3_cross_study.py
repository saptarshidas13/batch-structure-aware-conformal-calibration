"""
Tier 3: batch-structure-aware weighted conformal prediction across a
genuine cross-study shift -- GSE178360 (healthy lung) vs. GSE127465 (tumor
lung). Two directions are tested (healthy-as-reference/tumor-as-query, and
the reverse), each preserving per-sample/per-patient sub-batch identity so
structure-aware weighting has real structure to exploit. A shared PCA is
fit on the reference direction's cells only and applied to both, since the
two studies only share a common gene space, not a common embedding.
"""
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
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

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "tier3_prepared.h5ad"
ALPHAS = [0.01, 0.05, 0.10]
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
N_PCS = 50


def run_direction(ref_study: str, query_study: str, adata: ad.AnnData, seed: int = 42) -> dict:
    print(f"\n=== DIRECTION: reference={ref_study}  query={query_study} ===")
    rng = np.random.default_rng(seed)

    ref = adata[adata.obs["study"] == ref_study]
    query = adata[adata.obs["study"] == query_study]

    X_ref_raw = ref.X.toarray() if hasattr(ref.X, "toarray") else ref.X
    y_ref = ref.obs["broad_cell_type"].astype(str).values
    batch_ref = ref.obs["donor"].astype(str).values

    X_query_raw = query.X.toarray() if hasattr(query.X, "toarray") else query.X
    y_query = query.obs["broad_cell_type"].astype(str).values
    batch_query = query.obs["donor"].astype(str).values

    pca = PCA(n_components=N_PCS, random_state=seed)
    X_ref_pca = pca.fit_transform(X_ref_raw)
    X_query_pca = pca.transform(X_query_raw)
    print(f"  PCA explained variance (ref-fit): {pca.explained_variance_ratio_.sum():.3f}")

    X_train, X_cal, y_train, y_cal, _, batch_cal = train_test_split(
        X_ref_pca, y_ref, batch_ref, test_size=0.3, random_state=seed, stratify=y_ref
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_cal_s = scaler.transform(X_cal)
    X_query_s = scaler.transform(X_query_pca)

    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X_train_s, y_train)
    class_to_idx = {c: i for i, c in enumerate(clf.classes_)}

    cal_true_idx = np.array([class_to_idx[y] for y in y_cal])
    query_true_idx = np.array([class_to_idx.get(y, -1) for y in y_query])
    valid = query_true_idx >= 0
    print(
        f"  train={len(X_train_s)} cal={len(X_cal_s)} query={len(X_query_s)} "
        f"(valid: {valid.sum()}/{len(valid)})"
    )

    probs_cal = clf.predict_proba(X_cal_s)
    probs_query = clf.predict_proba(X_query_s)[valid]
    query_true_idx = query_true_idx[valid]
    X_query_s_valid = X_query_s[valid]
    batch_query_valid = batch_query[valid]

    cal_scores = aps_scores_calibration(probs_cal, cal_true_idx, rng)
    top1_acc = float(np.mean(clf.predict(X_query_s)[valid] == y_query[valid]))
    print(f"  top-1 accuracy (cross-study): {top1_acc:.4f}")

    results = {
        "reference": ref_study,
        "query": query_study,
        "top1_accuracy": top1_acc,
        "per_alpha": {},
    }

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
            f"domain_clf_naive cov={row['domain_clf_naive']['coverage']:.3f} size={row['domain_clf_naive']['avg_size']:.2f} | "
            f"domain_clf cov={row['domain_clf']['coverage']:.3f} size={row['domain_clf']['avg_size']:.2f} | "
            f"centroid cov={row['structure_aware_centroid']['coverage']:.3f} size={row['structure_aware_centroid']['avg_size']:.2f} | "
            f"propensity cov={row['structure_aware_propensity']['coverage']:.3f} size={row['structure_aware_propensity']['avg_size']:.2f}"
        )
        results["per_alpha"][str(alpha)] = row

    return results


def main():
    adata = ad.read_h5ad(DATA_PATH)

    all_results = [
        run_direction("healthy_GSE178360", "tumor_GSE127465", adata),
        run_direction("tumor_GSE127465", "healthy_GSE178360", adata),
    ]

    out_path = RESULTS_DIR / "tier3_cross_study_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {out_path}")

    rows = []
    for r in all_results:
        direction = f"{r['reference'][:7]}->{r['query'][:5]}"
        for alpha, methods in r["per_alpha"].items():
            for method in [
                "naive",
                "domain_clf_naive",
                "domain_clf",
                "structure_aware_centroid",
                "structure_aware_propensity",
            ]:
                rows.append(
                    {
                        "direction": direction,
                        "alpha": alpha,
                        "method": method,
                        "coverage": methods[method]["coverage"],
                        "avg_size": methods[method]["avg_size"],
                        "target_coverage": methods["target_coverage"],
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "tier3_cross_study_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("SUMMARY (both directions)")
    print("=" * 70)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
