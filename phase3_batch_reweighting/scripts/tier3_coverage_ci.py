"""
Peer-review response: bootstrap confidence intervals for Tier 3 coverage.

Reuses the exact same fold-fitting pipeline as tier3_cross_study.py (same
seed, same PCA/scaler/classifier/calibration steps), so the per-cell
predictions are identical to the main-text run. The only addition is a
nonparametric bootstrap over the query cells: for each method/alpha/
direction, the query set's per-cell covered/not-covered indicator is
resampled with replacement B times, and the resulting distribution of
mean-coverage values gives a 95% percentile confidence interval on the
achieved coverage rate. This answers "how much would this coverage number
move if we saw a new query sample from the same distribution", which is the
relevant uncertainty question given that each fold's query set is a fixed
few-thousand-cell sample, not a repeated experiment.
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
    evaluate_coverage_percell,
    naive_weights,
    structure_aware_propensity_weights,
)

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "tier3_prepared.h5ad"
ALPHAS = [0.01, 0.05, 0.10]
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
N_PCS = 50
N_BOOTSTRAP = 2000
METHODS = ["naive", "structure_aware_propensity", "domain_clf"]


def bootstrap_ci(covered: np.ndarray, n_boot: int, rng: np.random.Generator):
    n = len(covered)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = covered[idx].mean()
    lo, med, hi = np.percentile(boot_means, [2.5, 50, 97.5])
    return float(lo), float(med), float(hi)


def run_direction(ref_study: str, query_study: str, adata: ad.AnnData, seed: int = 42) -> list:
    print(f"\n=== DIRECTION: reference={ref_study}  query={query_study} ===")
    rng = np.random.default_rng(seed)
    boot_rng = np.random.default_rng(seed + 1000)

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

    probs_cal = clf.predict_proba(X_cal_s)
    probs_query = clf.predict_proba(X_query_s)[valid]
    query_true_idx = query_true_idx[valid]
    X_query_s_valid = X_query_s[valid]
    batch_query_valid = batch_query[valid]

    cal_scores = aps_scores_calibration(probs_cal, cal_true_idx, rng)

    weights_by_method = {
        "naive": naive_weights(len(X_cal_s), len(X_query_s_valid)),
        "domain_clf": domain_classifier_weights(X_cal_s, X_query_s_valid),
        "structure_aware_propensity": structure_aware_propensity_weights(
            X_cal_s, batch_cal, X_query_s_valid, batch_query_valid
        ),
    }

    direction_label = f"{ref_study[:7]}->{query_study[:5]}"
    rows = []
    for alpha in ALPHAS:
        for method in METHODS:
            w_cal, w_test = weights_by_method[method]
            covered, _ = evaluate_coverage_percell(
                cal_scores, w_cal, probs_query, w_test, query_true_idx, alpha, rng
            )
            point_estimate = float(covered.mean())
            lo, med, hi = bootstrap_ci(covered, N_BOOTSTRAP, boot_rng)
            print(f"  alpha={alpha:<5} {method:<28} point={point_estimate:.3f}  "
                  f"95% CI [{lo:.3f}, {hi:.3f}]  n={len(covered)}")
            rows.append({
                "direction": direction_label, "alpha": alpha, "method": method,
                "coverage": point_estimate, "ci_lo": lo, "ci_median": med, "ci_hi": hi,
                "n_query": len(covered),
            })
    return rows


def main():
    adata = ad.read_h5ad(DATA_PATH)
    all_rows = []
    all_rows += run_direction("healthy_GSE178360", "tumor_GSE127465", adata)
    all_rows += run_direction("tumor_GSE127465", "healthy_GSE178360", adata)

    df = pd.DataFrame(all_rows)
    out_path = RESULTS_DIR / "tier3_coverage_ci.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")
    print("\n" + "=" * 90)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
