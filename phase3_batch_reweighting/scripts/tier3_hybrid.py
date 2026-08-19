"""
Hybrid method test: does adding OOD rejection on top of structure-aware
reweighting restore coverage in Tier 3's severe cross-study shift case,
where pure reweighting alone failed (see tier3_cross_study.py results)?

Same data, same reference/query splits, same classifier setup as
tier3_cross_study.py, with one addition: query points are scored for
anomalousness (KNN distance) against the reference, and those with a
conformal p-value below alpha_o are rejected before evaluating coverage on
the survivors. Tests alpha_o in {0.1, 0.3} to show the rejection-rate /
coverage-restoration tradeoff.
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
from hybrid_reject import conformal_pvalues, knn_anomaly_score, reject_mask  # noqa: E402
from weighted_conformal import (  # noqa: E402
    aps_scores_calibration,
    evaluate_coverage,
    naive_weights,
    structure_aware_weights,
)

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "tier3_prepared.h5ad"
ALPHAS = [0.01, 0.05, 0.10]
ALPHA_O_VALUES = [0.1, 0.3]
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

    X_train, X_cal, y_train, y_cal, batch_train, batch_cal = train_test_split(
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
    probs_query_all = clf.predict_proba(X_query_s)
    cal_scores = aps_scores_calibration(probs_cal, cal_true_idx, rng)

    # --- OOD rejection layer ---
    # score X_cal (as the "in-distribution" reference) and X_query against
    # X_train's neighborhood structure
    cal_anomaly = knn_anomaly_score(X_train_s, X_cal_s, k=10)
    query_anomaly = knn_anomaly_score(X_train_s, X_query_s, k=10)
    query_pvalues = conformal_pvalues(cal_anomaly, query_anomaly)

    results = {"reference": ref_study, "query": query_study, "alpha_o_results": {}}

    for alpha_o in ALPHA_O_VALUES:
        rejected = reject_mask(query_pvalues, alpha_o)
        keep = (~rejected) & valid
        n_rejected_of_valid = int((rejected & valid).sum())
        print(
            f"\n  --- alpha_o={alpha_o}: rejected {n_rejected_of_valid}/{valid.sum()} "
            f"valid query cells ({100 * n_rejected_of_valid / valid.sum():.1f}%) ---"
        )

        probs_query = probs_query_all[keep]
        query_true_idx_kept = query_true_idx[keep]
        X_query_s_kept = X_query_s[keep]
        batch_query_kept = batch_query[keep]

        alpha_o_row = {"n_rejected": n_rejected_of_valid, "n_total_valid": int(valid.sum()), "per_alpha": {}}

        for alpha in ALPHAS:
            row = {"target_coverage": 1 - alpha}

            w_cal, w_test = naive_weights(len(X_cal_s), len(X_query_s_kept))
            cov, size = evaluate_coverage(
                cal_scores, w_cal, probs_query, w_test, query_true_idx_kept, alpha, rng
            )
            row["naive"] = {"coverage": cov, "avg_size": size}

            w_cal, w_test = structure_aware_weights(X_cal_s, batch_cal, X_query_s_kept, batch_query_kept)
            cov, size = evaluate_coverage(
                cal_scores, w_cal, probs_query, w_test, query_true_idx_kept, alpha, rng
            )
            row["structure_aware"] = {"coverage": cov, "avg_size": size}

            print(
                f"    alpha={alpha:<5} target={1 - alpha:.2f} | "
                f"naive+reject cov={row['naive']['coverage']:.3f} size={row['naive']['avg_size']:.2f} | "
                f"structure+reject cov={row['structure_aware']['coverage']:.3f} size={row['structure_aware']['avg_size']:.2f}"
            )
            alpha_o_row["per_alpha"][str(alpha)] = row

        results["alpha_o_results"][str(alpha_o)] = alpha_o_row

    return results


def main():
    adata = ad.read_h5ad(DATA_PATH)

    all_results = [
        run_direction("healthy_GSE178360", "tumor_GSE127465", adata),
        run_direction("tumor_GSE127465", "healthy_GSE178360", adata),
    ]

    out_path = RESULTS_DIR / "tier3_hybrid_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {out_path}")

    rows = []
    for r in all_results:
        direction = f"{r['reference'][:7]}->{r['query'][:5]}"
        for alpha_o, alpha_o_data in r["alpha_o_results"].items():
            reject_pct = 100 * alpha_o_data["n_rejected"] / alpha_o_data["n_total_valid"]
            for alpha, methods in alpha_o_data["per_alpha"].items():
                for method in ["naive", "structure_aware"]:
                    rows.append(
                        {
                            "direction": direction,
                            "alpha_o": alpha_o,
                            "reject_pct": reject_pct,
                            "alpha": alpha,
                            "method": method + "+reject",
                            "coverage": methods[method]["coverage"],
                            "avg_size": methods[method]["avg_size"],
                            "target_coverage": methods["target_coverage"],
                        }
                    )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "tier3_hybrid_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("FULL SUMMARY")
    print("=" * 70)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
