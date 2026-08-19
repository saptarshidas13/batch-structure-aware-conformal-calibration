"""
Tier 1: batch-structure-aware weighted conformal prediction on Baron
pancreas data (donor-as-batch). Reuses the Phase 2 Geneformer embeddings
(validated at 85.9% linear-probe accuracy) purely as a feature space for
classification and for computing batch centroids -- Phase 3 is testing the
CALIBRATION layer, not backbone choice.

Scope note: unlike Phases 1-2, this experiment does NOT apply the Table S8
"simulate OOD by excluding cell types" design -- all 14 cell types are kept
in both reference and query for every fold. This isolates the covariate
(batch) shift correction question cleanly from the separate novel-class-OOD
problem Phases 1-2 addressed.

Compares four weighting schemes at alpha in {0.01, 0.05, 0.10}:
  - naive: standard (unweighted) split conformal prediction
  - domain_clf: generic weighted CP (black-box density-ratio classifier)
  - structure_aware_centroid: batch-structure-aware HEURISTIC (donor
    centroid distance) -- no theorem behind it, kept as a baseline
  - structure_aware_propensity: batch-structure-aware, theoretically
    grounded via generalized propensity scores (see weighted_conformal.py)
"""
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
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

DATA_DIR = Path(__file__).resolve().parents[2] / "phase2_geneformer" / "data" / "embedding_adata"
DONORS = ["human1", "human2", "human3", "human4"]
ALPHAS = [0.01, 0.05, 0.10]
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_all_donors():
    return {d: ad.read_h5ad(DATA_DIR / f"baron_{d}_geneformer.h5ad") for d in DONORS}


def run_fold(held_out: str, donor_data: dict, seed: int = 42) -> dict:
    print(f"\n=== FOLD: held-out donor = {held_out} ===")
    rng = np.random.default_rng(seed)

    Xs, ys, batch_ids = [], [], []
    for d in DONORS:
        if d == held_out:
            continue
        adata = donor_data[d]
        Xs.append(adata.X)
        ys.append(adata.obs["cell_type"].astype(str).values)
        batch_ids.append(np.full(adata.n_obs, d))
    X_ref = np.vstack(Xs)
    y_ref = np.concatenate(ys)
    batch_ref = np.concatenate(batch_ids)

    query = donor_data[held_out]
    X_query = query.X
    y_query = query.obs["cell_type"].astype(str).values
    batch_query = np.full(query.n_obs, held_out)

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

    # weights are alpha-independent -- compute once per fold, reuse across alphas
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
    donor_data = load_all_donors()
    all_results = [run_fold(d, donor_data) for d in DONORS]

    out_path = RESULTS_DIR / "tier1_baron_pancreas_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {out_path}")

    rows = []
    for r in all_results:
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
                        "held_out": r["held_out"],
                        "alpha": alpha,
                        "method": method,
                        "coverage": methods[method]["coverage"],
                        "avg_size": methods[method]["avg_size"],
                        "target_coverage": methods["target_coverage"],
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "tier1_baron_pancreas_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("AGGREGATE (mean across folds)")
    print("=" * 70)
    agg = df.groupby(["alpha", "method"])[["coverage", "avg_size", "target_coverage"]].mean()
    print(agg.to_string())


if __name__ == "__main__":
    main()
