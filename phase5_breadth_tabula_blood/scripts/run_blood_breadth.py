"""
Phase 5 breadth test: same naive / domain_clf / structure_aware weighted
conformal comparison as Phase 3's Tiers 1-2, now on a third, genuinely
distinct tissue (peripheral blood, Tabula Sapiens) with a different batch
axis (donor+chemistry combo, "donor_assay") -- tests whether the method's
behavior (safe when batch effects are mild, unstable domain_clf baseline)
generalizes beyond pancreas and lung.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "phase3_batch_reweighting" / "scripts"))
from weighted_conformal import (  # noqa: E402
    aps_scores_calibration,
    domain_classifier_weights,
    evaluate_coverage,
    naive_weights,
    structure_aware_weights,
)

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "blood_prepared.h5ad"
ALPHAS = [0.01, 0.05, 0.10]
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MIN_CLASS_COUNT = 30


def run_fold(held_out: str, adata: ad.AnnData, seed: int = 42) -> dict:
    print(f"\n=== FOLD: held-out donor_assay = {held_out} ===")
    rng = np.random.default_rng(seed)

    ref_mask = adata.obs["donor"] != held_out
    adata_ref = adata[ref_mask]
    X_ref = adata_ref.X
    y_ref = adata_ref.obs["cell_type"].astype(str).values
    batch_ref = adata_ref.obs["donor"].astype(str).values

    query = adata[adata.obs["donor"] == held_out]
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

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train_s, y_train)
    class_to_idx = {c: i for i, c in enumerate(clf.classes_)}

    cal_true_idx = np.array([class_to_idx[y] for y in y_cal])
    query_true_idx = np.array([class_to_idx.get(y, -1) for y in y_query])
    valid = query_true_idx >= 0
    print(
        f"  train={len(X_train_s)} cal={len(X_cal_s)} query={len(X_query_s)} "
        f"(valid: {valid.sum()}/{len(valid)}, {len(class_to_idx)} classes)"
    )
    if valid.sum() < 10:
        print("  too few valid query cells, skipping fold")
        return None

    probs_cal = clf.predict_proba(X_cal_s)
    probs_query = clf.predict_proba(X_query_s)[valid]
    query_true_idx = query_true_idx[valid]
    X_query_s_valid = X_query_s[valid]
    batch_query_valid = batch_query[valid]

    cal_scores = aps_scores_calibration(probs_cal, cal_true_idx, rng)
    top1_acc = float(np.mean(clf.predict(X_query_s)[valid] == y_query[valid]))
    print(f"  top-1 accuracy: {top1_acc:.4f}")

    results = {"held_out": held_out, "top1_accuracy": top1_acc, "per_alpha": {}}

    for alpha in ALPHAS:
        row = {"target_coverage": 1 - alpha}

        w_cal, w_test = naive_weights(len(X_cal_s), len(X_query_s_valid))
        cov, size = evaluate_coverage(cal_scores, w_cal, probs_query, w_test, query_true_idx, alpha, rng)
        row["naive"] = {"coverage": cov, "avg_size": size}

        w_cal, w_test = domain_classifier_weights(X_cal_s, X_query_s_valid)
        cov, size = evaluate_coverage(cal_scores, w_cal, probs_query, w_test, query_true_idx, alpha, rng)
        row["domain_clf"] = {"coverage": cov, "avg_size": size}

        w_cal, w_test = structure_aware_weights(X_cal_s, batch_cal, X_query_s_valid, batch_query_valid)
        cov, size = evaluate_coverage(cal_scores, w_cal, probs_query, w_test, query_true_idx, alpha, rng)
        row["structure_aware"] = {"coverage": cov, "avg_size": size}

        print(
            f"  alpha={alpha:<5} target={1 - alpha:.2f} | "
            f"naive cov={row['naive']['coverage']:.3f} size={row['naive']['avg_size']:.2f} | "
            f"domain_clf cov={row['domain_clf']['coverage']:.3f} size={row['domain_clf']['avg_size']:.2f} | "
            f"structure cov={row['structure_aware']['coverage']:.3f} size={row['structure_aware']['avg_size']:.2f}"
        )
        results["per_alpha"][str(alpha)] = row

    return results


def main():
    adata = ad.read_h5ad(DATA_PATH)

    counts = adata.obs["cell_type"].value_counts()
    keep_types = counts[counts >= MIN_CLASS_COUNT].index
    n_before = adata.n_obs
    adata = adata[adata.obs["cell_type"].isin(keep_types)].copy()
    print(f"Dropped rare cell types (<{MIN_CLASS_COUNT}): {n_before} -> {adata.n_obs} cells")

    batches = sorted(adata.obs["donor"].unique())
    print(f"\nBatches (donor_assay, {len(batches)}): {batches}")

    all_results = [r for b in batches if (r := run_fold(b, adata)) is not None]

    out_path = RESULTS_DIR / "blood_breadth_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {out_path}")

    rows = []
    for r in all_results:
        for alpha, methods in r["per_alpha"].items():
            for method in ["naive", "domain_clf", "structure_aware"]:
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
    df.to_csv(RESULTS_DIR / "blood_breadth_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("AGGREGATE (mean across folds)")
    print("=" * 70)
    agg = df.groupby(["alpha", "method"])[["coverage", "avg_size", "target_coverage"]].mean()
    print(agg.to_string())


if __name__ == "__main__":
    main()
