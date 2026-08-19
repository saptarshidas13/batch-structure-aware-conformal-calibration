"""
The audit: leave-one-dataset-out structure-aware weighted conformal
prediction across HLCA core, using ann_level_3 as the classification
target and `dataset` as the batch covariate. The actual audit question:
does our calibration method's uncertainty signal (nonconformity score,
prediction-set size, top-1 correctness) systematically differ between
cells HLCA's own curators marked "Correctly annotated" vs "Misannotated"
vs "Underannotated" (the `reannotation_type` column -- comparing each
contributing study's ORIGINAL label to HLCA's integrated consensus label)?

If uncertainty is higher for Misannotated/Underannotated cells, that's a
real, independently-validated finding: our calibration signal tracks
genuine annotation problems the field's own experts already identified.
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
    evaluate_coverage,
    naive_weights,
    structure_aware_weights,
)

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "hlca_audit_prepared.h5ad"
ALPHAS = [0.05, 0.10]
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MIN_CLASS_COUNT = 30


def per_cell_nonconformity(probs: np.ndarray, true_idx: np.ndarray) -> np.ndarray:
    """Deterministic APS-style score per cell (no randomization needed here
    -- this is for the audit's descriptive uncertainty analysis, not for
    building calibrated prediction sets, so conservativeness bias doesn't
    matter; we only compare relative scores across reannotation_type groups)."""
    order = np.argsort(-probs, axis=1)
    ranks = np.argsort(order, axis=1)
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cumsum = np.cumsum(sorted_probs, axis=1)
    true_rank = ranks[np.arange(len(true_idx)), true_idx]
    return cumsum[np.arange(len(true_idx)), true_rank]


def run_fold(held_out: str, adata: ad.AnnData, seed: int = 42) -> dict:
    print(f"\n=== FOLD: held-out dataset = {held_out} ===")
    rng = np.random.default_rng(seed)

    ref_mask = adata.obs["dataset"] != held_out
    adata_ref = adata[ref_mask]
    X_ref = adata_ref.X
    y_ref = adata_ref.obs["ann_level_3"].astype(str).values
    batch_ref = adata_ref.obs["dataset"].astype(str).values

    query = adata[adata.obs["dataset"] == held_out]
    X_query = query.X
    y_query = query.obs["ann_level_3"].astype(str).values
    reannot_query = query.obs["reannotation_type"].astype(str).values
    batch_query = np.full(query.n_obs, held_out)

    X_train, X_cal, y_train, y_cal, _, batch_cal = train_test_split(
        X_ref, y_ref, batch_ref, test_size=0.3, random_state=seed, stratify=y_ref
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

    probs_cal = clf.predict_proba(X_cal_s)
    probs_query = clf.predict_proba(X_query_s)[valid]
    query_true_idx_v = query_true_idx[valid]
    X_query_s_v = X_query_s[valid]
    batch_query_v = batch_query[valid]
    reannot_query_v = reannot_query[valid]

    cal_scores = aps_scores_calibration(probs_cal, cal_true_idx, rng)
    top1_correct = clf.predict(X_query_s)[valid] == y_query[valid]
    top1_acc = float(np.mean(top1_correct))
    print(f"  top-1 accuracy: {top1_acc:.4f}")

    query_nonconformity = per_cell_nonconformity(probs_query, query_true_idx_v)

    fold_result = {
        "held_out_dataset": held_out,
        "n_query_valid": int(valid.sum()),
        "top1_accuracy": top1_acc,
        "per_cell": {
            "reannotation_type": reannot_query_v.tolist(),
            "nonconformity_score": query_nonconformity.tolist(),
            "top1_correct": top1_correct.tolist(),
        },
        "per_alpha": {},
    }

    for alpha in ALPHAS:
        row = {"target_coverage": 1 - alpha}
        for method_name, weight_fn in [
            ("naive", lambda: naive_weights(len(X_cal_s), len(X_query_s_v))),
            (
                "structure_aware",
                lambda: structure_aware_weights(X_cal_s, batch_cal, X_query_s_v, batch_query_v),
            ),
        ]:
            w_cal, w_test = weight_fn()
            # per-cell set sizes needed for the reannotation-type breakdown too
            n_test = len(X_query_s_v)
            set_sizes = np.zeros(n_test, dtype=int)
            covered = np.zeros(n_test, dtype=bool)
            from weighted_conformal import aps_prediction_set, weighted_quantile_threshold

            for i in range(n_test):
                tau = weighted_quantile_threshold(cal_scores, w_cal, w_test[i], alpha)
                pset = aps_prediction_set(probs_query[i], tau, rng)
                set_sizes[i] = len(pset)
                covered[i] = query_true_idx_v[i] in pset
            row[method_name] = {
                "coverage": float(covered.mean()),
                "avg_size": float(set_sizes.mean()),
                "set_sizes": set_sizes.tolist(),
            }
        print(
            f"  alpha={alpha:<5} naive cov={row['naive']['coverage']:.3f} size={row['naive']['avg_size']:.2f} | "
            f"structure cov={row['structure_aware']['coverage']:.3f} size={row['structure_aware']['avg_size']:.2f}"
        )
        fold_result["per_alpha"][str(alpha)] = row

    return fold_result


def main():
    adata = ad.read_h5ad(DATA_PATH)

    counts = adata.obs["ann_level_3"].value_counts()
    keep_types = counts[counts >= MIN_CLASS_COUNT].index
    n_before = adata.n_obs
    adata = adata[adata.obs["ann_level_3"].isin(keep_types)].copy()
    print(f"Dropped rare ann_level_3 types (<{MIN_CLASS_COUNT}): {n_before} -> {adata.n_obs} cells")

    datasets = sorted(adata.obs["dataset"].unique())
    print(f"Datasets ({len(datasets)}): {datasets}")

    all_results = [run_fold(d, adata) for d in datasets]

    out_path = RESULTS_DIR / "hlca_audit_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {out_path}")

    # ==== THE KEY AUDIT ANALYSIS ====
    print("\n" + "=" * 70)
    print("AUDIT: uncertainty signal vs. HLCA's own reannotation_type")
    print("=" * 70)

    rows = []
    for r in all_results:
        for reannot, score, correct in zip(
            r["per_cell"]["reannotation_type"],
            r["per_cell"]["nonconformity_score"],
            r["per_cell"]["top1_correct"],
        ):
            rows.append(
                {
                    "held_out_dataset": r["held_out_dataset"],
                    "reannotation_type": reannot,
                    "nonconformity_score": score,
                    "top1_correct": correct,
                }
            )
    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(RESULTS_DIR / "hlca_audit_per_cell.csv", index=False)

    summary = audit_df.groupby("reannotation_type").agg(
        n_cells=("nonconformity_score", "size"),
        mean_nonconformity=("nonconformity_score", "mean"),
        median_nonconformity=("nonconformity_score", "median"),
        top1_accuracy=("top1_correct", "mean"),
    )
    print(summary.to_string())
    summary.to_csv(RESULTS_DIR / "hlca_audit_summary_by_reannotation_type.csv")

    # coverage/set-size summary by method+alpha
    cov_rows = []
    for r in all_results:
        for alpha, methods in r["per_alpha"].items():
            for method in ["naive", "structure_aware"]:
                cov_rows.append(
                    {
                        "held_out_dataset": r["held_out_dataset"],
                        "alpha": alpha,
                        "method": method,
                        "coverage": methods[method]["coverage"],
                        "avg_size": methods[method]["avg_size"],
                        "target_coverage": methods["target_coverage"],
                    }
                )
    cov_df = pd.DataFrame(cov_rows)
    cov_df.to_csv(RESULTS_DIR / "hlca_audit_coverage_summary.csv", index=False)
    print("\n" + "=" * 70)
    print("AGGREGATE COVERAGE (mean across dataset folds)")
    print("=" * 70)
    print(cov_df.groupby(["alpha", "method"])[["coverage", "avg_size", "target_coverage"]].mean().to_string())


if __name__ == "__main__":
    main()
