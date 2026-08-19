"""
Tier 3 (step c): does UPSTREAM BATCH INTEGRATION fix the severe cross-study
shift that neither naive nor either reweighting scheme (centroid,
propensity) could fix in tier3_cross_study.py?

Rationale: Tibshirani et al. (2019) weighted conformal prediction can only
REWEIGHT existing calibration points -- it cannot manufacture calibration
structure that isn't there. Tier 3's failure mode (identified in step b) is
a violated positivity/overlap assumption: healthy and tumor lung occupy
near-disjoint regions of the raw PCA embedding (cross-study top-1 accuracy
only 0.50-0.59), so no reweighting of existing points can help. Harmony
(Korsunsky et al. 2019) attacks the problem differently -- rather than
reweighting points in a fixed embedding, it MOVES points in embedding space
to remove batch-associated variation, using only batch identity (here,
"study"), never cell-type labels. This is a legitimate, realistic upstream
step (exactly what atlas-building pipelines like HLCA itself use before any
downstream classifier or conformal calibration).

Design: PCA is computed ONCE on the pooled healthy+tumor cells (unlike
tier3_cross_study.py's reference-only PCA), then Harmony integrates on
"study" as the batch key using both study's cells jointly (a transductive
setup: only batch metadata is used, not cell-type labels -- analogous to
reference mapping). The same downstream split/train/calibrate/query
procedure and weighted_conformal.py methods are then re-run on the
harmonized embedding for both directions, exactly as in tier3_cross_study.py,
so the two scripts are directly comparable.
"""
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from harmonypy import run_harmony
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from weighted_conformal import (  # noqa: E402
    aps_scores_calibration,
    domain_classifier_weights,
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
SEED = 42


def build_harmonized_embedding(adata: ad.AnnData) -> np.ndarray:
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    pca = PCA(n_components=N_PCS, random_state=SEED)
    X_pca = pca.fit_transform(X_s)
    print(f"  pooled PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    meta = adata.obs[["study", "donor"]].copy()
    ho = run_harmony(X_pca, meta, ["study"], random_state=SEED)
    X_harmony = ho.Z_corr.T  # (n_cells, N_PCS)
    return np.asarray(X_harmony)


def run_direction(ref_study: str, query_study: str, adata: ad.AnnData, X_harmony: np.ndarray, seed: int = 42) -> dict:
    print(f"\n=== DIRECTION: reference={ref_study}  query={query_study} ===")
    rng = np.random.default_rng(seed)

    ref_mask = (adata.obs["study"] == ref_study).values
    query_mask = (adata.obs["study"] == query_study).values

    X_ref_h = X_harmony[ref_mask]
    y_ref = adata.obs["broad_cell_type"].astype(str).values[ref_mask]
    batch_ref = adata.obs["donor"].astype(str).values[ref_mask]

    X_query_h = X_harmony[query_mask]
    y_query = adata.obs["broad_cell_type"].astype(str).values[query_mask]
    batch_query = adata.obs["donor"].astype(str).values[query_mask]

    X_train, X_cal, y_train, y_cal, _, batch_cal = train_test_split(
        X_ref_h, y_ref, batch_ref, test_size=0.3, random_state=seed, stratify=y_ref
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_cal_s = scaler.transform(X_cal)
    X_query_s = scaler.transform(X_query_h)

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
    print(f"  top-1 accuracy (cross-study, POST-HARMONY): {top1_acc:.4f}")

    results = {
        "reference": ref_study,
        "query": query_study,
        "top1_accuracy": top1_acc,
        "per_alpha": {},
    }

    weights_by_method = {
        "naive": naive_weights(len(X_cal_s), len(X_query_s_valid)),
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
            f"centroid cov={row['structure_aware_centroid']['coverage']:.3f} size={row['structure_aware_centroid']['avg_size']:.2f} | "
            f"propensity cov={row['structure_aware_propensity']['coverage']:.3f} size={row['structure_aware_propensity']['avg_size']:.2f}"
        )
        results["per_alpha"][str(alpha)] = row

    return results


def main():
    adata = ad.read_h5ad(DATA_PATH)
    print(f"Loaded {adata.shape}")
    print(adata.obs.groupby(["study", "broad_cell_type"]).size())

    print("\nComputing pooled PCA + Harmony integration on study...")
    X_harmony = build_harmonized_embedding(adata)

    all_results = [
        run_direction("healthy_GSE178360", "tumor_GSE127465", adata, X_harmony),
        run_direction("tumor_GSE127465", "healthy_GSE178360", adata, X_harmony),
    ]

    out_path = RESULTS_DIR / "tier3_harmony_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {out_path}")

    rows = []
    for r in all_results:
        direction = f"{r['reference'][:7]}->{r['query'][:5]}"
        for alpha, methods in r["per_alpha"].items():
            for method in ["naive", "domain_clf", "structure_aware_centroid", "structure_aware_propensity"]:
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
    df.to_csv(RESULTS_DIR / "tier3_harmony_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("SUMMARY (both directions, post-Harmony)")
    print("=" * 70)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
