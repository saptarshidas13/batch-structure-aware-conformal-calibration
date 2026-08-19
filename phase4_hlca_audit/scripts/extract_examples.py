"""
Follow-up to run_hlca_audit.py: rerun the same leave-one-dataset-out loop,
but this time save full per-cell detail (predicted label, original
per-study label, HLCA consensus label, prediction sets) so we can pull
concrete, traceable illustrative examples of specific misannotated cells
-- not just aggregate statistics.

Three patterns are specifically checked for each Misannotated cell:
  A) predicted == original (wrong) label, != consensus
     -> our independent classifier makes the SAME mistake the original
        study did; suggests a genuinely hard boundary case that likely
        needed expert/manual correction, not just better automated calling
  B) predicted == consensus (correct) label, != original
     -> our classifier independently "catches" the correction HLCA made
  C) predicted == neither
     -> a third, different confusion
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
from weighted_conformal import aps_prediction_set, weighted_quantile_threshold  # noqa: E402
from weighted_conformal import aps_scores_calibration, naive_weights, structure_aware_weights  # noqa: E402

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "hlca_audit_prepared.h5ad"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
ALPHA = 0.10
MIN_CLASS_COUNT = 30


def run_fold(held_out: str, adata: ad.AnnData, seed: int = 42) -> pd.DataFrame:
    print(f"=== FOLD: held-out dataset = {held_out} ===")
    rng = np.random.default_rng(seed)

    ref_mask = adata.obs["dataset"] != held_out
    adata_ref = adata[ref_mask]
    X_ref = adata_ref.X
    y_ref = adata_ref.obs["ann_level_3"].astype(str).values
    batch_ref = adata_ref.obs["dataset"].astype(str).values

    query = adata[adata.obs["dataset"] == held_out]
    X_query = query.X
    y_query_consensus = query.obs["ann_level_3"].astype(str).values
    y_query_original = query.obs["original_ann_level_3"].astype(str).values
    reannot_query = query.obs["reannotation_type"].astype(str).values
    cell_type_query = query.obs["cell_type"].astype(str).values
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
    query_true_idx = np.array([class_to_idx.get(y, -1) for y in y_query_consensus])
    valid = query_true_idx >= 0

    probs_cal = clf.predict_proba(X_cal_s)
    probs_query = clf.predict_proba(X_query_s)[valid]
    cal_scores = aps_scores_calibration(probs_cal, cal_true_idx, rng)

    preds = clf.predict(X_query_s)[valid]
    pred_probs = probs_query.max(axis=1)

    X_query_s_v = X_query_s[valid]
    batch_query_v = batch_query[valid]
    w_cal_struct, w_test_struct = structure_aware_weights(X_cal_s, batch_cal, X_query_s_v, batch_query_v)
    w_cal_naive, w_test_naive = naive_weights(len(X_cal_s), len(X_query_s_v))

    set_sizes_naive = np.zeros(len(X_query_s_v), dtype=int)
    set_sizes_struct = np.zeros(len(X_query_s_v), dtype=int)
    for i in range(len(X_query_s_v)):
        tau_n = weighted_quantile_threshold(cal_scores, w_cal_naive, w_test_naive[i], ALPHA)
        set_sizes_naive[i] = len(aps_prediction_set(probs_query[i], tau_n, rng))
        tau_s = weighted_quantile_threshold(cal_scores, w_cal_struct, w_test_struct[i], ALPHA)
        set_sizes_struct[i] = len(aps_prediction_set(probs_query[i], tau_s, rng))

    df = pd.DataFrame(
        {
            "dataset": held_out,
            "cell_type_full": cell_type_query[valid],
            "consensus_label": y_query_consensus[valid],
            "original_label": y_query_original[valid],
            "reannotation_type": reannot_query[valid],
            "predicted_label": preds,
            "pred_confidence": pred_probs,
            "set_size_naive": set_sizes_naive,
            "set_size_structure": set_sizes_struct,
        }
    )
    return df


def main():
    adata = ad.read_h5ad(DATA_PATH)
    counts = adata.obs["ann_level_3"].value_counts()
    keep_types = counts[counts >= MIN_CLASS_COUNT].index
    adata = adata[adata.obs["ann_level_3"].isin(keep_types)].copy()

    datasets = sorted(adata.obs["dataset"].unique())
    all_dfs = [run_fold(d, adata) for d in datasets]
    full_df = pd.concat(all_dfs, ignore_index=True)

    out_path = RESULTS_DIR / "hlca_examples_full.csv"
    full_df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}  ({len(full_df)} cells)")

    misann = full_df[full_df["reannotation_type"] == "Misannotated"].copy()
    misann["pattern"] = np.select(
        [
            misann["predicted_label"] == misann["original_label"],
            misann["predicted_label"] == misann["consensus_label"],
        ],
        ["A_matches_original_wrong_label", "B_catches_correction"],
        default="C_predicts_something_else",
    )

    print("\n" + "=" * 70)
    print("PATTERN BREAKDOWN (Misannotated cells only)")
    print("=" * 70)
    print(misann["pattern"].value_counts())
    print(misann["pattern"].value_counts(normalize=True))

    print("\n" + "=" * 70)
    print("TOP (original_label -> consensus_label) CONFUSION PAIRS, Misannotated cells")
    print("=" * 70)
    pair_counts = misann.groupby(["original_label", "consensus_label"]).size().sort_values(ascending=False)
    print(pair_counts.head(15))

    print("\n" + "=" * 70)
    print("For the top confusion pair, how often does our classifier's prediction")
    print("match the (original wrong) vs (consensus correct) label?")
    print("=" * 70)
    top_pair = pair_counts.index[0]
    subset = misann[
        (misann["original_label"] == top_pair[0]) & (misann["consensus_label"] == top_pair[1])
    ]
    print(f"Pair: original='{top_pair[0]}' -> consensus='{top_pair[1]}'  (n={len(subset)})")
    print(subset["pattern"].value_counts())
    print("\nSample rows:")
    print(
        subset[
            ["dataset", "original_label", "consensus_label", "predicted_label", "pred_confidence",
             "set_size_naive", "set_size_structure"]
        ].head(10).to_string(index=False)
    )

    misann.to_csv(RESULTS_DIR / "hlca_misannotated_with_patterns.csv", index=False)
    print(f"\nSaved -> {RESULTS_DIR / 'hlca_misannotated_with_patterns.csv'}")


if __name__ == "__main__":
    main()
