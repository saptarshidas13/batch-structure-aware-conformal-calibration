"""
Phase 2: same leave-one-donor-out conformal cell-type annotation design as
Phase 1, but with the classifier backbone swapped from CellTypist (raw gene
expression) to a small feed-forward net ("torch_net") trained on frozen
Geneformer-V1-10M cell embeddings (obsm path). Everything else -- Table S8
OOD-simulation design, autoencoder OOD detector, APS conformal calibration,
alpha in {0.01, 0.05, 0.1} -- is held identical to Phase 1 for a fair,
direct comparison.

Must be run in annotator-env (needs torchcp / the patched ConformalSCAnnotator
repo from Phase 1), NOT geneformer-env.
"""
import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

PHASE1_REPO_DIR = Path(__file__).resolve().parents[2] / "phase1_baseline" / "repo"
sys.path.insert(0, str(PHASE1_REPO_DIR))

from conformalSC_annotator import ConformalSCAnnotator  # noqa: E402
from torchcp.classification.score import APS  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "embedding_adata"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DONORS = ["human1", "human2", "human3", "human4"]
ALPHAS = [0.01, 0.05, 0.1]

# Identical to Phase 1's Table S8 design (see phase1_baseline/scripts/run_phase1_baseline.py)
OOD_SIMULATED_EXCLUSIONS = {
    "human1": ["quiescent_stellate", "macrophage", "mast", "schwann", "epsilon", "t_cell"],
    "human2": ["macrophage", "mast", "schwann", "epsilon", "t_cell"],
    "human3": ["quiescent_stellate", "gamma", "macrophage", "mast", "schwann", "epsilon", "t_cell"],
    "human4": ["quiescent_stellate", "gamma", "macrophage", "mast", "schwann", "epsilon", "t_cell"],
}

NETWORK_ARCHITECTURE = {
    "hidden_sizes": [128, 64, 32],
    "dropout_rates": [0.15, 0.15, 0.15],
    "learning_rate": 1e-4,
}


def load_all_donors() -> dict:
    data = {}
    for d in DONORS:
        path = DATA_DIR / f"baron_{d}_geneformer.h5ad"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run build_embedding_adata.py first.")
        data[d] = ad.read_h5ad(path)
    return data


def prediction_set_to_list(cell_set):
    if cell_set is None:
        return []
    if isinstance(cell_set, (list, tuple, set, np.ndarray)):
        return list(cell_set)
    return [cell_set]


def run_fold(held_out_donor: str, donor_data: dict, quick: bool) -> dict:
    print(f"\n{'=' * 70}\nFOLD: held-out donor = {held_out_donor}\n{'=' * 70}")

    ref_parts = [donor_data[d] for d in DONORS if d != held_out_donor]
    adata_reference = ad.concat(ref_parts, join="inner", label="donor_src")
    adata_reference.var["features"] = adata_reference.var_names

    excluded_types = OOD_SIMULATED_EXCLUSIONS[held_out_donor]
    n_before = adata_reference.n_obs
    adata_reference = adata_reference[
        ~adata_reference.obs["cell_type"].isin(excluded_types)
    ].copy()
    print(
        f"  Excluded {excluded_types} from reference "
        f"({n_before} -> {adata_reference.n_obs} cells) to simulate OOD, per Table S8"
    )

    adata_query = donor_data[held_out_donor].copy()

    # Standardize embeddings (fit on reference only, applied to both) --
    # the torch_net/obsm path does not normalize input the way the
    # gene-expression paths do, and un-standardized embeddings fed into a
    # small net with a fixed lr=1e-4 converges poorly (see sanity-check
    # comparison against a StandardScaler'd linear probe, which scored
    # 85.9% vs torch_net's ~51% on identical folds).
    scaler = StandardScaler()
    ref_scaled = scaler.fit_transform(adata_reference.X)
    query_scaled = scaler.transform(adata_query.X)
    adata_reference.X = ref_scaled
    adata_reference.obsm["X_geneformer"] = ref_scaled
    adata_query.X = query_scaled
    adata_query.obsm["X_geneformer"] = query_scaled

    n_epochs_ood = 60 if quick else 850
    n_epochs_net = 60 if quick else 1000

    OOD_detector_config = {
        "pvalues": "marginal",
        "alpha": None,
        "delta": 0.1,
        "hidden_sizes": [50, 48, 32, 24],
        "dropout_rates": [0.15, 0.15, 0.15, 0.15],
        "learning_rate": 1e-4,
        "batch_size": 72,
        "n_epochs": n_epochs_ood,
    }

    annotator = ConformalSCAnnotator(
        adata_query,
        var_query_gene_column_name="features",
        underlying_model="torch_net",
    )

    annotator.configure(
        reference_path=adata_reference,
        model_architecture=NETWORK_ARCHITECTURE,
        OOD_detector=OOD_detector_config,
        CP_predictor="standard",
        cell_names_column="cell_type",
        cell_types_excluded_treshold=45,
        test=True,
        alpha=ALPHAS,
        non_conformity_function=APS(),
        epoch=n_epochs_net,
        batch_size=72,
        random_state=42,
    )

    # random_state=42 above only seeds sklearn's internal train/val/test
    # splits -- it does NOT seed PyTorch's weight initialization or the
    # WeightedRandomSampler used by fit_network(), which is the likely
    # cause of the run-to-run instability seen across folds. Seed
    # explicitly right before triggering fit() (via annotate()).
    torch.manual_seed(42)
    np.random.seed(42)

    annotator.annotate(obsm_layer="obsm", obsm="X_geneformer", obsm_OOD="X_geneformer")

    obs = annotator.adata_query.obs
    ground_truth = obs["cell_type"].astype(str).values
    is_id_cell = ~np.isin(ground_truth, excluded_types)
    print(f"  ID cells in query: {is_id_cell.sum()} / {len(ground_truth)}")

    fold_result = {
        "held_out_donor": held_out_donor,
        "n_query_cells": int(len(ground_truth)),
        "test_results": annotator.test_results,
        "ood_performance": annotator.OOD_performance_scores,
        "alpha_OOD": annotator.alpha_OOD,
        "per_alpha": {},
    }

    for alpha in ALPHAS:
        col = f"prediction_sets_{alpha}"
        pred_sets = [prediction_set_to_list(s) for s in obs[col]]
        covered = np.array([gt in pset for gt, pset in zip(ground_truth, pred_sets)])
        set_sizes = np.array([len(pset) for pset in pred_sets])

        coverage_id = float(covered[is_id_cell].mean())
        avg_size_id = float(set_sizes[is_id_cell].mean())
        coverage_all = float(covered.mean())
        avg_size_all = float(set_sizes.mean())

        print(
            f"  alpha={alpha:<5} target_coverage={1 - alpha:.2f}  "
            f"[ID-only] coverage={coverage_id:.4f} size={avg_size_id:.3f}  "
            f"[full query] coverage={coverage_all:.4f} size={avg_size_all:.3f}"
        )

        fold_result["per_alpha"][str(alpha)] = {
            "target_coverage": 1 - alpha,
            "empirical_coverage_id_only": coverage_id,
            "avg_set_size_id_only": avg_size_id,
            "empirical_coverage_full_query": coverage_all,
            "avg_set_size_full_query": avg_size_all,
        }

    acc = float(np.mean(obs["predicted_labels"].astype(str).values == ground_truth))
    fold_result["top1_accuracy"] = acc
    print(f"  top-1 accuracy: {acc:.4f}")

    return fold_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    donor_data = load_all_donors()

    all_results = []
    for held_out in DONORS:
        result = run_fold(held_out, donor_data, quick=args.quick)
        all_results.append(result)

    suffix = "_quick" if args.quick else ""
    out_path = RESULTS_DIR / f"phase2_results{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved results -> {out_path}")

    rows = []
    for r in all_results:
        for alpha, m in r["per_alpha"].items():
            rows.append(
                {
                    "held_out_donor": r["held_out_donor"],
                    "alpha": alpha,
                    "target_coverage": m["target_coverage"],
                    "coverage_id_only": m["empirical_coverage_id_only"],
                    "size_id_only": m["avg_set_size_id_only"],
                    "coverage_full_query": m["empirical_coverage_full_query"],
                    "size_full_query": m["avg_set_size_full_query"],
                    "top1_accuracy": r["top1_accuracy"],
                }
            )
    summary_df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("SUMMARY (per fold, per alpha)")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    summary_csv = RESULTS_DIR / f"phase2_summary{suffix}.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved summary table -> {summary_csv}")

    print("\n" + "=" * 70)
    print("AGGREGATE (mean across folds, per alpha)")
    print("=" * 70)
    agg = summary_df.groupby("alpha")[
        ["target_coverage", "coverage_id_only", "size_id_only", "coverage_full_query", "size_full_query"]
    ].mean()
    print(agg.to_string())


if __name__ == "__main__":
    main()
