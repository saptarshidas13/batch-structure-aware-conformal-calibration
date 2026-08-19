"""
Phase 1 baseline reproduction: leave-one-donor-out conformal cell-type
annotation on the Baron et al. 2016 human pancreas dataset (GSE84133),
mirroring the design in "Conformal inference for reliable single cell
RNA-seq annotation" (Bioinformatics, 2025) -- CellTypist underlying model,
APS non-conformity function, alpha in {0.01, 0.05, 0.1}.

For each donor D in {human1, human2, human3, human4}:
    reference = the other 3 donors (concatenated)
    query     = donor D
    -> fit CellTypist + autoencoder OOD detector on reference
    -> conformal-calibrate, annotate the query
    -> compute empirical coverage and average prediction-set size per alpha

Usage:
    python run_phase1_baseline.py [--quick]

--quick reduces OOD-detector training epochs for a fast smoke test before
committing to the full (paper-matching) run.
"""
import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

REPO_DIR = Path(__file__).resolve().parents[1] / "repo"
sys.path.insert(0, str(REPO_DIR))

from conformalSC_annotator import ConformalSCAnnotator  # noqa: E402
from torchcp.classification.score import APS  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DONORS = ["human1", "human2", "human3", "human4"]
ALPHAS = [0.01, 0.05, 0.1]

# Table S8 (supplementary materials): cell types deliberately excluded from the
# reference set to simulate OOD conditions, keyed by which donor is held out
# as the query. Matches paper's Pancreas Experiments 1-4 (donor query sizes in
# Table S3 -- 1903/1705/3518/1279 -- align with human1/2/3/4 respectively).
OOD_SIMULATED_EXCLUSIONS = {
    "human1": ["quiescent_stellate", "macrophage", "mast", "schwann", "epsilon", "t_cell"],
    "human2": ["macrophage", "mast", "schwann", "epsilon", "t_cell"],
    "human3": ["quiescent_stellate", "gamma", "macrophage", "mast", "schwann", "epsilon", "t_cell"],
    "human4": ["quiescent_stellate", "gamma", "macrophage", "mast", "schwann", "epsilon", "t_cell"],
}


def load_all_donors() -> dict:
    data = {}
    for d in DONORS:
        path = DATA_DIR / f"baron_{d}.h5ad"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run prepare_baron_pancreas.py first."
            )
        data[d] = ad.read_h5ad(path)
    return data


def prediction_set_to_list(cell_set):
    """Prediction sets may come back as lists, sets, or numpy arrays; normalize."""
    if cell_set is None:
        return []
    if isinstance(cell_set, (list, tuple, set, np.ndarray)):
        return list(cell_set)
    return [cell_set]


def densify(adata: ad.AnnData) -> ad.AnnData:
    """The annotator's OOD-detector autoencoder calls torch.tensor() directly
    on adata.X without handling scipy sparse matrices. These donor datasets
    are small enough (thousands of cells) that dense storage is cheap."""
    import scipy.sparse as sp

    if sp.issparse(adata.X):
        adata.X = np.asarray(adata.X.todense(), dtype=np.float32)
    return adata


def run_fold(held_out_donor: str, donor_data: dict, quick: bool) -> dict:
    print(f"\n{'=' * 70}\nFOLD: held-out donor = {held_out_donor}\n{'=' * 70}")

    ref_parts = [donor_data[d] for d in DONORS if d != held_out_donor]
    adata_reference = ad.concat(ref_parts, join="inner", label="donor_src")
    adata_reference.var["features"] = adata_reference.var_names

    # Match the paper's Table S8 design: drop specific cell types from the
    # reference (but keep them in the query) to simulate OOD conditions.
    excluded_types = OOD_SIMULATED_EXCLUSIONS[held_out_donor]
    n_before = adata_reference.n_obs
    adata_reference = adata_reference[
        ~adata_reference.obs["cell_type"].isin(excluded_types)
    ].copy()
    print(
        f"  Excluded {excluded_types} from reference "
        f"({n_before} -> {adata_reference.n_obs} cells) to simulate OOD, per Table S8"
    )

    # Paper's S4.2 gene filter ("detected in <3 cells removed"), applied to
    # the assembled reference so all folds get a consistent feature space --
    # the query is then subset to match, rather than filtered independently.
    n_genes_before = adata_reference.n_vars
    sc.pp.filter_genes(adata_reference, min_cells=3)
    print(
        f"  Gene filter on reference (detected in >=3 cells): "
        f"{n_genes_before} -> {adata_reference.n_vars} genes"
    )
    adata_reference.var["features"] = adata_reference.var_names
    adata_reference = densify(adata_reference)

    adata_query = donor_data[held_out_donor].copy()
    adata_query = adata_query[:, adata_reference.var_names].copy()
    adata_query.var["features"] = adata_query.var_names
    adata_query = densify(adata_query)

    n_epochs_ood = 60 if quick else 850

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
        underlying_model="celltypist",
    )

    annotator.configure(
        reference_path=adata_reference,
        OOD_detector=OOD_detector_config,
        CP_predictor="standard",
        cell_names_column="cell_type",
        cell_types_excluded_treshold=45,  # matches paper's S4.2 threshold
        test=True,
        alpha=ALPHAS,
        non_conformity_function=APS(),
        random_state=42,
    )

    annotator.annotate(obsm_layer=None)

    obs = annotator.adata_query.obs
    ground_truth = obs["cell_type"].astype(str).values

    fold_result = {
        "held_out_donor": held_out_donor,
        "n_query_cells": int(len(ground_truth)),
        "test_results": annotator.test_results,
        "ood_performance": annotator.OOD_performance_scores,
        "alpha_OOD": annotator.alpha_OOD,
        "per_alpha": {},
    }

    # Cells whose true type was deliberately excluded from the reference
    # (Table S8) can never appear in a prediction set by construction -- their
    # label isn't in the reference's label space. Report both the raw
    # full-query metric and an ID-only metric restricted to cells whose type
    # WAS in the reference; the ID-only view is what's comparable to the
    # paper's "Query data" coverage columns (Tables S10-S16).
    is_id_cell = ~np.isin(ground_truth, excluded_types)
    print(f"  ID cells in query: {is_id_cell.sum()} / {len(ground_truth)}")

    for alpha in ALPHAS:
        col = f"prediction_sets_{alpha}"
        pred_sets = [prediction_set_to_list(s) for s in obs[col]]
        covered = np.array([gt in pset for gt, pset in zip(ground_truth, pred_sets)])
        set_sizes = np.array([len(pset) for pset in pred_sets])

        coverage_all = float(covered.mean())
        avg_size_all = float(set_sizes.mean())
        coverage_id = float(covered[is_id_cell].mean())
        avg_size_id = float(set_sizes[is_id_cell].mean())

        print(
            f"  alpha={alpha:<5} target_coverage={1 - alpha:.2f}  "
            f"[ID-only] coverage={coverage_id:.4f} size={avg_size_id:.3f}  "
            f"[full query, incl. simulated-OOD] coverage={coverage_all:.4f} size={avg_size_all:.3f}"
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
    parser.add_argument(
        "--quick", action="store_true", help="Fast smoke test (fewer OOD epochs)"
    )
    args = parser.parse_args()

    donor_data = load_all_donors()

    all_results = []
    for held_out in DONORS:
        result = run_fold(held_out, donor_data, quick=args.quick)
        all_results.append(result)

    out_path = RESULTS_DIR / ("phase1_results_quick.json" if args.quick else "phase1_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved results -> {out_path}")

    # Summary table across folds
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

    summary_csv = RESULTS_DIR / ("phase1_summary_quick.csv" if args.quick else "phase1_summary.csv")
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
