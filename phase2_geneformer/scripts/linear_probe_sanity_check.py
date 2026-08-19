"""
Sanity check for Phase 2's negative result: is the small from-scratch
feed-forward net ("torch_net") underperforming because it's an unstable/
undertrained probe, or because the Geneformer embeddings themselves aren't
very discriminative for this task?

Fits a plain multinomial logistic regression (standard "linear probe" for
evaluating frozen embeddings) on the same leave-one-donor-out folds and
Table S8 OOD exclusions used in Phase 2, and reports top-1 accuracy on
ID-only query cells -- directly comparable to torch_net's 38-58% and
Phase 1 CellTypist's ~82-94%. No OOD detector / conformal calibration here;
this isolates the classification-quality question only.
"""
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "embedding_adata"
DONORS = ["human1", "human2", "human3", "human4"]

OOD_SIMULATED_EXCLUSIONS = {
    "human1": ["quiescent_stellate", "macrophage", "mast", "schwann", "epsilon", "t_cell"],
    "human2": ["macrophage", "mast", "schwann", "epsilon", "t_cell"],
    "human3": ["quiescent_stellate", "gamma", "macrophage", "mast", "schwann", "epsilon", "t_cell"],
    "human4": ["quiescent_stellate", "gamma", "macrophage", "mast", "schwann", "epsilon", "t_cell"],
}


def load_all_donors():
    return {d: ad.read_h5ad(DATA_DIR / f"baron_{d}_geneformer.h5ad") for d in DONORS}


def main():
    donor_data = load_all_donors()
    accs = []

    for held_out in DONORS:
        ref_parts = [donor_data[d] for d in DONORS if d != held_out]
        adata_ref = ad.concat(ref_parts, join="inner")
        excluded = OOD_SIMULATED_EXCLUSIONS[held_out]
        adata_ref = adata_ref[~adata_ref.obs["cell_type"].isin(excluded)].copy()

        adata_query = donor_data[held_out].copy()
        gt = adata_query.obs["cell_type"].astype(str).values
        is_id = ~np.isin(gt, excluded)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(adata_ref.X)
        y_train = adata_ref.obs["cell_type"].astype(str).values
        X_query = scaler.transform(adata_query.X)

        clf = LogisticRegression(max_iter=2000, C=1.0, multi_class="multinomial")
        clf.fit(X_train, y_train)
        preds = clf.predict(X_query)

        acc_id = float(np.mean(preds[is_id] == gt[is_id]))
        acc_all = float(np.mean(preds == gt))
        accs.append(acc_id)
        print(
            f"held-out={held_out:8s}  n_ref={adata_ref.n_obs:5d}  n_query={adata_query.n_obs:4d}  "
            f"ID-only acc={acc_id:.4f}  full-query acc={acc_all:.4f}"
        )

    print(f"\nMean ID-only accuracy across folds: {np.mean(accs):.4f}")
    print("For comparison: Phase 2 torch_net accuracy was 0.378-0.585 (mean ~0.51)")
    print("                 Phase 1 CellTypist accuracy was ~0.82-0.94")


if __name__ == "__main__":
    main()
