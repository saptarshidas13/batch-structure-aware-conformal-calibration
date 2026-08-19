"""
Peer-review response: bootstrap confidence intervals for the HLCA audit's
aggregate accuracy-by-reannotation_type finding (main text Table 2 / Result
1). Resamples the per-cell top1_correct indicator within each
reannotation_type group with replacement, B times, to get a 95% percentile
CI on top-1 accuracy per group. Also reports, across the same bootstrap
resamples, how often the full ordering (Correct > Underannotated >
Misannotated) holds, as a robustness check on the qualitative claim, not
just the point estimate.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
IN_PATH = RESULTS_DIR / "hlca_audit_per_cell.csv"
N_BOOTSTRAP = 5000
ORDER = ["Correctly annotated", "Underannotated", "Misannotated"]
SEED = 42


def bootstrap_ci(correct: np.ndarray, n_boot: int, rng: np.random.Generator):
    n = len(correct)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = correct[idx].mean()
    lo, med, hi = np.percentile(boot_means, [2.5, 50, 97.5])
    return boot_means, float(lo), float(med), float(hi)


def main():
    df = pd.read_csv(IN_PATH)
    df["top1_correct"] = df["top1_correct"].astype(bool)
    rng = np.random.default_rng(SEED)

    boot_means_by_group = {}
    rows = []
    for group in ORDER:
        correct = df.loc[df["reannotation_type"] == group, "top1_correct"].values
        boot_means, lo, med, hi = bootstrap_ci(correct, N_BOOTSTRAP, rng)
        boot_means_by_group[group] = boot_means
        point = float(correct.mean())
        print(f"{group:<22} n={len(correct):>6}  accuracy={point:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
        rows.append({
            "reannotation_type": group, "n_cells": len(correct),
            "accuracy": point, "ci_lo": lo, "ci_median": med, "ci_hi": hi,
        })

    # robustness of the ordering across paired bootstrap draws
    ordering_holds = (
        (boot_means_by_group["Correctly annotated"] > boot_means_by_group["Underannotated"])
        & (boot_means_by_group["Underannotated"] > boot_means_by_group["Misannotated"])
    )
    frac_ordering_holds = float(ordering_holds.mean())
    print(f"\nFraction of {N_BOOTSTRAP} bootstrap resamples where "
          f"Correct > Underannotated > Misannotated holds: {frac_ordering_holds:.4f}")

    out_df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "hlca_accuracy_ci.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")

    with open(RESULTS_DIR / "hlca_accuracy_ci_ordering.txt", "w") as f:
        f.write(f"Fraction of {N_BOOTSTRAP} bootstrap resamples where the full ordering "
                f"(Correct > Underannotated > Misannotated) holds: {frac_ordering_holds:.4f}\n")


if __name__ == "__main__":
    main()
