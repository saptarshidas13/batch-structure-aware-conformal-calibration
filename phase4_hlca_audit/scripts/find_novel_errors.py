"""
Phase 4, step (d): push past "confidence correlates with cells HLCA already
flagged as wrong" (Result 1) toward a genuinely surprising finding -- a
SPECIFIC population that HLCA's own reannotation_type column calls
"Correctly annotated" (i.e., the field's own expert curation process did NOT
flag it), but where an independently-trained classifier, with zero access to
HLCA's curation history, confidently and reproducibly disagrees.

This reuses hlca_examples_full.csv (already computed by extract_examples.py
-- 27,974 cells across 14 leave-one-dataset-out folds), so no re-training is
needed. The question is purely a mining question over existing predictions:

  Among cells labeled "Correctly annotated" by HLCA, which specific
  (fine-grained cell_type_full -> consensus_label -> predicted_label)
  triples show our classifier CONFIDENTLY (set_size_naive == 1, high
  pred_confidence) and REPRODUCIBLY (across >=3 independent held-out
  datasets, so it's not one study's batch idiosyncrasy) disagreeing with
  the consensus label HLCA's curators did not flag?

Candidates surviving this filter are genuine candidate errors HLCA's own
reannotation process may have missed -- not a re-statement of Result 1.
"""
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
IN_PATH = RESULTS_DIR / "hlca_examples_full.csv"

MIN_CONFIDENCE = 0.90
MIN_DATASETS = 3
MIN_CELLS = 15


def main():
    df = pd.read_csv(IN_PATH)
    print(f"Loaded {len(df)} cells")
    print(df["reannotation_type"].value_counts())

    correct = df[df["reannotation_type"] == "Correctly annotated"].copy()
    print(f"\n'Correctly annotated' cells (per HLCA's own curators): {len(correct)}")

    disagree = correct[correct["predicted_label"] != correct["consensus_label"]].copy()
    print(f"Of those, classifier disagrees with consensus: {len(disagree)} "
          f"({100 * len(disagree) / len(correct):.1f}%)")

    confident = disagree[
        (disagree["set_size_naive"] == 1) & (disagree["pred_confidence"] >= MIN_CONFIDENCE)
    ].copy()
    print(f"Of those, CONFIDENT disagreement (set_size=1, conf>={MIN_CONFIDENCE}): {len(confident)}")

    grouped = (
        confident.groupby(["cell_type_full", "consensus_label", "predicted_label"])
        .agg(
            n_cells=("dataset", "size"),
            n_datasets=("dataset", "nunique"),
            mean_confidence=("pred_confidence", "mean"),
            datasets=("dataset", lambda s: sorted(s.unique())),
        )
        .reset_index()
        .sort_values(["n_datasets", "n_cells"], ascending=False)
    )

    print("\n" + "=" * 100)
    print("FULL RANKED TABLE (no threshold filter, top 25 by n_datasets then n_cells)")
    print("=" * 100)
    with pd.option_context("display.max_colwidth", 40, "display.width", 160):
        print(grouped.head(25).to_string(index=False))

    reproducible = grouped[(grouped["n_datasets"] >= MIN_DATASETS) & (grouped["n_cells"] >= MIN_CELLS)]

    print("\n" + "=" * 100)
    print(f"CANDIDATE NOVEL ERRORS: confident, reproducible (>={MIN_DATASETS} datasets, "
          f">={MIN_CELLS} cells) disagreements on cells HLCA calls 'Correctly annotated'")
    print("=" * 100)
    with pd.option_context("display.max_colwidth", 40, "display.width", 160):
        print(reproducible.to_string(index=False))

    out_path = RESULTS_DIR / "hlca_candidate_novel_errors.csv"
    reproducible.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")

    # also dump full per-cell detail for the top candidate for manual/marker-gene follow-up
    if len(reproducible) > 0:
        top = reproducible.iloc[0]
        top_cells = confident[
            (confident["cell_type_full"] == top["cell_type_full"])
            & (confident["consensus_label"] == top["consensus_label"])
            & (confident["predicted_label"] == top["predicted_label"])
        ]
        top_out = RESULTS_DIR / "hlca_top_candidate_cells.csv"
        top_cells.to_csv(top_out, index=False)
        print(f"\nTop candidate: {top['cell_type_full']} labeled '{top['consensus_label']}' "
              f"but classifier says '{top['predicted_label']}' "
              f"({top['n_cells']} cells across {top['n_datasets']} datasets, "
              f"mean confidence {top['mean_confidence']:.3f})")
        print(f"Per-cell detail saved -> {top_out}")


if __name__ == "__main__":
    main()
