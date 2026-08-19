from pathlib import Path
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
df = pd.read_csv(RESULTS_DIR / "hlca_examples_full.csv")

sub = df[
    (df["cell_type_full"] == "respiratory basal cell")
    & (df["consensus_label"] == "Basal")
    & (df["reannotation_type"] == "Correctly annotated")
]
print(f"All 'respiratory basal cell / Basal / Correctly annotated' rows in the subsample: {len(sub)}")
print(sub.groupby("dataset")["predicted_label"].value_counts())

print("\nPer-dataset: how many predicted Secretory with set_size_naive==1 and conf>=0.9?")
conf = sub[(sub["predicted_label"] == "Secretory") & (sub["set_size_naive"] == 1) & (sub["pred_confidence"] >= 0.9)]
print(conf.groupby("dataset").size())
print(f"\nTotal basal-labeled cells per dataset in subsample:")
print(sub.groupby("dataset").size())
