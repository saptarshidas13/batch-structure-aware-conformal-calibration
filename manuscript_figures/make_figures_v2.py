"""Generate manuscript figures (v2: professional light palette) from existing
Phase 3/4 result CSVs. Figures are saved as both TIFF (300 dpi, Nature-
portfolio's preferred raster format for submission) and PNG (for quick
preview) into manuscript/figures_v2/. This is a v2 pass on visual design only
-- the underlying data and figure content are unchanged from figures/.

Palette: validated categorical + sequential hues from a colorblind-safe
reference palette (blue/orange/aqua categorical triplet; blue sequential
ramp for ordinal encodings), on a near-white surface with muted chart chrome.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- palette -----------------------------------------------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
BLUE_LIGHT = "#86b6ef"   # sequential step 250
BLUE_MID = "#2a78d6"     # sequential step 450
BLUE_DARK = "#184f95"    # sequential step 600

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.titlecolor": INK_PRIMARY,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.labelcolor": INK_SECONDARY,
    "axes.edgecolor": BASELINE,
    "axes.linewidth": 0.8,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "text.color": INK_PRIMARY,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

BASE = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures_out"
OUT.mkdir(parents=True, exist_ok=True)

METHOD_COLORS = {
    "naive": BLUE,
    "structure_aware_propensity": AQUA,
    "domain_clf": ORANGE,
}
METHOD_LABELS = {
    "naive": "Naive",
    "structure_aware_propensity": "Batch-mixture propensity",
    "domain_clf": "Domain classifier (fixed)",
}


def clean_axes(ax, y_grid=True, x_grid=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    if y_grid:
        ax.grid(axis="y", color=GRIDLINE, lw=0.8, zorder=0)
    if x_grid:
        ax.grid(axis="x", color=GRIDLINE, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.tif", dpi=300, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {name}.tif / .png")


# ---------------------------------------------------------------------------
# Figure 2: coverage vs alpha across the three severity tiers
# ---------------------------------------------------------------------------
def fig2_coverage_by_tier():
    t1 = pd.read_csv(BASE / "phase3_batch_reweighting/results/tier1_baron_pancreas_summary.csv")
    t2 = pd.read_csv(BASE / "phase3_batch_reweighting/results/tier2_gse127465_summary.csv")
    t3 = pd.read_csv(BASE / "phase3_batch_reweighting/results/tier3_cross_study_summary.csv")
    t3 = t3.rename(columns={"direction": "held_out"})

    methods = ["naive", "structure_aware_propensity", "domain_clf"]
    tiers = [
        ("Tier 1: Baron pancreas\n(donor-level, mild)", t1),
        ("Tier 2: GSE127465\n(patient-level, moderate)", t2),
        ("Tier 3: cross-study lung\n(healthy vs. tumor, severe)", t3),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), sharey=True)
    for ax, (title, df) in zip(axes, tiers):
        agg = df.groupby(["alpha", "method"])["coverage"].mean().reset_index()
        alphas = sorted(agg["alpha"].unique())
        target = [1 - a for a in alphas]
        ax.plot(alphas, target, linestyle=(0, (4, 2)), color=INK_MUTED, lw=1.1,
                 label="Target" if ax is axes[0] else None, zorder=2)
        for m in methods:
            sub = agg[agg["method"] == m].sort_values("alpha")
            if sub.empty:
                continue
            ax.plot(sub["alpha"], sub["coverage"], marker="o", ms=5, lw=1.6,
                     markeredgecolor=SURFACE, markeredgewidth=0.6,
                     color=METHOD_COLORS[m], label=METHOD_LABELS[m] if ax is axes[0] else None,
                     zorder=3)
        ax.set_title(title, pad=8)
        ax.set_xlabel(r"$\alpha$")
        ax.invert_xaxis()
        ax.set_ylim(0.55, 1.02)
        clean_axes(ax)
    axes[2].annotate("domain-classifier sets are\nnear-vacuous (~3 of 3 classes)",
                      xy=(0.34, 0.72), xycoords="axes fraction", fontsize=6.5,
                      color=ORANGE, va="top", ha="left",
                      bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec=GRIDLINE, lw=0.8, alpha=0.95))
    axes[0].set_ylabel("Empirical coverage")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.13),
               frameon=False, labelcolor=INK_SECONDARY)
    fig.suptitle("Figure 2", x=0.02, ha="left", fontsize=10, fontweight="bold", y=1.05, color=INK_PRIMARY)
    save(fig, "Figure2_coverage_by_tier")


# ---------------------------------------------------------------------------
# Figure 3: Harmony rescue effect on Tier 3
# ---------------------------------------------------------------------------
def fig3_harmony_rescue():
    pre = pd.read_csv(BASE / "phase3_batch_reweighting/results/tier3_cross_study_summary.csv")
    post = pd.read_csv(BASE / "phase3_batch_reweighting/results/tier3_harmony_summary.csv")
    pre = pre[pre["method"] == "naive"]
    post = post[post["method"] == "naive"]

    directions = ["healthy->tumor", "tumor_G->healt"]
    dir_labels = ["Healthy \u2192 tumor", "Tumor \u2192 healthy"]
    alphas = sorted(pre["alpha"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
    width = 0.34
    x = np.arange(len(alphas))
    for ax, direction, dlabel in zip(axes, directions, dir_labels):
        pre_cov = [pre[(pre["direction"] == direction) & (pre["alpha"] == a)]["coverage"].values[0] for a in alphas]
        post_cov = [post[(post["direction"] == direction) & (post["alpha"] == a)]["coverage"].values[0] for a in alphas]
        target = [1 - a for a in alphas]
        ax.bar(x - width / 2, pre_cov, width, label="Pre-Harmony" if ax is axes[0] else None,
               color=BLUE, zorder=3)
        ax.bar(x + width / 2, post_cov, width, label="Post-Harmony" if ax is axes[0] else None,
               color=AQUA, zorder=3)
        for xi, t in zip(x, target):
            ax.plot([xi - width, xi + width], [t, t], linestyle=(0, (4, 2)), color=INK_MUTED, lw=1.1, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{a:g}" for a in alphas])
        ax.set_xlabel(r"$\alpha$")
        ax.set_title(dlabel, pad=8)
        clean_axes(ax)
    axes[0].set_ylabel("Naive coverage")
    axes[0].set_ylim(0.55, 1.05)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.15),
               frameon=False, labelcolor=INK_SECONDARY)
    fig.suptitle("Figure 3", x=0.02, ha="left", fontsize=10, fontweight="bold", y=1.05, color=INK_PRIMARY)
    save(fig, "Figure3_harmony_rescue")


# ---------------------------------------------------------------------------
# Figure 4: HLCA audit -- accuracy/confidence by reannotation_type, and the
# Krasnow_2020 novel-error finding
# ---------------------------------------------------------------------------
def fig4_hlca_audit():
    summ = pd.read_csv(BASE / "phase4_hlca_audit/results/hlca_audit_summary_by_reannotation_type.csv")
    order = ["Correctly annotated", "Underannotated", "Misannotated"]
    summ = summ.set_index("reannotation_type").loc[order].reset_index()

    marker = pd.read_csv(BASE / "phase4_hlca_audit/results/hlca_marker_verification.csv")
    basal = marker[marker["group"] == "Basal_correctly_annotated"].copy()
    basal["secretory_dominant"] = basal["secretory_score"] > basal["basal_score"]
    rate = basal.groupby("dataset")["secretory_dominant"].mean().reindex(
        ["Barbry_Leroy_2020", "Jain_Misharin_2021_10Xv2", "Krasnow_2020"]
    )
    dataset_labels = ["Barbry_\nLeroy_2020", "Jain_\nMisharin_2021", "Krasnow_\n2020"]

    stress = pd.read_csv(BASE / "phase4_hlca_audit/results/hlca_stress_confound_check.csv")

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8))

    ax = axes[0]
    ordinal_colors = [BLUE_LIGHT, BLUE_MID, BLUE_DARK]
    ax.bar(order, summ["top1_accuracy"] * 100, color=ordinal_colors, zorder=3)
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("a. Accuracy vs. HLCA\ncuration label", pad=8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["Correct", "Under-\nannotated", "Mis-\nannotated"], rotation=0)
    ax.set_ylim(80, 100)
    clean_axes(ax)

    ax = axes[1]
    bars = ax.bar(dataset_labels, rate.values * 100, color=ORANGE, width=0.6, zorder=3)
    ax.set_ylabel("Secretory-marker-dominant\n\"Basal, correct\" cells (%)")
    ax.set_title("b. Candidate discrepancy\nby dataset", pad=8)
    ax.set_xticks(range(len(dataset_labels)))
    ax.set_xticklabels(dataset_labels, fontsize=6.5)
    clean_axes(ax)
    for b, v in zip(bars, rate.values * 100):
        v_display = int(v * 10) / 10  # truncate to 1 decimal for consistency with main text (76.3%)
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v_display:.1f}%", ha="center", fontsize=7,
                 color=INK_SECONDARY)

    ax = axes[2]
    kr = stress[stress["dataset"] == "Krasnow_2020"].copy()
    kr["group"] = np.where(kr["secretory_dominant"], "Secretory-\ndominant", "Basal-\ndominant")
    groups = ["Basal-\ndominant", "Secretory-\ndominant"]
    data = [kr[kr["group"] == g]["stress_score"].values for g in groups]
    bp = ax.boxplot(data, tick_labels=groups, showfliers=False, patch_artist=True, widths=0.5,
                     medianprops=dict(color=INK_PRIMARY, lw=1.3),
                     boxprops=dict(lw=0.8, edgecolor=INK_SECONDARY),
                     whiskerprops=dict(color=INK_SECONDARY, lw=0.8),
                     capprops=dict(color=INK_SECONDARY, lw=0.8))
    for patch, c in zip(bp["boxes"], [BLUE, AQUA]):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax.set_ylabel("Dissociation-stress score")
    ax.set_title("c. Krasnow_2020: stress by\nmarker-dominance subgroup", pad=8)
    clean_axes(ax)

    fig.suptitle("Figure 4", x=0.02, ha="left", fontsize=10, fontweight="bold", y=1.06, color=INK_PRIMARY)
    fig.tight_layout()
    save(fig, "Figure4_hlca_audit")


# ---------------------------------------------------------------------------
# Figure 5: cross-tissue generalization -- domain-classifier instability
# vs. structure-aware safety, all five datasets
# ---------------------------------------------------------------------------
def fig5_cross_tissue():
    t1 = pd.read_csv(BASE / "phase3_batch_reweighting/results/tier1_baron_pancreas_summary.csv")
    t2 = pd.read_csv(BASE / "phase3_batch_reweighting/results/tier2_gse127465_summary.csv")
    blood = pd.read_csv(BASE / "phase5_breadth_tabula_blood/results/blood_breadth_summary.csv")

    def mean_size(df, method, alpha=0.10):
        sub = df[(df["method"] == method) & (df["alpha"] == alpha)]
        return sub["avg_size"].mean() if len(sub) else np.nan

    datasets = ["Baron\npancreas", "GSE127465\nNSCLC", "Tabula Sapiens\nblood"]
    naive_sizes = [mean_size(t1, "naive"), mean_size(t2, "naive"), mean_size(blood, "naive")]
    struct_sizes = [
        mean_size(t1, "structure_aware_propensity"),
        mean_size(t2, "structure_aware_propensity"),
        mean_size(blood, "structure_aware"),
    ]
    clf_sizes = [mean_size(t1, "domain_clf"), mean_size(t2, "domain_clf"), mean_size(blood, "domain_clf")]

    x = np.arange(len(datasets))
    width = 0.27
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    ax.bar(x - width, naive_sizes, width, label="Naive", color=BLUE, zorder=3)
    ax.bar(x, struct_sizes, width, label="Structure-aware", color=AQUA, zorder=3)
    ax.bar(x + width, clf_sizes, width, label="Domain classifier", color=ORANGE, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel(r"Mean prediction-set size ($\alpha$=0.10)")
    ax.set_title("Generic domain-classifier baseline inflates\nprediction sets across independent tissues", pad=8)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY)
    clean_axes(ax)
    fig.suptitle("Figure 5", x=0.02, ha="left", fontsize=10, fontweight="bold", y=1.03, color=INK_PRIMARY)
    save(fig, "Figure5_cross_tissue_generalization")


if __name__ == "__main__":
    fig2_coverage_by_tier()
    fig3_harmony_rescue()
    fig4_hlca_audit()
    fig5_cross_tissue()
