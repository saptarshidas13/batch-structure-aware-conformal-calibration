"""Generate the workflow/overview schematic (Figure 1), v2: professional
light palette pass. Content is unchanged from the v1 figure; only the visual
design (colors, borders, arrow styling) is revised.

Per Nature Portfolio artwork guidelines, line art / schematics should be
supplied as vector format (EPS/PDF/AI) rather than raster, unlike
photographic images which need >=300 dpi raster. This figure is therefore
saved as a vector PDF (submission-quality source) in addition to a
high-resolution PNG/TIFF (for embedding in the manuscript Word file, which
cannot natively embed a PDF as an inline picture).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
})

OUT = Path(__file__).resolve().parent / "figures_out"
OUT.mkdir(parents=True, exist_ok=True)

# --- palette (same categorical family as the data figures) -------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
ARROW_COLOR = "#8a8985"

STAGE_COLOR = "#cde2fb"      # light blue tint (sequential step 100)
STAGE_EDGE = "#2a78d6"       # blue
BRANCH_A_COLOR = "#fbe1d2"   # light orange tint
BRANCH_A_EDGE = "#eb6834"    # orange
BRANCH_B_COLOR = "#cdefe1"   # light aqua tint
BRANCH_B_EDGE = "#1baf7a"    # aqua
SYNTH_COLOR = "#e5e0f6"      # light violet tint
SYNTH_EDGE = "#4a3aa7"       # violet
PIVOT_COLOR = "#ffffff"
PIVOT_EDGE = "#52514e"


def box(ax, xy, w, h, text, facecolor=STAGE_COLOR, edgecolor=STAGE_EDGE, fontsize=8.3, weight="normal"):
    x, y = xy
    p = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.09",
        linewidth=1.4, edgecolor=edgecolor, facecolor=facecolor,
    )
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, weight=weight,
             color=INK_PRIMARY, linespacing=1.4)
    return (x, y, w, h)


def arrow(ax, start, end, color=ARROW_COLOR, lw=1.3):
    a = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=11,
        linewidth=lw, color=color, shrinkA=2, shrinkB=2,
    )
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13.6)
    ax.axis("off")

    # Stage 1: data
    b1 = box(
        ax, (5, 12.7), 9.4, 1.5,
        "5 datasets \u00b7 3 organ systems\n"
        "Pancreas (Baron et al.) \u00b7 NSCLC (Zilionis et al.) \u00b7 Cross-study lung (healthy vs. tumor)\n"
        "Blood (Tabula Sapiens) \u00b7 Lung atlas (HLCA core)",
        weight="bold", fontsize=8.0,
    )

    # Stage 2: features
    b2 = box(ax, (5, 11.0), 6.6, 0.95, "Feature extraction\nGeneformer embeddings (Tier 1) or PCA (all other tiers)")
    arrow(ax, (5, b1[1] - b1[3] / 2), (5, b2[1] + b2[3] / 2))

    # Stage 3: classifier
    b3 = box(ax, (5, 9.5), 6.6, 0.85, "Classifier\nCellTypist or multinomial logistic regression")
    arrow(ax, (5, b2[1] - b2[3] / 2), (5, b3[1] + b3[3] / 2))

    # Stage 4: conformal weighting schemes
    b4 = box(
        ax, (5, 7.85), 8.0, 1.15,
        "Weighted conformal calibration (APS non-conformity score)\n"
        "naive \u00b7 domain-classifier \u00b7 batch-centroid \u00b7 batch-mixture propensity",
    )
    arrow(ax, (5, b3[1] - b3[3] / 2), (5, b4[1] + b4[3] / 2))

    # Stage 5: severity tiers (pivot box, neutral)
    b5 = box(
        ax, (5, 6.3), 8.6, 0.95,
        "Evaluated across 3 tiers of batch-effect severity\n"
        "mild (donor) \u2192 moderate (patient) \u2192 severe (cross-study)",
        facecolor=PIVOT_COLOR, edgecolor=PIVOT_EDGE, weight="bold",
    )
    arrow(ax, (5, b4[1] - b4[3] / 2), (5, b5[1] + b5[3] / 2))

    # Branch point: a short T-shaped connector (drop, then horizontal split)
    # avoids the overlapping-arrowhead artifact of two arrows sharing one
    # start point directly under the box edge.
    branch_y = 5.15
    box_bottom = b5[1] - b5[3] / 2
    split_y = box_bottom - 0.25
    ax.plot([5, 5], [box_bottom, split_y], color=ARROW_COLOR, lw=1.3)
    ax.plot([2.7, 7.3], [split_y, split_y], color=ARROW_COLOR, lw=1.3)
    arrow(ax, (2.7, split_y), (2.7, branch_y + 0.55))
    arrow(ax, (7.3, split_y), (7.3, branch_y + 0.55))

    # Branch A: severe-shift diagnosis (orange family)
    a1 = box(
        ax, (2.7, 4.6), 4.6, 1.05,
        "Severe cross-study shift breaks naive\nand reweighted coverage alike\n(Fig. 2, right panel)",
        facecolor=BRANCH_A_COLOR, edgecolor=BRANCH_A_EDGE,
    )
    arrow(ax, (2.7, a1[1] - a1[3] / 2), (2.7, 3.4), color=BRANCH_A_EDGE)
    a2 = box(
        ax, (2.7, 2.85), 4.6, 1.05,
        "Diagnosis: violated positivity assumption\n\u2192 upstream Harmony integration\npartially rescues coverage (Fig. 3)",
        facecolor=BRANCH_A_COLOR, edgecolor=BRANCH_A_EDGE,
    )

    # Branch B: HLCA audit (aqua family)
    b_1 = box(
        ax, (7.3, 4.6), 4.6, 1.05,
        "HLCA audit: independent classifier\nconfidence tracks curator-flagged\nerrors (Fig. 4a)",
        facecolor=BRANCH_B_COLOR, edgecolor=BRANCH_B_EDGE,
    )
    arrow(ax, (7.3, b_1[1] - b_1[3] / 2), (7.3, 3.4), color=BRANCH_B_EDGE)
    b_2 = box(
        ax, (7.3, 2.85), 4.6, 1.05,
        "Mining confident disagreement on\n\"correct\" cells finds a specific,\nconfound-checked discrepancy (Fig. 4b,c)",
        facecolor=BRANCH_B_COLOR, edgecolor=BRANCH_B_EDGE,
    )

    # Converge to shared bottom conclusion (violet family)
    conv_y = 1.15
    arrow(ax, (2.7, a2[1] - a2[3] / 2), (4.6, conv_y + 0.55), color=ARROW_COLOR)
    arrow(ax, (7.3, b_2[1] - b_2[3] / 2), (5.4, conv_y + 0.55), color=ARROW_COLOR)
    box(
        ax, (5, conv_y), 8.6, 1.0,
        "Boundary conditions for trustworthy batch-aware calibration,\n"
        "and a validated atlas-auditing application",
        facecolor=SYNTH_COLOR, edgecolor=SYNTH_EDGE, weight="bold", fontsize=8.8,
    )

    fig.text(0.01, 0.985, "Figure 1", fontsize=10, fontweight="bold", va="top", color=INK_PRIMARY)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    fig.savefig(OUT / "Figure1_workflow_overview.pdf", bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(OUT / "Figure1_workflow_overview.tif", dpi=300, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"}, facecolor=SURFACE)
    fig.savefig(OUT / "Figure1_workflow_overview.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("Saved Figure1_workflow_overview.{pdf,tif,png}")


if __name__ == "__main__":
    main()
