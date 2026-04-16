#!/usr/bin/env python3
"""Figure 9: Paired difference plot (Model A - Model B) of ECL_0.9 at matched loci.

Shows the distribution of ECL_0.9 differences between two models at 500 matched
loci, with kernel density, mean +/- 95% CI, and stratification by locus class.
Uses the permutation test from ecl.estimation to assess statistical significance.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ecl.ecl import ECL
from ecl.estimation import permutation_test
from ecl.models.base import SyntheticModel


def generate_paired_ecl(
    n_loci: int = 500,
    rng: np.random.Generator | None = None,
) -> dict:
    """Generate paired ECL_0.9 values for two models at matched loci.

    Model A has decay_length=100, Model B has decay_length=180, producing
    a systematic positive shift in Model B's ECL values.
    """
    rng = rng or np.random.default_rng(2024)

    model_a = SyntheticModel(seq_length=1000, embed_dim=64, decay_length=100.0)
    model_b = SyntheticModel(seq_length=1000, embed_dim=64, decay_length=180.0)

    locus_classes = [
        "Promoter",
        "Prox. Enhancer",
        "Dist. Enhancer",
        "CTCF",
        "Intronic",
        "Intergenic",
    ]
    class_decay_mods = {
        "Promoter": 0.3,
        "Prox. Enhancer": 0.8,
        "Dist. Enhancer": 2.5,
        "CTCF": 1.5,
        "Intronic": 0.5,
        "Intergenic": 0.2,
    }

    # Assign locus classes
    class_assignments = rng.choice(locus_classes, size=n_loci)

    ecl_a = np.empty(n_loci)
    ecl_b = np.empty(n_loci)
    max_dist = 500
    distances = np.arange(max_dist + 1, dtype=np.float64)

    for i in range(n_loci):
        cls = class_assignments[i]
        dm = class_decay_mods[cls]

        # Model A influence profile
        decay_a = model_a._decay_length * dm
        infl_a = np.exp(-distances / decay_a)
        infl_a += rng.exponential(scale=0.02, size=len(infl_a))
        ecl_a[i] = ECL(distances, np.maximum(infl_a, 0.0), beta=0.9)

        # Model B influence profile
        decay_b = model_b._decay_length * dm
        infl_b = np.exp(-distances / decay_b)
        infl_b += rng.exponential(scale=0.02, size=len(infl_b))
        ecl_b[i] = ECL(distances, np.maximum(infl_b, 0.0), beta=0.9)

    return {
        "ecl_a": ecl_a,
        "ecl_b": ecl_b,
        "classes": class_assignments,
    }


def main() -> None:
    from _config import FIGURE_DIR as output_dir

    rng = np.random.default_rng(2024)
    data = generate_paired_ecl(n_loci=500, rng=rng)

    diff = data["ecl_a"] - data["ecl_b"]

    # Run permutation test
    mean_diff, p_value, ci_hw = permutation_test(
        data["ecl_a"], data["ecl_b"], n_permutations=5000, rng=rng
    )

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [2, 1.2]})

    # --- Left panel: overall density of differences ---
    ax = axes[0]
    sns.histplot(diff, kde=True, color="#4C72B0", alpha=0.5, bins=40, ax=ax, stat="density")
    ax.axvline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axvline(
        mean_diff, color="#C44E52", linestyle="-", linewidth=2.0, label=f"Mean = {mean_diff:.1f}"
    )
    ax.axvspan(
        mean_diff - ci_hw,
        mean_diff + ci_hw,
        alpha=0.15,
        color="#C44E52",
        label=f"95% CI [{mean_diff - ci_hw:.1f}, {mean_diff + ci_hw:.1f}]",
    )
    ax.set_xlabel(r"$\Delta$ECL$_{0.9}$ (Model A $-$ Model B)", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title(
        f"Paired Difference (N=500, permutation p={p_value:.4f})",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper right")

    # --- Right panel: stratified by locus class ---
    ax2 = axes[1]
    classes_unique = [
        "Promoter",
        "Prox. Enhancer",
        "Dist. Enhancer",
        "CTCF",
        "Intronic",
        "Intergenic",
    ]
    class_colors = sns.color_palette("Set2", n_colors=len(classes_unique))

    class_means = []
    class_cis = []
    for cls in classes_unique:
        mask = data["classes"] == cls
        d = diff[mask]
        m = np.mean(d)
        se = np.std(d, ddof=1) / np.sqrt(len(d)) * 1.96
        class_means.append(m)
        class_cis.append(se)

    y_pos = np.arange(len(classes_unique))
    ax2.barh(
        y_pos,
        class_means,
        xerr=class_cis,
        color=class_colors,
        edgecolor="gray",
        linewidth=0.5,
        height=0.6,
        capsize=3,
    )
    ax2.axvline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(classes_unique, fontsize=10)
    ax2.set_xlabel(r"Mean $\Delta$ECL$_{0.9}$", fontsize=12)
    ax2.set_title("Stratified by Locus Class", fontsize=12, fontweight="bold")
    ax2.invert_yaxis()

    fig.suptitle(
        "Figure 9: Model Comparison -- Paired ECL$_{0.9}$ Differences",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    fig.savefig(output_dir / "fig09_model_comparison_paired.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(output_dir / "fig09_model_comparison_paired.png", bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"Figure 9 saved to {output_dir / 'fig09_model_comparison_paired.pdf'}")
    print(f"Figure 9 saved to {output_dir / 'fig09_model_comparison_paired.png'}")
    print(
        f"  Permutation test: mean diff = {mean_diff:.2f}, p = {p_value:.4f}, 95% CI half-width = {ci_hw:.2f}"
    )


if __name__ == "__main__":
    main()
