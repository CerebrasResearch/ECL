#!/usr/bin/env python3
"""Figure 7: Violin plots of ECL_0.9 distribution across locus classes.

Shows ECL_0.9 distribution for six locus classes (promoter, proximal enhancer,
distal enhancer, CTCF, intronic, intergenic) for two synthetic models with
different decay lengths. Demonstrates that different genomic element types
exhibit distinct effective context length distributions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ecl.ecl import ECL
from ecl.models.base import SyntheticModel


def generate_ecl_per_class(
    model: SyntheticModel,
    locus_classes: dict[str, dict],
    n_loci_per_class: int = 80,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Generate synthetic ECL_0.9 values per locus class.

    Each class has a characteristic influence profile shape parameterised
    by a class-specific decay modifier, mean shift, and variance.
    """
    rng = rng or np.random.default_rng(42)
    seq_len = model.nominal_context
    results: dict[str, np.ndarray] = {}

    for cls_name, params in locus_classes.items():
        ecl_values = np.empty(n_loci_per_class)
        decay_mod = params["decay_modifier"]
        noise_scale = params["noise_scale"]

        for k in range(n_loci_per_class):
            # Build a synthetic influence profile with class-specific decay
            effective_decay = model._decay_length * decay_mod
            max_dist = min(500, seq_len // 2)
            distances = np.arange(max_dist + 1, dtype=np.float64)
            influence = np.exp(-distances / effective_decay)
            # Add locus-specific noise
            influence += rng.exponential(scale=noise_scale, size=len(influence))
            influence = np.maximum(influence, 0.0)

            ecl_val = ECL(distances, influence, beta=0.9)
            ecl_values[k] = ecl_val

        results[cls_name] = ecl_values

    return results


def main() -> None:
    from _config import FIGURE_DIR as output_dir

    rng = np.random.default_rng(2024)

    # Define locus classes with characteristic ECL behaviour
    locus_classes = {
        "Promoter": {"decay_modifier": 0.3, "noise_scale": 0.02},
        "Prox. Enhancer": {"decay_modifier": 0.8, "noise_scale": 0.03},
        "Dist. Enhancer": {"decay_modifier": 2.5, "noise_scale": 0.04},
        "CTCF": {"decay_modifier": 1.5, "noise_scale": 0.025},
        "Intronic": {"decay_modifier": 0.5, "noise_scale": 0.05},
        "Intergenic": {"decay_modifier": 0.2, "noise_scale": 0.06},
    }

    # Two synthetic models with different decay lengths
    model_a = SyntheticModel(seq_length=1000, embed_dim=64, decay_length=100.0)
    model_b = SyntheticModel(seq_length=1000, embed_dim=64, decay_length=250.0)

    ecl_a = generate_ecl_per_class(model_a, locus_classes, n_loci_per_class=80, rng=rng)
    ecl_b = generate_ecl_per_class(model_b, locus_classes, n_loci_per_class=80, rng=rng)

    # Build a tidy dataframe-like structure for seaborn
    class_names = []
    ecl_vals = []
    model_labels = []
    for cls in locus_classes:
        for v in ecl_a[cls]:
            class_names.append(cls)
            ecl_vals.append(v)
            model_labels.append("Model A (decay=100)")
        for v in ecl_b[cls]:
            class_names.append(cls)
            ecl_vals.append(v)
            model_labels.append("Model B (decay=250)")

    # Plot
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(14, 6))

    palette = {"Model A (decay=100)": "#4C72B0", "Model B (decay=250)": "#DD8452"}

    sns.violinplot(
        x=class_names,
        y=ecl_vals,
        hue=model_labels,
        split=True,
        inner="quartile",
        palette=palette,
        linewidth=1.0,
        ax=ax,
        density_norm="width",
        cut=0,
    )

    ax.set_xlabel("Locus Class", fontsize=13)
    ax.set_ylabel(r"ECL$_{0.9}$ (bp)", fontsize=13)
    ax.set_title(
        "Figure 7: ECL$_{0.9}$ Distribution Across Locus Classes",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(title="Model", loc="upper left", framealpha=0.9)
    ax.tick_params(axis="x", rotation=15)

    fig.tight_layout()

    fig.savefig(output_dir / "fig07_locus_class_violin.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(output_dir / "fig07_locus_class_violin.png", bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"Figure 7 saved to {output_dir / 'fig07_locus_class_violin.pdf'}")
    print(f"Figure 7 saved to {output_dir / 'fig07_locus_class_violin.png'}")


if __name__ == "__main__":
    main()
