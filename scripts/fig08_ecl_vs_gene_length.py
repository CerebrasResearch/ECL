#!/usr/bin/env python3
"""Figure 8: Scatter plot of ECL_0.9 vs gene length, colored by expression level.

Shows the relationship between effective context length and gene body length,
with points colored by expression level (low / medium / high). Uses synthetic
data generated from SyntheticModel with varying decay lengths to simulate
the observation that longer genes tend to have longer effective contexts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ecl.ecl import ECL


def generate_synthetic_gene_data(
    n_genes: int = 400,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Generate synthetic gene length, ECL, and expression data.

    Longer genes are given larger effective decay lengths (positive
    correlation between gene length and ECL), with expression level
    adding a secondary modulation.
    """
    rng = rng or np.random.default_rng(2024)

    # Gene lengths span 500 bp to 2.5 Mb (log-uniform)
    log_lengths = rng.uniform(np.log10(500), np.log10(2_500_000), size=n_genes)
    gene_lengths = 10**log_lengths

    # Expression categories
    expr_idx = rng.choice(3, size=n_genes, p=[0.35, 0.40, 0.25])
    expr_labels = np.array(["Low", "Medium", "High"])[expr_idx]

    # Expression modulates the decay: higher expression -> slightly shorter ECL
    expr_multiplier = np.array([1.2, 1.0, 0.7])[expr_idx]

    ecl_values = np.empty(n_genes)

    for i in range(n_genes):
        # Decay length scales with sqrt of gene length, modulated by expression
        base_decay = 5.0 * np.sqrt(gene_lengths[i]) * expr_multiplier[i] / 100.0
        base_decay = np.clip(base_decay, 5.0, 500.0)

        max_dist = 500
        distances = np.arange(max_dist + 1, dtype=np.float64)
        influence = np.exp(-distances / base_decay)
        influence += rng.exponential(scale=0.01, size=len(influence))
        influence = np.maximum(influence, 0.0)

        ecl_values[i] = ECL(distances, influence, beta=0.9)

    return {
        "gene_length": gene_lengths,
        "ecl": ecl_values,
        "expression": expr_labels,
    }


def main() -> None:
    from _config import FIGURE_DIR as output_dir

    rng = np.random.default_rng(2024)
    data = generate_synthetic_gene_data(n_genes=500, rng=rng)

    # Colour palette for expression levels
    palette = {"Low": "#636EFA", "Medium": "#EF553B", "High": "#00CC96"}
    markers = {"Low": "o", "Medium": "s", "High": "D"}

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(10, 7))

    for level in ["Low", "Medium", "High"]:
        mask = data["expression"] == level
        ax.scatter(
            data["gene_length"][mask],
            data["ecl"][mask],
            c=palette[level],
            marker=markers[level],
            s=25,
            alpha=0.6,
            edgecolors="none",
            label=f"{level} expression",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Gene Length (bp)", fontsize=13)
    ax.set_ylabel(r"ECL$_{0.9}$ (bp)", fontsize=13)
    ax.set_title(
        "Figure 8: ECL$_{0.9}$ vs Gene Length by Expression Level",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(title="Expression", fontsize=11, title_fontsize=12, loc="upper left")

    # Add trend line (all data pooled)
    from numpy.polynomial import polynomial as P

    log_gl = np.log10(data["gene_length"])
    coeffs = P.polyfit(log_gl, data["ecl"], deg=1)
    x_fit = np.linspace(log_gl.min(), log_gl.max(), 200)
    y_fit = P.polyval(x_fit, coeffs)
    ax.plot(10**x_fit, y_fit, "k--", linewidth=1.5, alpha=0.5, label="Linear trend (log scale)")
    ax.legend(title="Expression", fontsize=10, title_fontsize=11, loc="upper left")

    fig.tight_layout()

    fig.savefig(output_dir / "fig08_ecl_vs_gene_length.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(output_dir / "fig08_ecl_vs_gene_length.png", bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"Figure 8 saved to {output_dir / 'fig08_ecl_vs_gene_length.pdf'}")
    print(f"Figure 8 saved to {output_dir / 'fig08_ecl_vs_gene_length.png'}")


if __name__ == "__main__":
    main()
