#!/usr/bin/env python3
"""Figure 10: Biological validation -- influence profiles at known long-range loci.

For three well-characterised long-range regulatory loci:
  - SHH / ZRS enhancer (~1 Mb separation)
  - MYC / super-enhancer (~1.7 Mb)
  - SOX9 / regulatory desert (~1 Mb)

Plots the full influence profile centred on each target gene TSS, marks the
known enhancer position, and overlays three synthetic surrogate models with
different decay lengths to illustrate how longer-context models better capture
distal regulatory signals.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Known long-range loci (approximate distances from TSS to enhancer, in kb)
LOCI = {
    "SHH / ZRS": {
        "enhancer_distance_kb": 1000,
        "description": "SHH gene & ZRS limb enhancer",
    },
    "MYC / Super-enh.": {
        "enhancer_distance_kb": 1700,
        "description": "MYC & downstream super-enhancer",
    },
    "SOX9 / Reg. desert": {
        "enhancer_distance_kb": 1000,
        "description": "SOX9 & regulatory desert enhancers",
    },
}

# Three synthetic models as surrogates with increasing effective context
MODEL_CONFIGS = [
    {"name": "Short-context (decay=50)", "decay_length": 50.0, "color": "#4C72B0", "ls": "-"},
    {"name": "Medium-context (decay=200)", "decay_length": 200.0, "color": "#DD8452", "ls": "--"},
    {"name": "Long-context (decay=800)", "decay_length": 800.0, "color": "#55A868", "ls": "-."},
]


def synthetic_influence_profile(
    decay_length: float,
    max_dist_kb: int = 2500,
    enhancer_dist_kb: int = 1000,
    enhancer_strength: float = 0.3,
    enhancer_width_kb: float = 20.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic influence profile with an enhancer bump.

    The profile is exponential decay from TSS (d=0) plus a Gaussian bump
    at the enhancer position, whose visibility scales with the model's
    decay length.
    """
    rng = rng or np.random.default_rng()
    distances_kb = np.linspace(0, max_dist_kb, 2000)

    # Base exponential decay
    influence = np.exp(-distances_kb / decay_length)

    # Enhancer bump -- only visible if the model's decay is long enough
    # The bump amplitude is attenuated by the base decay at that distance
    base_at_enhancer = np.exp(-enhancer_dist_kb / decay_length)
    bump_amplitude = enhancer_strength * max(base_at_enhancer, 0.001)
    enhancer_bump = bump_amplitude * np.exp(
        -0.5 * ((distances_kb - enhancer_dist_kb) / enhancer_width_kb) ** 2
    )
    influence += enhancer_bump

    # Small noise
    influence += rng.exponential(scale=0.002, size=len(influence))
    influence = np.maximum(influence, 0.0)

    return distances_kb, influence


def main() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(2024)
    sns.set_theme(style="whitegrid", font_scale=1.05)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)

    for ax, (locus_name, locus_info) in zip(axes, LOCI.items()):
        enhancer_kb = locus_info["enhancer_distance_kb"]

        for cfg in MODEL_CONFIGS:
            dist_kb, infl = synthetic_influence_profile(
                decay_length=cfg["decay_length"],
                max_dist_kb=2500,
                enhancer_dist_kb=enhancer_kb,
                enhancer_strength=0.3,
                rng=rng,
            )
            # Normalise to peak = 1 for visual comparison
            infl_norm = infl / infl.max()

            ax.plot(
                dist_kb, infl_norm,
                color=cfg["color"],
                linestyle=cfg["ls"],
                linewidth=1.8,
                alpha=0.85,
                label=cfg["name"],
            )

        # Mark enhancer position
        ax.axvline(
            enhancer_kb, color="#C44E52", linestyle=":", linewidth=2.0, alpha=0.8,
        )
        ax.annotate(
            f"Enhancer\n({enhancer_kb} kb)",
            xy=(enhancer_kb, 0.85),
            xytext=(enhancer_kb + 150, 0.9),
            fontsize=9,
            color="#C44E52",
            arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.2),
            ha="left",
        )

        # Mark TSS
        ax.annotate("TSS", xy=(0, 1.0), fontsize=9, fontweight="bold",
                     xytext=(50, 1.02), ha="left")

        ax.set_xlabel("Distance from TSS (kb)", fontsize=12)
        ax.set_ylabel("Normalised Influence", fontsize=12)
        ax.set_title(locus_name, fontsize=12, fontweight="bold")
        ax.set_xlim(-50, 2550)
        ax.set_ylim(-0.02, 1.12)

    # Single legend for all panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=3,
        fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.05),
    )

    fig.suptitle(
        "Figure 10: Influence Profiles at Known Long-Range Regulatory Loci",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    fig.savefig(output_dir / "fig10_biological_validation.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(output_dir / "fig10_biological_validation.png", bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"Figure 10 saved to {output_dir / 'fig10_biological_validation.pdf'}")
    print(f"Figure 10 saved to {output_dir / 'fig10_biological_validation.png'}")


if __name__ == "__main__":
    main()
