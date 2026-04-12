#!/usr/bin/env python3
"""Figure 12: gNIAH sensitivity vs distance for three motifs across models.

Plots Genomic Needle-in-a-Haystack (gNIAH) sensitivity as a function of
insertion distance for three regulatory motifs (CTCF, GATA, SP1) across
multiple synthetic models with different decay lengths.

Uses the ecl.gniah module with SyntheticModel surrogates.
gNIAH(d, m) = E[d_Z(f(X_neutral), f(X_neutral^{+m@d}))]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ecl.gniah import MOTIFS, gniah_sensitivity
from ecl.models.base import SyntheticModel

# Use a smaller sequence length and fewer samples for tractability
SEQ_LENGTH = 2000
N_SAMPLES = 20

# Models to compare
MODEL_CONFIGS = [
    {"name": "Short-context (decay=30)", "decay_length": 30.0},
    {"name": "Medium-context (decay=100)", "decay_length": 100.0},
    {"name": "Long-context (decay=400)", "decay_length": 400.0},
]

# Motifs to test
MOTIF_NAMES = ["CTCF", "GATA", "SP1"]


def main() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(2024)

    # Distance grid (in bp, log-spaced for wide coverage)
    distances = np.unique(
        np.concatenate([
            np.arange(1, 50, 5),
            np.arange(50, 200, 20),
            np.arange(200, SEQ_LENGTH // 2, 100),
        ])
    ).astype(int)

    # Colours and line styles for models
    model_colors = ["#4C72B0", "#DD8452", "#55A868"]
    model_ls = ["-", "--", "-."]

    sns.set_theme(style="whitegrid", font_scale=1.05)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)

    for mi, motif_name in enumerate(MOTIF_NAMES):
        ax = axes[mi]

        for ci, cfg in enumerate(MODEL_CONFIGS):
            model = SyntheticModel(
                seq_length=SEQ_LENGTH,
                embed_dim=64,
                decay_length=cfg["decay_length"],
            )

            print(f"  Computing gNIAH: {motif_name} x {cfg['name']}...")
            sensitivity = gniah_sensitivity(
                model_fn=model,
                motif_name=motif_name,
                distances=distances,
                seq_length=SEQ_LENGTH,
                n_samples=N_SAMPLES,
                rng=rng,
                show_progress=False,
            )

            # Normalise to peak sensitivity for visual comparison
            peak = sensitivity.max()
            if peak > 0:
                sensitivity_norm = sensitivity / peak
            else:
                sensitivity_norm = sensitivity

            ax.plot(
                distances,
                sensitivity_norm,
                color=model_colors[ci],
                linestyle=model_ls[ci],
                linewidth=2.0,
                alpha=0.85,
                label=cfg["name"],
            )

        ax.set_xlabel("Insertion Distance from Center (bp)", fontsize=12)
        if mi == 0:
            ax.set_ylabel("Normalised gNIAH Sensitivity", fontsize=12)
        ax.set_title(f"{motif_name} motif", fontsize=13, fontweight="bold")

        # Mark the motif consensus on the panel
        motif_seq = MOTIFS[motif_name]
        ax.text(
            0.97, 0.95, f"Motif: {motif_seq}",
            transform=ax.transAxes, fontsize=8,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray", alpha=0.8),
        )

        ax.set_xlim(0, distances.max() * 1.02)
        ax.set_ylim(-0.02, 1.15)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=3,
        fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.06),
    )

    fig.suptitle(
        "Figure 12: gNIAH Sensitivity vs Distance for Regulatory Motifs",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    fig.savefig(output_dir / "fig12_gniah_sensitivity.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(output_dir / "fig12_gniah_sensitivity.png", bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"Figure 12 saved to {output_dir / 'fig12_gniah_sensitivity.pdf'}")
    print(f"Figure 12 saved to {output_dir / 'fig12_gniah_sensitivity.png'}")


if __name__ == "__main__":
    main()
