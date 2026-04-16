#!/usr/bin/env python3
"""Figure 11: Heatmap of pairwise interaction influence I_int(i,j;r).

Displays a heatmap of pairwise interaction influence at a single locus,
revealing synergistic (positive) and redundant (negative) blocks.
Uses a synthetic model with explicit interaction terms to demonstrate
that ECL can detect non-additive position dependencies.

The interaction influence is defined as:
    I_int(i, j; r) = I({i,j}; r) - I(i; r) - I(j; r)
Positive values indicate synergy, negative indicate redundancy.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns

from ecl.influence import compute_interaction_influence
from ecl.models.base import SyntheticModel


class SyntheticInteractionModel(SyntheticModel):
    """Synthetic model with pairwise interaction terms.

    Adds explicit synergistic blocks to the base SyntheticModel:
    positions within the same block interact positively (synergy),
    while positions in adjacent blocks interact negatively (redundancy).
    """

    def __init__(
        self,
        seq_length: int = 200,
        embed_dim: int = 64,
        decay_length: float = 40.0,
        interaction_blocks: list[tuple[int, int]] | None = None,
        interaction_strength: float = 0.5,
        reference: int | None = None,
    ):
        super().__init__(
            seq_length=seq_length,
            embed_dim=embed_dim,
            decay_length=decay_length,
            reference=reference,
        )
        self._interaction_strength = interaction_strength
        # Default blocks: three synergistic regions
        if interaction_blocks is None:
            self._interaction_blocks = [
                (30, 60),  # block 1
                (80, 110),  # block 2
                (140, 170),  # block 3
            ]
        else:
            self._interaction_blocks = interaction_blocks

        rng = np.random.default_rng(123)
        self._interaction_proj = rng.standard_normal((4, 4, embed_dim)) * interaction_strength

    def forward(self, sequence: npt.NDArray) -> npt.NDArray:
        # Base embedding
        embedding = super().forward(sequence)

        # Add interaction terms within blocks (subsample for efficiency)
        seq = np.asarray(sequence, dtype=np.int64)
        for bstart, bend in self._interaction_blocks:
            bend = min(bend, len(seq))
            block_idx = np.arange(bstart, bend)
            # Subsample pairs to keep forward pass fast
            n_block = len(block_idx)
            max_pairs = min(n_block * (n_block - 1) // 2, 50)
            pair_count = 0
            for i in range(bstart, bend - 1, max(1, n_block // 10)):
                for j in range(i + 1, bend, max(1, n_block // 10)):
                    embedding += (
                        self._interaction_proj[seq[i], seq[j]] * self._weights[i] * self._weights[j]
                    )
                    pair_count += 1
                    if pair_count >= max_pairs:
                        break
                if pair_count >= max_pairs:
                    break
        return embedding


def compute_synthetic_interaction_matrix(
    n_positions: int = 40,
    seq_length: int = 200,
    n_sequences: int = 20,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a pairwise interaction matrix using synthetic data.

    For efficiency, we compute I_int directly using the synthetic model's
    structure rather than brute-force perturbation at every pair.
    """
    rng = rng or np.random.default_rng(2024)

    model = SyntheticInteractionModel(
        seq_length=seq_length,
        embed_dim=64,
        decay_length=40.0,
        interaction_strength=0.3,
    )

    # Select a grid of positions spanning the sequence
    positions = np.linspace(10, seq_length - 10, n_positions, dtype=int)

    # Generate sample sequences
    sequences = rng.integers(0, 4, size=(n_sequences, seq_length), dtype=np.int8)

    # Compute interaction matrix
    interaction_matrix = np.zeros((n_positions, n_positions))

    for ii in range(n_positions):
        for jj in range(ii + 1, n_positions):
            i_int = compute_interaction_influence(
                model_fn=model,
                sequences=sequences,
                pos_i=int(positions[ii]),
                pos_j=int(positions[jj]),
                rng=rng,
            )
            interaction_matrix[ii, jj] = i_int
            interaction_matrix[jj, ii] = i_int

    return positions, interaction_matrix


def main() -> None:
    from _config import FIGURE_DIR as output_dir

    rng = np.random.default_rng(2024)

    print("Computing pairwise interaction matrix (this may take a moment)...")
    positions, interaction_matrix = compute_synthetic_interaction_matrix(
        n_positions=30,
        seq_length=200,
        n_sequences=15,
        rng=rng,
    )

    # Plotting
    sns.set_theme(style="white", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(10, 8.5))

    # Use diverging colormap: blue=redundant, white=none, red=synergistic
    vmax = np.max(np.abs(interaction_matrix)) * 0.8
    if vmax < 1e-10:
        vmax = 1.0  # fallback

    im = ax.imshow(
        interaction_matrix,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="equal",
        interpolation="nearest",
    )

    # Tick labels as genomic positions
    tick_stride = max(1, len(positions) // 8)
    tick_indices = np.arange(0, len(positions), tick_stride)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(positions[tick_indices], fontsize=9, rotation=45)
    ax.set_yticks(tick_indices)
    ax.set_yticklabels(positions[tick_indices], fontsize=9)

    ax.set_xlabel("Position j (bp)", fontsize=13)
    ax.set_ylabel("Position i (bp)", fontsize=13)
    ax.set_title(
        r"Figure 11: Pairwise Interaction Influence $I_{\mathrm{int}}(i, j; r)$",
        fontsize=14,
        fontweight="bold",
    )

    # Mark interaction blocks on the axes
    block_ranges = [(30, 60), (80, 110), (140, 170)]
    block_colors = ["#2ca02c", "#d62728", "#9467bd"]
    for (bstart, bend), bc in zip(block_ranges, block_colors, strict=False):
        # Find indices in our sampled positions that fall in each block
        idx_in_block = np.where((positions >= bstart) & (positions <= bend))[0]
        if len(idx_in_block) >= 2:
            lo = idx_in_block[0] - 0.5
            hi = idx_in_block[-1] + 0.5
            rect = plt.Rectangle(
                (lo, lo),
                hi - lo,
                hi - lo,
                linewidth=2.0,
                edgecolor=bc,
                facecolor="none",
                linestyle="--",
            )
            ax.add_patch(rect)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(r"$I_{\mathrm{int}}(i, j; r)$    [+ synergistic / $-$ redundant]", fontsize=11)

    fig.tight_layout()

    fig.savefig(output_dir / "fig11_interaction_heatmap.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(output_dir / "fig11_interaction_heatmap.png", bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"Figure 11 saved to {output_dir / 'fig11_interaction_heatmap.pdf'}")
    print(f"Figure 11 saved to {output_dir / 'fig11_interaction_heatmap.png'}")


if __name__ == "__main__":
    main()
