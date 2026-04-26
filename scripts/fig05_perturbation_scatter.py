"""Figure 5: Scatter plot of ECL_0.9(shuffle) vs ECL_0.9(substitution).

Plots ECL at beta=0.9 under dinucleotide shuffle vs random substitution
perturbation across 500 synthetic promoter loci, colored by model.
Points above the diagonal indicate that shuffle perturbation yields
larger ECL estimates than substitution.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ecl.ecl import ECL
from ecl.metrics import squared_euclidean
from ecl.models.base import SyntheticModel
from ecl.perturbations import DinucleotideShuffle, RandomSubstitution

# ---------------------------------------------------------------------------
# To use real genomic models, replace the SyntheticModel entries below with:
#
#   from ecl.models.enformer import EnformerWrapper
#   from ecl.models.borzoi import BorzoiWrapper
#   from ecl.models.hyenadna import HyenaDNAWrapper
#   from ecl.models.caduceus import CaduceusWrapper
#   from ecl.models.dnabert2 import DNABERT2Wrapper
#   from ecl.models.evo2 import Evo2Wrapper
#
#   models = {
#       "Enformer": EnformerWrapper(),
#       ...
#   }
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "Enformer": 150.0,
    "Borzoi": 200.0,
    "HyenaDNA": 80.0,
    "Caduceus": 120.0,
    "DNABERT-2": 60.0,
    "Evo 2 (7B)": 220.0,
    "NT-v2": 90.0,
    "NT-v3": 100.0,
}

SEQ_LENGTH = 300
EMBED_DIM = 32
N_LOCI = 500
N_SEQUENCES_PER_LOCUS = 3
MAX_DISTANCE = 120
BETA = 0.9
SEED = 99


def fast_ecl_at_locus(
    model_fn, sequences, reference, max_distance, perturbation, rng, block_width=1
):
    """Compute ECL_beta for a single locus from a small batch of sequences.

    Uses a lightweight single-pass estimation (one position per distance)
    to keep runtime tractable for 500 loci.
    """
    n, L = sequences.shape
    D = max_distance
    half_block = block_width // 2
    distances = np.arange(D + 1)
    influence = np.zeros(D + 1, dtype=np.float64)

    for d in range(D + 1):
        candidates = []
        if reference - d >= 0:
            candidates.append(reference - d)
        if d > 0 and reference + d < L:
            candidates.append(reference + d)
        if not candidates:
            continue

        pos = rng.choice(candidates)
        if block_width <= 1:
            pos_arr = np.array([pos])
        else:
            lo = max(0, int(pos) - half_block)
            hi = min(L, int(pos) + half_block + 1)
            pos_arr = np.arange(lo, hi)

        accum = 0.0
        for t in range(n):
            seq = sequences[t]
            z = model_fn(seq)
            perturbed = perturbation(seq, pos_arr, rng)
            z_pert = model_fn(perturbed)
            accum += float(squared_euclidean(z, z_pert))
        influence[d] = accum / n

    return ECL(distances, influence, beta=BETA)


def main():
    rng = np.random.default_rng(SEED)
    from _config import FIGURE_DIR as output_dir

    reference = SEQ_LENGTH // 2

    perturbations = {
        "substitution": (RandomSubstitution(), 1),
        "shuffle": (DinucleotideShuffle(), 20),
    }

    # Pre-generate all locus sequences
    all_sequences = rng.integers(
        0, 4, size=(N_LOCI, N_SEQUENCES_PER_LOCUS, SEQ_LENGTH), dtype=np.int8
    )

    # Build synthetic models
    models = {}
    for name, decay in MODEL_CONFIGS.items():
        models[name] = SyntheticModel(
            seq_length=SEQ_LENGTH,
            embed_dim=EMBED_DIM,
            decay_length=decay,
            reference=reference,
        )

    # Compute ECL for each model x perturbation x locus
    scatter_data = {name: {"substitution": [], "shuffle": []} for name in MODEL_CONFIGS}

    for model_name, model in models.items():
        print(f"  Computing {N_LOCI} loci for {model_name}...")
        for locus_idx in range(N_LOCI):
            seqs = all_sequences[locus_idx]
            for pert_name, (pert, bw) in perturbations.items():
                ecl_val = fast_ecl_at_locus(
                    model_fn=model,
                    sequences=seqs,
                    reference=reference,
                    max_distance=MAX_DISTANCE,
                    perturbation=pert,
                    rng=rng,
                    block_width=bw,
                )
                scatter_data[model_name][pert_name].append(ecl_val)

    # Plot scatter
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    palette = sns.color_palette("husl", len(MODEL_CONFIGS))
    fig, ax = plt.subplots(figsize=(9, 8))

    # Diagonal reference line
    max_ecl = MAX_DISTANCE
    ax.plot([0, max_ecl], [0, max_ecl], "k--", linewidth=0.8, alpha=0.5, label="y = x (equal ECL)")

    for idx, (model_name, data) in enumerate(scatter_data.items()):
        ecl_sub = np.array(data["substitution"])
        ecl_shuf = np.array(data["shuffle"])
        color = palette[idx]

        ax.scatter(
            ecl_sub,
            ecl_shuf,
            color=color,
            alpha=0.35,
            s=12,
            label=model_name,
            edgecolors="none",
        )
        # Overlay mean point
        ax.scatter(
            ecl_sub.mean(),
            ecl_shuf.mean(),
            color=color,
            s=100,
            marker="D",
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )

    ax.set_xlabel(r"ECL$_{0.9}$ (Substitution) [bp]", fontsize=12)
    ax.set_ylabel(r"ECL$_{0.9}$ (Shuffle) [bp]", fontsize=12)
    ax.set_title(
        "Perturbation Comparison: ECL(Shuffle) vs ECL(Substitution)\n"
        f"({N_LOCI} promoter loci per model; diamonds = means)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.set_xlim(0, max_ecl)
    ax.set_ylim(0, max_ecl)
    ax.set_aspect("equal")

    plt.tight_layout()

    # Save
    for ext in ("pdf", "png"):
        out_path = output_dir / f"fig05_perturbation_scatter.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure 5 saved to {output_dir}/fig05_perturbation_scatter.[pdf|png]")


if __name__ == "__main__":
    main()
