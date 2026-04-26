"""Figure 1: Influence profiles I_hat(d; r) vs distance for promoter-centered loci.

Eight-panel log-scale plot showing influence decay for eight synthetic models
(surrogates for Enformer, Borzoi, HyenaDNA, Caduceus, DNABERT-2, Evo 2 7B,
NT-v2, NT-v3), averaged over promoter-centered loci. Two perturbation types:
substitution and dinucleotide shuffle. Shaded +/-1 SE bands.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ecl.influence import compute_influence_profile
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
#       "Borzoi": BorzoiWrapper(),
#       ...
#   }
# ---------------------------------------------------------------------------

# Synthetic model configurations: name -> decay_length (bp)
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

SEQ_LENGTH = 500
EMBED_DIM = 64
N_LOCI = 10  # Number of promoter-centered loci to average over
N_SEQUENCES = 5  # Sequences per locus
MAX_DISTANCE = 200
SEED = 42


def generate_promoter_sequences(n_loci, n_seq, seq_length, rng):
    """Generate random sequences simulating promoter-centered loci."""
    sequences = rng.integers(0, 4, size=(n_loci, n_seq, seq_length), dtype=np.int8)
    return sequences


def compute_profiles_with_se(
    model_fn,
    sequences_all_loci,
    reference,
    max_distance,
    perturbation,
    rng,
    block_width=1,
):
    """Compute mean influence profile and SE across loci."""
    n_loci = sequences_all_loci.shape[0]
    all_profiles = []

    for locus_idx in range(n_loci):
        seqs = sequences_all_loci[locus_idx]
        distances, influence = compute_influence_profile(
            model_fn=model_fn,
            sequences=seqs,
            reference=reference,
            max_distance=max_distance,
            positions_per_distance=2,
            perturbation=perturbation,
            rng=rng,
            show_progress=False,
            block_width=block_width,
        )
        all_profiles.append(influence)

    all_profiles = np.array(all_profiles)
    mean_profile = np.mean(all_profiles, axis=0)
    se_profile = np.std(all_profiles, axis=0, ddof=1) / np.sqrt(n_loci)
    return distances, mean_profile, se_profile


def main():
    rng = np.random.default_rng(SEED)
    from _config import FIGURE_DIR as output_dir

    # Generate promoter-centered loci
    sequences = generate_promoter_sequences(N_LOCI, N_SEQUENCES, SEQ_LENGTH, rng)
    reference = SEQ_LENGTH // 2

    perturbations = {
        "Substitution": (RandomSubstitution(), 1),
        "Shuffle": (DinucleotideShuffle(), 20),
    }

    # Build synthetic models
    models = {}
    for name, decay in MODEL_CONFIGS.items():
        models[name] = SyntheticModel(
            seq_length=SEQ_LENGTH,
            embed_dim=EMBED_DIM,
            decay_length=decay,
            reference=reference,
        )

    # Compute profiles
    results = {}
    for model_name, model in models.items():
        results[model_name] = {}
        for pert_name, (pert, bw) in perturbations.items():
            print(f"  Computing: {model_name} / {pert_name}...")
            d, mean_prof, se_prof = compute_profiles_with_se(
                model_fn=model,
                sequences_all_loci=sequences,
                reference=reference,
                max_distance=MAX_DISTANCE,
                perturbation=pert,
                rng=rng,
                block_width=bw,
            )
            results[model_name][pert_name] = (d, mean_prof, se_prof)

    # Plot: 2x4 grid
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey=True)
    model_names = list(MODEL_CONFIGS.keys())
    colors = {"Substitution": "#1f77b4", "Shuffle": "#ff7f0e"}

    for idx, model_name in enumerate(model_names):
        row, col = divmod(idx, 4)
        ax = axes[row, col]

        for pert_name, color in colors.items():
            d, mean_prof, se_prof = results[model_name][pert_name]
            # Avoid log(0): offset distances by 1
            d_plot = d[1:]  # skip d=0 for log scale
            mean_plot = mean_prof[1:]
            se_plot = se_prof[1:]

            ax.semilogy(d_plot, mean_plot, color=color, label=pert_name, linewidth=1.5)
            ax.fill_between(
                d_plot,
                np.maximum(mean_plot - se_plot, 1e-12),
                mean_plot + se_plot,
                alpha=0.2,
                color=color,
            )

        ax.set_title(model_name, fontsize=12, fontweight="bold")
        if row == 1:
            ax.set_xlabel("Distance d (bp)")
        if col == 0:
            ax.set_ylabel(r"$\hat{I}(d; r)$")
        if idx == 0:
            ax.legend(fontsize=9, loc="upper right")

    fig.suptitle(
        "Influence Profiles for Promoter-Centered Loci",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    for ext in ("pdf", "png"):
        out_path = output_dir / f"fig01_influence_profiles_promoter.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure 1 saved to {output_dir}/fig01_influence_profiles_promoter.[pdf|png]")


if __name__ == "__main__":
    main()
