"""Figure 2: Influence profiles I_hat(d; r) vs distance for enhancer-centered loci.

Same format as Figure 1 but for enhancer-centered loci. Synthetic models use
different decay characteristics to reflect the distinct chromatin context of
distal regulatory elements.
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

# Enhancer-centered models: different decay profiles from promoter loci
# (enhancers tend to have broader influence, so we shift decay lengths)
MODEL_CONFIGS = {
    "Enformer": 180.0,
    "Borzoi": 240.0,
    "HyenaDNA": 100.0,
    "Caduceus": 140.0,
    "DNABERT-2": 70.0,
    "Evo-2": 300.0,
}

SEQ_LENGTH = 500
EMBED_DIM = 64
N_LOCI = 10  # Number of enhancer-centered loci to average over
N_SEQUENCES = 5  # Sequences per locus
MAX_DISTANCE = 200
SEED = 123


def generate_enhancer_sequences(n_loci, n_seq, seq_length, rng):
    """Generate random sequences simulating enhancer-centered loci.

    Enhancer regions may have slightly higher GC content; here we use
    a simple approximation with weighted nucleotide sampling.
    """
    # Slightly elevated GC for enhancer regions
    gc_prob = 0.55
    at_prob = 1.0 - gc_prob
    probs = [at_prob / 2, gc_prob / 2, gc_prob / 2, at_prob / 2]
    sequences = rng.choice(
        4, size=(n_loci, n_seq, seq_length), p=probs
    ).astype(np.int8)
    return sequences


def compute_profiles_with_se(model_fn, sequences_all_loci, reference, max_distance,
                              perturbation, rng):
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
        )
        all_profiles.append(influence)

    all_profiles = np.array(all_profiles)
    mean_profile = np.mean(all_profiles, axis=0)
    se_profile = np.std(all_profiles, axis=0, ddof=1) / np.sqrt(n_loci)
    return distances, mean_profile, se_profile


def main():
    rng = np.random.default_rng(SEED)
    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate enhancer-centered loci
    sequences = generate_enhancer_sequences(N_LOCI, N_SEQUENCES, SEQ_LENGTH, rng)
    reference = SEQ_LENGTH // 2

    perturbations = {
        "Substitution": RandomSubstitution(),
        "Shuffle": DinucleotideShuffle(),
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
        for pert_name, pert in perturbations.items():
            print(f"  Computing: {model_name} / {pert_name}...")
            d, mean_prof, se_prof = compute_profiles_with_se(
                model_fn=model,
                sequences_all_loci=sequences,
                reference=reference,
                max_distance=MAX_DISTANCE,
                perturbation=pert,
                rng=rng,
            )
            results[model_name][pert_name] = (d, mean_prof, se_prof)

    # Plot: 2x3 grid
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    model_names = list(MODEL_CONFIGS.keys())
    colors = {"Substitution": "#2ca02c", "Shuffle": "#d62728"}

    for idx, model_name in enumerate(model_names):
        row, col = divmod(idx, 3)
        ax = axes[row, col]

        for pert_name, color in colors.items():
            d, mean_prof, se_prof = results[model_name][pert_name]
            # Skip d=0 for log scale
            d_plot = d[1:]
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
        "Figure 2: Influence Profiles for Enhancer-Centered Loci",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    for ext in ("pdf", "png"):
        out_path = output_dir / f"fig02_influence_profiles_enhancer.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure 2 saved to {output_dir}/fig02_influence_profiles_enhancer.[pdf|png]")


if __name__ == "__main__":
    main()
