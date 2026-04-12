"""Figure 6: Influence profiles for trained vs random-weight models.

Overlays influence profiles from a 'trained' synthetic model (with structured
exponential decay) and a 'random' model (near-uniform weights, simulating
untrained random initialization). Demonstrates that trained models develop
distance-dependent context usage, while random models show flat profiles.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ecl.ecl import ECL, cumulative_influence
from ecl.influence import compute_influence_profile
from ecl.models.base import SyntheticModel
from ecl.perturbations import RandomSubstitution

# ---------------------------------------------------------------------------
# To use real genomic models with trained vs random weights:
#
#   from ecl.models.enformer import EnformerWrapper
#
#   trained_model = EnformerWrapper(checkpoint="path/to/trained/weights")
#   random_model = EnformerWrapper(checkpoint=None, random_init=True)
# ---------------------------------------------------------------------------

SEQ_LENGTH = 500
EMBED_DIM = 64
N_SEQUENCES = 15
MAX_DISTANCE = 200
N_LOCI = 8  # Average over multiple loci for smoother curves
SEED = 31


class RandomWeightModel:
    """Synthetic model simulating a randomly initialized (untrained) network.

    Weights are drawn iid from a Gaussian distribution with no distance
    structure, producing a nearly flat influence profile (all positions
    contribute equally in expectation).

    Parameters
    ----------
    seq_length : int
    embed_dim : int
    reference : int or None
    noise_scale : float
        Scale of random noise added to break exact uniformity.
    """

    def __init__(self, seq_length=500, embed_dim=64, reference=None, noise_scale=0.01):
        self._seq_length = seq_length
        self._embed_dim = embed_dim
        self._reference = reference if reference is not None else seq_length // 2

        rng = np.random.default_rng(12345)
        # Near-uniform weights (no distance structure)
        base_weight = 1.0 / seq_length
        noise = rng.normal(0, noise_scale * base_weight, size=seq_length)
        self._weights = np.abs(base_weight + noise)
        self._weights /= self._weights.sum()

        self._projection = rng.standard_normal((4, embed_dim))

    def __call__(self, sequence):
        seq = np.asarray(sequence, dtype=np.int64)
        one_hot = np.eye(4, dtype=np.float64)[seq]
        weighted = np.einsum("i,ij->j", self._weights, one_hot)
        return weighted @ self._projection

    @property
    def nominal_context(self):
        return self._seq_length

    @property
    def embedding_dim(self):
        return self._embed_dim


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

    reference = SEQ_LENGTH // 2
    perturbation = RandomSubstitution()

    # Generate loci
    sequences = rng.integers(
        0, 4, size=(N_LOCI, N_SEQUENCES, SEQ_LENGTH), dtype=np.int8
    )

    # Models to compare
    model_configs = {
        "Trained (decay=150 bp)": {
            "model": SyntheticModel(
                seq_length=SEQ_LENGTH,
                embed_dim=EMBED_DIM,
                decay_length=150.0,
                reference=reference,
            ),
            "color": "#1f77b4",
            "linestyle": "-",
        },
        "Trained (decay=80 bp)": {
            "model": SyntheticModel(
                seq_length=SEQ_LENGTH,
                embed_dim=EMBED_DIM,
                decay_length=80.0,
                reference=reference,
            ),
            "color": "#2ca02c",
            "linestyle": "-",
        },
        "Random weights": {
            "model": RandomWeightModel(
                seq_length=SEQ_LENGTH,
                embed_dim=EMBED_DIM,
                reference=reference,
            ),
            "color": "#d62728",
            "linestyle": "--",
        },
    }

    # Compute profiles
    results = {}
    for name, cfg in model_configs.items():
        print(f"  Computing: {name}...")
        d, mean_prof, se_prof = compute_profiles_with_se(
            model_fn=cfg["model"],
            sequences_all_loci=sequences,
            reference=reference,
            max_distance=MAX_DISTANCE,
            perturbation=perturbation,
            rng=rng,
        )
        # ECL at beta=0.9
        ecl_val = ECL(d, mean_prof, beta=0.9)
        results[name] = {
            "distances": d,
            "mean": mean_prof,
            "se": se_prof,
            "ecl": ecl_val,
            "color": cfg["color"],
            "linestyle": cfg["linestyle"],
        }

    # Plot
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left panel: influence profiles (log scale)
    for name, res in results.items():
        d = res["distances"][1:]  # skip d=0 for log scale
        mean = res["mean"][1:]
        se = res["se"][1:]

        ax1.semilogy(
            d, mean,
            color=res["color"],
            linestyle=res["linestyle"],
            linewidth=2.0,
            label=f"{name} (ECL={res['ecl']} bp)",
        )
        ax1.fill_between(
            d,
            np.maximum(mean - se, 1e-12),
            mean + se,
            alpha=0.15,
            color=res["color"],
        )

    ax1.set_xlabel("Distance d (bp)", fontsize=12)
    ax1.set_ylabel(r"$\hat{I}(d; r)$ (log scale)", fontsize=12)
    ax1.set_title("Influence Profiles", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper right")

    # Right panel: cumulative influence fraction
    for name, res in results.items():
        d = res["distances"]
        mean = res["mean"]
        radii, cumul = cumulative_influence(d, mean)
        total = cumul[-1]
        fraction = cumul / total if total > 0 else np.zeros_like(cumul)

        ax2.plot(
            radii, fraction,
            color=res["color"],
            linestyle=res["linestyle"],
            linewidth=2.0,
            label=f"{name}",
        )
        # Mark ECL
        ax2.axvline(x=res["ecl"], color=res["color"], linestyle=":",
                     linewidth=1.0, alpha=0.6)

    # Beta threshold
    ax2.axhline(y=0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.text(MAX_DISTANCE * 0.98, 0.91, r"$\beta=0.9$", ha="right", fontsize=9,
             color="gray")

    ax2.set_xlabel("Radius l (bp)", fontsize=12)
    ax2.set_ylabel(r"$I_{\leq l}(r) \;/\; I_{\mathrm{tot}}(r)$", fontsize=12)
    ax2.set_title("Cumulative Influence", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9, loc="lower right")
    ax2.set_xlim(0, MAX_DISTANCE)
    ax2.set_ylim(0, 1.05)

    fig.suptitle(
        "Figure 6: Trained vs Random-Weight Model Influence",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    # Save
    for ext in ("pdf", "png"):
        out_path = output_dir / f"fig06_trained_vs_random.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure 6 saved to {output_dir}/fig06_trained_vs_random.[pdf|png]")


if __name__ == "__main__":
    main()
