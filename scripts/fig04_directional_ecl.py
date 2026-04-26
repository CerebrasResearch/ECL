"""Figure 4: Directional ECL -- upstream vs downstream influence profiles.

Two-panel figure showing upstream (I^-) and downstream (I^+) influence
profiles for synthetic surrogates of Enformer and Borzoi. Demonstrates
directional asymmetry in context usage.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ecl.ecl import directional_ecl
from ecl.metrics import squared_euclidean
from ecl.perturbations import RandomSubstitution

# ---------------------------------------------------------------------------
# To use real genomic models, replace the SyntheticModel entries below with:
#
#   from ecl.models.enformer import EnformerWrapper
#   from ecl.models.borzoi import BorzoiWrapper
#
#   enformer = EnformerWrapper()
#   borzoi = BorzoiWrapper()
# ---------------------------------------------------------------------------

SEQ_LENGTH = 500
EMBED_DIM = 64
N_SEQUENCES = 15
MAX_DISTANCE = 200
SEED = 77


class AsymmetricSyntheticModel:
    """Synthetic model with directional asymmetry in influence decay.

    Upstream and downstream regions have different decay lengths, simulating
    models that attend more to one side (e.g., promoter upstream vs gene body).

    Parameters
    ----------
    seq_length : int
    embed_dim : int
    decay_upstream : float
        Decay length for positions upstream of reference (i < r).
    decay_downstream : float
        Decay length for positions downstream of reference (i > r).
    reference : int or None
    """

    def __init__(
        self,
        seq_length=500,
        embed_dim=64,
        decay_upstream=100.0,
        decay_downstream=200.0,
        reference=None,
    ):
        self._seq_length = seq_length
        self._embed_dim = embed_dim
        self._reference = reference if reference is not None else seq_length // 2

        positions = np.arange(seq_length)
        distances = (positions - self._reference).astype(np.float64)

        # Asymmetric weights
        weights = np.zeros(seq_length, dtype=np.float64)
        upstream_mask = distances <= 0
        downstream_mask = distances > 0
        weights[upstream_mask] = np.exp(distances[upstream_mask] / decay_upstream)
        weights[downstream_mask] = np.exp(-distances[downstream_mask] / decay_downstream)
        weights /= weights.sum()
        self._weights = weights

        rng = np.random.default_rng(42)
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


def compute_directional_profiles(model_fn, sequences, reference, max_distance, perturbation, rng):
    """Compute upstream and downstream influence profiles separately."""
    n, L = sequences.shape
    D = max_distance

    distances = np.arange(1, D + 1)
    influence_upstream = np.zeros(D, dtype=np.float64)
    influence_downstream = np.zeros(D, dtype=np.float64)

    for d_idx, d in enumerate(distances):
        # Upstream: position r - d
        up_pos = reference - d
        if up_pos >= 0:
            accum = 0.0
            for t in range(n):
                seq = sequences[t]
                z = model_fn(seq)
                perturbed = perturbation(seq, np.array([up_pos]), rng)
                z_pert = model_fn(perturbed)
                accum += float(squared_euclidean(z, z_pert))
            influence_upstream[d_idx] = accum / n

        # Downstream: position r + d
        down_pos = reference + d
        if down_pos < L:
            accum = 0.0
            for t in range(n):
                seq = sequences[t]
                z = model_fn(seq)
                perturbed = perturbation(seq, np.array([down_pos]), rng)
                z_pert = model_fn(perturbed)
                accum += float(squared_euclidean(z, z_pert))
            influence_downstream[d_idx] = accum / n

    return distances, influence_upstream, influence_downstream


def main():
    rng = np.random.default_rng(SEED)
    from _config import FIGURE_DIR as output_dir

    reference = SEQ_LENGTH // 2
    perturbation = RandomSubstitution()
    sequences = rng.integers(0, 4, size=(N_SEQUENCES, SEQ_LENGTH), dtype=np.int8)

    # Asymmetric synthetic surrogates
    models = {
        "Enformer (synthetic)": AsymmetricSyntheticModel(
            seq_length=SEQ_LENGTH,
            embed_dim=EMBED_DIM,
            decay_upstream=100.0,
            decay_downstream=180.0,
            reference=reference,
        ),
        "Borzoi (synthetic)": AsymmetricSyntheticModel(
            seq_length=SEQ_LENGTH,
            embed_dim=EMBED_DIM,
            decay_upstream=160.0,
            decay_downstream=250.0,
            reference=reference,
        ),
    }

    # Compute directional profiles
    results = {}
    for model_name, model in models.items():
        print(f"  Computing: {model_name}...")
        distances, infl_up, infl_down = compute_directional_profiles(
            model_fn=model,
            sequences=sequences,
            reference=reference,
            max_distance=MAX_DISTANCE,
            perturbation=perturbation,
            rng=rng,
        )
        ecl_up, ecl_down, asymmetry = directional_ecl(distances, infl_up, infl_down, beta=0.9)
        results[model_name] = {
            "distances": distances,
            "upstream": infl_up,
            "downstream": infl_down,
            "ecl_up": ecl_up,
            "ecl_down": ecl_down,
            "asymmetry": asymmetry,
        }

    # Plot: 1x2 panels
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for idx, (model_name, res) in enumerate(results.items()):
        ax = axes[idx]
        d = res["distances"]

        ax.semilogy(
            d,
            res["upstream"],
            color="#1f77b4",
            linewidth=2.0,
            label=f"Upstream (ECL={res['ecl_up']} bp)",
        )
        ax.semilogy(
            d,
            res["downstream"],
            color="#ff7f0e",
            linewidth=2.0,
            label=f"Downstream (ECL={res['ecl_down']} bp)",
        )

        # Mark ECLs
        ax.axvline(x=res["ecl_up"], color="#1f77b4", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.axvline(x=res["ecl_down"], color="#ff7f0e", linestyle="--", linewidth=1.0, alpha=0.7)

        ax.set_title(
            f"{model_name}\nAsymmetry ratio: {res['asymmetry']:.2f}", fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Distance d (bp)", fontsize=11)
        if idx == 0:
            ax.set_ylabel(r"Influence $I^{\pm}(d; r)$", fontsize=11)
        ax.legend(fontsize=10, loc="upper right")

    fig.suptitle(
        "Directional Influence Profiles (Upstream vs Downstream)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    # Save
    for ext in ("pdf", "png"):
        out_path = output_dir / f"fig04_directional_ecl.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure 4 saved to {output_dir}/fig04_directional_ecl.[pdf|png]")


if __name__ == "__main__":
    main()
