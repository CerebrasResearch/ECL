"""Figure 3: Cumulative influence fraction with ECL estimates and bootstrap CIs.

Plots I_{<=l}(r) / I_tot(r) vs radius l (bp) for all six synthetic models
overlaid on one panel. Horizontal lines at beta thresholds (0.5, 0.8, 0.9,
0.95, 0.99). Vertical lines at ECL estimates. Bootstrap 95% CIs shown as
horizontal error bars at each model's ECL.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ecl.ecl import cumulative_influence
from ecl.estimation import bootstrap_ecl_ci
from ecl.models.base import SyntheticModel
from ecl.perturbations import RandomSubstitution

# ---------------------------------------------------------------------------
# To use real genomic models, replace the SyntheticModel entries below with:
#
#   from ecl.models.enformer import EnformerWrapper
#   from ecl.models.borzoi import BorzoiWrapper
#   ...
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "Enformer": 150.0,
    "Borzoi": 200.0,
    "HyenaDNA": 80.0,
    "Caduceus": 120.0,
    "DNABERT-2": 60.0,
    "Evo-2": 250.0,
}

SEQ_LENGTH = 500
EMBED_DIM = 64
N_SEQUENCES = 20  # More sequences for bootstrap estimation
MAX_DISTANCE = 200
N_BOOTSTRAP = 500
BETA_THRESHOLDS = [0.5, 0.8, 0.9, 0.95, 0.99]
SEED = 42


def compute_per_sample_influence(model_fn, sequences, reference, max_distance, perturbation, rng):
    """Compute per-sample influence values for bootstrap.

    Returns influence_samples of shape (n, D+1) and distances of shape (D+1,).
    """
    from ecl.metrics import squared_euclidean

    n, L = sequences.shape
    if max_distance is None:
        max_distance = L // 2
    D = max_distance

    distances = np.arange(D + 1)
    influence_samples = np.zeros((n, D + 1), dtype=np.float64)

    for t in range(n):
        seq = sequences[t]
        z = model_fn(seq)
        for d in range(D + 1):
            candidates = []
            if reference - d >= 0:
                candidates.append(reference - d)
            if d > 0 and reference + d < L:
                candidates.append(reference + d)
            if not candidates:
                continue
            pos = rng.choice(candidates)
            perturbed = perturbation(seq, np.array([pos]), rng)
            z_pert = model_fn(perturbed)
            influence_samples[t, d] = float(squared_euclidean(z, z_pert))

    return distances, influence_samples


def main():
    rng = np.random.default_rng(SEED)
    from _config import FIGURE_DIR as output_dir

    reference = SEQ_LENGTH // 2
    perturbation = RandomSubstitution()
    sequences = rng.integers(0, 4, size=(N_SEQUENCES, SEQ_LENGTH), dtype=np.int8)

    # Build models
    models = {}
    for name, decay in MODEL_CONFIGS.items():
        models[name] = SyntheticModel(
            seq_length=SEQ_LENGTH,
            embed_dim=EMBED_DIM,
            decay_length=decay,
            reference=reference,
        )

    # Compute cumulative influence and bootstrap CIs for each model
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    palette = sns.color_palette("husl", len(MODEL_CONFIGS))

    fig, ax = plt.subplots(figsize=(10, 7))

    ecl_results = {}
    for idx, (model_name, model) in enumerate(models.items()):
        print(f"  Computing: {model_name}...")
        color = palette[idx]

        # Per-sample influence for bootstrap
        distances, infl_samples = compute_per_sample_influence(
            model_fn=model,
            sequences=sequences,
            reference=reference,
            max_distance=MAX_DISTANCE,
            perturbation=perturbation,
            rng=rng,
        )

        mean_influence = infl_samples.mean(axis=0)

        # Cumulative influence fraction
        radii, cumul = cumulative_influence(distances, mean_influence)
        total = cumul[-1]
        fraction = cumul / total if total > 0 else np.zeros_like(cumul)

        ax.plot(radii, fraction, color=color, linewidth=2.0, label=model_name)

        # ECL at beta=0.9 with bootstrap CI
        ecl_point, ci_lo, ci_hi = bootstrap_ecl_ci(
            influence_samples=infl_samples,
            distances=distances,
            beta=0.9,
            n_bootstrap=N_BOOTSTRAP,
            alpha=0.05,
            rng=rng,
        )
        ecl_results[model_name] = (ecl_point, ci_lo, ci_hi, color)

    # Horizontal threshold lines
    beta_colors = {0.5: "#aaaaaa", 0.8: "#888888", 0.9: "#555555", 0.95: "#333333", 0.99: "#111111"}
    for beta in BETA_THRESHOLDS:
        ax.axhline(
            y=beta,
            color=beta_colors[beta],
            linestyle="--",
            linewidth=0.8,
            alpha=0.6,
        )
        ax.text(
            MAX_DISTANCE * 0.98,
            beta + 0.008,
            rf"$\beta={beta}$",
            ha="right",
            va="bottom",
            fontsize=8,
            color=beta_colors[beta],
        )

    # Vertical ECL lines with bootstrap CI error bars
    for ecl_pt, ci_lo, ci_hi, color in ecl_results.values():
        ax.axvline(x=ecl_pt, color=color, linestyle=":", linewidth=1.0, alpha=0.5)
        ax.errorbar(
            ecl_pt,
            0.9,
            xerr=[[ecl_pt - ci_lo], [ci_hi - ecl_pt]],
            fmt="o",
            color=color,
            markersize=5,
            capsize=3,
            capthick=1.5,
            linewidth=1.5,
        )

    ax.set_xlabel("Radius l (bp)", fontsize=12)
    ax.set_ylabel(r"$I_{\leq l}(r) \;/\; I_{\mathrm{tot}}(r)$", fontsize=12)
    ax.set_title(
        "Figure 3: Cumulative Influence Fraction with ECL Estimates",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.set_xlim(0, MAX_DISTANCE)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    # Save
    for ext in ("pdf", "png"):
        out_path = output_dir / f"fig03_cumulative_influence.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure 3 saved to {output_dir}/fig03_cumulative_influence.[pdf|png]")


if __name__ == "__main__":
    main()
