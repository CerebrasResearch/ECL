#!/usr/bin/env python
"""ECL Model Comparison Tutorial.

Demonstrates how to compare effective context lengths across multiple models
using the permutation test (Algorithm 5), directional ECL, and Context Decay
Spectroscopy.

Uses synthetic models with different decay characteristics to simulate
the comparison between architectures (e.g., CNN vs Transformer vs SSM).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ecl.cds import fit_cds, select_n_components
from ecl.ecl import ECL, directional_ecl
from ecl.estimation import bootstrap_ecl_ci, permutation_test
from ecl.influence import compute_influence_profile
from ecl.models.base import SyntheticModel
from ecl.perturbations import DinucleotideShuffle, RandomSubstitution

# ---------------------------------------------------------------------------
# Step 1: Create models with different context utilization profiles
# ---------------------------------------------------------------------------
print("=== ECL Model Comparison Tutorial ===\n")
print("Step 1: Creating synthetic models...")

SEQ_LEN = 500
EMBED_DIM = 32
N_LOCI = 50  # Number of loci for comparison
N_SEQS = 15  # Sequences per locus

# Simulate three architectures
models = {
    "CNN (short context)": SyntheticModel(SEQ_LEN, EMBED_DIM, decay_length=20),
    "Transformer (medium)": SyntheticModel(SEQ_LEN, EMBED_DIM, decay_length=80),
    "SSM (long context)": SyntheticModel(SEQ_LEN, EMBED_DIM, decay_length=150),
}

rng = np.random.default_rng(123)
ref = SEQ_LEN // 2

# ---------------------------------------------------------------------------
# Step 2: Estimate influence profiles and ECL for each model
# ---------------------------------------------------------------------------
print("\nStep 2: Estimating influence profiles...")

profiles = {}
ecl_values = {}
perturbation = RandomSubstitution()

for name, model in models.items():
    print(f"\n  Processing {name}...")
    seqs = rng.integers(0, 4, size=(N_SEQS, SEQ_LEN)).astype(np.int8)
    distances, influence = compute_influence_profile(
        model_fn=model,
        sequences=seqs,
        reference=ref,
        max_distance=200,
        positions_per_distance=3,
        perturbation=perturbation,
        rng=rng,
        show_progress=False,
    )
    profiles[name] = (distances, influence)

    for beta in [0.5, 0.9, 0.95]:
        ecl = ECL(distances, influence, beta=beta)
        print(f"    ECL_{beta} = {ecl} bp")

# ---------------------------------------------------------------------------
# Step 3: Per-locus ECL comparison with permutation test
# ---------------------------------------------------------------------------
print("\nStep 3: Per-locus comparison (permutation test)...")

# Collect per-locus ECL estimates for two models
model_a = models["CNN (short context)"]
model_b = models["Transformer (medium)"]

ecl_a_loci = np.zeros(N_LOCI)
ecl_b_loci = np.zeros(N_LOCI)

for j in range(N_LOCI):
    seqs = rng.integers(0, 4, size=(N_SEQS, SEQ_LEN)).astype(np.int8)

    _, infl_a = compute_influence_profile(
        model_a, seqs, ref, max_distance=200, positions_per_distance=2,
        perturbation=perturbation, rng=rng, show_progress=False,
    )
    ecl_a_loci[j] = ECL(profiles["CNN (short context)"][0], infl_a, beta=0.9)

    _, infl_b = compute_influence_profile(
        model_b, seqs, ref, max_distance=200, positions_per_distance=2,
        perturbation=perturbation, rng=rng, show_progress=False,
    )
    ecl_b_loci[j] = ECL(profiles["Transformer (medium)"][0], infl_b, beta=0.9)

mean_diff, p_value, ci_width = permutation_test(
    ecl_b_loci, ecl_a_loci, n_permutations=2000, rng=rng
)
print(f"  Mean delta(ECL_0.9): Transformer - CNN = {mean_diff:.1f} bp")
print(f"  95% CI: [{mean_diff - ci_width:.1f}, {mean_diff + ci_width:.1f}]")
print(f"  Permutation p-value: {p_value:.4f}")

# ---------------------------------------------------------------------------
# Step 4: Context Decay Spectroscopy comparison
# ---------------------------------------------------------------------------
print("\nStep 4: Context Decay Spectroscopy...")

for name in models:
    distances, influence = profiles[name]
    best_K, results = select_n_components(distances, influence, max_K=3)
    best = results[best_K - 1]
    print(f"\n  {name}: best K={best_K}")
    for k in range(best_K):
        print(f"    Component {k+1}: a={best['amplitudes'][k]:.3f}, "
              f"lambda={best['decay_rates'][k]:.4f} "
              f"(half-life={0.693/best['decay_rates'][k]:.0f} bp)")

# ---------------------------------------------------------------------------
# Step 5: Visualization
# ---------------------------------------------------------------------------
print("\nStep 5: Generating comparison plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors = {"CNN (short context)": "#e41a1c", "Transformer (medium)": "#377eb8",
          "SSM (long context)": "#4daf4a"}

# Panel A: Influence profiles (log scale)
ax = axes[0, 0]
for name in models:
    d, infl = profiles[name]
    ax.semilogy(d, infl, label=name, color=colors[name], linewidth=1.5)
ax.set_xlabel("Distance (bp)")
ax.set_ylabel("Influence energy (log scale)")
ax.set_title("A. Influence Profiles")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel B: Cumulative influence
ax = axes[0, 1]
for name in models:
    d, infl = profiles[name]
    from ecl.ecl import cumulative_influence as cum_infl
    _, cumul = cum_infl(d, infl)
    total = cumul[-1]
    if total > 0:
        ax.plot(d, cumul / total, label=name, color=colors[name], linewidth=1.5)
for beta in [0.5, 0.9]:
    ax.axhline(beta, color="gray", linestyle=":", alpha=0.5, label=f"beta={beta}")
ax.set_xlabel("Radius (bp)")
ax.set_ylabel("Cumulative fraction")
ax.set_title("B. Cumulative Influence")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Panel C: Paired difference (Transformer - CNN)
ax = axes[1, 0]
diffs = ecl_b_loci - ecl_a_loci
ax.hist(diffs, bins=20, alpha=0.7, color="#377eb8", edgecolor="black")
ax.axvline(np.mean(diffs), color="red", linewidth=2, label=f"Mean={np.mean(diffs):.0f}")
ax.axvline(0, color="black", linewidth=1, linestyle="--")
ax.set_xlabel("Delta ECL_0.9 (Transformer - CNN) (bp)")
ax.set_ylabel("Count")
ax.set_title(f"C. Paired Differences (p={p_value:.3f})")
ax.legend()
ax.grid(True, alpha=0.3)

# Panel D: CDS fits
ax = axes[1, 1]
for name in models:
    d, infl = profiles[name]
    cds = fit_cds(d, infl, n_components=2)
    ax.semilogy(d, infl, color=colors[name], alpha=0.3, linewidth=1)
    ax.semilogy(d, cds["fitted"], color=colors[name], linewidth=2, label=f"{name} (CDS fit)")
ax.set_xlabel("Distance (bp)")
ax.set_ylabel("Influence energy (log scale)")
ax.set_title("D. CDS Mixture Fits")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_dir = Path(__file__).resolve().parents[2] / "outputs"
out_dir.mkdir(exist_ok=True)
fig.savefig(out_dir / "model_comparison.png", dpi=150, bbox_inches="tight")
fig.savefig(out_dir / "model_comparison.pdf", bbox_inches="tight")
plt.close()

print(f"\n  Plots saved to {out_dir}/model_comparison.{{png,pdf}}")
print("\nTutorial complete!")
