#!/usr/bin/env python
"""ECL Quickstart Tutorial.

This tutorial demonstrates the core ECL workflow:
1. Create a model (using a synthetic model for demonstration)
2. Generate sequences
3. Compute influence profiles
4. Estimate ECL and related quantities
5. Visualize results

For real genomic models, replace SyntheticModel with one of the model wrappers
(EnformerWrapper, HyenaDNAWrapper, etc.) and use real genomic sequences.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ecl.cds import fit_cds, spectral_ecl
from ecl.ecl import AECP, ECD, ECL, ECP, cumulative_influence, normalized_influence
from ecl.estimation import bootstrap_ecl_ci
from ecl.influence import compute_influence_profile
from ecl.models.base import SyntheticModel
from ecl.perturbations import RandomSubstitution

# ---------------------------------------------------------------------------
# Step 1: Set up the model
# ---------------------------------------------------------------------------
print("Step 1: Creating synthetic model...")

# The SyntheticModel has exponentially decaying influence from a reference position.
# decay_length controls how far influence extends.
model = SyntheticModel(
    seq_length=500,  # 500 bp sequence
    embed_dim=64,  # 64-dimensional embedding
    decay_length=50.0,  # Influence decays over ~50 bp
)

print(f"  Nominal context: {model.nominal_context} bp")
print(f"  Embedding dim:   {model.embedding_dim}")

# ---------------------------------------------------------------------------
# Step 2: Generate sequences
# ---------------------------------------------------------------------------
print("\nStep 2: Generating random sequences...")

rng = np.random.default_rng(42)
n_sequences = 30
sequences = rng.integers(0, 4, size=(n_sequences, model.nominal_context)).astype(np.int8)
reference = model.nominal_context // 2  # Center position

print(f"  {n_sequences} sequences of length {model.nominal_context}")
print(f"  Reference position: {reference}")

# ---------------------------------------------------------------------------
# Step 3: Compute influence profile (Algorithm 2)
# ---------------------------------------------------------------------------
print("\nStep 3: Computing binned influence profile...")

perturbation = RandomSubstitution()
distances, influence = compute_influence_profile(
    model_fn=model,
    sequences=sequences,
    reference=reference,
    max_distance=200,
    positions_per_distance=5,
    perturbation=perturbation,
    rng=rng,
    show_progress=True,
)

print(f"  Profile computed for distances 0..{distances[-1]} bp")
print(f"  Max influence: {influence.max():.4f} at d={distances[influence.argmax()]} bp")

# ---------------------------------------------------------------------------
# Step 4: Compute ECL and related quantities
# ---------------------------------------------------------------------------
print("\nStep 4: Computing ECL quantities...")

# ECL at various beta thresholds (Definition 4.1)
betas = [0.5, 0.8, 0.9, 0.95, 0.99]
for beta in betas:
    ecl = ECL(distances, influence, beta=beta)
    print(f"  ECL_{beta:.2f} = {ecl} bp")

# Effective Context Profile (Definition 4.2)
ecp_betas, ecp_values = ECP(distances, influence)

# Area Under ECP (Definition 4.3)
aecp = AECP(distances, influence)
print(f"\n  AECP (mean influence distance) = {aecp:.1f} bp")

# Effective Context Dimension (Definition 4.5)
ecd = ECD(distances, influence)
print(f"  ECD (effective number of positions) = {ecd:.1f}")

# ---------------------------------------------------------------------------
# Step 5: Bootstrap confidence intervals (Algorithm 3)
# ---------------------------------------------------------------------------
print("\nStep 5: Bootstrap confidence intervals...")

# Create per-sample influence data for bootstrap
influence_samples = np.zeros((n_sequences, len(distances)))
for t in range(n_sequences):
    seq = sequences[t]
    z_orig = model(seq)
    for di, d in enumerate(distances):
        if reference - d >= 0:
            pos = reference - d
        elif reference + d < model.nominal_context:
            pos = reference + d
        else:
            continue
        perturbed = perturbation(seq, np.array([pos]), rng)
        z_pert = model(perturbed)
        influence_samples[t, di] = float(np.sum((z_orig - z_pert) ** 2))

ecl_point, ci_lo, ci_hi = bootstrap_ecl_ci(
    influence_samples, distances, beta=0.9, n_bootstrap=500, rng=rng
)
print(f"  ECL_0.9 = {ecl_point:.0f} bp  (95% CI: [{ci_lo:.0f}, {ci_hi:.0f}])")

# ---------------------------------------------------------------------------
# Step 6: Context Decay Spectroscopy (Section 5.8)
# ---------------------------------------------------------------------------
print("\nStep 6: Fitting CDS model...")

cds_result = fit_cds(distances, influence, n_components=2, method="nls")
print(f"  Amplitudes: {cds_result['amplitudes']}")
print(f"  Decay rates: {cds_result['decay_rates']}")

s_ecl = spectral_ecl(cds_result["amplitudes"], cds_result["decay_rates"], beta=0.9)
print(f"  Spectral ECL_0.9 = {s_ecl} bp")

# ---------------------------------------------------------------------------
# Step 7: Visualization
# ---------------------------------------------------------------------------
print("\nStep 7: Generating plots...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: Influence profile (log scale)
ax = axes[0, 0]
ax.semilogy(distances, influence, "b-", linewidth=1.5)
ax.semilogy(distances, cds_result["fitted"], "r--", linewidth=1, label="CDS fit")
ax.set_xlabel("Distance from reference (bp)")
ax.set_ylabel("Influence energy I(d; r)")
ax.set_title("A. Influence Profile")
ax.legend()
ax.grid(True, alpha=0.3)

# Panel B: Cumulative influence
ax = axes[0, 1]
_, cumul = cumulative_influence(distances, influence)
total = cumul[-1]
ax.plot(distances, cumul / total, "b-", linewidth=1.5)
for beta in [0.5, 0.8, 0.9, 0.95]:
    ax.axhline(beta, color="gray", linestyle=":", alpha=0.5)
    ecl_val = ECL(distances, influence, beta=beta)
    ax.axvline(ecl_val, color="red", linestyle="--", alpha=0.3)
    ax.annotate(f"ECL_{beta}", (ecl_val, beta), fontsize=8)
ax.set_xlabel("Radius l (bp)")
ax.set_ylabel("Cumulative influence fraction")
ax.set_title("B. Cumulative Influence & ECL")
ax.grid(True, alpha=0.3)

# Panel C: Effective Context Profile
ax = axes[1, 0]
ax.plot(ecp_betas, ecp_values, "b-", linewidth=1.5)
ax.fill_between(ecp_betas, ecp_values, alpha=0.2)
ax.set_xlabel("Beta")
ax.set_ylabel("ECL_beta (bp)")
ax.set_title(f"C. Effective Context Profile (AECP={aecp:.0f})")
ax.grid(True, alpha=0.3)

# Panel D: Normalized influence
ax = axes[1, 1]
norm_infl = normalized_influence(distances, influence)
ax.bar(distances, norm_infl, width=1.0, alpha=0.7, color="steelblue")
ax.set_xlabel("Distance from reference (bp)")
ax.set_ylabel("Normalized influence")
ax.set_title(f"D. Normalized Influence (ECD={ecd:.0f})")
ax.set_xlim(0, 150)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_dir = Path(__file__).resolve().parents[2] / "outputs"
out_dir.mkdir(exist_ok=True)
fig.savefig(out_dir / "quickstart_overview.png", dpi=150, bbox_inches="tight")
fig.savefig(out_dir / "quickstart_overview.pdf", bbox_inches="tight")
plt.close()

print(f"\n  Plots saved to {out_dir}/quickstart_overview.{{png,pdf}}")
print("\nDone! See the tutorial code for details on each step.")
