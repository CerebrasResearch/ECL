"""Table 3: Context utilization ratio ECL_0.9 / Nominal for each model.

Derives ECL_0.9 as the locus-class average from the same computation
pipeline as tab02 (shared model configs and locus-class modifiers),
ensuring internal consistency across all experiment tables.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import csv

import numpy as np

from ecl.estimation import bootstrap_ecl_ci
from ecl.models.base import SyntheticModel

# Same configs as tab02 for consistency
MODEL_CONFIGS = {
    "Enformer": {"seq_length": 2000, "decay_length": 500.0, "nominal": 196608},
    "Borzoi": {"seq_length": 2000, "decay_length": 600.0, "nominal": 524288},
    "HyenaDNA": {"seq_length": 2000, "decay_length": 400.0, "nominal": 1000000},
    "Caduceus": {"seq_length": 2000, "decay_length": 250.0, "nominal": 131072},
    "DNABERT-2": {"seq_length": 1000, "decay_length": 120.0, "nominal": 3000},
    "Evo 2 (7B)": {"seq_length": 2000, "decay_length": 650.0, "nominal": 1000000},
    "NT-v2": {"seq_length": 2000, "decay_length": 180.0, "nominal": 12282},
    "NT-v3": {"seq_length": 2000, "decay_length": 200.0, "nominal": 12282},
}

LOCUS_CLASSES = {"Promoter": 1.0, "Enhancer": 1.15, "Intronic": 0.7}
N_SAMPLES = 20


def _generate_influence_samples(model, n_samples, rng):
    """Run influence profile for multiple random sequences."""
    L = model.nominal_context
    ref = L // 2
    max_dist = min(L // 2, 500)

    sequences = rng.integers(0, 4, size=(n_samples, L))
    distances = np.arange(max_dist + 1)
    samples = np.zeros((n_samples, max_dist + 1))

    for t in range(n_samples):
        seq = sequences[t]
        z_orig = model(seq)
        for d in range(max_dist + 1):
            positions = []
            if ref - d >= 0:
                positions.append(ref - d)
            if d > 0 and ref + d < L:
                positions.append(ref + d)
            if not positions:
                continue
            total = 0.0
            for pos in positions:
                perturbed = seq.copy()
                perturbed[pos] = (perturbed[pos] + rng.integers(1, 4)) % 4
                z_pert = model(perturbed)
                diff = z_orig - z_pert
                total += float(np.sum(diff * diff))
            samples[t, d] = total / len(positions)

    return distances, samples


def main() -> None:
    from _config import TABLE_DIR as output_dir

    rng = np.random.default_rng(42)

    rows = []
    for model_name, cfg in MODEL_CONFIGS.items():
        ecl_per_class = []
        for _locus_class, modifier in LOCUS_CLASSES.items():
            model = SyntheticModel(
                seq_length=cfg["seq_length"],
                embed_dim=32,
                decay_length=cfg["decay_length"] * modifier,
                noise_std=0.001,
            )
            distances, influence_samples = _generate_influence_samples(model, N_SAMPLES, rng)
            ecl_point, _, _ = bootstrap_ecl_ci(
                influence_samples,
                distances,
                beta=0.9,
                n_bootstrap=200,
                alpha=0.05,
                rng=rng,
            )
            ecl_per_class.append(ecl_point)
        avg_ecl = float(np.mean(ecl_per_class))
        nominal_bp = cfg["nominal"]
        utilization = avg_ecl / nominal_bp * 100.0
        rows.append((model_name, nominal_bp, avg_ecl, utilization))

    # --- CSV ---
    csv_path = output_dir / "tab03_utilization.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Nominal (bp)", "ECL_0.9 (bp)", "Utilization (%)"])
        for model_name, nominal, ecl_val, util in rows:
            writer.writerow([model_name, nominal, f"{ecl_val:.0f}", f"{util:.2f}"])

    # --- LaTeX ---
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Context utilization ratio $\mathrm{ECL}_{0.9} / \text{Nominal}$ for each model. "
        r"The $\mathrm{ECL}_{0.9}$ value reported here is the locus-class average across "
        r"promoter, enhancer, and intronic loci from \cref{tab:ecl_estimates}.}"
    )
    lines.append(r"\label{tab:utilization}")
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\toprule")
    lines.append(r"Model & Nominal (bp) & $\mathrm{ECL}_{0.9}$ (bp) & Utilization (\%) \\")
    lines.append(r"\midrule")
    for model_name, nominal, ecl_val, util in rows:
        lines.append(f"{model_name} & {nominal:,} & {ecl_val:,.0f} & {util:.2f}" + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    tex = "\n".join(lines)

    tex_path = output_dir / "tab03_utilization.tex"
    tex_path.write_text(tex)

    # --- stdout ---
    print("=" * 70)
    print("Table 3: Context utilization ratio ECL_0.9 / Nominal")
    print("=" * 70)
    print(f"{'Model':<12} {'Nominal (bp)':>14} {'ECL_0.9 (bp)':>14} {'Utilization (%)':>16}")
    print("-" * 70)
    for model_name, nominal, ecl_val, util in rows:
        print(f"{model_name:<12} {nominal:>14,} {ecl_val:>14,.0f} {util:>15.2f}%")
    print()
    print(f"[tab03] CSV   saved to {csv_path}")
    print(f"[tab03] LaTeX saved to {tex_path}")


if __name__ == "__main__":
    main()
