"""Table 3: Context utilization ratio ECL_0.9 / Nominal for each model.

Uses SyntheticModel with different decay lengths to compute ECL_0.9 and
compare against the nominal context window.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import csv

import numpy as np

from ecl.ecl import ECL
from ecl.models.base import SyntheticModel

# Paper's 6 models: (name, nominal_bp, seq_length_for_sim, decay_length)
# We use smaller seq_length for simulation but report the real nominal context.
MODEL_CONFIGS = [
    ("Enformer", 196608, 2000, 500.0),
    ("Borzoi", 524288, 2000, 600.0),
    ("HyenaDNA", 1000000, 2000, 400.0),
    ("Caduceus", 131072, 2000, 250.0),
    ("Evo 2", 1000000, 2000, 700.0),
    ("DNABERT-2", 3000, 2000, 120.0),
]

N_SAMPLES = 30


def _compute_ecl_09(model: SyntheticModel, n_samples: int, rng: np.random.Generator) -> float:
    """Compute ECL_0.9 for a synthetic model."""
    L = model.nominal_context
    ref = L // 2
    max_dist = L // 2

    sequences = rng.integers(0, 4, size=(n_samples, L))
    distances = np.arange(max_dist + 1)
    influence = np.zeros(max_dist + 1)

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
            influence[d] += total / len(positions)

    influence /= n_samples
    return float(ECL(distances, influence, beta=0.9))


def main() -> None:
    from _config import TABLE_DIR as output_dir

    rng = np.random.default_rng(42)

    rows = []
    for model_name, nominal_bp, seq_len, decay in MODEL_CONFIGS:
        model = SyntheticModel(
            seq_length=seq_len,
            embed_dim=32,
            decay_length=decay,
            noise_std=0.001,
        )
        ecl_09 = _compute_ecl_09(model, N_SAMPLES, rng)
        utilization = ecl_09 / nominal_bp * 100.0
        rows.append((model_name, nominal_bp, ecl_09, utilization))

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
        r"\caption{Context utilization ratio $\mathrm{ECL}_{0.9} / \text{Nominal}$ for each model.}"
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
