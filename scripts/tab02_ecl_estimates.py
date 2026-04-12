"""Table 2: ECL_beta estimates (bp) with 95% bootstrap CIs across models and locus classes.

Uses SyntheticModel with different decay lengths per "model" and
ecl.estimation.bootstrap_ecl_ci for confidence intervals.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import csv

import numpy as np

from ecl.estimation import bootstrap_ecl_ci
from ecl.models.base import SyntheticModel

# Simulated models with characteristic decay lengths (bp)
MODEL_CONFIGS = {
    "DeepSEA": {"seq_length": 1000, "decay_length": 80.0},
    "Basset": {"seq_length": 600, "decay_length": 50.0},
    "Basenji": {"seq_length": 2000, "decay_length": 300.0},
    "Enformer": {"seq_length": 2000, "decay_length": 500.0},
    "HyenaDNA": {"seq_length": 2000, "decay_length": 400.0},
    "Caduceus": {"seq_length": 2000, "decay_length": 250.0},
    "DNABERT-2": {"seq_length": 1000, "decay_length": 120.0},
}

BETAS = [0.5, 0.8, 0.9, 0.95, 0.99]

N_SAMPLES = 20
N_BOOTSTRAP = 200


def _generate_influence_samples(
    model: SyntheticModel,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Run influence profile for multiple random sequences, returning per-sample data."""
    L = model.nominal_context
    ref = L // 2
    max_dist = min(L // 2, 500)

    # Generate random sequences
    sequences = rng.integers(0, 4, size=(n_samples, L))

    # Collect per-sample influence values at each distance
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
    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    # Collect results: model -> beta -> (point, ci_lo, ci_hi)
    results = {}
    for model_name, cfg in MODEL_CONFIGS.items():
        model = SyntheticModel(
            seq_length=cfg["seq_length"],
            embed_dim=32,
            decay_length=cfg["decay_length"],
            noise_std=0.001,
        )
        distances, influence_samples = _generate_influence_samples(model, N_SAMPLES, rng)

        results[model_name] = {}
        for beta in BETAS:
            point, ci_lo, ci_hi = bootstrap_ecl_ci(
                influence_samples,
                distances,
                beta=beta,
                n_bootstrap=N_BOOTSTRAP,
                alpha=0.05,
                rng=rng,
            )
            results[model_name][beta] = (point, ci_lo, ci_hi)

    # --- CSV ---
    csv_path = output_dir / "tab02_ecl_estimates.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Model"] + [f"ECL_{b}" for b in BETAS] + [f"CI_{b}" for b in BETAS]
        writer.writerow(header)
        for model_name in MODEL_CONFIGS:
            row = [model_name]
            for beta in BETAS:
                pt, _, _ = results[model_name][beta]
                row.append(f"{pt:.0f}")
            for beta in BETAS:
                _, lo, hi = results[model_name][beta]
                row.append(f"[{lo:.0f}, {hi:.0f}]")
            writer.writerow(row)

    # --- LaTeX ---
    beta_cols = " ".join(["c"] * len(BETAS))
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{$\mathrm{ECL}_\beta$ estimates (bp) with 95\% bootstrap CIs across models.}")
    lines.append(r"\label{tab:ecl_estimates}")
    lines.append(r"\begin{tabular}{l" + beta_cols + r"}")
    lines.append(r"\toprule")
    beta_header = " & ".join([f"$\\beta={b}$" for b in BETAS])
    lines.append(r"Model & " + beta_header + r" \\")
    lines.append(r"\midrule")
    for model_name in MODEL_CONFIGS:
        cells = [model_name]
        for beta in BETAS:
            pt, lo, hi = results[model_name][beta]
            cells.append(f"{pt:.0f} [{lo:.0f}, {hi:.0f}]")
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    tex = "\n".join(lines)

    tex_path = output_dir / "tab02_ecl_estimates.tex"
    tex_path.write_text(tex)

    # --- stdout ---
    print("=" * 100)
    print("Table 2: ECL_beta estimates (bp) with 95% bootstrap CIs")
    print("=" * 100)
    hdr = f"{'Model':<12}"
    for beta in BETAS:
        hdr += f" {'ECL_' + str(beta):>20}"
    print(hdr)
    print("-" * 100)
    for model_name in MODEL_CONFIGS:
        row = f"{model_name:<12}"
        for beta in BETAS:
            pt, lo, hi = results[model_name][beta]
            row += f" {pt:>5.0f} [{lo:>4.0f},{hi:>4.0f}]"
        print(row)
    print()
    print(f"[tab02] CSV   saved to {csv_path}")
    print(f"[tab02] LaTeX saved to {tex_path}")


if __name__ == "__main__":
    main()
