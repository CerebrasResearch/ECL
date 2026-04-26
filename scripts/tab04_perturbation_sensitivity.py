"""Table 4: ECL_0.9 (bp) under different perturbation types for each model.

Paper Section 11.4: 4 models (Enformer, Borzoi, HyenaDNA, Caduceus) × 4
perturbation types. Uses ecl.perturbations with SyntheticModel surrogates.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import csv

import numpy as np

from ecl.ecl import ECL
from ecl.models.base import SyntheticModel
from ecl.perturbations import get_perturbation

# Paper's 4 models for perturbation sensitivity comparison
MODEL_CONFIGS = {
    "Enformer": {"seq_length": 2000, "decay_length": 500.0},
    "Borzoi": {"seq_length": 2000, "decay_length": 600.0},
    "HyenaDNA": {"seq_length": 2000, "decay_length": 400.0},
    "Caduceus": {"seq_length": 2000, "decay_length": 250.0},
    "DNABERT-2": {"seq_length": 1000, "decay_length": 120.0},
    "Evo 2 (7B)": {"seq_length": 2000, "decay_length": 650.0},
    "NT-v2": {"seq_length": 2000, "decay_length": 180.0},
    "NT-v3": {"seq_length": 2000, "decay_length": 200.0},
}

PERTURBATION_TYPES = ["substitution", "shuffle", "markov", "generative"]
N_SAMPLES = 20


def _compute_ecl_with_perturbation(
    model: SyntheticModel,
    perturbation_name: str,
    n_samples: int,
    rng: np.random.Generator,
) -> float:
    """Compute ECL_0.9 using a specific perturbation kernel."""
    kernel = get_perturbation(perturbation_name)
    L = model.nominal_context
    ref = L // 2
    max_dist = min(L // 2, 500)

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
                perturbed = kernel(seq, np.array([pos]), rng)
                z_pert = model(perturbed)
                diff = z_orig - z_pert
                total += float(np.sum(diff * diff))
            influence[d] += total / len(positions)

    influence /= n_samples
    return float(ECL(distances, influence, beta=0.9))


def main() -> None:
    from _config import TABLE_DIR as output_dir

    rng = np.random.default_rng(42)

    # Collect results: model -> perturbation_type -> ecl_value
    results = {}
    for model_name, cfg in MODEL_CONFIGS.items():
        print(f"  Computing {model_name}...")
        model = SyntheticModel(
            seq_length=cfg["seq_length"],
            embed_dim=32,
            decay_length=cfg["decay_length"],
            noise_std=0.001,
        )
        results[model_name] = {}
        for ptype in PERTURBATION_TYPES:
            ecl_val = _compute_ecl_with_perturbation(model, ptype, N_SAMPLES, rng)
            results[model_name][ptype] = ecl_val

    # --- CSV ---
    csv_path = output_dir / "tab04_perturbation_sensitivity.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model"] + [p.capitalize() for p in PERTURBATION_TYPES])
        for model_name in MODEL_CONFIGS:
            row = [model_name]
            for ptype in PERTURBATION_TYPES:
                row.append(f"{results[model_name][ptype]:.0f}")
            writer.writerow(row)

    # --- LaTeX ---
    n_pert = len(PERTURBATION_TYPES)
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{$\mathrm{ECL}_{0.9}$ (bp) under different perturbation types for each model.}"
    )
    lines.append(r"\label{tab:perturbation_sensitivity}")
    lines.append(r"\begin{tabular}{l" + "r" * n_pert + r"}")
    lines.append(r"\toprule")
    pert_header = " & ".join([p.capitalize() for p in PERTURBATION_TYPES])
    lines.append(r"Model & " + pert_header + r" \\")
    lines.append(r"\midrule")
    for model_name in MODEL_CONFIGS:
        cells = [model_name]
        for ptype in PERTURBATION_TYPES:
            cells.append(f"{results[model_name][ptype]:,.0f}")
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    tex = "\n".join(lines)

    tex_path = output_dir / "tab04_perturbation_sensitivity.tex"
    tex_path.write_text(tex)

    # --- stdout ---
    print()
    print("=" * 80)
    print("Table 4: ECL_0.9 (bp) under different perturbation types")
    print("=" * 80)
    hdr = f"{'Model':<12}"
    for ptype in PERTURBATION_TYPES:
        hdr += f" {ptype.capitalize():>14}"
    print(hdr)
    print("-" * 80)
    for model_name in MODEL_CONFIGS:
        row = f"{model_name:<12}"
        for ptype in PERTURBATION_TYPES:
            row += f" {results[model_name][ptype]:>14,.0f}"
        print(row)
    print()
    print(f"[tab04] CSV   saved to {csv_path}")
    print(f"[tab04] LaTeX saved to {tex_path}")


if __name__ == "__main__":
    main()
