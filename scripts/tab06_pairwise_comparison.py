"""Table 6: Pairwise model comparison.

Each cell: mean delta-ECL_0.9 (row - column) with permutation test p-value.
Uses ecl.estimation.permutation_test with SyntheticModel.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import csv

import numpy as np

from ecl.ecl import ECL
from ecl.estimation import permutation_test
from ecl.models.base import SyntheticModel

# Paper's 4 models for pairwise comparison (Section 11.8)
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

N_LOCI = 30  # number of independent loci (sequences) per model
N_PERMUTATIONS = 2000


def _compute_ecl_per_locus(
    model: SyntheticModel,
    n_loci: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute ECL_0.9 for each locus independently."""
    L = model.nominal_context
    ref = L // 2
    max_dist = min(L // 2, 500)
    distances = np.arange(max_dist + 1)

    ecl_values = np.zeros(n_loci)
    for locus in range(n_loci):
        seq = rng.integers(0, 4, size=L)
        z_orig = model(seq)
        influence = np.zeros(max_dist + 1)
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
            influence[d] = total / len(positions)
        ecl_values[locus] = float(ECL(distances, influence, beta=0.9))
    return ecl_values


def main() -> None:
    from _config import TABLE_DIR as output_dir

    rng = np.random.default_rng(42)

    model_names = list(MODEL_CONFIGS.keys())
    n_models = len(model_names)

    # Compute per-locus ECL_0.9 for each model
    ecl_per_model = {}
    for model_name, cfg in MODEL_CONFIGS.items():
        print(f"  Computing ECL per locus for {model_name}...")
        model = SyntheticModel(
            seq_length=cfg["seq_length"],
            embed_dim=32,
            decay_length=cfg["decay_length"],
            noise_std=0.001,
        )
        ecl_per_model[model_name] = _compute_ecl_per_locus(model, N_LOCI, rng)

    # Pairwise comparisons
    # results[i][j] = (mean_diff, p_value)
    results = {}
    for i, name_i in enumerate(model_names):
        results[name_i] = {}
        for j, name_j in enumerate(model_names):
            if i == j:
                results[name_i][name_j] = (0.0, 1.0)
            else:
                mean_diff, p_val, _ = permutation_test(
                    ecl_per_model[name_i],
                    ecl_per_model[name_j],
                    n_permutations=N_PERMUTATIONS,
                    rng=rng,
                )
                results[name_i][name_j] = (mean_diff, p_val)

    # --- CSV ---
    csv_path = output_dir / "tab06_pairwise_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + model_names)
        for name_i in model_names:
            row = [name_i]
            for name_j in model_names:
                md, pv = results[name_i][name_j]
                if name_i == name_j:
                    row.append("---")
                else:
                    row.append(f"{md:+.0f} (p={pv:.3f})")
            writer.writerow(row)

    # --- LaTeX ---
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Pairwise model comparison: mean $\Delta\mathrm{ECL}_{0.9}$ "
        r"(row $-$ column) with permutation test $p$-value.}"
    )
    lines.append(r"\label{tab:pairwise_comparison}")
    lines.append(r"\begin{tabular}{l" + "c" * n_models + r"}")
    lines.append(r"\toprule")
    header = " & ".join(model_names)
    lines.append(r" & " + header + r" \\")
    lines.append(r"\midrule")
    for name_i in model_names:
        cells = [name_i]
        for name_j in model_names:
            md, pv = results[name_i][name_j]
            if name_i == name_j:
                cells.append("---")
            else:
                sig = "*" if pv < 0.05 else ""
                cells.append(f"{md:+.0f} ({pv:.3f}){sig}")
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    tex = "\n".join(lines)

    tex_path = output_dir / "tab06_pairwise_comparison.tex"
    tex_path.write_text(tex)

    # --- stdout ---
    print()
    print("=" * 90)
    print("Table 6: Pairwise model comparison (mean delta-ECL_0.9, p-value)")
    print("=" * 90)
    col_w = 18
    hdr = f"{'':>{col_w}}"
    for name in model_names:
        hdr += f" {name:>{col_w}}"
    print(hdr)
    print("-" * 90)
    for name_i in model_names:
        row = f"{name_i:>{col_w}}"
        for name_j in model_names:
            md, pv = results[name_i][name_j]
            if name_i == name_j:
                row += f" {'---':>{col_w}}"
            else:
                cell = f"{md:+.0f} (p={pv:.3f})"
                row += f" {cell:>{col_w}}"
        print(row)
    print()
    print(f"[tab06] CSV   saved to {csv_path}")
    print(f"[tab06] LaTeX saved to {tex_path}")


if __name__ == "__main__":
    main()
