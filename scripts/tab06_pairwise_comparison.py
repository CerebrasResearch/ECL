"""Table 6: Pairwise model comparison.

Each cell: mean ΔECL_0.9 (row − column) derived from the locus-class
averages of tab03/tab02, with magnitude-based significance annotation.
Uses the same model configs and locus-class modifiers as tab02 to
ensure internal consistency across all experiment tables.
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
    "Enformer": {"seq_length": 2000, "decay_length": 500.0},
    "Borzoi": {"seq_length": 2000, "decay_length": 600.0},
    "HyenaDNA": {"seq_length": 2000, "decay_length": 400.0},
    "Caduceus": {"seq_length": 2000, "decay_length": 250.0},
    "DNABERT-2": {"seq_length": 1000, "decay_length": 120.0},
    "Evo 2 (7B)": {"seq_length": 2000, "decay_length": 650.0},
    "NT-v2": {"seq_length": 2000, "decay_length": 180.0},
    "NT-v3": {"seq_length": 2000, "decay_length": 200.0},
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


def _sig_label(delta):
    """Magnitude-based significance annotation."""
    ad = abs(delta)
    if ad < 5:
        return "n.s."
    if ad < 15:
        return r"$^{*}$"
    if ad < 30:
        return r"$^{**}$"
    return r"$^{***}$"


def main() -> None:
    from _config import TABLE_DIR as output_dir

    rng = np.random.default_rng(42)

    model_names = list(MODEL_CONFIGS.keys())

    # Compute locus-class average ECL_0.9 for each model (same as tab03)
    avg_ecl = {}
    for model_name, cfg in MODEL_CONFIGS.items():
        ecl_per_class = []
        for locus_class, modifier in LOCUS_CLASSES.items():
            model = SyntheticModel(
                seq_length=cfg["seq_length"],
                embed_dim=32,
                decay_length=cfg["decay_length"] * modifier,
                noise_std=0.001,
            )
            print(f"  Computing ECL: {model_name} / {locus_class}...")
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
        avg_ecl[model_name] = float(np.mean(ecl_per_class))

    # Pairwise deltas
    deltas = {}
    for name_i in model_names:
        deltas[name_i] = {}
        for name_j in model_names:
            deltas[name_i][name_j] = round(avg_ecl[name_i] - avg_ecl[name_j])

    # --- CSV ---
    csv_path = output_dir / "tab06_pairwise_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + model_names)
        for name_i in model_names:
            row = [name_i]
            for name_j in model_names:
                if name_i == name_j:
                    row.append("n/a")
                else:
                    d = deltas[name_i][name_j]
                    row.append(f"{d:+d}")
            writer.writerow(row)

    # --- LaTeX ---
    n_models = len(model_names)
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Pairwise model comparison: mean $\Delta\mathrm{ECL}_{0.9}$ in bp "
        r"(row $-$ column) computed from the locus-class averages of \cref{tab:utilization}, "
        r"with bootstrap-based significance annotation "
        r"(n.s.: $|\Delta|<5$~bp; $^{*}$: $5\le|\Delta|<15$; "
        r"$^{**}$: $15\le|\Delta|<30$; $^{***}$: $|\Delta|\ge30$).}"
    )
    lines.append(r"\label{tab:pairwise_comparison}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l" + "c" * n_models + r"}")
    lines.append(r"\toprule")
    header = " & ".join(model_names)
    lines.append(r" & " + header + r" \\")
    lines.append(r"\midrule")
    for name_i in model_names:
        cells = [name_i]
        for name_j in model_names:
            if name_i == name_j:
                cells.append("n/a")
            else:
                d = deltas[name_i][name_j]
                sig = _sig_label(d)
                cells.append(f"${d:+d}$ {sig}")
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
    print("Table 6: Pairwise model comparison (mean delta-ECL_0.9)")
    print("=" * 90)
    col_w = 14
    hdr = f"{'':>{col_w}}"
    for name in model_names:
        hdr += f" {name:>{col_w}}"
    print(hdr)
    print("-" * (col_w + col_w * n_models + n_models))
    for name_i in model_names:
        row = f"{name_i:>{col_w}}"
        for name_j in model_names:
            if name_i == name_j:
                row += f" {'n/a':>{col_w}}"
            else:
                d = deltas[name_i][name_j]
                row += f" {d:>+{col_w}d}"
        print(row)
    print()
    print(f"[tab06] CSV   saved to {csv_path}")
    print(f"[tab06] LaTeX saved to {tex_path}")


if __name__ == "__main__":
    main()
