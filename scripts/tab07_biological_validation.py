"""Table 7: Biological validation against known long-range interactions.

Derives per-model ECL_0.9 from the locus-class averages (same as tab03)
and compares against known regulatory interaction distances. Under
surrogate calibration, all model ECLs are sub-1 kb, so none detect the
listed long-range interactions; the table serves as a workflow template.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import csv

import numpy as np

from ecl.estimation import bootstrap_ecl_ci
from ecl.models.base import SyntheticModel

# Known long-range regulatory interactions
KNOWN_LOCI = [
    ("SHH/ZRS", 1000),  # kb
    ("MYC enhancer", 1700),
    ("SOX9 desert", 1000),
    ("HBB/LCR", 60),
    ("PAX6/ELP4", 150),
    ("SHH/MACS1", 900),
]

# Models for biological validation (subset of tab02 configs)
MODEL_CONFIGS = {
    "Enformer": {"seq_length": 2000, "decay_length": 500.0},
    "Borzoi": {"seq_length": 2000, "decay_length": 600.0},
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


def main() -> None:
    from _config import TABLE_DIR as output_dir

    rng = np.random.default_rng(42)

    # Compute locus-class average ECL_0.9 for each model (same pipeline as tab03)
    model_ecl_kb = {}
    for model_name, cfg in MODEL_CONFIGS.items():
        ecl_per_class = []
        for locus_class, modifier in LOCUS_CLASSES.items():
            model = SyntheticModel(
                seq_length=cfg["seq_length"],
                embed_dim=32,
                decay_length=cfg["decay_length"] * modifier,
                noise_std=0.001,
            )
            print(f"  Computing ECL_0.9 for {model_name} / {locus_class}...")
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
        model_ecl_kb[model_name] = float(np.mean(ecl_per_class)) / 1000.0  # bp -> kb

    model_names = list(MODEL_CONFIGS.keys())

    # Build table rows
    rows = []
    for locus_name, known_dist_kb in KNOWN_LOCI:
        row = {"Locus": locus_name, "Known dist (kb)": known_dist_kb}
        for mn in model_names:
            ecl_kb = model_ecl_kb[mn]
            detected = "Yes" if ecl_kb >= known_dist_kb * 0.5 else "No"
            row[mn] = ecl_kb
            row[f"{mn}_detected"] = detected
        rows.append(row)

    # --- CSV ---
    csv_path = output_dir / "tab07_biological_validation.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Locus", "Known dist (kb)"]
        for mn in model_names:
            header += [f"ECL_{mn} (kb)", f"Detected_{mn}"]
        writer.writerow(header)
        for row in rows:
            csv_row = [row["Locus"], row["Known dist (kb)"]]
            for mn in model_names:
                csv_row += [f"{row[mn]:.2f}", row[f"{mn}_detected"]]
            writer.writerow(csv_row)

    # --- LaTeX ---
    n_models = len(model_names)
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Biological validation against known long-range interactions. "
        r"``Detected'' (Det?) indicates whether $\mathrm{ECL}_{0.9}$ reaches at least "
        r"$50\%$ of the known interaction distance. Per-model $\mathrm{ECL}_{0.9}$ values "
        r"are taken from \cref{tab:utilization} and converted to kb. Under the surrogate "
        r"calibration of \cref{sec:experiments}, all reported model ECLs are well below "
        r"$1$~kb, so none of the listed long-range interactions are detected; the table "
        r"illustrates the workflow for surrogate inputs and serves as a template for "
        r"real-data deployment, where rows should be backed by explicit experimental "
        r"references.}"
    )
    lines.append(r"\label{tab:biological_validation}")
    lines.append(r"\small")
    col_spec = "lr" + "rc" * n_models
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\toprule")
    model_header_parts = [r"\multicolumn{2}{c}{" + mn + r"}" for mn in model_names]
    lines.append(r"Locus & Known & " + " & ".join(model_header_parts) + r" \\")
    sub_parts = [r"ECL (kb) & Det?"] * n_models
    lines.append(r" & (kb) & " + " & ".join(sub_parts) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        cells = [row["Locus"], str(row["Known dist (kb)"])]
        for mn in model_names:
            cells.append(f"{row[mn]:.2f}")
            cells.append(row[f"{mn}_detected"])
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    tex = "\n".join(lines)

    tex_path = output_dir / "tab07_biological_validation.tex"
    tex_path.write_text(tex)

    # --- stdout ---
    print()
    print("=" * 100)
    print("Table 7: Biological validation against known long-range interactions")
    print("=" * 100)
    hdr = f"{'Locus':<16} {'Known (kb)':>10}"
    for mn in model_names:
        hdr += f" {'ECL_' + mn + ' (kb)':>16} {'Det?':>5}"
    print(hdr)
    print("-" * 100)
    for row in rows:
        line = f"{row['Locus']:<16} {row['Known dist (kb)']:>10}"
        for mn in model_names:
            line += f" {row[mn]:>16.2f} {row[f'{mn}_detected']:>5}"
        print(line)
    print()
    print(f"[tab07] CSV   saved to {csv_path}")
    print(f"[tab07] LaTeX saved to {tex_path}")


if __name__ == "__main__":
    main()
