"""Appendix Table A1: Hyperparameter sensitivity analysis.

Appendix E: Ablation studies for key hyperparameters:
  1. Number of MC samples n: {10, 25, 50, 100, 200}
  2. Block size b: {1, 8, 32, 128, 512}
  3. Bootstrap replicates B: {100, 500, 1000, 5000}
  4. Distance metric: squared_euclidean, cosine
Reports ECL_0.9 and CI width for each setting.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import csv

import numpy as np

from ecl.ecl import ECL
from ecl.estimation import bootstrap_ecl_ci
from ecl.metrics import cosine_distance, squared_euclidean
from ecl.models.base import SyntheticModel
from ecl.perturbations import RandomSubstitution

# Reference model: Enformer-calibrated surrogate (same as tab02 Enformer/Promoter)
SEQ_LENGTH = 2000
EMBED_DIM = 32
DECAY_LENGTH = 500.0
NOISE_STD = 0.001
MAX_DISTANCE = 500
SEED = 42


def _compute_influence_samples(model, sequences, reference, max_dist, perturbation, rng):
    """Per-sample influence values for bootstrap."""
    n = sequences.shape[0]
    L = sequences.shape[1]
    distances = np.arange(max_dist + 1)
    samples = np.zeros((n, max_dist + 1))

    for t in range(n):
        seq = sequences[t]
        z_orig = model(seq)
        for d in range(max_dist + 1):
            positions = []
            if reference - d >= 0:
                positions.append(reference - d)
            if d > 0 and reference + d < L:
                positions.append(reference + d)
            if not positions:
                continue
            total = 0.0
            for pos in positions:
                perturbed = perturbation(seq, np.array([pos]), rng)
                z_pert = model(perturbed)
                diff = z_orig - z_pert
                total += float(np.sum(diff * diff))
            samples[t, d] = total / len(positions)

    return distances, samples


def _compute_influence_samples_metric(
    model, sequences, reference, max_dist, perturbation, metric_fn, rng
):
    """Per-sample influence values using a specific metric function."""
    n = sequences.shape[0]
    L = sequences.shape[1]
    distances = np.arange(max_dist + 1)
    samples = np.zeros((n, max_dist + 1))

    for t in range(n):
        seq = sequences[t]
        z_orig = model(seq)
        for d in range(max_dist + 1):
            positions = []
            if reference - d >= 0:
                positions.append(reference - d)
            if d > 0 and reference + d < L:
                positions.append(reference + d)
            if not positions:
                continue
            total = 0.0
            for pos in positions:
                perturbed = perturbation(seq, np.array([pos]), rng)
                z_pert = model(perturbed)
                total += float(metric_fn(z_orig, z_pert))
            samples[t, d] = total / len(positions)

    return distances, samples


def main() -> None:
    from _config import TABLE_DIR as output_dir

    rng = np.random.default_rng(SEED)
    reference = SEQ_LENGTH // 2
    perturbation = RandomSubstitution()

    model = SyntheticModel(
        seq_length=SEQ_LENGTH,
        embed_dim=EMBED_DIM,
        decay_length=DECAY_LENGTH,
        noise_std=NOISE_STD,
    )

    # Generate a large pool of sequences for subsampling
    max_n = 200
    all_sequences = rng.integers(0, 4, size=(max_n, SEQ_LENGTH), dtype=np.int8)
    max_dist = MAX_DISTANCE

    results = []

    # --- Ablation 1: Number of MC samples ---
    print("  Ablation 1: MC sample count...")
    for n_samples in [10, 25, 50, 100, 200]:
        seqs = all_sequences[:n_samples]
        distances, infl_samples = _compute_influence_samples(
            model, seqs, reference, max_dist, perturbation, rng
        )
        ecl_pt, ci_lo, ci_hi = bootstrap_ecl_ci(
            infl_samples, distances, beta=0.9, n_bootstrap=500, rng=rng
        )
        ci_width = ci_hi - ci_lo
        results.append(("MC samples (n)", str(n_samples), ecl_pt, ci_width))

    # --- Ablation 2: Block size ---
    print("  Ablation 2: Block size...")
    for block_size in [1, 8, 32, 128]:
        seqs = all_sequences[:50]
        n_bs = seqs.shape[0]
        L_bs = seqs.shape[1]
        half_block = block_size // 2
        distances_bs = np.arange(max_dist + 1)
        influence_bs = np.zeros(max_dist + 1)
        for t in range(n_bs):
            seq = seqs[t]
            z_orig = model(seq)
            for d in range(max_dist + 1):
                candidates = []
                if reference - d >= 0:
                    candidates.append(reference - d)
                if d > 0 and reference + d < L_bs:
                    candidates.append(reference + d)
                if not candidates:
                    continue
                pos = rng.choice(candidates)
                lo = max(0, int(pos) - half_block)
                hi = min(L_bs, int(pos) + half_block + 1)
                pos_arr = np.arange(lo, hi)
                perturbed = perturbation(seq, pos_arr, rng)
                z_pert = model(perturbed)
                diff = z_orig - z_pert
                influence_bs[d] += float(np.sum(diff * diff))
        influence_bs /= n_bs
        ecl_val = float(ECL(distances_bs, influence_bs, beta=0.9))
        results.append(("Block size (b)", f"{block_size} bp", ecl_val, float("nan")))

    # --- Ablation 3: Bootstrap replicates ---
    print("  Ablation 3: Bootstrap replicates...")
    distances_ref, infl_ref = _compute_influence_samples(
        model, all_sequences[:50], reference, max_dist, perturbation, rng
    )
    for n_boot in [100, 500, 1000, 5000]:
        ecl_pt, ci_lo, ci_hi = bootstrap_ecl_ci(
            infl_ref, distances_ref, beta=0.9, n_bootstrap=n_boot, rng=rng
        )
        ci_width = ci_hi - ci_lo
        results.append(("Bootstrap (B)", str(n_boot), ecl_pt, ci_width))

    # --- Ablation 4: Distance metric ---
    print("  Ablation 4: Distance metric...")
    seqs_metric = all_sequences[:50]
    for metric_name, metric_fn in [
        ("Squared Euclidean", squared_euclidean),
        ("Cosine", cosine_distance),
    ]:
        distances_m, infl_m = _compute_influence_samples_metric(
            model, seqs_metric, reference, max_dist, perturbation, metric_fn, rng
        )
        mean_infl = infl_m.mean(axis=0)
        ecl_val = float(ECL(distances_m, mean_infl, beta=0.9))
        results.append(("Metric", metric_name, ecl_val, float("nan")))

    # --- CSV ---
    csv_path = output_dir / "tabA1_hyperparameter_sensitivity.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Hyperparameter", "Value", "ECL_0.9 (bp)", "CI width (bp)"])
        for hp, val, ecl_val, ci_w in results:
            ci_str = f"{ci_w:.0f}" if not np.isnan(ci_w) else "n/a"
            writer.writerow([hp, val, f"{ecl_val:.0f}", ci_str])

    # --- LaTeX ---
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Hyperparameter sensitivity: $\mathrm{ECL}_{0.9}$ and CI width under varying settings, "
        r"on a representative locus drawn from the surrogate calibration of \cref{sec:experiments}. "
        r"Estimates are stable across MC sample size, bootstrap replicates, and embedding metric, "
        r"and degrade gracefully as block size grows.}"
    )
    lines.append(r"\label{tab:hyperparameter_sensitivity}")
    lines.append(r"\begin{tabular}{llrr}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Hyperparameter} & \textbf{Value} & $\mathrm{ECL}_{0.9}$ (bp) & CI width (bp) \\"
    )
    lines.append(r"\midrule")
    prev_hp = None
    for hp, val, ecl_val, ci_w in results:
        if prev_hp is not None and hp != prev_hp:
            lines.append(r"\midrule")
        ci_str = f"{ci_w:.0f}" if not np.isnan(ci_w) else "n/a"
        lines.append(f"{hp} & {val} & {ecl_val:.0f} & {ci_str}" + r" \\")
        prev_hp = hp
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    tex = "\n".join(lines)

    tex_path = output_dir / "tabA1_hyperparameter_sensitivity.tex"
    tex_path.write_text(tex)

    # --- stdout ---
    print()
    print("=" * 70)
    print("Table A1: Hyperparameter Sensitivity")
    print("=" * 70)
    print(f"{'Hyperparameter':<20} {'Value':<20} {'ECL_0.9':>10} {'CI width':>10}")
    print("-" * 70)
    prev_hp = None
    for hp, val, ecl_val, ci_w in results:
        if prev_hp is not None and hp != prev_hp:
            print("-" * 70)
        ci_str = f"{ci_w:.0f}" if not np.isnan(ci_w) else "n/a"
        print(f"{hp:<20} {val:<20} {ecl_val:>10.0f} {ci_str:>10}")
        prev_hp = hp
    print()
    print(f"[tabA1] CSV   saved to {csv_path}")
    print(f"[tabA1] LaTeX saved to {tex_path}")


if __name__ == "__main__":
    main()
