"""Table 5: Multi-scale block ECL for Borzoi at promoter loci.

Rows: beta in {0.5, 0.9, 0.95}.
Columns: block sizes b in {1, 32, 128, 512} bp.
Uses ecl.influence.compute_block_influence with SyntheticModel.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import csv

import numpy as np

from ecl.ecl import ECL
from ecl.influence import compute_block_influence
from ecl.models.base import SyntheticModel

BETAS = [0.5, 0.9, 0.95]
BLOCK_SIZES = [1, 32, 128, 512]

# Borzoi-like synthetic model with long decay
SEQ_LENGTH = 4096
DECAY_LENGTH = 600.0
N_SAMPLES = 15


def _compute_block_ecl(
    model: SyntheticModel,
    block_size: int,
    beta: float,
    sequences: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """Compute ECL at a given beta using block-level influence."""
    ref = model.nominal_context // 2
    block_distances, block_influences = compute_block_influence(
        model_fn=model,
        sequences=sequences,
        reference=ref,
        block_size=block_size,
        rng=rng,
        show_progress=False,
    )
    return float(ECL(block_distances, block_influences, beta=beta))


def main() -> None:
    from _config import TABLE_DIR as output_dir

    rng = np.random.default_rng(42)

    model = SyntheticModel(
        seq_length=SEQ_LENGTH,
        embed_dim=32,
        decay_length=DECAY_LENGTH,
        noise_std=0.001,
    )

    sequences = rng.integers(0, 4, size=(N_SAMPLES, SEQ_LENGTH))

    # results[beta][block_size] = ecl_value
    results = {}
    for beta in BETAS:
        results[beta] = {}
        for bs in BLOCK_SIZES:
            print(f"  beta={beta}, block_size={bs}...")
            ecl_val = _compute_block_ecl(model, bs, beta, sequences, rng)
            results[beta][bs] = ecl_val

    # --- CSV ---
    csv_path = output_dir / "tab05_multiscale_block.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Beta"] + [f"b={bs}" for bs in BLOCK_SIZES])
        for beta in BETAS:
            row = [str(beta)]
            for bs in BLOCK_SIZES:
                row.append(f"{results[beta][bs]:.0f}")
            writer.writerow(row)

    # --- LaTeX ---
    n_bs = len(BLOCK_SIZES)
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Multi-scale block $\mathrm{ECL}$ for Borzoi at promoter loci. " r"Values in bp.}"
    )
    lines.append(r"\label{tab:multiscale_block}")
    lines.append(r"\begin{tabular}{l" + "r" * n_bs + r"}")
    lines.append(r"\toprule")
    bs_header = " & ".join([f"$b={bs}$" for bs in BLOCK_SIZES])
    lines.append(r"$\beta$ & " + bs_header + r" \\")
    lines.append(r"\midrule")
    for beta in BETAS:
        cells = [f"{beta}"]
        for bs in BLOCK_SIZES:
            cells.append(f"{results[beta][bs]:,.0f}")
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    tex = "\n".join(lines)

    tex_path = output_dir / "tab05_multiscale_block.tex"
    tex_path.write_text(tex)

    # --- stdout ---
    print()
    print("=" * 60)
    print("Table 5: Multi-scale block ECL for Borzoi (promoter loci)")
    print("=" * 60)
    hdr = f"{'Beta':<8}"
    for bs in BLOCK_SIZES:
        hdr += f" {'b=' + str(bs):>10}"
    print(hdr)
    print("-" * 60)
    for beta in BETAS:
        row = f"{beta:<8}"
        for bs in BLOCK_SIZES:
            row += f" {results[beta][bs]:>10,.0f}"
        print(row)
    print()
    print(f"[tab05] CSV   saved to {csv_path}")
    print(f"[tab05] LaTeX saved to {tex_path}")


if __name__ == "__main__":
    main()
