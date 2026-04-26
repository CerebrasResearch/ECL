"""Appendix Figure A1: Context Decay Spectroscopy (CDS) analysis.

Appendix F: Fits mixture-of-exponentials model to influence profiles for
multiple models. Reports optimal K (BIC), decay rates, amplitudes, and
spectral ECL. Produces a multi-panel figure showing raw profiles with
fitted CDS overlays and component decomposition.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ecl.cds import select_n_components, spectral_ecl
from ecl.influence import compute_influence_profile
from ecl.models.base import SyntheticModel
from ecl.perturbations import RandomSubstitution

# ---------------------------------------------------------------------------
# To use real genomic models, replace the SyntheticModel entries below with
# real model wrappers (see fig01 for example).
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "Enformer": {"seq_length": 1000, "decay_length": 150.0},
    "Borzoi": {"seq_length": 1000, "decay_length": 200.0},
    "HyenaDNA": {"seq_length": 1000, "decay_length": 80.0},
    "Caduceus": {"seq_length": 1000, "decay_length": 120.0},
    "DNABERT-2": {"seq_length": 1000, "decay_length": 60.0},
    "Evo 2 (7B)": {"seq_length": 1000, "decay_length": 220.0},
    "NT-v2": {"seq_length": 1000, "decay_length": 90.0},
    "NT-v3": {"seq_length": 1000, "decay_length": 100.0},
}

SEQ_LENGTH = 1000
EMBED_DIM = 64
N_SEQUENCES = 10
MAX_DISTANCE = 400
MAX_K = 4
SEED = 42


def main():
    rng = np.random.default_rng(SEED)
    from _config import FIGURE_DIR as output_dir

    perturbation = RandomSubstitution()
    sequences = rng.integers(0, 4, size=(N_SEQUENCES, SEQ_LENGTH), dtype=np.int8)
    reference = SEQ_LENGTH // 2

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    n_models = len(MODEL_CONFIGS)
    n_cols = 4
    n_model_rows = (n_models + n_cols - 1) // n_cols
    n_total_rows = n_model_rows * 2  # influence + BIC per row group
    fig, axes = plt.subplots(
        n_total_rows,
        n_cols,
        figsize=(4 * n_cols, 4 * n_model_rows + 1),
        gridspec_kw={"height_ratios": [2, 1] * n_model_rows},
    )
    if n_total_rows == 2 and n_cols == 1:
        axes = axes.reshape(2, 1)

    cds_summary = []

    for idx, (model_name, cfg) in enumerate(MODEL_CONFIGS.items()):
        print(f"  CDS analysis: {model_name}...")
        model = SyntheticModel(
            seq_length=cfg["seq_length"],
            embed_dim=EMBED_DIM,
            decay_length=cfg["decay_length"],
            reference=reference,
        )

        # Compute influence profile
        distances, influence = compute_influence_profile(
            model_fn=model,
            sequences=sequences,
            reference=reference,
            max_distance=MAX_DISTANCE,
            positions_per_distance=2,
            perturbation=perturbation,
            rng=rng,
            show_progress=False,
        )

        # Select optimal K and fit CDS
        best_K, all_fits = select_n_components(distances, influence, max_K=MAX_K, method="nls")
        best_fit = all_fits[best_K - 1]

        # Spectral ECL
        spec_ecl = spectral_ecl(best_fit["amplitudes"], best_fit["decay_rates"])

        cds_summary.append(
            {
                "model": model_name,
                "best_K": best_K,
                "amplitudes": best_fit["amplitudes"],
                "decay_rates": best_fit["decay_rates"],
                "spectral_ecl": spec_ecl,
                "residual": best_fit["residual"],
                "bic": best_fit["bic"],
            }
        )

        # Top panel: influence profile with CDS fit
        row_group = idx // n_cols
        col = idx % n_cols
        ax_top = axes[row_group * 2, col]
        d_plot = distances[1:]
        ax_top.semilogy(d_plot, influence[1:], "k.", markersize=3, alpha=0.5, label="Data")
        ax_top.semilogy(
            d_plot, best_fit["fitted"][1:], "r-", linewidth=2, label=f"CDS (K={best_K})"
        )

        # Show individual components
        component_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        for k in range(best_K):
            a_k = best_fit["amplitudes"][k]
            lam_k = best_fit["decay_rates"][k]
            component = a_k * np.exp(-lam_k * d_plot)
            ax_top.semilogy(
                d_plot,
                component,
                "--",
                color=component_colors[k % 4],
                linewidth=1,
                alpha=0.7,
                label=f"$a_{k+1}={a_k:.2e}, \\lambda_{k+1}={lam_k:.4f}$",
            )

        ax_top.set_title(
            f"{model_name}\n(K={best_K}, Spectral ECL={spec_ecl} bp)",
            fontsize=10,
            fontweight="bold",
        )
        if col == 0:
            ax_top.set_ylabel(r"$\hat{I}(d; r)$")
        ax_top.legend(fontsize=6, loc="upper right")

        # Bottom panel: BIC vs K
        ax_bot = axes[row_group * 2 + 1, col]
        bics = [f["bic"] for f in all_fits]
        ks = list(range(1, MAX_K + 1))
        ax_bot.bar(ks, bics, color=["#2ca02c" if k == best_K else "#cccccc" for k in ks])
        ax_bot.set_xlabel("K (components)")
        if col == 0:
            ax_bot.set_ylabel("BIC")
        ax_bot.set_xticks(ks)

    fig.suptitle(
        "Context Decay Spectroscopy (CDS) Analysis",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    for ext in ("pdf", "png"):
        out_path = output_dir / f"figA1_cds_analysis.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Print summary table
    print()
    print("=" * 90)
    print("Appendix: CDS Summary")
    print("=" * 90)
    print(f"{'Model':<12} {'K':>3} {'Spectral ECL':>14} {'Decay rates':>30} {'BIC':>12}")
    print("-" * 90)
    for s in cds_summary:
        rates_str = ", ".join([f"{r:.4f}" for r in s["decay_rates"]])
        print(
            f"{s['model']:<12} {s['best_K']:>3} {s['spectral_ecl']:>14} {rates_str:>30} {s['bic']:>12.2f}"
        )

    print(f"\nFigure A1 saved to {output_dir}/figA1_cds_analysis.[pdf|png]")


if __name__ == "__main__":
    main()
