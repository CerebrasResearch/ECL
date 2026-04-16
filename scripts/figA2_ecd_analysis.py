"""Appendix Figure A2: Effective Context Dimension (ECD) analysis.

Appendix F: Computes ECD distributions across models and locus classes.
ECD = exp(-sum_i I_bar(i;r) * log(I_bar(i;r))) measures the effective
number of contributing positions (analogous to perplexity of influence).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ecl.ecl import ECD
from ecl.influence import compute_influence_profile
from ecl.models.base import SyntheticModel
from ecl.perturbations import RandomSubstitution

# ---------------------------------------------------------------------------
# To use real genomic models, replace the SyntheticModel entries below.
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "Enformer": 150.0,
    "Borzoi": 200.0,
    "HyenaDNA": 80.0,
    "Caduceus": 120.0,
    "DNABERT-2": 60.0,
    "Evo-2": 250.0,
}

LOCUS_CLASSES = {
    "Promoter": 1.0,
    "Enhancer": 1.15,
    "CTCF": 0.6,
    "Intronic": 0.7,
    "Intergenic": 0.5,
}

SEQ_LENGTH = 500
EMBED_DIM = 64
N_LOCI = 30  # loci per class
N_SEQUENCES = 5
MAX_DISTANCE = 200
SEED = 42


def main():
    rng = np.random.default_rng(SEED)
    from _config import FIGURE_DIR as output_dir

    perturbation = RandomSubstitution()
    reference = SEQ_LENGTH // 2

    # Compute ECD per model × locus class
    records = []
    for model_name, base_decay in MODEL_CONFIGS.items():
        for locus_class, modifier in LOCUS_CLASSES.items():
            decay = base_decay * modifier
            model = SyntheticModel(
                seq_length=SEQ_LENGTH,
                embed_dim=EMBED_DIM,
                decay_length=decay,
                reference=reference,
            )

            for _ in range(N_LOCI):
                seqs = rng.integers(0, 4, size=(N_SEQUENCES, SEQ_LENGTH), dtype=np.int8)
                distances, influence = compute_influence_profile(
                    model_fn=model,
                    sequences=seqs,
                    reference=reference,
                    max_distance=MAX_DISTANCE,
                    positions_per_distance=2,
                    perturbation=perturbation,
                    rng=rng,
                    show_progress=False,
                )
                ecd_val = ECD(distances, influence)
                records.append(
                    {
                        "model": model_name,
                        "locus_class": locus_class,
                        "ecd": ecd_val,
                    }
                )

            print(f"  ECD: {model_name} / {locus_class} done")

    # Build DataFrame-like structure for plotting
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

    # Figure: ECD distributions grouped by model, colored by locus class
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: boxplot by model (aggregated across locus classes)
    ax = axes[0]
    model_names = list(MODEL_CONFIGS.keys())
    model_ecd_data = {m: [] for m in model_names}
    for r in records:
        model_ecd_data[r["model"]].append(r["ecd"])

    positions = np.arange(len(model_names))
    bp = ax.boxplot(
        [model_ecd_data[m] for m in model_names],
        positions=positions,
        widths=0.6,
        patch_artist=True,
    )
    palette = sns.color_palette("husl", len(model_names))
    for patch, color in zip(bp["boxes"], palette, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(model_names, rotation=30, ha="right")
    ax.set_ylabel("ECD (effective positions)")
    ax.set_title("(A) ECD by Model", fontweight="bold")

    # Panel B: boxplot by locus class (aggregated across models)
    ax = axes[1]
    locus_names = list(LOCUS_CLASSES.keys())
    locus_ecd_data = {lc: [] for lc in locus_names}
    for r in records:
        locus_ecd_data[r["locus_class"]].append(r["ecd"])

    positions = np.arange(len(locus_names))
    bp = ax.boxplot(
        [locus_ecd_data[lc] for lc in locus_names],
        positions=positions,
        widths=0.6,
        patch_artist=True,
    )
    palette2 = sns.color_palette("Set2", len(locus_names))
    for patch, color in zip(bp["boxes"], palette2, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(locus_names, rotation=30, ha="right")
    ax.set_ylabel("ECD (effective positions)")
    ax.set_title("(B) ECD by Locus Class", fontweight="bold")

    fig.suptitle(
        "Appendix: Effective Context Dimension (ECD) Analysis",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    for ext in ("pdf", "png"):
        out_path = output_dir / f"figA2_ecd_analysis.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Print summary
    print()
    print("=" * 70)
    print("Appendix: ECD Summary (mean +/- std)")
    print("=" * 70)
    print(f"{'Model':<12} {'Locus class':<14} {'ECD mean':>10} {'ECD std':>10}")
    print("-" * 70)
    for model_name in model_names:
        for locus_class in locus_names:
            vals = [
                r["ecd"]
                for r in records
                if r["model"] == model_name and r["locus_class"] == locus_class
            ]
            print(
                f"{model_name:<12} {locus_class:<14} {np.mean(vals):>10.1f} {np.std(vals):>10.1f}"
            )

    print(f"\nFigure A2 saved to {output_dir}/figA2_ecd_analysis.[pdf|png]")


if __name__ == "__main__":
    main()
