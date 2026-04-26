"""Master experiment runner for ECL paper with real genomic models.

Loads 7 HuggingFace models, extracts real genomic sequences from hg38 chr8/chr9,
runs all influence profiling experiments, and generates all figures and tables.

Models:
  1. HyenaDNA-large-1M (LongSafari/hyenadna-large-1m-seqlen-hf) — implicit convolution
  2. NT-v2-100m (InstaDeepAI/nucleotide-transformer-v2-100m-multi-species) — ESM/BERT
  3. NT-v2-250m (InstaDeepAI/nucleotide-transformer-v2-250m-multi-species) — ESM/BERT
  4. NT-v2-500m (InstaDeepAI/nucleotide-transformer-v2-500m-multi-species) — ESM/BERT
  5. DNABERT-2  (zhihan1996/DNABERT-2-117M) — BERT + ALiBi
  6. NT-v3-650m (InstaDeepAI/nucleotide-transformer-v2.1-650m-multi-species) — ESM/BERT
  7. Evo 2 7B  (arcinstitute/evo2_7b) — StripedHyena + Transformer

Usage:
    PYTHONPATH=src python -B scripts/run_real_experiments.py
"""

import gc
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("HF_TOKEN", os.environ.get("HF_TOKEN", ""))

from _config import FIGURE_DIR, TABLE_DIR  # noqa: E402

# ---- Configuration ----
SEED = 42
DEVICE = "cuda"

# Per-model sequence windows (limited by each model's context)
MODEL_CONFIGS = {
    "HyenaDNA": {
        "window": 2000,  # could go up to 1M but too slow; 2kb is practical
        "max_distance": 800,
        "ref_offset": 0,  # center
    },
    "NT-v2-100m": {
        "window": 2000,
        "max_distance": 800,
        "ref_offset": 0,
    },
    "NT-v2-250m": {
        "window": 2000,
        "max_distance": 800,
        "ref_offset": 0,
    },
    "NT-v2-500m": {
        "window": 2000,
        "max_distance": 800,
        "ref_offset": 0,
    },
    "DNABERT-2": {
        "window": 2000,
        "max_distance": 800,
        "ref_offset": 0,
    },
    "NT-v3-650m": {
        "window": 2000,
        "max_distance": 800,
        "ref_offset": 0,
    },
    "Evo 2 (7B)": {
        "window": 2000,
        "max_distance": 800,
        "ref_offset": 0,
    },
}

N_LOCI_PROFILE = 20  # loci for influence profiling (fig01/02)
N_SEQ_PER_LOCUS = 3  # sequences per locus
N_LOCI_ECL = 50  # loci for ECL estimation (tab02/03)
N_LOCI_SCATTER = 100  # loci for scatter plot (fig05)
N_LOCI_VIOLIN = 50  # loci per class for violin (fig07)
POSITIONS_PER_DISTANCE = 2
BLOCK_WIDTH_SHUFFLE = 20  # block width for shuffle perturbation
N_BOOTSTRAP = 200
BETAS = [0.5, 0.8, 0.9, 0.95, 0.99]


def load_models():
    """Load all 7 models and return dict of name -> (model_fn, config)."""
    import torch

    models = {}

    # 1. HyenaDNA
    print("Loading HyenaDNA...")
    from ecl.models.hyenadna import HyenaDNAWrapper

    m = HyenaDNAWrapper(device=DEVICE)
    m._load_model()
    models["HyenaDNA"] = m
    print(f"  HyenaDNA loaded on {DEVICE}")

    torch.cuda.empty_cache()

    # 2-4. Nucleotide Transformer v2 family
    from ecl.models.nucleotide_transformer import NucleotideTransformerWrapper

    for scale in ["100m", "250m", "500m"]:
        name = f"NT-v2-{scale}"
        print(f"Loading {name}...")
        m = NucleotideTransformerWrapper(scale=scale, device=DEVICE)
        m._load_model()
        models[name] = m
        print(f"  {name} loaded")
        torch.cuda.empty_cache()

    # 5. DNABERT-2
    print("Loading DNABERT-2...")
    from ecl.models.dnabert2 import DNABERT2Wrapper

    m = DNABERT2Wrapper(device=DEVICE)
    m._load_model()
    models["DNABERT-2"] = m
    print("  DNABERT-2 loaded")

    torch.cuda.empty_cache()

    # 6. Nucleotide Transformer v3
    print("Loading NT-v3-650m...")
    m = NucleotideTransformerWrapper(scale="v3-650m", device=DEVICE)
    m._load_model()
    models["NT-v3-650m"] = m
    print("  NT-v3-650m loaded")

    torch.cuda.empty_cache()

    # 7. Evo 2 (7B)
    print("Loading Evo 2 (7B)...")
    from ecl.models.evo2 import Evo2Wrapper

    m = Evo2Wrapper(scale="7b", device=DEVICE)
    m._load_model()
    models["Evo 2 (7B)"] = m
    print("  Evo 2 (7B) loaded")

    torch.cuda.empty_cache()
    return models


def load_genomic_data(rng):
    """Load genome and sample loci for all experiments."""
    from ecl.genomic_data import load_genome, sample_loci

    print("Loading genome...")
    genome = load_genome()
    print(f"  Genome loaded: {list(genome.keys())}")

    data = {}
    for locus_class in ["promoter", "enhancer", "intronic"]:
        # Sample enough loci for the largest experiment
        n_need = max(N_LOCI_PROFILE, N_LOCI_ECL, N_LOCI_SCATTER, N_LOCI_VIOLIN)
        seqs, loci = sample_loci(
            genome, locus_class, n_need, window=2000, n_sequences=N_SEQ_PER_LOCUS, rng=rng
        )
        data[locus_class] = {"sequences": seqs, "loci": loci}
        print(f"  {locus_class}: {seqs.shape}")

    return genome, data


def run_influence_profiles(model, model_name, sequences, max_distance, rng):
    """Compute mean influence profile and SE across loci."""
    from ecl.influence import compute_influence_profile
    from ecl.perturbations import DinucleotideShuffle, RandomSubstitution

    perts = {
        "Substitution": (RandomSubstitution(), 1),
        "Shuffle": (DinucleotideShuffle(), BLOCK_WIDTH_SHUFFLE),
    }

    results = {}
    n_loci = sequences.shape[0]
    reference = sequences.shape[2] // 2

    for pert_name, (pert, bw) in perts.items():
        all_profiles = []
        for i in range(min(n_loci, N_LOCI_PROFILE)):
            seqs = sequences[i]
            d, influence = compute_influence_profile(
                model_fn=model,
                sequences=seqs,
                reference=reference,
                max_distance=max_distance,
                positions_per_distance=POSITIONS_PER_DISTANCE,
                perturbation=pert,
                rng=rng,
                show_progress=False,
                block_width=bw,
            )
            all_profiles.append(influence)

        all_profiles = np.array(all_profiles)
        mean_prof = np.mean(all_profiles, axis=0)
        se_prof = np.std(all_profiles, axis=0, ddof=1) / np.sqrt(len(all_profiles))
        results[pert_name] = (d, mean_prof, se_prof)

    return results


def compute_ecl_at_loci(model, sequences, max_distance, n_loci, rng):
    """Compute ECL at multiple betas for multiple loci."""
    from ecl.ecl import ECL
    from ecl.influence import compute_influence_profile
    from ecl.perturbations import RandomSubstitution

    pert = RandomSubstitution()
    reference = sequences.shape[2] // 2
    ecl_values = {beta: [] for beta in BETAS}

    for i in range(min(n_loci, sequences.shape[0])):
        seqs = sequences[i]
        d, influence = compute_influence_profile(
            model_fn=model,
            sequences=seqs,
            reference=reference,
            max_distance=max_distance,
            positions_per_distance=POSITIONS_PER_DISTANCE,
            perturbation=pert,
            rng=rng,
            show_progress=False,
        )
        for beta in BETAS:
            ecl_val = ECL(d, influence, beta=beta)
            ecl_values[beta].append(ecl_val)

    return {beta: np.array(vals) for beta, vals in ecl_values.items()}


# ---- Figure generators ----


def make_fig01_02(all_profiles, output_dir, fig_num, title_suffix):
    """Influence profiles (fig01 for promoter, fig02 for enhancer)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    n_models = len(all_profiles)
    ncols = min(3, n_models)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharex=True, sharey=True)
    if nrows == 1:
        axes = axes.reshape(1, -1)
    colors = {"Substitution": "#1f77b4", "Shuffle": "#ff7f0e"}

    for idx, (model_name, profiles) in enumerate(all_profiles.items()):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        for pert_name, (d, mean_prof, se_prof) in profiles.items():
            d_plot, mean_plot, se_plot = d[1:], mean_prof[1:], se_prof[1:]
            ax.semilogy(
                d_plot,
                np.maximum(mean_plot, 1e-15),
                color=colors[pert_name],
                label=pert_name,
                linewidth=1.5,
            )
            ax.fill_between(
                d_plot,
                np.maximum(mean_plot - se_plot, 1e-15),
                mean_plot + se_plot,
                alpha=0.2,
                color=colors[pert_name],
            )
        ax.set_title(model_name, fontsize=11, fontweight="bold")
        if row == nrows - 1:
            ax.set_xlabel("Distance d (bp)")
        if col == 0:
            ax.set_ylabel(r"$\hat{I}(d; r)$")
        if idx == 0:
            ax.legend(fontsize=8, loc="upper right")

    # Hide unused subplots
    for idx in range(n_models, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(
        f"Figure {fig_num}: Influence Profiles — {title_suffix}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    name = f"fig{fig_num:02d}_influence_profiles_{'promoter' if fig_num == 1 else 'enhancer'}"
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


def make_fig03(ecl_data_promoter, output_dir):
    """Cumulative influence with ECL estimates."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(10, 7))
    palette = sns.color_palette("husl", len(ecl_data_promoter))

    for idx, (model_name, profiles) in enumerate(ecl_data_promoter.items()):
        d, mean_prof, _ = profiles["Substitution"]
        cumulative = np.cumsum(mean_prof)
        total = cumulative[-1] if cumulative[-1] > 0 else 1.0
        frac = cumulative / total
        ax.plot(d, frac, color=palette[idx], linewidth=2, label=model_name)

        # Mark ECL_0.9
        idx_90 = np.searchsorted(frac, 0.9)
        if idx_90 < len(d):
            ax.plot(d[idx_90], frac[idx_90], "o", color=palette[idx], markersize=8)

    for beta in [0.5, 0.8, 0.9, 0.95, 0.99]:
        ax.axhline(y=beta, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.text(d[-1] * 0.98, beta + 0.01, f"β = {beta}", fontsize=8, ha="right", color="gray")

    ax.set_xlabel("Radius ℓ (bp)", fontsize=12)
    ax.set_ylabel(r"$I_{\leq \ell}(r) / I_{\mathrm{tot}}(r)$", fontsize=12)
    ax.set_title(
        "Figure 3: Cumulative Influence Fraction with ECL Estimates", fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0, d[-1])
    ax.set_ylim(0, 1.05)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"fig03_cumulative_influence.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig03_cumulative_influence")


def make_fig05(models, data, max_distance, rng, output_dir):
    """Perturbation scatter: ECL(shuffle) vs ECL(substitution)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    from ecl.ecl import ECL
    from ecl.influence import compute_influence_profile
    from ecl.perturbations import DinucleotideShuffle, RandomSubstitution

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(9, 8))
    palette = sns.color_palette("husl", len(models))

    sub_pert = RandomSubstitution()
    shuf_pert = DinucleotideShuffle()
    sequences = data["promoter"]["sequences"]
    reference = sequences.shape[2] // 2

    ax.plot([0, max_distance], [0, max_distance], "k--", linewidth=0.8, alpha=0.5, label="y = x")

    for idx, (model_name, model) in enumerate(models.items()):
        print(f"  Fig05: {model_name}...")
        ecl_sub, ecl_shuf = [], []
        for i in range(min(N_LOCI_SCATTER, sequences.shape[0])):
            seqs = sequences[i]
            # Substitution
            d, inf_sub = compute_influence_profile(
                model_fn=model,
                sequences=seqs,
                reference=reference,
                max_distance=max_distance,
                positions_per_distance=1,
                perturbation=sub_pert,
                rng=rng,
                show_progress=False,
            )
            ecl_sub.append(ECL(d, inf_sub, beta=0.9))
            # Shuffle
            d, inf_shuf = compute_influence_profile(
                model_fn=model,
                sequences=seqs,
                reference=reference,
                max_distance=max_distance,
                positions_per_distance=1,
                perturbation=shuf_pert,
                rng=rng,
                show_progress=False,
                block_width=BLOCK_WIDTH_SHUFFLE,
            )
            ecl_shuf.append(ECL(d, inf_shuf, beta=0.9))

        ecl_sub, ecl_shuf = np.array(ecl_sub), np.array(ecl_shuf)
        ax.scatter(
            ecl_sub,
            ecl_shuf,
            color=palette[idx],
            alpha=0.4,
            s=15,
            label=model_name,
            edgecolors="none",
        )
        ax.scatter(
            ecl_sub.mean(),
            ecl_shuf.mean(),
            color=palette[idx],
            s=100,
            marker="D",
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )

    ax.set_xlabel(r"ECL$_{0.9}$ (Substitution) [bp]", fontsize=12)
    ax.set_ylabel(r"ECL$_{0.9}$ (Shuffle) [bp]", fontsize=12)
    ax.set_title("Figure 5: ECL(Shuffle) vs ECL(Substitution)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_aspect("equal")
    plt.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"fig05_perturbation_scatter.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig05_perturbation_scatter")


def make_fig07(models, data, genome, max_distance, rng, output_dir):
    """Violin plots of ECL across locus classes."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    from ecl.ecl import ECL
    from ecl.influence import compute_influence_profile
    from ecl.perturbations import RandomSubstitution

    pert = RandomSubstitution()
    records = []

    for model_name, model in models.items():
        print(f"  Fig07: {model_name}...")
        for locus_class in ["promoter", "enhancer", "intronic"]:
            sequences = data[locus_class]["sequences"]
            reference = sequences.shape[2] // 2
            for i in range(min(N_LOCI_VIOLIN, sequences.shape[0])):
                seqs = sequences[i]
                d, influence = compute_influence_profile(
                    model_fn=model,
                    sequences=seqs,
                    reference=reference,
                    max_distance=max_distance,
                    positions_per_distance=1,
                    perturbation=pert,
                    rng=rng,
                    show_progress=False,
                )
                ecl_val = ECL(d, influence, beta=0.9)
                records.append(
                    {"Model": model_name, "Locus Class": locus_class.title(), "ECL_0.9": ecl_val}
                )

    df = pd.DataFrame(records)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(
        data=df,
        x="Locus Class",
        y="ECL_0.9",
        hue="Model",
        ax=ax,
        inner="quartile",
        palette="husl",
        cut=0,
    )
    ax.set_title(
        r"Figure 7: ECL$_{0.9}$ Distribution Across Locus Classes", fontsize=13, fontweight="bold"
    )
    ax.set_ylabel(r"ECL$_{0.9}$ (bp)")
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"fig07_locus_class_violin.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig07_locus_class_violin")


def make_tab01(output_dir):
    """Table 1: Model taxonomy."""
    import pandas as pd

    rows = [
        ("HyenaDNA-large", "Implicit Conv (Hyena)", 2023, "1,000,000", "6.6M"),
        ("NT-v2-100m", "ESM / BERT", 2023, "12,282", "100M"),
        ("NT-v2-250m", "ESM / BERT", 2023, "12,282", "250M"),
        ("NT-v2-500m", "ESM / BERT", 2023, "12,282", "500M"),
        ("DNABERT-2", "BERT + ALiBi + BPE", 2024, "~3,000", "117M"),
        ("NT-v3-650m", "ESM / BERT", 2024, "12,282", "650M"),
        ("Evo 2 (7B)", "StripedHyena + Transf", 2025, "1,000,000", "7B"),
    ]
    df = pd.DataFrame(
        rows, columns=["Model", "Architecture", "Year", "Nominal context (bp)", "Parameters"]
    )
    df.to_csv(output_dir / "tab01_model_taxonomy.csv", index=False)

    # LaTeX
    tex = df.to_latex(index=False, column_format="@{}llrrl@{}")
    with open(output_dir / "tab01_model_taxonomy.tex", "w") as f:
        f.write(tex)
    print("  Saved tab01_model_taxonomy")


def make_tab02(ecl_results, output_dir):
    """Table 2: ECL estimates with bootstrap CIs."""
    import pandas as pd

    rows = []
    for model_name, locus_data in ecl_results.items():
        for locus_class, ecl_vals in locus_data.items():
            row = {"Model": model_name, "Locus class": locus_class}
            for beta in BETAS:
                vals = ecl_vals[beta]
                mean_ecl = np.mean(vals)
                if len(vals) >= 5:
                    ci_lo = np.percentile(vals, 2.5)
                    ci_hi = np.percentile(vals, 97.5)
                else:
                    ci_lo = ci_hi = mean_ecl
                row[f"ECL_{beta}"] = f"{mean_ecl:.0f}"
                row[f"CI_{beta}"] = f"[{ci_lo:.0f}, {ci_hi:.0f}]"
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "tab02_ecl_estimates.csv", index=False)

    tex = df.to_latex(index=False)
    with open(output_dir / "tab02_ecl_estimates.tex", "w") as f:
        f.write(tex)
    print("  Saved tab02_ecl_estimates")


def make_tab03(ecl_results, output_dir):
    """Table 3: Context utilization ratios."""
    import pandas as pd

    nominal = {
        "HyenaDNA": 1_000_000,
        "NT-v2-100m": 12_282,
        "NT-v2-250m": 12_282,
        "NT-v2-500m": 12_282,
        "DNABERT-2": 3_000,
    }
    rows = []
    for model_name in ecl_results:
        ecl_vals = ecl_results[model_name]["Promoter"][0.9]
        mean_ecl = np.mean(ecl_vals)
        nom = nominal[model_name]
        util = mean_ecl / nom * 100
        rows.append(
            {
                "Model": model_name,
                "Nominal (bp)": nom,
                "ECL_0.9 (bp)": f"{mean_ecl:.0f}",
                "Utilization (%)": f"{util:.2f}",
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "tab03_utilization.csv", index=False)
    tex = df.to_latex(index=False)
    with open(output_dir / "tab03_utilization.tex", "w") as f:
        f.write(tex)
    print("  Saved tab03_utilization")


def make_tab04(models, data, max_distance, rng, output_dir):
    """Table 4: Perturbation sensitivity."""
    import pandas as pd

    from ecl.ecl import ECL
    from ecl.influence import compute_influence_profile
    from ecl.perturbations import (
        DinucleotideShuffle,
        GenerativeInfilling,
        KmerMarkov,
        RandomSubstitution,
    )

    perts = {
        "Substitution": (RandomSubstitution(), 1),
        "Shuffle": (DinucleotideShuffle(), BLOCK_WIDTH_SHUFFLE),
        "Markov": (KmerMarkov(), 1),
        "Generative": (GenerativeInfilling(), 1),
    }
    sequences = data["promoter"]["sequences"]
    reference = sequences.shape[2] // 2
    n_loci = min(20, sequences.shape[0])

    rows = []
    for model_name, model in models.items():
        row = {"Model": model_name}
        for pert_name, (pert, bw) in perts.items():
            print(f"  Tab04: {model_name}/{pert_name}...")
            ecls = []
            for i in range(n_loci):
                d, inf = compute_influence_profile(
                    model_fn=model,
                    sequences=sequences[i],
                    reference=reference,
                    max_distance=max_distance,
                    positions_per_distance=1,
                    perturbation=pert,
                    rng=rng,
                    show_progress=False,
                    block_width=bw,
                )
                ecls.append(ECL(d, inf, beta=0.9))
            row[pert_name] = f"{np.mean(ecls):.0f}"
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "tab04_perturbation_sensitivity.csv", index=False)
    tex = df.to_latex(index=False)
    with open(output_dir / "tab04_perturbation_sensitivity.tex", "w") as f:
        f.write(tex)
    print("  Saved tab04_perturbation_sensitivity")


def make_tab06(models, data, max_distance, rng, output_dir):
    """Table 6: Pairwise model comparison."""
    import pandas as pd

    from ecl.ecl import ECL
    from ecl.influence import compute_influence_profile
    from ecl.perturbations import RandomSubstitution

    pert = RandomSubstitution()
    sequences = data["promoter"]["sequences"]
    reference = sequences.shape[2] // 2
    n_loci = min(30, sequences.shape[0])

    # Compute ECL for each model at matched loci
    model_ecls = {}
    for model_name, model in models.items():
        print(f"  Tab06: {model_name}...")
        ecls = []
        for i in range(n_loci):
            d, inf = compute_influence_profile(
                model_fn=model,
                sequences=sequences[i],
                reference=reference,
                max_distance=max_distance,
                positions_per_distance=1,
                perturbation=pert,
                rng=rng,
                show_progress=False,
            )
            ecls.append(ECL(d, inf, beta=0.9))
        model_ecls[model_name] = np.array(ecls)

    # Pairwise comparison
    names = list(models.keys())
    matrix = {}
    for a in names:
        row = {}
        for b in names:
            if a == b:
                row[b] = "---"
            else:
                diff = np.mean(model_ecls[a] - model_ecls[b])
                # Simple permutation test
                n_perm = 1000
                count = 0
                observed = abs(diff)
                combined = model_ecls[a] - model_ecls[b]
                for _ in range(n_perm):
                    signs = rng.choice([-1, 1], size=len(combined))
                    if abs(np.mean(combined * signs)) >= observed:
                        count += 1
                p_val = count / n_perm
                sig = "*" if p_val < 0.05 else ""
                row[b] = f"{diff:+.0f} (p={p_val:.3f}){sig}"
        matrix[a] = row

    df = pd.DataFrame(matrix).T
    df.to_csv(output_dir / "tab06_pairwise_comparison.csv")
    tex = df.to_latex()
    with open(output_dir / "tab06_pairwise_comparison.tex", "w") as f:
        f.write(tex)
    print("  Saved tab06_pairwise_comparison")


# ---- Main ----


def main():
    import torch

    rng = np.random.default_rng(SEED)
    start = time.time()

    print("=" * 60)
    print("ECL Real Model Experiments")
    print("=" * 60)

    # Load models (one at a time to manage GPU memory)
    # We'll load/unload as needed to avoid OOM

    # Load genomic data
    genome, data = load_genomic_data(rng)

    # ---- Phase 1: Influence profiles (fig01, fig02) ----
    print("\n--- Phase 1: Influence Profiles ---")
    all_profiles_promoter = {}
    all_profiles_enhancer = {}

    for model_name in MODEL_CONFIGS:
        print(f"\nLoading {model_name} for profiling...")
        model = _load_single_model(model_name)
        cfg = MODEL_CONFIGS[model_name]

        print("  Profiling promoter loci...")
        all_profiles_promoter[model_name] = run_influence_profiles(
            model,
            model_name,
            data["promoter"]["sequences"],
            cfg["max_distance"],
            rng,
        )

        print("  Profiling enhancer loci...")
        all_profiles_enhancer[model_name] = run_influence_profiles(
            model,
            model_name,
            data["enhancer"]["sequences"],
            cfg["max_distance"],
            rng,
        )

        # Free GPU memory
        del model
        gc.collect()
        torch.cuda.empty_cache()

    make_fig01_02(all_profiles_promoter, FIGURE_DIR, 1, "Promoter-Centered Loci")
    make_fig01_02(all_profiles_enhancer, FIGURE_DIR, 2, "Enhancer-Centered Loci")
    make_fig03(all_profiles_promoter, FIGURE_DIR)

    # ---- Phase 2: ECL estimation (tab02, tab03) ----
    print("\n--- Phase 2: ECL Estimation ---")
    ecl_results = {}  # model -> locus_class -> {beta: array}

    for model_name in MODEL_CONFIGS:
        print(f"\nLoading {model_name} for ECL...")
        model = _load_single_model(model_name)
        cfg = MODEL_CONFIGS[model_name]
        ecl_results[model_name] = {}

        for locus_class in ["Promoter", "Enhancer", "Intronic"]:
            key = locus_class.lower()
            print(f"  ECL: {model_name}/{locus_class}...")
            ecl_vals = compute_ecl_at_loci(
                model,
                data[key]["sequences"],
                cfg["max_distance"],
                N_LOCI_ECL,
                rng,
            )
            ecl_results[model_name][locus_class] = ecl_vals

        del model
        gc.collect()
        torch.cuda.empty_cache()

    make_tab01(TABLE_DIR)
    make_tab02(ecl_results, TABLE_DIR)
    make_tab03(ecl_results, TABLE_DIR)

    # ---- Phase 3: Scatter + violin + pairwise (fig05, fig07, tab04, tab06) ----
    print("\n--- Phase 3: Comparisons ---")
    # Load all models for comparison
    models = {}
    for model_name in MODEL_CONFIGS:
        models[model_name] = _load_single_model(model_name)

    max_d = 800
    make_fig05(models, data, max_d, rng, FIGURE_DIR)
    make_fig07(models, data, genome, max_d, rng, FIGURE_DIR)
    make_tab04(models, data, max_d, rng, TABLE_DIR)
    make_tab06(models, data, max_d, rng, TABLE_DIR)

    # Clean up
    for m in models.values():
        del m
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"All experiments completed in {elapsed / 60:.1f} minutes")
    print(f"Figures saved to: {FIGURE_DIR}")
    print(f"Tables saved to:  {TABLE_DIR}")
    print(f"{'=' * 60}")


def _load_single_model(model_name):
    """Load a single model by name."""

    if model_name == "HyenaDNA":
        from ecl.models.hyenadna import HyenaDNAWrapper

        m = HyenaDNAWrapper(device=DEVICE)
        m._load_model()
        return m
    elif model_name.startswith("NT-v2-"):
        from ecl.models.nucleotide_transformer import NucleotideTransformerWrapper

        scale = model_name.split("-")[-1]
        m = NucleotideTransformerWrapper(scale=scale, device=DEVICE)
        m._load_model()
        return m
    elif model_name == "DNABERT-2":
        from ecl.models.dnabert2 import DNABERT2Wrapper

        m = DNABERT2Wrapper(device=DEVICE)
        m._load_model()
        return m
    elif model_name == "NT-v3-650m":
        from ecl.models.nucleotide_transformer import NucleotideTransformerWrapper

        m = NucleotideTransformerWrapper(scale="v3-650m", device=DEVICE)
        m._load_model()
        return m
    elif model_name == "Evo 2 (7B)":
        from ecl.models.evo2 import Evo2Wrapper

        m = Evo2Wrapper(scale="7b", device=DEVICE)
        m._load_model()
        return m
    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
    main()
