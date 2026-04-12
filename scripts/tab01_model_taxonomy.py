"""Table 1: Taxonomy of genomic sequence models.

Generates a reference table of model architectures, years, nominal context
lengths, and parameter counts as LaTeX and CSV.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import csv

MODELS = [
    ("DeepSEA", "CNN", 2015, "1,000", "~0.3M"),
    ("Basset", "CNN", 2016, "600", "~2M"),
    ("Basenji", "Dilated CNN", 2018, "131,072", "~4M"),
    ("Basenji2", "Dilated CNN", 2020, "131,072", "~4M"),
    ("Enformer", "CNN+Transformer", 2021, "196,608", "~250M"),
    ("Borzoi", "CNN+Transformer", 2025, "524,288", "~250M"),
    ("HyenaDNA", "Hyena", 2023, "1,000,000", "~6.6M"),
    ("Caduceus", "BiMamba", 2024, "131,072", "~8M"),
    ("Evo 2", "StripedHyena+Transf", 2025, "1,000,000", "40B"),
    ("DNABERT-2", "BERT+ALiBi", 2024, "3,000", "117M"),
]

HEADER = ["Model", "Architecture", "Year", "Nominal ctx (bp)", "Parameters"]


def generate_csv(output_path: Path) -> None:
    """Write table to CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for row in MODELS:
            writer.writerow(row)


def generate_latex(output_path: Path) -> str:
    """Write table to LaTeX and return the string."""
    col_fmt = "llcrr"
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Taxonomy of genomic sequence models.}")
    lines.append(r"\label{tab:model_taxonomy}")
    lines.append(r"\begin{tabular}{" + col_fmt + r"}")
    lines.append(r"\toprule")
    lines.append(" & ".join(HEADER) + r" \\")
    lines.append(r"\midrule")
    for model, arch, year, ctx, params in MODELS:
        lines.append(f"{model} & {arch} & {year} & {ctx} & {params}" + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    tex = "\n".join(lines)
    output_path.write_text(tex)
    return tex


def main() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_csv(output_dir / "tab01_model_taxonomy.csv")
    tex = generate_latex(output_dir / "tab01_model_taxonomy.tex")

    # Print to stdout
    print("=" * 80)
    print("Table 1: Taxonomy of genomic sequence models")
    print("=" * 80)
    header_fmt = f"{'Model':<12} {'Architecture':<22} {'Year':<6} {'Nominal ctx (bp)':<18} {'Parameters':<12}"
    print(header_fmt)
    print("-" * 80)
    for model, arch, year, ctx, params in MODELS:
        print(f"{model:<12} {arch:<22} {year:<6} {ctx:<18} {params:<12}")
    print()
    print("LaTeX output:")
    print(tex)
    print()
    print(f"[tab01] CSV  saved to {output_dir / 'tab01_model_taxonomy.csv'}")
    print(f"[tab01] LaTeX saved to {output_dir / 'tab01_model_taxonomy.tex'}")


if __name__ == "__main__":
    main()
