"""Shared output path configuration and style for all experiment scripts.

Figure scripts write PDFs/PNGs to docs/report/figures/.
Table scripts write TEX/CSVs to docs/report/tables/.
These directories are tracked by git (not gitignored).
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "docs" / "report" / "figures"
TABLE_DIR = PROJECT_ROOT / "docs" / "report" / "tables"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# Unified color scheme for the paper
PERTURBATION_COLORS = {
    "Substitution": "#1f77b4",  # blue
    "Shuffle": "#ff7f0e",  # orange
    "Markov": "#2ca02c",  # green
    "Generative": "#9467bd",  # purple
}

MODEL_COLORS = {
    "Enformer": "#e41a1c",
    "Borzoi": "#ff7f00",
    "HyenaDNA": "#4daf4a",
    "Caduceus": "#377eb8",
    "DNABERT-2": "#984ea3",
    "Evo 2 (7B)": "#a65628",
    "NT-v2": "#f781bf",
    "NT-v3": "#17becf",
}


def set_paper_style():
    """Set consistent seaborn/matplotlib style for all paper figures."""
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
