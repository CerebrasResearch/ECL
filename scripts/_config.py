"""Shared output path configuration for all experiment scripts.

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
