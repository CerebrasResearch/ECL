#!/usr/bin/env python
"""Master runner: generate all results for the ECL paper.

Usage:
    PYTHONPATH=src python -B scripts/run_all.py              # all experiments
    PYTHONPATH=src python -B scripts/run_all.py --figures     # main figures only
    PYTHONPATH=src python -B scripts/run_all.py --tables      # main tables only
    PYTHONPATH=src python -B scripts/run_all.py --appendix    # appendix only
    PYTHONPATH=src python -B scripts/run_all.py --quick       # fast subset for CI
    PYTHONPATH=src python -B scripts/run_all.py fig01 tab02   # specific scripts

Or equivalently via Make:
    make all-experiments    # everything
    make figures            # all main figures
    make tables             # all main tables
    make appendix           # appendix scripts
    make fig01              # individual script
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# -----------------------------------------------------------------------
# Script registry — maps short names to script files
# Ordering matches the paper's Section 11 experiment numbering.
# -----------------------------------------------------------------------
MAIN_FIGURES = [
    ("fig01", "fig01_influence_profiles_promoter.py", "Exp 1a: Influence profiles — promoter"),
    ("fig02", "fig02_influence_profiles_enhancer.py", "Exp 1b: Influence profiles — enhancer"),
    ("fig03", "fig03_cumulative_influence.py", "Exp 2a: Cumulative influence + bootstrap CIs"),
    ("fig04", "fig04_directional_ecl.py", "Exp 3:  Directional ECL"),
    ("fig05", "fig05_perturbation_scatter.py", "Exp 4b: Perturbation scatter (sub vs shuf)"),
    ("fig06", "fig06_trained_vs_random.py", "Exp 5:  Trained vs random weights"),
    ("fig07", "fig07_locus_class_violin.py", "Exp 7a: Locus-class violin plots"),
    ("fig08", "fig08_ecl_vs_gene_length.py", "Exp 7b: ECL vs gene length"),
    ("fig09", "fig09_model_comparison_paired.py", "Exp 8a: Paired model comparison"),
    ("fig10", "fig10_biological_validation.py", "Exp 9a: Biological validation profiles"),
    ("fig11", "fig11_interaction_heatmap.py", "Exp 10a: Interaction heatmap"),
    ("fig12", "fig12_gniah_sensitivity.py", "Exp 10b: gNIAH sensitivity"),
]

MAIN_TABLES = [
    ("tab01", "tab01_model_taxonomy.py", "Table 1: Model taxonomy (static)"),
    ("tab02", "tab02_ecl_estimates.py", "Table 2: ECL estimates + bootstrap CIs"),
    ("tab03", "tab03_utilization.py", "Table 3: Context utilization ratios"),
    ("tab04", "tab04_perturbation_sensitivity.py", "Table 4: Perturbation sensitivity"),
    ("tab05", "tab05_multiscale_block.py", "Table 5: Multi-scale block ECL"),
    ("tab06", "tab06_pairwise_comparison.py", "Table 6: Pairwise model comparison"),
    ("tab07", "tab07_biological_validation.py", "Table 7: Biological validation"),
]

APPENDIX = [
    ("figA1", "figA1_cds_analysis.py", "Appendix: CDS analysis"),
    ("figA2", "figA2_ecd_analysis.py", "Appendix: ECD analysis"),
    ("tabA1", "tabA1_hyperparameter_sensitivity.py", "Appendix: Hyperparameter sensitivity"),
]

# Lightweight subset for quick smoke-testing / CI
QUICK = ["tab01", "fig01", "fig03", "tab02"]

ALL_SCRIPTS = {
    name: (filename, desc) for name, filename, desc in MAIN_FIGURES + MAIN_TABLES + APPENDIX
}


def run_script(name: str, filename: str, desc: str) -> tuple[str, float, bool]:
    """Run a single experiment script. Returns (name, elapsed_sec, success)."""
    script_path = SCRIPT_DIR / filename
    print(f"\n{'='*72}")
    print(f"  [{name}] {desc}")
    print(f"  {script_path}")
    print(f"{'='*72}")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "-B", str(script_path)],
        env={**__import__("os").environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
    )
    elapsed = time.time() - t0
    ok = result.returncode == 0

    status = "OK" if ok else "FAILED"
    print(f"  [{name}] {status} ({elapsed:.1f}s)")
    return name, elapsed, ok


def main():
    parser = argparse.ArgumentParser(
        description="Run ECL paper experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "scripts", nargs="*", help="Specific script names to run (e.g., fig01 tab02)"
    )
    parser.add_argument(
        "--figures", action="store_true", help="Run main figures only (fig01-fig12)"
    )
    parser.add_argument("--tables", action="store_true", help="Run main tables only (tab01-tab07)")
    parser.add_argument("--appendix", action="store_true", help="Run appendix scripts only")
    parser.add_argument("--quick", action="store_true", help="Quick smoke-test subset")
    parser.add_argument("--list", action="store_true", help="List all available scripts and exit")
    args = parser.parse_args()

    if args.list:
        print(f"\n{'Name':<8} {'Description':<50} {'File'}")
        print("-" * 90)
        for name, filename, desc in MAIN_FIGURES + MAIN_TABLES + APPENDIX:
            print(f"{name:<8} {desc:<50} {filename}")
        return

    # Determine which scripts to run
    to_run = []
    if args.scripts:
        for s in args.scripts:
            if s in ALL_SCRIPTS:
                filename, desc = ALL_SCRIPTS[s]
                to_run.append((s, filename, desc))
            else:
                print(f"Unknown script: {s}. Use --list to see available scripts.")
                sys.exit(1)
    elif args.figures:
        to_run = MAIN_FIGURES
    elif args.tables:
        to_run = MAIN_TABLES
    elif args.appendix:
        to_run = APPENDIX
    elif args.quick:
        to_run = [(n, f, d) for n, f, d in MAIN_FIGURES + MAIN_TABLES + APPENDIX if n in QUICK]
    else:
        to_run = MAIN_FIGURES + MAIN_TABLES + APPENDIX

    # Ensure output directories exist (report directories, tracked by git)
    (PROJECT_ROOT / "docs" / "report" / "figures").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "docs" / "report" / "tables").mkdir(parents=True, exist_ok=True)

    # Run scripts
    print(f"\nRunning {len(to_run)} experiment(s)...")
    total_t0 = time.time()
    report = []
    for name, filename, desc in to_run:
        name, elapsed, ok = run_script(name, filename, desc)
        report.append((name, elapsed, ok))

    total_elapsed = time.time() - total_t0

    # Summary report
    print(f"\n\n{'='*72}")
    print("  RESULTS SUMMARY")
    print(f"{'='*72}")
    n_ok = sum(1 for _, _, ok in report if ok)
    n_fail = len(report) - n_ok
    for name, elapsed, ok in report:
        status = "OK" if ok else "FAILED"
        print(f"  [{name}] {status:>6} ({elapsed:>6.1f}s)")
    print(f"\n  Total: {n_ok} passed, {n_fail} failed, {total_elapsed:.1f}s elapsed")
    print(f"  Figures in: {PROJECT_ROOT / 'docs' / 'report' / 'figures'}/")
    print(f"  Tables in:  {PROJECT_ROOT / 'docs' / 'report' / 'tables'}/")

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
