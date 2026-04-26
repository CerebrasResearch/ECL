# CLAUDE.md — ECL Project Guide

## What This Project Is

Implementation of the paper "Effective Context Length: A Perturbation-Variance Framework for Estimating and Comparing Context Utilization in Sequence Models" by Omid Shams Solari. The paper introduces ECL — a formal, statistically rigorous measure of how much input context a sequence model actually uses, focused on genomic language models.

## Quick Commands

```bash
# Run tests (no pip install needed — use PYTHONPATH)
PYTHONPATH=src python -B -m pytest tests/ -v --tb=short

# Lint
ruff check src/ tests/ scripts/

# Run a figure/table script
PYTHONPATH=src python -B scripts/fig01_influence_profiles_promoter.py

# Run all experiments
make figures && make tables
```

## Architecture

### Core Library (`src/ecl/`)

| Module | Purpose | Key paper reference |
|--------|---------|-------------------|
| `perturbations.py` | 4 perturbation kernels (substitution, shuffle, Markov, generative) | Section 3.3 |
| `metrics.py` | Embedding distance functions (Euclidean, cosine, Mahalanobis) | Section 3.5 |
| `influence.py` | Influence energy computation — Algorithms 1, 2, 4 | Section 5.1, 7.1-7.4 |
| `ecl.py` | ECL, ECP, ECD, AECP, directional ECL computation | Section 5, Definitions 4.1-4.5 |
| `estimation.py` | Bernstein bounds, bootstrap CI, permutation tests | Section 8, Algorithms 3, 5 |
| `cds.py` | Context Decay Spectroscopy (mixture-of-exponentials) | Section 5.8 |
| `gniah.py` | Genomic Needle-in-a-Haystack protocol | Section 5.10 |

### Models (`src/ecl/models/`)

- `base.py`: `BaseGenomicModel` ABC + 3 synthetic models for testing:
  - `SyntheticModel` — exponential decay, controllable `decay_length`
  - `LocalModel` — exact finite receptive field
  - `AdditiveModel` — verifies Sobol equivalence (Theorem 6.4)
- `enformer.py`, `borzoi.py`, `hyenadna.py`, `caduceus.py`, `evo2.py`, `dnabert2.py` — real model wrappers (require separate model weights)

### Scripts (`scripts/`)

One script per figure (fig01-fig12) and table (tab01-tab07) from the paper's Section 11. All scripts work out-of-the-box with synthetic models. Each has commented placeholders showing how to swap in real models.

## Conventions

- **No pip install needed for dev**: use `PYTHONPATH=src` or `sys.path.insert(0, "src")` in scripts
- **Integer DNA encoding**: A=0, C=1, G=2, T=3 as `np.int8` arrays
- **Model interface**: any callable `f(sequence: NDArray[int8]) -> NDArray[float64]` works with the library
- **Perturbation interface**: `kernel(sequence, positions, rng) -> perturbed_sequence` — always preserves non-target positions
- **Influence profiles**: tuple of `(distances, influence)` arrays — distances are always `np.arange(D+1)`
- **All randomness**: via `np.random.Generator` (no global state)
- **Tests run with**: `python -B` flag to avoid `__pycache__` writes (NFS disk space is limited)

## NFS Disk Space Warning

The NFS mount at `/net/omids-dev/srv/nfs/omids-data/` has very limited free space. Always use `python -B` to suppress `.pyc` writes. Avoid `pip install -e .` (it writes `.egg-info` to the NFS). Use `PYTHONPATH=src` instead.

## Key Theoretical Properties to Know

These are verified by the test suite:

- **Monotonicity** (Prop 6.1): `ECL_beta` is non-decreasing in beta
- **Isometry invariance** (Prop 6.2): ECL unchanged under orthogonal transforms
- **Locality bound** (Prop 6.3): `ECL <= receptive_field` for exactly local models
- **Sobol equivalence** (Thm 6.4): for additive models, influence = 2 * Var(g_i)
- **Exponential decay bound** (Thm 6.2): `ECL = O(1/lambda * log(1/(1-beta)))`
- **Bernstein concentration** (Thm 8.1): finite-sample guarantees for influence estimates
- **Bootstrap validity** (Alg 3): percentile CI for ECL

## Paper Completion Plan (ICLR 2026)

6-phase plan. Full details: `/cb/home/omids/.claude/plans/radiant-cuddling-umbrella.md`

### Phase 1: Fix Shuffle Bug in SyntheticModel
- [x] Add dinucleotide interaction term to `SyntheticModel.forward()` in `src/ecl/models/base.py`
- [x] Add `_dinuc_projection` to `SyntheticModel.__init__()`
- [x] Fix `DinucleotideShuffle` single-position fallback (was no-op, now substitutes)
- [x] Update fig01, fig02, fig05 to pass `block_width=20` for shuffle perturbation
- [x] Verify: fig01 orange Shuffle line shows nonzero decay; fig05 scatter spreads above y=0

### Phase 2: Fix Generic Labels
- [x] `scripts/fig07_locus_class_violin.py`: "Model A/B" -> "Enformer/Borzoi (synthetic)"
- [x] `scripts/fig09_model_comparison_paired.py`: same

### Phase 3: CDS Fix + Aesthetics
- [x] `src/ecl/cds.py`: tighten lambda upper bound 100->1.0, log-spaced initial guesses
- [x] Remove "Figure N:" prefix from all 14 figure script suptitles
- [x] Add unified `PERTURBATION_COLORS`, `MODEL_COLORS`, `set_paper_style()` to `scripts/_config.py`
- [x] Fix fig02 colors to match fig01 (blue/orange not green/red)
- [ ] Standardize `font_scale=1.2` across all scripts (deferred — minor)

### Phase 4: LaTeX Section 11 Rewrite
- [x] `docs/report/sections/11_experiments.tex`: title -> "Experiments"
- [x] Replace "Proposed Figure N." paragraphs with `\includegraphics`
- [x] Replace inline tables (`---`) with `\input{tables/tabNN_*.tex}`
- [x] Remove "Proposed" / "Expected:" language; add synthetic disclaimer

### Phase 5: Real Model Experiments (GPU required)
- [x] Remove hardcoded HF token (use env var)
- [ ] Add missing experiments to `scripts/run_real_experiments.py` (fig04,09,12; tab05; figA1,A2; tabA1)
- [ ] Run full suite on GPU

### Phase 6: Final Integration
- [x] `make all-experiments` to regenerate all outputs (running)
- [x] `make test` (106 tests pass)
- [ ] `pdflatex` + `bibtex` compile
- [ ] Visual spot-check all figures/tables in compiled PDF

## Testing

106 tests across 8 files. Key test categories:
- `test_perturbations.py` — kernel correctness (locality, range)
- `test_metrics.py` — metric properties (symmetry, identity, range)
- `test_ecl.py` — ECL properties (monotonicity, invariance, bounds)
- `test_estimation.py` — statistical validity (coverage, type-I/II error)
- `test_models.py` — model properties (additivity, locality, determinism)
- `test_cds.py` — CDS fitting (recovery of known parameters)
- `test_gniah.py` — gNIAH protocol correctness
