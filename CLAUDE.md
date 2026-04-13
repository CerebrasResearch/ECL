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

## Testing

106 tests across 8 files. Key test categories:
- `test_perturbations.py` — kernel correctness (locality, range)
- `test_metrics.py` — metric properties (symmetry, identity, range)
- `test_ecl.py` — ECL properties (monotonicity, invariance, bounds)
- `test_estimation.py` — statistical validity (coverage, type-I/II error)
- `test_models.py` — model properties (additivity, locality, determinism)
- `test_cds.py` — CDS fitting (recovery of known parameters)
- `test_gniah.py` — gNIAH protocol correctness
