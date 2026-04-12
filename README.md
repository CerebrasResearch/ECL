# ECL: Effective Context Length

A perturbation-variance framework for estimating and comparing context utilization in sequence models.

## Overview

Modern sequence models accept nominal context windows spanning hundreds of kilobases, yet only a fraction of the input context is functionally utilized. ECL provides a rigorous, estimable, and comparable notion of **effective context length** -- the input span that materially affects a model's embedding at a reference locus.

### Key Quantities

| Quantity | Description |
|----------|-------------|
| **ECL_beta** | Minimum radius capturing beta-fraction of total influence |
| **ECP** | Effective Context Profile: full beta -> ECL curve |
| **ECD** | Effective Context Dimension: entropy-based count of contributing positions |
| **AECP** | Area Under the ECP (mean influence distance) |
| **CDS** | Context Decay Spectroscopy: mixture-of-exponentials decomposition |
| **gNIAH** | Genomic Needle-in-a-Haystack protocol |

### Supported Models

Enformer, Borzoi, HyenaDNA, Caduceus, Evo 2, DNABERT-2 (via model wrappers), plus synthetic models for testing.

## Installation

```bash
# Create environment
bash create_env.sh

# Or install manually
pip install -e ".[all]"
```

## Quick Start

```python
from ecl.models.base import SyntheticModel
from ecl.influence import compute_influence_profile
from ecl.ecl import ECL, ECP, ECD
from ecl.perturbations import RandomSubstitution
import numpy as np

# Create model and sequences
model = SyntheticModel(seq_length=500, embed_dim=64, decay_length=50)
rng = np.random.default_rng(42)
sequences = rng.integers(0, 4, size=(20, 500)).astype(np.int8)

# Compute influence profile
distances, influence = compute_influence_profile(
    model_fn=model, sequences=sequences, reference=250,
    max_distance=200, perturbation=RandomSubstitution(), rng=rng,
)

# Compute ECL
ecl_90 = ECL(distances, influence, beta=0.9)
print(f"ECL_0.9 = {ecl_90} bp")
```

## Project Structure

```
ECL/
├── src/ecl/                    # Core library
│   ├── perturbations.py        # Perturbation kernels (substitution, shuffle, Markov, generative)
│   ├── metrics.py              # Embedding discrepancy metrics
│   ├── influence.py            # Influence energy computation (Algorithms 1-2)
│   ├── ecl.py                  # ECL, ECP, ECD, AECP, directional ECL
│   ├── estimation.py           # Bernstein bounds, bootstrap CI, permutation tests
│   ├── cds.py                  # Context Decay Spectroscopy
│   ├── gniah.py                # Genomic Needle-in-a-Haystack
│   └── models/                 # Model wrappers
│       ├── base.py             # SyntheticModel, LocalModel, AdditiveModel
│       ├── enformer.py         # Enformer wrapper
│       ├── borzoi.py           # Borzoi wrapper
│       ├── hyenadna.py         # HyenaDNA wrapper
│       ├── caduceus.py         # Caduceus wrapper
│       ├── evo2.py             # Evo 2 wrapper
│       └── dnabert2.py         # DNABERT-2 wrapper
├── scripts/                    # One script per figure/table
│   ├── fig01..fig12_*.py       # 12 figure scripts
│   └── tab01..tab07_*.py       # 7 table scripts
├── tests/                      # Comprehensive test suite (106 tests)
├── docs/
│   ├── report/                 # LaTeX paper source
│   ├── tutorials/              # Python tutorials
│   ├── api.md                  # API reference
│   ├── theory.md               # Mathematical background
│   ├── getting_started.md      # Getting started guide
│   ├── model_guide.md          # Model wrapper guide
│   └── experiments.md          # Experiment reproduction guide
├── notebooks/                  # Jupyter tutorial notebooks
├── pyproject.toml
├── Makefile
├── create_env.sh
└── .pre-commit-config.yaml
```

## Running

```bash
# Run tests
make test

# Run linter
make lint

# Generate all figures
make figures

# Generate all tables
make tables

# Run everything
make all-experiments
```

## Citation

```bibtex
@inproceedings{shamssolari2026ecl,
  title={Effective Context Length: A Perturbation--Variance Framework for
         Estimating and Comparing Context Utilization in Sequence Models},
  author={Shams Solari, Omid},
  year={2026}
}
```
