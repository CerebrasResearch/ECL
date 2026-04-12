# Getting Started

## Installation

### From source (recommended for development)

```bash
git clone <repo-url> ECL
cd ECL
pip install -e ".[dev]"
```

### Core dependencies only

```bash
pip install -e .
```

### With genomic model support

```bash
pip install -e ".[models]"
```

This adds `enformer-pytorch`, `transformers`, and `einops`. Individual model wrappers may require additional packages (see the [Model Guide](model_guide.md)).

### Requirements

- Python 3.10+
- NumPy >= 1.24
- SciPy >= 1.10
- PyTorch >= 2.0
- matplotlib >= 3.7, seaborn >= 0.12
- tqdm >= 4.65

---

## Quickstart

The following example creates a synthetic model, computes an influence profile, estimates ECL, and visualizes the result -- all in about 20 lines of code.

```python
import numpy as np
from ecl.models.base import SyntheticModel
from ecl.influence import compute_influence_profile
from ecl.ecl import ECL, ECP, AECP, ECD

# 1. Create a synthetic model with 50 bp decay length
model = SyntheticModel(seq_length=500, embed_dim=64, decay_length=50.0)

# 2. Generate random DNA sequences
rng = np.random.default_rng(42)
sequences = rng.integers(0, 4, size=(30, 500)).astype(np.int8)
reference = 250  # center position

# 3. Compute binned influence profile (Algorithm 2)
distances, influence = compute_influence_profile(
    model_fn=model,
    sequences=sequences,
    reference=reference,
    max_distance=200,
)

# 4. Compute ECL quantities
ecl_90 = ECL(distances, influence, beta=0.9)
aecp = AECP(distances, influence)
ecd = ECD(distances, influence)

print(f"ECL_0.9 = {ecl_90} bp")
print(f"AECP    = {aecp:.1f} bp")
print(f"ECD     = {ecd:.1f}")
```

---

## Basic Usage Patterns

### Using a different perturbation kernel

```python
from ecl.perturbations import get_perturbation

# Dinucleotide shuffle instead of random substitution
kernel = get_perturbation("shuffle")

distances, influence = compute_influence_profile(
    model_fn=model,
    sequences=sequences,
    reference=reference,
    max_distance=200,
    perturbation=kernel,
)
```

Available kernels: `"substitution"` (default), `"shuffle"`, `"markov"`, `"generative"`.

### Using a different embedding metric

```python
from ecl.metrics import cosine_distance

distances, influence = compute_influence_profile(
    model_fn=model,
    sequences=sequences,
    reference=reference,
    metric=cosine_distance,
)
```

### Computing the Effective Context Profile (ECP)

```python
from ecl.ecl import ECP

betas, ecl_values = ECP(distances, influence)

# Plot ECP
import matplotlib.pyplot as plt
plt.plot(betas, ecl_values)
plt.xlabel("Beta")
plt.ylabel("ECL_beta (bp)")
plt.title("Effective Context Profile")
plt.show()
```

### Bootstrap confidence intervals

```python
from ecl.estimation import bootstrap_ecl_ci

# Collect per-sample influence values (shape: n_sequences x n_distances)
influence_samples = ...  # see tutorials for full example

ecl_point, ci_lo, ci_hi = bootstrap_ecl_ci(
    influence_samples, distances, beta=0.9, n_bootstrap=1000
)
print(f"ECL_0.9 = {ecl_point:.0f} bp  (95% CI: [{ci_lo:.0f}, {ci_hi:.0f}])")
```

### Comparing two models with a permutation test

```python
from ecl.estimation import permutation_test

# ecl_model_a, ecl_model_b: arrays of ECL values at matched loci
mean_diff, p_value, ci_width = permutation_test(ecl_model_a, ecl_model_b)
print(f"Mean difference = {mean_diff:.1f} bp, p = {p_value:.4f}")
```

### Context Decay Spectroscopy

```python
from ecl.cds import fit_cds, select_n_components

# Automatic component selection
best_K, results = select_n_components(distances, influence, max_K=4)
best = results[best_K - 1]
print(f"Best K={best_K}")
for k in range(best_K):
    half_life = 0.693 / best["decay_rates"][k]
    print(f"  Component {k+1}: amplitude={best['amplitudes'][k]:.3f}, "
          f"half-life={half_life:.0f} bp")
```

### Genomic Needle-in-a-Haystack (gNIAH)

```python
from ecl.gniah import gniah_sensitivity

distances_gniah = np.arange(0, 1000, 50)
sensitivity = gniah_sensitivity(
    model_fn=model,
    motif_name="CTCF",
    distances=distances_gniah,
    seq_length=2000,
    n_samples=20,
)
```

---

## Using Real Genomic Models

Replace `SyntheticModel` with any of the provided wrappers:

```python
from ecl.models.enformer import EnformerWrapper

model = EnformerWrapper(device="cuda", head="human")
# model accepts int arrays of length 196,608
```

See the [Model Guide](model_guide.md) for details on each supported model, hardware requirements, and how to write a custom wrapper.

---

## Next Steps

- [API Reference](api.md) -- complete function and class reference.
- [Theory Guide](theory.md) -- mathematical background and definitions.
- [Model Guide](model_guide.md) -- supported models and custom wrappers.
- [Experiments Guide](experiments.md) -- reproduce the paper's figures and tables.
- Jupyter notebooks in `notebooks/` for interactive exploration.
