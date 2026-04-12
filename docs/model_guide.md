# Model Guide

This guide describes how to use each supported genomic model wrapper and how to write your own.

---

## Overview

All model wrappers implement the `BaseGenomicModel` interface:

```python
class BaseGenomicModel(ABC):
    def forward(self, sequence: npt.NDArray) -> npt.NDArray: ...
    @property
    def nominal_context(self) -> int: ...
    @property
    def embedding_dim(self) -> int: ...
    def token_to_bp(self, token_idx: int) -> int: ...
```

Every wrapper is callable: `model(sequence)` is equivalent to `model.forward(sequence)`. Input sequences are integer-encoded NumPy arrays with values in `{0, 1, 2, 3}` corresponding to `{A, C, G, T}`.

---

## Supported Models

### Enformer

| Property | Value |
|---|---|
| Architecture | CNN + Transformer |
| Nominal context | 196,608 bp |
| Output resolution | 128 bp per bin (896 bins) |
| Embedding dim | 5,313 tracks (human head) |
| Input format | Single-nucleotide |
| Hardware | 1x GPU with >= 16 GB VRAM |

**Installation:**

```bash
pip install enformer-pytorch
```

**Usage:**

```python
from ecl.models.enformer import EnformerWrapper

model = EnformerWrapper(device="cuda", head="human", output_bin=448)
# output_bin=448 is the central bin; set to None for default (center)

import numpy as np
seq = np.random.randint(0, 4, size=196_608).astype(np.int8)
embedding = model(seq)  # shape: (5313,)
```

The model is lazy-loaded on the first call to `forward()`.

---

### Borzoi

| Property | Value |
|---|---|
| Architecture | CNN + Transformer |
| Nominal context | 524,288 bp |
| Output resolution | 32 bp per bin |
| Embedding dim | 7,611 tracks |
| Input format | Single-nucleotide |
| Hardware | 1x GPU with >= 40 GB VRAM |

**Setup:** Borzoi requires manual weight download from the [Calico repository](https://github.com/calico/borzoi).

**Usage:**

```python
from ecl.models.borzoi import BorzoiWrapper

# Load the model yourself first, then wrap it:
# borzoi_model = ...  (load via Borzoi's own API)
model = BorzoiWrapper.from_loaded(borzoi_model, device="cuda")
```

---

### HyenaDNA

| Property | Value |
|---|---|
| Architecture | Hyena operator (implicit long convolution) |
| Nominal context | Up to 1,000,000 bp |
| Output resolution | Single-nucleotide |
| Embedding dim | 256 (large variant) |
| Input format | Character-level DNA |
| Hardware | 1x GPU with >= 24 GB VRAM for 1M context |

**Installation:**

```bash
pip install transformers
```

**Usage:**

```python
from ecl.models.hyenadna import HyenaDNAWrapper

model = HyenaDNAWrapper(
    model_name="LongSafari/hyenadna-large-1m-seqlen-hf",
    device="cuda",
    reference_position=500_000,  # extract embedding at this position
)

import numpy as np
seq = np.random.randint(0, 4, size=1_000_000).astype(np.int8)
embedding = model(seq)  # shape: (256,)
```

---

### Caduceus

| Property | Value |
|---|---|
| Architecture | BiMamba (bidirectional SSM) |
| Nominal context | 131,072 bp |
| Output resolution | Single-nucleotide |
| Embedding dim | 256 |
| Input format | Character-level DNA |
| Hardware | 1x GPU with >= 16 GB VRAM |

**Installation:**

```bash
pip install transformers mamba-ssm caduceus
```

**Usage:**

```python
from ecl.models.caduceus import CaduceusWrapper

model = CaduceusWrapper(device="cuda", reference_position=65_536)

import numpy as np
seq = np.random.randint(0, 4, size=131_072).astype(np.int8)
embedding = model(seq)  # shape: (256,)
```

Caduceus is bidirectional and reverse-complement equivariant, meaning the influence profile is expected to be symmetric around the reference position.

---

### Evo 2

| Property | Value |
|---|---|
| Architecture | StripedHyena + Transformer hybrid |
| Nominal context | Up to 1,000,000 bp |
| Output resolution | Single-nucleotide |
| Embedding dim | 6,144 |
| Input format | Character-level DNA |
| Hardware | **4x A100 80 GB** (minimum for 40B model) |

**Setup:** Evo 2 requires its own multi-GPU inference harness. See the [Arc Institute repository](https://github.com/arcinstitute/evo2).

**Usage:**

```python
from ecl.models.evo2 import Evo2Wrapper

# evo2_model = ...  (load via Evo 2's own API with model parallelism)
model = Evo2Wrapper.from_loaded(evo2_model, device="auto")
```

Because Evo 2 is autoregressive (causal), only positions $i < r$ can influence the embedding at $r$. The influence profile is expected to be one-sided (upstream only). Use `directional_ecl` to analyze this asymmetry.

---

### DNABERT-2

| Property | Value |
|---|---|
| Architecture | BERT + BPE tokenization + ALiBi |
| Nominal context | ~3,000 bp |
| Output resolution | Variable (BPE tokens, ~6 bp each) |
| Embedding dim | 768 |
| Input format | BPE-tokenized DNA string |
| Hardware | 1x GPU with >= 8 GB VRAM |

**Installation:**

```bash
pip install transformers
```

**Usage:**

```python
from ecl.models.dnabert2 import DNABERT2Wrapper

model = DNABERT2Wrapper(device="cuda", reference_position=1500)

import numpy as np
seq = np.random.randint(0, 4, size=3000).astype(np.int8)
embedding = model(seq)  # shape: (768,)
```

**Position mapping note:** DNABERT-2 uses BPE tokenization, which means the mapping from base-pair positions to token indices ($\phi$) is non-trivial. The wrapper handles this internally via `_bp_to_token`. The `token_to_bp` method provides the inverse approximation (~6 bp per token).

---

## Hardware Summary

| Model | Min GPU VRAM | Recommended Setup |
|---|---|---|
| Enformer | 16 GB | 1x V100 / A100 |
| Borzoi | 40 GB | 1x A100 80 GB |
| HyenaDNA (1M) | 24 GB | 1x A100 40 GB |
| Caduceus (131k) | 16 GB | 1x V100 / A100 |
| Evo 2 (40B) | 320 GB total | 4x A100 80 GB |
| DNABERT-2 | 8 GB | 1x T4 / V100 |

For CPU-only usage (small sequences, slow), set `device="cpu"` in any wrapper.

---

## Writing a Custom Wrapper

To integrate a new model, subclass `BaseGenomicModel` and implement three required members:

```python
import numpy as np
import numpy.typing as npt
from ecl.models.base import BaseGenomicModel


class MyModelWrapper(BaseGenomicModel):
    """Wrapper for my custom genomic model."""

    def __init__(self, device: str = "cpu"):
        self._device = device
        self._model = None

    def _load_model(self):
        """Lazy-load model weights."""
        if self._model is not None:
            return
        # import my_model_lib
        # self._model = my_model_lib.load("my-model-name")
        # self._model.to(self._device).eval()

    def forward(self, sequence: npt.NDArray) -> npt.NDArray:
        """Map integer DNA sequence to embedding vector.

        Parameters
        ----------
        sequence : int array of shape (L,), values in {0, 1, 2, 3}.

        Returns
        -------
        embedding : float64 array of shape (d,).
        """
        self._load_model()
        # Convert sequence to model's expected format
        # Run inference
        # Extract embedding at the desired position
        # Return as np.float64 array
        raise NotImplementedError

    @property
    def nominal_context(self) -> int:
        """Nominal input length in base pairs."""
        return 10_000  # your model's context

    @property
    def embedding_dim(self) -> int:
        """Output embedding dimensionality."""
        return 512  # your model's hidden dim

    def token_to_bp(self, token_idx: int) -> int:
        """Map token index to bp coordinate.

        Override if your model uses multi-nucleotide tokens (BPE, k-mer).
        Default is identity (single-nucleotide resolution).
        """
        return token_idx
```

**Key requirements:**

1. `forward` must accept an integer NumPy array of shape `(L,)` with values in `{0, 1, 2, 3}`.
2. `forward` must return a float64 NumPy array. Shape `(d,)` for single-output models, or `(T, d)` for multi-track models.
3. Use lazy loading (`_load_model`) so the wrapper can be instantiated without immediately loading weights.
4. Override `token_to_bp` if the model does not operate at single-nucleotide resolution.

Once implemented, the wrapper works seamlessly with all ECL functions:

```python
model = MyModelWrapper(device="cuda")
distances, influence = compute_influence_profile(
    model_fn=model, sequences=sequences, reference=ref
)
ecl = ECL(distances, influence, beta=0.9)
```
