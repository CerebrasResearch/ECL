# API Reference

Complete reference for every public function and class in the `ecl` library.

---

## `ecl.perturbations` -- Perturbation Kernels

Implements four perturbation types used to probe context sensitivity (Section 3.3 of the paper).

### Classes

| Class | Description |
|---|---|
| `PerturbationKernel` | Abstract base class for all perturbation kernels $\Pi_S(\cdot \mid x)$. |
| `RandomSubstitution` | Replace each nucleotide in $S$ with a uniformly random alternative (3 choices). |
| `DinucleotideShuffle` | Shuffle positions in $S$ preserving dinucleotide frequencies (Altschul--Erickson algorithm). |
| `KmerMarkov(k=3, context_flank=500)` | Resample from a $k$-th order Markov model fitted to flanking context. |
| `GenerativeInfilling(k=5, context_flank=1000)` | Placeholder for MLM-based infilling; falls back to Markov resampling. |

### `PerturbationKernel.perturb`

```python
def perturb(
    self,
    sequence: npt.NDArray,     # int array (L,), values in {0,1,2,3}
    positions: npt.NDArray,    # int array of positions to perturb
    rng: np.random.Generator | None = None,
) -> npt.NDArray:              # perturbed int array (L,)
```

Perturb the specified positions in the sequence while preserving all other coordinates. All kernels are also callable: `kernel(sequence, positions, rng)`.

### `get_perturbation`

```python
def get_perturbation(name: str = "substitution", **kwargs) -> PerturbationKernel
```

Instantiate a perturbation kernel by name. Valid names: `"substitution"`, `"shuffle"`, `"markov"`, `"generative"`.

---

## `ecl.metrics` -- Embedding Discrepancy Metrics

Distance functions $d_Z(z, z')$ for comparing model embeddings (Section 3.5).

### Functions

| Function | Signature | Description |
|---|---|---|
| `squared_euclidean` | `(z, z') -> (...)` | $d_Z(z,z') = \lVert z - z' \rVert_2^2$ |
| `cosine_distance` | `(z, z') -> (...)` | $d_{\cos}(z,z') = 1 - \langle z, z' \rangle / (\lVert z \rVert \lVert z' \rVert)$ |
| `mahalanobis_distance` | `(z, z', precision) -> (...)` | $d_\Sigma(z,z') = (z-z')^T \Sigma^{-1}(z-z')$ |
| `task_weighted_distance` | `(z, z', W) -> (...)` | $d_W(z,z') = (z-z')^T W (z-z')$ |
| `estimate_precision` | `(embeddings, regularization=0.01) -> (d, d)` | Estimate precision matrix from embedding samples. |
| `get_metric` | `(name) -> callable` | Look up metric by name (`"squared_euclidean"`, `"cosine"`). |

All metric functions accept arrays of shape `(..., d)` and return arrays of shape `(...)`.

---

## `ecl.influence` -- Influence Energy Computation

Core Monte Carlo estimators for influence energy (Algorithms 1 and 2 from the paper).

### `compute_single_influence`

```python
def compute_single_influence(
    model_fn: Callable[[npt.NDArray], npt.NDArray],
    sequences: npt.NDArray,    # (n, L) int array
    position: int,
    reference: int,
    perturbation: PerturbationKernel | None = None,
    metric: Callable = squared_euclidean,
    rng: np.random.Generator | None = None,
) -> float
```

Estimate single-position influence energy $I(i; r)$ averaged over `n` sequences. Implements Equation 1.

### `compute_influence_profile`

```python
def compute_influence_profile(
    model_fn: Callable[[npt.NDArray], npt.NDArray],
    sequences: npt.NDArray,    # (n, L)
    reference: int,
    max_distance: int | None = None,
    positions_per_distance: int = 10,
    perturbation: PerturbationKernel | None = None,
    metric: Callable = squared_euclidean,
    rng: np.random.Generator | None = None,
    show_progress: bool = True,
) -> tuple[npt.NDArray, npt.NDArray]
```

Binned influence profile estimator (Algorithm 2). Returns `(distances, influence)` arrays of shape `(D+1,)`.

### `compute_block_influence`

```python
def compute_block_influence(
    model_fn: Callable[[npt.NDArray], npt.NDArray],
    sequences: npt.NDArray,    # (n, L)
    reference: int,
    block_size: int = 128,
    perturbation: PerturbationKernel | None = None,
    metric: Callable = squared_euclidean,
    rng: np.random.Generator | None = None,
    show_progress: bool = True,
) -> tuple[npt.NDArray, npt.NDArray]
```

Block-level influence (Algorithm 4, multi-scale block ECL). Partitions the sequence into blocks of `block_size` bp and estimates $I(B_k; r)$ for each block. Returns `(block_distances, block_influences)`.

### `compute_interaction_influence`

```python
def compute_interaction_influence(
    model_fn: Callable[[npt.NDArray], npt.NDArray],
    sequences: npt.NDArray,    # (n, L)
    pos_i: int,
    pos_j: int,
    perturbation: PerturbationKernel | None = None,
    metric: Callable = squared_euclidean,
    rng: np.random.Generator | None = None,
) -> float
```

Pairwise interaction influence $I_{\text{int}}(i, j; r) = I(\{i,j\}; r) - I(i; r) - I(j; r)$ (Equation 9). Positive values indicate synergy; negative values indicate redundancy.

---

## `ecl.ecl` -- ECL Quantities

Core ECL definitions computed from influence profiles (Section 5).

### `cumulative_influence`

```python
def cumulative_influence(
    distances: npt.NDArray,
    influence: npt.NDArray,
    reference: int | None = None,
    L: int | None = None,
) -> tuple[npt.NDArray, npt.NDArray]
```

Compute $I_{\le l}(r) = \sum_{d=0}^{l} m(d) \cdot I(d; r)$ where $m(d)$ is the multiplicity at distance $d$. Returns `(radii, cumulative)`.

### `normalized_influence`

```python
def normalized_influence(
    distances: npt.NDArray,
    influence: npt.NDArray,
    reference: int | None = None,
    L: int | None = None,
) -> npt.NDArray
```

Normalized influence $\bar{I}(d; r)$ (Equation 2): a probability distribution over distances.

### `ECL`

```python
def ECL(
    distances: npt.NDArray,
    influence: npt.NDArray,
    beta: float = 0.9,
    reference: int | None = None,
    L: int | None = None,
) -> int
```

Perturbation-variance effective context length (Definition 4.1): $\text{ECL}_\beta(r) = \min\{l : I_{\le l}(r) \ge \beta \cdot I_{\text{tot}}(r)\}$.

### `ECP`

```python
def ECP(
    distances: npt.NDArray,
    influence: npt.NDArray,
    betas: npt.NDArray | None = None,
    reference: int | None = None,
    L: int | None = None,
) -> tuple[npt.NDArray, npt.NDArray]
```

Effective Context Profile (Definition 4.2): the map $\beta \mapsto \text{ECL}_\beta(r)$. Returns `(betas, ecl_values)`.

### `AECP`

```python
def AECP(
    distances: npt.NDArray,
    influence: npt.NDArray,
    reference: int | None = None,
    L: int | None = None,
    n_betas: int = 200,
) -> float
```

Area Under the Effective Context Profile (Definition 4.3): $\text{AECP}(r) = \sum_i |i - r| \cdot \bar{I}(i; r)$.

### `ECD`

```python
def ECD(
    distances: npt.NDArray,
    influence: npt.NDArray,
    reference: int | None = None,
    L: int | None = None,
) -> float
```

Effective Context Dimension (Definition 4.5): $\text{ECD}(r) = \exp\bigl(-\sum_i \bar{I}(i; r) \log \bar{I}(i; r)\bigr)$. Measures the effective number of positions contributing to the embedding.

### `directional_ecl`

```python
def directional_ecl(
    distances: npt.NDArray,
    influence_upstream: npt.NDArray,
    influence_downstream: npt.NDArray,
    beta: float = 0.9,
) -> tuple[int, int, float]
```

Directional ECL (Equations 10--11). Returns `(ecl_upstream, ecl_downstream, asymmetry_ratio)`.

---

## `ecl.estimation` -- Statistical Estimation

Statistical tools for reliable ECL inference (Section 8).

### `bernstein_confidence_radius`

```python
def bernstein_confidence_radius(
    n: int, variance: float, bound: float, delta: float = 0.05,
) -> float
```

Bernstein confidence radius (Theorem 8.1, Equation 14): $\varepsilon_n(\delta) = \frac{M \log(2/\delta)}{3n} + \sqrt{\frac{2\sigma^2 \log(2/\delta)}{n}}$.

### `sample_complexity`

```python
def sample_complexity(
    L: int, bound: float, margin: float, delta: float = 0.05,
) -> int
```

Minimum sample size for exact ECL estimation (Corollary 8.3, Equation 15).

### `bootstrap_ecl_ci`

```python
def bootstrap_ecl_ci(
    influence_samples: npt.NDArray,   # (n, D+1)
    distances: npt.NDArray,
    beta: float = 0.9,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    reference: int | None = None,
    L: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]
```

Bootstrap confidence interval for $\text{ECL}_\beta(r)$ (Algorithm 3). Returns `(ecl_point, ci_lower, ci_upper)`.

### `antithetic_estimate`

```python
def antithetic_estimate(
    d_plus: npt.NDArray, d_minus: npt.NDArray,
) -> tuple[float, float]
```

Antithetic variance-reduced influence estimator (Proposition 8.4). Returns `(mean, variance)`.

### `importance_weighted_estimate`

```python
def importance_weighted_estimate(
    influence_values: npt.NDArray,
    proposal_probs: npt.NDArray,
    pilot_probs: npt.NDArray,
) -> npt.NDArray
```

Importance-weighted influence estimator (Proposition 8.5). Returns corrected influence estimates.

### `permutation_test`

```python
def permutation_test(
    ecl_f: npt.NDArray, ecl_g: npt.NDArray,
    n_permutations: int = 10000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]
```

Paired sign-flip permutation test for comparing ECL between two models (Algorithm 5). Tests $H_0: \mathbb{E}[\text{ECL}^f(R) - \text{ECL}^g(R)] = 0$. Returns `(mean_diff, p_value, ci_95_halfwidth)`.

### `asymptotic_ci`

```python
def asymptotic_ci(
    influence_samples: npt.NDArray, alpha: float = 0.05,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]
```

Asymptotic (CLT-based) confidence intervals for the influence profile (Theorem 8.6). Returns `(means, ci_lower, ci_upper)`.

---

## `ecl.cds` -- Context Decay Spectroscopy

Fits the influence profile to a mixture of exponentials to decompose context channels (Section 5.8).

### `fit_cds`

```python
def fit_cds(
    distances: npt.NDArray,
    influence: npt.NDArray,
    n_components: int = 2,
    method: str = "nls",
) -> dict
```

Fit Context Decay Spectroscopy model: $I(d; r) \approx \sum_k a_k \exp(-\lambda_k d)$. Methods: `"nls"` (non-linear least squares) or `"nnls"` (non-negative LS on a $\lambda$ grid). Returns a dict with keys: `"amplitudes"`, `"decay_rates"`, `"fitted"`, `"residual"`, `"bic"`.

### `select_n_components`

```python
def select_n_components(
    distances: npt.NDArray,
    influence: npt.NDArray,
    max_K: int = 4,
    method: str = "nls",
) -> tuple[int, list[dict]]
```

Select optimal number of CDS components by BIC. Returns `(best_K, results_list)`.

### `spectral_ecl`

```python
def spectral_ecl(
    amplitudes: npt.NDArray,
    decay_rates: npt.NDArray,
    beta: float = 0.9,
    max_distance: int = 1_000_000,
) -> int
```

Compute spectral ECL from the CDS mixture using the closed-form cumulative: $\sum_k (a_k / \lambda_k)(1 - e^{-\lambda_k l})$.

---

## `ecl.gniah` -- Genomic Needle-in-a-Haystack

The gNIAH protocol for probing motif-level sensitivity as a function of distance (Section 5.10).

### `gniah_sensitivity`

```python
def gniah_sensitivity(
    model_fn: Callable[[npt.NDArray], npt.NDArray],
    motif_name: str,
    distances: npt.NDArray,
    seq_length: int = 196_608,
    center: int | None = None,
    n_samples: int = 50,
    metric: Callable = squared_euclidean,
    rng: np.random.Generator | None = None,
    show_progress: bool = True,
) -> npt.NDArray
```

Compute gNIAH sensitivity $\text{gNIAH}(d, m) = \mathbb{E}[d_Z(f(X_{\text{neutral}}), f(X_{\text{neutral}}^{+m@d}))]$ for each distance $d$. `motif_name` can be a key in the built-in `MOTIFS` dict or a raw DNA string.

### Helper Functions

| Function | Description |
|---|---|
| `encode_motif(motif, rng=None)` | Encode a motif string (with IUPAC ambiguity codes) to an integer array. |
| `generate_neutral_background(length, gc_content=0.42, rng=None)` | Generate a neutral background sequence with specified GC content. |
| `insert_motif_at_distance(sequence, center, distance, motif, upstream=True)` | Insert an encoded motif at a given distance from center. |

### Built-in Motifs

```python
MOTIFS = {
    "CTCF": "CCGCGNGGNGGCAG",
    "GATA": "AGATAAGG",
    "SP1":  "GGGCGG",
    "TATA": "TATAAAA",
    "CAAT": "CCAAT",
}
```

---

## `ecl.models.base` -- Base Model Interface

### `BaseGenomicModel` (ABC)

```python
class BaseGenomicModel(ABC):
    def forward(self, sequence: npt.NDArray) -> npt.NDArray: ...
    @property
    def nominal_context(self) -> int: ...
    @property
    def embedding_dim(self) -> int: ...
    def token_to_bp(self, token_idx: int) -> int: ...
    def __call__(self, sequence) -> npt.NDArray: ...
```

Abstract interface for embedding-generating sequence models. All model wrappers must implement `forward`, `nominal_context`, and `embedding_dim`.

### `SyntheticModel`

```python
class SyntheticModel(
    seq_length: int = 1000,
    embed_dim: int = 64,
    decay_length: float = 100.0,
    reference: int | None = None,
    noise_std: float = 0.0,
)
```

Synthetic model with controllable exponential influence decay for testing. The embedding is a weighted sum of one-hot inputs: $f(x)_j = \sum_i w(|i-r|) \cdot x_{i,j}$ where $w(d) = \exp(-d / \ell)$.

### `LocalModel`

```python
class LocalModel(
    seq_length: int = 1000,
    embed_dim: int = 64,
    receptive_field: int = 50,
    reference: int | None = None,
)
```

Synthetic model with exact locality (finite receptive field). Only positions within radius $R$ of the reference affect the embedding. Used to verify Proposition 6.3.

### `AdditiveModel`

```python
class AdditiveModel(
    seq_length: int = 200,
    embed_dim: int = 32,
    decay_length: float = 30.0,
    reference: int | None = None,
)
```

Synthetic additive model $f(X) = \sum_i g_i(X_i)$ with per-position random projections scaled by exponential decay. Used to verify Theorem 6.4 (Sobol equivalence).

---

## `ecl.models` -- Genomic Model Wrappers

All wrappers inherit from `BaseGenomicModel` and provide lazy model loading.

### `EnformerWrapper`

```python
class EnformerWrapper(
    device: str = "cpu",
    head: str = "human",
    output_bin: int | None = None,
)
```

Wrapper for Enformer (CNN + Transformer). 196,608 bp input, 5,313 output tracks at 128 bp resolution. Requires `enformer-pytorch`.

### `BorzoiWrapper`

```python
class BorzoiWrapper(
    device: str = "cpu",
    output_bin: int | None = None,
)
```

Wrapper for Borzoi. 524,288 bp input, 7,611 tracks at 32 bp resolution. Use `BorzoiWrapper.from_loaded(model)` with a pre-loaded model.

### `HyenaDNAWrapper`

```python
class HyenaDNAWrapper(
    model_name: str = "LongSafari/hyenadna-large-1m-seqlen-hf",
    device: str = "cpu",
    reference_position: int | None = None,
)
```

Wrapper for HyenaDNA (implicit long convolution). Up to 1 Mb single-nucleotide resolution. Requires `transformers`.

### `CaduceusWrapper`

```python
class CaduceusWrapper(
    model_name: str = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16",
    device: str = "cpu",
    reference_position: int | None = None,
)
```

Wrapper for Caduceus (BiMamba with reverse-complement equivariance). 131,072 bp context. Requires `transformers`, `mamba-ssm`, `caduceus`.

### `Evo2Wrapper`

```python
class Evo2Wrapper(
    model_name: str = "arcinstitute/evo2_40b",
    device: str = "auto",
    reference_position: int | None = None,
)
```

Wrapper for Evo 2 (40B StripedHyena + Transformer). Up to 1 Mb. Autoregressive (causal). Requires multi-GPU (4x A100 80 GB minimum). Use `Evo2Wrapper.from_loaded(model)`.

### `DNABERT2Wrapper`

```python
class DNABERT2Wrapper(
    model_name: str = "zhihan1996/DNABERT-2-117M",
    device: str = "cpu",
    reference_position: int | None = None,
)
```

Wrapper for DNABERT-2 (BERT + BPE + ALiBi). Approximately 3 kb context. Handles non-trivial BPE-to-bp position mapping via `_bp_to_token`. Requires `transformers`.
