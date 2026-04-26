"""Base model interface and synthetic test models (Appendix G of the paper)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class BaseGenomicModel(ABC):
    """Abstract interface for embedding-generating sequence models.

    Required interface (per Appendix G):
      - forward(sequence) -> embedding
      - nominal_context: int (bp)
      - embedding_dim: int
      - token_to_bp(token_idx) -> bp_position
    """

    @abstractmethod
    def forward(self, sequence: npt.NDArray) -> npt.NDArray:
        """Compute embedding from an integer-encoded DNA sequence.

        Parameters
        ----------
        sequence : int array of shape (L,) with values in {0, 1, 2, 3}.

        Returns
        -------
        embedding : float array of shape (d,) or (T, d) for multi-track models.
        """

    @property
    @abstractmethod
    def nominal_context(self) -> int:
        """Nominal context length in base pairs."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the output embedding."""

    def token_to_bp(self, token_idx: int) -> int:
        """Map token index to base-pair coordinate.

        Default: identity mapping (single-nucleotide resolution).
        """
        return token_idx

    def __call__(self, sequence: npt.NDArray) -> npt.NDArray:
        return self.forward(sequence)


class SyntheticModel(BaseGenomicModel):
    """Synthetic model with controllable influence decay for testing.

    The embedding at position r is a weighted sum of one-hot encoded inputs,
    where weights decay exponentially from r:

        f(x)_j = sum_i w(|i - r|) * x_{i,j}

    with w(d) = exp(-d / decay_length).

    Parameters
    ----------
    seq_length : int
        Input sequence length L.
    embed_dim : int
        Embedding dimensionality d.
    decay_length : float
        Characteristic decay length in bp. Larger = longer effective context.
    reference : int or None
        Reference position. Defaults to seq_length // 2.
    noise_std : float
        Additive Gaussian noise to the embedding.
    """

    def __init__(
        self,
        seq_length: int = 1000,
        embed_dim: int = 64,
        decay_length: float = 100.0,
        reference: int | None = None,
        noise_std: float = 0.0,
    ):
        self._seq_length = seq_length
        self._embed_dim = embed_dim
        self._decay_length = decay_length
        self._reference = reference if reference is not None else seq_length // 2
        self._noise_std = noise_std

        # Precompute weights
        positions = np.arange(seq_length)
        distances = np.abs(positions - self._reference).astype(np.float64)
        self._weights = np.exp(-distances / decay_length)
        self._weights /= self._weights.sum()

        # Random projection for embedding
        self._rng = np.random.default_rng(42)
        self._projection = self._rng.standard_normal((4, embed_dim))

        # Dinucleotide interaction projection — makes the model sensitive to
        # nucleotide ORDER (not just composition), so that shuffle perturbations
        # produce nonzero influence.  16 dinucleotides, scaled to ~10% of the
        # main signal so the exponential decay shape is preserved.
        self._dinuc_projection = self._rng.standard_normal((16, embed_dim)) * 0.1

    def forward(self, sequence: npt.NDArray) -> npt.NDArray:
        seq = np.asarray(sequence, dtype=np.int64)
        # One-hot encode: (L, 4)
        one_hot = np.eye(4, dtype=np.float64)[seq]
        # Weighted sum: sum_i w_i * one_hot[i] -> (4,) then project -> (d,)
        weighted = np.einsum("i,ij->j", self._weights, one_hot)
        embedding = weighted @ self._projection

        # Dinucleotide interaction term: weighted sum of dinucleotide embeddings.
        # This depends on the ORDER of nucleotides at consecutive positions,
        # making the model sensitive to shuffle perturbations.
        dinuc_indices = seq[:-1] * 4 + seq[1:]  # (L-1,) values in 0..15
        dinuc_embeds = self._dinuc_projection[dinuc_indices]  # (L-1, d)
        embedding += np.einsum("i,ij->j", self._weights[:-1], dinuc_embeds)

        if self._noise_std > 0:
            embedding += self._rng.normal(0, self._noise_std, size=embedding.shape)
        return embedding

    @property
    def nominal_context(self) -> int:
        return self._seq_length

    @property
    def embedding_dim(self) -> int:
        return self._embed_dim


class LocalModel(BaseGenomicModel):
    """Synthetic model with exact locality (finite receptive field).

    Only positions within radius R of the reference affect the embedding.
    Used to verify Proposition 6.3 (ECL upper bound under exact locality).
    """

    def __init__(
        self,
        seq_length: int = 1000,
        embed_dim: int = 64,
        receptive_field: int = 50,
        reference: int | None = None,
    ):
        self._seq_length = seq_length
        self._embed_dim = embed_dim
        self._receptive_field = receptive_field
        self._reference = reference if reference is not None else seq_length // 2
        self._rng = np.random.default_rng(42)
        self._projection = self._rng.standard_normal((4, embed_dim))

    def forward(self, sequence: npt.NDArray) -> npt.NDArray:
        seq = np.asarray(sequence, dtype=np.int64)
        r = self._reference
        R = self._receptive_field
        lo = max(0, r - R)
        hi = min(len(seq), r + R + 1)
        one_hot = np.eye(4, dtype=np.float64)[seq[lo:hi]]
        summed = one_hot.mean(axis=0)
        return summed @ self._projection

    @property
    def nominal_context(self) -> int:
        return self._seq_length

    @property
    def embedding_dim(self) -> int:
        return self._embed_dim


class AdditiveModel(BaseGenomicModel):
    """Synthetic additive model f(X) = sum_i g_i(X_i) for Sobol equivalence testing.

    Each g_i is a random linear map from one-hot to embedding space,
    with influence magnitude proportional to exp(-|i - r| / decay_length).
    Used to verify Theorem 6.4 (Sobol equivalence).
    """

    def __init__(
        self,
        seq_length: int = 200,
        embed_dim: int = 32,
        decay_length: float = 30.0,
        reference: int | None = None,
    ):
        self._seq_length = seq_length
        self._embed_dim = embed_dim
        self._reference = reference if reference is not None else seq_length // 2
        rng = np.random.default_rng(42)

        positions = np.arange(seq_length)
        distances = np.abs(positions - self._reference).astype(np.float64)
        scales = np.exp(-distances / decay_length)

        # Per-position random projections g_i: R^4 -> R^d, scaled by distance
        self._projections = np.array(
            [scales[i] * rng.standard_normal((4, embed_dim)) for i in range(seq_length)]
        )

    def forward(self, sequence: npt.NDArray) -> npt.NDArray:
        seq = np.asarray(sequence, dtype=np.int64)
        embedding = np.zeros(self._embed_dim, dtype=np.float64)
        for i in range(len(seq)):
            one_hot = np.eye(4, dtype=np.float64)[seq[i]]
            embedding += one_hot @ self._projections[i]
        return embedding

    @property
    def nominal_context(self) -> int:
        return self._seq_length

    @property
    def embedding_dim(self) -> int:
        return self._embed_dim
