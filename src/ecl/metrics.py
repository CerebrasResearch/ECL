"""Embedding discrepancy metrics (Section 3.5 of the paper).

Provides squared Euclidean, cosine, Mahalanobis, and task-weighted metrics
for computing d_Z(z, z').
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def squared_euclidean(z: npt.NDArray, z_prime: npt.NDArray) -> npt.NDArray:
    """d_Z(z, z') = ||z - z'||_2^2.

    Parameters
    ----------
    z, z_prime : array of shape (..., d)

    Returns
    -------
    array of shape (...)
    """
    diff = z - z_prime
    return np.sum(diff * diff, axis=-1)


def cosine_distance(z: npt.NDArray, z_prime: npt.NDArray) -> npt.NDArray:
    """d_cos(z, z') = 1 - <z, z'> / (||z|| ||z'||).

    Parameters
    ----------
    z, z_prime : array of shape (..., d)

    Returns
    -------
    array of shape (...)
    """
    dot = np.sum(z * z_prime, axis=-1)
    norm_z = np.sqrt(np.sum(z * z, axis=-1))
    norm_zp = np.sqrt(np.sum(z_prime * z_prime, axis=-1))
    denom = norm_z * norm_zp
    # Avoid division by zero
    denom = np.maximum(denom, 1e-12)
    return 1.0 - dot / denom


def mahalanobis_distance(
    z: npt.NDArray,
    z_prime: npt.NDArray,
    precision: npt.NDArray,
) -> npt.NDArray:
    """d_Sigma(z, z') = (z - z')^T Sigma^{-1} (z - z').

    Parameters
    ----------
    z, z_prime : array of shape (..., d)
    precision : array of shape (d, d), the inverse covariance matrix.

    Returns
    -------
    array of shape (...)
    """
    diff = z - z_prime  # (..., d)
    # (..., d) @ (d, d) -> (..., d)
    transformed = diff @ precision
    return np.sum(transformed * diff, axis=-1)


def task_weighted_distance(
    z: npt.NDArray,
    z_prime: npt.NDArray,
    W: npt.NDArray,
) -> npt.NDArray:
    """d_W(z, z') = (z - z')^T W (z - z').

    Parameters
    ----------
    z, z_prime : array of shape (..., d)
    W : array of shape (d, d), positive-definite weight matrix.

    Returns
    -------
    array of shape (...)
    """
    diff = z - z_prime
    transformed = diff @ W
    return np.sum(transformed * diff, axis=-1)


def estimate_precision(
    embeddings: npt.NDArray,
    regularization: float = 0.01,
) -> npt.NDArray:
    """Estimate the precision matrix (inverse covariance) from embeddings.

    Parameters
    ----------
    embeddings : array of shape (n, d)
    regularization : ridge regularization lambda for numerical stability.

    Returns
    -------
    precision : array of shape (d, d)
    """
    cov = np.cov(embeddings, rowvar=False)
    cov += regularization * np.eye(cov.shape[0])
    return np.linalg.inv(cov)


# Registry for convenient lookup
METRICS = {
    "squared_euclidean": squared_euclidean,
    "cosine": cosine_distance,
}


def get_metric(name: str = "squared_euclidean"):
    """Look up a metric function by name."""
    if name not in METRICS:
        raise ValueError(f"Unknown metric '{name}'. Choose from: {list(METRICS.keys())}")
    return METRICS[name]
