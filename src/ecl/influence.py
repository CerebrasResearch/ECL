"""Influence energy computation (Algorithms 1 and 2 from the paper).

Implements:
  - Single-position influence energy I(i; r)  [Eq. 1]
  - Binned influence profile I(d; r)           [Eq. 3, Algorithm 2]
  - Full Monte Carlo influence estimation      [Algorithm 1]
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from ecl.metrics import squared_euclidean
from ecl.perturbations import PerturbationKernel, RandomSubstitution


def compute_single_influence(
    model_fn: Callable[[npt.NDArray], npt.NDArray],
    sequences: npt.NDArray,
    position: int,
    reference: int,
    perturbation: PerturbationKernel | None = None,
    metric: Callable = squared_euclidean,
    rng: np.random.Generator | None = None,
) -> float:
    """Estimate influence energy I(position; reference) from multiple sequences.

    Parameters
    ----------
    model_fn : callable
        Maps integer sequence (L,) -> embedding (d,).
    sequences : array of shape (n, L)
        Sample of n sequences from P_X.
    position : int
        Position i to perturb.
    reference : int
        Reference position r (used for extracting per-position embeddings if needed).
    perturbation : PerturbationKernel
        Defaults to RandomSubstitution.
    metric : callable
        Embedding discrepancy d_Z. Defaults to squared Euclidean.
    rng : numpy random generator.

    Returns
    -------
    float
        Estimated I(position; reference).
    """
    perturbation = perturbation or RandomSubstitution()
    rng = rng or np.random.default_rng()
    n = len(sequences)

    total = 0.0
    positions_arr = np.array([position])
    for t in range(n):
        seq = sequences[t]
        z = model_fn(seq)
        perturbed = perturbation(seq, positions_arr, rng)
        z_pert = model_fn(perturbed)
        total += float(metric(z, z_pert))

    return total / n


def compute_influence_profile(
    model_fn: Callable[[npt.NDArray], npt.NDArray],
    sequences: npt.NDArray,
    reference: int,
    max_distance: int | None = None,
    positions_per_distance: int = 10,
    perturbation: PerturbationKernel | None = None,
    metric: Callable = squared_euclidean,
    rng: np.random.Generator | None = None,
    show_progress: bool = True,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Binned influence profile estimator (Algorithm 2).

    Estimates I(d; r) for each distance d by sub-sampling positions at that distance.

    Parameters
    ----------
    model_fn : callable
        Maps integer sequence (L,) -> embedding (d,).
    sequences : array of shape (n, L)
        Sample sequences from P_X.
    reference : int
        Reference position r.
    max_distance : int or None
        Maximum distance D to estimate. Defaults to L // 2.
    positions_per_distance : int
        Number of positions m(d) to sample per distance bin.
    perturbation : PerturbationKernel
        Defaults to RandomSubstitution.
    metric : callable
        Embedding discrepancy. Defaults to squared Euclidean.
    rng : numpy random generator.
    show_progress : bool
        Whether to show tqdm progress bar.

    Returns
    -------
    distances : array of shape (D+1,)
        Distance values [0, 1, ..., D].
    influence : array of shape (D+1,)
        Estimated I(d; r) for each distance.
    """
    perturbation = perturbation or RandomSubstitution()
    rng = rng or np.random.default_rng()
    n, L = sequences.shape

    if max_distance is None:
        max_distance = L // 2

    distances = np.arange(max_distance + 1)
    influence = np.zeros(max_distance + 1, dtype=np.float64)

    dist_iter = tqdm(range(max_distance + 1), desc="Distances", disable=not show_progress)
    for d in dist_iter:
        # Collect all positions at distance d from reference
        candidates = []
        if reference - d >= 0:
            candidates.append(reference - d)
        if d > 0 and reference + d < L:
            candidates.append(reference + d)

        if not candidates:
            continue

        # Sub-sample positions
        m = min(positions_per_distance, len(candidates))
        sampled_positions = rng.choice(candidates, size=m, replace=True)

        accum = 0.0
        count = 0
        for t in range(n):
            seq = sequences[t]
            z = model_fn(seq)
            for pos in sampled_positions:
                perturbed = perturbation(seq, np.array([pos]), rng)
                z_pert = model_fn(perturbed)
                accum += float(metric(z, z_pert))
                count += 1

        influence[d] = accum / max(count, 1)

    return distances, influence


def compute_block_influence(
    model_fn: Callable[[npt.NDArray], npt.NDArray],
    sequences: npt.NDArray,
    reference: int,
    block_size: int = 128,
    perturbation: PerturbationKernel | None = None,
    metric: Callable = squared_euclidean,
    rng: np.random.Generator | None = None,
    show_progress: bool = True,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Compute block-level influence (Algorithm 4, multi-scale block ECL).

    Partitions the sequence into blocks of size `block_size` and estimates
    I(B_k; r) for each block.

    Parameters
    ----------
    model_fn, sequences, reference, perturbation, metric, rng : as above.
    block_size : int
        Block size b in base pairs.

    Returns
    -------
    block_distances : array of shape (K,)
        Distance of each block from reference (min |i - r| for i in B_k).
    block_influences : array of shape (K,)
        Estimated I(B_k; r).
    """
    perturbation = perturbation or RandomSubstitution()
    rng = rng or np.random.default_rng()
    n, L = sequences.shape

    K = int(np.ceil(L / block_size))
    block_distances = np.zeros(K, dtype=np.float64)
    block_influences = np.zeros(K, dtype=np.float64)

    blocks_iter = tqdm(range(K), desc="Blocks", disable=not show_progress)
    for k in blocks_iter:
        start = k * block_size
        end = min((k + 1) * block_size, L)
        positions = np.arange(start, end)
        block_distances[k] = float(np.min(np.abs(positions - reference)))

        accum = 0.0
        for t in range(n):
            seq = sequences[t]
            z = model_fn(seq)
            perturbed = perturbation(seq, positions, rng)
            z_pert = model_fn(perturbed)
            accum += float(metric(z, z_pert))

        block_influences[k] = accum / n

    # Sort by distance
    order = np.argsort(block_distances)
    return block_distances[order], block_influences[order]


def compute_interaction_influence(
    model_fn: Callable[[npt.NDArray], npt.NDArray],
    sequences: npt.NDArray,
    pos_i: int,
    pos_j: int,
    perturbation: PerturbationKernel | None = None,
    metric: Callable = squared_euclidean,
    rng: np.random.Generator | None = None,
) -> float:
    """Pairwise interaction influence I_int(i, j; r) [Eq. 9].

    I_int(i,j;r) = I({i,j}; r) - I(i; r) - I(j; r)

    Positive => synergistic; negative => redundant.
    """
    perturbation = perturbation or RandomSubstitution()
    rng = rng or np.random.default_rng()
    n = len(sequences)

    sum_i = 0.0
    sum_j = 0.0
    sum_ij = 0.0

    for t in range(n):
        seq = sequences[t]
        z = model_fn(seq)

        # Single-site perturbations
        z_i = model_fn(perturbation(seq, np.array([pos_i]), rng))
        z_j = model_fn(perturbation(seq, np.array([pos_j]), rng))
        # Joint perturbation
        z_ij = model_fn(perturbation(seq, np.array([pos_i, pos_j]), rng))

        sum_i += float(metric(z, z_i))
        sum_j += float(metric(z, z_j))
        sum_ij += float(metric(z, z_ij))

    I_i = sum_i / n
    I_j = sum_j / n
    I_ij = sum_ij / n
    return I_ij - I_i - I_j
