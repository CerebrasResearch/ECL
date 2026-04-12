"""ECL computation from influence profiles (Section 5 of the paper).

Core quantities:
  - Cumulative influence I_{<=l}(r)          [Eq. 4]
  - Perturbation-variance ECL_beta(r)        [Def. 4.1, Eq. 5]
  - Effective Context Profile (ECP)          [Def. 4.2]
  - Area Under the ECP (AECP)               [Def. 4.3]
  - Effective Context Dimension (ECD)        [Def. 4.5]
  - Directional ECL                         [Eq. 10-11]
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def cumulative_influence(
    distances: npt.NDArray,
    influence: npt.NDArray,
    reference: int | None = None,
    L: int | None = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Compute cumulative influence I_{<=l}(r) as a function of radius l.

    Parameters
    ----------
    distances : array of shape (D+1,)
        Distance values [0, 1, ..., D].
    influence : array of shape (D+1,)
        Binned influence I(d; r).
    reference : int or None
        Reference position (used to compute multiplicity). If None, assumes
        interior reference with multiplicity 2 for d > 0.
    L : int or None
        Sequence length (used for boundary multiplicity correction).

    Returns
    -------
    radii : array of shape (D+1,)
    cumulative : array of shape (D+1,), cumulative influence I_{<=l}(r).
    """
    D = len(distances) - 1

    # Compute multiplicity: number of positions at each distance
    if reference is not None and L is not None:
        multiplicity = np.array(
            [_count_at_distance(d, reference, L) for d in range(D + 1)], dtype=np.float64
        )
    else:
        # Assume interior: multiplicity 1 for d=0, 2 for d>0
        multiplicity = np.ones(D + 1, dtype=np.float64)
        multiplicity[1:] = 2.0

    weighted = influence * multiplicity
    cumul = np.cumsum(weighted)
    return distances.copy(), cumul


def _count_at_distance(d: int, r: int, L: int) -> int:
    """Count positions at distance d from reference r in [0, L)."""
    count = 0
    if r - d >= 0:
        count += 1
    if d > 0 and r + d < L:
        count += 1
    return count


def normalized_influence(
    distances: npt.NDArray,
    influence: npt.NDArray,
    reference: int | None = None,
    L: int | None = None,
) -> npt.NDArray:
    """Normalized influence profile I_bar(d; r) [Eq. 2].

    Returns a probability distribution over distances.
    """
    _, cumul = cumulative_influence(distances, influence, reference, L)
    total = cumul[-1]
    if total <= 0:
        return np.zeros_like(influence)

    # Compute per-distance weighted influence
    D = len(distances) - 1
    if reference is not None and L is not None:
        multiplicity = np.array(
            [_count_at_distance(d, reference, L) for d in range(D + 1)], dtype=np.float64
        )
    else:
        multiplicity = np.ones(D + 1, dtype=np.float64)
        multiplicity[1:] = 2.0

    weighted = influence * multiplicity
    return weighted / total


def ECL(
    distances: npt.NDArray,
    influence: npt.NDArray,
    beta: float = 0.9,
    reference: int | None = None,
    L: int | None = None,
) -> int:
    """Perturbation-variance effective context length [Definition 4.1].

    ECL_beta(r) = min{l : I_{<=l}(r) >= beta * I_tot(r)}

    Parameters
    ----------
    distances : array of shape (D+1,)
    influence : array of shape (D+1,)
    beta : float in (0, 1]
        Fraction of total influence to capture.
    reference, L : for boundary correction.

    Returns
    -------
    int : ECL_beta(r) in distance units.
    """
    _, cumul = cumulative_influence(distances, influence, reference, L)
    total = cumul[-1]
    if total <= 0:
        return 0

    threshold = beta * total
    crossing = np.where(cumul >= threshold)[0]
    if len(crossing) == 0:
        return int(distances[-1])
    return int(distances[crossing[0]])


def ECP(
    distances: npt.NDArray,
    influence: npt.NDArray,
    betas: npt.NDArray | None = None,
    reference: int | None = None,
    L: int | None = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Effective Context Profile: beta -> ECL_beta(r) [Definition 4.2].

    Parameters
    ----------
    distances, influence : binned influence profile.
    betas : array of beta values. Defaults to linspace(0.01, 1.0, 100).

    Returns
    -------
    betas : array of shape (B,)
    ecl_values : array of shape (B,), ECL at each beta.
    """
    if betas is None:
        betas = np.linspace(0.01, 1.0, 100)
    ecl_values = np.array(
        [ECL(distances, influence, beta=b, reference=reference, L=L) for b in betas]
    )
    return betas, ecl_values


def AECP(
    distances: npt.NDArray,
    influence: npt.NDArray,
    reference: int | None = None,
    L: int | None = None,
    n_betas: int = 200,
) -> float:
    """Area Under the Effective Context Profile [Definition 4.3].

    AECP(r) = integral_0^1 ECL_beta(r) d_beta = sum_i |i - r| * I_bar(i; r)
    """
    norm_infl = normalized_influence(distances, influence, reference, L)
    return float(np.sum(distances * norm_infl))


def ECD(
    distances: npt.NDArray,
    influence: npt.NDArray,
    reference: int | None = None,
    L: int | None = None,
) -> float:
    """Effective Context Dimension [Definition 4.5].

    ECD(r) = exp(-sum_i I_bar(i;r) log I_bar(i;r))

    Measures the effective number of positions contributing to the embedding.
    """
    norm_infl = normalized_influence(distances, influence, reference, L)
    # Filter out zeros to avoid log(0)
    mask = norm_infl > 0
    if not np.any(mask):
        return 1.0
    p = norm_infl[mask]
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def directional_ecl(
    distances: npt.NDArray,
    influence_upstream: npt.NDArray,
    influence_downstream: npt.NDArray,
    beta: float = 0.9,
) -> tuple[int, int, float]:
    """Directional ECL (upstream/downstream) [Eq. 10-11].

    Parameters
    ----------
    distances : array [1, 2, ..., D] (excluding d=0).
    influence_upstream : I^-(d; r) for d = 1, ..., D.
    influence_downstream : I^+(d; r) for d = 1, ..., D.
    beta : fraction threshold.

    Returns
    -------
    ecl_upstream : ECL^-_beta(r)
    ecl_downstream : ECL^+_beta(r)
    asymmetry_ratio : ECL^+ / ECL^-
    """
    # Upstream ECL
    cumul_up = np.cumsum(influence_upstream)
    total_up = cumul_up[-1] if len(cumul_up) > 0 else 0.0
    if total_up > 0:
        crossing_up = np.where(cumul_up >= beta * total_up)[0]
        ecl_up = int(distances[crossing_up[0]]) if len(crossing_up) > 0 else int(distances[-1])
    else:
        ecl_up = 0

    # Downstream ECL
    cumul_down = np.cumsum(influence_downstream)
    total_down = cumul_down[-1] if len(cumul_down) > 0 else 0.0
    if total_down > 0:
        crossing_down = np.where(cumul_down >= beta * total_down)[0]
        ecl_down = (
            int(distances[crossing_down[0]]) if len(crossing_down) > 0 else int(distances[-1])
        )
    else:
        ecl_down = 0

    ratio = ecl_down / ecl_up if ecl_up > 0 else float("inf")
    return ecl_up, ecl_down, ratio
