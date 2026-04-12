"""Context Decay Spectroscopy (CDS) — Section 5.8 of the paper.

Fits the distance-binned influence profile to a mixture of exponentials:
    I(d; r) ~= sum_k a_k * exp(-lambda_k * d)

Each component represents a distinct context channel with decay rate lambda_k.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit, nnls


def _single_exp(d: npt.NDArray, a: float, lam: float) -> npt.NDArray:
    return a * np.exp(-lam * d)


def _mixture_exp(d: npt.NDArray, *params) -> npt.NDArray:
    """Mixture of K exponentials: sum_k a_k * exp(-lambda_k * d).

    params = [a_1, lambda_1, a_2, lambda_2, ...].
    """
    K = len(params) // 2
    result = np.zeros_like(d, dtype=np.float64)
    for k in range(K):
        a_k = params[2 * k]
        lam_k = params[2 * k + 1]
        result += a_k * np.exp(-lam_k * d)
    return result


def fit_cds(
    distances: npt.NDArray,
    influence: npt.NDArray,
    n_components: int = 2,
    method: str = "nls",
) -> dict:
    """Fit Context Decay Spectroscopy model.

    Parameters
    ----------
    distances : array of shape (D+1,), distance values.
    influence : array of shape (D+1,), binned influence I(d; r).
    n_components : int
        Number of exponential components K.
    method : str
        'nls' for non-linear least squares (scipy curve_fit),
        'nnls' for non-negative least squares on a fixed lambda grid.

    Returns
    -------
    dict with keys:
        'amplitudes': array of shape (K,)
        'decay_rates': array of shape (K,), lambda_k (sorted ascending = slowest first)
        'fitted': array of shape (D+1,), fitted values
        'residual': float, sum of squared residuals
        'bic': float, Bayesian Information Criterion
    """
    d = np.asarray(distances, dtype=np.float64)
    y = np.asarray(influence, dtype=np.float64)

    # Filter positive influence for fitting
    mask = y > 0
    if mask.sum() < 2 * n_components:
        # Not enough data; return uniform fallback
        return _fallback_result(d, y, n_components)

    if method == "nnls":
        return _fit_nnls(d, y, n_components)
    return _fit_nls(d, y, n_components)


def _fit_nls(d: npt.NDArray, y: npt.NDArray, K: int) -> dict:
    """Fit via non-linear least squares."""
    # Initial guesses: spread lambda values logarithmically
    a_init = float(np.max(y)) / K
    lam_max = 1.0 / max(1.0, d[d > 0].min()) if np.any(d > 0) else 1.0
    lam_min = 1.0 / max(1.0, d.max()) if d.max() > 0 else 0.01

    p0 = []
    bounds_lo = []
    bounds_hi = []
    for k in range(K):
        p0.extend([a_init, lam_min + (lam_max - lam_min) * (k + 1) / (K + 1)])
        bounds_lo.extend([0.0, 1e-6])
        bounds_hi.extend([np.inf, 100.0])

    try:
        popt, _ = curve_fit(
            _mixture_exp, d, y, p0=p0, bounds=(bounds_lo, bounds_hi), maxfev=10000
        )
    except RuntimeError:
        return _fallback_result(d, y, K)

    amplitudes = np.array([popt[2 * k] for k in range(K)])
    decay_rates = np.array([popt[2 * k + 1] for k in range(K)])

    # Sort by decay rate ascending (slowest first)
    order = np.argsort(decay_rates)
    amplitudes = amplitudes[order]
    decay_rates = decay_rates[order]

    fitted = _mixture_exp(d, *popt)
    residual = float(np.sum((y - fitted) ** 2))
    n = len(y)
    bic = n * np.log(residual / n + 1e-12) + 2 * K * np.log(n)

    return {
        "amplitudes": amplitudes,
        "decay_rates": decay_rates,
        "fitted": fitted,
        "residual": residual,
        "bic": bic,
    }


def _fit_nnls(d: npt.NDArray, y: npt.NDArray, K: int) -> dict:
    """Fit via NNLS on a grid of fixed decay rates."""
    # Create grid of candidate lambda values
    n_candidates = max(K * 10, 50)
    if d.max() > 0:
        lam_grid = np.logspace(np.log10(1.0 / d.max()), np.log10(1.0), n_candidates)
    else:
        lam_grid = np.logspace(-3, 0, n_candidates)

    # Design matrix
    A = np.column_stack([np.exp(-lam * d) for lam in lam_grid])

    # NNLS solve
    coefs, residual_norm = nnls(A, y)

    # Select top K components
    top_k = np.argsort(coefs)[-K:]
    top_k = top_k[coefs[top_k] > 0]

    if len(top_k) == 0:
        return _fallback_result(d, y, K)

    amplitudes = coefs[top_k]
    decay_rates = lam_grid[top_k]

    # Sort by decay rate ascending
    order = np.argsort(decay_rates)
    amplitudes = amplitudes[order]
    decay_rates = decay_rates[order]

    fitted = sum(a * np.exp(-lam * d) for a, lam in zip(amplitudes, decay_rates, strict=True))
    residual = float(np.sum((y - fitted) ** 2))
    n = len(y)
    bic = n * np.log(residual / n + 1e-12) + 2 * len(amplitudes) * np.log(n)

    return {
        "amplitudes": amplitudes,
        "decay_rates": decay_rates,
        "fitted": fitted,
        "residual": residual,
        "bic": bic,
    }


def _fallback_result(d: npt.NDArray, y: npt.NDArray, K: int) -> dict:
    """Fallback when fitting fails."""
    return {
        "amplitudes": np.full(K, float(np.mean(y))),
        "decay_rates": np.linspace(0.01, 1.0, K),
        "fitted": np.full_like(d, float(np.mean(y)), dtype=np.float64),
        "residual": float(np.sum(y**2)),
        "bic": float("inf"),
    }


def select_n_components(
    distances: npt.NDArray,
    influence: npt.NDArray,
    max_K: int = 4,
    method: str = "nls",
) -> tuple[int, list[dict]]:
    """Select the optimal number of CDS components by BIC.

    Parameters
    ----------
    distances, influence : binned influence profile.
    max_K : maximum number of components to try.
    method : fitting method.

    Returns
    -------
    best_K : int, optimal number of components.
    results : list of fit results for K=1, ..., max_K.
    """
    results = []
    for K in range(1, max_K + 1):
        res = fit_cds(distances, influence, n_components=K, method=method)
        results.append(res)

    bics = [r["bic"] for r in results]
    best_K = int(np.argmin(bics)) + 1
    return best_K, results


def spectral_ecl(
    amplitudes: npt.NDArray,
    decay_rates: npt.NDArray,
    beta: float = 0.9,
    max_distance: int = 1_000_000,
) -> int:
    """Compute spectral ECL from the CDS mixture.

    Analytically computes the ECL from the fitted mixture of exponentials
    using the closed-form cumulative: sum_k (a_k / lambda_k) * (1 - exp(-lambda_k * l)).
    """
    # Total influence: sum_k a_k / lambda_k (integral from 0 to inf)
    total = np.sum(amplitudes / decay_rates)
    if total <= 0:
        return 0

    threshold = beta * total
    for ell in range(max_distance + 1):
        cumul = np.sum(amplitudes / decay_rates * (1 - np.exp(-decay_rates * ell)))
        if cumul >= threshold:
            return ell
    return max_distance
