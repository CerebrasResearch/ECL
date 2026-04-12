"""Statistical estimation utilities (Section 8 of the paper).

Implements:
  - Bernstein concentration bounds       [Theorem 8.1]
  - Bootstrap confidence intervals        [Algorithm 3]
  - Antithetic variance reduction         [Proposition 8.4]
  - Importance sampling                   [Proposition 8.5]
  - Permutation test for ECL comparison   [Algorithm 5]
  - Asymptotic confidence intervals       [Theorem 8.6]
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ecl.ecl import ECL


def bernstein_confidence_radius(
    n: int,
    variance: float,
    bound: float,
    delta: float = 0.05,
) -> float:
    """Bernstein confidence radius [Theorem 8.1, Eq. 14].

    epsilon_n(delta) = M * log(2/delta) / (3n) + sqrt(2 * sigma^2 * log(2/delta) / n)

    Parameters
    ----------
    n : number of samples.
    variance : estimated variance sigma_i^2 of U_i.
    bound : upper bound M on |U_i|.
    delta : failure probability.

    Returns
    -------
    float : confidence radius.
    """
    log_term = np.log(2.0 / delta)
    return bound * log_term / (3.0 * n) + np.sqrt(2.0 * variance * log_term / n)


def sample_complexity(
    L: int,
    bound: float,
    margin: float,
    delta: float = 0.05,
) -> int:
    """Minimum sample size for exact ECL estimation [Corollary 8.3, Eq. 15].

    n >= 128 * L^2 * M^2 / gamma^2 * log(2L / delta)
    """
    n = 128 * L**2 * bound**2 / margin**2 * np.log(2 * L / delta)
    return int(np.ceil(n))


def bootstrap_ecl_ci(
    influence_samples: npt.NDArray,
    distances: npt.NDArray,
    beta: float = 0.9,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    reference: int | None = None,
    L: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for ECL_beta(r) [Algorithm 3].

    Parameters
    ----------
    influence_samples : array of shape (n, D+1)
        Per-sample influence values at each distance.
        influence_samples[t, d] = d_Z(f(X^t), f(X^{t,d})) for sample t, distance d.
    distances : array of shape (D+1,)
    beta : ECL threshold.
    n_bootstrap : number of bootstrap replicates B.
    alpha : significance level (yields 1-alpha CI).
    reference, L : for boundary correction.
    rng : random generator.

    Returns
    -------
    ecl_point : float, point estimate.
    ci_lower : float
    ci_upper : float
    """
    rng = rng or np.random.default_rng()
    n = influence_samples.shape[0]

    # Point estimate
    mean_influence = influence_samples.mean(axis=0)
    ecl_point = float(ECL(distances, mean_influence, beta=beta, reference=reference, L=L))

    # Bootstrap
    ecl_boots = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_influence = influence_samples[idx].mean(axis=0)
        ecl_boots[b] = ECL(distances, boot_influence, beta=beta, reference=reference, L=L)

    ci_lower = float(np.percentile(ecl_boots, 100 * alpha / 2))
    ci_upper = float(np.percentile(ecl_boots, 100 * (1 - alpha / 2)))
    return ecl_point, ci_lower, ci_upper


def antithetic_estimate(
    d_plus: npt.NDArray,
    d_minus: npt.NDArray,
) -> tuple[float, float]:
    """Antithetic variance-reduced influence estimator [Proposition 8.4].

    Parameters
    ----------
    d_plus : array of shape (n,), d_Z(f(X), f(X^{i,+})) for original perturbation.
    d_minus : array of shape (n,), d_Z(f(X), f(X^{i,-})) for antithetic perturbation.

    Returns
    -------
    mean : antithetic estimate of I(i; r).
    variance : estimated variance of the antithetic estimator.
    """
    combined = (d_plus + d_minus) / 2.0
    mean = float(np.mean(combined))
    variance = float(np.var(combined, ddof=1) / len(combined))
    return mean, variance


def importance_weighted_estimate(
    influence_values: npt.NDArray,
    proposal_probs: npt.NDArray,
    pilot_probs: npt.NDArray,
) -> npt.NDArray:
    """Importance-weighted influence estimator [Proposition 8.5].

    Parameters
    ----------
    influence_values : array of shape (M,), raw influence at sampled positions.
    proposal_probs : array of shape (M,), q(i) for each sampled position.
    pilot_probs : array of shape (M,), pilot estimate I_bar_0(i; r) for each position.

    Returns
    -------
    corrected : array of shape (M,), importance-corrected influence estimates.
    """
    weights = pilot_probs / np.maximum(proposal_probs, 1e-12)
    return influence_values * weights


def permutation_test(
    ecl_f: npt.NDArray,
    ecl_g: npt.NDArray,
    n_permutations: int = 10000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Permutation test for ECL comparison between models [Algorithm 5].

    Tests H0: E[ECL^f(R) - ECL^g(R)] = 0 using paired sign-flip permutation.

    Parameters
    ----------
    ecl_f : array of shape (N,), ECL estimates for model f at N loci.
    ecl_g : array of shape (N,), ECL estimates for model g at same loci.
    n_permutations : number of permutations P.
    rng : random generator.

    Returns
    -------
    mean_diff : mean(ECL_f - ECL_g).
    p_value : two-sided p-value.
    ci_95 : 95% CI half-width for the mean difference (bootstrap SE * 1.96).
    """
    rng = rng or np.random.default_rng()
    D = ecl_f - ecl_g
    N = len(D)
    T_obs = np.abs(np.mean(D))

    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=N)
        T_perm = np.abs(np.mean(D * signs))
        if T_perm >= T_obs:
            count += 1

    p_value = count / n_permutations
    mean_diff = float(np.mean(D))
    se = float(np.std(D, ddof=1) / np.sqrt(N))
    return mean_diff, p_value, 1.96 * se


def asymptotic_ci(
    influence_samples: npt.NDArray,
    alpha: float = 0.05,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Asymptotic (CLT-based) confidence intervals [Theorem 8.6].

    Parameters
    ----------
    influence_samples : array of shape (n, D+1)
        Per-sample influence at each distance.
    alpha : significance level.

    Returns
    -------
    means : array of shape (D+1,)
    ci_lower : array of shape (D+1,)
    ci_upper : array of shape (D+1,)
    """
    from scipy import stats

    n = influence_samples.shape[0]
    means = influence_samples.mean(axis=0)
    stds = influence_samples.std(axis=0, ddof=1)
    z = stats.norm.ppf(1 - alpha / 2)
    half_width = z * stds / np.sqrt(n)
    return means, means - half_width, means + half_width
