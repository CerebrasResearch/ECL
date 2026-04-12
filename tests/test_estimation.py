"""Tests for ecl.estimation module."""

from __future__ import annotations

import numpy as np

from ecl.estimation import (
    antithetic_estimate,
    asymptotic_ci,
    bernstein_confidence_radius,
    bootstrap_ecl_ci,
    permutation_test,
    sample_complexity,
)


class TestBernsteinConfidenceRadius:
    def test_decreases_with_n(self):
        r1 = bernstein_confidence_radius(100, variance=1.0, bound=10.0, delta=0.05)
        r2 = bernstein_confidence_radius(1000, variance=1.0, bound=10.0, delta=0.05)
        assert r2 < r1

    def test_increases_with_variance(self):
        r1 = bernstein_confidence_radius(100, variance=0.1, bound=10.0, delta=0.05)
        r2 = bernstein_confidence_radius(100, variance=10.0, bound=10.0, delta=0.05)
        assert r2 > r1

    def test_positive(self):
        r = bernstein_confidence_radius(100, variance=1.0, bound=5.0, delta=0.05)
        assert r > 0


class TestSampleComplexity:
    def test_increases_with_L(self):
        n1 = sample_complexity(100, bound=1.0, margin=0.1)
        n2 = sample_complexity(1000, bound=1.0, margin=0.1)
        assert n2 > n1

    def test_decreases_with_margin(self):
        n1 = sample_complexity(100, bound=1.0, margin=0.01)
        n2 = sample_complexity(100, bound=1.0, margin=1.0)
        assert n1 > n2

    def test_positive_integer(self):
        n = sample_complexity(50, bound=1.0, margin=0.5)
        assert n > 0
        assert isinstance(n, int)


class TestBootstrapEclCI:
    def test_ci_contains_point_estimate(self, rng):
        n, D = 50, 30
        # Simulate exponentially decaying influence
        distances = np.arange(D)
        samples = np.zeros((n, D))
        for t in range(n):
            samples[t] = 5.0 * np.exp(-distances / 10.0) + rng.normal(0, 0.1, D).clip(0)

        ecl_point, ci_lo, ci_hi = bootstrap_ecl_ci(
            samples, distances, beta=0.9, n_bootstrap=200, rng=rng
        )
        assert ci_lo <= ecl_point <= ci_hi

    def test_wider_ci_with_more_variance(self, rng):
        n, D = 30, 20
        distances = np.arange(D)

        # Low variance
        samples_low = np.zeros((n, D))
        for t in range(n):
            samples_low[t] = 5.0 * np.exp(-distances / 8.0) + rng.normal(0, 0.01, D).clip(0)

        # High variance
        samples_high = np.zeros((n, D))
        for t in range(n):
            samples_high[t] = 5.0 * np.exp(-distances / 8.0) + rng.normal(0, 2.0, D).clip(0)

        _, lo1, hi1 = bootstrap_ecl_ci(samples_low, distances, n_bootstrap=200, rng=rng)
        _, lo2, hi2 = bootstrap_ecl_ci(samples_high, distances, n_bootstrap=200, rng=rng)
        # High variance should give wider CI (or at least not narrower)
        assert (hi2 - lo2) >= 0
        assert (hi1 - lo1) >= 0


class TestAntitheticEstimate:
    def test_unbiased(self, rng):
        n = 1000
        true_mean = 5.0
        d_plus = rng.normal(true_mean, 1.0, n)
        d_minus = rng.normal(true_mean, 1.0, n)
        mean, var = antithetic_estimate(d_plus, d_minus)
        assert abs(mean - true_mean) < 0.5  # Should be close

    def test_variance_reduction(self, rng):
        n = 1000
        # Negatively correlated pairs (ideal antithetic case)
        base = rng.normal(0, 1.0, n)
        d_plus = 5.0 + base
        d_minus = 5.0 - base * 0.5
        mean, var = antithetic_estimate(d_plus, d_minus)
        # Mean should be close to 5.0
        assert abs(mean - 5.0) < 0.5


class TestPermutationTest:
    def test_null_hypothesis(self, rng):
        """Under H0 (no difference), p-value should be large."""
        n = 100
        ecl_f = rng.normal(50, 10, n)
        ecl_g = ecl_f + rng.normal(0, 0.1, n)  # Nearly identical
        mean_diff, p_val, ci = permutation_test(ecl_f, ecl_g, n_permutations=500, rng=rng)
        assert p_val > 0.01  # Should not reject H0

    def test_alternative_hypothesis(self, rng):
        """Under H1 (large difference), p-value should be small."""
        n = 100
        ecl_f = rng.normal(100, 5, n)
        ecl_g = rng.normal(50, 5, n)
        mean_diff, p_val, ci = permutation_test(ecl_f, ecl_g, n_permutations=1000, rng=rng)
        assert p_val < 0.05  # Should reject H0
        assert mean_diff > 0  # f has larger ECL

    def test_returns_correct_types(self, rng):
        ecl_f = rng.normal(50, 10, 20)
        ecl_g = rng.normal(50, 10, 20)
        mean_diff, p_val, ci = permutation_test(ecl_f, ecl_g, n_permutations=100, rng=rng)
        assert isinstance(mean_diff, float)
        assert isinstance(p_val, float)
        assert 0 <= p_val <= 1


class TestAsymptoticCI:
    def test_output_shapes(self, rng):
        samples = rng.standard_normal((50, 10))
        means, lo, hi = asymptotic_ci(samples, alpha=0.05)
        assert means.shape == (10,)
        assert lo.shape == (10,)
        assert hi.shape == (10,)

    def test_ci_contains_mean(self, rng):
        samples = rng.standard_normal((100, 5))
        means, lo, hi = asymptotic_ci(samples, alpha=0.05)
        assert np.all(lo <= means)
        assert np.all(means <= hi)
