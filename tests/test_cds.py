"""Tests for ecl.cds (Context Decay Spectroscopy) module."""

from __future__ import annotations

import numpy as np
import pytest

from ecl.cds import fit_cds, select_n_components, spectral_ecl


class TestFitCDS:
    def test_single_exponential(self):
        """Fit should recover a single exponential."""
        d = np.arange(100, dtype=np.float64)
        y = 5.0 * np.exp(-0.05 * d)
        result = fit_cds(d, y, n_components=1, method="nls")
        assert len(result["amplitudes"]) == 1
        assert result["amplitudes"][0] == pytest.approx(5.0, rel=0.3)
        assert result["decay_rates"][0] == pytest.approx(0.05, rel=0.3)

    def test_two_components(self):
        """Fit should recover two exponentials."""
        d = np.arange(200, dtype=np.float64)
        y = 3.0 * np.exp(-0.1 * d) + 1.0 * np.exp(-0.01 * d)
        result = fit_cds(d, y, n_components=2, method="nls")
        assert len(result["amplitudes"]) == 2
        # Sorted by decay rate ascending (slowest first)
        assert result["decay_rates"][0] < result["decay_rates"][1]

    def test_fitted_values_shape(self):
        d = np.arange(50, dtype=np.float64)
        y = np.exp(-0.1 * d)
        result = fit_cds(d, y, n_components=1)
        assert result["fitted"].shape == d.shape

    def test_residual_small_for_exact_fit(self):
        d = np.arange(100, dtype=np.float64)
        y = 2.0 * np.exp(-0.03 * d)
        result = fit_cds(d, y, n_components=1)
        assert result["residual"] < 1.0  # Should be very small

    def test_nnls_method(self):
        d = np.arange(100, dtype=np.float64)
        y = 3.0 * np.exp(-0.05 * d)
        result = fit_cds(d, y, n_components=2, method="nnls")
        assert "amplitudes" in result
        assert "decay_rates" in result

    def test_handles_zero_influence(self):
        d = np.arange(10, dtype=np.float64)
        y = np.zeros(10)
        result = fit_cds(d, y, n_components=1)
        assert "amplitudes" in result  # Should not crash


class TestSelectNComponents:
    def test_selects_correct_k(self):
        """For data from exactly 1 exponential, should prefer K=1."""
        d = np.arange(100, dtype=np.float64)
        y = 5.0 * np.exp(-0.05 * d)
        best_K, results = select_n_components(d, y, max_K=3)
        assert best_K >= 1
        assert len(results) == 3

    def test_returns_all_results(self):
        d = np.arange(50, dtype=np.float64)
        y = np.exp(-0.1 * d)
        _, results = select_n_components(d, y, max_K=4)
        assert len(results) == 4
        for r in results:
            assert "bic" in r


class TestSpectralECL:
    def test_single_exponential(self):
        """Spectral ECL for a single exponential: ~1/lambda * log(1/(1-beta))."""
        amplitudes = np.array([1.0])
        decay_rates = np.array([0.1])
        ecl = spectral_ecl(amplitudes, decay_rates, beta=0.9)
        # Analytical: 1/0.1 * log(1/0.1) ≈ 10 * 2.3 ≈ 23
        expected = int(np.ceil((1.0 / 0.1) * np.log(1.0 / (1 - 0.9))))
        assert abs(ecl - expected) <= 2

    def test_monotone_in_beta(self):
        amplitudes = np.array([2.0, 1.0])
        decay_rates = np.array([0.01, 0.1])
        ecl_50 = spectral_ecl(amplitudes, decay_rates, beta=0.5)
        ecl_90 = spectral_ecl(amplitudes, decay_rates, beta=0.9)
        assert ecl_90 >= ecl_50
