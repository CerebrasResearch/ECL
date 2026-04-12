"""Tests for ecl.ecl module — core ECL computation."""

from __future__ import annotations

import numpy as np
import pytest

from ecl.ecl import AECP, ECD, ECL, ECP, cumulative_influence, directional_ecl, normalized_influence


class TestCumulativeInfluence:
    def test_monotonically_increasing(self, sample_influence_profile):
        distances, influence = sample_influence_profile
        _, cumul = cumulative_influence(distances, influence)
        assert np.all(np.diff(cumul) >= 0)

    def test_starts_at_influence_zero(self, sample_influence_profile):
        distances, influence = sample_influence_profile
        _, cumul = cumulative_influence(distances, influence)
        # First value should be influence at d=0 (multiplicity 1)
        assert cumul[0] == pytest.approx(influence[0])

    def test_with_boundary_correction(self):
        distances = np.arange(10)
        influence = np.ones(10)
        _, cumul = cumulative_influence(distances, influence, reference=5, L=20)
        # d=0: mult=1, d=1..5: mult=2, d=6..9: mult=2 (if within bounds)
        assert cumul[-1] > cumul[0]


class TestNormalizedInfluence:
    def test_sums_to_one(self, sample_influence_profile):
        distances, influence = sample_influence_profile
        norm = normalized_influence(distances, influence)
        assert np.sum(norm) == pytest.approx(1.0, abs=1e-10)

    def test_non_negative(self, sample_influence_profile):
        distances, influence = sample_influence_profile
        norm = normalized_influence(distances, influence)
        assert np.all(norm >= 0)


class TestECL:
    def test_monotone_in_beta(self, sample_influence_profile):
        """ECL_beta should be non-decreasing in beta (Proposition 6.1)."""
        distances, influence = sample_influence_profile
        betas = [0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
        ecls = [ECL(distances, influence, beta=b) for b in betas]
        for i in range(len(ecls) - 1):
            assert ecls[i] <= ecls[i + 1]

    def test_beta_one_returns_max(self, sample_influence_profile):
        distances, influence = sample_influence_profile
        ecl = ECL(distances, influence, beta=1.0)
        # Should return the max distance where influence > 0
        assert ecl <= int(distances[-1])

    def test_small_beta_small_ecl(self):
        """For concentrated influence, small beta -> small ECL."""
        distances = np.arange(100)
        influence = np.zeros(100)
        influence[0] = 100.0  # All influence at d=0
        ecl = ECL(distances, influence, beta=0.5)
        assert ecl == 0

    def test_uniform_influence(self):
        """For uniform influence, ECL should scale with beta."""
        distances = np.arange(100)
        influence = np.ones(100)
        ecl_50 = ECL(distances, influence, beta=0.5)
        ecl_90 = ECL(distances, influence, beta=0.9)
        assert ecl_50 < ecl_90

    def test_zero_influence(self):
        distances = np.arange(10)
        influence = np.zeros(10)
        assert ECL(distances, influence, beta=0.9) == 0


class TestECP:
    def test_non_decreasing(self, sample_influence_profile):
        distances, influence = sample_influence_profile
        betas, ecl_values = ECP(distances, influence)
        assert np.all(np.diff(ecl_values) >= 0)

    def test_output_shape(self, sample_influence_profile):
        distances, influence = sample_influence_profile
        betas, ecl_values = ECP(distances, influence, betas=np.linspace(0.1, 1.0, 50))
        assert len(betas) == 50
        assert len(ecl_values) == 50


class TestAECP:
    def test_zero_for_concentrated(self):
        """AECP should be 0 when all influence is at d=0."""
        distances = np.arange(50)
        influence = np.zeros(50)
        influence[0] = 1.0
        assert AECP(distances, influence) == pytest.approx(0.0, abs=1e-10)

    def test_positive_for_spread(self, sample_influence_profile):
        distances, influence = sample_influence_profile
        aecp = AECP(distances, influence)
        assert aecp > 0

    def test_equals_mean_distance(self):
        """AECP = sum_i |i-r| * I_bar(i;r) (Proposition 5.1)."""
        distances = np.arange(5)
        influence = np.array([0.4, 0.3, 0.2, 0.1, 0.0])
        # Normalized: [0.4, 0.6, 0.4, 0.2, 0] / 1.6 with multiplicity [1,2,2,2,2]
        aecp = AECP(distances, influence)
        assert aecp > 0


class TestECD:
    def test_one_for_single_position(self):
        """ECD = 1 when all influence is at one position."""
        distances = np.arange(50)
        influence = np.zeros(50)
        influence[5] = 1.0
        ecd = ECD(distances, influence)
        assert ecd == pytest.approx(1.0, abs=0.01)

    def test_bounded_by_L(self, sample_influence_profile):
        distances, influence = sample_influence_profile
        ecd = ECD(distances, influence)
        assert 1.0 <= ecd <= len(distances)

    def test_uniform_maximizes_ecd(self):
        """Uniform influence should give maximum ECD."""
        distances = np.arange(20)
        uniform_infl = np.ones(20)
        concentrated_infl = np.zeros(20)
        concentrated_infl[:3] = 1.0
        ecd_uniform = ECD(distances, uniform_infl)
        ecd_concentrated = ECD(distances, concentrated_infl)
        assert ecd_uniform > ecd_concentrated


class TestDirectionalECL:
    def test_symmetric_profiles(self):
        """Symmetric profiles should give equal directional ECLs."""
        distances = np.arange(1, 51)
        influence = np.exp(-distances / 10.0)
        ecl_up, ecl_down, ratio = directional_ecl(distances, influence, influence, beta=0.9)
        assert ecl_up == ecl_down
        assert ratio == pytest.approx(1.0)

    def test_asymmetric_profiles(self):
        distances = np.arange(1, 51)
        upstream = np.exp(-distances / 5.0)  # Fast decay
        downstream = np.exp(-distances / 20.0)  # Slow decay
        ecl_up, ecl_down, ratio = directional_ecl(distances, upstream, downstream, beta=0.9)
        assert ecl_down > ecl_up
        assert ratio > 1.0
