"""Tests for ecl.metrics module."""

from __future__ import annotations

import numpy as np
import pytest

from ecl.metrics import (
    cosine_distance,
    estimate_precision,
    get_metric,
    mahalanobis_distance,
    squared_euclidean,
    task_weighted_distance,
)


class TestSquaredEuclidean:
    def test_identical_vectors(self):
        z = np.array([1.0, 2.0, 3.0])
        assert squared_euclidean(z, z) == pytest.approx(0.0)

    def test_known_value(self):
        z1 = np.array([1.0, 0.0])
        z2 = np.array([0.0, 1.0])
        assert squared_euclidean(z1, z2) == pytest.approx(2.0)

    def test_batch(self):
        z1 = np.array([[1.0, 0.0], [0.0, 0.0]])
        z2 = np.array([[0.0, 1.0], [3.0, 4.0]])
        result = squared_euclidean(z1, z2)
        np.testing.assert_allclose(result, [2.0, 25.0])

    def test_symmetry(self, rng):
        z1 = rng.standard_normal(10)
        z2 = rng.standard_normal(10)
        assert squared_euclidean(z1, z2) == pytest.approx(squared_euclidean(z2, z1))


class TestCosineDistance:
    def test_identical_vectors(self):
        z = np.array([1.0, 2.0, 3.0])
        assert cosine_distance(z, z) == pytest.approx(0.0, abs=1e-10)

    def test_orthogonal_vectors(self):
        z1 = np.array([1.0, 0.0])
        z2 = np.array([0.0, 1.0])
        assert cosine_distance(z1, z2) == pytest.approx(1.0)

    def test_opposite_vectors(self):
        z1 = np.array([1.0, 0.0])
        z2 = np.array([-1.0, 0.0])
        assert cosine_distance(z1, z2) == pytest.approx(2.0)

    def test_range(self, rng):
        z1 = rng.standard_normal(10)
        z2 = rng.standard_normal(10)
        d = cosine_distance(z1, z2)
        assert 0.0 <= d <= 2.0 + 1e-10


class TestMahalanobisDistance:
    def test_identity_precision(self):
        """With identity precision, Mahalanobis = squared Euclidean."""
        z1 = np.array([1.0, 2.0])
        z2 = np.array([3.0, 4.0])
        eye = np.eye(2)
        assert mahalanobis_distance(z1, z2, eye) == pytest.approx(squared_euclidean(z1, z2))

    def test_scaled_precision(self):
        z1 = np.array([1.0, 0.0])
        z2 = np.array([0.0, 0.0])
        P = np.diag([4.0, 1.0])  # Scale first dimension by 4
        assert mahalanobis_distance(z1, z2, P) == pytest.approx(4.0)


class TestTaskWeightedDistance:
    def test_identity_weight(self):
        z1 = np.array([1.0, 2.0])
        z2 = np.array([3.0, 4.0])
        W = np.eye(2)
        assert task_weighted_distance(z1, z2, W) == pytest.approx(squared_euclidean(z1, z2))


class TestEstimatePrecision:
    def test_shape(self, rng):
        embeddings = rng.standard_normal((100, 5))
        P = estimate_precision(embeddings)
        assert P.shape == (5, 5)

    def test_symmetry(self, rng):
        embeddings = rng.standard_normal((100, 5))
        P = estimate_precision(embeddings)
        np.testing.assert_allclose(P, P.T, atol=1e-10)


class TestRegistry:
    def test_get_known_metric(self):
        assert get_metric("squared_euclidean") is squared_euclidean
        assert get_metric("cosine") is cosine_distance

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            get_metric("nonexistent")
