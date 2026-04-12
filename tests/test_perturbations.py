"""Tests for ecl.perturbations module."""

from __future__ import annotations

import numpy as np
import pytest

from ecl.perturbations import (
    DinucleotideShuffle,
    GenerativeInfilling,
    KmerMarkov,
    RandomSubstitution,
    get_perturbation,
)


class TestRandomSubstitution:
    def test_preserves_non_target_positions(self, random_sequence, rng):
        kernel = RandomSubstitution()
        positions = np.array([10, 20, 30])
        result = kernel(random_sequence, positions, rng)
        # Non-target positions unchanged
        mask = np.ones(len(random_sequence), dtype=bool)
        mask[positions] = False
        np.testing.assert_array_equal(result[mask], random_sequence[mask])

    def test_modifies_target_positions(self, random_sequence, rng):
        kernel = RandomSubstitution()
        positions = np.array([10, 20, 30])
        result = kernel(random_sequence, positions, rng)
        # Target positions differ from original
        assert np.all(result[positions] != random_sequence[positions])

    def test_values_in_range(self, random_sequence, rng):
        kernel = RandomSubstitution()
        positions = np.arange(len(random_sequence))
        result = kernel(random_sequence, positions, rng)
        assert np.all((result >= 0) & (result <= 3))

    def test_empty_positions(self, random_sequence, rng):
        kernel = RandomSubstitution()
        result = kernel(random_sequence, np.array([], dtype=int), rng)
        np.testing.assert_array_equal(result, random_sequence)

    def test_copy_semantics(self, random_sequence, rng):
        kernel = RandomSubstitution()
        original = random_sequence.copy()
        kernel(random_sequence, np.array([5]), rng)
        # Original should not be modified
        np.testing.assert_array_equal(random_sequence, original)


class TestDinucleotideShuffle:
    def test_preserves_non_target(self, random_sequence, rng):
        kernel = DinucleotideShuffle()
        positions = np.arange(50, 100)
        result = kernel(random_sequence, positions, rng)
        np.testing.assert_array_equal(result[:50], random_sequence[:50])
        np.testing.assert_array_equal(result[100:], random_sequence[100:])

    def test_preserves_length_and_range(self, rng):
        """Shuffled region should have valid nucleotides and correct length."""
        seq = rng.integers(0, 4, size=200).astype(np.int8)
        kernel = DinucleotideShuffle()
        positions = np.arange(50, 150)
        result = kernel(seq, positions, rng)
        assert len(result) == len(seq)
        assert np.all((result[positions] >= 0) & (result[positions] <= 3))

    def test_short_region_fallback(self, random_sequence, rng):
        kernel = DinucleotideShuffle()
        positions = np.array([10, 11])
        result = kernel(random_sequence, positions, rng)
        assert result.shape == random_sequence.shape


class TestKmerMarkov:
    def test_preserves_non_target(self, random_sequence, rng):
        kernel = KmerMarkov(k=2, context_flank=50)
        positions = np.array([100, 101, 102])
        result = kernel(random_sequence, positions, rng)
        mask = np.ones(len(random_sequence), dtype=bool)
        mask[positions] = False
        np.testing.assert_array_equal(result[mask], random_sequence[mask])

    def test_values_in_range(self, random_sequence, rng):
        kernel = KmerMarkov(k=3)
        positions = np.arange(50, 60)
        result = kernel(random_sequence, positions, rng)
        assert np.all((result >= 0) & (result <= 3))


class TestGenerativeInfilling:
    def test_preserves_non_target(self, random_sequence, rng):
        kernel = GenerativeInfilling()
        positions = np.array([50, 51, 52])
        result = kernel(random_sequence, positions, rng)
        mask = np.ones(len(random_sequence), dtype=bool)
        mask[positions] = False
        np.testing.assert_array_equal(result[mask], random_sequence[mask])


class TestRegistry:
    def test_get_known_perturbation(self):
        for name in ["substitution", "shuffle", "markov", "generative"]:
            kernel = get_perturbation(name)
            assert kernel is not None

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown perturbation"):
            get_perturbation("nonexistent")
