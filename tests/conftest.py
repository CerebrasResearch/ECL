"""Shared fixtures for ECL test suite."""

from __future__ import annotations

import numpy as np
import pytest

from ecl.models.base import AdditiveModel, LocalModel, SyntheticModel


@pytest.fixture
def rng():
    """Deterministic random generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def random_sequence(rng):
    """A random DNA sequence of length 200."""
    return rng.integers(0, 4, size=200).astype(np.int8)


@pytest.fixture
def short_sequence(rng):
    """A short DNA sequence of length 50."""
    return rng.integers(0, 4, size=50).astype(np.int8)


@pytest.fixture
def synthetic_model():
    """SyntheticModel with decay_length=30, seq_length=200."""
    return SyntheticModel(seq_length=200, embed_dim=32, decay_length=30.0)


@pytest.fixture
def local_model():
    """LocalModel with receptive_field=20, seq_length=200."""
    return LocalModel(seq_length=200, embed_dim=32, receptive_field=20)


@pytest.fixture
def additive_model():
    """AdditiveModel with decay_length=20, seq_length=100."""
    return AdditiveModel(seq_length=100, embed_dim=16, decay_length=20.0)


@pytest.fixture
def sample_sequences(rng):
    """Batch of 10 random sequences of length 200."""
    return rng.integers(0, 4, size=(10, 200)).astype(np.int8)


@pytest.fixture
def sample_influence_profile():
    """A synthetic exponentially-decaying influence profile."""
    distances = np.arange(100)
    influence = 5.0 * np.exp(-distances / 20.0)
    influence += np.random.default_rng(0).normal(0, 0.01, size=100).clip(0)
    return distances, influence
