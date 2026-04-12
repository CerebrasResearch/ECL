"""Tests for ecl.models module."""

from __future__ import annotations

import numpy as np

from ecl.models.base import AdditiveModel, LocalModel, SyntheticModel


class TestSyntheticModel:
    def test_output_shape(self, rng):
        model = SyntheticModel(seq_length=100, embed_dim=32)
        seq = rng.integers(0, 4, size=100).astype(np.int8)
        emb = model(seq)
        assert emb.shape == (32,)

    def test_deterministic(self, rng):
        model = SyntheticModel(seq_length=100, embed_dim=32, noise_std=0.0)
        seq = rng.integers(0, 4, size=100).astype(np.int8)
        emb1 = model(seq)
        emb2 = model(seq)
        np.testing.assert_array_equal(emb1, emb2)

    def test_different_sequences_different_embeddings(self, rng):
        model = SyntheticModel(seq_length=100, embed_dim=32)
        seq1 = rng.integers(0, 4, size=100).astype(np.int8)
        seq2 = rng.integers(0, 4, size=100).astype(np.int8)
        emb1 = model(seq1)
        emb2 = model(seq2)
        assert not np.allclose(emb1, emb2)

    def test_nominal_context(self):
        model = SyntheticModel(seq_length=500)
        assert model.nominal_context == 500

    def test_embedding_dim(self):
        model = SyntheticModel(embed_dim=64)
        assert model.embedding_dim == 64

    def test_token_to_bp_identity(self):
        model = SyntheticModel()
        assert model.token_to_bp(42) == 42

    def test_nearby_perturbation_larger_effect(self, rng):
        """Perturbing near reference should change embedding more than far."""
        model = SyntheticModel(seq_length=200, embed_dim=32, decay_length=20)
        seq = rng.integers(0, 4, size=200).astype(np.int8)
        ref = 100
        emb_orig = model(seq)

        # Perturb near
        seq_near = seq.copy()
        seq_near[ref + 1] = (seq[ref + 1] + 1) % 4
        emb_near = model(seq_near)
        diff_near = np.sum((emb_orig - emb_near) ** 2)

        # Perturb far
        seq_far = seq.copy()
        seq_far[ref + 90] = (seq[ref + 90] + 1) % 4
        emb_far = model(seq_far)
        diff_far = np.sum((emb_orig - emb_far) ** 2)

        assert diff_near > diff_far


class TestLocalModel:
    def test_output_shape(self, rng):
        model = LocalModel(seq_length=200, embed_dim=32, receptive_field=20)
        seq = rng.integers(0, 4, size=200).astype(np.int8)
        emb = model(seq)
        assert emb.shape == (32,)

    def test_outside_rf_no_effect(self, rng):
        """Perturbing outside receptive field should not change embedding."""
        model = LocalModel(seq_length=200, embed_dim=32, receptive_field=20)
        seq = rng.integers(0, 4, size=200).astype(np.int8)
        emb_orig = model(seq)

        # Perturb outside RF
        seq_pert = seq.copy()
        seq_pert[0] = (seq[0] + 1) % 4  # Position 0, far from center 100
        emb_pert = model(seq_pert)
        np.testing.assert_array_equal(emb_orig, emb_pert)

    def test_inside_rf_has_effect(self, rng):
        """Perturbing inside receptive field should change embedding."""
        model = LocalModel(seq_length=200, embed_dim=32, receptive_field=20)
        seq = rng.integers(0, 4, size=200).astype(np.int8)
        emb_orig = model(seq)

        # Perturb inside RF
        seq_pert = seq.copy()
        seq_pert[100] = (seq[100] + 1) % 4
        emb_pert = model(seq_pert)
        assert not np.allclose(emb_orig, emb_pert)


class TestAdditiveModel:
    def test_output_shape(self, rng):
        model = AdditiveModel(seq_length=100, embed_dim=16)
        seq = rng.integers(0, 4, size=100).astype(np.int8)
        emb = model(seq)
        assert emb.shape == (16,)

    def test_additivity(self, rng):
        """For additive model, single-position perturbation should give independent effects."""
        model = AdditiveModel(seq_length=50, embed_dim=8, decay_length=10)
        seq = rng.integers(0, 4, size=50).astype(np.int8)
        emb_orig = model(seq)

        # Perturb positions 10 and 20 separately and jointly
        seq_10 = seq.copy()
        seq_10[10] = (seq[10] + 1) % 4
        delta_10 = model(seq_10) - emb_orig

        seq_20 = seq.copy()
        seq_20[20] = (seq[20] + 1) % 4
        delta_20 = model(seq_20) - emb_orig

        seq_both = seq.copy()
        seq_both[10] = (seq[10] + 1) % 4
        seq_both[20] = (seq[20] + 1) % 4
        delta_both = model(seq_both) - emb_orig

        # Additivity: delta_both ≈ delta_10 + delta_20
        np.testing.assert_allclose(delta_both, delta_10 + delta_20, atol=1e-10)
