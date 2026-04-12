"""Tests for ecl.gniah (Genomic Needle-in-a-Haystack) module."""

from __future__ import annotations

import numpy as np

from ecl.gniah import (
    encode_motif,
    generate_neutral_background,
    gniah_sensitivity,
    insert_motif_at_distance,
)
from ecl.models.base import SyntheticModel


class TestEncodeMotif:
    def test_known_motif(self):
        encoded = encode_motif("ACGT")
        np.testing.assert_array_equal(encoded, [0, 1, 2, 3])

    def test_length_preserved(self):
        encoded = encode_motif("ACGTACGT")
        assert len(encoded) == 8

    def test_ambiguous_base(self, rng):
        encoded = encode_motif("ACNGT", rng)
        assert len(encoded) == 5
        assert 0 <= encoded[2] <= 3  # N resolved to a valid base


class TestGenerateNeutralBackground:
    def test_length(self, rng):
        bg = generate_neutral_background(1000, rng=rng)
        assert len(bg) == 1000

    def test_values_in_range(self, rng):
        bg = generate_neutral_background(1000, rng=rng)
        assert np.all((bg >= 0) & (bg <= 3))

    def test_gc_content(self, rng):
        bg = generate_neutral_background(100_000, gc_content=0.5, rng=rng)
        gc = np.mean((bg == 1) | (bg == 2))
        assert abs(gc - 0.5) < 0.02  # Should be close to 50%


class TestInsertMotifAtDistance:
    def test_motif_inserted(self, rng):
        seq = rng.integers(0, 4, size=1000).astype(np.int8)
        motif = np.array([0, 1, 2, 3], dtype=np.int8)
        result = insert_motif_at_distance(seq, center=500, distance=100, motif=motif)
        assert len(result) == len(seq)
        # Motif should be present somewhere around position 400
        start = 500 - 100 - 2
        assert np.array_equal(result[start : start + 4], motif)

    def test_preserves_length(self, rng):
        seq = rng.integers(0, 4, size=500).astype(np.int8)
        motif = encode_motif("ACGT")
        result = insert_motif_at_distance(seq, center=250, distance=50, motif=motif)
        assert len(result) == len(seq)

    def test_edge_handling(self, rng):
        seq = rng.integers(0, 4, size=100).astype(np.int8)
        motif = encode_motif("ACGTACGT")
        # Insert near edge should not crash
        result = insert_motif_at_distance(seq, center=5, distance=3, motif=motif)
        assert len(result) == len(seq)


class TestGNIAHSensitivity:
    def test_output_shape(self, rng):
        model = SyntheticModel(seq_length=500, embed_dim=16, decay_length=50)
        distances = np.array([10, 50, 100, 200])
        sens = gniah_sensitivity(
            model_fn=model,
            motif_name="GATA",
            distances=distances,
            seq_length=500,
            n_samples=3,
            rng=rng,
            show_progress=False,
        )
        assert sens.shape == (4,)

    def test_non_negative(self, rng):
        model = SyntheticModel(seq_length=500, embed_dim=16, decay_length=50)
        distances = np.array([10, 50, 100])
        sens = gniah_sensitivity(
            model_fn=model,
            motif_name="CTCF",
            distances=distances,
            seq_length=500,
            n_samples=3,
            rng=rng,
            show_progress=False,
        )
        assert np.all(sens >= 0)

    def test_custom_motif_string(self, rng):
        model = SyntheticModel(seq_length=200, embed_dim=8, decay_length=30)
        distances = np.array([10, 30])
        sens = gniah_sensitivity(
            model_fn=model,
            motif_name="AAACCCGGG",
            distances=distances,
            seq_length=200,
            n_samples=2,
            rng=rng,
            show_progress=False,
        )
        assert sens.shape == (2,)
