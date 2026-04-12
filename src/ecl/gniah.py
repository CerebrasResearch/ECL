"""Genomic Needle-in-a-Haystack (gNIAH) protocol — Section 5.10 of the paper.

Insert a known regulatory motif at varying distances from the prediction center
in a neutral (shuffled) background and measure the embedding change.

    gNIAH(d, m) = E[d_Z(f(X_neutral), f(X_neutral^{+m@d}))]
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from ecl.metrics import squared_euclidean

# Common regulatory motifs (consensus sequences)
MOTIFS = {
    "CTCF": "CCGCGNGGNGGCAG",  # CTCF binding consensus (N = any)
    "GATA": "AGATAAGG",  # GATA factor consensus
    "SP1": "GGGCGG",  # SP1 GC-box
    "TATA": "TATAAAA",  # TATA box
    "CAAT": "CCAAT",  # CAAT box
}

# Integer-encoded versions (N -> random)
DNA_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}


def encode_motif(motif: str, rng: np.random.Generator | None = None) -> npt.NDArray:
    """Encode a motif string to integer array, resolving ambiguous bases randomly."""
    rng = rng or np.random.default_rng()
    encoded = []
    for c in motif.upper():
        if c in DNA_MAP:
            encoded.append(DNA_MAP[c])
        else:
            encoded.append(rng.integers(0, 4))
    return np.array(encoded, dtype=np.int8)


def generate_neutral_background(
    length: int,
    gc_content: float = 0.42,
    rng: np.random.Generator | None = None,
) -> npt.NDArray:
    """Generate a neutral background sequence with specified GC content.

    Optionally applies dinucleotide shuffle to make it more realistic.
    """
    rng = rng or np.random.default_rng()
    # Probabilities: A, C, G, T
    p_gc = gc_content / 2.0
    p_at = (1.0 - gc_content) / 2.0
    probs = [p_at, p_gc, p_gc, p_at]
    return rng.choice(4, size=length, p=probs).astype(np.int8)


def insert_motif_at_distance(
    sequence: npt.NDArray,
    center: int,
    distance: int,
    motif: npt.NDArray,
    upstream: bool = True,
) -> npt.NDArray:
    """Insert a motif at a given distance from center.

    Parameters
    ----------
    sequence : int array of shape (L,).
    center : reference position.
    distance : distance from center in bp.
    motif : int array of motif.
    upstream : if True, insert upstream (left); otherwise downstream (right).

    Returns
    -------
    modified : sequence with motif inserted (replacing existing bases).
    """
    out = sequence.copy()
    m_len = len(motif)
    start = center - distance - m_len // 2 if upstream else center + distance - m_len // 2

    start = max(0, min(start, len(out) - m_len))
    out[start : start + m_len] = motif
    return out


def gniah_sensitivity(
    model_fn: Callable[[npt.NDArray], npt.NDArray],
    motif_name: str,
    distances: npt.NDArray,
    seq_length: int = 196_608,
    center: int | None = None,
    n_samples: int = 50,
    metric: Callable = squared_euclidean,
    rng: np.random.Generator | None = None,
    show_progress: bool = True,
) -> npt.NDArray:
    """Compute gNIAH sensitivity profile [Eq. 12].

    Parameters
    ----------
    model_fn : callable
        Maps integer sequence (L,) -> embedding (d,).
    motif_name : str
        Key in MOTIFS dict, or a raw DNA string.
    distances : array of int
        Distances (bp) at which to test motif insertion.
    seq_length : int
        Length of the neutral background sequence.
    center : int or None
        Reference position. Defaults to seq_length // 2.
    n_samples : int
        Number of neutral backgrounds to average over.
    metric : callable
        Embedding discrepancy.
    rng : random generator.

    Returns
    -------
    sensitivities : array of shape (len(distances),)
        gNIAH(d, m) for each distance d.
    """
    from tqdm import tqdm

    rng = rng or np.random.default_rng()
    center = center or seq_length // 2

    # Resolve motif
    motif_str = MOTIFS.get(motif_name, motif_name)
    motif = encode_motif(motif_str, rng)

    distances = np.asarray(distances)
    sensitivities = np.zeros(len(distances), dtype=np.float64)

    dist_iter = tqdm(range(len(distances)), desc="gNIAH", disable=not show_progress)
    for di in dist_iter:
        d = int(distances[di])
        accum = 0.0
        for _ in range(n_samples):
            neutral = generate_neutral_background(seq_length, rng=rng)
            z_neutral = model_fn(neutral)
            # Insert motif (try both upstream and downstream, average)
            seq_with_motif = insert_motif_at_distance(neutral, center, d, motif, upstream=True)
            z_with = model_fn(seq_with_motif)
            accum += float(metric(z_neutral, z_with))

        sensitivities[di] = accum / n_samples

    return sensitivities
