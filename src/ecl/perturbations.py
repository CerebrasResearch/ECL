"""Perturbation kernels for ECL estimation (Section 3.3 of the paper).

Implements four perturbation types:
  (i)   Random substitution (Pi^sub)
  (ii)  Dinucleotide shuffle (Pi^shuf)
  (iii) k-mer Markov resampling (Pi^Markov)
  (iv)  Generative infilling (Pi^gen) — stub for MLM-based infilling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

# Standard DNA alphabet
DNA_ALPHABET = np.array(list("ACGT"))
DNA_CHAR_TO_IDX = {c: i for i, c in enumerate(DNA_ALPHABET)}


def _str_to_idx(seq: str | npt.NDArray) -> npt.NDArray:
    """Convert a string DNA sequence to integer indices (A=0, C=1, G=2, T=3)."""
    if isinstance(seq, str):
        return np.array([DNA_CHAR_TO_IDX[c] for c in seq], dtype=np.int8)
    return np.asarray(seq, dtype=np.int8)


def _idx_to_str(idx: npt.NDArray) -> str:
    """Convert integer indices back to DNA string."""
    return "".join(DNA_ALPHABET[idx])


class PerturbationKernel(ABC):
    """Base class for perturbation kernels Pi_S(. | x)."""

    @abstractmethod
    def perturb(
        self,
        sequence: npt.NDArray,
        positions: npt.NDArray,
        rng: np.random.Generator | None = None,
    ) -> npt.NDArray:
        """Perturb positions in the sequence, preserving all other coordinates.

        Parameters
        ----------
        sequence : int array of shape (L,) with values in {0,1,2,3}.
        positions : int array of positions to perturb.
        rng : numpy random generator.

        Returns
        -------
        perturbed : int array of shape (L,), modified only at `positions`.
        """

    def __call__(
        self,
        sequence: npt.NDArray,
        positions: npt.NDArray,
        rng: np.random.Generator | None = None,
    ) -> npt.NDArray:
        return self.perturb(sequence, positions, rng)


class RandomSubstitution(PerturbationKernel):
    """Replace each nucleotide in S with a uniformly random *alternative*.

    Guarantees the perturbed nucleotide differs from the original (3 choices).
    """

    def perturb(
        self,
        sequence: npt.NDArray,
        positions: npt.NDArray,
        rng: np.random.Generator | None = None,
    ) -> npt.NDArray:
        rng = rng or np.random.default_rng()
        out = sequence.copy()
        positions = np.asarray(positions)
        if positions.size == 0:
            return out
        originals = out[positions]
        # For each position, pick from the 3 alternatives
        offsets = rng.integers(1, 4, size=len(positions))
        out[positions] = (originals + offsets) % 4
        return out


class DinucleotideShuffle(PerturbationKernel):
    """Shuffle positions in S preserving dinucleotide frequencies.

    Uses the Altschul-Erickson algorithm for the targeted region.
    If the region is very short (< 4 nt), falls back to random permutation.
    """

    def perturb(
        self,
        sequence: npt.NDArray,
        positions: npt.NDArray,
        rng: np.random.Generator | None = None,
    ) -> npt.NDArray:
        rng = rng or np.random.default_rng()
        out = sequence.copy()
        positions = np.sort(np.asarray(positions))
        if len(positions) < 4:
            # Too short for dinucleotide structure — random permutation
            out[positions] = rng.permutation(out[positions])
            return out

        # Extract contiguous sub-regions and shuffle each
        sub = out[positions].copy()
        shuffled = _dinucleotide_shuffle_array(sub, rng)
        out[positions] = shuffled
        return out


def _dinucleotide_shuffle_array(seq: npt.NDArray, rng: np.random.Generator) -> npt.NDArray:
    """Altschul-Erickson dinucleotide-preserving shuffle on integer array.

    Constructs a directed Eulerian graph from dinucleotide transitions and
    finds a random Eulerian path, preserving exact dinucleotide frequencies.
    """
    n = len(seq)
    if n <= 2:
        return seq.copy()

    # Build edge lists for each starting nucleotide
    edges: dict[int, list[int]] = {i: [] for i in range(4)}
    for i in range(n - 1):
        edges[int(seq[i])].append(int(seq[i + 1]))

    # Shuffle edge lists to randomize path
    for k in edges:
        rng.shuffle(edges[k])

    # Follow Eulerian path starting from seq[0]
    result = [int(seq[0])]
    idx = {k: 0 for k in range(4)}
    for _ in range(n - 1):
        cur = result[-1]
        if idx[cur] < len(edges[cur]):
            nxt = edges[cur][idx[cur]]
            idx[cur] += 1
            result.append(nxt)
        else:
            # Fallback: shouldn't happen with a valid Eulerian graph
            break

    out = np.array(result, dtype=seq.dtype)
    # Pad if path was short (edge case)
    if len(out) < n:
        out = np.concatenate([out, seq[len(out):]])
    return out


class KmerMarkov(PerturbationKernel):
    """Resample from a k-th order Markov model fitted to the local context.

    Fits transition probabilities from a flanking context window, then
    resamples positions in S auto-regressively.

    Parameters
    ----------
    k : int
        Markov order (default 3, i.e., 3-mer context).
    context_flank : int
        Number of flanking bp on each side used to fit the Markov model.
    """

    def __init__(self, k: int = 3, context_flank: int = 500):
        self.k = k
        self.context_flank = context_flank

    def perturb(
        self,
        sequence: npt.NDArray,
        positions: npt.NDArray,
        rng: np.random.Generator | None = None,
    ) -> npt.NDArray:
        rng = rng or np.random.default_rng()
        out = sequence.copy()
        positions = np.sort(np.asarray(positions))
        if len(positions) == 0:
            return out

        # Fit Markov model from flanking context
        lo = max(0, int(positions[0]) - self.context_flank)
        hi = min(len(sequence), int(positions[-1]) + self.context_flank + 1)
        context_seq = sequence[lo:hi]
        trans = self._fit_transitions(context_seq)

        # Resample each position auto-regressively
        for pos in positions:
            pos = int(pos)
            # Get preceding k-mer
            start = max(0, pos - self.k)
            kmer = tuple(out[start:pos])
            # Pad if near start
            while len(kmer) < self.k:
                kmer = (0,) + kmer
            probs = trans.get(kmer)
            if probs is None:
                # Fallback to uniform
                out[pos] = rng.integers(0, 4)
            else:
                out[pos] = rng.choice(4, p=probs)
        return out

    def _fit_transitions(self, seq: npt.NDArray) -> dict[tuple, npt.NDArray]:
        """Estimate k-th order transition probabilities with add-1 smoothing."""
        counts: dict[tuple, npt.NDArray] = {}
        for i in range(self.k, len(seq)):
            kmer = tuple(seq[i - self.k : i])
            if kmer not in counts:
                counts[kmer] = np.ones(4, dtype=np.float64)  # add-1 smoothing
            counts[kmer][int(seq[i])] += 1.0
        trans = {}
        for kmer, c in counts.items():
            trans[kmer] = c / c.sum()
        return trans


class GenerativeInfilling(PerturbationKernel):
    """Placeholder for generative (MLM-based) infilling perturbation.

    In a full implementation this would mask positions in S and infill
    from a pretrained masked language model. Here we fall back to
    k-mer Markov resampling as a lightweight proxy.
    """

    def __init__(self, k: int = 5, context_flank: int = 1000):
        self._proxy = KmerMarkov(k=k, context_flank=context_flank)

    def perturb(
        self,
        sequence: npt.NDArray,
        positions: npt.NDArray,
        rng: np.random.Generator | None = None,
    ) -> npt.NDArray:
        return self._proxy.perturb(sequence, positions, rng)


# Registry
PERTURBATION_KERNELS = {
    "substitution": RandomSubstitution,
    "shuffle": DinucleotideShuffle,
    "markov": KmerMarkov,
    "generative": GenerativeInfilling,
}


def get_perturbation(name: str = "substitution", **kwargs) -> PerturbationKernel:
    """Instantiate a perturbation kernel by name."""
    if name not in PERTURBATION_KERNELS:
        raise ValueError(
            f"Unknown perturbation '{name}'. Choose from: {list(PERTURBATION_KERNELS.keys())}"
        )
    return PERTURBATION_KERNELS[name](**kwargs)
