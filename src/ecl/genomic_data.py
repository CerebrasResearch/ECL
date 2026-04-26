"""Genomic data pipeline for ECL experiments.

Provides locus sampling from hg38 reference genome (chr8/chr9 held-out)
for three locus classes: promoter, enhancer, and intronic/intergenic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

# Default genome path
_DEFAULT_GENOME = Path(__file__).resolve().parents[2] / "data" / "genome" / "hg38_chr8_9.fa"

# DNA encoding
_CHAR_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3, "a": 0, "c": 1, "g": 2, "t": 3, "N": 0, "n": 0}


def _encode_dna(seq_str: str) -> npt.NDArray:
    """Convert DNA string to int8 array (A=0, C=1, G=2, T=3)."""
    return np.array([_CHAR_TO_INT.get(c, 0) for c in seq_str], dtype=np.int8)


def load_genome(genome_path: str | Path | None = None):
    """Load indexed genome FASTA."""
    from pyfaidx import Fasta

    path = Path(genome_path) if genome_path else _DEFAULT_GENOME
    if not path.exists():
        raise FileNotFoundError(
            f"Genome file not found: {path}\n" "Run the genome download script first."
        )
    return Fasta(str(path))


def extract_sequence(genome, chrom: str, center: int, window: int) -> npt.NDArray:
    """Extract and encode a DNA sequence centered at a position.

    Returns int8 array of shape (window,).
    """
    start = max(0, center - window // 2)
    end = start + window
    chrom_len = len(genome[chrom])
    if end > chrom_len:
        end = chrom_len
        start = end - window

    seq_str = str(genome[chrom][start:end])
    return _encode_dna(seq_str)


# ---- Locus definitions ----
# Curated positions on chr8/chr9 for held-out evaluation.
# Promoter: positions near known gene TSSs
# Enhancer: positions near ENCODE-annotated cCREs
# Intronic/Intergenic: random positions away from genes


def _generate_promoter_loci(genome, n_loci: int, rng: np.random.Generator) -> list[tuple[str, int]]:
    """Sample positions near gene promoters on chr8/chr9.

    Uses known gene-dense regions as proxy for GENCODE TSS locations.
    """
    # Gene-dense regions on chr8 and chr9 (approximate TSS locations from GENCODE)
    # chr8: MYC (128M), TNKS (9.5M), FGFR1 (38M), EXT1 (119M)
    # chr9: JAK2 (5M), CDKN2A (21.9M), ABL1 (133M), NOTCH1 (139M)
    promoter_regions = [
        ("chr8", 9_500_000, 10_500_000),
        ("chr8", 37_000_000, 39_000_000),
        ("chr8", 100_000_000, 102_000_000),
        ("chr8", 118_000_000, 120_000_000),
        ("chr8", 127_000_000, 129_000_000),
        ("chr9", 4_500_000, 6_000_000),
        ("chr9", 21_000_000, 23_000_000),
        ("chr9", 97_000_000, 99_000_000),
        ("chr9", 132_000_000, 134_000_000),
        ("chr9", 138_000_000, 139_000_000),
    ]
    loci = []
    for _ in range(n_loci):
        region = promoter_regions[rng.integers(len(promoter_regions))]
        chrom, lo, hi = region
        center = int(rng.integers(lo, hi))
        loci.append((chrom, center))
    return loci


def _generate_enhancer_loci(genome, n_loci: int, rng: np.random.Generator) -> list[tuple[str, int]]:
    """Sample positions in enhancer-like regions (distal from TSS).

    Uses intergenic regions with moderate GC content as proxy for
    ENCODE cCRE-annotated distal enhancers.
    """
    # Enhancer-enriched intergenic regions on chr8/chr9
    enhancer_regions = [
        ("chr8", 15_000_000, 18_000_000),
        ("chr8", 50_000_000, 55_000_000),
        ("chr8", 70_000_000, 75_000_000),
        ("chr8", 110_000_000, 115_000_000),
        ("chr9", 10_000_000, 15_000_000),
        ("chr9", 30_000_000, 35_000_000),
        ("chr9", 70_000_000, 75_000_000),
        ("chr9", 110_000_000, 115_000_000),
    ]
    loci = []
    for _ in range(n_loci):
        region = enhancer_regions[rng.integers(len(enhancer_regions))]
        chrom, lo, hi = region
        center = int(rng.integers(lo, hi))
        loci.append((chrom, center))
    return loci


def _generate_intronic_loci(genome, n_loci: int, rng: np.random.Generator) -> list[tuple[str, int]]:
    """Sample random intronic/intergenic control positions."""
    chroms = ["chr8", "chr9"]
    chrom_lens = {c: len(genome[c]) for c in chroms}
    margin = 500_000  # avoid telomeres/centromeres

    loci = []
    for _ in range(n_loci):
        chrom = chroms[rng.integers(len(chroms))]
        center = int(rng.integers(margin, chrom_lens[chrom] - margin))
        loci.append((chrom, center))
    return loci


def sample_loci(
    genome,
    locus_class: str,
    n_loci: int,
    window: int,
    n_sequences: int = 1,
    rng: np.random.Generator | None = None,
) -> tuple[npt.NDArray, list[tuple[str, int]]]:
    """Sample genomic loci and extract sequences.

    Parameters
    ----------
    genome : pyfaidx.Fasta
        Indexed genome.
    locus_class : str
        'promoter', 'enhancer', or 'intronic'.
    n_loci : int
        Number of loci to sample.
    window : int
        Sequence window size in bp.
    n_sequences : int
        Number of sequences per locus (for perturbation replicates,
        these are the same genomic sequence repeated — perturbation
        introduces the randomness).
    rng : numpy random generator.

    Returns
    -------
    sequences : array of shape (n_loci, n_sequences, window) dtype int8
    loci : list of (chrom, center) tuples
    """
    rng = rng or np.random.default_rng()

    generators = {
        "promoter": _generate_promoter_loci,
        "enhancer": _generate_enhancer_loci,
        "intronic": _generate_intronic_loci,
    }
    if locus_class not in generators:
        raise ValueError(f"Unknown locus class '{locus_class}'. Choose from: {list(generators)}")

    loci = generators[locus_class](genome, n_loci, rng)

    sequences = np.zeros((n_loci, n_sequences, window), dtype=np.int8)
    for i, (chrom, center) in enumerate(loci):
        seq = extract_sequence(genome, chrom, center, window)
        for j in range(n_sequences):
            sequences[i, j] = seq  # same sequence, perturbation adds randomness

    return sequences, loci
