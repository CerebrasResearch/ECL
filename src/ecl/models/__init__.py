"""Model wrappers for ECL estimation.

Each wrapper implements the interface defined in BaseGenomicModel:
  - forward(sequence) -> embedding
  - nominal_context: int (bp)
  - embedding_dim: int
  - token_to_bp(token_idx) -> bp_position
"""

from ecl.models.base import BaseGenomicModel, SyntheticModel

__all__ = ["BaseGenomicModel", "SyntheticModel"]
