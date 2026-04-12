"""Evo 2 model wrapper (Appendix G of the paper).

Evo 2: 40B parameter StripedHyena + Transformer hybrid, up to 1 Mb inputs.
Autoregressive (next-token prediction) — causal masking means only i < r
can influence embedding at r.

Requires multi-GPU inference (>= 4x A100 80GB).
"""

from __future__ import annotations

import numpy.typing as npt

from ecl.models.base import BaseGenomicModel


class Evo2Wrapper(BaseGenomicModel):
    """Wrapper for the Evo 2 model.

    Parameters
    ----------
    model_name : str
        Model identifier for Evo 2.
    device : str
        Torch device or 'auto' for multi-GPU.
    reference_position : int or None
        Position from which to extract the hidden state.
    """

    NOMINAL_CONTEXT = 1_000_000

    def __init__(
        self,
        model_name: str = "arcinstitute/evo2_40b",
        device: str = "auto",
        reference_position: int | None = None,
    ):
        self._model_name = model_name
        self._device = device
        self._reference_position = reference_position
        self._model = None
        self._embed_dim = 6144  # Evo 2 hidden dimension

    def _load_model(self):
        if self._model is not None:
            return
        raise NotImplementedError(
            "Evo 2 (40B) requires multi-GPU setup. "
            "See https://github.com/arcinstitute/evo2 for setup instructions. "
            "Use Evo2Wrapper.from_loaded(model) with a pre-loaded model."
        )

    @classmethod
    def from_loaded(cls, model, device: str = "auto", reference_position: int | None = None):
        """Create wrapper from a pre-loaded Evo 2 model."""
        wrapper = cls(device=device, reference_position=reference_position)
        wrapper._model = model
        return wrapper

    def forward(self, sequence: npt.NDArray) -> npt.NDArray:
        self._load_model()
        raise NotImplementedError("Full Evo 2 inference requires model weights and multi-GPU.")

    @property
    def nominal_context(self) -> int:
        return self.NOMINAL_CONTEXT

    @property
    def embedding_dim(self) -> int:
        return self._embed_dim

    def token_to_bp(self, token_idx: int) -> int:
        return token_idx  # character-level
