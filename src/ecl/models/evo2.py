"""Evo 2 model wrapper (Appendix G of the paper).

Evo 2 family: StripedHyena + Transformer hybrid, up to 1 Mb inputs.
Autoregressive (next-token prediction) — causal masking means only i < r
can influence embedding at r.

Variants:
  - evo2_7b:  7B params, 4096-dim hidden (single A100 80GB)
  - evo2_40b: 40B params, 6144-dim hidden (>= 4x A100 80GB)
"""

from __future__ import annotations

import numpy.typing as npt

from ecl.models.base import BaseGenomicModel

_EVO2_MODELS = {
    "7b": {
        "hf_name": "arcinstitute/evo2_7b",
        "embed_dim": 4096,
    },
    "40b": {
        "hf_name": "arcinstitute/evo2_40b",
        "embed_dim": 6144,
    },
}


class Evo2Wrapper(BaseGenomicModel):
    """Wrapper for the Evo 2 model family.

    Parameters
    ----------
    scale : str
        Model scale: '7b' or '40b'.
    device : str
        Torch device or 'auto' for multi-GPU.
    reference_position : int or None
        Position from which to extract the hidden state.
    """

    NOMINAL_CONTEXT = 1_000_000

    def __init__(
        self,
        scale: str = "7b",
        device: str = "auto",
        reference_position: int | None = None,
    ):
        if scale not in _EVO2_MODELS:
            raise ValueError(f"Unknown scale '{scale}'. Choose from: {list(_EVO2_MODELS)}")
        cfg = _EVO2_MODELS[scale]
        self._model_name = cfg["hf_name"]
        self._device = device
        self._reference_position = reference_position
        self._model = None
        self._embed_dim = cfg["embed_dim"]
        self._scale = scale

    def _load_model(self):
        if self._model is not None:
            return
        raise NotImplementedError(
            f"Evo 2 ({self._scale}) requires GPU setup. "
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
