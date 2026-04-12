"""Borzoi model wrapper (Appendix G of the paper).

Borzoi: CNN + Transformer, 524,288 bp input, 32 bp output resolution.
"""

from __future__ import annotations

import numpy.typing as npt

from ecl.models.base import BaseGenomicModel


class BorzoiWrapper(BaseGenomicModel):
    """Wrapper for the Borzoi model.

    Parameters
    ----------
    device : str
        Torch device.
    output_bin : int or None
        Which output bin to extract. None returns the central bin.
    """

    NOMINAL_CONTEXT = 524_288
    BIN_SIZE = 32
    N_TRACKS = 7611

    def __init__(self, device: str = "cpu", output_bin: int | None = None):
        self._device = device
        n_bins = self.NOMINAL_CONTEXT // self.BIN_SIZE
        self._output_bin = output_bin if output_bin is not None else n_bins // 2
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        raise NotImplementedError(
            "Borzoi loading requires model weights. "
            "See https://github.com/calico/borzoi for setup instructions. "
            "Provide a loaded model via BorzoiWrapper.from_loaded(model)."
        )

    @classmethod
    def from_loaded(cls, model, device: str = "cpu", output_bin: int | None = None):
        """Create wrapper from a pre-loaded Borzoi model object."""
        wrapper = cls(device=device, output_bin=output_bin)
        wrapper._model = model
        return wrapper

    def forward(self, sequence: npt.NDArray) -> npt.NDArray:
        self._load_model()
        raise NotImplementedError("Full Borzoi inference requires model weights.")

    @property
    def nominal_context(self) -> int:
        return self.NOMINAL_CONTEXT

    @property
    def embedding_dim(self) -> int:
        return self.N_TRACKS

    def token_to_bp(self, token_idx: int) -> int:
        return token_idx
