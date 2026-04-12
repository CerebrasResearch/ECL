"""Enformer model wrapper (Appendix G of the paper).

Enformer: CNN + Transformer, 196,608 bp input, 5,313 tracks at 128 bp resolution.
Available via enformer-pytorch or TensorFlow Hub.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ecl.models.base import BaseGenomicModel


class EnformerWrapper(BaseGenomicModel):
    """Wrapper for the Enformer model.

    Parameters
    ----------
    device : str
        Torch device ('cpu' or 'cuda').
    head : str
        'human' or 'mouse'.
    output_bin : int or None
        Which output bin (of 896) to extract. None returns the central bin.
    """

    NOMINAL_CONTEXT = 196_608
    OUTPUT_BINS = 896
    BIN_SIZE = 128  # bp per output bin
    N_TRACKS_HUMAN = 5313

    def __init__(self, device: str = "cpu", head: str = "human", output_bin: int | None = None):
        self._device = device
        self._head = head
        self._output_bin = output_bin if output_bin is not None else self.OUTPUT_BINS // 2
        self._model = None

    def _load_model(self):
        """Lazy-load the Enformer model."""
        if self._model is not None:
            return
        try:
            import torch
            from enformer_pytorch import Enformer

            self._model = Enformer.from_pretrained("EleutherAI/enformer-official-rough")
            self._model = self._model.to(self._device).eval()
            self._torch = torch
        except ImportError as e:
            raise ImportError(
                "Enformer requires: pip install enformer-pytorch\n"
                "Install with: pip install ecl[models]"
            ) from e

    def forward(self, sequence: npt.NDArray) -> npt.NDArray:
        self._load_model()
        torch = self._torch

        # Convert integer sequence to one-hot
        seq = np.asarray(sequence, dtype=np.int64)
        one_hot = np.eye(4, dtype=np.float32)[seq]  # (L, 4)
        x = torch.tensor(one_hot, dtype=torch.float32).unsqueeze(0).to(self._device)

        with torch.no_grad():
            output = self._model(x)

        # output['human']: (1, 896, 5313)
        head_key = self._head
        embedding = output[head_key][0, self._output_bin].cpu().numpy()
        return embedding.astype(np.float64)

    @property
    def nominal_context(self) -> int:
        return self.NOMINAL_CONTEXT

    @property
    def embedding_dim(self) -> int:
        return self.N_TRACKS_HUMAN

    def token_to_bp(self, token_idx: int) -> int:
        return token_idx  # single-nucleotide input
