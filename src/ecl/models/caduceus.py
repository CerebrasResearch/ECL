"""Caduceus model wrapper (Appendix G of the paper).

Caduceus: BiMamba (bidirectional SSM) with reverse-complement equivariance, 131,072 bp.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ecl.models.base import BaseGenomicModel


class CaduceusWrapper(BaseGenomicModel):
    """Wrapper for the Caduceus model.

    Parameters
    ----------
    model_name : str
        HuggingFace model name.
    device : str
        Torch device.
    reference_position : int or None
        Position from which to extract the embedding.
    """

    NOMINAL_CONTEXT = 131_072

    def __init__(
        self,
        model_name: str = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16",
        device: str = "cpu",
        reference_position: int | None = None,
    ):
        self._model_name = model_name
        self._device = device
        self._reference_position = reference_position
        self._model = None
        self._tokenizer = None
        self._embed_dim = 256

    def _load_model(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name, trust_remote_code=True
            )
            self._model = AutoModelForMaskedLM.from_pretrained(
                self._model_name, trust_remote_code=True
            )
            self._model = self._model.to(self._device).eval()
            self._torch = torch
        except ImportError as e:
            raise ImportError(
                "Caduceus requires: pip install transformers mamba-ssm caduceus\n"
                "Install with: pip install ecl[models]"
            ) from e

    def forward(self, sequence: npt.NDArray) -> npt.NDArray:
        self._load_model()
        torch = self._torch

        alphabet = "ACGT"
        dna_str = "".join(alphabet[int(b)] for b in sequence)

        tokens = self._tokenizer(dna_str, return_tensors="pt")
        input_ids = tokens["input_ids"].to(self._device)

        with torch.no_grad():
            outputs = self._model(input_ids, output_hidden_states=True)

        hidden = outputs.hidden_states[-1][0]
        ref = self._reference_position or len(sequence) // 2
        ref = min(ref, hidden.shape[0] - 1)
        embedding = hidden[ref].cpu().numpy()
        return embedding.astype(np.float64)

    @property
    def nominal_context(self) -> int:
        return self.NOMINAL_CONTEXT

    @property
    def embedding_dim(self) -> int:
        return self._embed_dim

    def token_to_bp(self, token_idx: int) -> int:
        return token_idx
