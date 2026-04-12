"""HyenaDNA model wrapper (Appendix G of the paper).

HyenaDNA: implicit long convolution (Hyena operator), up to 1 Mb single-nucleotide resolution.
Available via HuggingFace.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ecl.models.base import BaseGenomicModel

# Character-level DNA tokenization for HyenaDNA
_HYENA_VOCAB = {"A": 7, "C": 8, "G": 9, "T": 10, "N": 11}
_IDX_TO_HYENA = {0: 7, 1: 8, 2: 9, 3: 10}  # our int encoding -> HyenaDNA tokens


class HyenaDNAWrapper(BaseGenomicModel):
    """Wrapper for HyenaDNA models.

    Parameters
    ----------
    model_name : str
        HuggingFace model name (e.g., 'LongSafari/hyenadna-large-1m-seqlen-hf').
    device : str
        Torch device.
    reference_position : int or None
        Position from which to extract the embedding. None = center.
    """

    NOMINAL_CONTEXT = 1_000_000

    def __init__(
        self,
        model_name: str = "LongSafari/hyenadna-large-1m-seqlen-hf",
        device: str = "cpu",
        reference_position: int | None = None,
    ):
        self._model_name = model_name
        self._device = device
        self._reference_position = reference_position
        self._model = None
        self._tokenizer = None
        self._embed_dim = 256  # HyenaDNA-large hidden dim

    def _load_model(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name, trust_remote_code=True
            )
            self._model = self._model.to(self._device).eval()
            self._torch = torch

            if hasattr(self._model.config, "d_model"):
                self._embed_dim = self._model.config.d_model
        except ImportError as e:
            raise ImportError(
                "HyenaDNA requires: pip install transformers\n"
                "Install with: pip install ecl[models]"
            ) from e

    def forward(self, sequence: npt.NDArray) -> npt.NDArray:
        self._load_model()
        torch = self._torch

        # Convert integer array to DNA string
        alphabet = "ACGT"
        dna_str = "".join(alphabet[int(b)] for b in sequence)

        tokens = self._tokenizer(dna_str, return_tensors="pt")
        input_ids = tokens["input_ids"].to(self._device)

        with torch.no_grad():
            outputs = self._model(input_ids, output_hidden_states=True)

        # Extract last hidden state at reference position
        hidden = outputs.hidden_states[-1][0]  # (seq_len, d_model)
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
        return token_idx  # character-level, phi = identity
