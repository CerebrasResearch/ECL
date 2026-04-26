"""Nucleotide Transformer v2/v3 wrappers (InstaDeepAI).

NT-v2 scale variants:
  - NT-v2-100m: 100M params, 512-dim hidden, 12 layers
  - NT-v2-250m: 250M params, 768-dim hidden, 20 layers
  - NT-v2-500m: 500M params, 1024-dim hidden, 29 layers

NT-v3 (next-generation):
  - NT-v3-650m: 650M params, 1280-dim hidden

All use 6-mer tokenization with ~12 kb effective context.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ecl.models.base import BaseGenomicModel

_NT_MODELS = {
    # NT-v2 family
    "v2-100m": {
        "hf_name": "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
        "embed_dim": 512,
        "nominal_context": 12_282,
    },
    "v2-250m": {
        "hf_name": "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
        "embed_dim": 768,
        "nominal_context": 12_282,
    },
    "v2-500m": {
        "hf_name": "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
        "embed_dim": 1024,
        "nominal_context": 12_282,
    },
    # NT-v3 family
    "v3-650m": {
        "hf_name": "InstaDeepAI/nucleotide-transformer-v2.1-650m-multi-species",
        "embed_dim": 1280,
        "nominal_context": 12_282,
    },
    # Legacy aliases (backward compat with run_real_experiments.py)
    "100m": {
        "hf_name": "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
        "embed_dim": 512,
        "nominal_context": 12_282,
    },
    "250m": {
        "hf_name": "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
        "embed_dim": 768,
        "nominal_context": 12_282,
    },
    "500m": {
        "hf_name": "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
        "embed_dim": 1024,
        "nominal_context": 12_282,
    },
}


class NucleotideTransformerWrapper(BaseGenomicModel):
    """Unified wrapper for Nucleotide Transformer v2/v3 family.

    Parameters
    ----------
    scale : str
        Model scale: 'v2-100m', 'v2-250m', 'v2-500m', 'v3-650m',
        or legacy aliases '100m', '250m', '500m'.
    device : str
        Torch device.
    reference_position : int or None
        Position for embedding extraction. None = center token.
    """

    TOKEN_SIZE = 6  # 6-mer tokenization

    def __init__(
        self,
        scale: str = "v2-500m",
        device: str = "cuda",
        reference_position: int | None = None,
    ):
        if scale not in _NT_MODELS:
            raise ValueError(f"Unknown scale '{scale}'. Choose from: {list(_NT_MODELS)}")
        cfg = _NT_MODELS[scale]
        self._hf_name = cfg["hf_name"]
        self._embed_dim = cfg["embed_dim"]
        self._nominal_context = cfg["nominal_context"]
        self._device = device
        self._reference_position = reference_position
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._hf_name, trust_remote_code=True)
        self._model = AutoModelForMaskedLM.from_pretrained(self._hf_name, trust_remote_code=True)
        self._model = self._model.to(self._device).eval()
        self._torch = torch

    def forward(self, sequence: npt.NDArray) -> npt.NDArray:
        self._load_model()
        torch = self._torch

        alphabet = "ACGT"
        dna_str = "".join(alphabet[int(b)] for b in sequence)

        tokens = self._tokenizer(dna_str, return_tensors="pt", truncation=True)
        input_ids = tokens["input_ids"].to(self._device)

        with torch.no_grad():
            outputs = self._model(input_ids, output_hidden_states=True)

        hidden = outputs.hidden_states[-1][0]  # (n_tokens, embed_dim)

        if self._reference_position is not None:
            token_idx = self._reference_position // self.TOKEN_SIZE + 1  # +1 for CLS
            token_idx = min(token_idx, hidden.shape[0] - 1)
        else:
            token_idx = hidden.shape[0] // 2

        embedding = hidden[token_idx].cpu().numpy()
        return embedding.astype(np.float64)

    @property
    def nominal_context(self) -> int:
        return self._nominal_context

    @property
    def embedding_dim(self) -> int:
        return self._embed_dim

    def token_to_bp(self, token_idx: int) -> int:
        return token_idx * self.TOKEN_SIZE
