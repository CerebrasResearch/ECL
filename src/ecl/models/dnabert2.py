"""DNABERT-2 model wrapper (Appendix G of the paper).

DNABERT-2: BERT with BPE tokenization and ALiBi positional encoding, ~3 kb context.
Position mapping phi is non-trivial due to BPE: each token spans variable bp.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ecl.models.base import BaseGenomicModel


class DNABERT2Wrapper(BaseGenomicModel):
    """Wrapper for DNABERT-2.

    Parameters
    ----------
    model_name : str
        HuggingFace model name.
    device : str
        Torch device.
    reference_position : int or None
        Reference position in bp coordinates.
    """

    NOMINAL_CONTEXT = 3000  # approx 3 kb

    def __init__(
        self,
        model_name: str = "zhihan1996/DNABERT-2-117M",
        device: str = "cpu",
        reference_position: int | None = None,
    ):
        self._model_name = model_name
        self._device = device
        self._reference_position = reference_position
        self._model = None
        self._tokenizer = None
        self._embed_dim = 768

    def _load_model(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name, trust_remote_code=True
            )
            self._model = AutoModel.from_pretrained(self._model_name, trust_remote_code=True)
            self._model = self._model.to(self._device).eval()
            self._torch = torch
        except ImportError as e:
            raise ImportError(
                "DNABERT-2 requires: pip install transformers\n"
                "Install with: pip install ecl[models]"
            ) from e

    def forward(self, sequence: npt.NDArray) -> npt.NDArray:
        self._load_model()
        torch = self._torch

        alphabet = "ACGT"
        dna_str = "".join(alphabet[int(b)] for b in sequence)

        tokens = self._tokenizer(dna_str, return_tensors="pt", truncation=True, max_length=512)
        input_ids = tokens["input_ids"].to(self._device)

        with torch.no_grad():
            outputs = self._model(input_ids)

        # Use [CLS] token embedding or position-specific
        hidden = outputs.last_hidden_state[0]  # (seq_len, 768)
        if self._reference_position is not None:
            # Map bp position to token position (approximate)
            token_idx = self._bp_to_token(self._reference_position, dna_str)
            token_idx = min(token_idx, hidden.shape[0] - 1)
            embedding = hidden[token_idx].cpu().numpy()
        else:
            # Use CLS token
            embedding = hidden[0].cpu().numpy()
        return embedding.astype(np.float64)

    def _bp_to_token(self, bp_pos: int, dna_str: str) -> int:
        """Approximate mapping from bp position to token index."""
        if self._tokenizer is None:
            return 0
        # Use offset mapping if available
        tokens = self._tokenizer(dna_str, return_offsets_mapping=True, truncation=True)
        if "offset_mapping" in tokens:
            for idx, (start, end) in enumerate(tokens["offset_mapping"]):
                if start <= bp_pos < end:
                    return idx
        # Fallback: linear approximation
        n_tokens = len(tokens["input_ids"])
        return min(int(bp_pos / len(dna_str) * n_tokens), n_tokens - 1)

    def token_to_bp(self, token_idx: int) -> int:
        """Approximate: average ~6 bp per BPE token for DNABERT-2."""
        return token_idx * 6

    @property
    def nominal_context(self) -> int:
        return self.NOMINAL_CONTEXT

    @property
    def embedding_dim(self) -> int:
        return self._embed_dim
