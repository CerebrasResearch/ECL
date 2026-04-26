"""DNABERT-2 model wrapper.

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
        device: str = "cuda",
        reference_position: int | None = None,
    ):
        self._model_name = model_name
        self._device = device
        self._reference_position = reference_position
        self._model = None
        self._tokenizer = None
        self._embed_dim = 768

    _HF_REVISION = "7bce263b15377fc15361f52cfab88f8b586abda0"

    def _load_model(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        rev = self._HF_REVISION

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            trust_remote_code=True,
            revision=rev,
        )
        BertModel = get_class_from_dynamic_module(
            "bert_layers.BertModel",
            self._model_name,
            revision=rev,
        )
        BertConfig = get_class_from_dynamic_module(
            "configuration_bert.BertConfig",
            self._model_name,
            revision=rev,
        )
        config = BertConfig.from_pretrained(self._model_name, revision=rev)
        if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
            config.pad_token_id = self._tokenizer.pad_token_id or 0

        # Patch Triton flash attention for compatibility with newer Triton
        self._patch_triton_flash_attn()

        self._model = BertModel.from_pretrained(
            self._model_name,
            config=config,
            revision=rev,
        )
        self._model = self._model.to(self._device).eval()
        self._torch = torch

    def _patch_triton_flash_attn(self):
        """Patch the cached Triton flash attention for compatibility."""
        from pathlib import Path as _P

        cache_dir = _P.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"
        triton_files = list(cache_dir.rglob("flash_attn_triton.py"))
        for tf in triton_files:
            if "DNABERT" not in str(tf):
                continue
            content = tf.read_text()
            if "trans_b=True" in content or "trans_a=True" in content:
                content = content.replace("tl.dot(q, k, trans_b=True)", "tl.dot(q, tl.trans(k))")
                content = content.replace(", trans_b=True)", ")")
                content = content.replace(", trans_a=True)", ")")
                tf.write_text(content)

    def forward(self, sequence: npt.NDArray) -> npt.NDArray:
        self._load_model()
        torch = self._torch

        alphabet = "ACGT"
        dna_str = "".join(alphabet[int(b)] for b in sequence)

        tokens = self._tokenizer(dna_str, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self._device) for k, v in tokens.items()}

        with torch.no_grad():
            out = self._model(**inputs)

        # Model returns tuple: (last_hidden_state, pooler_output)
        hidden = out[0][0] if isinstance(out, tuple) else out.last_hidden_state[0]

        if self._reference_position is not None:
            token_idx = self._bp_to_token(self._reference_position, dna_str)
            token_idx = min(token_idx, hidden.shape[0] - 1)
            embedding = hidden[token_idx].cpu().numpy()
        else:
            embedding = hidden[0].cpu().numpy()  # CLS token
        return embedding.astype(np.float64)

    def _bp_to_token(self, bp_pos: int, dna_str: str) -> int:
        """Approximate mapping from bp position to token index."""
        if self._tokenizer is None:
            return 0
        tokens = self._tokenizer(dna_str, return_offsets_mapping=True, truncation=True)
        if "offset_mapping" in tokens:
            for idx, (start, end) in enumerate(tokens["offset_mapping"]):
                if start <= bp_pos < end:
                    return idx
        n_tokens = len(tokens["input_ids"])
        return min(int(bp_pos / len(dna_str) * n_tokens), n_tokens - 1)

    def token_to_bp(self, token_idx: int) -> int:
        return token_idx * 6

    @property
    def nominal_context(self) -> int:
        return self.NOMINAL_CONTEXT

    @property
    def embedding_dim(self) -> int:
        return self._embed_dim
