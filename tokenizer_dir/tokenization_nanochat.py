import os
import pickle
import shutil
from typing import Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizer


class NanochatTokenizer(PreTrainedTokenizer):
    # Use `vocab_file` (not `tokenizer_file`) to avoid collision with the
    # internal `tokenizer_file` reserved for fast-tokenizer JSON handling.
    vocab_files_names = {"vocab_file": "tokenizer.pkl"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file=None, **kwargs):
        default_name = self.vocab_files_names["vocab_file"]
        if vocab_file is None:
            vocab_file = default_name

        # Resolve both absolute and module-relative tokenizer paths so loading
        # works from local dirs and HF cache snapshots.
        candidate_paths = [vocab_file]
        if not os.path.isabs(vocab_file):
            module_dir = os.path.dirname(__file__)
            candidate_paths.append(os.path.join(module_dir, vocab_file))
            candidate_paths.append(os.path.join(module_dir, default_name))
        resolved = next((p for p in candidate_paths if p and os.path.exists(p)), None)
        if resolved is None:
            raise FileNotFoundError(
                f"Tokenizer file not found. Tried: {candidate_paths}. "
                "Ensure tokenizer.pkl is present in the model repo."
            )

        self.vocab_file = resolved
        with open(resolved, "rb") as f:
            self._enc = pickle.load(f)
        self._special_to_id = dict(getattr(self._enc, "_special_tokens", {}))
        self._id_to_special = {v: k for k, v in self._special_to_id.items()}
        bos = kwargs.pop("bos_token", "<|bos|>")
        eos = kwargs.pop("eos_token", bos)
        pad = kwargs.pop("pad_token", bos)
        super().__init__(bos_token=bos, eos_token=eos, pad_token=pad, **kwargs)

    @property
    def vocab_size(self) -> int:
        return int(self._enc.n_vocab)

    def get_vocab(self) -> Dict[str, int]:
        vocab = {str(i): i for i in range(self.vocab_size)}
        for token, token_id in self._special_to_id.items():
            tid = int(token_id)
            if 0 <= tid < self.vocab_size:
                vocab[token] = tid
        return vocab

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        return [str(i) for i in self._enc.encode_ordinary(text)]

    def _convert_token_to_id(self, token: str) -> int:
        if token in self._special_to_id:
            return int(self._special_to_id[token])
        try:
            return int(token)
        except ValueError:
            if self.unk_token_id is not None:
                return int(self.unk_token_id)
            # Fallback to BOS so conversion never yields out-of-range IDs.
            return int(self._special_to_id.get("<|bos|>", 0))

    def _convert_id_to_token(self, index: int) -> str:
        if index in self._id_to_special:
            return self._id_to_special[index]
        return str(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        ids = []
        for token in tokens:
            if token in self._special_to_id:
                continue
            ids.append(int(token))
        return self._enc.decode(ids)

    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> str:
        pieces = []
        buf = []
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id in self._id_to_special:
                if buf:
                    pieces.append(self._enc.decode(buf))
                    buf = []
                if not skip_special_tokens:
                    pieces.append(self._id_to_special[token_id])
            else:
                buf.append(token_id)
        if buf:
            pieces.append(self._enc.decode(buf))
        return "".join(pieces)

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        os.makedirs(save_directory, exist_ok=True)
        name = "tokenizer.pkl" if filename_prefix is None else f"{filename_prefix}-tokenizer.pkl"
        out = os.path.join(save_directory, name)
        shutil.copy2(self.vocab_file, out)
        return (out,)
