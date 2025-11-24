from pathlib import Path
from typing import List

import sentencepiece as spm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_DIR = PROJECT_ROOT / "tokenizer"
DEFAULT_MODEL_PATH = TOKENIZER_DIR / "earnings_bpe.model"


class EarningsTokenizer:
    """
    Thin wrapper around SentencePiece for our earnings-call tokenizer.

    Usage:
        tok = EarningsTokenizer()   # uses default earnings_bpe.model
        ids = tok.encode("Hello world")
        text = tok.decode(ids)
    """

    def __init__(self, model_path: str | Path | None = None):
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Tokenizer model not found at {self.model_path}. "
                f"Did you run `python -m src.tokenizer_train`?"
            )

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(self.model_path))

    @property
    def vocab_size(self) -> int:
        return self.sp.GetPieceSize()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """
        Convert a string into a list of token IDs.
        """
        ids = self.sp.EncodeAsIds(text)
        if add_bos:
            ids = [self.sp.bos_id()] + ids
        if add_eos:
            ids = ids + [self.sp.eos_id()]
        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Convert a list of token IDs back into a string.
        """
        # SentencePiece expects Python ints
        ids = list(map(int, ids))
        return self.sp.DecodeIds(ids)

    def bos_id(self) -> int:
        return self.sp.bos_id()

    def eos_id(self) -> int:
        return self.sp.eos_id()

    def pad_id(self) -> int:
        return self.sp.pad_id()

    def unk_id(self) -> int:
        return self.sp.unk_id()
