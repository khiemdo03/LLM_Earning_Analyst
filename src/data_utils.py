import re
import random
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# Paths and corpus utilities
# ----------------------------

# Base paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_PATH = PROCESSED_DIR / "train.txt"
VAL_PATH = PROCESSED_DIR / "val.txt"


def find_raw_text_files(raw_dir: Path) -> List[Path]:
    """
    Recursively find all .txt files under data/raw/.
    Example structure:
        data/raw/NVDA/NVDA_2025_Q2.txt
        data/raw/META/META_2025_Q2.txt
    """
    return sorted(raw_dir.rglob("*.txt"))


def clean_transcript(text: str) -> str:
    """
    Clean a raw earnings call transcript.

    Heuristics:
    - Try to skip website junk at the top (menus, market tickers, etc.)
      by starting from the first line that looks like the real call:
        * contains 'earnings call', 'results conference call',
          'conference call', 'prepared remarks'
    - Remove lines that clearly look like site navigation or stock widgets.
    - Collapse multiple blank lines into single blank lines.
    """

    # Normalize line endings and split into lines
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    # Try to find where the actual transcript starts
    start_markers = [
        "earnings call transcript",
        "earnings call",
        "results conference call",
        "earnings conference call",
        "prepared remarks",
        "results call",
        "conference call",
    ]
    start_idx = 0
    for i, line in enumerate(lines):
        low = line.lower()
        if any(marker in low for marker in start_markers):
            start_idx = i
            break

    # Slice from start_idx to end
    lines = lines[start_idx:]

    cleaned_lines: List[str] = []
    last_was_blank = False

    # Patterns / keywords that usually indicate junk (site UI, tickers, etc.)
    junk_keywords = [
        "accessibility menu",
        "the motley fool",
        "follow us",
        "daily stock gainers",
        "daily stock losers",
        "most active stocks",
        "free article",
        "image source:",
        "logo of jester cap",
        "s&p 500",
        "nasdaq",
        "bitcoin",
        "dji",
        "stock gainers",
        "stock losers",
        "privacy policy",
        "terms of use",
    ]

    # Ticker / quote style lines: crude heuristic
    ticker_line_re = re.compile(
        r"^[A-Z]{2,5}\s*\$?\d+(\.\d+)?\s*[\+\-]\d", re.ASCII
    )

    for raw_line in lines:
        line = raw_line.strip()

        # Skip fully empty lines but keep single blank lines
        if line == "":
            if not last_was_blank:
                cleaned_lines.append("")
                last_was_blank = True
            continue

        low = line.lower()

        # Drop junk keyword lines
        if any(kw in low for kw in junk_keywords):
            continue

        # Drop obvious ticker price lines
        if ticker_line_re.match(line):
            continue

        # Some pages have standalone single numbers (page numbers, etc.)
        if re.fullmatch(r"\d+", line):
            # we treat these as likely junk and drop
            continue

        # If a line is extremely short and looks like site navigation or stray labels, drop it
        if len(line) <= 2 and not any(ch.isalpha() for ch in line):
            continue

        cleaned_lines.append(line)
        last_was_blank = False

    # Join back together
    cleaned_text = "\n".join(cleaned_lines).strip()

    return cleaned_text


def build_corpus(
    raw_dir: Path,
    processed_dir: Path,
    train_path: Path,
    val_path: Path,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> Tuple[str, str]:
    """
    Read all raw transcripts, clean them, and split into train/val corpora.

    We split by-document:
    - Each file becomes one cleaned document (string).
    - We shuffle the documents and assign ~train_ratio to train, rest to val.
    - Inside each split, documents are joined with two newlines between them.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)

    files = find_raw_text_files(raw_dir)
    if not files:
        raise FileNotFoundError(f"No .txt files found under {raw_dir}")

    print(f"Found {len(files)} raw transcript files under {raw_dir}")

    documents: List[str] = []

    for path in files:
        try:
            raw_text = path.read_text(encoding="utf-8", errors="ignore")
        except UnicodeDecodeError:
            raw_text = path.read_text(encoding="latin-1", errors="ignore")

        cleaned = clean_transcript(raw_text)
        if cleaned:
            documents.append(cleaned)
        else:
            print(f"Warning: cleaned document is empty for {path}")

    if not documents:
        raise RuntimeError("All documents were empty after cleaning; check your data.")

    # Shuffle and split into train/val
    random.seed(seed)
    random.shuffle(documents)

    n_docs = len(documents)
    n_train = max(1, int(train_ratio * n_docs))
    train_docs = documents[:n_train]
    val_docs = documents[n_train:] if n_train < n_docs else []

    train_text = "\n\n".join(train_docs).strip()
    val_text = "\n\n".join(val_docs).strip() if val_docs else ""

    # Save to disk
    train_path.write_text(train_text, encoding="utf-8")
    print(f"Wrote train corpus to {train_path} (from {len(train_docs)} documents)")

    val_path.write_text(val_text, encoding="utf-8")
    print(f"Wrote val corpus to {val_path} (from {len(val_docs)} documents)")

    return train_text, val_text

# ----------------------------
# Dataset & DataLoader for LM
# ----------------------------


class EarningsLMDataset(Dataset):
    """
    Language modeling dataset for earnings-call text.

    It:
      - reads a single .txt file (train or val)
      - tokenizes the entire text once
      - returns overlapping blocks of length `block_size`
        with targets shifted by one position
    """

    def __init__(
        self,
        text_path: Path,
        tokenizer,
        block_size: int = 128,
        stride: int = 1,
        add_bos: bool = False,
        add_eos: bool = False,
    ):
        """
        Args:
            text_path: path to .txt file with raw text.
            tokenizer: an object with .encode(text) -> List[int]
            block_size: input sequence length.
            stride: step between consecutive blocks (1 = max overlap).
            add_bos, add_eos: whether to add BOS/EOS tokens to the entire corpus before slicing.
        """
        if not text_path.exists():
            raise FileNotFoundError(f"Text file not found: {text_path}")

        raw_text = text_path.read_text(encoding="utf-8", errors="ignore")

        # Tokenize the entire corpus
        ids = tokenizer.encode(raw_text, add_bos=add_bos, add_eos=add_eos)
        self.tokens = torch.tensor(ids, dtype=torch.long)

        self.block_size = block_size
        self.stride = max(1, stride)

        # Precompute starting indices for blocks
        self.indices = list(range(0, len(self.tokens) - block_size - 1, self.stride))

        if len(self.indices) == 0:
            raise ValueError(
                f"Text too short ({len(self.tokens)} tokens) for block_size={block_size}"
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        start = self.indices[idx]
        end = start + self.block_size
        x = self.tokens[start:end]              # input
        y = self.tokens[start + 1 : end + 1]    # target is next token
        return x, y


def create_lm_dataloader(
    text_path: Path,
    tokenizer,
    block_size: int,
    batch_size: int,
    stride: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Helper to create a DataLoader for language modeling.

    Args:
        text_path: path to the corpus (.txt)
        tokenizer: EarningsTokenizer instance
        block_size: sequence length
        batch_size: batch size
        stride: stride between segments
        shuffle: whether to shuffle dataset order
        num_workers: for DataLoader
    """
    dataset = EarningsLMDataset(
        text_path=text_path,
        tokenizer=tokenizer,
        block_size=block_size,
        stride=stride,
        add_bos=False,
        add_eos=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )
    return loader


def main():
    """
    Entry point so you can run:

        python -m src.data_utils

    from the project root (llm-earnings-call-analyst/).

    This will (re)build train.txt and val.txt from data/raw/.
    """
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw data dir: {RAW_DIR}")
    print(f"Processed dir: {PROCESSED_DIR}")

    build_corpus(
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
        train_path=TRAIN_PATH,
        val_path=VAL_PATH,
        train_ratio=0.9,
        seed=42,
    )


if __name__ == "__main__":
    main()
