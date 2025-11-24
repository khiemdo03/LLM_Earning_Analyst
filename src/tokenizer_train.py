import argparse
from pathlib import Path

import sentencepiece as spm


# Paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
TOKENIZER_DIR = PROJECT_ROOT / "tokenizer"


def train_tokenizer(
    input_paths,
    model_prefix: str,
    vocab_size: int = 8000,
    character_coverage: float = 1.0,
    model_type: str = "bpe",
):
    """
    Train a SentencePiece tokenizer on the given input text files.

    Args:
        input_paths: list of Path objects pointing to text files.
        model_prefix: prefix for the output model files
                      (SentencePiece will create <prefix>.model and <prefix>.vocab)
        vocab_size: vocabulary size for BPE model.
        character_coverage: fraction of characters covered (1.0 for English).
        model_type: 'bpe', 'unigram', etc. (we use 'bpe' here).
    """

    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    # SentencePiece expects a comma-separated list of file paths
    input_str = ",".join(str(p) for p in input_paths)

    print("=== Training tokenizer ===")
    print(f"Input files: {input_str}")
    print(f"Model prefix: {model_prefix}")
    print(f"Vocab size: {vocab_size}")
    print(f"Model type: {model_type}")

    spm.SentencePieceTrainer.Train(
        input=input_str,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        bos_id=1,   # <s>
        eos_id=2,   # </s>
        pad_id=0,   # <pad>
        unk_id=3,   # <unk>
        input_sentence_size=0,  # use all sentences
        shuffle_input_sentence=True,
    )

    print("Tokenizer training complete.")
    print(f"Model saved to: {model_prefix}.model")
    print(f"Vocab saved to: {model_prefix}.vocab")


def main():
    parser = argparse.ArgumentParser(
        description="Train a SentencePiece BPE tokenizer on processed earnings call text."
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=str(DATA_DIR / "train.txt"),
        help="Path to the training text file.",
    )
    parser.add_argument(
        "--extra_file",
        type=str,
        default=str(DATA_DIR / "val.txt"),
        help="Optional extra text file (e.g., val) to include in tokenizer training.",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=8000,
        help="Vocabulary size for the tokenizer.",
    )
    parser.add_argument(
        "--model_prefix",
        type=str,
        default=str(TOKENIZER_DIR / "earnings_bpe"),
        help="Output prefix for tokenizer model files.",
    )
    args = parser.parse_args()

    train_path = Path(args.train_file)
    extra_path = Path(args.extra_file)

    input_files = []
    if train_path.exists():
        input_files.append(train_path)
    else:
        raise FileNotFoundError(f"Training file not found: {train_path}")

    # extra file is optional
    if extra_path.exists():
        input_files.append(extra_path)

    train_tokenizer(
        input_paths=input_files,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=1.0,
        model_type="bpe",
    )


if __name__ == "__main__":
    main()
