import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ModelConfig
from model import TransformerModel
from data_utils import PROJECT_ROOT, TRAIN_PATH, VAL_PATH, create_lm_dataloader
from tokenizer_utils import EarningsTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a small GPT-style language model on earnings call transcripts."
    )

    # Model size
    parser.add_argument(
        "--model_size",
        type=str,
        default="tiny",
        choices=["tiny", "small"],
        help="Predefined model size: 'tiny' (good for quick tests) or 'small' (more capable).",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=128,
        help="Maximum sequence length (context window).",
    )

    # Training hyperparams
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride between consecutive training blocks in the corpus.",
    )

    parser.add_argument(
        "--train_file",
        type=str,
        default=str(TRAIN_PATH),
        help="Path to train.txt corpus.",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default=str(VAL_PATH),
        help="Path to val.txt corpus.",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints"),
        help="Directory to save model checkpoints.",
    )

    return parser.parse_args()


def get_model_config(
    vocab_size: int, model_size: str, max_seq_len: int
) -> ModelConfig:
    """
    Return a ModelConfig for either 'tiny' or 'small'.
    """
    if model_size == "tiny":
        # Very small model (good for quick CPU tests)
        return ModelConfig(
            vocab_size=vocab_size,
            d_model=128,
            n_layers=2,
            n_heads=2,
            d_ff=512,
            max_seq_len=max_seq_len,
            dropout=0.1,
        )
    elif model_size == "small":
        # Slightly larger model for real training
        return ModelConfig(
            vocab_size=vocab_size,
            d_model=256,
            n_layers=4,
            n_heads=4,
            d_ff=1024,
            max_seq_len=max_seq_len,
            dropout=0.1,
        )
    else:
        raise ValueError(f"Unknown model_size: {model_size}")


def evaluate(
    model: TransformerModel,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Compute average loss over the given DataLoader.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)  # (b, t, vocab)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=-100,
            )
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(1, n_batches)


def train():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = EarningsTokenizer()
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    # Build model config
    config = get_model_config(vocab_size, args.model_size, args.max_seq_len)
    print(f"Model config: {config}")

    # Instantiate model and move to device
    model = TransformerModel(config).to(device)
    print(f"Number of parameters: {model.num_parameters():,}")

    # Create data loaders
    train_path = Path(args.train_file)
    val_path = Path(args.val_file)

    print(f"Loading training data from: {train_path}")
    train_loader = create_lm_dataloader(
        text_path=train_path,
        tokenizer=tokenizer,
        block_size=args.max_seq_len,
        batch_size=args.batch_size,
        stride=args.stride,
        shuffle=True,
        num_workers=0,
    )

    if val_path.exists():
        print(f"Loading validation data from: {val_path}")
        val_loader = create_lm_dataloader(
            text_path=val_path,
            tokenizer=tokenizer,
            block_size=args.max_seq_len,
            batch_size=args.batch_size,
            stride=args.stride,
            shuffle=False,
            num_workers=0,
        )
    else:
        print(f"Validation file not found at {val_path}; skipping validation.")
        val_loader = None

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Where to save checkpoints
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(100, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        model.train()
        running_loss = 0.0
        n_batches = 0

        for x, y in tqdm(train_loader, desc=f"Train epoch {epoch}"):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)  # (b, t, vocab)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=-100,
            )

            loss.backward()
            # Optional: gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(1, n_batches)
        print(f"Train loss: {train_loss:.4f}")

        # Validation
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device)
            print(f"Val loss: {val_loss:.4f}")
        else:
            val_loss = train_loss  # fallback

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = save_dir / f"lm_{args.model_size}_best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.to_dict(),
                    "vocab_size": vocab_size,
                },
                ckpt_path,
            )
            print(f"Saved new best model to: {ckpt_path}")

    print("\nTraining complete.")
    print(f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()
