import argparse
from pathlib import Path

import torch

from config import ModelConfig
from model import TransformerModel
from tokenizer_utils import EarningsTokenizer
from data_utils import PROJECT_ROOT


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference demo for the earnings-call mini LLM."
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "lm_tiny_best.pt"),
        help="Path to the trained model checkpoint (.pt).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Nvidia expects",
        help="Text prompt to start generation from.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (>0, higher = more random).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="If >0, keep only top_k tokens for sampling.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run on. 'auto' uses CUDA if available, else CPU.",
    )
    return parser.parse_args()


def load_model_and_tokenizer(ckpt_path: Path):
    """
    Load tokenizer, config, and model from a checkpoint.
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Load config
    cfg_dict = ckpt.get("config")
    if cfg_dict is None:
        raise KeyError("Checkpoint missing 'config' entry.")
    config = ModelConfig(**cfg_dict)

    # Load tokenizer
    tokenizer = EarningsTokenizer()

    # Build model and load weights
    model = TransformerModel(config)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    return model, tokenizer, config


def main():
    args = parse_args()

    # Decide device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    ckpt_path = Path(args.ckpt_path)

    # Load model + tokenizer
    model, tokenizer, config = load_model_and_tokenizer(ckpt_path)
    model.to(device)
    model.eval()

    print("Model config:", config)
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Prompt: {args.prompt!r}")

    # Encode prompt
    # We can optionally add BOS; for now just plain encode.
    input_ids = tokenizer.encode(args.prompt)
    if len(input_ids) == 0:
        raise ValueError("Prompt produced empty token sequence; try a different prompt.")

    # Convert to tensor (batch_size=1)
    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Generate continuation
    with torch.no_grad():
        out_ids = model.generate(
            input_ids_tensor,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
        )

    # Decode full output (prompt + generated tokens)
    out_ids_list = out_ids[0].tolist()
    generated_text = tokenizer.decode(out_ids_list)

    print("\n=== Generated Text ===")
    print(generated_text)
    print("=======================")


if __name__ == "__main__":
    main()
