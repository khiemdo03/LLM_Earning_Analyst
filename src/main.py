from __future__ import annotations
import argparse
from pathlib import Path
import sys
import json

# Developer-provided local uploaded path (useful default for single-file tests)
EXAMPLE_UPLOADED_PATH = "/mnt/data/cab89646-21bb-44fa-a8d8-3a038c9d259f.png"

# import project modules (assumes src is a package; running as `python -m src.main` keeps imports local)
from data_utils import build_corpus, PROJECT_ROOT
from tokenizer_train import train_tokenizer
from tokenizer_utils import EarningsTokenizer
from train_lm import train as train_lm_entrypoint, parse_args as train_parse_args
from rag_pipeline import build_index_from_raw, answer_question_with_model, INDEX_FILE, METADATA_FILE
from inference_demo import load_model_and_tokenizer
from train_lm import TRAIN_PATH, VAL_PATH  # if available; fallback below

import torch


def cmd_build_corpus(args):
    print("Building/cleaning corpus...")
    build_corpus(
        raw_dir=PROJECT_ROOT / "data" / "raw",
        processed_dir=PROJECT_ROOT / "data" / "processed",
        train_path=PROJECT_ROOT / "data" / "processed" / "train.txt",
        val_path=PROJECT_ROOT / "data" / "processed" / "val.txt",
        train_ratio=0.9,
        seed=42,
    )
    print("Corpus built.")


def cmd_train_tokenizer(args):
    print(f"Training tokenizer with vocab_size={args.vocab_size} ...")
    train_tokenizer(
        input_paths=[Path(args.train_file)] if Path(args.train_file).exists() else [PROJECT_ROOT / "data" / "processed" / "train.txt"],
        model_prefix=str((PROJECT_ROOT / "tokenizer" / args.model_prefix).resolve()),
        vocab_size=args.vocab_size,
        character_coverage=1.0,
        model_type="bpe",
    )
    print("Tokenizer trained.")


def cmd_train_lm(args):
    # train_lm.py's train() expects to be run as module, but we provided train() function.
    # Simplest approach: forward args to training script via sys.argv hack.
    print("Starting LM training. This will call the training entrypoint (train_lm.train).")
    # Recreate command-line style args for train_lm.train to parse if needed.
    # If train_lm.train() parses its own args, just call it (it will parse from sys.argv). So set sys.argv accordingly.
    sys_argv_backup = sys.argv[:]
    sys.argv = [sys.argv[0], "--model_size", args.model_size, "--max_seq_len", str(args.max_seq_len),
                "--batch_size", str(args.batch_size), "--epochs", str(args.epochs), "--lr", str(args.lr),
                "--stride", str(args.stride)]
    if args.train_file:
        sys.argv += ["--train_file", args.train_file]
    if args.val_file:
        sys.argv += ["--val_file", args.val_file]
    if args.save_dir:
        sys.argv += ["--save_dir", args.save_dir]
    try:
        train_lm_entrypoint()
    finally:
        sys.argv = sys_argv_backup
    print("LM training finished.")


def cmd_build_index(args):
    print("Building TF-IDF RAG index from data/raw ...")
    build_index_from_raw(
        raw_root=PROJECT_ROOT / "data" / "raw",
        index_path=Path(args.index_file) if args.index_file else None or (PROJECT_ROOT / "data" / "rag_index" / "tfidf_index.joblib"),
        metadata_path=Path(args.metadata_file) if args.metadata_file else (PROJECT_ROOT / "data" / "rag_index" / "chunks_metadata.json"),
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
        rebuild=True,
    )
    print("Index built.")


def cmd_ask(args):
    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        print(f"Checkpoint not found at {ckpt}. Train a model first or provide --ckpt.")
        return
    print(f"Answering question using RAG + model {ckpt} ...")
    ans = answer_question_with_model(
        question=args.question,
        ckpt_path=ckpt,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print("\n=== RAG Answer ===\n")
    print(ans)
    print("\n==================\n")


def cmd_generate(args):
    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        print(f"Checkpoint not found at {ckpt}. Train a model first or provide --ckpt.")
        return
    print(f"Generating from model {ckpt} ...")
    model, tokenizer, config = load_model_and_tokenizer(ckpt_path=ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ids = tokenizer.encode(args.prompt)
    input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(input_tensor, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k if args.top_k>0 else None)
    text = tokenizer.decode(out[0].tolist())
    print("\n=== Generated ===\n")
    print(text)
    print("\n=================\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llm-earnings-call-analyst", description="Project CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # build_corpus
    sp = sub.add_parser("build_corpus", help="Clean raw transcripts and build processed train/val files")
    sp.set_defaults(func=cmd_build_corpus)

    # train_tokenizer
    sp = sub.add_parser("train_tokenizer", help="Train SentencePiece BPE tokenizer on processed train.txt")
    sp.add_argument("--vocab_size", type=int, default=16000)
    sp.add_argument("--train_file", type=str, default=str(PROJECT_ROOT / "data" / "processed" / "train.txt"))
    sp.add_argument("--model_prefix", type=str, default="earnings_bpe")
    sp.set_defaults(func=cmd_train_tokenizer)

    # train_lm
    sp = sub.add_parser("train_lm", help="Train language model (tiny or small)")
    sp.add_argument("--model_size", type=str, default="tiny", choices=["tiny", "small"])
    sp.add_argument("--max_seq_len", type=int, default=128)
    sp.add_argument("--batch_size", type=int, default=8)
    sp.add_argument("--epochs", type=int, default=1)
    sp.add_argument("--lr", type=float, default=3e-4)
    sp.add_argument("--stride", type=int, default=1)
    sp.add_argument("--train_file", type=str, default=str(PROJECT_ROOT / "data" / "processed" / "train.txt"))
    sp.add_argument("--val_file", type=str, default=str(PROJECT_ROOT / "data" / "processed" / "val.txt"))
    sp.add_argument("--save_dir", type=str, default=str(PROJECT_ROOT / "checkpoints"))
    sp.set_defaults(func=cmd_train_lm)

    # build_index
    sp = sub.add_parser("build_index", help="Build or rebuild TF-IDF RAG index from data/raw")
    sp.add_argument("--index_file", type=str, default=str(PROJECT_ROOT / "data" / "rag_index" / "tfidf_index.joblib"))
    sp.add_argument("--metadata_file", type=str, default=str(PROJECT_ROOT / "data" / "rag_index" / "chunks_metadata.json"))
    sp.add_argument("--max_chars", type=int, default=1500)
    sp.add_argument("--overlap_chars", type=int, default=200)
    sp.set_defaults(func=cmd_build_index)

    # ask (RAG)
    sp = sub.add_parser("ask", help="Ask a question using RAG + local model")
    sp.add_argument("--question", type=str, required=True)
    sp.add_argument("--ckpt", type=str, default=str(PROJECT_ROOT / "checkpoints" / "lm_tiny_best.pt"))
    sp.add_argument("--top_k", type=int, default=3)
    sp.add_argument("--max_new_tokens", type=int, default=200)
    sp.add_argument("--temperature", type=float, default=0.8)
    sp.set_defaults(func=cmd_ask)

    # generate
    sp = sub.add_parser("generate", help="Generate continuation from a prompt using local model")
    sp.add_argument("--prompt", type=str, required=True)
    sp.add_argument("--ckpt", type=str, default=str(PROJECT_ROOT / "checkpoints" / "lm_tiny_best.pt"))
    sp.add_argument("--max_new_tokens", type=int, default=100)
    sp.add_argument("--temperature", type=float, default=1.0)
    sp.add_argument("--top_k", type=int, default=50)
    sp.set_defaults(func=cmd_generate)

    # quick index-single helper (copy an external file into data/raw/tmp_single and build index)
    sp = sub.add_parser("index_single", help="Index a single transcript file (path provided) for quick tests")
    sp.add_argument("--file", type=str, default=EXAMPLE_UPLOADED_PATH, help="Path to a .txt transcript to index (or other file).")
    sp.add_argument("--rebuild", action="store_true", help="Force rebuild the index")
    sp.set_defaults(func=lambda a: _index_single_helper(a))

    return p


def _index_single_helper(args):
    # copy given path into data/raw/tmp_single/<name> then call cmd_build_index
    src = Path(args.file)
    dest_dir = PROJECT_ROOT / "data" / "raw" / "tmp_single"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    try:
        dest.write_bytes(src.read_bytes())
        print(f"Copied {src} -> {dest} (for indexing).")
    except Exception as e:
        print("Copy failed:", e)
        return
    # call build_index (rebuild True)
    build_index_from_raw(raw_root=PROJECT_ROOT / "data" / "raw", rebuild=True)
    print("Index rebuilt including single file.")


def main():
    p = build_parser()
    args = p.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
