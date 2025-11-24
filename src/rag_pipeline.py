from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

import torch

from tokenizer_utils import EarningsTokenizer
from model import TransformerModel
from config import ModelConfig
from data_utils import find_raw_text_files, clean_transcript, PROJECT_ROOT

# Where index and metadata are stored
RAG_DIR = PROJECT_ROOT / "data" / "rag_index"
RAG_DIR.mkdir(parents=True, exist_ok=True)
INDEX_FILE = RAG_DIR / "tfidf_index.joblib"
METADATA_FILE = RAG_DIR / "chunks_metadata.json"

# Example local uploaded file path (from your session). Replace or pass via CLI.
EXAMPLE_UPLOADED_PATH = "/mnt/data/cab89646-21bb-44fa-a8d8-3a038c9d259f.png"


def chunk_text(text: str, max_chars: int = 1500, overlap_chars: int = 200) -> List[str]:
    """
    Naive char-level chunker that attempts to break on paragraph / sentence boundaries.
    - max_chars: preferred max chunk length in characters.
    - overlap_chars: number of characters to overlap between consecutive chunks.
    """
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    def flush_current():
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
            current = ""

    for p in paragraphs:
        if not current:
            current = p
        elif len(current) + 1 + len(p) <= max_chars:
            current = current + "\n\n" + p
        else:
            # current + p would exceed max, so flush current and start new (with overlap)
            flush_current()
            # create overlap by starting new chunk with trailing characters from last chunk if possible
            # simplest: start new chunk with p
            current = p

    flush_current()

    # If chunks are still large, further split by sentences
    final_chunks = []
    for c in chunks:
        if len(c) <= max_chars:
            final_chunks.append(c)
        else:
            # split by sentences (simple split on periods)
            sentences = [s.strip() for s in c.split(". ") if s.strip()]
            cur = ""
            for s in sentences:
                to_add = (s + ".") if not s.endswith(".") else s
                if not cur:
                    cur = to_add
                elif len(cur) + 1 + len(to_add) <= max_chars:
                    cur = cur + " " + to_add
                else:
                    final_chunks.append(cur.strip())
                    cur = to_add
            if cur:
                final_chunks.append(cur.strip())

    # Add small overlap windows between consecutive chunks
    if overlap_chars > 0 and len(final_chunks) > 1:
        overlapped = []
        for i, c in enumerate(final_chunks):
            if i == 0:
                overlapped.append(c)
            else:
                prev = overlapped[-1]
                # create overlap snippet from end of prev
                overlap = prev[-overlap_chars:] if len(prev) > overlap_chars else prev
                new_c = overlap + "\n\n" + c
                overlapped.append(new_c)
        final_chunks = overlapped

    return final_chunks


def build_index_from_raw(
    raw_root: Path,
    index_path: Path = INDEX_FILE,
    metadata_path: Path = METADATA_FILE,
    max_chars: int = 1500,
    overlap_chars: int = 200,
    rebuild: bool = False,
) -> Tuple[TfidfVectorizer, np.ndarray, List[Dict]]:
    """
    Walk data/raw/, read each .txt file, clean it, chunk it, build TF-IDF on chunks.
    Returns (vectorizer, matrix, metadata_list)
    - metadata_list: list of dicts {doc_path, chunk_id, text}
    """

    if index_path.exists() and metadata_path.exists() and not rebuild:
        print(f"Index exists at {index_path}, load with load_index(). Pass --build_index to rebuild.")
        loaded = joblib.load(index_path)
        vectorizer = loaded["vectorizer"]
        matrix = loaded["matrix"]
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return vectorizer, matrix, metadata

    files = find_raw_text_files(raw_root)
    if not files:
        raise FileNotFoundError(f"No .txt transcripts found under {raw_root}")

    all_chunks = []
    metadata = []

    print(f"Found {len(files)} transcripts. Chunking...")
    for path in files:
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_transcript(raw_text)
        chunks = chunk_text(cleaned, max_chars=max_chars, overlap_chars=overlap_chars)
        for i, ch in enumerate(chunks):
            metadata.append({"doc_path": str(path.relative_to(PROJECT_ROOT)), "chunk_id": i})
            all_chunks.append(ch)

    print(f"Total chunks: {len(all_chunks)}. Building TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
    matrix = vectorizer.fit_transform(all_chunks)  # sparse matrix (n_chunks x n_features)

    # Persist index
    print(f"Saving TF-IDF index to {index_path} and metadata to {metadata_path}")
    joblib.dump({"vectorizer": vectorizer, "matrix": matrix}, index_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks_to_metadata(all_chunks, metadata), f, ensure_ascii=False, indent=2)

    return vectorizer, matrix, metadata


def all_chunks_to_metadata(all_chunks: List[str], metadata_stub: List[Dict]) -> List[Dict]:
    """
    Combine chunk text and stub metadata into a list of dicts with 'text', 'doc_path', 'chunk_id'
    """
    out = []
    for chunk_text, meta in zip(all_chunks, metadata_stub):
        entry = {"text": chunk_text, "doc_path": meta["doc_path"], "chunk_id": meta["chunk_id"]}
        out.append(entry)
    return out


def load_index(index_path: Path = INDEX_FILE, metadata_path: Path = METADATA_FILE):
    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Index or metadata not found. Build it first with --build_index")
    loaded = joblib.load(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    vectorizer = loaded["vectorizer"]
    matrix = loaded["matrix"]
    return vectorizer, matrix, metadata


def retrieve_top_k(
    question: str,
    vectorizer: TfidfVectorizer,
    matrix,
    metadata: List[Dict],
    top_k: int = 3,
) -> List[Tuple[Dict, float]]:
    """
    Return a list of (metadata_entry, score) sorted by descending relevance.
    """
    q_vec = vectorizer.transform([question])  # (1, n_features)
    sims = cosine_similarity(q_vec, matrix).flatten()  # (n_chunks,)
    if sims.size == 0:
        return []

    top_idx = np.argsort(-sims)[:top_k]
    results = []
    for idx in top_idx:
        results.append((metadata[idx], float(sims[idx])))
    return results


def compose_prompt(question: str, retrieved: List[Tuple[Dict, float]], max_chars: int = 4000) -> str:
    """
    Compose a prompt for the LM from question + retrieved chunks.
    Keeps total characters under max_chars by truncating long chunks (end kept).
    """
    header = "You are a helpful assistant specialized in summarizing corporate earnings call transcripts.\n"
    header += "Answer concisely and base your answer ONLY on the provided CONTEXT. If the answer is not present, say 'Not stated in the provided transcript.'\n\n"

    ctx_parts = []
    for entry, score in retrieved:
        text = entry.get("text", "")
        # truncate from the front if too long
        if len(text) > 1000:
            text = text[-1000:]
        src = f"[SOURCE] {entry.get('doc_path')} (chunk {entry.get('chunk_id')})\n{text}\n"
        ctx_parts.append(src)

    context = "\n\n".join(ctx_parts)
    prompt = f"{header}\n[QUESTION]\n{question}\n\n[CONTEXT]\n{context}\n\n[ANSWER]\n"
    # ensure prompt not too long
    if len(prompt) > max_chars:
        prompt = prompt[: max_chars - 50] + "\n\n[TRUNCATED]"
    return prompt


def load_model_from_ckpt(ckpt_path: Path, device: torch.device):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = ckpt.get("config")
    if cfg_dict is None:
        raise RuntimeError("Checkpoint missing config entry")
    config = ModelConfig(**cfg_dict)
    model = TransformerModel(config)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, config


def answer_question_with_model(
    question: str,
    ckpt_path: Path,
    top_k: int = 3,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    device: Optional[torch.device] = None,
):
    """
    High-level wrapper: load index, retrieve chunks, compose prompt, run model to generate an answer.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading index...")
    vectorizer, matrix, metadata = load_index()

    print("Retrieving top-k chunks...")
    retrieved = retrieve_top_k(question, vectorizer, matrix, metadata, top_k=top_k)
    if not retrieved:
        return "No relevant chunks found."

    print("Composing prompt...")
    prompt = compose_prompt(question, retrieved)

    # load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = EarningsTokenizer()
    model, config = load_model_from_ckpt(Path(ckpt_path), device)

    # encode prompt (may be long) and truncate to model.max_seq_len
    ids = tokenizer.encode(prompt)
    input_ids = ids[-config.max_seq_len :]  # keep last tokens if longer than context
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # generate continuation
    with torch.no_grad():
        out = model.generate(input_tensor, max_new_tokens=max_new_tokens, temperature=temperature, top_k=50)

    out_ids = out[0].tolist()
    # decode entire output
    text = tokenizer.decode(out_ids)

    # Optionally, post-process: remove the prompt portion to show only answer after [ANSWER]
    # Attempt to find where the [ANSWER] token starts in decoded text
    answer_marker = "[ANSWER]"
    if answer_marker in text:
        answer = text.split(answer_marker, 1)[1].strip()
    else:
        # fallback: return last portion
        answer = text

    return answer


# ---------------------------
# CLI helpers
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="RAG-lite retrieval + generation demo")
    p.add_argument("--build_index", action="store_true", help="(Re)build TF-IDF index from data/raw/")
    p.add_argument("--raw_root", type=str, default=str(PROJECT_ROOT / "data" / "raw"), help="Root dir for transcripts")
    p.add_argument("--ckpt", type=str, default=str(PROJECT_ROOT / "checkpoints" / "lm_tiny_best.pt"), help="Model checkpoint")
    p.add_argument("--question", type=str, default="", help="Question to ask")
    p.add_argument("--top_k", type=int, default=3, help="Number of chunks to retrieve")
    p.add_argument("--max_new_tokens", type=int, default=200, help="Max tokens to generate")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--single_file", type=str, default="", help="Optional: path to a single transcript to index (for quick tests)")
    p.add_argument("--example_local_file", type=str, default=EXAMPLE_UPLOADED_PATH, help="Example local file path (uploaded).")
    return p.parse_args()


def main():
    args = parse_args()

    raw_root = Path(args.raw_root)
    if args.build_index:
        # Optionally index a single provided file path (useful for quick tests)
        if args.single_file:
            # copy the single file into data/raw/tmp/ for indexing
            single = Path(args.single_file)
            if not single.exists():
                print(f"Provided single_file does not exist: {single}")
            else:
                tmp = raw_root / "tmp_single"
                tmp.mkdir(parents=True, exist_ok=True)
                dest = tmp / single.name
                # If user passes a path to an image (uploaded path), we warn; typical transcript should be .txt
                if single.suffix.lower() != ".txt":
                    print(f"Warning: single_file extension is {single.suffix}; expected .txt transcript files.")
                # copy text
                try:
                    dest.write_bytes(single.read_bytes())
                    print(f"Copied {single} -> {dest} for indexing.")
                except Exception as e:
                    print("Copy failed:", e)
            # rebuild index using raw_root which now has tmp_single
        print("Building index from:", raw_root)
        build_index_from_raw(raw_root, rebuild=True)
        print("Index built.")
        return

    if not args.question:
        print("No question provided. Use --question 'your question' or --build_index to build index.")
        return

    # Ensure index exists; if not, instruct user
    try:
        answer = answer_question_with_model(
            question=args.question,
            ckpt_path=Path(args.ckpt),
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print("\n=== RAG Answer ===\n")
        print(answer)
        print("\n==================\n")
    except FileNotFoundError as e:
        print("File error:", e)
        print("If no index exists, run with --build_index first.")


if __name__ == "__main__":
    main()
