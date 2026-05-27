"""Build BM25 and FAISS retrieval indexes from the approved training JSONL."""

import argparse
import os
import re
import pickle
import json
from pathlib import Path
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── config ────────────────────────────────────────────────────────────────────
BM25_OUT       = "models/bm25.pkl"
FAISS_OUT      = "models/faiss.index"
EMBEDDINGS_OUT = "models/embeddings.npy"
ADDRESSES_OUT  = "models/addresses.npy"
ADDRESS_IDS_OUT = "models/address_ids.npy"
TRAIN_JSONL    = "data/address_training_kier_v1_strict_clean.jsonl"
EMBED_MODEL    = "multi-qa-mpnet-base-dot-v1"   # best accuracy for addresses
BATCH_SIZE     = 64


# ── helpers ───────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Build BM25 + FAISS retrieval artifacts atomically."
    )
    parser.add_argument("--train-jsonl", default=TRAIN_JSONL)
    parser.add_argument("--embed-model", default=EMBED_MODEL)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only locally cached SentenceTransformer files.",
    )
    return parser.parse_args()


def normalize(addr: str) -> str:
    addr = str(addr).lower().strip()
    addr = re.sub(r"[^\w\s]", " ", addr)
    addr = re.sub(r"_+", " ", addr)
    addr = re.sub(r"(?<=\d)(?!(?:st|nd|rd|th)\b)(?=[a-z])|(?<=[a-z])(?=\d)", " ", addr)
    return re.sub(r"\s+", " ", addr).strip()


def load_addresses_from_jsonl(path: str):
    """Load unique canonical addresses from approved JSONL dataset."""
    if not os.path.exists(path):
        raise RuntimeError(
            f"Dataset not found: {path}. Build indexes only from Kier strict dataset."
        )

    seen = set()
    paired = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"Invalid JSON at line {line_no} in {path}: {exc}")

            addr_id = int(obj.get("id", line_no))
            clean_target = normalize(obj.get("clean_target", ""))
            if len(clean_target) <= 4:
                continue

            if clean_target in seen:
                continue
            seen.add(clean_target)
            paired.append((addr_id, clean_target))

    if not paired:
        raise RuntimeError(f"No valid addresses found in {path}")
    return paired


# ── main ──────────────────────────────────────────────────────────────────────
def _tmp_path(path: str) -> str:
    p = Path(path)
    return str(p.with_name(p.name + ".tmp"))


def _publish_artifacts(temp_to_final: list[tuple[str, str]]) -> None:
    for temp_path, _ in temp_to_final:
        if not os.path.exists(temp_path):
            raise RuntimeError(f"Temporary artifact missing: {temp_path}")
    for temp_path, final_path in temp_to_final:
        os.replace(temp_path, final_path)


def main():
    args = parse_args()
    os.makedirs("models", exist_ok=True)

    print("Loading addresses from approved JSONL dataset ...")
    paired = load_addresses_from_jsonl(args.train_jsonl)

    address_ids = [addr_id for addr_id, _ in paired]
    addresses = [addr for _, addr in paired]
    print(f"  {len(addresses):,} addresses loaded")

    # ── BM25 index ────────────────────────────────────────────────────────────
    print("\nBuilding BM25 index...")
    tokenized = [addr.split() for addr in tqdm(addresses)]
    bm25      = BM25Okapi(tokenized)

    bm25_tmp = _tmp_path(BM25_OUT)
    faiss_tmp = _tmp_path(FAISS_OUT)
    embeddings_tmp = _tmp_path(EMBEDDINGS_OUT)
    addresses_tmp = _tmp_path(ADDRESSES_OUT)
    address_ids_tmp = _tmp_path(ADDRESS_IDS_OUT)

    with open(bm25_tmp, "wb") as f:
        pickle.dump(bm25, f)
    print(f"  BM25 staged -> {bm25_tmp}")
    print(f"  Vocab size : {len(bm25.idf):,} unique terms")

    # ── FAISS semantic index ──────────────────────────────────────────────────
    print(f"\nBuilding FAISS semantic index with {args.embed_model}...")
    print("  (This may take 10-15 minutes for 100K addresses on CPU)")

    try:
        embedder = SentenceTransformer(
            args.embed_model,
            local_files_only=args.local_files_only,
        )
    except Exception as exc:
        if os.path.exists(bm25_tmp):
            os.remove(bm25_tmp)
        hint = (
            "Embedding model load failed before publishing artifacts. "
            "Existing model files were left unchanged. "
            "If the model is not cached locally, rerun without --local-files-only "
            "when network access is available."
        )
        raise RuntimeError(f"{hint}\nOriginal error: {exc}") from exc

    embeddings = embedder.encode(
        addresses,
        batch_size=args.batch_size,
        normalize_embeddings=True,   # L2 normalise so cosine = dot product
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    embeddings = embeddings.astype("float32")

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # exact inner product = cosine on normalised vecs
    index.add(embeddings)

    faiss.write_index(index, faiss_tmp)
    with open(embeddings_tmp, "wb") as f:
        np.save(f, embeddings)
    with open(addresses_tmp, "wb") as f:
        np.save(f, np.array(addresses, dtype=object))
    with open(address_ids_tmp, "wb") as f:
        np.save(f, np.array(address_ids, dtype=np.int64))

    _publish_artifacts([
        (bm25_tmp, BM25_OUT),
        (faiss_tmp, FAISS_OUT),
        (embeddings_tmp, EMBEDDINGS_OUT),
        (addresses_tmp, ADDRESSES_OUT),
        (address_ids_tmp, ADDRESS_IDS_OUT),
    ])

    print(f"\n  FAISS index saved -> {FAISS_OUT}")
    print(f"  Embeddings  saved -> {EMBEDDINGS_OUT}")
    print(f"  Addresses   saved -> {ADDRESSES_OUT}")
    print(f"  Address IDs saved -> {ADDRESS_IDS_OUT}")
    print(f"  Total vectors    : {index.ntotal:,}")
    print(f"  Embedding dim    : {dim}")

    # ── quick sanity check ────────────────────────────────────────────────────
    print("\nSanity check - querying 'mg road bangalore':")
    q_vec   = embedder.encode(
        ["mg road bangalore"], normalize_embeddings=True
    ).astype("float32")
    D, I    = index.search(q_vec, 5)
    for rank, (idx, score) in enumerate(zip(I[0], D[0]), 1):
        print(f"  {rank}. [{score:.3f}] {addresses[idx]}")

    print("\nIndexes built successfully!")


if __name__ == "__main__":
    main()
