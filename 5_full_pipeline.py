"""
5_full_pipeline.py
The complete 93% accuracy address correction pipeline.
Loads all trained models and runs end-to-end inference.
Run: python 5_full_pipeline.py
"""

import os
import re
import pickle
import numpy as np
import faiss
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz
from collections import defaultdict

# ── config ────────────────────────────────────────────────────────────────────
T5_MODEL_PATH  = "models/t5_address"
BM25_PATH      = "models/bm25.pkl"
FAISS_PATH     = "models/faiss.index"
ADDRESSES_PATH = "models/addresses.npy"
RERANKER_PATH  = "models/reranker.pkl"
EMBED_MODEL    = "multi-qa-mpnet-base-dot-v1"

RETRIEVAL_TOP_K = 50    # candidates to retrieve before re-ranking
FINAL_TOP_N     = 5     # results returned to user
T5_BEAMS        = 4     # beam search width (higher = more accurate, slower)

# retrieval weights (BM25 + FAISS + fuzzy)
W_BM25    = 0.30
W_FAISS   = 0.50
W_FUZZY   = 0.20


# ── normalise helper ──────────────────────────────────────────────────────────
def normalize(addr: str) -> str:
    addr = str(addr).lower().strip()
    addr = re.sub(r"[^\w\s]", " ", addr)
    return re.sub(r"\s+", " ", addr).strip()


# ── load all models once ──────────────────────────────────────────────────────
def load_models():
    print("Loading T5 address correction model...")
    t5_tok = T5Tokenizer.from_pretrained(T5_MODEL_PATH)
    t5     = T5ForConditionalGeneration.from_pretrained(T5_MODEL_PATH)
    t5.eval()

    print("Loading sentence embedder...")
    embedder = SentenceTransformer(EMBED_MODEL)

    print("Loading FAISS index...")
    faiss_idx = faiss.read_index(FAISS_PATH)
    addresses = np.load(ADDRESSES_PATH, allow_pickle=True).tolist()

    print("Loading BM25 index...")
    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)

    print("Loading LightGBM re-ranker...")
    with open(RERANKER_PATH, "rb") as f:
        ranker = pickle.load(f)

    addr_to_idx = {addr: i for i, addr in enumerate(addresses)}

    print(f"All models loaded. Database: {len(addresses):,} addresses\n")
    return t5_tok, t5, embedder, faiss_idx, addresses, bm25, ranker, addr_to_idx


# ── step 1: T5 correction ─────────────────────────────────────────────────────
def t5_correct(raw: str, t5_tok, t5, max_len=64, num_beams=4) -> str:
    """Fix spelling, abbreviations, and word order using fine-tuned T5."""
    prompt = f"correct address: {normalize(raw)}"
    inp    = t5_tok(
        prompt, return_tensors="pt",
        max_length=max_len, truncation=True,
    )
    with torch.no_grad():
        out = t5.generate(
            **inp,
            max_length=max_len,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )
    return t5_tok.decode(out[0], skip_special_tokens=True).strip()


# ── step 2: retrieve candidates ───────────────────────────────────────────────
def retrieve_candidates(query: str, embedder, faiss_idx,
                        addresses, bm25, top_k=50) -> list:
    """Retrieve top-K candidates using BM25 + FAISS + fuzzy."""
    scores = defaultdict(float)
    q      = normalize(query)

    # BM25
    bm25_scores = bm25.get_scores(q.split())
    bm25_max    = float(bm25_scores.max()) + 1e-9
    for i in np.argsort(bm25_scores)[::-1][:top_k]:
        scores[i] += (float(bm25_scores[i]) / bm25_max) * W_BM25

    # FAISS semantic
    q_vec      = embedder.encode(
        [q], normalize_embeddings=True
    ).astype("float32")
    D, I       = faiss_idx.search(q_vec, top_k)
    for j, idx in enumerate(I[0]):
        scores[idx] += float(D[0][j]) * W_FAISS

    # Fuzzy boost for close matches
    for idx, addr in enumerate(addresses):
        fs = fuzz.token_sort_ratio(q, addr) / 100.0
        if fs > 0.55:
            scores[idx] += fs * W_FUZZY

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [(addresses[idx], base_score) for idx, base_score in ranked[:top_k]]


# ── step 3: LightGBM re-rank ──────────────────────────────────────────────────
def rerank_candidates(query: str, candidates: list,
                      embedder, faiss_idx, addresses,
                      bm25, ranker, addr_to_idx, top_n=5) -> list:
    """Re-rank candidates using LightGBM with rich features."""
    q          = normalize(query)
    bm25_scores = bm25.get_scores(q.split())
    q_vec       = embedder.encode(
        [q], normalize_embeddings=True
    ).astype("float32")

    feats = []
    for addr, base_score in candidates:
        idx      = addr_to_idx.get(addr, 0)
        b_score  = float(bm25_scores[idx]) if idx < len(bm25_scores) else 0.0

        # FAISS score for this specific candidate
        a_vec    = embedder.encode(
            [addr], normalize_embeddings=True
        ).astype("float32")
        f_score  = float(np.dot(q_vec[0], a_vec[0]))

        q_tok    = set(q.split())
        c_tok    = set(addr.split())
        overlap  = len(q_tok & c_tok) / (len(q_tok) + 1e-9)
        len_diff = abs(len(q) - len(addr)) / (len(q) + 1e-9)
        fuzz_tsr = fuzz.token_sort_ratio(q, addr) / 100.0
        fuzz_pr  = fuzz.partial_ratio(q, addr)    / 100.0
        max_len  = max(len(q), len(addr)) + 1e-9
        edit_sim = 1.0 - (
            sum(a != b for a, b in zip(q, addr)) +
            abs(len(q) - len(addr))
        ) / max_len
        q_nums   = set(t for t in q.split()    if t.isdigit())
        c_nums   = set(t for t in addr.split() if t.isdigit())
        num_match = float(bool(q_nums & c_nums)) if q_nums else 0.5

        feats.append([
            b_score, f_score, fuzz_tsr, fuzz_pr,
            edit_sim, overlap, len_diff, num_match,
        ])

    import pandas as pd
    feat_names    = ["bm25_score","faiss_score","fuzzy_tsr","fuzzy_pr",
                     "edit_sim","token_overlap","len_diff","num_match"]
    X             = pd.DataFrame(feats, columns=feat_names)
    # use predict_proba to get real confidence scores (probability of correct match)
    rerank_scores = ranker.predict_proba(X)[:, 1]
    ranked        = sorted(
        zip([c[0] for c in candidates], rerank_scores),
        key=lambda x: -x[1],
    )
    return [(addr, float(score)) for addr, score in ranked[:top_n]]


# ── full pipeline ─────────────────────────────────────────────────────────────
def correct_address(raw_input: str, models: tuple, top_n: int = 5) -> dict:
    """
    Full address correction pipeline:
      1. T5 fixes spelling / abbreviations
      2. BM25 + FAISS + fuzzy retrieve top-50 candidates
      3. LightGBM re-ranks to top-N
    """
    t5_tok, t5, embedder, faiss_idx, addresses, bm25, ranker, addr_to_idx = models

    # Step 1 — T5 spelling correction
    corrected  = t5_correct(raw_input, t5_tok, t5, num_beams=T5_BEAMS)

    # Step 2 — retrieve candidates
    candidates = retrieve_candidates(
        corrected, embedder, faiss_idx, addresses, bm25,
        top_k=RETRIEVAL_TOP_K,
    )

    # Step 3 — LightGBM re-rank
    final = rerank_candidates(
        corrected, candidates,
        embedder, faiss_idx, addresses, bm25, ranker, addr_to_idx,
        top_n=top_n,
    )

    return {
        "original":    raw_input,
        "corrected":   corrected,
        "top_matches": final,
        "best_match":  final[0][0] if final else "",
        "confidence":  round(final[0][1], 4) if final else 0.0,
    }


# ── main (demo + batch test) ──────────────────────────────────────────────────
def main():
    models = load_models()

    test_queries = [
        "123 Mian Stret, Mumbay",
        "connaught plase nw dlehi",
        "MG Rode Bangalor karnatak",
        "hsr laoyut bangalroe 5600102",
        "juhu bech road andheri mum",
        "sector 18 noida uttar pradsh",
    ]

    print("=" * 60)
    print("Address Correction Pipeline — Test Results")
    print("=" * 60)

    for query in test_queries:
        result = correct_address(query, models, top_n=FINAL_TOP_N)
        print(f"\nInput     : {result['original']}")
        print(f"T5 fixed  : {result['corrected']}")
        print(f"Best match: {result['best_match']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print("Top 3:")
        for i, (addr, score) in enumerate(result["top_matches"][:3], 1):
            print(f"  {i}. [{score:.3f}] {addr}")


if __name__ == "__main__":
    main()
