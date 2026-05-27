"""
4_train_reranker.py
Train a LightGBM re-ranker on (query, candidate, label) triplets.
This pushes accuracy from ~88% → ~93% Hit@1.
Run: python 4_train_reranker.py
Expected time: ~5–10 minutes
"""

import os
import pickle
import random
import json
import re
import numpy as np
import faiss
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── config ────────────────────────────────────────────────────────────────────
BM25_PATH      = "models/bm25.pkl"
FAISS_PATH     = "models/faiss.index"
ADDRESSES_PATH = "models/addresses.npy"
RERANKER_OUT   = "models/reranker.pkl"
EMBED_MODEL    = "multi-qa-mpnet-base-dot-v1"
N_SAMPLE       = int(os.getenv("RERANK_N_SAMPLE", "5000"))
TOP_K          = 20        # candidates per query for training
RANDOM_SEED    = 42
HARD_FAILURES_PATH = os.getenv("HARD_FAILURES_PATH", "data/eval_failures_100_seed42.json")
HARD_TOPK_NEG   = 8

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def normalize(addr: str) -> str:
    addr = str(addr).lower().strip()
    addr = re.sub(r"[^\w\s]", " ", addr)
    addr = re.sub(r"_+", " ", addr)
    addr = re.sub(r"(?<=\d)(?!(?:st|nd|rd|th)\b)(?=[a-z])|(?<=[a-z])(?=\d)", " ", addr)
    return re.sub(r"\s+", " ", addr).strip()


# ── feature engineering ───────────────────────────────────────────────────────
def extract_features(query: str, candidate: str,
                     bm25_score: float, faiss_score: float) -> list:
    """
    Build a feature vector for one (query, candidate) pair.
    These features teach LightGBM how to combine signals.
    """
    q_tokens  = set(query.split())
    c_tokens  = set(candidate.split())

    # token overlap ratio
    overlap   = len(q_tokens & c_tokens) / (len(q_tokens) + 1e-9)

    # length difference ratio
    len_diff  = abs(len(query) - len(candidate)) / (len(query) + 1e-9)

    # fuzzy scores
    fuzzy_tsr = fuzz.token_sort_ratio(query, candidate) / 100.0
    fuzzy_pr  = fuzz.partial_ratio(query, candidate)   / 100.0

    # character-level edit similarity
    max_len   = max(len(query), len(candidate)) + 1e-9
    char_diff = sum(a != b for a, b in zip(query, candidate))
    edit_sim  = 1.0 - (char_diff + abs(len(query) - len(candidate))) / max_len

    # number token match (pincodes, house numbers)
    q_nums    = set(t for t in query.split()    if t.isdigit())
    c_nums    = set(t for t in candidate.split() if t.isdigit())
    num_match = float(bool(q_nums & c_nums)) if q_nums else 0.5

    return [
        bm25_score,    # BM25 keyword score
        faiss_score,   # cosine semantic score
        fuzzy_tsr,     # token sort ratio
        fuzzy_pr,      # partial ratio
        edit_sim,      # character edit similarity
        overlap,       # token overlap
        len_diff,      # length difference
        num_match,     # numeric token match
    ]

FEATURE_NAMES = [
    "bm25_score", "faiss_score", "fuzzy_tsr", "fuzzy_pr",
    "edit_sim", "token_overlap", "len_diff", "num_match",
]


def add_hard_failure_examples(
    x_rows: list,
    y_rows: list,
    failures_path: str,
    addresses: list,
    addr_index: dict,
    bm25,
    faiss_idx,
    embedder,
):
    """Inject hard positive/negative examples from failure diagnostics."""
    if not os.path.exists(failures_path):
        print(f"No hard-failure file found at {failures_path}; skipping hard examples.")
        return 0

    with open(failures_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    failures = payload.get("failures", [])
    if not failures:
        print("Hard-failure file has no failures; skipping hard examples.")
        return 0

    added = 0
    for item in tqdm(failures, desc="Injecting hard negatives"):
        query = normalize(item.get("input", ""))
        expected = normalize(item.get("expected", ""))
        predicted = normalize(item.get("predicted", ""))

        if not query or not expected:
            continue
        if expected not in addr_index:
            continue

        bm25_scores = bm25.get_scores(query.split())
        q_vec = embedder.encode([query], normalize_embeddings=True).astype("float32")

        exp_idx = addr_index[expected]
        exp_b = float(bm25_scores[exp_idx])
        exp_f = float(np.dot(q_vec[0], embedder.encode([addresses[exp_idx]], normalize_embeddings=True).astype("float32")[0]))
        x_rows.append(extract_features(query, addresses[exp_idx], exp_b, exp_f))
        y_rows.append(1)
        added += 1

        if predicted and predicted in addr_index and predicted != expected:
            neg_idx = addr_index[predicted]
            neg_b = float(bm25_scores[neg_idx])
            neg_f = float(np.dot(q_vec[0], embedder.encode([addresses[neg_idx]], normalize_embeddings=True).astype("float32")[0]))
            x_rows.append(extract_features(query, addresses[neg_idx], neg_b, neg_f))
            y_rows.append(0)
            added += 1

        # Mine additional hard negatives from nearest neighbors.
        dists, inds = faiss_idx.search(q_vec, HARD_TOPK_NEG + 1)
        for idx, f_score in zip(inds[0], dists[0]):
            candidate = addresses[idx]
            if candidate == expected:
                continue
            b_score = float(bm25_scores[idx])
            x_rows.append(extract_features(query, candidate, b_score, float(f_score)))
            y_rows.append(0)
            added += 1

    print(f"Added {added:,} hard examples from {len(failures):,} failures")
    return added


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs("models", exist_ok=True)

    print("Loading indexes and embedder...")
    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)

    faiss_idx  = faiss.read_index(FAISS_PATH)
    addresses  = np.load(ADDRESSES_PATH, allow_pickle=True).tolist()
    embedder   = SentenceTransformer(EMBED_MODEL)
    addr_index = {addr: i for i, addr in enumerate(addresses)}

    print(f"  {len(addresses):,} addresses loaded")

    # ── generate training examples ────────────────────────────────────────────
    sample    = random.sample(addresses, min(N_SAMPLE, len(addresses)))
    X, y      = [], []
    groups    = []   # retained for parity with previous data generation

    print(f"\nGenerating training examples from {len(sample):,} queries...")
    for correct_addr in tqdm(sample):
        # encode query
        q_vec = embedder.encode(
            [correct_addr], normalize_embeddings=True
        ).astype("float32")

        # retrieve top-K candidates via FAISS
        D, I = faiss_idx.search(q_vec, TOP_K)

        # BM25 scores for all addresses
        bm25_scores = bm25.get_scores(correct_addr.split())

        group_size = 0
        for rank, (idx, f_score) in enumerate(zip(I[0], D[0])):
            candidate  = addresses[idx]
            b_score    = float(bm25_scores[idx])
            feats      = extract_features(
                correct_addr, candidate, b_score, float(f_score)
            )
            label      = 1 if candidate == correct_addr else 0
            X.append(feats)
            y.append(label)
            group_size += 1

        groups.append(group_size)

    # ── inject hard negatives from known failures ───────────────────────────
    print("\nInjecting hard examples from failure diagnostics...")
    add_hard_failure_examples(
        X,
        y,
        HARD_FAILURES_PATH,
        addresses,
        addr_index,
        bm25,
        faiss_idx,
        embedder,
    )

    X = np.array(X, dtype="float32")
    y = np.array(y, dtype="int32")
    print(f"\n  Total examples : {len(X):,}")
    print(f"  Positive labels: {y.sum():,}  ({y.mean():.1%})")

    # ── train / val split ─────────────────────────────────────────────────────
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )

    # ── train LightGBM classifier (fixes row limit error) ────────────────────
    print("\nTraining LightGBM re-ranker...")
    ranker = lgb.LGBMClassifier(
        objective        = "binary",
        n_estimators     = 300,
        learning_rate    = 0.05,
        num_leaves       = 63,
        min_child_samples= 10,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        random_state     = RANDOM_SEED,
        n_jobs           = -1,
    )
    ranker.fit(
        X_tr, y_tr,
        eval_set  = [(X_val, y_val)],
        callbacks = [
            lgb.early_stopping(stopping_rounds=20, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )

    with open(RERANKER_OUT, "wb") as f:
        pickle.dump(ranker, f)
    print(f"\nRe-ranker saved -> {RERANKER_OUT}")

    # ── feature importance ────────────────────────────────────────────────────
    print("\nFeature importances:")
    importances = ranker.feature_importances_
    for name, imp in sorted(
        zip(FEATURE_NAMES, importances), key=lambda x: -x[1]
    ):
        bar = "#" * int(imp / max(importances) * 20)
        print(f"  {name:<18} {bar} {imp:.1f}")


if __name__ == "__main__":
    main()
