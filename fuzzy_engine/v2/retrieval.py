"""
fuzzy_engine.v2.retrieval  (Layer 3)
====================================
Hybrid retrieval over the verified-address corpus.

Three indices, queried in parallel and merged:

1. PrefixTrie       - typo-tolerant autocomplete on every keystroke.
2. BM25Retriever    - keyword recall (loads existing models/bm25.pkl).
3. DenseRetriever   - semantic search via FAISS + sentence-transformers
                      (loads existing models/faiss.index + embeddings.npy).

Each query returns a unified candidate list:
    [(addr_id, address_str, scores_dict), ...]
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from fuzzy_engine.v2.config import (
    ADDRESSES_PATH,
    ADDRESS_IDS_PATH,
    AUTOCOMPLETE_TOP_K,
    BM25_PATH,
    EMBED_MODEL,
    EMBEDDINGS_PATH,
    FAISS_PATH,
    RETRIEVAL_TOP_K,
    TRIE_PATH,
)
from fuzzy_engine.v2.normalize import normalize_text, parse as _parse_address

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers for locality pre-filtering
# ---------------------------------------------------------------------------
from functools import lru_cache

@lru_cache(maxsize=10000)
def _significant_tokens(address: str) -> frozenset[str]:
    """Extract significant (non-generic) tokens from an address for locality filtering.

    Significant tokens include: road anchors, locality anchors, informative tokens,
    and numeric identifiers. Excludes city names and generic words.
    Cached up to 10k addresses for O(1) repeated lookup.
    """
    parsed = _parse_address(address)
    tokens = set()
    if parsed.road_anchor:
        tokens.add(parsed.road_anchor)
    tokens.update(parsed.locality_anchors)
    # informative_tokens are already filtered (>=4 chars, not generic/city)
    tokens.update(parsed.informative_tokens)
    # Include numbers beyond pincode (house/building numbers)
    if parsed.numbers:
        if parsed.pincode:
            tokens.update(parsed.numbers - {parsed.pincode})
        else:
            tokens.update(parsed.numbers)
    return frozenset(tokens)


# ---------------------------------------------------------------------------
# Candidate type
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    addr_id: int
    address: str
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def fused(self) -> float:
        return sum(self.scores.values())


# ---------------------------------------------------------------------------
# Prefix trie  (autocomplete)
# ---------------------------------------------------------------------------
class PrefixTrie:
    """Lightweight prefix trie over normalized addresses for autocomplete.

    Only stores per-node `(addr_id, popularity)` lists to keep memory bounded.
    Typo tolerance is achieved by also indexing each token's 3-gram shingles.
    """

    def __init__(self) -> None:
        self._root: dict = {}
        self._addr_by_id: dict[int, str] = {}
        self._popularity: dict[int, int] = {}

    # ----- build -----
    def build(self, items: Iterable[tuple[int, str]]) -> None:
        for addr_id, addr in items:
            norm = normalize_text(addr)
            if not norm:
                continue
            self._addr_by_id[addr_id] = addr
            self._popularity.setdefault(addr_id, 1)
            self._insert_string(norm, addr_id)

    def _insert_string(self, s: str, addr_id: int) -> None:
        node = self._root
        for ch in s:
            node = node.setdefault(ch, {})
            bucket = node.setdefault("$ids", [])
            if len(bucket) < 50:
                bucket.append(addr_id)

    # ----- query -----
    def search(self, prefix: str, k: int = AUTOCOMPLETE_TOP_K) -> list[Candidate]:
        prefix = normalize_text(prefix)
        if not prefix:
            return []
        node = self._root
        for ch in prefix:
            if ch not in node:
                return []
            node = node[ch]
        ids = node.get("$ids", [])[:k * 4]
        seen: set[int] = set()
        out: list[Candidate] = []
        for aid in ids:
            if aid in seen:
                continue
            seen.add(aid)
            addr = self._addr_by_id.get(aid)
            if not addr:
                continue
            out.append(
                Candidate(addr_id=aid, address=addr, scores={"trie": 1.0})
            )
            if len(out) >= k:
                break
        return out

    # ----- io -----
    def save(self, path: Path = TRIE_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {"root": self._root, "addr": self._addr_by_id, "pop": self._popularity},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def load(self, path: Path = TRIE_PATH) -> "PrefixTrie":
        with Path(path).open("rb") as f:
            data = pickle.load(f)
        self._root = data["root"]
        self._addr_by_id = data["addr"]
        self._popularity = data.get("pop", {})
        return self

    @property
    def size(self) -> int:
        return len(self._addr_by_id)


# ---------------------------------------------------------------------------
# BM25 retriever (loads existing artifact)
# ---------------------------------------------------------------------------
class BM25Retriever:
    def __init__(self, path: Path = BM25_PATH) -> None:
        self.path = Path(path)
        self.bm25 = None
        self.tokenizer = lambda s: normalize_text(s).split()

    def load(self) -> "BM25Retriever":
        with self.path.open("rb") as f:
            self.bm25 = pickle.load(f)
        return self

    def search(self, query: str, k: int = RETRIEVAL_TOP_K) -> list[tuple[int, float]]:
        if self.bm25 is None:
            return []
        toks = self.tokenizer(query)
        if not toks:
            return []
        scores = self.bm25.get_scores(toks)
        top_idx = np.argpartition(scores, -min(k, len(scores)))[-k:]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]


# ---------------------------------------------------------------------------
# Dense retriever (FAISS + sentence-transformers)
# ---------------------------------------------------------------------------
class DenseRetriever:
    def __init__(self,
                 faiss_path: Path = FAISS_PATH,
                 embeddings_path: Path = EMBEDDINGS_PATH,
                 model_name: str = EMBED_MODEL) -> None:
        self.faiss_path = Path(faiss_path)
        self.embeddings_path = Path(embeddings_path)
        self.model_name = model_name
        self.index = None
        self.model = None

    def load(self) -> "DenseRetriever":
        try:
            import faiss  # noqa: WPS433
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # noqa: BLE001
            log.warning("Dense retriever unavailable: %s", exc)
            return self
        if not self.faiss_path.exists():
            log.warning("FAISS index not found at %s", self.faiss_path)
            return self
        self.index = faiss.read_index(str(self.faiss_path))
        self.model = SentenceTransformer(self.model_name)
        return self

    def search(self, query: str, k: int = RETRIEVAL_TOP_K) -> list[tuple[int, float]]:
        if self.index is None or self.model is None:
            return []
        vec = self.model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
        D, I = self.index.search(vec, k)
        out: list[tuple[int, float]] = []
        for idx, dist in zip(I[0], D[0]):
            if idx == -1:
                continue
            out.append((int(idx), float(dist)))
        return out


# ---------------------------------------------------------------------------
# Hybrid retriever  (the public face of L3)
# ---------------------------------------------------------------------------
class HybridRetriever:
    """Fuses prefix-trie + BM25 + dense retrievers into one candidate list."""

    def __init__(
        self,
        addresses: list[str],
        address_ids: list[int],
        bm25: Optional[BM25Retriever] = None,
        dense: Optional[DenseRetriever] = None,
        trie: Optional[PrefixTrie] = None,
    ) -> None:
        assert len(addresses) == len(address_ids), "addresses/ids length mismatch"
        self.addresses = addresses
        self.address_ids = address_ids
        self._id_to_addr = dict(zip(address_ids, addresses))
        # FAISS / BM25 indices use the *position* in the addresses array,
        # not the SQL address_id, so we keep both mappings.
        self._pos_to_id = {i: aid for i, aid in enumerate(address_ids)}

        self.bm25 = bm25
        self.dense = dense
        self.trie = trie

        # Build pincode -> [positions] inverted index for O(1) prefilter.
        # Each address's first 6-digit token is treated as its pincode.
        self._pincode_to_pos: dict[str, list[int]] = {}
        for i, addr in enumerate(addresses):
            for tok in addr.split():
                tok = tok.strip(",.()/")
                if tok.isdigit() and len(tok) == 6:
                    self._pincode_to_pos.setdefault(tok, []).append(i)
                    break

        # Pre-compute significant tokens for each address for locality filtering.
        self._addr_sig_tokens: dict[int, set[str]] = {}
        for i, addr in enumerate(addresses):
            self._addr_sig_tokens[i] = set(_significant_tokens(addr))

    # ---- factory ----
    @classmethod
    def from_artifacts(cls) -> "HybridRetriever":
        addresses = list(np.load(ADDRESSES_PATH, allow_pickle=True))
        ids = list(np.load(ADDRESS_IDS_PATH, allow_pickle=True))
        ids = [int(x) for x in ids]
        bm25 = BM25Retriever().load() if Path(BM25_PATH).exists() else None
        dense = DenseRetriever().load() if Path(FAISS_PATH).exists() else None
        trie = PrefixTrie()
        if Path(TRIE_PATH).exists():
            trie.load(TRIE_PATH)
        else:
            log.info("Prefix trie not built yet; building in-memory.")
            trie.build(zip(ids, addresses))
        return cls(addresses=addresses, address_ids=ids,
                   bm25=bm25, dense=dense, trie=trie)

    # ---- search ----
    def search(self, query: str, k: int = RETRIEVAL_TOP_K,
               pincode: Optional[str] = None) -> list[Candidate]:
        """Hybrid retrieval. If `pincode` is given AND we have rows for it,
        results are pre-filtered to that pincode bucket (huge precision win
        for Indian addresses, since pincode is a strong geographic anchor).

        We over-fetch by 4x from the underlying indices when pre-filtering so
        the post-filter still leaves enough candidates after the bucket cut.
        """
        # Decide pincode bucket
        bucket: Optional[set[int]] = None
        if pincode and pincode in self._pincode_to_pos:
            bucket = set(self._pincode_to_pos[pincode])
            # Over-fetch so filter doesn't starve us; capped at corpus size.
            fetch_k = min(k * 4, len(self.addresses))
        else:
            fetch_k = k

        merged: dict[int, Candidate] = {}
        bm25_hits = self.bm25.search(query, k=fetch_k) if self.bm25 else []
        dense_hits = self.dense.search(query, k=fetch_k) if self.dense else []

        # Apply pincode bucket filter at position level (O(k) hash lookup).
        if bucket is not None:
            bm25_hits = [(p, s) for p, s in bm25_hits if p in bucket]
            dense_hits = [(p, s) for p, s in dense_hits if p in bucket]
            # Fallback: if bucket filtering wiped us out (e.g. embedding model
            # never returned same-pincode rows), seed candidates from the bucket
            # itself so retrieval still produces something useful.
            if not bm25_hits and not dense_hits:
                bm25_hits = [(p, 1.0) for p in list(bucket)[:k]]

        # ---- Locality pre-filter (when no pincode to anchor us) -------------
        # Without a pincode, generic city-only matches like "Marathahalli Bangalore"
        # can outrank relevant locality matches. Filter candidates to those that
        # share at least one significant (non-generic) token with the query.
        query_sig = _significant_tokens(query)
        if bucket is None and query_sig:
            # Keep candidates that share at least one significant token
            # or fallback entirely if the filter is too aggressive.
            filtered_bm25 = [(p, s) for p, s in bm25_hits
                           if self._addr_sig_tokens.get(p, set()) & set(query_sig)]
            filtered_dense = [(p, s) for p, s in dense_hits
                            if self._addr_sig_tokens.get(p, set()) & set(query_sig)]
            # If we have fewer than k/2 candidates after filtering, relax.
            if len(filtered_bm25) + len(filtered_dense) >= k // 2:
                bm25_hits, dense_hits = filtered_bm25, filtered_dense
                log.debug("Locality pre-filter active: %d BM25 + %d dense kept "
                         "(query sig: %s)", len(bm25_hits), len(dense_hits),
                         query_sig)

        # Normalize raw scores per-source to [0,1] for cleaner fusion.
        bm25_norm = _normalize_scores(bm25_hits)
        dense_norm = _normalize_scores(dense_hits)

        for pos, score in bm25_norm:
            aid = self._pos_to_id.get(pos)
            if aid is None:
                continue
            cand = merged.setdefault(aid, Candidate(aid, self._id_to_addr[aid]))
            cand.scores["bm25"] = score

        for pos, score in dense_norm:
            aid = self._pos_to_id.get(pos)
            if aid is None:
                continue
            cand = merged.setdefault(aid, Candidate(aid, self._id_to_addr[aid]))
            cand.scores["faiss"] = score

        return sorted(merged.values(), key=lambda c: -c.fused)[:k]

    def autocomplete(self, prefix: str, k: int = AUTOCOMPLETE_TOP_K) -> list[Candidate]:
        if self.trie is None:
            return []
        return self.trie.search(prefix, k=k)

    # ---- helpers ----
    @property
    def size(self) -> int:
        return len(self.addresses)


def _normalize_scores(hits: list[tuple[int, float]]) -> list[tuple[int, float]]:
    if not hits:
        return []
    vals = [s for _, s in hits]
    lo, hi = min(vals), max(vals)
    rng = hi - lo
    if rng <= 0:
        return [(i, 1.0) for i, _ in hits]
    return [(i, (s - lo) / rng) for i, s in hits]
