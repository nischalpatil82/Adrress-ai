"""
fuzzy_engine.v2.reranker  (Layer 4)
===================================
Re-ranks the top-K retrieval candidates with hand-engineered features +
LightGBM (loaded from existing models/reranker.pkl) and a calibration
post-processor that turns the LightGBM raw score into a *probability*.

Calibration is critical: v1's confidence numbers were uninterpretable
(everything came out at 89.9). We fix that by:

1. Fitting an isotonic regressor on (raw_score -> 0/1 correctness) over a
   held-out validation set, saved at models/v2/calibrator.pkl.
2. Falling back to a sigmoid if the calibrator file is absent.

Features (8) match the v1 reranker so the existing artifact is reusable:
    bm25_score, faiss_score, fuzzy_tsr, fuzzy_pr, edit_sim,
    token_overlap, len_diff, num_match
"""
from __future__ import annotations

import logging
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from rapidfuzz import fuzz

from fuzzy_engine.v2.config import CALIBRATOR_PATH, RERANKER_PATH
from fuzzy_engine.v2.normalize import normalize_text, parse
from fuzzy_engine.v2.retrieval import Candidate

log = logging.getLogger(__name__)

FEATURE_NAMES = [
    "bm25_score", "faiss_score", "fuzzy_tsr", "fuzzy_pr",
    "edit_sim", "token_overlap", "len_diff", "num_match",
    "query_specificity",
]


@dataclass
class RerankResult:
    candidate: Candidate
    raw_score: float
    probability: float
    features: dict[str, float]


class Reranker:
    def __init__(self, model_path: Path = RERANKER_PATH,
                 calibrator_path: Path = CALIBRATOR_PATH) -> None:
        self.model_path = Path(model_path)
        self.calibrator_path = Path(calibrator_path)
        self.model = None
        self.calibrator = None

    def load(self) -> "Reranker":
        if self.model_path.exists():
            with self.model_path.open("rb") as f:
                self.model = pickle.load(f)
        else:
            log.warning("Reranker model not found at %s; using fallback fusion.",
                        self.model_path)
        if self.calibrator_path.exists():
            with self.calibrator_path.open("rb") as f:
                self.calibrator = pickle.load(f)
        return self

    # ------------------------------------------------------------------
    def rerank(self, query: str, candidates: list[Candidate],
               top_n: int = 5) -> list[RerankResult]:
        if not candidates:
            return []

        feats = np.array(
            [self._featurize(query, c) for c in candidates], dtype="float32"
        )
        raw = self._predict(feats)
        base_probs = self._calibrate(raw)

        # No boost - let the address features speak for themselves
        # Different addresses should get different scores based on actual match quality
        boosted = np.clip(base_probs, 0.0, 1.0)

        ranked = sorted(
            (
                RerankResult(
                    candidate=c,
                    raw_score=float(r),
                    probability=float(p),
                    features=dict(zip(FEATURE_NAMES, row.tolist())),
                )
                for c, r, p, row in zip(candidates, raw, boosted, feats)
            ),
            key=lambda x: -x.probability,
        )
        return ranked[:top_n]

    # ------------------------------------------------------------------
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Return one raw score per row.

        Strategy (in order):
        1. If model exposes `predict_proba` (sklearn classifier, LightGBMClassifier),
           use the positive-class probability — this gives a continuous score.
        2. Else use `predict` (regressor or raw_score regression model).
        3. Else linear blend of features as a sanity fallback.
        """
        # Fallback blend with 9 features (including query_specificity)
        def fallback_blend(X):
            if X.shape[1] >= 9:
                return (X[:, 0] * 0.35 +      # bm25_score (address relevance - highest weight)
                        X[:, 1] * 0.05 +      # faiss_score (semantic similarity - low weight)
                        X[:, 2] * 0.25 +     # fuzzy_tsr (token similarity - high weight)
                        X[:, 3] * 0.20 +     # fuzzy_pr (partial matching)
                        X[:, 4] * 0.10 +     # edit_sim (edit distance)
                        X[:, 5] * 0.05 +      # token_overlap (shared tokens)
                        X[:, 8] * 0.00)      # query_specificity (no boost - let features speak)
            else:
                # Old 8-feature fallback
                return X[:, 0] * 0.3 + X[:, 1] * 0.4 + X[:, 2] * 0.3

        if self.model is None:
            return fallback_blend(X)

        # If model was trained with fewer features than we provide,
        # use fallback blend so the new features (e.g., query_specificity) count.
        try:
            n_model_features = getattr(self.model, 'n_features_in_', None)
            if n_model_features and X.shape[1] > n_model_features:
                log.warning("Model expects %d features but we have %d; using fallback blend.", n_model_features, X.shape[1])
                return fallback_blend(X)
        except Exception:
            pass

        try:
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(X)
                if proba.ndim == 2 and proba.shape[1] >= 2:
                    return proba[:, 1]
                return proba.ravel()
            return self.model.predict(X)
        except Exception as exc:  # noqa: BLE001
            log.warning("Reranker predict failed: %s; falling back to blend.", exc)
            return fallback_blend(X)

    def _calibrate(self, raw: np.ndarray) -> np.ndarray:
        if self.calibrator is not None:
            try:
                result = np.clip(self.calibrator.predict(raw), 0.0, 1.0)
                # If calibrator collapses all values to the same extreme,
                # it was trained on a different score distribution; use sigmoid.
                if len(set(np.round(result, 3))) > 1:
                    return result
                log.warning("Calibrator collapsed all scores to %.3f; using sigmoid.", result[0])
            except Exception as exc:  # noqa: BLE001
                log.warning("Calibrator failed: %s; falling back to sigmoid.", exc)
        # Sigmoid fallback centred on 0 (assumes raw was z-ish).
        return 1.0 / (1.0 + np.exp(-raw))

    # ------------------------------------------------------------------
    @staticmethod
    def _featurize(query: str, c: Candidate) -> list[float]:
        q = normalize_text(query)
        a = normalize_text(c.address)
        bm25 = float(c.scores.get("bm25", 0.0))
        faiss_s = float(c.scores.get("faiss", 0.0))
        tsr = float(fuzz.token_sort_ratio(q, a)) / 100.0
        pr = float(fuzz.partial_ratio(q, a)) / 100.0
        edit_sim = float(fuzz.ratio(q, a)) / 100.0
        q_tok, a_tok = set(q.split()), set(a.split())
        denom = max(len(q_tok | a_tok), 1)
        token_overlap = len(q_tok & a_tok) / denom
        len_diff = abs(len(q) - len(a)) / max(len(q), len(a), 1)
        q_p = parse(query)
        a_p = parse(c.address)
        num_match = 1.0 if q_p.numbers and (q_p.numbers & a_p.numbers) else 0.0

        # Query specificity: higher for more specific queries
        q_specificity = 0.1
        if q_p.pincode:
            q_specificity = 0.2
        if q_p.locality_anchors:
            q_specificity += 0.25
        if q_p.road_anchor:
            q_specificity += 0.2
        if q_p.numbers:
            non_pincode = q_p.numbers - {q_p.pincode} if q_p.pincode else q_p.numbers
            if non_pincode:
                q_specificity += 0.15
        if q_p.informative_tokens:
            non_pincode_tokens = q_p.informative_tokens
            if q_p.pincode:
                non_pincode_tokens = non_pincode_tokens - {q_p.pincode}
            if len(non_pincode_tokens) > 0:
                q_specificity += min(0.15 * len(non_pincode_tokens), 0.3)
        q_specificity = min(q_specificity, 1.0)

        return [bm25, faiss_s, tsr, pr, edit_sim, token_overlap, len_diff, num_match, q_specificity]
