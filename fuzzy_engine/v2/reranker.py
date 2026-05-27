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
        probs = self._calibrate(raw)

        ranked = sorted(
            (
                RerankResult(
                    candidate=c,
                    raw_score=float(r),
                    probability=float(p),
                    features=dict(zip(FEATURE_NAMES, row.tolist())),
                )
                for c, r, p, row in zip(candidates, raw, probs, feats)
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
        if self.model is None:
            return X[:, 0] * 0.3 + X[:, 1] * 0.4 + X[:, 2] * 0.3
        try:
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(X)
                if proba.ndim == 2 and proba.shape[1] >= 2:
                    return proba[:, 1]
                return proba.ravel()
            return self.model.predict(X)
        except Exception as exc:  # noqa: BLE001
            log.warning("Reranker predict failed: %s; falling back to blend.", exc)
            return X[:, 0] * 0.3 + X[:, 1] * 0.4 + X[:, 2] * 0.3

    def _calibrate(self, raw: np.ndarray) -> np.ndarray:
        if self.calibrator is not None:
            try:
                return np.clip(self.calibrator.predict(raw), 0.0, 1.0)
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
        return [bm25, faiss_s, tsr, pr, edit_sim, token_overlap, len_diff, num_match]
