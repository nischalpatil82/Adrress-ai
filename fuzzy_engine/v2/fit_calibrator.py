"""
fuzzy_engine.v2.fit_calibrator
==============================
Fits an isotonic regressor that turns the LightGBM raw reranker score into
a calibrated probability of "this candidate is the right answer".

Source of truth for labels: the kier-cleaned training jsonl, where each row
has both `noisy_input` (real-world typed address) and `clean_target`
(verified canonical address). We treat clean_target's tokens as gold.

Usage:
    python -m fuzzy_engine.v2.fit_calibrator
    python -m fuzzy_engine.v2.fit_calibrator --samples 3000
    python -m fuzzy_engine.v2.fit_calibrator --jsonl path/to/file.jsonl

Output:
    models/v2/calibrator.pkl   (sklearn IsotonicRegression)
    models/v2/calibrator_report.json  (eval metrics)
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, roc_auc_score,
)

from fuzzy_engine.v2 import AddressPipeline
from fuzzy_engine.v2.config import (
    CALIBRATOR_PATH,
    PROJECT_ROOT,
    V2_ARTIFACTS_DIR,
)
from fuzzy_engine.v2.normalize import normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("v2.fit_calibrator")

DEFAULT_JSONL = PROJECT_ROOT / "data" / "address_training_kier_v1_strict_clean.jsonl"
REPORT_PATH = V2_ARTIFACTS_DIR / "calibrator_report.json"

# Token-overlap threshold above which we call a candidate "matches the gold".
MATCH_THRESHOLD = 0.75


def _token_overlap(a: str, b: str) -> float:
    a_tok = {t for t in normalize_text(a).split() if len(t) >= 3}
    b_tok = {t for t in normalize_text(b).split() if len(t) >= 3}
    if not a_tok:
        return 0.0
    return len(a_tok & b_tok) / len(a_tok)


def _iter_eval_rows(jsonl_path: Path, n: int, seed: int = 13):
    """Sample n rows that have both noisy_input and clean_target."""
    rng = random.Random(seed)
    rows: list[dict] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not r.get("noisy_input") or not r.get("clean_target"):
                continue
            rows.append({
                "noisy": r["noisy_input"],
                "clean": r["clean_target"],
            })
    rng.shuffle(rows)
    return rows[:n]


def collect_pairs(pipeline: AddressPipeline, rows: list[dict],
                  k_per_query: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Run the pipeline on each row and emit (raw_score, label) pairs."""
    raws: list[float] = []
    labels: list[int] = []
    for i, r in enumerate(rows, 1):
        if i % 100 == 0:
            log.info("  progress %d/%d", i, len(rows))
        try:
            cands = pipeline.retriever.search(r["noisy"], k=50)
            if not cands:
                continue
            reranked = pipeline.reranker.rerank(r["noisy"], cands,
                                                top_n=k_per_query)
        except Exception as exc:  # noqa: BLE001
            log.warning("query %d failed: %s", i, exc)
            continue
        for rr in reranked:
            label = 1 if _token_overlap(r["clean"], rr.candidate.address) \
                >= MATCH_THRESHOLD else 0
            raws.append(rr.raw_score)
            labels.append(label)
    return np.asarray(raws, dtype="float64"), np.asarray(labels, dtype="int32")


def fit(raws: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raws, labels)
    return iso


def evaluate(raws: np.ndarray, labels: np.ndarray,
             cal: IsotonicRegression) -> dict:
    pos_rate = float(labels.mean())
    out: dict = {
        "n_samples": int(len(labels)),
        "positive_rate": round(pos_rate, 4),
        "raw_score_min": float(raws.min()),
        "raw_score_max": float(raws.max()),
        "raw_score_mean": float(raws.mean()),
    }
    if 0 < pos_rate < 1:
        out["roc_auc_raw"] = round(float(roc_auc_score(labels, raws)), 4)
        out["pr_auc_raw"] = round(float(average_precision_score(labels, raws)), 4)
        out["brier_raw"] = round(float(brier_score_loss(labels, raws)), 4)
        cal_p = np.clip(cal.predict(raws), 0.0, 1.0)
        out["brier_calibrated"] = round(float(brier_score_loss(labels, cal_p)), 4)
        out["calibrated_score_p10"] = round(float(np.percentile(cal_p, 10)), 4)
        out["calibrated_score_p50"] = round(float(np.percentile(cal_p, 50)), 4)
        out["calibrated_score_p90"] = round(float(np.percentile(cal_p, 90)), 4)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", default=str(DEFAULT_JSONL),
                   help="Path to training jsonl with noisy_input + clean_target")
    p.add_argument("--samples", type=int, default=1500,
                   help="Number of queries to evaluate")
    p.add_argument("--out", default=str(CALIBRATOR_PATH),
                   help="Output pickle path")
    args = p.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        log.error("Training jsonl not found: %s", jsonl_path)
        return 2

    log.info("Loading pipeline (no T5, no geocoder)... [first import takes ~30s]")
    pipeline = AddressPipeline.from_config(use_t5=False, use_geocoder=False)
    log.info("Pipeline loaded.")

    log.info("Sampling %d rows from %s", args.samples, jsonl_path)
    rows = _iter_eval_rows(jsonl_path, n=args.samples)
    log.info("Got %d eval rows.", len(rows))

    log.info("Collecting (raw_score, label) pairs ...")
    raws, labels = collect_pairs(pipeline, rows)
    log.info("Collected %d pairs (positives=%d, %.1f%%)",
             len(labels), int(labels.sum()),
             100.0 * float(labels.mean() if len(labels) else 0))

    if len(labels) < 100 or labels.sum() < 10 or labels.sum() == len(labels):
        log.error("Not enough positive/negative diversity to fit calibrator.")
        return 3

    log.info("Fitting IsotonicRegression ...")
    cal = fit(raws, labels)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(cal, f)
    log.info("Saved calibrator -> %s", out_path)

    report = evaluate(raws, labels, cal)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Eval report -> %s", REPORT_PATH)
    log.info("\n%s", json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
