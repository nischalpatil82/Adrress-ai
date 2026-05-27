"""
fuzzy_engine.v2.build_indexes
=============================
One-shot builder for v2 artifacts.

Currently builds:
- models/v2/prefix_trie.pkl  (autocomplete index)

Run:
    python -m fuzzy_engine.v2.build_indexes

Optional flags:
    --calibrate <val_jsonl>   fit isotonic calibrator from validation pairs
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np

from fuzzy_engine.v2.config import (
    ADDRESSES_PATH,
    ADDRESS_IDS_PATH,
    CALIBRATOR_PATH,
    TRIE_PATH,
)
from fuzzy_engine.v2.retrieval import PrefixTrie

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("v2.build_indexes")


def build_trie() -> None:
    log.info("Loading addresses from %s", ADDRESSES_PATH)
    addresses = list(np.load(ADDRESSES_PATH, allow_pickle=True))
    ids = list(np.load(ADDRESS_IDS_PATH, allow_pickle=True))
    ids = [int(x) for x in ids]
    assert len(addresses) == len(ids)
    log.info("Building prefix trie over %d addresses", len(addresses))
    trie = PrefixTrie()
    trie.build(zip(ids, addresses))
    trie.save(TRIE_PATH)
    log.info("Saved trie -> %s (size=%d)", TRIE_PATH, trie.size)


def fit_calibrator(val_jsonl: Path) -> None:
    """Fit an isotonic regressor over (raw_score, label) pairs.

    Expected JSONL rows: {"raw_score": float, "label": 0|1}
    Produced by 6_evaluate.py once we wire it to dump rerank scores.
    """
    try:
        from sklearn.isotonic import IsotonicRegression  # noqa: WPS433
    except Exception as exc:  # noqa: BLE001
        log.error("scikit-learn required for calibration: %s", exc)
        return
    raws, labels = [], []
    with Path(val_jsonl).open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            raws.append(float(row["raw_score"]))
            labels.append(int(row["label"]))
    if not raws:
        log.error("No rows found in %s", val_jsonl)
        return
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raws, labels)
    CALIBRATOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CALIBRATOR_PATH.open("wb") as f:
        pickle.dump(iso, f)
    log.info("Saved calibrator -> %s (n=%d)", CALIBRATOR_PATH, len(raws))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-trie", action="store_true")
    p.add_argument("--calibrate", type=str, default=None,
                   help="Path to validation JSONL with raw_score+label.")
    args = p.parse_args()

    if not args.skip_trie:
        build_trie()
    if args.calibrate:
        fit_calibrator(Path(args.calibrate))


if __name__ == "__main__":
    main()
