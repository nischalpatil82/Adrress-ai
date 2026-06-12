"""
eval_v2.py
==========
Accuracy harness for the *v2* pipeline that the API actually serves
(fuzzy_engine.v2.orchestrator.AddressPipeline), as opposed to 6_evaluate.py
which measures the older 5_full_pipeline_sql.py.

Measures Exact / Fuzzy Hit@1 and Hit@5 against the strict-clean validation set.
Deterministic by default: fixed sample seed, geocoder OFF (no network, no API
cost, reproducible). T5 follows V2_USE_T5 env (default off, matching Docker).

Run:
    python eval_v2.py                  # default: 200 pairs, no dense, no geocode
    python eval_v2.py --n 300          # more pairs
    python eval_v2.py --dense          # force dense retrieval ON (needs faiss.index)
    python eval_v2.py --no-dense       # force dense retrieval OFF
    python eval_v2.py --geocode        # enable geocoder (slow, may hit network)
    python eval_v2.py --seed 7 --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VAL_JSONL = ROOT / "data" / "address_training_kier_v1_strict_clean.jsonl"
FUZZY_THR = 80


def load_pairs(path: Path) -> list[dict]:
    pairs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            noisy = str(obj.get("noisy_input", "")).strip()
            target = str(obj.get("clean_target", "")).strip()
            if noisy and target:
                pairs.append({"noisy": noisy, "target": target})
    return pairs


def _norm(s: str) -> str:
    """Case/space-insensitive normalization for scoring.

    Production output is title-cased for display and may append a trailing
    country / zero-padded pincode for geocoded results; none of that carries
    address meaning, so we lowercase and collapse whitespace before comparing.
    """
    return " ".join(str(s).lower().split())


def preds_from_result(result) -> list[str]:
    """Ordered prediction list: best_address first, then suggestions."""
    preds: list[str] = []
    best = (result.best_address or "").strip()
    if best:
        preds.append(best)
    for s in result.suggestions:
        addr = (getattr(s, "address", "") or "").strip()
        if addr and addr not in preds:
            preds.append(addr)
    return preds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dense", dest="dense", action="store_true", default=None)
    ap.add_argument("--no-dense", dest="dense", action="store_false")
    ap.add_argument("--geocode", action="store_true")
    ap.add_argument("--json", type=str, default="")
    ap.add_argument("--label", type=str, default="")
    args = ap.parse_args()

    from fuzzy_engine.v2 import config as v2config
    from fuzzy_engine.v2.orchestrator import AddressPipeline

    dense_available = v2config.FAISS_PATH.exists()
    dense_on = dense_available if args.dense is None else args.dense

    pairs = load_pairs(VAL_JSONL)
    if not pairs:
        raise RuntimeError(f"No usable validation rows in {VAL_JSONL}")

    rng = random.Random(args.seed)
    sample = rng.sample(pairs, min(args.n, len(pairs)))

    use_t5 = os.getenv("V2_USE_T5", "0").lower() not in ("0", "false", "no", "")

    print(f"\nv2 accuracy eval")
    print(f"  val file     : {VAL_JSONL.name}")
    print(f"  sample       : {len(sample)} pairs (seed={args.seed})")
    print(f"  dense/faiss  : {'ON' if dense_on else 'OFF'} "
          f"(index {'present' if dense_available else 'MISSING'})")
    print(f"  geocoder     : {'ON' if args.geocode else 'OFF'}")
    print(f"  t5 speller   : {'ON' if use_t5 else 'OFF'}")
    print()

    t0 = time.time()
    pipe = AddressPipeline.from_config(use_t5=use_t5, use_geocoder=args.geocode)
    # Hard-disable dense if requested, regardless of index presence.
    if not dense_on and getattr(pipe, "retriever", None) is not None:
        if getattr(pipe.retriever, "dense", None) is not None:
            pipe.retriever.dense = None
    print(f"Pipeline loaded in {time.time()-t0:.1f}s. Scoring...\n")

    exact_1 = exact_5 = fuzzy_1 = fuzzy_5 = 0
    failures = []
    t0 = time.time()
    for i, pair in enumerate(sample, 1):
        noisy, expected = pair["noisy"], pair["target"]
        try:
            result = pipe.correct(noisy, top_n=5)
        except Exception as exc:  # noqa: BLE001
            failures.append({"input": noisy, "expected": expected,
                             "error": repr(exc)})
            continue
        preds = preds_from_result(result)
        p1 = preds[0] if preds else ""
        exp_n = _norm(expected)
        preds_n = [_norm(p) for p in preds]
        p1_n = preds_n[0] if preds_n else ""

        e1 = bool(preds_n) and p1_n == exp_n
        e5 = exp_n in preds_n
        f1 = bool(preds_n) and fuzz.token_sort_ratio(p1_n, exp_n) >= FUZZY_THR
        f5 = any(fuzz.token_sort_ratio(p, exp_n) >= FUZZY_THR for p in preds_n)
        exact_1 += e1; exact_5 += e5; fuzzy_1 += f1; fuzzy_5 += f5

        if not f1:
            failures.append({
                "input": noisy, "expected": expected, "predicted": p1,
                "status": result.status,
                "similarity": round(fuzz.token_sort_ratio(p1_n, exp_n), 1),
            })
        if i % 50 == 0:
            print(f"  ...{i}/{len(sample)}")

    n = len(sample)
    dt = time.time() - t0
    summary = {
        "label": args.label,
        "n_eval": n, "seed": args.seed,
        "dense": dense_on, "geocode": args.geocode, "t5": use_t5,
        "exact_hit_at_1": round(exact_1 / n, 4),
        "exact_hit_at_5": round(exact_5 / n, 4),
        "fuzzy_hit_at_1": round(fuzzy_1 / n, 4),
        "fuzzy_hit_at_5": round(fuzzy_5 / n, 4),
        "sec_per_query": round(dt / n, 3),
        "n_failures_below_fuzzy1": len(failures),
    }

    print("\n" + "=" * 44)
    print(f"{'Metric':<26}{'Score':>8}")
    print("-" * 44)
    print(f"{'Exact Hit@1':<26}{summary['exact_hit_at_1']:>8.1%}")
    print(f"{'Exact Hit@5':<26}{summary['exact_hit_at_5']:>8.1%}")
    print(f"{'Fuzzy Hit@1 (>=80%)':<26}{summary['fuzzy_hit_at_1']:>8.1%}")
    print(f"{'Fuzzy Hit@5 (>=80%)':<26}{summary['fuzzy_hit_at_5']:>8.1%}")
    print("-" * 44)
    print(f"{'sec/query':<26}{summary['sec_per_query']:>8.3f}")
    print("=" * 44)

    if args.json:
        out = {"summary": summary, "failures": failures[:50]}
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json} ({len(failures)} failures, first 50 saved)")

    return 0


if __name__ == "__main__":
    sys.exit(main())