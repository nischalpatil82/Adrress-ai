"""
1_prepare_data.py
Generate noisy→clean training pairs from your address dataset.
Run: python 1_prepare_data.py
"""

import random
import re
import pickle
import os
import json
from collections import Counter

# ── config ────────────────────────────────────────────────────────────────────
TRAIN_OUT       = "data/train_pairs.pkl"
VAL_OUT         = "data/val_pairs.pkl"
TRAIN_JSONL     = "data/address_training_kier_v1_strict_clean.jsonl"
AUGMENT_FACTOR  = 4       # noisy versions per clean address
VAL_SPLIT       = 0.10
RANDOM_SEED     = 42

QUALITY_REPORT_OUT = "data/training_data_quality.json"

STRUCTURED_FIELDS = [
    "house_no",
    "flat_no",
    "floor",
    "house_name",
    "apartment_name",
    "landmark",
    "street",
    "area",
    "town",
    "village",
    "district",
    "state",
    "country",
    "pincode",
]

random.seed(RANDOM_SEED)

# ── abbreviation maps ─────────────────────────────────────────────────────────
ABBR = {
    "street": "st",   "road": "rd",    "avenue": "ave",
    "nagar":  "ngr",  "colony": "clny","district": "dist",
    "bangalore": "blr","mumbai": "mum", "delhi": "del",
    "hyderabad": "hyd","chennai": "chn","kolkata": "kol",
    "sector": "sec",  "layout": "lyt", "cross": "xing",
}
REV_ABBR = {v: k for k, v in ABBR.items()}

# ── helpers ───────────────────────────────────────────────────────────────────
def normalize(addr: str) -> str:
    addr = str(addr).lower().strip()
    addr = re.sub(r"[^\w\s]", " ", addr)
    addr = re.sub(r"_+", " ", addr)
    addr = re.sub(r"(?<=\d)(?!(?:st|nd|rd|th)\b)(?=[a-z])|(?<=[a-z])(?=\d)", " ", addr)
    addr = re.sub(r"\s+", " ", addr)
    return addr.strip()


def inject_noise(addr: str) -> str:
    """Simulate realistic address typos for training data augmentation."""
    tokens = addr.split()
    out = []
    for t in tokens:
        r = random.random()
        if r < 0.12 and len(t) > 3:
            # swap two adjacent characters
            i = random.randint(0, len(t) - 2)
            t = t[:i] + t[i+1] + t[i] + t[i+2:]
        elif r < 0.20 and len(t) > 3:
            # delete a random character
            i = random.randint(1, len(t) - 1)
            t = t[:i] + t[i+1:]
        elif r < 0.27 and len(t) > 3:
            # duplicate a random character
            i = random.randint(0, len(t) - 1)
            t = t[:i] + t[i] + t[i:]
        elif r < 0.34:
            # abbreviate or expand
            t = ABBR.get(t, REV_ABBR.get(t, t))
        elif r < 0.38 and len(tokens) > 3:
            # drop token entirely
            continue
        out.append(t)
    return " ".join(out) if out else addr


def compose_from_structured(row: dict) -> str:
    parts = []
    for key in STRUCTURED_FIELDS:
        val = row.get(key)
        if val is None:
            continue
        txt = str(val).strip()
        if txt:
            parts.append(txt)
    return " ".join(parts)


def choose_clean_address(row: dict) -> str:
    return (
        row.get("normalized_full_address")
        or row.get("source_raw_address")
        or compose_from_structured(row)
        or ""
    )


def build_quality_report(rows, addresses):
    source_counts = Counter()
    empty_source_rows = 0
    too_short_rows = 0

    for row, addr in zip(rows, addresses):
        has_norm = bool(str(row.get("normalized_full_address") or "").strip())
        has_raw = bool(str(row.get("source_raw_address") or "").strip())
        has_structured = bool(compose_from_structured(row).strip())

        if has_norm:
            source_counts["normalized_full_address"] += 1
        elif has_raw:
            source_counts["source_raw_address"] += 1
        elif has_structured:
            source_counts["structured_columns"] += 1
        else:
            source_counts["empty"] += 1
            empty_source_rows += 1

        if len(addr) <= 5:
            too_short_rows += 1

    valid = [a for a in addresses if len(a) > 5]
    dup_count = len(valid) - len(set(valid))

    return {
        "rows_total": len(rows),
        "source_breakdown": dict(source_counts),
        "rows_with_no_usable_text": empty_source_rows,
        "rows_too_short_after_normalize": too_short_rows,
        "valid_rows_after_filter": len(valid),
        "duplicate_clean_addresses": dup_count,
    }


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(TRAIN_JSONL):
        raise RuntimeError(
            f"Dataset not found: {TRAIN_JSONL}. "
            "Run 'python 0_clean_training_data.py' first."
        )

    print(f"Loading addresses from JSONL dataset ... ({TRAIN_JSONL})")
    rows = []
    with open(TRAIN_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    addresses = [normalize(str(r.get("clean_target", ""))) for r in rows]
    report = {
        "rows_total": len(rows),
        "source_breakdown": {"jsonl_clean_target": len(rows)},
        "rows_with_no_usable_text": 0,
        "rows_too_short_after_normalize": sum(1 for a in addresses if len(a) <= 5),
        "valid_rows_after_filter": sum(1 for a in addresses if len(a) > 5),
        "duplicate_clean_addresses": len([a for a in addresses if len(a) > 5]) - len(set([a for a in addresses if len(a) > 5])),
    }

    with open(QUALITY_REPORT_OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("  Data quality report:")
    print(f"    Total rows                 : {report['rows_total']:,}")
    print(f"    Valid rows after filter    : {report['valid_rows_after_filter']:,}")
    print(f"    Empty source rows          : {report['rows_with_no_usable_text']:,}")
    print(f"    Duplicate clean addresses  : {report['duplicate_clean_addresses']:,}")
    print(f"  Report saved -> {QUALITY_REPORT_OUT}")

    addresses = [a for a in addresses if len(a) > 5]   # drop empty rows
    print(f"  Loaded {len(addresses):,} clean addresses")

    # Build T5-style pairs: input="correct address: <noisy>", target=<clean>
    pairs = []
    for addr in addresses:
        for _ in range(AUGMENT_FACTOR):
            noisy = inject_noise(addr)
            pairs.append({
                "input":  f"correct address: {noisy}",
                "target": addr,
            })

    random.shuffle(pairs)
    split     = int(len(pairs) * (1 - VAL_SPLIT))
    train_p   = pairs[:split]
    val_p     = pairs[split:]

    with open(TRAIN_OUT, "wb") as f: pickle.dump(train_p, f)
    with open(VAL_OUT,   "wb") as f: pickle.dump(val_p,   f)

    print(f"  Train pairs : {len(train_p):,}")
    print(f"  Val   pairs : {len(val_p):,}")
    print(f"  Saved -> {TRAIN_OUT}  {VAL_OUT}")

    # quick sanity check
    sample = random.choice(train_p)
    print("\nSample pair:")
    print(f"  INPUT : {sample['input']}")
    print(f"  TARGET: {sample['target']}")


if __name__ == "__main__":
    main()
