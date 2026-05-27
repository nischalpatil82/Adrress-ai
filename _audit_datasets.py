"""Dataset audit: duplicates, pincode inconsistencies across 3 sources.

Sources:
  1. Static corpus  : models/addresses.npy   (powers BM25/FAISS/trie)
  2. Live MySQL     : addresses table        (SQLRetriever)
  3. India Post CSV : data/india_post_pincodes.csv (pincode validation)
"""
from __future__ import annotations
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

# ---------- helpers ---------------------------------------------------------
PIN_RE = re.compile(r"\b(\d{6})\b")

def norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def pincode_of(addr: str) -> str | None:
    m = PIN_RE.search(addr or "")
    return m.group(1) if m else None


# ---------- 1. STATIC CORPUS  ----------------------------------------------
def audit_static():
    print("\n" + "=" * 60)
    print("1. STATIC CORPUS  models/addresses.npy")
    print("=" * 60)
    p = ROOT / "models" / "addresses.npy"
    if not p.exists():
        print("  MISSING:", p);  return
    arr = np.load(p, allow_pickle=True)
    n = len(arr)
    print(f"  rows: {n:,}")

    # Exact duplicates
    raw_dupes = Counter(arr.tolist())
    raw_dups = sum(1 for v in raw_dupes.values() if v > 1)
    print(f"  exact-duplicate strings : {raw_dups:,}")

    # Normalized duplicates
    norm_counter: Counter = Counter()
    for a in arr:
        norm_counter[norm(str(a))] += 1
    norm_dups = sum(c - 1 for c in norm_counter.values() if c > 1)
    print(f"  normalized duplicates   : {norm_dups:,} extra rows")

    # Same pincode + same locality skeleton (likely true duplicates)
    by_pin_loc: dict[tuple[str, str], list[str]] = defaultdict(list)
    no_pin = 0
    for a in arr:
        s = str(a)
        pc = pincode_of(s)
        if not pc:
            no_pin += 1
            continue
        # locality skeleton = first 3 alpha tokens (excluding city/state)
        toks = [t for t in norm(s).split() if t.isalpha()]
        sk = " ".join(toks[:3])
        by_pin_loc[(pc, sk)].append(s)
    multi = {k: v for k, v in by_pin_loc.items() if len(v) > 1}
    print(f"  rows missing pincode    : {no_pin:,}")
    print(f"  pincode+locality groups w/ >1 row: {len(multi):,}")

    # Show a few examples
    print("\n  Example dup groups (pincode, locality skeleton):")
    for (pc, sk), rows in list(multi.items())[:5]:
        print(f"    [{pc}] {sk!r}  ({len(rows)} rows)")
        for r in rows[:3]:
            print(f"       - {r[:90]}")


# ---------- 2. MYSQL  -------------------------------------------------------
def audit_mysql():
    print("\n" + "=" * 60)
    print("2. MYSQL  addresses table")
    print("=" * 60)
    try:
        from fuzzy_engine.db_loader import get_engine
        from sqlalchemy import text
    except Exception as e:
        print("  IMPORT FAIL:", e); return

    try:
        eng = get_engine()
        with eng.connect() as c:
            n = c.execute(text("SELECT COUNT(*) FROM addresses")).scalar()
            print(f"  rows: {n:,}")

            # Exact duplicate normalized_full_address
            row = c.execute(text(
                "SELECT COUNT(*) FROM (SELECT normalized_full_address, COUNT(*) ct "
                "FROM addresses WHERE normalized_full_address IS NOT NULL "
                "GROUP BY normalized_full_address HAVING ct > 1) x"
            )).scalar()
            print(f"  groups w/ duplicate normalized_full_address: {row:,}")

            # Rows missing pincode
            row = c.execute(text(
                "SELECT COUNT(*) FROM addresses "
                "WHERE pincode IS NULL OR pincode = ''"
            )).scalar()
            print(f"  rows missing pincode: {row:,}")

            # Same pincode, different state (mismatch)
            row = c.execute(text(
                "SELECT COUNT(*) FROM ("
                " SELECT pincode FROM addresses "
                " WHERE pincode IS NOT NULL AND pincode <> '' "
                "   AND state IS NOT NULL AND state <> '' "
                " GROUP BY pincode HAVING COUNT(DISTINCT LOWER(state)) > 1"
                ") x"
            )).scalar()
            print(f"  pincodes with >1 distinct state: {row:,}")

            # Bad pincode format (not 6 digits)
            row = c.execute(text(
                "SELECT COUNT(*) FROM addresses "
                "WHERE pincode IS NOT NULL AND pincode <> '' "
                "  AND pincode NOT REGEXP '^[0-9]{6}$'"
            )).scalar()
            print(f"  rows with malformed pincode: {row:,}")
    except Exception as e:
        print("  DB ERROR:", e)


# ---------- 3. INDIA POST CSV  ---------------------------------------------
def audit_pincode_csv():
    print("\n" + "=" * 60)
    print("3. INDIA POST CSV  data/india_post_pincodes.csv")
    print("=" * 60)
    import csv
    p = ROOT / "data" / "india_post_pincodes.csv"
    if not p.exists():
        print("  MISSING:", p); return
    rows = 0
    by_pin_state: dict[str, set] = defaultdict(set)
    bad_pin = 0
    with p.open("r", encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f)
        for rec in r:
            rows += 1
            pin = (rec.get("Pincode") or rec.get("pincode") or "").strip()
            state = (rec.get("StateName") or rec.get("State") or
                     rec.get("state") or "").strip().lower()
            if not re.fullmatch(r"\d{6}", pin):
                bad_pin += 1; continue
            if state:
                by_pin_state[pin].add(state)
    print(f"  rows: {rows:,}")
    print(f"  malformed pincodes: {bad_pin:,}")
    multi_state = sum(1 for v in by_pin_state.values() if len(v) > 1)
    print(f"  pincodes mapped to >1 state (data error): {multi_state:,}")
    if multi_state:
        print("  examples:")
        i = 0
        for pin, states in by_pin_state.items():
            if len(states) > 1:
                print(f"    {pin} -> {sorted(states)}")
                i += 1
                if i >= 5: break


if __name__ == "__main__":
    audit_static()
    audit_mysql()
    audit_pincode_csv()
