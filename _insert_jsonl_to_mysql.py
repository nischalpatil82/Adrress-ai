"""Insert JSONL addresses into MySQL `addresses` table, skipping duplicates.

For each JSONL row:
  1. Use `clean_target` (or fallback `raw_address`) as base.
  2. Re-normalize + canonicalize to catch "layout layout" glitches.
  3. Skip if the normalized string already exists in MySQL.
  4. Extract pincode from the text.
  5. Insert with `source_raw_address` and `normalized_full_address`.

Uses batch inserts for speed.
"""
from __future__ import annotations
import json
import os
import re
from pathlib import Path
from sqlalchemy import create_engine, text

JSONL_PATH = Path(__file__).resolve().parent / "data" / "address_training_kier_v1_strict_clean.jsonl"
BATCH_SIZE = 500

PIN_RE = re.compile(r"\b(\d{6})\b")


def get_engine():
    host = os.getenv("DB_HOST", "127.0.0.1")
    port = int(os.getenv("DB_PORT", "3306"))
    user = os.getenv("DB_USER", "root")
    pw = os.getenv("DB_PASSWORD", "root")
    db = os.getenv("DB_NAME", "address_ai")
    url = f"mysql+pymysql://{user}:{pw}@{host}:{port}/{db}"
    return create_engine(url, pool_pre_ping=True, future=True)


def _extract_pincode(s: str) -> str | None:
    m = PIN_RE.search(s or "")
    return m.group(1) if m else None


def _dedouble(s: str) -> str:
    """Remove consecutive duplicate tokens (e.g. 'layout layout')."""
    toks = s.split()
    out = []
    prev = None
    for t in toks:
        if t == prev:
            continue
        out.append(t)
        prev = t
    return " ".join(out)


def clean_address(addr: str) -> str:
    from fuzzy_engine.v2.normalize import normalize_text
    s = normalize_text(addr)
    s = _dedouble(s)
    return " ".join(s.split())


def main():
    eng = get_engine()
    inserted = 0
    skipped = 0
    batch: list[dict] = []

    # Build a set of existing normalized addresses in MySQL
    print("Fetching existing normalized addresses from MySQL ...")
    with eng.connect() as conn:
        result = conn.execute(text("SELECT normalized_full_address FROM addresses"))
        existing_norm = {row.normalized_full_address or "" for row in result}
    print(f"  {len(existing_norm):,} rows already in MySQL")

    print(f"Scanning JSONL: {JSONL_PATH}")
    with JSONL_PATH.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            raw = rec.get("raw_address") or ""
            clean = rec.get("clean_target") or rec.get("noisy_input") or raw
            if not clean:
                continue

            # Clean it
            norm = clean_address(clean)
            if not norm:
                continue

            # Skip if already in MySQL
            if norm in existing_norm:
                skipped += 1
                continue

            pin = _extract_pincode(norm)
            batch.append({
                "src": raw.strip(),
                "norm": norm,
                "pin": pin,
            })

            if len(batch) >= BATCH_SIZE:
                _insert_batch(eng, batch)
                inserted += len(batch)
                print(f"  inserted {inserted:,} ...")
                batch = []

    if batch:
        _insert_batch(eng, batch)
        inserted += len(batch)

    print(f"\nDone: {inserted:,} inserted, {skipped:,} skipped (already in DB)")


def _insert_batch(eng, batch: list[dict]) -> None:
    sql = text("""
        INSERT INTO addresses
        (source_raw_address, normalized_full_address, pincode)
        VALUES (:src, :norm, :pin)
    """)
    with eng.connect() as conn:
        conn.execute(sql, batch)
        conn.commit()


if __name__ == "__main__":
    main()
