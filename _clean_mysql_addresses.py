"""Clean ALL existing normalized_full_address values in MySQL.

Fixes:
  1. Re-run normalize_text + canonicalize_localities.
  2. Fix glued words  (stagebangalore -> stage bangalore, layoutnear -> layout near).
  3. Dedouble consecutive identical tokens (layout layout -> layout).
  4. Update only when changed.
"""
from __future__ import annotations
import os
import re
from sqlalchemy import create_engine, text


def get_engine():
    host = os.getenv("DB_HOST", "127.0.0.1")
    port = int(os.getenv("DB_PORT", "3306"))
    user = os.getenv("DB_USER", "root")
    pw = os.getenv("DB_PASSWORD", "root")
    db = os.getenv("DB_NAME", "address_ai")
    url = f"mysql+pymysql://{user}:{pw}@{host}:{port}/{db}"
    return create_engine(url, pool_pre_ping=True, future=True)


def _fix_spacing(s: str) -> str:
    s = re.sub(r"([a-z]{3,})(bangalore|bengaluru|karnataka)", r"\1 \2", s, flags=re.I)
    s = re.sub(r"(layout|nagar|colony|phase|stage|block|sector)([a-z]{4,})", r"\1 \2", s, flags=re.I)
    s = re.sub(r"(\d+(?:th|st|nd|rd))(stage|phase|block|sector|layout)", r"\1 \2", s, flags=re.I)
    s = re.sub(r"(main|cross|service)(road|rd|street|st)", r"\1 \2", s, flags=re.I)
    return s


def _dedouble(s: str) -> str:
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
    s = _fix_spacing(s)
    s = _dedouble(s)
    return " ".join(s.split())


def main():
    eng = get_engine()
    updated = 0
    unchanged = 0
    batch: list[dict] = []
    BATCH_SIZE = 500

    with eng.connect() as conn:
        result = conn.execute(
            text("SELECT address_id, normalized_full_address FROM addresses")
        )
        for row in result:
            aid = row.address_id
            old = row.normalized_full_address or ""
            new = clean_address(old)
            if new != old:
                batch.append({"addr": new, "id": aid})
            else:
                unchanged += 1

            if len(batch) >= BATCH_SIZE:
                conn.execute(
                    text("UPDATE addresses SET normalized_full_address = :addr WHERE address_id = :id"),
                    batch,
                )
                conn.commit()
                updated += len(batch)
                print(f"  updated {updated:,} ...")
                batch = []

        if batch:
            conn.execute(
                text("UPDATE addresses SET normalized_full_address = :addr WHERE address_id = :id"),
                batch,
            )
            conn.commit()
            updated += len(batch)

    print(f"\nDone: {updated:,} rows cleaned, {unchanged:,} rows already clean")


if __name__ == "__main__":
    main()
