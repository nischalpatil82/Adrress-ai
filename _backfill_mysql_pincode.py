"""Backfill the MySQL `addresses.pincode` column from the address text.

Queries every row, extracts the first 6-digit sequence from
`normalized_full_address` (falling back to `source_raw_address`),
and updates the `pincode` column.
"""
from __future__ import annotations
import re
import os
from sqlalchemy import create_engine, text

PIN_RE = re.compile(r"\b(\d{6})\b")


def _extract_pincode(addr: str | None) -> str | None:
    if not addr:
        return None
    m = PIN_RE.search(str(addr))
    return m.group(1) if m else None


def get_engine():
    host = os.getenv("DB_HOST", "127.0.0.1")
    port = int(os.getenv("DB_PORT", "3306"))
    user = os.getenv("DB_USER", "root")
    pw = os.getenv("DB_PASSWORD", "root")
    db = os.getenv("DB_NAME", "address_ai")
    url = f"mysql+pymysql://{user}:{pw}@{host}:{port}/{db}"
    return create_engine(url, pool_pre_ping=True, future=True)


def main():
    eng = get_engine()
    batch_size = 500
    updated = 0
    skipped = 0

    with eng.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM addresses")).scalar()
        print(f"Total rows: {total:,}")

        # Cursor-based fetch to avoid loading everything into RAM
        result = conn.execute(
            text("SELECT address_id, normalized_full_address, source_raw_address FROM addresses")
        )

        batch: list[tuple[str, int]] = []
        for row in result:
            addr_id = row.address_id
            pin = _extract_pincode(row.normalized_full_address)
            if not pin:
                pin = _extract_pincode(row.source_raw_address)
            if pin:
                batch.append((pin, addr_id))
            else:
                skipped += 1

            if len(batch) >= batch_size:
                conn.execute(
                    text("UPDATE addresses SET pincode = :pin WHERE address_id = :aid"),
                    [{"pin": p, "aid": aid} for p, aid in batch],
                )
                conn.commit()
                updated += len(batch)
                print(f"  updated {updated:,} ...")
                batch = []

        if batch:
            conn.execute(
                text("UPDATE addresses SET pincode = :pin WHERE address_id = :aid"),
                [{"pin": p, "aid": aid} for p, aid in batch],
            )
            conn.commit()
            updated += len(batch)

        print(f"\nDone: {updated:,} rows updated, {skipped:,} rows still without pincode.")


if __name__ == "__main__":
    main()
