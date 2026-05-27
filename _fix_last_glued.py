"""Fix last remaining extreme glued patterns."""
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


# Very specific remaining glued patterns
SPECIFIC_FIXES = [
    (r"halliraghavendra", "halli raghavendra"),
    (r"crossnear", "cross near"),
    (r"crosstalacavery", "cross talacvery"),
    (r"amruthahalliballary", "amrutha halli ballary"),
    (r"masjidk\s+g", "masjid k g"),
    (r"halliarabic", "halli arabic"),
    (r"guttahallikempe", "gutta halli kempe"),
    (r"extentiongavipuram", "extention gavipuram"),
    (r"buildinggandhi", "building gandhi"),
    (r"stcross", "st cross"),
    (r"ndcross", "nd cross"),
    (r"thcross", "th cross"),
    (r"rdcross", "rd cross"),
    (r"stmain", "st main"),
    (r"ndmain", "nd main"),
    (r"thmain", "th main"),
    (r"rdmain", "rd main"),
    (r"stphase", "st phase"),
    (r"ndphase", "nd phase"),
    (r"thphase", "th phase"),
    (r"rdphase", "rd phase"),
    (r"stblock", "st block"),
    (r"ndblock", "nd block"),
    (r"thblock", "th block"),
    (r"rdblock", "rd block"),
    (r"ststage", "st stage"),
    (r"ndstage", "nd stage"),
    (r"thstage", "th stage"),
    (r"rdstage", "rd stage"),
    (r"stlayout", "st layout"),
    (r"ndlayout", "nd layout"),
    (r"thlayout", "th layout"),
    (r"rdlayout", "rd layout"),
    (r"stnagar", "st nagar"),
    (r"ndnagar", "nd nagar"),
    (r"thnagar", "th nagar"),
    (r"rdnagar", "rd nagar"),
    (r"stroad", "st road"),
    (r"ndroad", "nd road"),
    (r"throad", "th road"),
    (r"rdroad", "rd road"),
]


def fix_address(addr: str) -> str:
    if not addr:
        return addr
    s = addr.strip()
    for pat, repl in SPECIFIC_FIXES:
        s = re.sub(pat, repl, s, flags=re.I)
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
            new = fix_address(old)
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

    print(f"\nDone: {updated:,} rows fixed, {unchanged:,} rows already clean")


if __name__ == "__main__":
    main()
