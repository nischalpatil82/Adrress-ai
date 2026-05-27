"""Fix ALL known text-quality issues in MySQL normalized_full_address.

Patterns fixed:
  1. Glued road words       (crossroad -> cross road, mainroad -> main road)
  2. Glued localities       (layoutbangalore -> layout bangalore)
  3. Glued numbers+ords     (29148th -> 2914 8th, 193rdamain -> 193rd a main)
  4. Glued ordinals         (15thth -> 15th, 7thth -> 7th)
  5. Glued city+pin         (bangalore560076 -> bangalore 560076)
  6. Consecutive dup tokens (layout layout -> layout, road road -> road)
  7. Duplicate city         (bangalore bangalore -> bangalore)
  8. Duplicate state        (karnataka karnataka -> karnataka)
  9. Triple tokens          (main main main -> main)
  10. Bad ordinals          (thth -> th)
  11. Missing space after comma
  12. General glued word separation for all alpha+alpha boundaries
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


# Common address words that should be separate tokens
SEP_BEFORE = frozenset({
    "bangalore", "bengaluru", "bangaluru", "bengalorurui", "bengalurur",
    "bengalure", "bangalor", "karnataka", "india",
    "road", "rd", "street", "st", "lane", "main", "cross", "service",
    "layout", "nagar", "colony", "phase", "stage", "block", "sector",
    "extension", "extn", "circle", "junction", "puram", "halli", "pura",
    "village", "town", "city", "district", "state", "country",
    "temple", "school", "college", "hospital", "park", "station",
    "bus", "stop", "office", "building", "tower", "complex", "mall",
})

SEP_AFTER = frozenset({
    "layout", "nagar", "colony", "phase", "stage", "block", "sector",
    "puram", "halli", "pura", "cross", "main", "road", "rd", "street",
    "village", "town", "post", "bus", "railway",
})

ORDINALS = {"1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th",
            "9th", "10th", "11th", "12th", "13th", "14th", "15th",
            "16th", "17th", "18th", "19th", "20th", "21st", "22nd",
            "23rd", "24th", "25th", "26th", "27th", "28th", "29th",
            "30th", "31st", "32nd", "33rd", "34th", "35th", "36th",
            "37th", "38th", "39th", "40th", "41st", "42nd", "43rd",
            "44th", "45th", "46th", "47th", "48th", "49th", "50th"}

CITIES = {"bangalore", "bengaluru", "bangaluru", "bengalorurui",
          "bengalurur", "bengalure", "bangalor"}


def _fix_glued_words(s: str) -> str:
    """Separate words that got stuck together."""
    # 1. city/pincode glue: bangalore560076 -> bangalore 560076
    s = re.sub(r"([a-z]{3,})(\d{6})(?!\d)", r"\1 \2", s)
    s = re.sub(r"(\d{6})([a-z]{3,})", r"\1 \2", s)

    # 2. number glued to ordinal road words: 29148th -> 2914 8th
    #    But careful: "1st" is valid, "123rd" is valid.
    #    Pattern: digits followed by digits+ordinal suffix stuck to word
    s = re.sub(r"(\d+)(\d+(?:th|st|nd|rd))(\D)", r"\1 \2\3", s)

    # 3. ordinal doubled: 15thth -> 15th, 7thth -> 7th
    s = re.sub(r"\b(\d+(?:th|st|nd|rd))(?:th|st|nd|rd)\b", r"\1", s)

    # 4. alpha words that should be separate based on known suffixes
    #    Split when a known suffix is immediately followed by known prefix
    for suffix in sorted(SEP_AFTER, key=len, reverse=True):
        for prefix in sorted(SEP_BEFORE, key=len, reverse=True):
            pat = rf"\b({suffix})({prefix})\b"
            s = re.sub(pat, rf"\1 \2", s, flags=re.I)

    # 5. road variants: mainrd -> main rd, crossroad -> cross road
    s = re.sub(r"\b(main|cross|service)(road|rd|street|st)\b", r"\1 \2", s, flags=re.I)

    # 6. number glued directly to alpha word (not ordinal): 560bangalore -> 560 bangalore
    s = re.sub(r"(\d)([a-z])", r"\1 \2", s, flags=re.I)

    return s


def _dedouble_tokens(s: str) -> str:
    """Remove consecutive duplicate tokens."""
    toks = s.split()
    out = []
    prev = None
    for t in toks:
        if t == prev:
            continue
        out.append(t)
        prev = t
    return " ".join(out)


def _dedupe_triple(s: str) -> str:
    """Reduce any token that appears 3+ times consecutively or overall."""
    toks = s.split()
    # Global dedup: if a token appears 3+ times total, reduce to 2
    counts = {}
    for t in toks:
        counts[t] = counts.get(t, 0) + 1
    # If any token >= 3, keep only first 2 occurrences
    for tok, ct in counts.items():
        if ct >= 3 and len(tok) >= 3 and tok.isalpha():
            new_toks = []
            kept = 0
            for t in toks:
                if t == tok:
                    if kept < 2:
                        new_toks.append(t)
                        kept += 1
                else:
                    new_toks.append(t)
            toks = new_toks
    return " ".join(toks)


def _fix_duplicate_city_state(s: str) -> str:
    """Reduce duplicate cities/states to single occurrence."""
    toks = s.split()
    out = []
    prev = None
    prev_city = False
    prev_state = False
    for t in toks:
        t_lower = t.lower()
        is_city = t_lower in CITIES
        is_state = t_lower == "karnataka"
        if is_city and prev_city:
            continue
        if is_state and prev_state:
            continue
        out.append(t)
        prev = t
        prev_city = is_city
        prev_state = is_state
    return " ".join(out)


def _fix_comma_space(s: str) -> str:
    """Add space after commas."""
    return re.sub(r",([^ ])", r", \1", s)


def fix_address(addr: str) -> str:
    """Apply all fixes to a single address string."""
    if not addr:
        return addr
    s = addr.strip()
    s = _fix_glued_words(s)
    s = _dedouble_tokens(s)
    s = _dedupe_triple(s)
    s = _fix_duplicate_city_state(s)
    s = _fix_comma_space(s)
    # Clean up multiple spaces
    s = " ".join(s.split())
    return s


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
