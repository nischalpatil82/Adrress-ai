"""Clean the static address corpus (models/addresses.npy).

Fixes applied:
  1. Re-run normalize_text + locality alias canonicalization.
  2. Fix stuck-together words  (e.g. "stagebangalore" -> "stage bangalore").
  3. De-double consecutive identical tokens (e.g. "layout layout" -> "layout").
  4. Strip orphaned punctuation / ordinals.
  5. Backup old .npy before overwriting.
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "models" / "addresses.npy"
BAK = ROOT / "models" / "addresses.npy.bak"


def _fix_spacing(s: str) -> str:
    """Separate words that got glued together."""
    # "stagebangalore" -> "stage bangalore"
    s = re.sub(r"([a-z]{3,})(bangalore|bengaluru|karnataka)", r"\1 \2", s, flags=re.I)
    # "layoutvijayanagar" -> "layout vijayanagar"
    s = re.sub(r"(layout|nagar|colony|phase|stage|block|sector)([a-z]{4,})", r"\1 \2", s, flags=re.I)
    # "2ndstage" -> "2nd stage"
    s = re.sub(r"(\d+(?:th|st|nd|rd))(stage|phase|block|sector|layout)", r"\1 \2", s, flags=re.I)
    # "mainroad" -> "main road"
    s = re.sub(r"(main|cross|service)(road|rd|street|st)", r"\1 \2", s, flags=re.I)
    return s


def _dedouble_tokens(s: str) -> str:
    """Remove consecutive duplicate tokens caused by alias over-expansion."""
    toks = s.split()
    out: list[str] = []
    prev: str | None = None
    for t in toks:
        if t == prev:
            continue
        out.append(t)
        prev = t
    return " ".join(out)


def clean_address(addr: str) -> str:
    from fuzzy_engine.v2.normalize import normalize_text
    # Step 1: full normalization (includes locality aliases)
    s = normalize_text(addr)
    # Step 2: fix glued words
    s = _fix_spacing(s)
    # Step 3: dedouble
    s = _dedouble_tokens(s)
    # Step 4: re-normalize whitespace
    s = " ".join(s.split())
    return s


def main():
    if not SRC.exists():
        raise FileNotFoundError(SRC)
    arr = np.load(SRC, allow_pickle=True)
    n = len(arr)
    print(f"Loaded {n:,} addresses from {SRC}")

    cleaned = []
    changed = 0
    for i, a in enumerate(arr):
        orig = str(a)
        c = clean_address(orig)
        if c != orig:
            changed += 1
        cleaned.append(c)
        if i and i % 20000 == 0:
            print(f"  processed {i:,} ...")

    print(f"Changed {changed:,} / {n:,} addresses")

    # Backup
    if BAK.exists():
        BAK.unlink()
    SRC.rename(BAK)
    print(f"Backup -> {BAK}")

    np.save(SRC, np.array(cleaned, dtype=object))
    print(f"Saved cleaned corpus -> {SRC}")

    # Also save ids.npy untouched (they stay aligned)
    ids_path = ROOT / "models" / "address_ids.npy"
    if ids_path.exists():
        print(f"  (address_ids.npy stays aligned, no change needed)")


if __name__ == "__main__":
    main()
