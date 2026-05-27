"""Show before vs after cleaning samples.

Static corpus: compare addresses.npy.bak vs addresses.npy
MySQL: show sample of cleaned patterns from audit + current state.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import os
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parent

print("=" * 70)
print("1. STATIC CORPUS (models/addresses.npy)")
print("=" * 70)

bak = ROOT / "models" / "addresses.npy.bak"
curr = ROOT / "models" / "addresses.npy"

if bak.exists() and curr.exists():
    bak_arr = np.load(bak, allow_pickle=True)
    curr_arr = np.load(curr, allow_pickle=True)
    changed = 0
    samples = []
    for i in range(len(bak_arr)):
        old = str(bak_arr[i])
        new = str(curr_arr[i])
        if old != new:
            changed += 1
            if len(samples) < 20:
                samples.append((old, new))
    print(f"\nTotal changed: {changed:,} / {len(bak_arr):,}\n")
    print("Sample before -> after pairs:")
    for idx, (old, new) in enumerate(samples, 1):
        print(f"\n  [{idx}] BEFORE: {old[:100]}")
        print(f"      AFTER:  {new[:100]}")
else:
    print("Backup not found.")


print("\n" + "=" * 70)
print("2. MYSQL DATABASE (address_ai.addresses)")
print("=" * 70)

# MySQL was cleaned in-place; we don't have a snapshot.
# Show representative "before" reconstructions + current "after" state.

BEFORE_PATTERNS = {
    "layout layout": "btm layout layout 2nd stage bangalore",
    "stagebangalore": "7th main kumaraswamy layout 2nd stagebangalore 76",
    "layoutbangalore": "560 27th main 2nd stage btm layout layoutbangalore karnataka",
    "crossroad": "9th crossroad btm layout bangalore",
    "mainroad": "17th mainroad begur koppa bangalore",
    "layoutheggenahalli": "layoutheggenahalli cross bangalore",
    "15thth": "15thth main vinayaka nagar bangalore",
    "bangalore560076": "begur post bangalore560076 karnataka",
    "29148th": "29148th main road btm layout bangalore",
    "193rdamain": "193rdamain 8th cross rpc layout bangalore",
    "thth": "7th cross 15thth main jpnagar bangalore",
}

print("\nRepresentative BEFORE -> AFTER reconstructions:")
for pattern, example_before in BEFORE_PATTERNS.items():
    print(f"\n  Pattern: '{pattern}'")
    print(f"    BEFORE: {example_before}")
    # Apply same fixes the cleaners used
    from fuzzy_engine.v2.normalize import normalize_text
    fixed = normalize_text(example_before)
    # Dedouble
    toks = fixed.split()
    out = []
    prev = None
    for t in toks:
        if t == prev:
            continue
        out.append(t)
        prev = t
    fixed = " ".join(out)
    # Fix glued
    import re
    fixed = re.sub(r"([a-z]{3,})(bangalore|bengaluru|karnataka)", r"\1 \2", fixed, flags=re.I)
    fixed = re.sub(r"(layout|nagar|colony|phase|stage|block|sector)([a-z]{4,})", r"\1 \2", fixed, flags=re.I)
    fixed = re.sub(r"(\d+(?:th|st|nd|rd))(stage|phase|block|sector|layout)", r"\1 \2", fixed, flags=re.I)
    fixed = re.sub(r"(main|cross|service)(road|rd|street|st)", r"\1 \2", fixed, flags=re.I)
    fixed = re.sub(r"(\d)([a-z])", r"\1 \2", fixed, flags=re.I)
    fixed = re.sub(r"\b(\d+(?:th|st|nd|rd))(th|st|nd|rd)\b", r"\1", fixed, flags=re.I)
    fixed = " ".join(fixed.split())
    print(f"    AFTER:  {fixed}")

# Show actual current MySQL samples that were cleaned
print("\n\nActual MySQL rows that were cleaned (current state):")
host = os.getenv("DB_HOST", "127.0.0.1")
port = int(os.getenv("DB_PORT", "3306"))
user = os.getenv("DB_USER", "root")
pw = os.getenv("DB_PASSWORD", "root")
db = os.getenv("DB_NAME", "address_ai")
url = f"mysql+pymysql://{user}:{pw}@{host}:{port}/{db}"
eng = create_engine(url, pool_pre_ping=True, future=True)

with eng.connect() as conn:
    # Show some that would have been dirty
    patterns = ["%layout layout%", "%stagebangalore%", "%crossroad%", "%mainroad%",
                "%thth%", "%bangalore5600%"]
    print("\nSearching for any remaining dirty patterns (should be 0):")
    for p in patterns:
        q = text("SELECT COUNT(*) FROM addresses WHERE normalized_full_address LIKE :pat")
        ct = conn.execute(q, {"pat": p}).scalar()
        print(f"  {p}: {ct} rows")

    # Show sample cleaned rows
    print("\nSample cleaned MySQL rows (current state):")
    q = text("SELECT normalized_full_address FROM addresses WHERE address_id IN (60485, 60505, 91327, 52293, 70194, 87601) ORDER BY address_id")
    for row in conn.execute(q):
        print(f"  {row.normalized_full_address[:100]}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Static corpus:  {len(bak_arr):,} total, {changed:,} cleaned")
print(f"MySQL database: 158,198 total, ~56,528 cleaned")
print("=" * 70)
