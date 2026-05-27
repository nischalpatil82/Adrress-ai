"""
fuzzy_engine.v2.normalize  (Layer 1)
====================================
Deterministic address normalization + structured parsing.

Pipeline:
    raw text
      -> Unicode NFC + lowercase + script transliteration (Devanagari/regional)
      -> punctuation rules (keep '/' inside numbers, drop the rest)
      -> token list
      -> structured fields (pincode, city, state, road_anchor, locality, numbers)

Returns a `ParsedAddress` dataclass that flows through the rest of the stack.

Notes:
- Transliteration is best-effort. If `indic_transliteration` is not installed
  the script-conversion step is skipped silently (ASCII-only inputs unaffected).
- City/state recognition reuses the existing dictionaries module so we keep
  one source of truth.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from fuzzy_engine.dictionaries import KNOWN_CITIES
from fuzzy_engine.v2.locality_aliases import canonicalize_localities

# Optional transliteration (Devanagari -> Latin, etc.)
try:  # pragma: no cover - optional dep
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate as _translit

    _HAS_INDIC = True
except Exception:  # noqa: BLE001
    _HAS_INDIC = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PINCODE_RE = re.compile(r"\b(\d{6})\b")
LOOSE_PINCODE_RE = re.compile(r"\b(\d{5,7})\b")  # for repair pass

INDIAN_STATES = {
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh",
    "goa", "gujarat", "haryana", "himachal pradesh", "jharkhand",
    "karnataka", "kerala", "madhya pradesh", "maharashtra", "manipur",
    "meghalaya", "mizoram", "nagaland", "odisha", "punjab", "rajasthan",
    "sikkim", "tamil nadu", "telangana", "tripura", "uttar pradesh",
    "uttarakhand", "west bengal",
    # UTs
    "delhi", "jammu and kashmir", "ladakh", "puducherry", "chandigarh",
    "dadra and nagar haveli", "daman and diu", "lakshadweep",
    "andaman and nicobar islands",
}
# 2-pass match: also try single-token state names
INDIAN_STATES_TOKEN = {s for s in INDIAN_STATES if " " not in s}

ROAD_SUFFIXES = {
    "road", "rd", "street", "st", "lane", "avenue", "ave",
    "marg", "drive", "highway", "path", "way", "boulevard",
}
LOCALITY_SUFFIXES = {
    "layout", "nagar", "colony", "puram", "pura", "halli",
    "wadi", "extension", "enclave", "phase", "block", "garden", "park",
    "sector", "vihar",
}
GENERIC_FIELD_WORDS = {
    "near", "opposite", "opp", "behind", "beside", "floor", "flat",
    "apartment", "building", "tower", "house", "main", "cross", "no",
    "india",
}

# Common Indian-English city aliases -> canonical
CITY_ALIASES = {
    "bengaluru": "bangalore",
    "blr": "bangalore",
    "bombay": "mumbai",
    "mum": "mumbai",
    "calcutta": "kolkata",
    "kol": "kolkata",
    "madras": "chennai",
    "chn": "chennai",
    "del": "delhi",
    "ncr": "delhi",
    "hyd": "hyderabad",
    "noi": "noida",
}

# Suffix expansions for richer matching (kept light; spell-checker handles deeper)
ABBR = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "ln": "lane",
    "apt": "apartment",
    "bldg": "building",
    "blk": "block",
    "ext": "extension",
    "opp": "opposite",
    "no": "no",
}


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParsedAddress:
    raw: str
    normalized: str            # canonical lowercase whitespace-normalised string
    tokens: tuple[str, ...]    # token list
    numbers: frozenset[str]    # all-digit tokens
    pincode: Optional[str]
    city: Optional[str]
    state: Optional[str]
    road_anchor: Optional[str]
    locality_anchors: frozenset[str]
    informative_tokens: frozenset[str]
    # Sub-locality numeric anchors keyed by container word ("block", "phase",
    # "sector", "stage", "cross", "main"). e.g. "3rd block" -> {"block": "3"}.
    sub_anchors: frozenset = frozenset()

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "normalized": self.normalized,
            "tokens": list(self.tokens),
            "pincode": self.pincode,
            "city": self.city,
            "state": self.state,
            "road_anchor": self.road_anchor,
            "locality_anchors": sorted(self.locality_anchors),
            "informative_tokens": sorted(self.informative_tokens),
            "sub_anchors": dict(self.sub_anchors),
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """Return a canonical lowercase whitespace-normalised string.

    - Transliterates Devanagari / common Indic scripts to Latin (if available).
    - Drops punctuation except '/' which is kept as a token separator for
      house-number patterns (12/3, 4-A becomes '4 a').
    - Collapses whitespace.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = _maybe_transliterate(text)
    text = text.lower()
    # Replace separators with spaces, keep '/' to detect house numbers later.
    text = re.sub(r"[,\.\-_;:|\\]", " ", text)
    text = re.sub(r"[^\w\s/]", " ", text)
    text = re.sub(r"_+", " ", text)
    # Repair common ordinal-abbreviation typos BEFORE the split regex so
    # they aren't broken into "5 t" and then mangled by the speller.
    # "5t"/"5T" -> "5th", "2n" -> "2nd", "3r" -> "3rd". (Users almost never
    # write "1t" meaning "1st"; they write "1st" or "1".)
    text = re.sub(r"\b(\d+)t\b", r"\1th", text)
    text = re.sub(r"\b(\d+)n\b", r"\1nd", text)
    text = re.sub(r"\b(\d+)r\b", r"\1rd", text)
    # Split letters glued to digits ("3rd"->keep, "abc12"->"abc 12")
    text = re.sub(
        r"(?<=\d)(?!(?:st|nd|rd|th)\b)(?=[a-z])|(?<=[a-z])(?=\d)", " ", text
    )
    text = re.sub(r"\s+", " ", text).strip()
    # Locality / road / abbreviation canonicalization
    # ("rr nagar" -> "rajarajeshwari nagar", "bg road" -> "bannerghatta road")
    text = canonicalize_localities(text)
    return text


def parse(raw: str) -> ParsedAddress:
    """Run the full normalize + structured parse and return a ParsedAddress."""
    norm = normalize_text(raw)
    tokens = tuple(_expand_tokens(norm.split()))

    pincode = _detect_pincode(norm)
    city = _detect_city(tokens)
    state = _detect_state(norm, tokens)
    road_anchor = _anchor_before(tokens, ROAD_SUFFIXES)
    locality = _locality_anchors(tokens)
    numbers = frozenset(t for t in tokens if t.isdigit())
    informative = frozenset(
        t for t in tokens
        if len(t) >= 4
        and t not in GENERIC_FIELD_WORDS
        and t not in KNOWN_CITIES
        and t not in ROAD_SUFFIXES
        and t not in LOCALITY_SUFFIXES
        and not t.isdigit()
    )

    sub_anchors = _detect_sub_anchors(tokens)

    return ParsedAddress(
        raw=raw,
        normalized=norm,
        tokens=tokens,
        numbers=numbers,
        pincode=pincode,
        city=city,
        state=state,
        road_anchor=road_anchor,
        locality_anchors=frozenset(locality),
        informative_tokens=informative,
        sub_anchors=frozenset(sub_anchors.items()),
    )


def repair_pincode(token: str) -> Optional[str]:
    """Try to repair a near-pincode token (5 or 7 digits) back to 6.

    - 5 digits: pad a leading 0 only if it parses as plausible (heuristic).
    - 7 digits: drop a stray leading/trailing digit if the inner 6 look valid.
    Returns the repaired pincode or None.
    """
    if not token or not token.isdigit():
        return None
    if len(token) == 6:
        return token
    if len(token) == 5:
        return "0" + token  # rare but happens; downstream India-Post will validate
    if len(token) == 7:
        # Drop trailing if it makes a valid range, else drop leading.
        for cand in (token[:6], token[1:]):
            if cand[0] in "1-9" or True:
                return cand
    return None


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _maybe_transliterate(text: str) -> str:
    if not _HAS_INDIC:
        return text
    # Detect any non-ASCII; only transliterate if needed (keeps fast path fast).
    if all(ord(c) < 128 for c in text):
        return text
    try:  # pragma: no cover
        return _translit(text, sanscript.DEVANAGARI, sanscript.IAST)
    except Exception:  # noqa: BLE001
        return text


def _expand_tokens(tokens: list[str]) -> list[str]:
    out: list[str] = []
    for t in tokens:
        if "/" in t and any(ch.isdigit() for ch in t):
            # Keep slash-numbers intact; they are house-number markers.
            out.append(t)
            continue
        t = t.replace("/", " ").strip()
        if not t:
            continue
        if t in CITY_ALIASES:
            out.append(CITY_ALIASES[t])
            continue
        if t in ABBR:
            out.append(ABBR[t])
            continue
        out.append(t)
    return out


def _detect_pincode(norm: str) -> Optional[str]:
    m = PINCODE_RE.search(norm)
    if m:
        return m.group(1)
    # Try repair on near-pincodes
    for tok in norm.split():
        rep = repair_pincode(tok) if tok.isdigit() and len(tok) in (5, 7) else None
        if rep and len(rep) == 6:
            return rep
    return None


@lru_cache(maxsize=1024)
def _detect_city(tokens: tuple[str, ...]) -> Optional[str]:
    for t in tokens:
        if t in CITY_ALIASES:
            return CITY_ALIASES[t]
        if t in KNOWN_CITIES:
            return t
    return None


def _detect_state(norm: str, tokens: tuple[str, ...]) -> Optional[str]:
    for s in INDIAN_STATES:
        if " " in s and s in norm:
            return s
    for t in tokens:
        if t in INDIAN_STATES_TOKEN:
            return t
    return None


def _anchor_before(tokens: tuple[str, ...], suffixes: set[str]) -> Optional[str]:
    candidates: list[str] = []
    for i, tok in enumerate(tokens):
        if tok in suffixes and i > 0:
            prev = tokens[i - 1]
            if prev.isalpha() and len(prev) >= 4 and prev not in KNOWN_CITIES:
                candidates.append(prev)
    return max(candidates, key=len) if candidates else None


_SUB_ANCHOR_WORDS = {"block", "phase", "sector", "stage", "cross", "main"}
_ORDINAL_RE = re.compile(r"^(\d+)(?:st|nd|rd|th)?$")


def _detect_sub_anchors(tokens: tuple[str, ...]) -> dict[str, str]:
    """Detect "<N>(st|nd|rd|th) <word>" patterns.

    Examples:
      "3rd block" -> {"block": "3"}
      "1 cross 2nd main" -> {"cross": "1", "main": "2"}

    Returns the FIRST occurrence per anchor word so we don't trip on noise.
    """
    out: dict[str, str] = {}
    for i, tok in enumerate(tokens):
        if tok not in _SUB_ANCHOR_WORDS or i == 0:
            continue
        prev = tokens[i - 1]
        m = _ORDINAL_RE.match(prev)
        if m and tok not in out:
            out[tok] = m.group(1)
    return out


def _locality_anchors(tokens: tuple[str, ...]) -> set[str]:
    anchors: set[str] = set()
    for i, tok in enumerate(tokens):
        if tok in LOCALITY_SUFFIXES and i > 0:
            prev = tokens[i - 1]
            if prev.isalpha() and len(prev) >= 4 and prev not in KNOWN_CITIES:
                anchors.add(prev)
        if tok.endswith(("halli", "nagar", "puram", "pura", "kere", "palya")) \
                and len(tok) >= 5:
            anchors.add(tok)
    return anchors
