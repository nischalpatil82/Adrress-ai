"""
fuzzy_engine.parser
===================
Small, deterministic address parser used by the fuzzy-first pipeline.

This is intentionally conservative. It extracts only fields we can use for
matching guardrails without pretending to fully understand every address.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fuzzy_engine.dictionaries import KNOWN_CITIES
from fuzzy_engine.normalizer import normalize


ROAD_SUFFIXES = {"road", "rd", "street", "st", "lane", "avenue", "ave", "marg", "drive", "highway"}
LOCALITY_SUFFIXES = {
    "layout", "nagar", "colony", "puram", "pura", "halli", "wadi",
    "extension", "enclave", "phase", "block", "garden", "park",
}
GENERIC_FIELD_WORDS = {
    "near", "opposite", "opp", "behind", "beside", "floor", "flat",
    "apartment", "building", "tower", "house", "main", "cross", "road",
    "street", "lane", "layout", "nagar", "colony", "phase", "block",
    "india", "karnataka",
}


@dataclass(frozen=True)
class ParsedAddress:
    normalized: str
    tokens: list[str] = field(default_factory=list)
    numbers: set[str] = field(default_factory=set)
    pincode: str | None = None
    city: str | None = None
    road_anchor: str | None = None
    locality_anchors: set[str] = field(default_factory=set)
    informative_tokens: set[str] = field(default_factory=set)


def _anchor_before_suffix(tokens: list[str], suffixes: set[str]) -> str | None:
    candidates = []
    for i, tok in enumerate(tokens):
        if tok not in suffixes or i == 0:
            continue
        prev = tokens[i - 1]
        if prev.isalpha() and len(prev) >= 4 and prev not in KNOWN_CITIES:
            candidates.append(prev)
    return max(candidates, key=len) if candidates else None


def _locality_anchors(tokens: list[str]) -> set[str]:
    anchors = set()
    for i, tok in enumerate(tokens):
        if tok in LOCALITY_SUFFIXES and i > 0:
            prev = tokens[i - 1]
            if prev.isalpha() and len(prev) >= 4 and prev not in KNOWN_CITIES:
                anchors.add(prev)
        if tok.endswith(("halli", "nagar", "puram", "pura", "kere", "palya")) and len(tok) >= 5:
            anchors.add(tok)
    return anchors


def parse_address(text: str) -> ParsedAddress:
    normalized = normalize(text)
    tokens = normalized.split()
    numbers = {tok for tok in tokens if tok.isdigit()}
    pincode = next((tok for tok in tokens if tok.isdigit() and len(tok) == 6), None)
    city = next((tok for tok in tokens if tok in KNOWN_CITIES), None)
    road_anchor = _anchor_before_suffix(tokens, ROAD_SUFFIXES)
    locality_anchors = _locality_anchors(tokens)
    informative = {
        tok for tok in tokens
        if len(tok) >= 4
        and tok not in GENERIC_FIELD_WORDS
        and tok not in KNOWN_CITIES
    }
    return ParsedAddress(
        normalized=normalized,
        tokens=tokens,
        numbers=numbers,
        pincode=pincode,
        city=city,
        road_anchor=road_anchor,
        locality_anchors=locality_anchors,
        informative_tokens=informative,
    )
