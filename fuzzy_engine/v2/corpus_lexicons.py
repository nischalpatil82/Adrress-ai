"""
fuzzy_engine.v2.corpus_lexicons
================================
Build data-driven road/locality name sets from the address corpus.

Ported from v1 lexicons.py + matcher._detect_areas.
These sets are used by the speller and matcher to recognize when a
token is a known geographic name (and therefore should NOT be blindly
corrected into a dictionary word).

Usage:
    lex = CorpusLexicons.build(AddressPipeline.from_config().retriever.addresses)
    "koramangala" in lex.locality_names  # True
    "bannerghatta" in lex.road_names     # True
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from fuzzy_engine.dictionaries import HARDCODED_AREAS, KNOWN_CITIES
from fuzzy_engine.v2.normalize import (
    LOCALITY_SUFFIXES,
    ROAD_SUFFIXES,
    normalize_text,
)

# Area tokens that survive the frequency filter.
MIN_AREA_FREQUENCY = 3

_AREA_STOP_WORDS = {
    "house", "flat", "no", "road", "street", "main", "cross", "near",
    "opposite", "behind", "beside", "floor", "block", "phase", "stage",
    "bangalore", "bengaluru", "karnataka", "india", "apartment",
    "building", "tower", "complex", "mall", "layout", "colony", "nagar",
    "city", "district", "state", "pin", "pincode", "post", "office",
}


def _area_name_tokens(addresses: list[str]) -> set[str]:
    """Auto-detect area names by corpus frequency (ported from matcher._detect_areas)."""
    freq: Counter[str] = Counter()
    for addr in addresses:
        for tok in normalize_text(addr).split():
            if (tok.isalpha() and len(tok) > 3
                    and tok not in _AREA_STOP_WORDS
                    and tok not in KNOWN_CITIES):
                freq[tok] += 1
    areas = {w for w, c in freq.items() if c >= MIN_AREA_FREQUENCY}
    areas.update(HARDCODED_AREAS)
    return areas


@dataclass(frozen=True)
class CorpusLexicons:
    road_names: set[str] = field(default_factory=set)
    locality_names: set[str] = field(default_factory=set)
    area_names: set[str] = field(default_factory=set)
    geo_names: set[str] = field(default_factory=set)

    @classmethod
    def build(cls, addresses: Iterable[str]) -> "CorpusLexicons":
        road_names: set[str] = set()
        locality_names: set[str] = set(HARDCODED_AREAS)
        geo_freq: Counter[str] = Counter()

        for addr in addresses:
            toks = normalize_text(addr).split()
            for i, tok in enumerate(toks):
                if not tok.isalpha() or len(tok) < 5 or tok in KNOWN_CITIES:
                    continue
                prev_tok = toks[i - 1] if i > 0 else ""
                next_tok = toks[i + 1] if i + 1 < len(toks) else ""

                # Road name: token adjacent to a road suffix
                if next_tok in ROAD_SUFFIXES or prev_tok in ROAD_SUFFIXES:
                    road_names.add(tok)
                    geo_freq[tok] += 2

                # Locality name: token adjacent to locality suffix or ends
                # with a known Bangalore-area suffix.
                if (
                    next_tok in LOCALITY_SUFFIXES
                    or prev_tok in LOCALITY_SUFFIXES
                    or tok.endswith(("halli", "nagar", "puram", "pura",
                                     "kere", "palya", "gudi", "kunte"))
                ):
                    locality_names.add(tok)
                    geo_freq[tok] += 2
                else:
                    geo_freq[tok] += 1

        # Frequency-based geo_names (>=2 occurrences from any source)
        geo_names = {w for w, c in geo_freq.items() if c >= 2}
        geo_names.update(road_names)
        geo_names.update(locality_names)
        geo_names.update(_area_name_tokens(list(addresses)))

        return cls(
            road_names=road_names,
            locality_names=locality_names,
            area_names=_area_name_tokens(list(addresses)),
            geo_names=geo_names,
        )
