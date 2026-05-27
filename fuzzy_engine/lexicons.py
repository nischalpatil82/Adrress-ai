"""
fuzzy_engine.lexicons
=====================
Reusable lexicon builder for the fuzzy-first correction engine.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from fuzzy_engine.dictionaries import COMMON_MISSPELLINGS, HARDCODED_AREAS, KNOWN_CITIES
from fuzzy_engine.normalizer import normalize
from fuzzy_engine.parser import LOCALITY_SUFFIXES, ROAD_SUFFIXES


DEFAULT_SUPPLEMENTAL_JSONL = Path("data/address_training_kier_v1_strict_clean.jsonl")


@dataclass(frozen=True)
class AddressLexicons:
    road_names: set[str] = field(default_factory=set)
    locality_names: set[str] = field(default_factory=set)
    geo_names: set[str] = field(default_factory=set)

    @classmethod
    def build(
        cls,
        addresses: list[str],
        supplemental_jsonl: Path = DEFAULT_SUPPLEMENTAL_JSONL,
    ) -> "AddressLexicons":
        road_names: set[str] = set()
        locality_names: set[str] = set(HARDCODED_AREAS)
        geo_freq = Counter()

        def absorb_text(text: str, weight: int = 1) -> None:
            toks = normalize(text).split()
            for i, tok in enumerate(toks):
                if not tok.isalpha() or len(tok) < 5 or tok in KNOWN_CITIES:
                    continue
                prev_tok = toks[i - 1] if i > 0 else ""
                next_tok = toks[i + 1] if i + 1 < len(toks) else ""
                if next_tok in ROAD_SUFFIXES or prev_tok in ROAD_SUFFIXES:
                    road_names.add(tok)
                    geo_freq[tok] += weight + 1
                if (
                    next_tok in LOCALITY_SUFFIXES
                    or prev_tok in LOCALITY_SUFFIXES
                    or tok.endswith(("halli", "nagar", "puram", "pura", "kere", "palya"))
                ):
                    locality_names.add(tok)
                    geo_freq[tok] += weight + 1
                else:
                    geo_freq[tok] += weight

        for addr in addresses:
            absorb_text(addr, weight=2)

        if supplemental_jsonl.exists():
            try:
                with supplemental_jsonl.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        absorb_text(row.get("clean_target", ""), weight=1)
            except OSError:
                pass

        for value in COMMON_MISSPELLINGS.values():
            toks = normalize(value).split()
            if len(toks) == 1 and len(toks[0]) >= 5:
                geo_freq[toks[0]] += 2

        geo_names = {word for word, count in geo_freq.items() if count >= 2}
        geo_names.update(road_names)
        geo_names.update(locality_names)

        return cls(
            road_names=road_names,
            locality_names=locality_names,
            geo_names=geo_names,
        )
