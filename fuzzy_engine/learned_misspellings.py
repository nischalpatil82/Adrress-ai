"""
fuzzy_engine.learned_misspellings
=================================
Data-driven misspelling learning for the address correction engine.

This module mines stable token-level substitutions from the noisy/clean
training dataset and stores them in a cache so the runtime spell checker can
reuse common corrections without hardcoding every typo.

A sanitization pass removes circular mappings, entries that corrupt valid
words, and entries that conflict with the authoritative COMMON_MISSPELLINGS
dictionary.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, Tuple

from fuzzy_engine.dictionaries import KNOWN_CITIES
from fuzzy_engine.normalizer import normalize


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_JSONL = ROOT_DIR / "data" / "address_training_kier_v1_strict_clean.jsonl"
DEFAULT_CACHE_PATH = ROOT_DIR / "data" / "learned_misspellings.json"
CACHE_VERSION = 2


def _iter_pairs(dataset_path: Path) -> Iterable[Tuple[str, str]]:
    if not dataset_path.exists():
        return []

    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            noisy = str(row.get("noisy_input", "")).strip()
            clean = str(row.get("clean_target", "")).strip()
            if noisy and clean:
                yield noisy, clean


def _extract_token_pairs(noisy: str, clean: str) -> Iterable[Tuple[str, str]]:
    noisy_tokens = normalize(noisy).split()
    clean_tokens = normalize(clean).split()
    if not noisy_tokens or not clean_tokens:
        return []

    matcher = SequenceMatcher(a=noisy_tokens, b=clean_tokens, autojunk=False)
    pairs = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        src = noisy_tokens[i1:i2]
        dst = clean_tokens[j1:j2]

        if len(src) == 1 and len(dst) == 1:
            if src[0] != dst[0]:
                pairs.append((src[0], dst[0]))
            continue

        if tag == "replace" and len(src) == len(dst):
            for noisy_tok, clean_tok in zip(src, dst):
                if noisy_tok != clean_tok:
                    pairs.append((noisy_tok, clean_tok))

    return pairs


def build_learned_misspellings(
    dataset_path: Path = DEFAULT_TRAIN_JSONL,
    min_count: int = 3,
    min_support_ratio: float = 0.60,
) -> Dict[str, str]:
    """Mine stable noisy->clean token substitutions from the training set."""
    pair_counts = defaultdict(Counter)

    for noisy, clean in _iter_pairs(Path(dataset_path)):
        for src, dst in _extract_token_pairs(noisy, clean):
            if not src.isalpha() or not dst.isalpha():
                continue
            if len(src) < 3 or len(dst) < 3:
                continue
            pair_counts[src][dst] += 1

    learned = {}
    for src, counter in pair_counts.items():
        dst, count = counter.most_common(1)[0]
        total = sum(counter.values())

        is_city_variant = dst in KNOWN_CITIES
        required_count = 2 if is_city_variant else min_count
        required_ratio = 0.50 if is_city_variant else min_support_ratio

        if count < required_count:
            continue
        if total and (count / total) < required_ratio:
            continue
        if src == dst:
            continue
        learned[src] = dst

    return _sanitize_learned(dict(sorted(learned.items())))


# ── Words that must NEVER be treated as misspellings ─────────────────────
_PROTECTED_VALID_WORDS = {
    # Common address structural words
    "stage", "main", "road", "street", "floor", "near", "plot", "flat",
    "house", "building", "apartment", "layout", "nagar", "colony", "sector",
    "block", "phase", "cross", "lane", "avenue", "marg", "garden", "tower",
    "complex", "society", "enclave", "extension", "villa", "bridge", "park",
    "market", "circle", "drive", "plaza", "square", "place", "highway",
    "temple", "church", "school", "hospital", "bank", "post", "office",
    "junction", "station", "chowk", "vihar", "puram", "ganj", "bazar",
    "gate", "india", "north", "south", "east", "west",
    "pura", "puram", "bazaar", "bazar", "vihar", "ganj",
    # Cities that must stay as-is
    "bangalore", "mumbai", "delhi", "hyderabad", "chennai", "kolkata",
    "pune", "ahmedabad", "jaipur", "noida", "gurgaon", "surat",
    "lucknow", "nagpur", "indore", "bhopal", "patna", "vadodara",
    "chandigarh", "kochi", "mysuru", "coimbatore", "mangalore",
    # Common Bangalore areas
    "hoskote", "nagarbhavi", "banashankari", "koramangala",
    "whitefield", "indiranagar", "jayanagar", "basavanagudi",
    "malleshwaram", "rajajinagar", "yelahanka", "hebbal",
    "marathahalli", "sarjapur", "bannerghatta", "electronic",
}


def _sanitize_learned(mappings: Dict[str, str]) -> Dict[str, str]:
    """Remove poisoned entries from learned misspellings.

    Removes:
      1. Entries where the source is a known valid word (e.g. stage→satge)
      2. Circular mappings (A→B→A)
      3. Entries that conflict with COMMON_MISSPELLINGS
    """
    from fuzzy_engine.dictionaries import COMMON_MISSPELLINGS

    # Build set of authoritative correct forms
    authoritative_correct = set(COMMON_MISSPELLINGS.values())
    for v in authoritative_correct.copy():
        for word in v.split():
            authoritative_correct.add(word)

    protected = _PROTECTED_VALID_WORDS | authoritative_correct

    clean = {}
    for src, dst in mappings.items():
        # Rule 1: Never corrupt a valid word
        if src in protected:
            continue

        # Rule 2: If COMMON_MISSPELLINGS already maps src, skip learned
        if src in COMMON_MISSPELLINGS:
            continue

        # Rule 3: Detect circular mappings (A→B and B→A)
        if dst in mappings and mappings[dst] == src:
            continue

        # Rule 4: Don't map TO something that COMMON_MISSPELLINGS would
        # then remap differently (chain conflict)
        if dst in COMMON_MISSPELLINGS and COMMON_MISSPELLINGS[dst] != dst:
            # Map directly to the final canonical form instead
            final = COMMON_MISSPELLINGS[dst]
            if final != src:  # avoid no-ops
                clean[src] = final
            continue

        clean[src] = dst

    return clean


def load_learned_misspellings(
    dataset_path: Path = DEFAULT_TRAIN_JSONL,
    cache_path: Path = DEFAULT_CACHE_PATH,
) -> Dict[str, str]:
    """Load learned misspellings from cache or rebuild them if needed."""
    dataset_path = Path(dataset_path)
    cache_path = Path(cache_path)

    if cache_path.exists():
        try:
            if not dataset_path.exists() or cache_path.stat().st_mtime >= dataset_path.stat().st_mtime:
                with cache_path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)

                if isinstance(data, dict) and "mappings" in data:
                    version = int(data.get("version", 0))
                    mappings = data.get("mappings", {})
                    if version == CACHE_VERSION and isinstance(mappings, dict):
                        return {
                            str(k): str(v)
                            for k, v in mappings.items()
                            if k and v
                        }
        except Exception:
            pass

    learned = build_learned_misspellings(dataset_path=dataset_path)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "version": CACHE_VERSION,
                    "mappings": learned,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
    except Exception:
        pass
    return learned


def get_merged_misspellings() -> Dict[str, str]:
    """Return the learned table. Kept as a narrow helper for callers."""
    return load_learned_misspellings()
