"""
fuzzy_engine.spell_checker
==========================
Enterprise token-level spell correction engine.

Provides the SpellChecker class which dynamically corrects ANY word
using a 5-stage pipeline:
  1. Exact lookup in hardcoded misspelling dictionary (Tier 0 override)
  2. Check if token is already a valid word in the database
  3. Probabilistic Norvig 1-Edit Check (fixes 1-char typos)
  4. Probabilistic Norvig 2-Edit Check (fixes severe 2-char typos)
  5. Phonetic Check via Indian Soundex (fixes pronunciation mistakes)

Protects short common words from false corrections.
"""

import json
import re
from collections import Counter
from pathlib import Path
from rapidfuzz import fuzz, process

from fuzzy_engine.config import (
    PROTECTED_WORDS,
    MIN_TOKEN_LEN_MISSPELLING,
    MAX_LEN_DIFF_VOCAB,
)
from fuzzy_engine.dictionaries import COMMON_MISSPELLINGS, EXTRA_VOCAB
from fuzzy_engine.learned_misspellings import load_learned_misspellings
from fuzzy_engine.normalizer import normalize
from fuzzy_engine.probabilistic import ProbabilisticEngine
from fuzzy_engine.phonetics import PhoneticEngine


_SLASH_NUM_PATTERN = re.compile(r"\b\d+(?:/\d+)+\b")
_ROAD_SUFFIX_TOKENS = {"road", "rd", "street", "st", "lane", "avenue", "ave", "marg", "drive", "highway"}
_LOCALITY_SUFFIX_TOKENS = {
    "layout", "nagar", "colony", "puram", "pura", "halli", "wadi",
    "extension", "enclave", "phase", "block", "cross", "main",
}
_SUPPLEMENTAL_GEO_JSONL = Path("data/address_training_kier_v1_strict_clean.jsonl")


class SpellChecker:
    """
    Token-level spell correction engine for Indian addresses using
    statistical frequency, edit-distance, and phonetics.
    """

    def __init__(self, addresses: list):
        """
        Initialize the spell checker by building the frequency map.
        """
        # Merge hardcoded and learned typo maps up front so every downstream
        # stage can reuse the same correction table.
        self._learned_misspellings = load_learned_misspellings()
        self._misspelling_map = {
            **self._learned_misspellings,
            **COMMON_MISSPELLINGS,
        }

        # Build frequency map
        self._freqs = self._build_frequency_map(addresses)
        self._vocab = set(self._freqs.keys())
        
        # Initialize engines
        self._prob_engine = ProbabilisticEngine(self._freqs)
        self._phonetic_engine = PhoneticEngine(list(self._vocab))
        
        # Hardcoded + learned dictionary keys
        self._misspelling_keys = list(self._misspelling_map.keys())

        # Build dedicated area/road name vocabulary for relaxed matching
        self._area_vocab = self._build_area_vocab(addresses)
        self._geo_vocab = self._build_geo_vocab(addresses)

        # Known city lexicon for final tail-token normalization.
        from fuzzy_engine.dictionaries import KNOWN_CITIES, HARDCODED_AREAS
        self._known_cities = set(KNOWN_CITIES)
        self._known_city_list = list(KNOWN_CITIES)
        self._hardcoded_areas = set(HARDCODED_AREAS)
        self._generic_address_tokens = {
            "road", "street", "avenue", "lane", "layout", "nagar", "colony",
            "sector", "phase", "block", "main", "cross", "post", "garden",
            "building", "apartment", "tower", "complex", "house", "villa",
            "north", "south", "east", "west", "near", "opposite", "behind",
        }
        self._trusted_short_aliases = {
            "x": "cross",
            "mn": "main",
        }

    # ── Public API ────────────────────────────────────────────────────────

    def correct(self, raw_address: str) -> tuple:
        """
        Spell-correct each word in the address independently.
        """
        normalized = normalize(raw_address)
        # Normalize symbol-heavy house prefixes, e.g. "no9&4" -> "no 9 4".
        normalized = re.sub(r"\b(?:no|num|number)\s*(\d+)\b", r"no \1", normalized)
        tokens = normalized.split()
        corrected = []
        changes = []

        # Track multi-word expansions to avoid duplicating them.
        # e.g. "btm" -> "btm layout" should only happen once, not twice
        # when "btm" appears multiple times in the input.
        _expansion_used = set()

        for i, tok in enumerate(tokens):
            prev_tok = tokens[i - 1] if i > 0 else ""
            next_tok = tokens[i + 1] if i + 1 < len(tokens) else ""
            fixed = self._correct_token(tok, prev_tok=prev_tok, next_tok=next_tok)
            # Normalize intermediate outputs to canonical mapped forms.
            if len(tok) > 2:
                fixed = self._canonicalize_token(fixed)

            # Prevent multi-word expansions from being applied more than once.
            if " " in fixed and tok != fixed:
                if tok in _expansion_used:
                    fixed = tok  # Already expanded once, keep original
                else:
                    # Also revert if the expansion's last word equals the next
                    # token (e.g. "btm"->"btm layout" when next_tok=="layout")
                    expansion_tail = fixed.split()[-1]
                    if next_tok and expansion_tail == next_tok:
                        fixed = tok  # Would create duplicate, skip expansion
                    else:
                        _expansion_used.add(tok)

            # Global safety net for unseen typos: keep original token when
            # the substitution looks too aggressive and is not explicitly
            # supported by the misspelling dictionary.
            if not self._is_safe_correction(tok, fixed):
                fixed = tok

            corrected.append(fixed)
            if fixed != tok:
                changes.append(f"'{tok}' -> '{fixed}'")

        # Final city normalization with stronger tail handling for city variants.
        for i in range(len(corrected)):
            tok = corrected[i]
            if (not tok.isalpha()) or len(tok) < 4 or tok in self._known_cities:
                continue
            best_city = process.extractOne(tok, self._known_city_list, scorer=fuzz.WRatio)
            if not best_city:
                continue
            city, score, _ = best_city
            near_tail = i >= max(0, len(corrected) - 4)
            city_prefix_match = tok[:3] == city[:3] if len(tok) >= 3 and len(city) >= 3 else False
            # Also check 2-char prefix for severely misspelled cities
            city_prefix2_match = tok[:2] == city[:2] if len(tok) >= 4 and len(city) >= 4 else False
            if (
                (score >= 88 and abs(len(city) - len(tok)) <= 4)
                or (near_tail and score >= 80 and city_prefix_match)
                or (near_tail and score >= 82 and city_prefix2_match and abs(len(city) - len(tok)) <= 3)
            ):
                corrected[i] = city
                if corrected[i] != tok:
                    changes.append(f"'{tok}' -> '{city}'")

        corrected_text = " ".join(corrected)
        corrected_text = self._restore_slash_numbers(raw_address, corrected_text)
        return corrected_text, changes

    @staticmethod
    def _restore_slash_numbers(raw_address: str, corrected_text: str) -> str:
        """Preserve house-number patterns like '16/17' in user-facing output."""
        restored = corrected_text
        for match in _SLASH_NUM_PATTERN.findall(str(raw_address).lower()):
            parts = match.split("/")
            if len(parts) < 2:
                continue
            joined = " ".join(parts)
            restored = re.sub(rf"\b{re.escape(joined)}\b", match, restored, count=1)
        return restored

    def _canonicalize_token(self, token: str) -> str:
        """Follow misspelling dictionary links to a stable canonical token."""
        cur = token
        seen = set()
        for _ in range(3):
            if cur in seen:
                break
            seen.add(cur)
            nxt = self._misspelling_map.get(cur)
            if not nxt or nxt == cur or " " in nxt:
                break
            cur = nxt
        return cur

    def _is_explicit_mapping_pair(self, src: str, dst: str) -> bool:
        """True when src can explicitly resolve to dst via dict mappings."""
        cur = src
        seen = set()
        for _ in range(4):
            if cur in seen:
                break
            seen.add(cur)
            nxt = self._misspelling_map.get(cur)
            if not nxt:
                break
            if nxt == dst:
                return True
            if " " in nxt:
                break
            cur = nxt
        return False

    def _is_safe_correction(self, original: str, fixed: str) -> bool:
        """Conservative filter to avoid risky substitutions on unseen input."""
        if original == fixed:
            return True
        if not original or not fixed:
            return False

        # Explicit dictionary mappings are trusted.
        if self._is_explicit_mapping_pair(original, fixed):
            return True
        if self._trusted_short_aliases.get(original) == fixed:
            return True

        # Allow known city normalization near token tail.
        if fixed in self._known_cities:
            return fuzz.ratio(original, fixed) >= 78

        # Allow structured locality/road corrections when token shape still
        # looks close to a trusted geo lexicon candidate.
        if fixed in self._geo_vocab:
            if original[0] != fixed[0]:
                return False
            if self._char_ngram_overlap(original, fixed) < 0.34:
                return False
            return fuzz.WRatio(original, fixed) >= 74

        # For non-address proper nouns (builder/person/local names), require
        # much stronger evidence unless explicitly mapped.
        is_generic = (
            original in self._generic_address_tokens
            or fixed in self._generic_address_tokens
        )
        if not is_generic:
            if abs(len(original) - len(fixed)) > 1:
                return False
            if len(original) >= 6 and len(fixed) >= 6 and original[:3] != fixed[:3]:
                return False
            if fuzz.ratio(original, fixed) < 90:
                return False

        # Reject substitutions that reshape token too much.
        if abs(len(original) - len(fixed)) > 3:
            return False
        if original[0] != fixed[0]:
            return False
        if len(original) >= 5 and len(fixed) >= 5 and original[:2] != fixed[:2]:
            return False
        if fuzz.ratio(original, fixed) < 76:
            return False

        # Prefer corrections that resolve to learned vocabulary.
        return fixed in self._vocab

    @property
    def vocabulary(self) -> set:
        """Return the full vocabulary set (read-only)."""
        return self._vocab.copy()

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return len(self._vocab)

    # ── Private Methods ──────────────────────────────────────────────────

    def _build_frequency_map(self, addresses: list) -> Counter:
        """
        Build exhaustive word frequency mapping from:
          1. Database frequencies (base truth)
          2. EXTRA_VOCAB (given high artificial frequency)
          3. COMMON_MISSPELLINGS targets (given high artificial frequency)
        """
        cnt = Counter()

        # Database words
        for addr in addresses:
            for tok in normalize(addr).split():
                if tok.isalpha() and len(tok) >= 2:
                    cnt[tok] += 1

        # Extra common address words
        for word in EXTRA_VOCAB:
            if word not in cnt:
                cnt[word] = 20  # Artificial high frequency

        # Correct forms from misspellings
        for correct_form in COMMON_MISSPELLINGS.values():
            for word in correct_form.split():
                if word not in cnt:
                    cnt[word] = 15

        for correct_form in self._learned_misspellings.values():
            for word in correct_form.split():
                if word not in cnt:
                    cnt[word] = 15
        
        return cnt

    def _build_area_vocab(self, addresses: list) -> set:
        """
        Build a focused vocabulary of area/road/locality names from addresses.
        These are multi-frequency location tokens (not generic words) used for
        relaxed fuzzy matching of severely misspelled area names.
        """
        from fuzzy_engine.dictionaries import HARDCODED_AREAS, KNOWN_CITIES

        area_freq = Counter()
        # Known area keywords that signal a preceding area name
        area_signals = {
            "road", "nagar", "layout", "colony", "puram", "pura",
            "halli", "wadi", "ganj", "enclave", "extension",
        }
        for addr in addresses:
            toks = normalize(addr).split()
            for tok in toks:
                if (tok.isalpha() and len(tok) >= 5
                        and tok not in KNOWN_CITIES
                        and tok not in PROTECTED_WORDS
                        and tok not in {"floor", "apartment", "building",
                                        "house", "block", "tower",
                                        "complex", "society", "near",
                                        "opposite", "behind", "cross",
                                        "india", "street", "avenue"}):
                    area_freq[tok] += 1

        # Keep words appearing 3+ times as likely area names.
        area_vocab = {w for w, c in area_freq.items() if c >= 3}
        area_vocab.update(HARDCODED_AREAS)
        # Add correct forms from misspelling dict (bannerghatta etc.)
        area_vocab.update(
            v for v in COMMON_MISSPELLINGS.values()
            if len(v) >= 5 and " " not in v
        )
        return list(area_vocab)

    def _build_geo_vocab(self, addresses: list) -> list:
        """Build a broad geo lexicon for localities and road names."""
        from fuzzy_engine.dictionaries import HARDCODED_AREAS, KNOWN_CITIES

        geo_freq = Counter()
        generic_exclude = {
            "india", "street", "avenue", "floor", "apartment", "building",
            "house", "block", "tower", "complex", "society", "near",
            "opposite", "behind", "post", "main", "cross",
        }
        hardcoded_areas = set(HARDCODED_AREAS)
        for addr in addresses:
            toks = normalize(addr).split()
            for i, tok in enumerate(toks):
                if not tok.isalpha() or len(tok) < 5:
                    continue
                if tok in KNOWN_CITIES or tok in PROTECTED_WORDS or tok in generic_exclude:
                    continue
                next_tok = toks[i + 1] if i + 1 < len(toks) else ""
                prev_tok = toks[i - 1] if i > 0 else ""
                looks_geo = (
                    tok in hardcoded_areas
                    or next_tok in _ROAD_SUFFIX_TOKENS
                    or next_tok in _LOCALITY_SUFFIX_TOKENS
                    or prev_tok in _ROAD_SUFFIX_TOKENS
                    or prev_tok in _LOCALITY_SUFFIX_TOKENS
                    or tok.endswith(("halli", "nagar", "puram", "pura", "pet", "kere"))
                )
                geo_freq[tok] += 2 if looks_geo else 1

        geo_vocab = {w for w, c in geo_freq.items() if c >= 2}
        geo_vocab.update(hardcoded_areas)
        geo_vocab.update(self._area_vocab)
        geo_vocab.update(self._load_supplemental_geo_vocab())
        geo_vocab.update(
            v for v in COMMON_MISSPELLINGS.values()
            if len(v) >= 5 and " " not in v
        )
        return sorted(geo_vocab)

    def _load_supplemental_geo_vocab(self) -> set[str]:
        """Load extra locality/road words from cleaned training data."""
        if not _SUPPLEMENTAL_GEO_JSONL.exists():
            return set()

        out = set()
        try:
            with _SUPPLEMENTAL_GEO_JSONL.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    toks = normalize(row.get("clean_target", "")).split()
                    for i, tok in enumerate(toks):
                        if not tok.isalpha() or len(tok) < 5:
                            continue
                        next_tok = toks[i + 1] if i + 1 < len(toks) else ""
                        prev_tok = toks[i - 1] if i > 0 else ""
                        if (
                            tok.endswith(("halli", "nagar", "puram", "pura", "kere"))
                            or next_tok in _ROAD_SUFFIX_TOKENS
                            or next_tok in _LOCALITY_SUFFIX_TOKENS
                            or prev_tok in _ROAD_SUFFIX_TOKENS
                            or prev_tok in _LOCALITY_SUFFIX_TOKENS
                        ):
                            out.add(tok)
        except OSError:
            return set()
        return out

    @staticmethod
    def _char_ngrams(token: str, n: int = 3) -> set[str]:
        token = str(token)
        if len(token) < n:
            return {token} if token else set()
        return {token[i:i + n] for i in range(len(token) - n + 1)}

    @classmethod
    def _char_ngram_overlap(cls, a: str, b: str) -> float:
        ag = cls._char_ngrams(a)
        bg = cls._char_ngrams(b)
        if not ag:
            return 0.0
        return len(ag & bg) / max(len(ag), 1)

    @staticmethod
    def _consonant_signature(token: str) -> str:
        return "".join(ch for ch in str(token) if ch not in "aeiou")

    def _geo_context_kind(self, token: str, prev_tok: str, next_tok: str) -> str | None:
        if next_tok in _ROAD_SUFFIX_TOKENS or prev_tok in _ROAD_SUFFIX_TOKENS:
            return "road"
        if next_tok in _LOCALITY_SUFFIX_TOKENS or prev_tok in _LOCALITY_SUFFIX_TOKENS:
            return "locality"
        if token in self._hardcoded_areas or token.endswith(("halli", "nagar", "puram", "pura", "kere")):
            return "locality"
        if len(token) >= 7:
            return "geo"
        return None

    def _correct_geo_token(self, token: str, prev_tok: str, next_tok: str) -> str | None:
        """Correct out-of-vocabulary locality/road tokens using geo lexicons."""
        kind = self._geo_context_kind(token, prev_tok, next_tok)
        if kind is None or not self._geo_vocab:
            return None

        pool = self._geo_vocab
        prefix2 = token[:2]
        sig = self._consonant_signature(token)
        shortlist = [
            cand for cand in pool
            if abs(len(cand) - len(token)) <= 4
            and cand[0] == token[0]
            and (
                cand.startswith(prefix2)
                or self._consonant_signature(cand).startswith(sig[:4])
                or self._char_ngram_overlap(token, cand) >= 0.30
            )
        ]
        if len(shortlist) < 8:
            shortlist = [
                cand for cand in pool
                if abs(len(cand) - len(token)) <= 4
                and cand[0] == token[0]
            ] or pool

        best = process.extractOne(token, shortlist, scorer=fuzz.WRatio)
        if not best:
            return None

        candidate, score, _ = best
        ngram_overlap = self._char_ngram_overlap(token, candidate)
        same_sig = self._consonant_signature(token)[:4] == self._consonant_signature(candidate)[:4]

        min_score = 78
        if kind == "road":
            min_score = 75
        elif kind == "locality":
            min_score = 76

        if score < min_score:
            return None
        if ngram_overlap < 0.34 and not same_sig:
            return None
        return candidate

    def _correct_token(self, token: str, prev_tok: str = "", next_tok: str = "") -> str:
        """
        Correct a single token using a 5-stage probabilistic pipeline.
        """
        org_tokens = {
            "sap", "labs", "lab", "epip", "itpl", "tech", "park",
            "nexus", "nexuss", "mall", "phase", "stage", "floor",
        }

        building_markers = {
            "apartment", "apertment", "apartmnt", "apartmet", "apartmnet", "apt", "apts",
            "building", "tower", "residency", "residences", "residence", "society", "complex",
        }

        def _looks_like_building_marker(tok: str) -> bool:
            if not tok:
                return False
            if tok in building_markers:
                return True
            return fuzz.ratio(tok, "apartment") >= 70

        near_building_name = _looks_like_building_marker(prev_tok) or _looks_like_building_marker(next_tok)
        name_anchors = {
            "nilaya", "nileya", "nivas", "nivasa", "residency", "apartment",
            "villa", "house", "garden", "bhavan", "building", "tower", "layout",
        }
        near_name_anchor = next_tok in name_anchors

        # Prevent area-name overcorrection for common apartment names.
        if near_building_name and token == "gravit":
            return "gravity"

        # Skip non-alphabetic or single-char tokens
        if not token.isalpha():
            return token

        if token in self._trusted_short_aliases:
            return self._trusted_short_aliases[token]

        if len(token) < 2:
            return token

        # Stage 0: explicit short-form expansion (rd -> road, st -> street,
        # ngr -> nagar, ...). Done BEFORE the length / protected-word guards
        # so 2-char abbreviations actually get expanded.
        if token in self._misspelling_map and not token in PROTECTED_WORDS:
            return self._misspelling_map[token]

        # Skip protected short words (e.g., 'no', 'mg', 'sv')
        if token in PROTECTED_WORDS or len(token) <= 2:
            return token

        # Keep common organization/place tokens stable to avoid regressions
        # like 'labs' -> 'lbs'.
        if token in org_tokens:
            return token

        # Stage 1: Absolute exact match against hardcoded map
        mapped = self._misspelling_map.get(token)
        if mapped:
            return mapped

        # Preserve likely house/building proper names unless we have an
        # explicit dictionary mapping above.
        if near_name_anchor:
            return token

        # Stage 2: Already a highly frequent known word? Don't touch it.
        if token in self._vocab:
            return token

        # Stage 2.5: Area/road name fuzzy matching (relaxed threshold)
        # Catches severe misspellings like 'berrergata' → 'bannerghatta'
        geo_fixed = None
        if len(token) >= 5 and not near_building_name:
            geo_fixed = self._correct_geo_token(token, prev_tok, next_tok)
        if geo_fixed:
            return geo_fixed

        if len(token) >= 6 and self._area_vocab and not near_building_name:
            best_area = process.extractOne(token, self._area_vocab, scorer=fuzz.ratio)
            if best_area and best_area[1] >= 72:
                return best_area[0]

        # Stage 3: Norvig 1-Edit Check
        if len(token) >= 4:
            edits1 = self._prob_engine.edits1(token)
            known_edits1 = self._prob_engine.known(edits1)
            if known_edits1:
                # Pick highest frequency word 1 edit away
                best_e1 = max(known_edits1, key=self._prob_engine.probability)
                # GUARDRAIL: Only apply if it's structurally very similar
                if fuzz.ratio(token, best_e1) >= 80.0:
                    return best_e1

        # Stage 4: Norvig 2-Edit Check
        # Generate edits2 and check against known vocab
        if len(token) >= 6:  # Only do 2-edits for longer words to avoid false pos
            edits2 = self._prob_engine.edits2(token)
            known_edits2 = self._prob_engine.known(edits2)
            if known_edits2:
                # Pick highest frequency word 2 edits away
                best_e2 = max(known_edits2, key=self._prob_engine.probability)
                # GUARDRAIL: Ensure the length hasn't deviated radically & similarity is high
                if abs(len(best_e2) - len(token)) <= MAX_LEN_DIFF_VOCAB and fuzz.ratio(token, best_e2) >= 75.0:
                    return best_e2

        # Stage 5: Phonetic matching (handles extreme pronunciation swaps)
        if len(token) >= 5:
            phonetic_matches = self._phonetic_engine.get_matches(token)
            if phonetic_matches:
                # Pick the most frequent phonetic match
                best_p = max(phonetic_matches, key=self._prob_engine.probability)
                if fuzz.ratio(token, best_p) >= 55.0:
                    return best_p

        # Uncorrectable — return original token
        return token
