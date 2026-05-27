"""
fuzzy_engine.matcher
====================
Enterprise address matching engine with geo-boosted scoring.

Provides the AddressMatcher class which:
  1. Loads and indexes addresses from a CSV file
  2. Auto-detects area names from address frequency
  3. Checks if an address already exists in the database (>=threshold)
  4. Finds best matching real addresses using multi-strategy fuzzy scoring
     with geographic component boosting
"""

import csv
from collections import defaultdict

from rapidfuzz import fuzz, process

from fuzzy_engine.config import (
    DB_MATCH_THRESHOLD,
    TOP_N_SUGGESTIONS,
    W_TOKEN_SORT_RATIO, W_TOKEN_SET_RATIO, W_PARTIAL_RATIO, W_RATIO,
    CANDIDATES_PER_STRATEGY, CANDIDATES_PER_STRATEGY_RAW,
    BOOST_CITY_MATCH, BOOST_AREA_MATCH, BOOST_STREET_MATCH,
    BOOST_NUMBER_MATCH, BOOST_TOKEN_OVERLAP,
    BOOST_ROAD_NAME_MATCH, BOOST_PINCODE_MATCH,
    PENALTY_STREET_MISMATCH, PENALTY_AREA_MISMATCH,
    MAX_SCORE,
    MIN_AREA_FREQUENCY, AREA_STOP_WORDS,
)
from fuzzy_engine.dictionaries import KNOWN_CITIES, HARDCODED_AREAS
from fuzzy_engine.lexicons import AddressLexicons
from fuzzy_engine.normalizer import normalize, extract_geo_tokens
from fuzzy_engine.parser import parse_address


# Penalties to reduce false positives when core structured tokens disagree.
PENALTY_CITY_MISMATCH = 0.25
PENALTY_NUMBER_MISMATCH = 0.20
PENALTY_PINCODE_MISMATCH = 0.28
PENALTY_LOCALITY_MISMATCH = 0.22

# Street-type keywords that signal a preceding token is a road/area name.
_ROAD_SUFFIXES = {"road", "rd", "marg", "highway", "lane", "drive", "avenue", "ave"}
_AREA_SUFFIXES = {"nagar", "layout", "colony", "puram", "pura", "halli",
                  "enclave", "extension", "vihar", "garden", "park"}


def _extract_primary_road_name(geo_tokens: dict) -> str | None:
    """Extract the dominant road/area name from parsed geo tokens.

    Looks for long alpha tokens that appear alongside street-type keywords.
    e.g. in 'bannerghatta road' → returns 'bannerghatta'.
    """
    area_toks = geo_tokens.get("area_tokens", [])
    other_toks = geo_tokens.get("other_tokens", [])
    street_toks = set(geo_tokens.get("street_tokens", []))

    # Prefer area tokens that are long, specific names.
    candidates = [t for t in area_toks if len(t) >= 5 and t.isalpha()]
    # Also consider "other" tokens adjacent to road keywords.
    candidates += [t for t in other_toks if len(t) >= 5 and t.isalpha()]

    if not candidates:
        return None

    # If a street keyword (road/lane/etc) is present, the road name is likely
    # the longest specific token — e.g. 'bannerghatta' in 'bannerghatta road'.
    has_road_keyword = bool(street_toks & _ROAD_SUFFIXES)
    if has_road_keyword and candidates:
        return max(candidates, key=len)

    # Fallback: return the longest area-like candidate.
    return max(candidates, key=len) if candidates else None


class AddressMatcher:
    """
    Multi-strategy fuzzy matcher with geographic boosting.

    Usage:
        matcher = AddressMatcher("data/realistic_addresses.csv")
        exists_result = matcher.check_exists("no 655 richmond road indiranagar chennai 600338")
        match_result  = matcher.find_best_matches("prestige apartment banashankari bangalore")
    """

    def __init__(self, csv_path: str = None, address_list: list = None):
        """
        Initialize the matcher by loading addresses and building indexes.

        Args:
            csv_path:     Path to the CSV file with a 'raw_address' column.
            address_list: Pre-built list of address strings (e.g. from MySQL).
                          If provided, csv_path is ignored.
        """
        if address_list is not None:
            self._addresses = address_list
        elif csv_path is not None:
            self._addresses = self._load_csv(csv_path)
        else:
            raise ValueError("Either csv_path or address_list must be provided.")

        self._normalized_map = self._build_normalized_map()
        self._norm_keys = list(self._normalized_map.keys())
        self._lexicons = AddressLexicons.build(self._addresses)
        self._known_areas = self._detect_areas()
        self._known_areas.update(self._lexicons.locality_names)

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def addresses(self) -> list:
        """Return the raw address list."""
        return self._addresses

    @property
    def address_count(self) -> int:
        """Return total number of addresses loaded."""
        return len(self._addresses)

    @property
    def unique_count(self) -> int:
        """Return number of unique normalized addresses."""
        return len(self._normalized_map)

    @property
    def known_areas(self) -> set:
        """Return the detected + hardcoded area names."""
        return self._known_areas.copy()

    @property
    def area_count(self) -> int:
        """Return count of known areas."""
        return len(self._known_areas)

    def check_exists(self, normalized_query: str, candidate_addresses: list = None) -> dict:
        """
        Check if the address already exists in the database.

        Args:
            normalized_query: A normalized (lowercase, cleaned) address string.
            candidate_addresses: Optional narrowed pool to match against.

        Returns:
            dict with:
                found       : bool
                address     : str or None   — the matched original address
                score       : float         — the match percentage
        """
        if candidate_addresses:
            local_map = {}
            for addr in candidate_addresses:
                n = normalize(addr)
                if n and n not in local_map:
                    local_map[n] = addr
            norm_keys = list(local_map.keys()) if local_map else self._norm_keys
            norm_map = local_map if local_map else self._normalized_map
        else:
            norm_keys = self._norm_keys
            norm_map = self._normalized_map

        best_sort = process.extractOne(normalized_query, norm_keys, scorer=fuzz.token_sort_ratio)
        best_set = process.extractOne(normalized_query, norm_keys, scorer=fuzz.token_set_ratio)

        best = None
        if best_sort and best_set:
            best = best_sort if best_sort[1] >= best_set[1] else best_set
        else:
            best = best_sort or best_set

        if best and best[1] >= DB_MATCH_THRESHOLD:
            if not self._structured_match_allowed(normalized_query, best[0]):
                return {"found": False, "address": None, "score": 0.0}
            return {
                "found": True,
                "address": norm_map[best[0]],
                "score": best[1],
            }
        return {"found": False, "address": None, "score": 0.0}

    def find_best_matches(self, normalized_query: str,
                          raw_query: str = None,
                          top_n: int = TOP_N_SUGGESTIONS,
                          candidate_addresses: list = None) -> list:
        """
        Find the best matching real addresses using multi-strategy
        fuzzy scoring with geo-boosted ranking.

        Args:
            normalized_query: The spell-corrected, normalized query.
            raw_query:        Optional raw (uncorrected) normalized query
                              for additional candidate retrieval.
            top_n:            Number of results to return.
            candidate_addresses: Optional blocked subset of addresses.

        Returns:
            list of (original_address, score) tuples, sorted by score desc.
        """
        q = normalized_query
        if candidate_addresses:
            local_map = {}
            for addr in candidate_addresses:
                n = normalize(addr)
                if n and n not in local_map:
                    local_map[n] = addr
            norm_keys = list(local_map.keys()) if local_map else self._norm_keys
            norm_map = local_map if local_map else self._normalized_map
        else:
            norm_keys = self._norm_keys
            norm_map = self._normalized_map

        # ── Gather candidates from multiple strategies ────────────────
        candidate_set = set()
        scorers = [
            fuzz.token_sort_ratio, fuzz.token_set_ratio,
            fuzz.partial_ratio, fuzz.ratio,
        ]
        for scorer in scorers:
            results = process.extract(
                q, norm_keys, scorer=scorer,
                limit=CANDIDATES_PER_STRATEGY
            )
            for match_str, score, _ in results:
                candidate_set.add(match_str)

        # Also search with raw (uncorrected) input for broader coverage
        if raw_query and raw_query != q:
            for scorer in [fuzz.token_sort_ratio, fuzz.token_set_ratio]:
                results = process.extract(
                    raw_query, norm_keys, scorer=scorer,
                    limit=CANDIDATES_PER_STRATEGY_RAW
                )
                for match_str, score, _ in results:
                    candidate_set.add(match_str)

        # ── Score each candidate with geo-boosted matching ────────────
        query_geo = extract_geo_tokens(q, KNOWN_CITIES, self._known_areas)
        query_city = query_geo["city"]
        query_areas = set(query_geo["area_tokens"])
        query_numbers = set(query_geo["number_tokens"])
        query_streets = set(query_geo["street_tokens"])
        query_parsed = parse_address(q)
        query_real_numbers = {n for n in query_parsed.numbers if len(n) != 6}
        # Extract road/area name tokens for targeted boosting.
        # These are the "other" tokens that look like area names (long, alpha).
        query_road_names = {
            t for t in query_geo["area_tokens"] + query_geo["other_tokens"]
            if len(t) >= 5 and t.isalpha()
        }

        scored = []
        for cand in candidate_set:
            cand_parsed = parse_address(cand)
            cand_real_numbers = {n for n in cand_parsed.numbers if len(n) != 6}
            hard_mismatch = False

            # Base fuzzy scores (4 strategies, weighted)
            s_tsr = fuzz.token_sort_ratio(q, cand) / 100.0
            s_tse = fuzz.token_set_ratio(q, cand)  / 100.0
            s_pr  = fuzz.partial_ratio(q, cand)    / 100.0
            s_r   = fuzz.ratio(q, cand)            / 100.0

            base = (s_tsr * W_TOKEN_SORT_RATIO
                    + s_tse * W_TOKEN_SET_RATIO
                    + s_pr  * W_PARTIAL_RATIO
                    + s_r   * W_RATIO)

            # ── Geographic boosting ───────────────────────────────────
            cand_geo = extract_geo_tokens(cand, KNOWN_CITIES, self._known_areas)
            boost = 0.0

            # City match
            if query_city and cand_geo["city"] == query_city:
                boost += BOOST_CITY_MATCH

            # Area match (per matching area)
            cand_areas = set(cand_geo["area_tokens"])
            common_areas = query_areas & cand_areas
            if common_areas:
                # Cap area boost so generic locality overlap cannot dominate score.
                boost += BOOST_AREA_MATCH * min(len(common_areas), 2)

            # Street type match
            cand_streets = set(cand_geo["street_tokens"])
            if query_streets and query_streets & cand_streets:
                boost += BOOST_STREET_MATCH

            # Number match
            cand_numbers = set(cand_geo["number_tokens"])
            if query_numbers and query_numbers & cand_numbers:
                boost += BOOST_NUMBER_MATCH

            # Penalize key mismatches for better precision on near-duplicate addresses.
            penalty = 0.0
            if query_city and cand_geo["city"] and cand_geo["city"] != query_city:
                penalty += PENALTY_CITY_MISMATCH

            if query_numbers and cand_numbers and not (query_numbers & cand_numbers):
                penalty += PENALTY_NUMBER_MISMATCH

            q_pins = {n for n in query_numbers if len(n) == 6}
            c_pins = {n for n in cand_numbers if len(n) == 6}
            if q_pins and c_pins and not (q_pins & c_pins):
                penalty += PENALTY_PINCODE_MISMATCH
                hard_mismatch = True

            if query_real_numbers and cand_real_numbers and not (query_real_numbers & cand_real_numbers):
                penalty += PENALTY_NUMBER_MISMATCH

            # Token overlap
            q_tok = set(q.split())
            c_tok = set(cand.split())
            overlap = len(q_tok & c_tok) / max(len(q_tok), 1)
            boost += overlap * BOOST_TOKEN_OVERLAP

            # Road/area name match boost.
            cand_road_names = {
                t for t in cand_geo["area_tokens"] + cand_geo["other_tokens"]
                if len(t) >= 5 and t.isalpha()
            }
            if query_road_names and query_road_names & cand_road_names:
                boost += BOOST_ROAD_NAME_MATCH

            # Street/area name MISMATCH penalty.
            # When query clearly says "bannerghatta road" but candidate says
            # "bellary road", apply a heavy penalty.
            q_road = _extract_primary_road_name(query_geo)
            c_road = _extract_primary_road_name(cand_geo)
            q_road = query_parsed.road_anchor or q_road
            c_road = cand_parsed.road_anchor or c_road
            if q_road and c_road and q_road != c_road:
                # Both have a specific road name and they differ.
                penalty += PENALTY_STREET_MISMATCH
                hard_mismatch = True
            elif q_road and cand_road_names and q_road not in cand_road_names:
                # Query has a specific area but candidate doesn't contain it.
                penalty += PENALTY_AREA_MISMATCH

            q_localities = query_parsed.locality_anchors
            c_localities = cand_parsed.locality_anchors
            if q_localities and c_localities and not (q_localities & c_localities):
                penalty += PENALTY_LOCALITY_MISMATCH

            # Pincode match boost.
            q_pins = {n for n in query_numbers if len(n) == 6}
            c_pins = {n for n in cand_numbers if len(n) == 6}
            if q_pins and c_pins and (q_pins & c_pins):
                boost += BOOST_PINCODE_MATCH

            final_score = min(max((base + boost - penalty) * 100, 0.0), MAX_SCORE)
            if hard_mismatch:
                final_score = min(final_score, 69.0)
            scored.append((cand, final_score))

        scored.sort(key=lambda x: -x[1])

        # Convert normalized keys back to original addresses
        results = [(norm_map[norm_addr], round(score, 1)) for norm_addr, score in scored[:top_n]]
        return results

    def _structured_match_allowed(self, query: str, candidate: str) -> bool:
        """Guard exact/exists checks from high fuzzy scores with bad fields."""
        q = parse_address(query)
        c = parse_address(candidate)

        if q.pincode and c.pincode and q.pincode != c.pincode:
            return False
        if q.city and c.city and q.city != c.city:
            return False
        if q.road_anchor and c.road_anchor and q.road_anchor != c.road_anchor:
            return False

        q_nums = {n for n in q.numbers if len(n) != 6}
        c_nums = {n for n in c.numbers if len(n) != 6}
        if q_nums and c_nums and not (q_nums & c_nums):
            return False

        if q.locality_anchors and c.locality_anchors and not (q.locality_anchors & c.locality_anchors):
            return False
        return True

    # ── Private Methods ──────────────────────────────────────────────────

    def _load_csv(self, csv_path: str) -> list:
        """Load addresses from CSV file."""
        addresses = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw = row.get("raw_address", "").strip()
                if raw:
                    addresses.append(raw)
        return addresses

    def _build_normalized_map(self) -> dict:
        """Build normalized_text -> original_text lookup."""
        mapping = {}
        for addr in self._addresses:
            n = normalize(addr)
            if n and n not in mapping:
                mapping[n] = addr
        return mapping

    def _detect_areas(self) -> set:
        """
        Auto-detect area names by analyzing word frequency in addresses.
        Words appearing >= MIN_AREA_FREQUENCY times (that aren't cities
        or stop words) are considered area names.
        """
        freq = defaultdict(int)
        for addr in self._addresses:
            for tok in normalize(addr).split():
                if (tok.isalpha() and len(tok) > 3
                        and tok not in AREA_STOP_WORDS
                        and tok not in KNOWN_CITIES):
                    freq[tok] += 1

        areas = {word for word, count in freq.items()
                 if count >= MIN_AREA_FREQUENCY}

        # Add manually curated areas
        areas.update(HARDCODED_AREAS)
        return areas
