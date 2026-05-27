"""
5_full_pipeline_sql.py
High-accuracy address correction pipeline with SQL structured output.

Compared to the plain SQL retrieval variant, this includes the LightGBM
re-ranker stage used in 5_full_pipeline.py for better Hit@1 accuracy.

Run: python 5_full_pipeline_sql.py
"""

import os
import re
import pickle
from collections import defaultdict

import faiss
import numpy as np
import pandas as pd
import torch
from rapidfuzz import fuzz
from rapidfuzz import process
from sqlalchemy import text
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer

from db import get_engine, fetch_structured_rows
from fuzzy_engine.dictionaries import KNOWN_CITIES as DICTIONARY_KNOWN_CITIES
from fuzzy_engine.spell_checker import SpellChecker
from fuzzy_engine.learned_misspellings import load_learned_misspellings

# -- config ------------------------------------------------------------------
T5_MODEL_PATH = "models/t5_address"
BM25_PATH = "models/bm25.pkl"
FAISS_PATH = "models/faiss.index"
EMBEDDINGS_PATH = "models/embeddings.npy"
ADDRESSES_PATH = "models/addresses.npy"
ADDR_IDS_PATH = "models/address_ids.npy"
RERANKER_PATH = "models/reranker.pkl"
EMBED_MODEL = "multi-qa-mpnet-base-dot-v1"

RETRIEVAL_TOP_K = 80
FINAL_TOP_N = 5
T5_BEAMS = 4

W_BM25 = 0.30
W_FAISS = 0.50
W_FUZZY = 0.20

FEATURE_NAMES = [
    "bm25_score",
    "faiss_score",
    "fuzzy_tsr",
    "fuzzy_pr",
    "edit_sim",
    "token_overlap",
    "len_diff",
    "num_match",
]

KNOWN_CITIES = {
    "mumbai",
    "bangalore",
    "delhi",
    "hyderabad",
    "chennai",
    "kolkata",
    "pune",
    "ahmedabad",
    "jaipur",
    "noida",
    "gurgaon",
    "surat",
    "lucknow",
    "nagpur",
    "indore",
    "bhopal",
    "patna",
    "vadodara",
    "ludhiana",
    "agra",
    "nashik",
    "faridabad",
    "meerut",
    "rajkot",
    "chandigarh",
    "madurai",
    "mysuru",
    "solapur",
    "margao",
    "blr",
    "mum",
    "del",
    "hyd",
    "chn",
    "kol",
    "noi",
}
KNOWN_CITIES.update(DICTIONARY_KNOWN_CITIES)

STREET_KEYWORDS = {
    "road",
    "rd",
    "street",
    "st",
    "avenue",
    "ave",
    "nagar",
    "colony",
    "layout",
    "sector",
    "marg",
    "lane",
    "ln",
    "chowk",
    "cross",
    "main",
    "extension",
    "block",
    "phase",
    "park",
    "circle",
    "drive",
    "plaza",
    "enclave",
    "vihar",
    "puram",
    "ganj",
    "bazar",
    "market",
}

COMMON_ADDR_WORDS = {
    "street",
    "road",
    "avenue",
    "nagar",
    "colony",
    "layout",
    "sector",
    "floor",
    "flat",
    "number",
    "apartment",
    "block",
    "phase",
    "near",
    "opposite",
    "behind",
    "main",
    "cross",
    "circle",
    "building",
    "society",
    "tower",
    "wing",
    "no",
    "correct",
    "address",
}

EXACT_MATCH_CONFIDENCE = 1.0
NO_MATCH_CONFIDENCE_THRESHOLD = 0.78
FUZZY_FIRST_CONFIDENCE_THRESHOLD = 0.88
FUZZY_EXISTS_THRESHOLD = 90.0
VERIFIED_TOKEN_SORT_THRESHOLD = 93.0
VERIFIED_TOKEN_SET_THRESHOLD = 92.0
VERIFIED_MIN_TOKEN_OVERLAP = 0.65
VERIFIED_MIN_INFORMATIVE_OVERLAP = 0.45
MIN_BLOCK_RESULTS = 10

# SQL blocking config (query-time DB candidate narrowing)
SQL_BLOCK_LIMIT = 2000
SQL_BLOCK_MIN_TOKEN_LEN = 4
SQL_BLOCK_STOP_WORDS = {
    "address", "house", "flat", "building", "near", "opp", "opposite",
    "road", "street", "avenue", "lane", "main", "cross", "sector",
    "nagar", "layout", "block", "phase", "india", "state", "city",
}

COMMON_MISSPELLINGS = {
    "plat": "plot",
    "apertment": "apartment",
    "apaertment": "apartment",
    "appartment": "apartment",
    "aprtment": "apartment",
    "pot": "plot",
    "rad": "road",
    "vijapar": "vijapur",
    "maharashra": "maharashtra",
    "mumbay": "mumbai",
    "bombay": "mumbai",
    "bangalor": "bangalore",
    "bangalroe": "bangalore",
    "bengaluru": "bangalore",
    "bengalor": "bangalore",
    "bnagalore": "bangalore",
    "dlehi": "delhi",
    "stret": "street",
    "strret": "street",
    "rode": "road",
    "raod": "road",
    "rood": "road",
    "roaad": "road",
    "pradsh": "pradesh",
    "laoyut": "layout",
    "nagr": "nagar",
    "cty": "city",
    "citi": "city",
    "flour": "floor",
    "hous": "house",
    "rchmond": "richmond",
    "mal": "mall",
    "indiranagr": "indiranagar",
    "berrergata": "bannerghatta",
    "bennergatha": "bannerghatta",
    "bennergata": "bannerghatta",
    "bannergatta": "bannerghatta",
    "banergatta": "bannerghatta",
    "banerghata": "bannerghatta",
    "bannerghata": "bannerghatta",
    "banergata": "bannerghatta",
    "sustems": "systems",
    "sidadiah": "siddaiah",
    "prastige": "prestige",
    "prestiege": "prestige",
    "benagalore": "bangalore",
    "benagalor": "bangalore",
    "benaglore": "bangalore",
    "banglaor": "bangalore",
    "maiin": "main",
    "maaiin": "main",
    "mailn": "main",
    "stag": "stage",
    "satage": "stage",
    "stge": "stage",
    "satge": "stage",
    "koraamangaala": "koramangala",
    "koraamangala": "koramangala",
    "koramangaala": "koramangala",
    "withefield": "whitefield",
    "whithefield": "whitefield",
    "sarejahpur": "sarjapur",
    "sarrjapura": "sarjapur",
    "sajapur": "sarjapur",
    "feeniks": "phoenix",
    "fenix": "phoenix",
    "udayaangar": "udayanagar",
    "vijaynagar": "vijayanagar",
    "viajya": "vijaya",
    "ajnaneya": "anjaneya",
    "tempe": "temple",
    "kbuer": "kuber",
    "buuilding": "building",
    "jabber": "jabbar",
    "swwamy": "swamy",
    "vivekanada": "vivekananda",
    "karataka": "karnataka",
    "kanataka": "karnataka",
    "kanrataka": "karnataka",
    "karnatkaa": "karnataka",
    "karnatakka": "karnataka",
    "karntaaka": "karnataka",
    "karnaataka": "karnataka",
    "karnaatka": "karnataka",
    "idia": "india",
    "indiia": "india",
    "indai": "india",
    "nidia": "india",
    "inndia": "india",
}

_LEARNED_MISSPELLINGS = load_learned_misspellings()
for _src, _dst in _LEARNED_MISSPELLINGS.items():
    COMMON_MISSPELLINGS.setdefault(_src, _dst)

# Map city -> state for geo-filtering.
CITY_STATE_MAP = {
    "bangalore": "karnataka", "mumbai": "maharashtra", "delhi": "delhi",
    "hyderabad": "telangana", "chennai": "tamil nadu", "kolkata": "west bengal",
    "pune": "maharashtra", "ahmedabad": "gujarat", "jaipur": "rajasthan",
    "noida": "uttar pradesh", "gurgaon": "haryana", "surat": "gujarat",
    "lucknow": "uttar pradesh", "nagpur": "maharashtra", "indore": "madhya pradesh",
    "bhopal": "madhya pradesh", "patna": "bihar", "vadodara": "gujarat",
    "chandigarh": "chandigarh", "kochi": "kerala", "mysuru": "karnataka",
}

KNOWN_STATES = {
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh",
    "goa", "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka",
    "kerala", "madhya pradesh", "maharashtra", "manipur", "meghalaya", "mizoram",
    "nagaland", "odisha", "punjab", "rajasthan", "sikkim", "tamil nadu",
    "telangana", "tripura", "uttar pradesh", "uttarakhand", "west bengal", "delhi",
    "chandigarh",
}

_SPELL_VOCAB_CACHE = None
_SPELL_BUCKETS_CACHE = None


# -- helpers -----------------------------------------------------------------
def normalize(addr: str) -> str:
    addr = re.sub(r"[^\w\s]", " ", str(addr).lower())
    addr = re.sub(r"_+", " ", addr)
    addr = re.sub(r"(?<=\d)(?!(?:st|nd|rd|th)\b)(?=[a-z])|(?<=[a-z])(?=\d)", " ", addr)
    return re.sub(r"\s+", " ", addr).strip()


def detect_fields(raw: str) -> dict:
    text = normalize(raw)
    tokens = text.split()

    has_number = any(t.isdigit() for t in tokens)
    has_pincode = any(t.isdigit() and len(t) == 6 for t in tokens)
    has_partial_pincode = any(t.isdigit() and len(t) in (4, 5) for t in tokens)
    has_street = any(t in STREET_KEYWORDS for t in tokens)
    has_city = any(t in KNOWN_CITIES for t in tokens)

    warnings = []
    if not has_number:
        warnings.append("No house or flat number detected.")
    if not has_street:
        warnings.append("No street type detected (road, street, nagar, etc.).")
    if not has_city:
        warnings.append("No city name detected.")
    if len(tokens) < 3:
        warnings.append("Address too short. Please provide more details.")
    if has_partial_pincode and not has_pincode:
        warnings.append("Pincode appears incomplete (expected 6 digits in India).")

    return {
        "has_number": has_number,
        "has_street": has_street,
        "has_city": has_city,
        "has_pincode": has_pincode,
        "has_partial_pincode": has_partial_pincode,
        "token_count": len(tokens),
        "warnings": warnings,
        "is_complete": len(warnings) == 0,
    }


def _extract_proper_nouns(text: str) -> set:
    tokens = normalize(text).split()
    return {t for t in tokens if len(t) > 5 and t not in COMMON_ADDR_WORDS}


def _merge_with_original(original: str, t5_output: str) -> str:
    orig_proper = _extract_proper_nouns(original)
    t5_tokens = set(normalize(t5_output).split())
    dropped = orig_proper - t5_tokens
    if not dropped:
        return t5_output

    restored = t5_output.strip()
    for word in sorted(dropped):
        if word not in normalize(restored):
            restored = restored + " " + word
    return restored.strip()


def _is_bad_t5_output(text: str) -> bool:
    t = normalize(text)
    if not t:
        return True

    # Guard against common hallucination patterns (links, non-address text).
    raw = text.lower()
    if "http://" in raw or "https://" in raw or "www." in raw:
        return True

    tokens = t.split()
    if len(tokens) < 2:
        return True

    has_digit = any(tok.isdigit() for tok in tokens)
    has_city = any(tok in KNOWN_CITIES for tok in tokens)
    has_street = any(tok in STREET_KEYWORDS for tok in tokens)

    # Require at least one address-like signal.
    return not (has_digit or has_city or has_street)


def _format_generated_address(text: str) -> str:
    """Normalize and title-case generated addresses with a readable geo tail."""
    tokens = normalize(text).split()
    if not tokens:
        return ""

    upper_tokens = {
        "ii", "iii", "iv", "vi", "vii", "viii", "ix", "x",
        "ncr", "uk", "usa", "uae",
        "jp", "mg", "hsr", "blr", "btm",
    }

    out = []
    for tok in tokens:
        if tok.isdigit():
            out.append(tok)
        elif tok in upper_tokens:
            out.append(tok.upper())
        else:
            out.append(tok.capitalize())

    def _title_tokens(seq):
        out_seq = []
        for tok in seq:
            if tok.isdigit():
                out_seq.append(tok)
            elif tok in upper_tokens:
                out_seq.append(tok.upper())
            else:
                out_seq.append(tok.capitalize())
        return out_seq

    def _format_house_prefix(seq):
        if not seq:
            return None, seq

        first = seq[0].lower()
        if first == "no" and len(seq) >= 3 and seq[1].isdigit() and seq[2].isdigit():
            return f"No. {seq[1]}/{seq[2]}", seq[3:]

        if first.startswith("no") and len(first) > 2 and first[2:].isdigit():
            base_number = first[2:]
            if len(seq) >= 2 and seq[1].isdigit():
                return f"No. {base_number}/{seq[1]}", seq[2:]
            return f"No. {base_number}", seq[1:]

        return None, seq

    def _chunk_body(seq):
        if not seq:
            return []
        breakers = {
            "road", "street", "avenue", "lane", "layout", "post",
            "colony", "garden", "phase", "block", "building", "tower",
            "complex", "villa", "cross",
        }
        chunks = []
        current = [seq[0]]
        for tok in seq[1:]:
            if tok.isdigit() or current[-1] in breakers:
                chunks.append(current)
                current = [tok]
            else:
                current.append(tok)
        if current:
            chunks.append(current)
        return chunks

    # Build a comma-separated display form when geo parts are present.
    country = None
    pincode = None
    state = None
    city = None

    geo_tokens = tokens[:]
    if geo_tokens and geo_tokens[-1].isdigit() and len(geo_tokens[-1]) == 6:
        pincode = geo_tokens.pop()

    if geo_tokens and geo_tokens[-1] in {"india"}:
        country = geo_tokens.pop()

    if len(geo_tokens) >= 2:
        two = f"{geo_tokens[-2]} {geo_tokens[-1]}"
        if two in KNOWN_STATES:
            state = two
            geo_tokens = geo_tokens[:-2]
    if state is None and geo_tokens and geo_tokens[-1] in KNOWN_STATES:
        state = geo_tokens.pop()

    if geo_tokens and geo_tokens[-1] in KNOWN_CITIES:
        city = geo_tokens.pop()

    house_prefix, geo_tokens = _format_house_prefix(geo_tokens)
    body_chunks = [" ".join(_title_tokens(chunk)) for chunk in _chunk_body(geo_tokens)]

    parts = []
    if house_prefix:
        parts.append(house_prefix)
    if body_chunks:
        parts.extend(body_chunks)
    if city:
        parts.append(city.capitalize())
    if state and pincode:
        parts.append(" ".join(w.capitalize() for w in state.split()) + f" {pincode}")
    elif state:
        parts.append(" ".join(w.capitalize() for w in state.split()))
    elif pincode:
        parts.append(pincode)
    if country:
        parts.append(country.capitalize())

    if len(parts) >= 2:
        return ", ".join(parts)
    if body_chunks:
        return ", ".join(body_chunks)
    if house_prefix:
        return house_prefix
    return " ".join(_title_tokens(tokens))


def _nearby_single_suggestion(query: str, addresses, addr_ids, rows_by_id, addr_to_idx):
    """Return one very-near DB suggestion (~99%) when available."""
    hit = process.extractOne(normalize(query), addresses, scorer=fuzz.token_set_ratio)
    if not hit:
        return None

    candidate, score, _ = hit
    if float(score) < 99.0:
        return None

    idx = addr_to_idx.get(candidate)
    db_id = int(addr_ids[idx]) if idx is not None and idx < len(addr_ids) else None
    record = rows_by_id.get(db_id, {}) if db_id is not None else {}
    return {
        "full_address": candidate,
        "score": round(float(score) / 100.0, 4),
        "db_id": db_id,
        "structured": record,
    }


def _ensure_spell_vocab(addresses):
    """Build lightweight correction vocabulary once from indexed addresses."""
    global _SPELL_VOCAB_CACHE, _SPELL_BUCKETS_CACHE
    if _SPELL_VOCAB_CACHE is not None and _SPELL_BUCKETS_CACHE is not None:
        return _SPELL_VOCAB_CACHE, _SPELL_BUCKETS_CACHE

    vocab = set(KNOWN_CITIES) | set(STREET_KEYWORDS) | set(COMMON_ADDR_WORDS)
    for addr in addresses:
        for tok in normalize(addr).split():
            if tok.isalpha() and len(tok) >= 3:
                vocab.add(tok)

    buckets = defaultdict(list)
    for tok in vocab:
        buckets[tok[0]].append(tok)

    _SPELL_VOCAB_CACHE = vocab
    _SPELL_BUCKETS_CACHE = buckets
    return _SPELL_VOCAB_CACHE, _SPELL_BUCKETS_CACHE


def _conservative_correct_from_raw_with_changes(raw_input: str, addresses):
    """Return conservative token corrections and a list of token-level edits."""
    vocab, buckets = _ensure_spell_vocab(addresses)
    tokens = normalize(raw_input).split()
    if not tokens:
        return "", []

    corrected = []
    changes = []
    for tok in tokens:
        if tok.isdigit() or len(tok) < 3:
            corrected.append(tok)
            continue

        mapped = COMMON_MISSPELLINGS.get(tok)
        if mapped:
            corrected.append(mapped)
            if mapped != tok:
                changes.append(f"'{tok}' -> '{mapped}'")
            continue

        if tok in vocab or not tok.isalpha():
            corrected.append(tok)
            continue

        pool = buckets.get(tok[0], [])
        if not pool:
            corrected.append(tok)
            continue

        best = process.extractOne(tok, pool, scorer=fuzz.ratio)
        if not best:
            corrected.append(tok)
            continue

        candidate, score, _ = best
        if score >= 90 and abs(len(candidate) - len(tok)) <= 3:
            corrected.append(candidate)
            if candidate != tok:
                changes.append(f"'{tok}' -> '{candidate}'")
        else:
            corrected.append(tok)

    return " ".join(corrected), changes


def _conservative_correct_from_raw(raw_input: str, addresses) -> str:
    """Backward-compatible wrapper that returns only corrected text."""
    corrected, _ = _conservative_correct_from_raw_with_changes(raw_input, addresses)
    return corrected


def _sql_block_terms(query: str):
    """Return selective query terms for SQL candidate blocking."""
    tokens = normalize(query).split()
    if not tokens:
        return []

    city_terms = [t for t in tokens if t in KNOWN_CITIES]
    number_terms = [t for t in tokens if t.isdigit()]
    text_terms = []
    for t in tokens:
        if t.isdigit() or len(t) < SQL_BLOCK_MIN_TOKEN_LEN or t in SQL_BLOCK_STOP_WORDS:
            continue
        text_terms.append(t)

    # Keep deterministic order and dedupe.
    terms = city_terms[:2] + number_terms[:2] + text_terms[:6]
    seen = set()
    out = []
    for t in terms:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


# Street-type keywords that signal a preceding token is a road/area name.
_ROAD_SUFFIXES = {"road", "rd", "marg", "highway", "lane", "drive", "avenue", "ave"}
_AREA_SUFFIXES = {"nagar", "layout", "colony", "puram", "pura", "halli",
                  "enclave", "extension", "vihar", "garden", "park"}


def _extract_street_anchor(query: str) -> str | None:
    """Extract the dominant street/area name from a query for SQL filtering.

    Finds tokens like 'bannerghatta' in 'bannerghatta road' by looking for
    long alpha tokens that appear before street-type suffixes.
    """
    tokens = normalize(query).split()
    if not tokens:
        return None

    all_suffixes = _ROAD_SUFFIXES | _AREA_SUFFIXES
    suffix_candidates = []
    generic_candidates = []

    for i, tok in enumerate(tokens):
        if tok in all_suffixes and i > 0:
            prev = tokens[i - 1]
            if prev.isalpha() and len(prev) >= 5 and prev not in KNOWN_CITIES:
                suffix_candidates.append(prev)

    # Also check for known area names that are long enough to be discriminative.
    for tok in tokens:
        if (tok.isalpha() and len(tok) >= 7
                and tok not in KNOWN_CITIES and tok not in STREET_KEYWORDS
                and tok not in SQL_BLOCK_STOP_WORDS and tok not in COMMON_ADDR_WORDS):
            if tok not in suffix_candidates and tok not in generic_candidates:
                generic_candidates.append(tok)

    if suffix_candidates:
        return max(suffix_candidates, key=len)
    return max(generic_candidates, key=len) if generic_candidates else None


def _detect_geo_anchors(query: str) -> dict:
    """Extract city/state/pincode/street anchors for geo-first SQL narrowing."""
    q = normalize(query)
    tokens = q.split()

    city = None
    pincode = None
    for tok in tokens:
        if tok in KNOWN_CITIES:
            city = tok
        if tok.isdigit() and len(tok) == 6:
            pincode = tok

    state = None
    for st in KNOWN_STATES:
        if st in q:
            state = st
            break

    if city and not state:
        state = CITY_STATE_MAP.get(city)

    street = _extract_street_anchor(q)

    return {"city": city, "state": state, "pincode": pincode, "street": street}


def _build_text_score_sql(text_terms, params):
    """Build LIKE score expressions for non-geo ranking within narrowed pool."""
    like_exprs = []
    score_exprs = []
    for i, term in enumerate(text_terms[:8]):
        key = f"t{i}"
        params[key] = f"%{term}%"
        expr = (
            f"(LOWER(COALESCE(normalized_full_address, '')) LIKE :{key} "
            f"OR LOWER(COALESCE(source_raw_address, '')) LIKE :{key})"
        )
        like_exprs.append(expr)
        score_exprs.append(f"CASE WHEN {expr} THEN 1 ELSE 0 END")

    if score_exprs:
        score_sql = " + ".join(score_exprs)
        order_by = f"({score_sql}) DESC, address_id ASC"
    else:
        score_sql = "1"
        order_by = "address_id ASC"
    return score_sql, order_by


def _execute_block_query(engine, addrid_to_idx, where_sql: str, params: dict, text_terms, limit: int):
    """Execute one blocking pass and return mapped indices + total matches."""
    count_sql = text(
        f"""
        SELECT COUNT(*) AS cnt
        FROM addresses
        WHERE {where_sql}
        """
    )

    local_params = dict(params)
    score_sql, order_by = _build_text_score_sql(text_terms, local_params)
    local_params["limit"] = int(limit)

    query_sql = text(
        f"""
        SELECT address_id, ({score_sql}) AS block_score
        FROM addresses
        WHERE {where_sql}
        ORDER BY {order_by}
        LIMIT :limit
        """
    )

    with engine.connect() as conn:
        total = int(conn.execute(count_sql, params).scalar_one())
        rows = conn.execute(query_sql, local_params).mappings().all()

    blocked = []
    for row in rows:
        idx = addrid_to_idx.get(int(row["address_id"]))
        if idx is not None:
            blocked.append(idx)
    return blocked, total


def _fetch_blocked_indices(
    engine,
    query: str,
    addrid_to_idx: dict,
    limit: int = SQL_BLOCK_LIMIT,
    allow_pincode: bool = True,
):
    """Fetch blocked candidate indices using strict geo-first narrowing with fallback.

    Blocking cascade (strict → loose):
      1. pincode + city + state + street
      2. city + state + street
      3. pincode + city + state
      4. city + state
      5. city + street
      6. pincode only
      7. city only
      8. legacy LIKE fallback
    """
    anchors = _detect_geo_anchors(query)
    detected_city = anchors["city"]
    detected_state = anchors["state"]
    detected_pincode = anchors["pincode"] if allow_pincode else None
    detected_street = anchors.get("street")
    terms = _sql_block_terms(query)

    text_terms = [
        t for t in terms
        if t not in KNOWN_CITIES and not (t.isdigit() and len(t) == 6)
    ]

    # Street LIKE clause for narrowing within geo-filtered results.
    street_like_clause = (
        "(LOWER(COALESCE(normalized_full_address, '')) LIKE :geo_street "
        "OR LOWER(COALESCE(source_raw_address, '')) LIKE :geo_street "
        "OR LOWER(COALESCE(street, '')) LIKE :geo_street "
        "OR LOWER(COALESCE(area, '')) LIKE :geo_street)"
    )

    passes = []

    # --- Passes WITH street anchor (highest precision) ---
    if detected_street:
        street_param = {"geo_street": f"%{detected_street}%"}

        if detected_pincode and detected_city and detected_state:
            passes.append((
                "pin_city_state_street",
                f"pincode = :geo_pin AND "
                f"(LOWER(COALESCE(town, '')) = :geo_city OR LOWER(COALESCE(district, '')) = :geo_city) AND "
                f"LOWER(COALESCE(state, '')) = :geo_state AND {street_like_clause}",
                {"geo_pin": detected_pincode, "geo_city": detected_city,
                 "geo_state": detected_state, **street_param},
            ))

        if detected_city and detected_state:
            passes.append((
                "city_state_street",
                f"(LOWER(COALESCE(town, '')) = :geo_city OR LOWER(COALESCE(district, '')) = :geo_city) AND "
                f"LOWER(COALESCE(state, '')) = :geo_state AND {street_like_clause}",
                {"geo_city": detected_city, "geo_state": detected_state, **street_param},
            ))

        if detected_city:
            passes.append((
                "city_street",
                f"(LOWER(COALESCE(town, '')) = :geo_city OR LOWER(COALESCE(district, '')) = :geo_city) AND "
                f"{street_like_clause}",
                {"geo_city": detected_city, **street_param},
            ))

    # --- Passes WITHOUT street anchor (broader fallback) ---
    if detected_pincode and detected_city and detected_state:
        passes.append((
            "pin_city_state",
            "pincode = :geo_pin AND "
            "(LOWER(COALESCE(town, '')) = :geo_city OR LOWER(COALESCE(district, '')) = :geo_city) AND "
            "LOWER(COALESCE(state, '')) = :geo_state",
            {"geo_pin": detected_pincode, "geo_city": detected_city, "geo_state": detected_state},
        ))
    if detected_city and detected_state:
        passes.append((
            "city_state",
            "(LOWER(COALESCE(town, '')) = :geo_city OR LOWER(COALESCE(district, '')) = :geo_city) AND "
            "LOWER(COALESCE(state, '')) = :geo_state",
            {"geo_city": detected_city, "geo_state": detected_state},
        ))
    if detected_pincode:
        passes.append(("pincode", "pincode = :geo_pin", {"geo_pin": detected_pincode}))
    if detected_city:
        passes.append((
            "city",
            "(LOWER(COALESCE(town, '')) = :geo_city OR LOWER(COALESCE(district, '')) = :geo_city)",
            {"geo_city": detected_city},
        ))

    for strategy, where_sql, params in passes:
        blocked, total = _execute_block_query(
            engine=engine,
            addrid_to_idx=addrid_to_idx,
            where_sql=where_sql,
            params=params,
            text_terms=text_terms,
            limit=limit,
        )
        if blocked:
            return blocked, total, strategy

    # -- Legacy fallback: plain LIKE blocking --
    if not terms:
        return None, 0, "none"

    min_hits = 2 if len(terms) >= 2 else 1

    params_legacy = {"limit": int(limit), "min_hits": int(min_hits)}
    like_exprs = []
    score_exprs = []
    for i, term in enumerate(terms):
        key = f"t{i}"
        params_legacy[key] = f"%{term}%"
        expr = (
            f"(LOWER(COALESCE(normalized_full_address, '')) LIKE :{key} "
            f"OR LOWER(COALESCE(source_raw_address, '')) LIKE :{key})"
        )
        like_exprs.append(expr)
        score_exprs.append(f"CASE WHEN {expr} THEN 1 ELSE 0 END")

    where_sql = " OR ".join(like_exprs)
    score_sql = " + ".join(score_exprs)
    query_sql = text(
        f"""
        SELECT address_id, ({score_sql}) AS block_score
        FROM addresses
                WHERE ({where_sql})
                    AND ({score_sql}) >= :min_hits
        ORDER BY block_score DESC, address_id ASC
        LIMIT :limit
        """
    )

    count_sql = text(
        f"""
        SELECT COUNT(*) AS cnt
        FROM addresses
        WHERE ({where_sql})
          AND ({score_sql}) >= :min_hits
        """
    )

    with engine.connect() as conn:
        total = int(conn.execute(count_sql, params_legacy).scalar_one())
        rows = conn.execute(query_sql, params_legacy).mappings().all()

    blocked = []
    for row in rows:
        idx = addrid_to_idx.get(int(row["address_id"]))
        if idx is not None:
            blocked.append(idx)
    return blocked or None, total, "legacy_like"


def _build_top_results(final, addresses, addr_ids, rows_by_id):
    """Convert ranked vector candidates to response payload rows."""
    top_results = []
    seen = set()
    for vec_idx, addr, score in final:
        addr_norm = normalize(addr)
        if addr_norm in seen:
            continue
        seen.add(addr_norm)
        db_id = int(addr_ids[vec_idx]) if vec_idx < len(addr_ids) else None
        record = rows_by_id.get(db_id, {}) if db_id is not None else {}
        top_results.append(
            {
                "full_address": addr,
                "score": round(float(score), 4),
                "db_id": db_id,
                "structured": record,
            }
        )
    return top_results


def _informative_tokens(text: str) -> set:
    """Return non-generic tokens that are more discriminative for matching."""
    toks = set(normalize(text).split())
    if not toks:
        return set()

    generic = set(COMMON_ADDR_WORDS) | set(STREET_KEYWORDS) | set(KNOWN_CITIES)
    generic.update({"india", "ind", "post", "po", "near", "opp", "opposite"})

    # Exclude state-name pieces so generic geo terms don't inflate overlap.
    for state in KNOWN_STATES:
        for part in state.split():
            generic.add(part)

    out = set()
    for tok in toks:
        if tok.isdigit():
            continue
        if len(tok) <= 2:
            continue
        if tok in generic:
            continue
        out.add(tok)
    return out


def _merge_candidates(primary, secondary, top_k: int):
    """Merge candidate lists by vec-index and keep the higher base score."""
    by_idx = {}
    for idx, addr, score in primary + secondary:
        old = by_idx.get(idx)
        if old is None or float(score) > float(old[2]):
            by_idx[idx] = (idx, addr, float(score))
    merged = sorted(by_idx.values(), key=lambda x: -float(x[2]))
    return merged[:top_k]


def _check_exists_in_pool(query: str, pool_indices, addresses, addr_ids, rows_by_id):
    """Find a strict verified DB match inside candidate pool with geo consistency."""
    if not pool_indices:
        return None

    norm_to_idx = {}
    pool_norm = []
    for idx in pool_indices:
        n = normalize(addresses[idx])
        if not n or n in norm_to_idx:
            continue
        norm_to_idx[n] = idx
        pool_norm.append(n)

    if not pool_norm:
        return None

    q_norm = normalize(query)
    def _combined_similarity(a: str, b: str, **_kwargs) -> float:
        return max(fuzz.token_sort_ratio(a, b), fuzz.token_set_ratio(a, b))

    best = process.extractOne(q_norm, pool_norm, scorer=_combined_similarity)
    if not best:
        return None

    best_norm, score, _ = best
    if float(score) < VERIFIED_TOKEN_SET_THRESHOLD:
        return None

    q_tokens = set(q_norm.split())
    cand_tokens = set(best_norm.split())
    token_overlap = len(q_tokens & cand_tokens) / max(len(q_tokens), 1)
    if len(q_tokens) >= 6 and token_overlap < VERIFIED_MIN_TOKEN_OVERLAP:
        return None

    # Guardrail: verified matches should preserve street-type cues.
    q_has_street = bool(q_tokens & STREET_KEYWORDS)
    c_has_street = bool(cand_tokens & STREET_KEYWORDS)
    if q_has_street and not c_has_street:
        return None

    # Stronger rule for road-style queries.
    q_has_road = bool(q_tokens & {"road", "rd"})
    c_has_road = bool(cand_tokens & {"road", "rd"})
    if q_has_road and not c_has_road:
        return None

    # Road/area name consistency: reject when primary road names explicitly differ.
    # e.g. query='bannerghatta road' vs candidate='bellary road' -> reject.
    q_street = _extract_street_anchor(q_norm)
    c_street = _extract_street_anchor(best_norm)
    if q_street and c_street and q_street != c_street:
        return None

    q_inf = _informative_tokens(q_norm)
    c_inf = _informative_tokens(best_norm)
    if q_inf:
        inf_overlap = len(q_inf & c_inf) / max(len(q_inf), 1)
        if len(q_inf) >= 3 and inf_overlap < VERIFIED_MIN_INFORMATIVE_OVERLAP:
            return None
    else:
        inf_overlap = 1.0

    idx = norm_to_idx[best_norm]
    db_id = int(addr_ids[idx]) if idx < len(addr_ids) else None
    record = rows_by_id.get(db_id, {}) if db_id is not None else {}

    q_anchors = _detect_geo_anchors(q_norm)
    rec_state = normalize(record.get("state", "")) if record else ""
    rec_town = normalize(record.get("town", "")) if record else ""
    rec_district = normalize(record.get("district", "")) if record else ""
    rec_pin = str(record.get("pincode", "")).strip() if record else ""

    if q_anchors["pincode"] and rec_pin and q_anchors["pincode"] != rec_pin:
        return None
    if q_anchors["state"] and rec_state and q_anchors["state"] != rec_state:
        return None
    if q_anchors["city"] and (rec_town or rec_district):
        if q_anchors["city"] not in {rec_town, rec_district}:
            return None

    return {
        "idx": idx,
        "address": addresses[idx],
        "verification_score": round(float(score), 2),
        "token_overlap": round(float(token_overlap), 3),
        "informative_overlap": round(float(inf_overlap), 3),
        "db_id": db_id,
        "record": record,
    }


def _validate_model_artifacts(addresses, addr_ids, bm25, faiss_idx, embeddings) -> None:
    """Fail fast when retrieval artifacts were built from different corpora."""
    n_addresses = len(addresses)
    checks = {
        "address_ids": len(addr_ids),
        "bm25_docs": len(getattr(bm25, "doc_freqs", [])),
    }
    if faiss_idx is not None:
        checks["faiss_vectors"] = int(faiss_idx.ntotal)
    if embeddings is not None:
        checks["embeddings"] = int(embeddings.shape[0])

    mismatched = {
        name: count
        for name, count in checks.items()
        if int(count) != n_addresses
    }
    if mismatched:
        details = ", ".join(f"{name}={count}" for name, count in mismatched.items())
        raise RuntimeError(
            "Model artifact mismatch detected. "
            f"addresses={n_addresses}, {details}. "
            "Run 'python 3_build_indexes.py' to rebuild all retrieval artifacts together."
        )


# -- loading -----------------------------------------------------------------
def load_models():
    print("Loading models...")
    t5_tok = T5Tokenizer.from_pretrained(T5_MODEL_PATH)
    t5 = T5ForConditionalGeneration.from_pretrained(T5_MODEL_PATH)
    t5.eval()

    embedder = None
    faiss_idx = None
    embeddings = None
    try:
        embedder = SentenceTransformer(EMBED_MODEL, local_files_only=True)
        faiss_idx = faiss.read_index(FAISS_PATH)
        embeddings = np.load(EMBEDDINGS_PATH)
    except Exception as exc:
        print(
            "Warning: semantic embedder unavailable in offline mode; "
            f"using BM25 + RapidFuzz fallback. Details: {exc}"
        )
    addresses = np.load(ADDRESSES_PATH, allow_pickle=True).tolist()
    addr_ids = np.load(ADDR_IDS_PATH, allow_pickle=True).tolist()

    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)

    _validate_model_artifacts(addresses, addr_ids, bm25, faiss_idx, embeddings)

    ranker = None
    if os.path.exists(RERANKER_PATH):
        with open(RERANKER_PATH, "rb") as f:
            ranker = pickle.load(f)

    engine = get_engine()
    structured_rows = fetch_structured_rows(engine)
    rows_by_id = {int(row["address_id"]): row for row in structured_rows}
    spell_checker = SpellChecker(addresses)
    addr_to_idx = {addr: i for i, addr in enumerate(addresses)}
    addrid_to_idx = {
        int(addr_id): i
        for i, addr_id in enumerate(addr_ids)
    }
    exact_lookup = {}
    for i, addr in enumerate(addresses):
        n = normalize(addr)
        # Keep first occurrence as canonical for deterministic pass-through.
        if n and n not in exact_lookup:
            exact_lookup[n] = i

    print(f"All models loaded. Search corpus: {len(addresses):,} addresses")
    print(f"Structured rows loaded: {len(rows_by_id):,}")
    if embedder is None:
        print("Semantic FAISS retrieval disabled; lexical fallback is active.")
    if ranker is None:
        print("Warning: reranker not found, falling back to retrieval-only scoring.")
    else:
        print("Reranker loaded: high-accuracy mode enabled.")
    print()

    return (
        t5_tok,
        t5,
        embedder,
        faiss_idx,
        embeddings,
        addresses,
        addr_ids,
        bm25,
        ranker,
        addr_to_idx,
        addrid_to_idx,
        rows_by_id,
        spell_checker,
        exact_lookup,
        engine,
    )


# -- step 1: T5 --------------------------------------------------------------
def t5_correct(raw: str, t5_tok, t5, max_len=96, num_beams=4) -> str:
    prompt = f"correct address: {normalize(raw)}"
    inp = t5_tok(prompt, return_tensors="pt", max_length=max_len, truncation=True)

    with torch.no_grad():
        out = t5.generate(
            **inp,
            max_length=max_len,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )

    decoded = t5_tok.decode(out[0], skip_special_tokens=True).strip()
    merged = _merge_with_original(raw, decoded)

    if _is_bad_t5_output(merged):
        return normalize(raw)
    return merged


# -- step 2: candidate retrieval ---------------------------------------------
def retrieve_candidates(query: str, embedder, faiss_idx, addresses, bm25, top_k=50,
                        blocked_indices=None, use_fuzzy=True):
    scores = defaultdict(float)
    q = normalize(query)

    candidate_indices = None
    if blocked_indices:
        candidate_indices = list(dict.fromkeys(blocked_indices))

    bm25_scores = bm25.get_scores(q.split())
    if candidate_indices:
        blocked_vals = [float(bm25_scores[i]) for i in candidate_indices]
        bm25_max = max(blocked_vals) + 1e-9
        ranked_bm25 = sorted(candidate_indices, key=lambda i: float(bm25_scores[i]), reverse=True)[:top_k]
        for i in ranked_bm25:
            scores[i] += (float(bm25_scores[i]) / bm25_max) * W_BM25
    else:
        bm25_max = float(bm25_scores.max()) + 1e-9
        for i in np.argsort(bm25_scores)[::-1][:top_k]:
            scores[i] += (float(bm25_scores[i]) / bm25_max) * W_BM25

    if embedder is not None and faiss_idx is not None:
        q_vec = embedder.encode([q], normalize_embeddings=True).astype("float32")
        if candidate_indices:
            faiss_limit = min(len(addresses), max(top_k * 20, 500))
            dists, inds = faiss_idx.search(q_vec, faiss_limit)
            kept = 0
            blocked_set = set(candidate_indices)
            for j, idx in enumerate(inds[0]):
                if idx not in blocked_set:
                    continue
                scores[idx] += float(dists[0][j]) * W_FAISS
                kept += 1
                if kept >= top_k:
                    break
        else:
            dists, inds = faiss_idx.search(q_vec, top_k)
            for j, idx in enumerate(inds[0]):
                scores[idx] += float(dists[0][j]) * W_FAISS

    if use_fuzzy:
        fuzzy_iter = candidate_indices if candidate_indices else range(len(addresses))
        for idx in fuzzy_iter:
            addr = addresses[idx]
            fs = fuzz.token_sort_ratio(q, addr) / 100.0
            if fs > 0.55:
                scores[idx] += fs * W_FUZZY

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [(idx, addresses[idx], float(base_score)) for idx, base_score in ranked[:top_k]]


# -- step 3: reranking -------------------------------------------------------
def rerank_candidates(
    query: str,
    candidates,
    embedder,
    embeddings,
    bm25,
    ranker,
    addresses,
    addr_to_idx,
    top_n=5,
):
    if ranker is None:
        return [(idx, addr, score) for idx, addr, score in candidates[:top_n]]

    q = normalize(query)
    bm25_scores = bm25.get_scores(q.split())
    q_vec = None
    if embedder is not None:
        q_vec = embedder.encode([q], normalize_embeddings=True).astype("float32")

    feats = []
    base_rows = []
    for idx, addr, _ in candidates:
        addr_idx = addr_to_idx.get(addr, idx)
        b_score = float(bm25_scores[addr_idx]) if addr_idx < len(bm25_scores) else 0.0

        if q_vec is not None and embeddings is not None and 0 <= addr_idx < len(embeddings):
            f_score = float(np.dot(q_vec[0], embeddings[addr_idx]))
        elif q_vec is not None:
            a_vec = embedder.encode([addr], normalize_embeddings=True).astype("float32")
            f_score = float(np.dot(q_vec[0], a_vec[0]))
        else:
            f_score = fuzz.token_sort_ratio(q, addr) / 100.0

        q_tok = set(q.split())
        c_tok = set(addr.split())
        overlap = len(q_tok & c_tok) / (len(q_tok) + 1e-9)
        len_diff = abs(len(q) - len(addr)) / (len(q) + 1e-9)
        fuzz_tsr = fuzz.token_sort_ratio(q, addr) / 100.0
        fuzz_pr = fuzz.partial_ratio(q, addr) / 100.0
        max_len = max(len(q), len(addr)) + 1e-9
        edit_sim = 1.0 - (
            sum(a != b for a, b in zip(q, addr)) + abs(len(q) - len(addr))
        ) / max_len

        q_nums = {t for t in q.split() if t.isdigit()}
        c_nums = {t for t in addr.split() if t.isdigit()}
        num_match = float(bool(q_nums & c_nums)) if q_nums else 0.5

        feats.append([
            b_score,
            f_score,
            fuzz_tsr,
            fuzz_pr,
            edit_sim,
            overlap,
            len_diff,
            num_match,
        ])
        base_rows.append((idx, addr))

    x = pd.DataFrame(feats, columns=FEATURE_NAMES)
    rerank_scores = ranker.predict_proba(x)[:, 1]

    ranked = sorted(
        zip(base_rows, rerank_scores),
        key=lambda x: -float(x[1]),
    )

    out = []
    for (idx, addr), score in ranked[:top_n]:
        out.append((idx, addr, float(score)))
    return out


# -- full pipeline ------------------------------------------------------------
def correct_address(raw_input: str, models, top_n=FINAL_TOP_N):
    (
        t5_tok,
        t5,
        embedder,
        faiss_idx,
        embeddings,
        addresses,
        addr_ids,
        bm25,
        ranker,
        addr_to_idx,
        addrid_to_idx,
        rows_by_id,
        spell_checker,
        exact_lookup,
        engine,
    ) = models

    if spell_checker is not None:
        fixed_raw, spell_changes = spell_checker.correct(raw_input)
    else:
        fixed_raw, spell_changes = _conservative_correct_from_raw_with_changes(raw_input, addresses)
    fixed_input = _format_generated_address(fixed_raw)

    raw_norm = normalize(raw_input)
    fixed_norm_for_lookup = normalize(fixed_raw)
    exact_idx = exact_lookup.get(raw_norm)
    exact_source = "raw"
    if exact_idx is None and fixed_norm_for_lookup:
        exact_idx = exact_lookup.get(fixed_norm_for_lookup)
        exact_source = "spell_corrected"
    if exact_idx is not None:
        exact_addr = addresses[exact_idx]
        db_id = int(addr_ids[exact_idx]) if exact_idx < len(addr_ids) else None
        record = rows_by_id.get(db_id, {}) if db_id is not None else {}
        return {
            "original": raw_input,
            "corrected": exact_addr,
            "corrected_input": fixed_input if spell_changes else None,
            "spell_changes": spell_changes,
            "top_matches": [
                {
                    "full_address": exact_addr,
                    "score": EXACT_MATCH_CONFIDENCE,
                    "db_id": db_id,
                    "structured": record,
                }
            ],
            "best_match": exact_addr,
            "confidence": EXACT_MATCH_CONFIDENCE,
            "fixed_input": fixed_input,
            "warnings": [],
            "status": "exact_match",
            "scoring": f"exact_lookup_{exact_source}",
            "sql_blocked_candidates": 1,
            "changed": normalize(raw_input) != normalize(exact_addr),
        }

    field_check = detect_fields(fixed_input)
    if not field_check["has_number"] and not field_check["has_city"]:
        return {
            "original": raw_input,
            "corrected": None,
            "corrected_input": fixed_input if spell_changes else None,
            "spell_changes": spell_changes,
            "top_matches": [],
            "best_match": None,
            "confidence": 0.0,
            "fixed_input": fixed_input,
            "warnings": field_check["warnings"],
            "status": "incomplete",
            "sql_blocked_candidates": 0,
        }

    # Step A: generation-first, then strict DB verification.
    t5_source = fixed_input or raw_input
    corrected = t5_correct(t5_source, t5_tok, t5, num_beams=T5_BEAMS)
    allow_pin_block = not field_check.get("has_partial_pincode", False)
    blocked_indices, blocked_total_count, block_strategy = _fetch_blocked_indices(
        engine,
        corrected,
        addrid_to_idx,
        allow_pincode=allow_pin_block,
    )
    blocked_count = len(blocked_indices) if blocked_indices else 0

    pool_indices = blocked_indices if blocked_indices else list(range(len(addresses)))
    exists_hit = _check_exists_in_pool(
        corrected,
        pool_indices,
        addresses,
        addr_ids,
        rows_by_id,
    )
    if exists_hit is None and blocked_indices:
        # The SQL block can occasionally miss the true row when the query is
        # missing city/state tokens. Fall back to the full corpus before
        # downgrading to a generated-only response.
        exists_hit = _check_exists_in_pool(
            corrected,
            list(range(len(addresses))),
            addresses,
            addr_ids,
            rows_by_id,
        )
    if exists_hit is not None:
        return {
            "original": raw_input,
            "corrected": exists_hit["address"],
            "t5_output": corrected,
            "corrected_input": fixed_input if spell_changes else None,
            "spell_changes": spell_changes,
            "best_match": exists_hit["address"],
            "confidence": round(exists_hit["verification_score"] / 100.0, 4),
            "verification_score": exists_hit["verification_score"],
            "similarity_score": exists_hit["verification_score"],
            "fixed_input": fixed_input,
            "top_matches": [
                {
                    "full_address": exists_hit["address"],
                    "score": round(exists_hit["verification_score"] / 100.0, 4),
                    "db_id": exists_hit["db_id"],
                    "structured": exists_hit["record"],
                }
            ],
            "warnings": field_check["warnings"],
            "status": "verified_db_match",
            "scoring": "strict_verifier",
            "sql_blocked_candidates": blocked_count,
            "sql_blocked_candidates_total": blocked_total_count,
            "sql_block_strategy": block_strategy,
            "changed": normalize(raw_input) != normalize(exists_hit["address"]),
        }

    candidates_pre = retrieve_candidates(
        corrected,
        embedder,
        faiss_idx,
        addresses,
        bm25,
        top_k=RETRIEVAL_TOP_K,
        blocked_indices=blocked_indices,
    )

    # Query fusion: combine retrieval from model-corrected and spell-fixed forms.
    fixed_norm = normalize(fixed_input)
    corrected_norm = normalize(corrected)
    if fixed_norm and fixed_norm != corrected_norm:
        candidates_fixed = retrieve_candidates(
            fixed_norm,
            embedder,
            faiss_idx,
            addresses,
            bm25,
            top_k=RETRIEVAL_TOP_K,
            blocked_indices=blocked_indices,
        )
        candidates_pre = _merge_candidates(candidates_pre, candidates_fixed, RETRIEVAL_TOP_K)

    final_pre = rerank_candidates(
        corrected,
        candidates_pre,
        embedder,
        embeddings,
        bm25,
        ranker,
        addresses,
        addr_to_idx,
        top_n=top_n,
    )
    top_results_pre = _build_top_results(final_pre, addresses, addr_ids, rows_by_id)
    best_pre_conf = top_results_pre[0]["score"] if top_results_pre else 0.0

    # If pin-based blocking produced weak confidence, retry with relaxed
    # city/state blocking (ignore pincode) before falling back to generation.
    pin_locked_strategy = block_strategy in {"pin_city_state", "pincode"}
    if pin_locked_strategy and best_pre_conf < NO_MATCH_CONFIDENCE_THRESHOLD:
        relaxed_indices, relaxed_total_count, relaxed_strategy = _fetch_blocked_indices(
            engine,
            corrected,
            addrid_to_idx,
            allow_pincode=False,
        )
        if relaxed_indices:
            relaxed_exists = _check_exists_in_pool(
                corrected,
                relaxed_indices,
                addresses,
                addr_ids,
                rows_by_id,
            )
            if relaxed_exists is None:
                relaxed_exists = _check_exists_in_pool(
                    corrected,
                    list(range(len(addresses))),
                    addresses,
                    addr_ids,
                    rows_by_id,
                )

            if relaxed_exists is not None:
                return {
                    "original": raw_input,
                    "corrected": relaxed_exists["address"],
                    "t5_output": corrected,
                    "corrected_input": fixed_input if spell_changes else None,
                    "spell_changes": spell_changes,
                    "best_match": relaxed_exists["address"],
                    "confidence": round(relaxed_exists["verification_score"] / 100.0, 4),
                    "verification_score": relaxed_exists["verification_score"],
                    "similarity_score": relaxed_exists["verification_score"],
                    "fixed_input": fixed_input,
                    "top_matches": [
                        {
                            "full_address": relaxed_exists["address"],
                            "score": round(relaxed_exists["verification_score"] / 100.0, 4),
                            "db_id": relaxed_exists["db_id"],
                            "structured": relaxed_exists["record"],
                        }
                    ],
                    "warnings": field_check["warnings"],
                    "status": "verified_db_match",
                    "scoring": "strict_verifier_relaxed_geo",
                    "sql_blocked_candidates": len(relaxed_indices),
                    "sql_blocked_candidates_total": relaxed_total_count,
                    "sql_block_strategy": relaxed_strategy,
                    "changed": normalize(raw_input) != normalize(relaxed_exists["address"]),
                }

            relaxed_candidates = retrieve_candidates(
                corrected,
                embedder,
                faiss_idx,
                addresses,
                bm25,
                top_k=RETRIEVAL_TOP_K,
                blocked_indices=relaxed_indices,
            )
            relaxed_final = rerank_candidates(
                corrected,
                relaxed_candidates,
                embedder,
                embeddings,
                bm25,
                ranker,
                addresses,
                addr_to_idx,
                top_n=top_n,
            )
            relaxed_top = _build_top_results(relaxed_final, addresses, addr_ids, rows_by_id)
            relaxed_best_conf = relaxed_top[0]["score"] if relaxed_top else 0.0
            if relaxed_best_conf > best_pre_conf:
                top_results_pre = relaxed_top
                best_pre_conf = relaxed_best_conf
                blocked_indices = relaxed_indices
                blocked_count = len(relaxed_indices)
                blocked_total_count = relaxed_total_count
                block_strategy = relaxed_strategy

    # Last accuracy attempt: if blocked retrieval is weak, try global retrieval.
    if best_pre_conf < NO_MATCH_CONFIDENCE_THRESHOLD and blocked_indices is not None:
        global_candidates = retrieve_candidates(
            corrected,
            embedder,
            faiss_idx,
            addresses,
            bm25,
            top_k=RETRIEVAL_TOP_K,
            blocked_indices=None,
            use_fuzzy=False,
        )
        if fixed_norm and fixed_norm != corrected_norm:
            global_candidates_fixed = retrieve_candidates(
                fixed_norm,
                embedder,
                faiss_idx,
                addresses,
                bm25,
                top_k=RETRIEVAL_TOP_K,
                blocked_indices=None,
                use_fuzzy=False,
            )
            global_candidates = _merge_candidates(global_candidates, global_candidates_fixed, RETRIEVAL_TOP_K)

        global_final = rerank_candidates(
            corrected,
            global_candidates,
            embedder,
            embeddings,
            bm25,
            ranker,
            addresses,
            addr_to_idx,
            top_n=top_n,
        )
        global_top = _build_top_results(global_final, addresses, addr_ids, rows_by_id)
        global_best_conf = global_top[0]["score"] if global_top else 0.0
        if global_best_conf > best_pre_conf:
            top_results_pre = global_top
            best_pre_conf = global_best_conf
            blocked_indices = None
            blocked_count = 0
            blocked_total_count = len(addresses)
            block_strategy = "global_fallback"

    best_conf = top_results_pre[0]["score"] if top_results_pre else 0.0

    # If model confidence is low but a DB candidate is strongly similar,
    # prefer the DB candidate over raw generated output.
    if top_results_pre and best_conf < NO_MATCH_CONFIDENCE_THRESHOLD:
        best_addr = top_results_pre[0]["full_address"]
        fuzzy_exists = max(
            fuzz.token_set_ratio(corrected, best_addr),
            fuzz.token_sort_ratio(corrected, best_addr),
        )
        if fuzzy_exists >= FUZZY_EXISTS_THRESHOLD:
            return {
                "original": raw_input,
                "corrected": best_addr,
                "t5_output": corrected,
                "corrected_input": fixed_input if spell_changes else None,
                "spell_changes": spell_changes,
                "best_match": best_addr,
                "confidence": round(float(best_conf), 4),
                "verification_score": 0.0,
                "similarity_score": round(float(fuzzy_exists), 2),
                "fixed_input": fixed_input,
                "top_matches": top_results_pre,
                "warnings": field_check["warnings"],
                "status": "candidate_db_match",
                "scoring": "fuzzy_candidate_preferred",
                "sql_blocked_candidates": blocked_count,
                "sql_blocked_candidates_total": blocked_total_count,
                "sql_block_strategy": block_strategy,
                "changed": normalize(raw_input) != normalize(best_addr),
            }

    if best_conf < NO_MATCH_CONFIDENCE_THRESHOLD:
        # Second relaxed pass for generated-only paths:
        # run one broader candidate search and promote high-fuzzy DB hits.
        rescue_query = fixed_norm or corrected_norm or raw_norm
        rescue_candidates = retrieve_candidates(
            rescue_query,
            embedder,
            faiss_idx,
            addresses,
            bm25,
            top_k=max(RETRIEVAL_TOP_K, 120),
            blocked_indices=None,
            use_fuzzy=False,
        )
        rescue_final = rerank_candidates(
            rescue_query,
            rescue_candidates,
            embedder,
            embeddings,
            bm25,
            ranker,
            addresses,
            addr_to_idx,
            top_n=top_n,
        )
        rescue_top = _build_top_results(rescue_final, addresses, addr_ids, rows_by_id)
        if rescue_top:
            rescue_best = rescue_top[0]["full_address"]
            rescue_best_conf = rescue_top[0]["score"]
            rescue_fuzzy = max(
                fuzz.token_set_ratio(rescue_query, rescue_best),
                fuzz.token_sort_ratio(rescue_query, rescue_best),
            )

            if rescue_fuzzy >= FUZZY_EXISTS_THRESHOLD:
                return {
                    "original": raw_input,
                    "corrected": rescue_best,
                    "t5_output": corrected,
                    "corrected_input": fixed_input if spell_changes else None,
                    "spell_changes": spell_changes,
                    "best_match": rescue_best,
                    "confidence": round(float(rescue_best_conf), 4),
                    "verification_score": 0.0,
                    "similarity_score": round(float(rescue_fuzzy), 2),
                    "fixed_input": fixed_input,
                    "top_matches": rescue_top,
                    "warnings": field_check["warnings"],
                    "status": "candidate_db_match",
                    "scoring": "fuzzy_candidate_rescue",
                    "sql_blocked_candidates": 0,
                    "sql_blocked_candidates_total": len(addresses),
                    "sql_block_strategy": "global_rescue",
                    "changed": normalize(raw_input) != normalize(rescue_best),
                }

        # Final safety net: direct global fuzzy against full address strings.
        # This helps recover records when city/pincode are missing in input but
        # locality and house tokens are strong enough to identify a row.
        fuzzy_hit = process.extractOne(
            rescue_query,
            addresses,
            scorer=fuzz.partial_ratio,
        )
        if fuzzy_hit:
            fuzzy_addr, fuzzy_partial, _ = fuzzy_hit
            fuzzy_set = fuzz.token_set_ratio(rescue_query, fuzzy_addr)
            q_inf = _informative_tokens(rescue_query)
            a_inf = _informative_tokens(fuzzy_addr)
            inf_overlap = (len(q_inf & a_inf) / max(len(q_inf), 1)) if q_inf else 0.0

            if float(fuzzy_partial) >= 93.0 and float(fuzzy_set) >= 74.0 and inf_overlap >= 0.50:
                fuzzy_idx = addr_to_idx.get(fuzzy_addr)
                fuzzy_db_id = int(addr_ids[fuzzy_idx]) if fuzzy_idx is not None and fuzzy_idx < len(addr_ids) else None
                fuzzy_record = rows_by_id.get(fuzzy_db_id, {}) if fuzzy_db_id is not None else {}
                fuzzy_top = [
                    {
                        "full_address": fuzzy_addr,
                        "score": round(float(fuzzy_partial) / 100.0, 4),
                        "db_id": fuzzy_db_id,
                        "structured": fuzzy_record,
                    }
                ]
                return {
                    "original": raw_input,
                    "corrected": fuzzy_addr,
                    "t5_output": corrected,
                    "corrected_input": fixed_input if spell_changes else None,
                    "spell_changes": spell_changes,
                    "best_match": fuzzy_addr,
                    "confidence": round(float(fuzzy_partial) / 100.0, 4),
                    "verification_score": 0.0,
                    "similarity_score": round(float(fuzzy_partial), 2),
                    "fixed_input": fixed_input,
                    "top_matches": fuzzy_top,
                    "warnings": field_check["warnings"],
                    "status": "candidate_db_match",
                    "scoring": "direct_fuzzy_global",
                    "sql_blocked_candidates": 0,
                    "sql_blocked_candidates_total": len(addresses),
                    "sql_block_strategy": "direct_fuzzy_global",
                    "changed": normalize(raw_input) != normalize(fuzzy_addr),
                }

        generated = fixed_input or _format_generated_address(corrected)
        near_one = _nearby_single_suggestion(
            generated,
            addresses,
            addr_ids,
            rows_by_id,
            addr_to_idx,
        )
        gen_top_matches = [
            {
                "full_address": generated,
                "score": round(float(best_conf), 4),
                "db_id": None,
                "structured": {},
            }
        ]
        if near_one is not None:
            gen_top_matches.append(near_one)
        return {
            "original": raw_input,
            "corrected": generated,
            "t5_output": corrected,
            "corrected_input": fixed_input if spell_changes else None,
            "spell_changes": spell_changes,
            "best_match": generated,
            "confidence": round(float(best_conf), 4),
            "verification_score": 0.0,
            "similarity_score": round(float(best_conf), 4),
            "fixed_input": fixed_input,
            "top_matches": gen_top_matches,
            "warnings": field_check["warnings"] + [
                "No reliable database match found; returning generated corrected format."
            ],
            "status": "generated_only",
            "scoring": "fallback_generated",
            "sql_blocked_candidates": blocked_count,
            "sql_blocked_candidates_total": blocked_total_count,
            "sql_block_strategy": block_strategy,
            "changed": normalize(raw_input) != normalize(generated),
        }

    return {
        "original": raw_input,
        "corrected": top_results_pre[0]["full_address"] if top_results_pre else corrected,
        "t5_output": corrected,
        "corrected_input": fixed_input if spell_changes else None,
        "spell_changes": spell_changes,
        "best_match": top_results_pre[0]["full_address"] if top_results_pre else "",
        "confidence": top_results_pre[0]["score"] if top_results_pre else 0.0,
        "verification_score": 0.0,
        "similarity_score": top_results_pre[0]["score"] if top_results_pre else 0.0,
        "fixed_input": fixed_input,
        "top_matches": top_results_pre,
        "warnings": field_check["warnings"],
        "status": "candidate_db_match",
        "scoring": "reranker_generated" if ranker is not None else "retrieval_only_generated",
        "sql_blocked_candidates": blocked_count,
        "sql_blocked_candidates_total": blocked_total_count,
        "sql_block_strategy": block_strategy,
        "changed": normalize(raw_input) != normalize(corrected),
    }


# -- interactive --------------------------------------------------------------
def print_structured_record(record: dict):
    if not record:
        print("    No structured row found.")
        return

    ordered_fields = [
        "address_id",
        "source_raw_address",
        "normalized_full_address",
        "house_no",
        "flat_no",
        "floor",
        "house_name",
        "apartment_name",
        "landmark",
        "street",
        "area",
        "town",
        "village",
        "district",
        "state",
        "country",
        "pincode",
    ]
    for field in ordered_fields:
        val = record.get(field)
        if val not in (None, ""):
            print(f"    {field:<24}: {val}")


def interactive(models):
    rows_by_id = models[-2]

    print("\n" + "=" * 68)
    print("  Address Correction - SQL + Re-ranker Mode")
    print(f"  Structured rows: {len(rows_by_id):,}")
    print("  Type 'quit' to exit  |  'show N' to see full record")
    print("=" * 68)

    last_result = None
    while True:
        print()
        raw = input("Enter address: ").strip()

        if not raw:
            continue
        if raw.lower() in ("quit", "exit", "q"):
            break

        if raw.lower().startswith("show"):
            parts = raw.split()
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
            if last_result and last_result["top_matches"]:
                picked = last_result["top_matches"][
                    min(n - 1, len(last_result["top_matches"]) - 1)
                ]
                print("\nFull structured record:")
                print_structured_record(picked["structured"])
            continue

        result = correct_address(raw, models)
        last_result = result

        if result.get("spell_changes"):
            print(f"  [SPELL] Corrections: {result['spell_changes']}")
            if result.get("corrected_input"):
                print(f"    Fixed input: {result['corrected_input']}")

        if result["warnings"]:
            print("Warnings:")
            for warning in result["warnings"]:
                print(f"  - {warning}")

        if result["status"] == "incomplete":
            print("\nToo incomplete to search. Add missing fields.")
            continue

        print(f"\nGenerated  : {result['corrected']}")
        print(f"Best match : {result['best_match']}")
        print(f"Status     : {result.get('status', 'unknown')}")
        print(f"Confidence : {result['confidence']:.3f} ({result['scoring']})")
        if "verification_score" in result:
            print(f"Verify     : {result.get('verification_score', 0.0):.2f}")
        if "similarity_score" in result:
            print(f"Similarity : {result.get('similarity_score', 0.0):.3f}")
        print(
            f"SQL Block  : {result.get('sql_blocked_candidates_total', 0)} "
            f"(used: {result.get('sql_blocked_candidates', 0)}) "
            f"[{result.get('sql_block_strategy', 'na')}]"
        )

        if result["top_matches"]:
            print("\nBest match structured fields:")
            print_structured_record(result["top_matches"][0]["structured"])

        if len(result["top_matches"]) > 1:
            print("\nOther possibilities:")
            for i, m in enumerate(result["top_matches"][1:2], 2):
                print(f"  {i}. [{m['score']:.3f}] {m['full_address']}")
            print("Type 'show 2' to inspect option 2 in detail.")


def main():
    models = load_models()
    interactive(models)


if __name__ == "__main__":
    main()
