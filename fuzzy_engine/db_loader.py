"""
fuzzy_engine.db_loader
======================
Database loader module for the enterprise address correction engine.

Loads addresses from MySQL/MariaDB using SQLAlchemy, serving as the
primary data source for the correction engine.

Connection is configured via environment variables or defaults:
    DB_HOST     = 127.0.0.1
    DB_PORT     = 3306
    DB_USER     = root
    DB_PASSWORD = root
    DB_NAME     = address_ai
"""

import os
import re
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from fuzzy_engine.config import (
    SQL_BLOCK_LIMIT,
    SQL_BLOCK_MIN_TOKEN_LEN,
    SQL_BLOCK_STOP_WORDS,
)
from fuzzy_engine.dictionaries import KNOWN_CITIES


def _normalize_for_blocking(addr: str) -> str:
    addr = re.sub(r"[^\w\s]", " ", str(addr).lower())
    return re.sub(r"\s+", " ", addr).strip()


def _blocking_terms(query: str) -> tuple[list[str], list[str], list[str]]:
    """Return (city_terms, number_terms, text_terms) for SQL blocking."""
    tokens = _normalize_for_blocking(query).split()
    if not tokens:
        return [], [], []

    city_terms = [t for t in tokens if t in KNOWN_CITIES]
    number_terms = [t for t in tokens if t.isdigit()]

    text_terms = []
    for t in tokens:
        if t.isdigit():
            continue
        if len(t) < SQL_BLOCK_MIN_TOKEN_LEN:
            continue
        if t in SQL_BLOCK_STOP_WORDS:
            continue
        text_terms.append(t)

    # Keep deterministic order while deduplicating.
    def _dedupe(seq):
        seen = set()
        out = []
        for x in seq:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    return _dedupe(city_terms), _dedupe(number_terms), _dedupe(text_terms)


def _build_db_url() -> str:
    """Build MySQL connection URL from env vars or defaults."""
    explicit = os.getenv("DB_URL")
    if explicit:
        return explicit

    host     = os.getenv("DB_HOST",     "127.0.0.1")
    port     = os.getenv("DB_PORT",     "3306")
    user     = os.getenv("DB_USER",     "root")
    password = os.getenv("DB_PASSWORD", "root")
    database = os.getenv("DB_NAME",     "address_ai")

    return f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4"


def get_engine() -> Engine:
    """Create and return a SQLAlchemy engine with connection pooling."""
    return create_engine(
        _build_db_url(),
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        future=True,
    )


def load_addresses_from_db(engine: Engine = None) -> list:
    """
    Load all addresses from the MySQL `addresses` table.

    Tries multiple columns in priority order:
      1. normalized_full_address
      2. source_raw_address

    Returns:
        list of address strings
    """
    if engine is None:
        engine = get_engine()

    query = text("""
        SELECT
            address_id,
            source_raw_address,
            normalized_full_address
        FROM addresses
        ORDER BY address_id
    """)

    addresses = []
    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()

    for row in rows:
        # Prefer normalized_full_address, fall back to source_raw_address
        addr = (
            (row.get("normalized_full_address") or "").strip()
            or (row.get("source_raw_address") or "").strip()
        )
        if addr:
            addresses.append(addr)

    return addresses


def load_blocked_addresses_from_db(
    query: str,
    engine: Engine = None,
    limit: int = SQL_BLOCK_LIMIT,
    with_count: bool = False,
) -> list | tuple[list, int]:
    """
    Fetch a narrowed candidate set for a query using SQL blocking clauses.

    This reduces in-memory fuzzy matching cost by searching only likely rows.
    Falls back to empty list when no meaningful block terms exist.
    """
    if engine is None:
        engine = get_engine()

    city_terms, number_terms, text_terms = _blocking_terms(query)
    all_terms = city_terms + number_terms + text_terms
    if not all_terms:
        return []

    # Prefer strong terms first so query remains selective.
    ordered_terms = city_terms[:2] + number_terms[:2] + text_terms[:6]
    if not ordered_terms:
        return []

    params = {"limit": int(limit)}
    like_clauses = []
    match_score_exprs = []

    for i, term in enumerate(ordered_terms):
        key = f"t{i}"
        params[key] = f"%{term}%"
        expr = (
            f"(LOWER(COALESCE(normalized_full_address, '')) LIKE :{key} "
            f"OR LOWER(COALESCE(source_raw_address, '')) LIKE :{key})"
        )
        like_clauses.append(expr)
        match_score_exprs.append(f"CASE WHEN {expr} THEN 1 ELSE 0 END")

    where_sql = " OR ".join(like_clauses)
    score_sql = " + ".join(match_score_exprs)

    total_count = None
    if with_count:
        count_sql = text(
            f"""
            SELECT COUNT(*) AS cnt
            FROM addresses
            WHERE {where_sql}
            """
        )
        with engine.connect() as conn:
            total_count = conn.execute(count_sql, params).scalar_one()

    query_sql = text(
        f"""
        SELECT
            source_raw_address,
            normalized_full_address,
            ({score_sql}) AS block_score
        FROM addresses
        WHERE {where_sql}
        ORDER BY block_score DESC, address_id ASC
        LIMIT :limit
        """
    )

    with engine.connect() as conn:
        rows = conn.execute(query_sql, params).mappings().all()

    out = []
    for row in rows:
        addr = (
            (row.get("normalized_full_address") or "").strip()
            or (row.get("source_raw_address") or "").strip()
        )
        if addr:
            out.append(addr)
    if with_count:
        return out, int(total_count or 0)
    return out


KNOWN_STATES = {
    "karnataka", "maharashtra", "delhi", "telangana", "tamil nadu",
    "west bengal", "uttar pradesh", "rajasthan", "gujarat", "kerala",
    "madhya pradesh", "andhra pradesh", "punjab", "haryana",
    "uttarakhand", "jharkhand", "chhattisgarh", "odisha", "assam",
    "goa", "bihar", "himachal pradesh", "jammu and kashmir",
}

# Map city → likely state for validation / enrichment.
CITY_STATE_MAP = {
    "bangalore": "karnataka", "mumbai": "maharashtra", "delhi": "delhi",
    "hyderabad": "telangana", "chennai": "tamil nadu", "kolkata": "west bengal",
    "pune": "maharashtra", "ahmedabad": "gujarat", "jaipur": "rajasthan",
    "noida": "uttar pradesh", "gurgaon": "haryana", "surat": "gujarat",
    "lucknow": "uttar pradesh", "nagpur": "maharashtra", "indore": "madhya pradesh",
    "bhopal": "madhya pradesh", "patna": "bihar", "vadodara": "gujarat",
    "chandigarh": "chandigarh", "kochi": "kerala", "mysuru": "karnataka",
    "coimbatore": "tamil nadu", "mangalore": "karnataka",
    "thiruvananthapuram": "kerala", "visakhapatnam": "andhra pradesh",
    "thane": "maharashtra", "navi mumbai": "maharashtra",
}


# Street-type keywords that signal a preceding token is a road/area name.
_ROAD_SUFFIXES = {"road", "rd", "marg", "highway", "lane", "drive", "avenue", "ave"}
_AREA_SUFFIXES = {"nagar", "layout", "colony", "puram", "pura", "halli",
                  "enclave", "extension", "vihar", "garden", "park"}
_COMMON_GENERIC = {
    "street", "road", "avenue", "nagar", "colony", "layout", "sector",
    "floor", "flat", "number", "apartment", "block", "phase", "near",
    "opposite", "behind", "main", "cross", "building", "society", "tower",
}


def _extract_street_anchor(query: str) -> str | None:
    """Extract the dominant street/area name from a query for SQL filtering."""
    tokens = _normalize_for_blocking(query).split()
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
                and tok not in KNOWN_CITIES
                and tok not in SQL_BLOCK_STOP_WORDS
                and tok not in _COMMON_GENERIC):
            if tok not in suffix_candidates and tok not in generic_candidates:
                generic_candidates.append(tok)

    if suffix_candidates:
        return max(suffix_candidates, key=len)
    return max(generic_candidates, key=len) if generic_candidates else None


def _extract_geo_anchors(query: str) -> dict:
    """Extract city, state, pincode, and street anchor from a query string."""
    tokens = _normalize_for_blocking(query).split()
    result = {"city": None, "state": None, "pincode": None, "street": None}

    for tok in tokens:
        if tok in KNOWN_CITIES:
            result["city"] = tok
        if tok.isdigit() and len(tok) == 6:
            result["pincode"] = tok

    # Check multi-word state names.
    norm = _normalize_for_blocking(query)
    for st in KNOWN_STATES:
        if st in norm:
            result["state"] = st
            break

    # Infer state from city if not explicitly provided.
    if result["city"] and not result["state"]:
        result["state"] = CITY_STATE_MAP.get(result["city"])

    result["street"] = _extract_street_anchor(query)

    return result


def load_geo_filtered_addresses_from_db(
    query: str,
    engine: Engine = None,
    limit: int = SQL_BLOCK_LIMIT,
    with_count: bool = False,
    min_geo_results: int = 10,
) -> list | tuple[list, int]:
    """
    Fetch candidates using structured column filtering (town, state, pincode)
    first, then refine with text LIKE terms within that set.

    Falls back to load_blocked_addresses_from_db if geo-filtering returns
    too few results or no geographic anchors are detected.
    """
    if engine is None:
        engine = get_engine()

    geo = _extract_geo_anchors(query)

    # If no geographic anchors detected, fall back to existing blocking.
    if not geo["city"] and not geo["state"] and not geo["pincode"]:
        return load_blocked_addresses_from_db(
            query, engine=engine, limit=limit, with_count=with_count,
        )

    # -- Phase 1: filter by structured columns --------------------------
    geo_clauses = []
    params = {"limit": int(limit)}

    if geo["pincode"]:
        params["pincode"] = geo["pincode"]
        geo_clauses.append("pincode = :pincode")

    if geo["city"]:
        params["city"] = geo["city"]
        geo_clauses.append(
            "(LOWER(COALESCE(town, '')) = :city "
            "OR LOWER(COALESCE(district, '')) = :city)"
        )

    if geo["state"]:
        params["state"] = geo["state"]
        geo_clauses.append("LOWER(COALESCE(state, '')) = :state")

    geo_where = " AND ".join(geo_clauses)

    # -- Phase 1b: add street anchor LIKE filter if detected ---------------
    street = geo.get("street")
    if street:
        params["street_anchor"] = f"%{street}%"
        geo_clauses.append(
            "(LOWER(COALESCE(normalized_full_address, '')) LIKE :street_anchor "
            "OR LOWER(COALESCE(source_raw_address, '')) LIKE :street_anchor "
            "OR LOWER(COALESCE(street, '')) LIKE :street_anchor "
            "OR LOWER(COALESCE(area, '')) LIKE :street_anchor)"
        )
        geo_where_with_street = " AND ".join(geo_clauses)
    else:
        geo_where_with_street = None

    # -- Phase 2: add text LIKE terms within geo-filtered set -----------
    _, number_terms, text_terms = _blocking_terms(query)
    text_like_terms = number_terms[:2] + text_terms[:6]

    like_clauses = []
    score_exprs = []
    for i, term in enumerate(text_like_terms):
        key = f"t{i}"
        params[key] = f"%{term}%"
        expr = (
            f"(LOWER(COALESCE(normalized_full_address, '')) LIKE :{key} "
            f"OR LOWER(COALESCE(source_raw_address, '')) LIKE :{key})"
        )
        like_clauses.append(expr)
        score_exprs.append(f"CASE WHEN {expr} THEN 1 ELSE 0 END")

    if score_exprs:
        score_sql = " + ".join(score_exprs)
        order_by = f"({score_sql}) DESC, address_id ASC"
    else:
        score_sql = "1"
        order_by = "address_id ASC"

    # Try street-narrowed query first for tighter candidate pool.
    active_where = geo_where_with_street if geo_where_with_street else geo_where

    query_sql = text(
        f"""
        SELECT
            source_raw_address,
            normalized_full_address,
            ({score_sql}) AS block_score
        FROM addresses
        WHERE {active_where}
        ORDER BY {order_by}
        LIMIT :limit
        """
    )

    total_count = None
    if with_count:
        count_sql = text(
            f"SELECT COUNT(*) AS cnt FROM addresses WHERE {active_where}"
        )
        with engine.connect() as conn:
            total_count = conn.execute(count_sql, params).scalar_one()

    with engine.connect() as conn:
        rows = conn.execute(query_sql, params).mappings().all()

    out = []
    for row in rows:
        addr = (
            (row.get("normalized_full_address") or "").strip()
            or (row.get("source_raw_address") or "").strip()
        )
        if addr:
            out.append(addr)

    # If the street-narrowed pass found anything, keep it. Broadening to
    # city/state here can reintroduce cross-street hallucinations such as
    # Bannerghatta Road queries returning Bellary Road candidates.
    if out and geo_where_with_street:
        if with_count:
            return out, int(total_count or len(out))
        return out

    # If street-narrowed returned nothing, retry with broader geo-only.
    if len(out) < min_geo_results and geo_where_with_street:
        query_sql_broad = text(
            f"""
            SELECT
                source_raw_address,
                normalized_full_address,
                ({score_sql}) AS block_score
            FROM addresses
            WHERE {geo_where}
            ORDER BY {order_by}
            LIMIT :limit
            """
        )
        # Remove street_anchor param for the broad query
        broad_params = {k: v for k, v in params.items() if k != "street_anchor"}
        if with_count:
            count_sql_broad = text(
                f"SELECT COUNT(*) AS cnt FROM addresses WHERE {geo_where}"
            )
            with engine.connect() as conn:
                total_count = conn.execute(count_sql_broad, broad_params).scalar_one()

        with engine.connect() as conn:
            rows = conn.execute(query_sql_broad, broad_params).mappings().all()

        out = []
        for row in rows:
            addr = (
                (row.get("normalized_full_address") or "").strip()
                or (row.get("source_raw_address") or "").strip()
            )
            if addr:
                out.append(addr)

    # Fall back to broader blocking if geo-filter returned too few.
    if len(out) < min_geo_results:
        return load_blocked_addresses_from_db(
            query, engine=engine, limit=limit, with_count=with_count,
        )

    if with_count:
        return out, int(total_count or 0)
    return out


def test_connection(engine: Engine = None) -> dict:
    """
    Test database connectivity and return basic stats.

    Returns:
        dict with keys: connected (bool), address_count (int), error (str or None)
    """
    if engine is None:
        engine = get_engine()

    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) AS cnt FROM addresses"))
            count = result.scalar()
        return {"connected": True, "address_count": count, "error": None}
    except Exception as e:
        return {"connected": False, "address_count": 0, "error": str(e)}
