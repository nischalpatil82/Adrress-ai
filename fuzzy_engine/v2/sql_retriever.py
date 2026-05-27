"""
fuzzy_engine.v2.sql_retriever
=============================
Live SQL blocking retriever for the v2 pipeline.

Connects to the client's MySQL database, uses blocking-term SQL LIKE
queries, and returns candidates in the v2 Candidate format.

Use case:
    - Client has a live address database (may contain typos).
    - We still want spell correction + fuzzy matching to work.
    - SQL results are merged with BM25/FAISS static-index candidates.

Configuration via environment variables:
    DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME
    V2_SQL_BLOCK_LIMIT (default 200)
    V2_SQL_MATCH_THRESHOLD (default 60, rapidfuzz ratio)

Enable / disable:
    V2_USE_SQL_DB=1 (default 1 — enabled)
"""
from __future__ import annotations

import logging
import os
from typing import Iterable

from rapidfuzz import fuzz

from fuzzy_engine.db_loader import (
    _build_db_url,
    _blocking_terms,
    get_engine as _legacy_get_engine,
)
from fuzzy_engine.v2.normalize import normalize_text
from fuzzy_engine.v2.retrieval import Candidate

log = logging.getLogger(__name__)

DEFAULT_SQL_LIMIT = int(os.getenv("V2_SQL_BLOCK_LIMIT", "200"))
DEFAULT_MATCH_THRESHOLD = int(os.getenv("V2_SQL_MATCH_THRESHOLD", "60"))


def _compose_address(
    source_raw: str | None,
    normalized: str | None,
    house_no: str | None,
    flat_no: str | None,
    floor: str | None,
    house_name: str | None,
    apartment_name: str | None,
    landmark: str | None,
    street: str | None,
    area: str | None,
    town: str | None,
    village: str | None,
    district: str | None,
    state: str | None,
    country: str | None,
    pincode: str | None,
) -> str:
    """Reconstruct a full address from structured DB columns."""
    parts = []
    for key in (
        "house_no", "flat_no", "floor", "house_name",
        "apartment_name", "landmark", "street", "area",
        "town", "village", "district", "state", "country", "pincode",
    ):
        val = locals()[key]
        if val is not None:
            txt = str(val).strip()
            if txt:
                parts.append(txt)
    if parts:
        return " ".join(parts)
    # fallback
    return (normalized or source_raw or "").strip()


class SQLRetriever:
    """Lightweight live-SQL candidate source for the v2 pipeline."""

    def __init__(
        self,
        engine=None,
        limit: int = DEFAULT_SQL_LIMIT,
        match_threshold: int = DEFAULT_MATCH_THRESHOLD,
    ) -> None:
        self._engine = engine
        self.limit = limit
        self.match_threshold = match_threshold

    # ------------------------------------------------------------------
    # lazy engine init (so import doesn't require a live DB)
    # ------------------------------------------------------------------
    @property
    def _db(self):
        if self._engine is None:
            self._engine = _legacy_get_engine()
        return self._engine

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def search(self, query: str) -> list[Candidate]:
        """
        Return Candidate objects from the live DB that fuzzily match `query`.
        Uses SQL blocking (LIKE on blocking terms) then rapidfuzz scoring.
        """
        if not query or len(query.strip()) < 3:
            return []

        norm_query = normalize_text(query)
        city_terms, number_terms, text_terms = _blocking_terms(norm_query)
        all_terms = city_terms + number_terms + text_terms
        if not all_terms:
            return []

        # Reuse the legacy blocking logic but adapt to our own SQL shape.
        ordered_terms = city_terms[:2] + number_terms[:2] + text_terms[:6]
        if not ordered_terms:
            return []

        rows = self._fetch_rows(ordered_terms)
        if not rows:
            return []

        # Score each DB row against the (spell-corrected) user query.
        # We use rapidfuzz because the DB itself may contain typos.
        scored: list[tuple[float, Candidate]] = []
        seen: set[int] = set()
        for row in rows:
            addr_id = int(row.get("address_id", 0))
            if addr_id in seen:
                continue
            seen.add(addr_id)

            addr = _compose_address(
                row.get("source_raw_address"),
                row.get("normalized_full_address"),
                row.get("house_no"),
                row.get("flat_no"),
                row.get("floor"),
                row.get("house_name"),
                row.get("apartment_name"),
                row.get("landmark"),
                row.get("street"),
                row.get("area"),
                row.get("town"),
                row.get("village"),
                row.get("district"),
                row.get("state"),
                row.get("country"),
                row.get("pincode"),
            )
            if not addr:
                continue

            norm_addr = normalize_text(addr)
            ratio = fuzz.ratio(norm_query, norm_addr)
            if ratio >= self.match_threshold:
                # Convert 0-100 score to 0-1 probability-like score.
                scored.append((ratio / 100.0, Candidate(addr_id, addr, {"sql": ratio / 100.0})))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored]

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _fetch_rows(self, ordered_terms: list[str]) -> list[dict]:
        """Run a SQL LIKE blocking query and return rows as dicts."""
        from sqlalchemy import text

        params: dict = {"limit": int(self.limit)}
        like_clauses: list[str] = []

        for i, term in enumerate(ordered_terms):
            key = f"t{i}"
            params[key] = f"%{term}%"
            expr = (
                f"LOWER(COALESCE(normalized_full_address, '')) LIKE :{key} "
                f"OR LOWER(COALESCE(source_raw_address, '')) LIKE :{key} "
                f"OR LOWER(COALESCE(street, '')) LIKE :{key} "
                f"OR LOWER(COALESCE(area, '')) LIKE :{key} "
                f"OR LOWER(COALESCE(house_name, '')) LIKE :{key} "
                f"OR LOWER(COALESCE(apartment_name, '')) LIKE :{key} "
                f"OR LOWER(COALESCE(landmark, '')) LIKE :{key} "
                f"OR LOWER(COALESCE(town, '')) LIKE :{key} "
                f"OR LOWER(COALESCE(village, '')) LIKE :{key} "
                f"OR LOWER(COALESCE(district, '')) LIKE :{key} "
                f"OR LOWER(COALESCE(state, '')) LIKE :{key} "
                f"OR LOWER(COALESCE(pincode, '')) LIKE :{key}"
            )
            like_clauses.append(expr)

        if not like_clauses:
            return []

        where_sql = " OR ".join(like_clauses)

        query_sql = text(
            f"""
            SELECT
                address_id,
                source_raw_address,
                normalized_full_address,
                house_no,
                flat_no,
                floor,
                house_name,
                apartment_name,
                landmark,
                street,
                area,
                town,
                village,
                district,
                state,
                country,
                pincode
            FROM addresses
            WHERE {where_sql}
            ORDER BY address_id ASC
            LIMIT :limit
            """
        )

        try:
            with self._db.connect() as conn:
                rows = conn.execute(query_sql, params).mappings().all()
            return [dict(row) for row in rows]
        except Exception as exc:
            log.warning("SQL blocking query failed: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Optional global instance (lazy, won't connect until first search)
# ---------------------------------------------------------------------------
_sql_retriever: SQLRetriever | None = None


def get_sql_retriever() -> SQLRetriever | None:
    """Return a cached SQLRetriever, or None if V2_USE_SQL_DB is disabled."""
    global _sql_retriever
    if os.getenv("V2_USE_SQL_DB", "1").lower() in ("0", "false", "no", "off"):
        return None
    if _sql_retriever is None:
        _sql_retriever = SQLRetriever()
    return _sql_retriever
