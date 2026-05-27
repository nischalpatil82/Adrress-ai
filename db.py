import os
from typing import Dict, List, Tuple

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


DEFAULT_DB_PORT = "3306"


def _build_db_url() -> str:
    explicit = os.getenv("DB_URL")
    if explicit:
        return explicit

    host = os.getenv("DB_HOST", "127.0.0.1")
    port = os.getenv("DB_PORT", "3306")
    user = os.getenv("DB_USER", "root")
    password = os.getenv("DB_PASSWORD", "root")
    database = os.getenv("DB_NAME", "address_ai")

    return f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4"


def get_engine() -> Engine:
    return create_engine(_build_db_url(), pool_pre_ping=True, future=True)


def _compose_structured_address(row: Dict[str, str]) -> str:
    parts = []
    for key in (
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
    ):
        val = row.get(key)
        if val is None:
            continue
        txt = str(val).strip()
        if txt:
            parts.append(txt)
    return " ".join(parts)


def fetch_canonical_addresses(engine: Engine) -> List[Tuple[int, str]]:
    query = text(
        """
        SELECT
            address_id,
            normalized_full_address,
            source_raw_address,
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
        ORDER BY address_id
        """
    )

    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()

    out: List[Tuple[int, str]] = []
    for row in rows:
        canonical = (
            (row.get("normalized_full_address") or "").strip()
            or (row.get("source_raw_address") or "").strip()
            or _compose_structured_address(dict(row)).strip()
        )
        if canonical:
            out.append((int(row["address_id"]), canonical))

    return out


def fetch_structured_rows(engine: Engine) -> List[Dict[str, str]]:
    query = text(
        """
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
        ORDER BY address_id
        """
    )

    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()

    return [dict(row) for row in rows]
