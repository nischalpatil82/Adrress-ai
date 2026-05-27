"""
fuzzy_engine.v2.fetch_pincodes
==============================
Downloads an open India Post pincode dataset and writes it to
    data/india_post_pincodes.csv
in the schema expected by `fuzzy_engine.v2.verify.PincodeIndex`:
    pincode,office,district,state

Source: a community-maintained mirror of the data.gov.in "All India Pincode
Directory". We try a few mirrors so that one going down does not break setup.

Run:
    python -m fuzzy_engine.v2.fetch_pincodes
"""
from __future__ import annotations

import csv
import io
import json
import logging
import sys
from pathlib import Path

import requests

from fuzzy_engine.v2.config import PINCODE_CSV_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("v2.fetch_pincodes")

# Stable open mirrors of the India Post pincode dataset.
# Each entry: (url, kind) where kind in {"csv", "json"}
SOURCES: list[tuple[str, str]] = [
    # kishorek/India-Codes: ~155k offices, columns:
    # PostOfficeName,Pincode,DistrictsName,City,State
    (
        "https://raw.githubusercontent.com/kishorek/India-Codes/master/csv/pincodes.csv",
        "csv",
    ),
    # dropdevrahul/pincodes-india: alternative aggregator
    (
        "https://raw.githubusercontent.com/dropdevrahul/pincodes-india/main/data/pincodes.csv",
        "csv",
    ),
    # saravanakumargn/All-India-Pincode-Directory
    (
        "https://raw.githubusercontent.com/saravanakumargn/All-India-Pincode-Directory/master/all_india_PO_list_without_APS_offices_ver2_lat_long.csv",
        "csv",
    ),
]


def _fetch(url: str, timeout: int = 60) -> bytes:
    log.info("GET %s", url)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content


def _normalise_csv(blob: bytes) -> list[dict]:
    """Read any CSV variant and remap columns to (pincode, office, district, state)."""
    text = blob.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    rows: list[dict] = []
    for row in reader:
        # Lower-case keys for matching
        keys = {k.lower().strip(): k for k in row.keys() if k}
        pin_key = next(
            (keys[k] for k in ("pincode", "pin", "pin_code", "postal_code")
             if k in keys),
            None,
        )
        office_key = next(
            (keys[k] for k in (
                "officename", "postofficename", "office",
                "office_name", "post_office",
            ) if k in keys),
            None,
        )
        district_key = next(
            (keys[k] for k in (
                "districtname", "districtsname", "district", "district_name"
            ) if k in keys),
            None,
        )
        state_key = next(
            (keys[k] for k in ("statename", "state", "state_name")
             if k in keys),
            None,
        )
        if not pin_key:
            continue
        pin = (row.get(pin_key) or "").strip()
        if not pin or not pin.isdigit() or len(pin) != 6:
            continue
        rows.append({
            "pincode": pin,
            "office": (row.get(office_key) or "").strip() if office_key else "",
            "district": (row.get(district_key) or "").strip() if district_key else "",
            "state": (row.get(state_key) or "").strip() if state_key else "",
        })
    return rows


def _normalise_json(blob: bytes) -> list[dict]:
    data = json.loads(blob.decode("utf-8", errors="replace"))
    rows: list[dict] = []
    for item in data:
        pin = str(item.get("pincode") or item.get("Pincode") or "").strip()
        if not pin or not pin.isdigit() or len(pin) != 6:
            continue
        rows.append({
            "pincode": pin,
            "office": str(item.get("office") or item.get("Office") or "").strip(),
            "district": str(item.get("district") or item.get("District") or "").strip(),
            "state": str(item.get("state") or item.get("State") or "").strip(),
        })
    return rows


def _write(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["pincode", "office", "district", "state"]
        )
        writer.writeheader()
        writer.writerows(rows)


def fetch(out_path: Path = PINCODE_CSV_PATH) -> int:
    last_err: Exception | None = None
    for url, kind in SOURCES:
        try:
            blob = _fetch(url)
            if kind.startswith("csv"):
                rows = _normalise_csv(blob)
            elif kind.startswith("json"):
                rows = _normalise_json(blob)
            else:
                continue
            if not rows:
                log.warning("Source %s returned 0 valid rows; trying next.", url)
                continue
            _write(rows, out_path)
            log.info("Saved %d rows to %s", len(rows), out_path)
            return len(rows)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            log.warning("Source failed (%s): %s", url, exc)
    raise RuntimeError(
        f"All pincode sources failed. Last error: {last_err}. "
        f"Manually download from data.gov.in and place CSV at {out_path}."
    )


def main() -> None:
    n = fetch()
    print(f"Done. Wrote {n} rows to {PINCODE_CSV_PATH}")


if __name__ == "__main__":
    sys.exit(main())
