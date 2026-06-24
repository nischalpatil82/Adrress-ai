"""
8_api.py
Flask REST API for the high-accuracy Address AI pipeline.
Version: 2024-06-04-v2 - Fixed reranker scoring with query specificity

Default mode loads 5_full_pipeline_sql.py (T5 + BM25 + FAISS + reranker).
Use --legacy or --csv to run the older fuzzy_engine.AddressCorrector path.

Run:
    python 8_api.py
    python 8_api.py --legacy
    python 8_api.py --csv
"""

from __future__ import annotations

import csv
import io
import importlib.util
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from session_store import create_session, get_session, update_session, delete_session

# Prevent HuggingFace tokenizers from deadlocking in Flask worker threads
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Prefer local model files. If Hugging Face's HTTP client is in a closed state,
# dense retrieval will be disabled instead of taking the whole API down.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# Keep interactive API calls responsive when an external geocoder is blocked
# or unavailable. Users can override this in .env with V2_GEOCODE_TIMEOUT.
os.environ.setdefault("V2_GEOCODE_TIMEOUT", "3")

from flask import Flask, Response, jsonify, redirect, render_template, request


ROOT = Path(__file__).resolve().parent
PIPELINE_PATH = ROOT / "5_full_pipeline_sql.py"
FEEDBACK_PATH = ROOT / "data" / "active_learning_feedback.jsonl"
# Persisted bulk-cleanse jobs (history + re-download). One folder per job:
#   <job_id>/meta.json  +  <job_id>/output.csv
CLEANSE_HISTORY_DIR = ROOT / "data" / "cleanse_history"

app = Flask(__name__)
# Re-read templates from disk when they change so UI edits show up on a plain
# browser refresh (no server restart needed). The mtime check per render is
# negligible and avoids wiping in-memory cleanse sessions on every tweak.
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True
engine_mode = "pipeline_sql"
pipeline_mod = None
pipeline_models = None
legacy_corrector = None
startup_error = None
v2_pipeline = None
v2_error: str | None = None


def _maybe_load_v2(force_reload: bool = False) -> None:
    """Lazy-load the v2 pipeline. Failures are non-fatal."""
    global v2_pipeline, v2_error
    if force_reload:
        v2_pipeline = None
        v2_error = None
    if v2_pipeline is not None or v2_error is not None:
        return
    try:
        from fuzzy_engine.v2 import AddressPipeline
        use_t5 = os.getenv("V2_USE_T5", "1").lower() not in (
            "0", "false", "no", "off",
        )
        use_geocoder = os.getenv("V2_USE_GEOCODER", "1").lower() not in (
            "0", "false", "no", "off",
        )
        v2_pipeline = AddressPipeline.from_config(
            use_t5=use_t5,
            use_geocoder=use_geocoder,
        )
        print("  [+] v2 pipeline loaded.")
    except Exception as exc:  # noqa: BLE001
        import traceback
        if _looks_like_closed_client_error(exc) and "AddressPipeline" in locals():
            try:
                print("  [!] v2 load hit a closed client; retrying without geocoder.")
                v2_pipeline = AddressPipeline.from_config(
                    use_t5=use_t5,
                    use_geocoder=False,
                )
                v2_error = None
                print("  [+] v2 pipeline loaded without geocoder.")
                return
            except Exception as fallback_exc:  # noqa: BLE001
                if _looks_like_closed_client_error(fallback_exc):
                    try:
                        print("  [!] v2 fallback still hit a closed client; retrying local-only.")
                        v2_pipeline = AddressPipeline.from_config(
                            use_t5=False,
                            use_geocoder=False,
                        )
                        v2_error = None
                        print("  [+] v2 pipeline loaded local-only.")
                        return
                    except Exception as local_exc:  # noqa: BLE001
                        exc = local_exc
                else:
                    exc = fallback_exc
        v2_error = str(exc)
        print(f"  [!] v2 pipeline failed to load: {exc}")
        traceback.print_exc()
        # Write full traceback to file for debugging
        try:
            with open("v2_pipeline_error.log", "w", encoding="utf-8") as f:
                f.write(f"v2 pipeline load error: {exc}\n")
                f.write(traceback.format_exc())
        except Exception:
            pass


def _looks_like_closed_client_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "client has been closed" in msg
        or "client is closed" in msg
        or "closed client" in msg
        or "cannot send a request" in msg
    )


def _v2_correct_with_recovery(query: str, top_n: int = 3):
    """Run v2 correction, rebuilding the pipeline once for closed-client errors."""
    if not _ensure_v2_available():
        raise RuntimeError(f"v2 pipeline unavailable: {v2_error}")
    try:
        return v2_pipeline.correct(query, top_n=top_n)
    except Exception as exc:  # noqa: BLE001
        if not _looks_like_closed_client_error(exc):
            raise
        print(f"  [!] v2 pipeline client closed; reloading and retrying once: {exc}")
        _maybe_load_v2(force_reload=True)
        if v2_pipeline is None:
            raise RuntimeError(f"v2 pipeline unavailable after reload: {v2_error}") from exc
        return v2_pipeline.correct(query, top_n=top_n)


def _ensure_v2_available() -> bool:
    """Ensure v2 is available, clearing stale closed-client errors once."""
    _maybe_load_v2()
    if v2_pipeline is None and v2_error and _looks_like_closed_client_error(Exception(v2_error)):
        _maybe_load_v2(force_reload=True)
    return v2_pipeline is not None


def _v2_unavailable_payload(raw_query: str, top_n: int = 5) -> dict:
    return {
        "query": raw_query,
        "status": "pipeline_unavailable",
        "confidence": 0.0,
        "best_address": None,
        "best_addr_id": None,
        "structured": {},
        "spell": {"applied": False, "corrected": raw_query, "changes": [], "used_t5": False},
        "parsed": {},
        "verification": {"geocoded": False, "notes": ["v2_pipeline_unavailable"]},
        "suggestions": [],
        "notes": [f"v2_pipeline_unavailable: {v2_error}"],
        "top_n": top_n,
    }


def _load_pipeline_module():
    spec = importlib.util.spec_from_file_location("full_pipeline_sql", PIPELINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load pipeline module: {PIPELINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_legacy_corrector(use_csv: bool):
    from fuzzy_engine import AddressCorrector

    if use_csv:
        return AddressCorrector("data/realistic_addresses.csv")

    try:
        return AddressCorrector.from_database()
    except Exception as exc:
        print(f"  [!] MySQL connection failed: {exc}")
        print("  [!] Falling back to CSV legacy engine.")
        return AddressCorrector("data/realistic_addresses.csv")


def load_engine() -> None:
    global engine_mode, pipeline_mod, pipeline_models, legacy_corrector, startup_error

    use_legacy = "--legacy" in sys.argv or "--csv" in sys.argv
    use_csv = "--csv" in sys.argv

    if use_legacy:
        engine_mode = "legacy_csv" if use_csv else "legacy_db"
        print(f"Loading legacy fuzzy engine ({engine_mode})...")
        legacy_corrector = _load_legacy_corrector(use_csv=use_csv)
        return

    print("Loading high-accuracy SQL pipeline...")
    try:
        pipeline_mod = _load_pipeline_module()
        pipeline_models = pipeline_mod.load_models()
        engine_mode = "pipeline_sql"
    except Exception as exc:
        startup_error = str(exc)
        print(f"  [!] High-accuracy pipeline failed to load: {exc}")
        print("  [!] Falling back to legacy database/CSV engine.")
        engine_mode = "legacy_fallback"
        legacy_corrector = _load_legacy_corrector(use_csv=False)


def _parse_top_n() -> int:
    raw_n = request.args.get("n", "5")
    return _coerce_top_n(raw_n)


def _coerce_top_n(raw_n, default: int = 5) -> int:
    try:
        return max(1, min(int(raw_n), 10))
    except (TypeError, ValueError):
        return default


def _pipeline_response(raw_query: str, top_n: int) -> dict:
    result = pipeline_mod.correct_address(raw_query, pipeline_models, top_n=top_n)
    top_matches = result.get("top_matches", [])
    return {
        "query": raw_query,
        "mode": engine_mode,
        "status": result.get("status"),
        "match": result.get("best_match") or result.get("corrected"),
        "corrected": result.get("corrected"),
        "corrected_input": result.get("corrected_input"),
        "confidence": result.get("confidence", 0.0),
        "spell_changes": result.get("spell_changes", []),
        "warnings": result.get("warnings", []),
        "scoring": result.get("scoring"),
        "sql_block_strategy": result.get("sql_block_strategy"),
        "sql_blocked_candidates": result.get("sql_blocked_candidates", 0),
        "sql_blocked_candidates_total": result.get("sql_blocked_candidates_total", 0),
        "suggestions": [
            {
                "address": item.get("full_address"),
                "score": item.get("score"),
                "db_id": item.get("db_id"),
                "structured": item.get("structured", {}),
            }
            for item in top_matches[:top_n]
        ],
    }


def _legacy_response(raw_query: str, top_n: int) -> dict:
    result = legacy_corrector.correct(raw_query, top_n=top_n)
    if result.get("error"):
        return {"error": result["error"]}

    if result.get("already_exists"):
        match = result.get("existing_address")
        confidence = result.get("existing_score", 0.0)
        suggestions = []
        status = "existing"
    else:
        match = result.get("corrected")
        confidence = result.get("confidence", 0.0)
        suggestions = [
            {"address": addr, "score": score}
            for addr, score in result.get("suggestions", [])
        ]
        status = result.get("status", "generated")

    return {
        "query": raw_query,
        "mode": engine_mode,
        "status": status,
        "match": match,
        "corrected": result.get("corrected"),
        "corrected_input": result.get("corrected_input"),
        "confidence": confidence,
        "spell_changes": result.get("spell_changes", []),
        "warnings": [],
        "scoring": "legacy_fuzzy",
        "sql_blocked_candidates": result.get("sql_blocked_candidates", 0),
        "sql_blocked_candidates_total": result.get("sql_blocked_candidates_total", 0),
        "suggestions": suggestions,
    }


@app.route("/", methods=["GET"])
def index():
    return render_template("v2.html")


@app.route("/legacy", methods=["GET"])
def legacy_index():
    return redirect("/")


@app.route("/v2", methods=["GET"])
def index_v2():
    return render_template("v2.html")


def _read_xlsx(raw_bytes: bytes):
    """Parse an .xlsx file into (headers, rows) matching csv.DictReader shape.

    Reads the first worksheet. The first non-empty row is treated as the
    header. Cell values are coerced to strings (empty string for blanks).
    """
    import openpyxl  # lazy import; only needed for Excel uploads

    wb = openpyxl.load_workbook(
        io.BytesIO(raw_bytes), read_only=True, data_only=True,
    )
    ws = wb.active
    rows_iter = ws.iter_rows(values_only=True)

    headers: list[str] = []
    for first in rows_iter:
        if first is None:
            continue
        headers = [
            (str(h).strip() if h is not None else f"column_{i + 1}")
            for i, h in enumerate(first)
        ]
        if any(h for h in headers):
            break

    rows: list[dict] = []
    for r in rows_iter:
        if r is None or all(v is None for v in r):
            continue
        row = {}
        for i, h in enumerate(headers):
            v = r[i] if i < len(r) else None
            row[h] = "" if v is None else str(v).strip()
        rows.append(row)

    wb.close()
    return headers, rows


_ADDRESS_KEYWORDS = [
    "address", "addr", "line", "area", "locality",
    "city", "town", "state", "province", "country", "pin", "zip", "postal",
]

_COLUMN_ROLE_KEYWORDS = {
    "address1": ("address line 1", "address 1", "addr 1", "line 1", "street", "address"),
    "address2": ("address line 2", "address 2", "addr 2", "line 2", "suite", "unit", "area", "locality"),
    "city": ("city", "town", "district"),
    "state": ("state", "province", "region"),
    "country": ("country", "nation"),
    "postal": ("zip code", "zipcode", "zip", "postal", "pincode", "pin code", "pin"),
}

_US_STATE_CODES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "IA",
    "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO",
    "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI",
    "WV", "WY", "DC",
}

_TRUSTED_STATUSES = {
    "verified", "high_confidence", "found_in_database", "generated",
}

# Max rows the model will process in one cleanse run. Default high enough to
# cover whole files; override with the env var for cost control. The previous
# default of 50 silently truncated large files.
_MODEL_ROW_LIMIT = int(os.getenv("ADDRESS_CLEANSE_MODEL_ROW_LIMIT", "5000"))

# Street-suffix + directional + unit abbreviations -> full words. Used to
# standardize geocoder output (which abbreviates: "St", "Rd", "Tpke", "Hwy")
# back into the expanded forms users expect from a cleanse tool.
_STREET_ABBREVIATIONS = {
    "st": "Street", "str": "Street", "rd": "Road", "ave": "Avenue",
    "av": "Avenue", "blvd": "Boulevard", "dr": "Drive", "ln": "Lane",
    "ct": "Court", "cir": "Circle", "pl": "Place", "sq": "Square",
    "ter": "Terrace", "trl": "Trail", "pkwy": "Parkway", "pky": "Parkway",
    "hwy": "Highway", "tpke": "Turnpike", "tpk": "Turnpike", "expy": "Expressway",
    "fwy": "Freeway", "cswy": "Causeway", "byp": "Bypass", "plz": "Plaza",
    "pt": "Point", "spg": "Spring", "spgs": "Springs", "mt": "Mount",
    "mtn": "Mountain", "jct": "Junction", "xing": "Crossing",
    "n": "North", "s": "South", "e": "East", "w": "West",
    "ne": "Northeast", "nw": "Northwest", "se": "Southeast", "sw": "Southwest",
    "apt": "Apartment", "ste": "Suite", "fl": "Floor", "bldg": "Building",
    "rm": "Room", "dept": "Department",
}

# US state two-letter codes -> full names (CT -> Connecticut).
_US_STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "IA": "Iowa",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "MA": "Massachusetts", "MD": "Maryland",
    "ME": "Maine", "MI": "Michigan", "MN": "Minnesota", "MO": "Missouri",
    "MS": "Mississippi", "MT": "Montana", "NC": "North Carolina",
    "ND": "North Dakota", "NE": "Nebraska", "NH": "New Hampshire",
    "NJ": "New Jersey", "NM": "New Mexico", "NV": "Nevada", "NY": "New York",
    "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VA": "Virginia",
    "VT": "Vermont", "WA": "Washington", "WI": "Wisconsin",
    "WV": "West Virginia", "WY": "Wyoming", "DC": "District of Columbia",
}


def _expand_abbreviations(text: str) -> str:
    """Expand street-suffix/directional/unit abbreviations to full words.

    Token-wise, case-insensitive, word-boundary safe. Preserves numbers,
    ZIPs and unknown tokens unchanged. "2233 NY-86" keeps "NY-86" (hyphenated
    route codes are not split). Title-cases expanded words.
    """
    if not text:
        return text

    def _fix_token(tok: str) -> str:
        # Keep tokens that contain digits (house numbers, ZIPs, "NY-86") as-is.
        if any(ch.isdigit() for ch in tok):
            return tok
        # Split trailing punctuation (commas) so "Rd," still expands.
        m = re.match(r"^([A-Za-z]+)([^A-Za-z]*)$", tok)
        if not m:
            return tok
        word, trail = m.group(1), m.group(2)
        repl = _STREET_ABBREVIATIONS.get(word.lower())
        return (repl + trail) if repl else tok

    return " ".join(_fix_token(t) for t in text.split())


def _expand_state(value: str) -> str:
    """Expand a US state code to its full name (CT -> Connecticut)."""
    v = (value or "").strip()
    return _US_STATE_NAMES.get(v.upper(), value)


def _read_upload(uploaded) -> tuple[list[str], list[dict], bytes]:
    fname = (uploaded.filename or "").lower()
    raw_bytes = uploaded.stream.read()
    if fname.endswith(".xlsx"):
        headers, rows = _read_xlsx(raw_bytes)
    elif fname.endswith(".xls"):
        raise ValueError(
            "Legacy .xls files are not supported. Please save as .xlsx or .csv and re-upload."
        )
    else:
        stream = io.StringIO(raw_bytes.decode("utf-8-sig"))
        reader = csv.DictReader(stream)
        headers = list(reader.fieldnames or [])
        rows = list(reader)
    if not headers:
        raise ValueError("File has no header row")
    return headers, rows, raw_bytes


def _detect_address_columns(headers: list[str]) -> list[str]:
    return [
        h for h in headers
        if any(k in h.lower() for k in _ADDRESS_KEYWORDS)
    ]


def _suggest_column_roles(headers: list[str], rows: list[dict] | None = None) -> dict[str, str]:
    roles: dict[str, str] = {}
    lowered = {h: re.sub(r"[^a-z0-9]+", " ", h.lower()).strip() for h in headers}
    for role, keywords in _COLUMN_ROLE_KEYWORDS.items():
        for h, low in lowered.items():
            if h in roles.values():
                continue
            if any(k in low for k in keywords):
                roles[role] = h
                break
    if rows:
        postal_candidates = [
            h for h, low in lowered.items()
            if any(k in low for k in ("zip code", "zipcode", "zip", "postal", "pincode", "pin code", "pin"))
        ]
        if postal_candidates:
            roles["postal"] = max(
                postal_candidates,
                key=lambda h: _postal_column_score(rows, h),
            )
    return roles


def _postal_column_score(rows: list[dict], col: str) -> float:
    score = 0.0
    checked = 0
    for row in rows[:100]:
        raw = str(row.get(col) or "").strip()
        if not raw:
            continue
        checked += 1
        digits = re.sub(r"\D", "", raw)
        if re.fullmatch(r"\d{5}", raw):
            score += 4
        elif re.fullmatch(r"\d{5}-\d{4}", raw):
            score += 4
        elif len(digits) == 5:
            score += 3
        elif len(digits) == 9:
            score += 2
        elif len(digits) == 6:
            score += 2
        else:
            score -= 1
    return score / max(checked, 1)


def _filtered_address_columns(headers: list[str], roles: dict[str, str]) -> list[str]:
    cols = _detect_address_columns(headers)
    postal_cols = [
        h for h in cols
        if any(k in h.lower() for k in ("zip", "postal", "pin"))
    ]
    preferred_postal = roles.get("postal")
    if preferred_postal and len(postal_cols) > 1:
        cols = [
            h for h in cols
            if h not in postal_cols or h == preferred_postal
        ]
    return cols


def _infer_country(rows: list[dict], roles: dict[str, str]) -> str:
    country_col = roles.get("country")
    if country_col:
        values = [
            str(row.get(country_col) or "").strip().lower()
            for row in rows[:50]
            if str(row.get(country_col) or "").strip()
        ]
        if values:
            joined = " ".join(values)
            if "india" in joined or joined.count("in") > max(2, len(values) // 3):
                return "India"
            if "usa" in joined or "united states" in joined or "us" in joined.split():
                return "United States"

    state_col = roles.get("state")
    postal_col = roles.get("postal")
    state_hits = 0
    zip_hits = 0
    checked = 0
    for row in rows[:50]:
        state = str(row.get(state_col) or "").strip().upper() if state_col else ""
        postal = re.sub(r"\D", "", str(row.get(postal_col) or "")) if postal_col else ""
        if state or postal:
            checked += 1
        if state in _US_STATE_CODES:
            state_hits += 1
        if len(postal) in (5, 9):
            zip_hits += 1
    if checked and (state_hits / checked >= 0.4 or zip_hits / checked >= 0.6):
        return "United States"
    return ""


def _build_query(row: dict, address_cols: list[str], country_hint: str = "") -> str:
    parts = [str(row.get(c) or "").strip() for c in address_cols]
    query = ", ".join(p for p in parts if p)
    if country_hint and country_hint.lower() not in query.lower():
        query = f"{query}, {country_hint}" if query else country_hint
    return query


def _quality_issues(row: dict, roles: dict[str, str]) -> list[str]:
    issues: list[str] = []
    postal_col = roles.get("postal")
    state_col = roles.get("state")
    address1_col = roles.get("address1")
    city_col = roles.get("city")

    if address1_col and not str(row.get(address1_col) or "").strip():
        issues.append("missing_address_line_1")
    if city_col and not str(row.get(city_col) or "").strip():
        issues.append("missing_city")
    if state_col and not str(row.get(state_col) or "").strip():
        issues.append("missing_state")
    if postal_col:
        raw = str(row.get(postal_col) or "").strip()
        digits = re.sub(r"\D", "", raw)
        if not raw:
            issues.append("missing_postal_code")
        elif len(digits) not in (5, 6, 9):
            issues.append("postal_code_unusual_length")
        elif len(digits) == 9 and raw == digits:
            issues.append("zip_plus4_stored_without_dash")
    return issues


def _safe_text(value) -> str:
    return "" if value is None else str(value).strip()


def _compare_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def _values_equivalent(before: str, after: str) -> bool:
    return _compare_text(before) == _compare_text(after)


def _address_similarity(before: str, after: str) -> float:
    before = before or ""
    after = after or ""
    if not before or not after:
        return 0.0
    try:
        from rapidfuzz import fuzz
        return float(fuzz.token_set_ratio(before, after)) / 100.0
    except Exception:
        b = set(_compare_text(before).split())
        a = set(_compare_text(after).split())
        return len(a & b) / max(len(a | b), 1)


def _title_if_caps(text: str) -> str:
    """Title-case ALL-CAPS / all-lowercase alpha tokens, preserve the rest.

    "ONE HOSPITAL PLAZA" -> "One Hospital Plaza", but mixed/identifier tokens
    like "NY-86" or "McDonald" are left untouched.
    """
    out = []
    for tok in (text or "").split():
        if tok.isalpha() and (tok.isupper() or tok.islower()):
            out.append(tok.capitalize())
        else:
            out.append(tok)
    return " ".join(out)


def _fuzzy_token_present(word: str, candidates: set[str]) -> bool:
    try:
        from rapidfuzz import fuzz
        return any(fuzz.ratio(word, c) >= 80 for c in candidates)
    except Exception:
        return False


def _loses_information(old: str, new: str) -> bool:
    """True when `new` drops content that `old` carried — i.e. a destructive
    "correction". Guards against the geocoder collapsing
    "2233 STATE ROUTE 86" down to "2233".

    A change loses information when it drops a number the original had, or when
    a majority of the original's words vanish (fuzzy-matched so genuine typo
    fixes like STEET->STREET are NOT treated as loss).
    """
    old = old or ""
    new = new or ""
    if not old.strip():
        return False  # nothing to lose
    if not re.search(r"[A-Za-z]{2,}", old):
        # No real alphabetic content (e.g. "1", "0", "-", "."): a junk placeholder
        # with nothing worth preserving, so replacing it with a verified value is
        # never a loss. This lets a garbage city like "1" become "Bengaluru".
        return False
    old_nums = set(re.findall(r"\d+", old))
    new_nums = set(re.findall(r"\d+", new))
    if old_nums - new_nums:
        return True
    old_words = [w for w in re.findall(r"[a-zA-Z]+", old.lower()) if len(w) >= 3]
    if not old_words:
        return False
    new_words = set(re.findall(r"[a-zA-Z]+", new.lower()))
    missing = 0
    for w in old_words:
        if w in new_words or _fuzzy_token_present(w, new_words):
            continue
        missing += 1
    return missing > len(old_words) // 2


def _normalize_postal(structured_pin, original: str, country: str = "") -> str:
    """Prefer the geocoder-verified postal code; otherwise clean the original.

    Pads US ZIPs to 5 digits, formats ZIP+4, keeps 6-digit Indian PINs, and
    never emits the corrupted concatenated form (e.g. "060522016").
    """
    pin = _safe_text(structured_pin)
    if pin:
        return pin  # geocoder-verified, already canonical
    digits = re.sub(r"\D", "", original or "")
    if not digits:
        return _safe_text(original)
    c = (country or "").lower()
    if "india" in c or "in" == c or len(digits) == 6:
        return digits[:6]
    if len(digits) >= 9:
        return f"{digits[:5]}-{digits[5:9]}"
    return digits[:5].zfill(5)


def _corrected_field(role: str, structured: dict, original: str,
                     country_hint: str = "") -> str:
    """Produce the corrected value for a single column, by its role.

    Geocoder-derived values are preferred, but only when they don't destroy
    information present in the original (see `_loses_information`). When the
    geocoder value is missing or destructive, the original is kept but still
    normalized (case + abbreviation/state expansion).
    """
    s = structured or {}
    original = _safe_text(original)
    country = _safe_text(s.get("country")) or country_hint

    if role == "postal":
        return _normalize_postal(s.get("pincode"), original, country)

    if role == "country":
        return _title_if_caps(_safe_text(s.get("country")) or original)

    if role == "state":
        return _title_if_caps(_expand_state(_safe_text(s.get("state")) or original))

    if role == "city":
        geo = _safe_text(s.get("city"))
        cand = _title_if_caps(geo) if geo else ""
        if cand and not _loses_information(original, cand):
            # Guard against the geocoder prepending street/landmark tokens to a
            # clean city (e.g. "COLUMBIA" -> "Marion Columbia"). If the original
            # is already fully contained in a longer candidate, keep the original.
            if original and len(cand.split()) > len(original.split()):
                o_words = set(re.findall(r"[a-z]+", original.lower()))
                c_words = set(re.findall(r"[a-z]+", cand.lower()))
                if o_words and o_words <= c_words:
                    return _title_if_caps(original)
            return cand
        return _title_if_caps(original)

    if role == "address1":
        hn = _safe_text(s.get("house_number"))
        street = _safe_text(s.get("street"))
        if street:
            cand = _expand_abbreviations(
                _title_if_caps((hn + " " + street).strip()))
            if cand and not _loses_information(original, cand):
                return cand
        return _expand_abbreviations(_title_if_caps(original))

    if role == "address2":
        sub = _safe_text(s.get("sublocality"))
        cand = _expand_abbreviations(_title_if_caps(sub)) if sub else ""
        if cand and not _loses_information(original, cand):
            return cand
        return _expand_abbreviations(_title_if_caps(original))

    return original


def _full_verified_address(structured: dict) -> str:
    return (
        _safe_text(structured.get("formatted"))
        or f"{structured.get('house_number') or ''} {structured.get('street') or ''}, "
           f"{structured.get('sublocality') or ''}, {structured.get('city') or ''}, "
           f"{structured.get('state') or ''}, {structured.get('country') or ''}, "
           f"{structured.get('pincode') or ''}".strip(", ")
    ).strip()


def _is_trusted_result(result, min_confidence: float) -> tuple[bool, list[str]]:
    if not result:
        return False, ["pipeline_returned_no_result"]
    status = getattr(result, "status", "no_match")
    confidence = float(getattr(result, "confidence", 0.0) or 0.0)
    structured = getattr(result, "structured", {}) or {}
    full = _full_verified_address(structured).strip().lower()
    issues: list[str] = []
    if status not in _TRUSTED_STATUSES:
        issues.append(f"status_{status}")
    if confidence < min_confidence:
        issues.append("below_approval_threshold")
    if full in {"", "india", "united states", "usa"}:
        issues.append("generic_geocode_result")
    return not issues, issues


def _norm_country(value: str) -> str:
    """Collapse country spelling variants so US != United States isn't a false
    mismatch."""
    v = _compare_text(value)
    if v in ("us", "usa", "united states", "united states of america", "america", "u s a"):
        return "us"
    if v in ("in", "ind", "india", "bharat"):
        return "in"
    if v in ("uk", "gb", "united kingdom", "great britain", "england"):
        return "uk"
    return v


def _detect_anomalies(row: dict, structured: dict, roles: dict,
                      country_hint: str, similarity: float) -> list[str]:
    """High-signal flags that a geocode match is probably WRONG (as opposed to
    a normal correction). These rank the review queue: a row can look confident
    yet still be flagged here ("confident but suspicious")."""
    s = structured or {}
    anomalies: list[str] = []

    geocoded = s.get("lat") is not None and s.get("lon") is not None
    if not geocoded:
        anomalies.append("not_geocoded")
    else:
        prec = s.get("precision")
        if prec is not None and float(prec) <= 0.45:
            anomalies.append("approximate_location")

    # Country flipped (e.g. expected US but resolved India).
    res_country = _safe_text(s.get("country"))
    if country_hint and res_country and _norm_country(res_country) != _norm_country(country_hint):
        anomalies.append("country_mismatch")

    # Resolved city is a DIFFERENT place than the user typed (not just casing).
    # Compare case/punctuation-normalized text — rapidfuzz is case-sensitive, so
    # "STAMFORD" vs "Stamford" would otherwise look like a mismatch.
    city_col = roles.get("city")
    in_city = _compare_text(_safe_text(row.get(city_col))) if city_col else ""
    res_city = _compare_text(_safe_text(s.get("city")))
    if in_city and res_city and in_city != res_city and _address_similarity(in_city, res_city) < 0.6:
        anomalies.append("city_mismatch")

    # Resolved state differs from the user's (after expanding CT->Connecticut).
    state_col = roles.get("state")
    in_state = _expand_state(_safe_text(row.get(state_col))) if state_col else ""
    res_state = _expand_state(_safe_text(s.get("state")))
    if in_state and res_state and _compare_text(in_state) != _compare_text(res_state):
        anomalies.append("state_mismatch")

    return anomalies


def _summarize_result(row: dict, result, roles: dict[str, str],
                      address_cols: list[str], min_confidence: float,
                      query: str = "") -> dict:
    structured = getattr(result, "structured", {}) if result else {}
    trusted, trust_issues = _is_trusted_result(result, min_confidence)
    # Reverse the role map (role -> column) into column -> role so each column
    # is corrected according to what it actually holds.
    col_to_role = {col: role for role, col in (roles or {}).items()}
    country_hint = _safe_text(structured.get("country"))
    field_changes = []
    input_fields = []   # column-wise "before" view of every selected column
    for col in address_cols:
        old_val = _safe_text(row.get(col))
        input_fields.append({"column": col, "value": old_val})
        role = col_to_role.get(col)
        if not role:
            # Unknown column (e.g. "Organization Name"): never rewrite its
            # meaning, but still normalize ALL-CAPS / all-lowercase casing so
            # "STAMFORD HOSPITAL" -> "Stamford Hospital".
            cased = _title_if_caps(old_val)
            if cased and cased.strip() != old_val.strip():
                field_changes.append({
                    "column": col, "from": old_val, "to": cased, "role": "casing",
                })
            continue
        # _corrected_field already guards the destructive text roles
        # (address1/address2/city) internally. postal/state/country are intended
        # to change value (e.g. ZIP 08046 -> 06902, CT -> Connecticut) and must
        # NOT be blocked by a number/word-loss check.
        new_val = _corrected_field(role, structured, old_val, country_hint)
        if new_val and new_val.strip() != old_val.strip():
            field_changes.append({
                "column": col, "from": old_val, "to": new_val, "role": role,
            })
    issues = _quality_issues(row, roles) + trust_issues
    full_address = _full_verified_address(structured)
    similarity = _address_similarity(query, full_address)
    confidence = float(getattr(result, "confidence", 0.0) or 0.0)

    # Graded per-address confidence. For geocode-derived results the pipeline
    # assigns a flat floor (e.g. 0.85 for any street-level match), so every row
    # looks identical. Replace it with a score that actually varies, blending:
    #   - Google's match precision (rooftop vs interpolated vs approximate)
    #   - input similarity (does the resolved address match what was typed?)
    #   - field completeness (street/city/state/postal/country resolved?)
    geocoded = structured.get("lat") is not None and structured.get("lon") is not None
    if geocoded and "precision" in structured:
        # Graded per-address confidence from the two TRUSTWORTHY signals:
        #   - geocode precision (rooftop > street-interpolated > area > approx)
        #   - field completeness (street/city/state/postal/country resolved?)
        # (Input similarity is deliberately NOT used — it's unreliable here:
        # the org-name column isn't in Google's formatted address, so it reads
        # low for correct rows and high for junk.)
        precision = float(structured.get("precision") or 0.0)
        # Remap so a clean area-level match reads ~0.83 (a good cleanse, just
        # not a rooftop pin) instead of being punished as if it were wrong.
        precision_score = {1.0: 0.97, 0.85: 0.90, 0.65: 0.83, 0.45: 0.60}.get(
            round(precision, 2), 0.55 + 0.42 * precision)
        key_fields = ("street", "city", "state", "pincode", "country")
        present = sum(1 for k in key_fields if _safe_text(structured.get(k)))
        completeness = present / len(key_fields)
        confidence = max(0.0, min(1.0, precision_score * (0.85 + 0.15 * completeness)))

    if full_address and query and similarity < 0.72:
        trusted = False
        issues.append("suggestion_differs_from_input")
    anomalies = _detect_anomalies(row, structured, roles, country_hint, similarity) if result else []
    if trusted and not field_changes:
        action = "already_clean"
    elif trusted and field_changes:
        action = "suggested_fix"
    else:
        action = "review_required"
    return {
        "status": getattr(result, "status", "no_match") if result else "no_match",
        "confidence": round(confidence, 4),
        "accuracy_percent": round(confidence * 100, 1),
        "input_similarity_percent": round(similarity * 100, 1),
        "trusted": trusted,
        "action": action,
        "issues": issues,
        "anomalies": anomalies,
        "field_changes": field_changes,
        "input_fields": input_fields,
        "changed_columns": [c["column"] for c in field_changes],
        "full_verified_address": full_address,
    }


# Bucket order is meaningful (used for filter chips / dashboard).
_BUCKETS = ("auto_fixable", "needs_review", "already_clean", "no_match")


def _bucket_of(summary: dict, threshold: float) -> str:
    """Categorize a processed row into one enterprise review bucket.

    Confidence-first so the user-adjustable threshold actually drives the
    buckets (a high-confidence row with a proposed fix is auto-fixable even if
    a soft guard like `suggestion_differs_from_input` downgraded its action —
    that issue still surfaces in the Issues column for transparency):
      - no_match     : engine produced nothing usable (confidence 0, no fix)
      - already_clean: trusted and nothing to change
      - auto_fixable : has corrections AND confidence >= threshold
      - needs_review : everything else (low-confidence, or flagged-only)
    """
    # Work in percent so it matches the lean record (which stores
    # accuracy_percent, not the 0..1 confidence). threshold is a 0..1 fraction.
    conf_pct = float(summary.get("accuracy_percent", 0.0) or 0.0)
    thresh_pct = threshold * 100.0
    action = summary.get("action")
    has_changes = bool(summary.get("field_changes"))
    if conf_pct <= 0 and not has_changes:
        return "no_match"
    if action == "already_clean":
        return "already_clean"
    if has_changes and conf_pct >= thresh_pct:
        return "auto_fixable"
    return "needs_review"


def _lean_record(summary: dict) -> dict:
    """Trim a per-row summary to what the review UI needs (keeps sessions small
    when processing hundreds of thousands of rows)."""
    return {
        "row_number": summary["row_number"],
        "accuracy_percent": summary["accuracy_percent"],
        "action": summary["action"],
        "issues": summary["issues"],
        "anomalies": summary.get("anomalies", []),
        "field_changes": summary["field_changes"],
        "input_fields": summary["input_fields"],
        "changed_columns": summary["changed_columns"],
    }


def _process_cleanse_row(row: dict, idx: int, roles: dict, address_cols: list,
                         country_hint: str, min_confidence: float,
                         cache: dict, model_budget: list | None = None):
    """Single source of truth for cleansing one row.

    Builds the query, runs the model (with a query-keyed cache so identical
    addresses and repeat calls across preview/final are computed once), and
    returns (summary, result). `model_budget` is a 1-element mutable list used
    as a shared counter to cap model runs; pass None for no cap.
    """
    query = _build_query(row, address_cols, country_hint)
    result = None
    if query and _ensure_v2_available():
        key = _compare_text(query)
        if key in cache:
            result = cache[key]
        elif model_budget is not None and model_budget[0] >= _MODEL_ROW_LIMIT:
            result = None
        else:
            try:
                result = _v2_correct_with_recovery(query, top_n=3)
            except Exception:
                result = None
            cache[key] = result
            if model_budget is not None:
                model_budget[0] += 1
    summary = _summarize_result(row, result, roles, address_cols, min_confidence, query)
    summary["row_number"] = idx
    summary["query"] = query
    return summary, result


def _cleanse_rows(rows: list[dict], headers: list[str], address_cols: list[str],
                  roles: dict[str, str], min_confidence: float,
                  country_hint: str = "",
                  apply_mode: str = "review_only") -> tuple[list[dict], dict]:
    needs_model = apply_mode == "auto_high_confidence"
    country_supported = (country_hint or "").strip().lower() in {"", "india"}
    pipeline_available = _ensure_v2_available() if needs_model and country_supported else False
    result_cache = {}
    model_runs = 0

    output_rows = []
    stats = {
        "processed": 0,
        "auto_corrected": 0,
        "review_required": 0,
        "unchanged": 0,
        "suggested": 0,
        "model_skipped": 0,
    }
    for idx, row in enumerate(rows, 1):
        query = _build_query(row, address_cols, country_hint)
        result = None
        if query and pipeline_available and model_runs < _MODEL_ROW_LIMIT:
            cache_key = _compare_text(query)
            if cache_key in result_cache:
                result = result_cache[cache_key]
            else:
                try:
                    result = _v2_correct_with_recovery(query, top_n=3)
                    model_runs += 1
                except Exception as exc:  # noqa: BLE001
                    result = None
                    row["_cleanse_exception"] = str(exc)
                result_cache[cache_key] = result
        elif query and needs_model:
            stats["model_skipped"] += 1

        summary = _summarize_result(row, result, roles, address_cols, min_confidence, query)
        structured = getattr(result, "structured", {}) if result else {}
        corrected_row = dict(row)

        should_apply = apply_mode == "auto_high_confidence" and summary["action"] == "suggested_fix"
        if should_apply:
            for change in summary["field_changes"]:
                corrected_row[change["column"]] = change["to"]
            stats["auto_corrected"] += 1
            output_action = "auto_corrected"
        elif summary["action"] == "suggested_fix":
            stats["suggested"] += 1
            output_action = "suggested_fix"
        elif query:
            stats["review_required"] += 1
            output_action = summary["action"]
        else:
            stats["unchanged"] += 1
            output_action = "unchanged"

        corrected_row["Cleanse Action"] = output_action
        corrected_row["Cleanse Status"] = summary["status"]
        corrected_row["Cleanse Confidence %"] = summary["accuracy_percent"]
        corrected_row["Input Similarity %"] = summary["input_similarity_percent"]
        issues = list(summary["issues"])
        if query and apply_mode == "review_only":
            issues.append("review_only_mode:model_not_run")
        elif query and needs_model and not country_supported:
            issues.append(f"country_not_supported_by_local_model:{country_hint}")
        elif query and needs_model and pipeline_available and result is None and model_runs >= _MODEL_ROW_LIMIT:
            issues.append(f"model_row_limit_reached:{_MODEL_ROW_LIMIT}")
        elif query and needs_model and not pipeline_available:
            issues.append(f"v2_pipeline_unavailable: {v2_error}")
        corrected_row["Cleanse Issues"] = "; ".join(issues)
        corrected_row["Cleanse Suggested Changes"] = json.dumps(
            summary["field_changes"], ensure_ascii=False,
        )
        corrected_row["Latitude"] = structured.get("lat") if structured.get("lat") is not None else ""
        corrected_row["Longitude"] = structured.get("lon") if structured.get("lon") is not None else ""
        corrected_row["Full Verified Address"] = summary["full_verified_address"]
        corrected_row["Original Address Query"] = query
        corrected_row["Row Number"] = idx
        output_rows.append(corrected_row)
        stats["processed"] += 1
    return output_rows, stats


@app.route("/v2/address_cleanse/preview", methods=["POST"])
def address_cleanse_preview_v2():
    uploaded = request.files.get("csv_file")
    if not uploaded or not uploaded.filename:
        return jsonify({"error": "No file provided"}), 400

    try:
        headers, rows, _raw = _read_upload(uploaded)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Failed to parse file: {exc}"}), 400

    original_count = len(rows)
    if len(rows) > 1000:
        rows = rows[:1000]

    roles = _suggest_column_roles(headers, rows)
    address_cols = _filtered_address_columns(headers, roles)
    country_hint = _infer_country(rows, roles)
    min_confidence = 0.80
    sample_rows = rows[:5]
    preview_results = []
    run_model_preview = len(rows) <= 100

    if sample_rows and address_cols:
        if run_model_preview and _ensure_v2_available():
            for idx, row in enumerate(sample_rows, 1):
                query = _build_query(row, address_cols, country_hint)
                result = None
                if query:
                    try:
                        result = _v2_correct_with_recovery(query, top_n=3)
                    except Exception:
                        result = None
                summary = _summarize_result(row, result, roles, address_cols, min_confidence, query)
                summary["row_number"] = idx
                summary["query"] = query
                preview_results.append(summary)
        else:
            for idx, row in enumerate(sample_rows, 1):
                query = _build_query(row, address_cols, country_hint)
                issues = _quality_issues(row, roles)
                preview_results.append({
                    "row_number": idx,
                    "query": query,
                    "status": "not_run",
                    "confidence": 0.0,
                    "accuracy_percent": 0.0,
                    "input_similarity_percent": 0.0,
                    "trusted": False,
                    "action": "ready_for_audit",
                    "issues": issues or ["large_file_fast_preview:model_not_run"],
                    "field_changes": [],
                    "full_verified_address": "",
                })

    return jsonify({
        "headers": headers,
        "row_count": original_count,
        "processed_limit": len(rows),
        "truncated": original_count > len(rows),
        "address_columns": address_cols,
        "roles": roles,
        "country_hint": country_hint,
        "min_confidence": min_confidence,
        "sample_rows": sample_rows,
        "preview_results": preview_results,
        "fast_preview": not run_model_preview,
        "pipeline_available": v2_pipeline is not None,
        "pipeline_error": v2_error,
    })


# ─── Interactive cleanse wizard endpoints ─────────────────────────────

@app.route("/v2/address_cleanse/analyze", methods=["POST"])
def address_cleanse_analyze_v2():
    """Step 1: Upload file, get column mapping + full data preview."""
    uploaded = request.files.get("csv_file")
    if not uploaded or not uploaded.filename:
        return jsonify({"error": "No file provided"}), 400

    try:
        headers, rows, raw_bytes = _read_upload(uploaded)
    except Exception as exc:
        return jsonify({"error": f"Failed to parse file: {exc}"}), 400

    original_count = len(rows)

    roles = _suggest_column_roles(headers, rows)
    address_cols = _filtered_address_columns(headers, roles)
    country_hint = _infer_country(rows, roles)

    # Create a session so subsequent steps can reference the full data. The full
    # row set is kept (no 1,000-row cap) so the wizard can process everything in
    # client-driven chunks.
    session_payload = {
        "headers": headers,
        "rows": rows,
        "roles": roles,
        "address_columns": address_cols,
        "country_hint": country_hint,
        "min_confidence": 0.80,
        "apply_mode": "review_only",
        "filename": uploaded.filename or "upload.csv",
    }
    sid = create_session(session_payload)

    return jsonify({
        "session_id": sid,
        "headers": headers,
        "row_count": original_count,
        "display_rows": rows[:5],
        "roles": roles,
        "address_columns": address_cols,
        "country_hint": country_hint,
    })


@app.route("/v2/address_cleanse/preview_rows", methods=["POST"])
def address_cleanse_preview_rows_v2():
    """Step 2: After column mapping, run AI preview on selected row range."""
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "")
    sess = get_session(session_id) if session_id else None

    if not sess:
        return jsonify({"error": "Session expired or invalid"}), 400

    rows = sess.get("rows", [])
    headers = sess.get("headers", [])
    roles = sess.get("roles", {})
    address_cols = data.get("address_columns", sess.get("address_columns", []))
    country_hint = data.get("country_hint", sess.get("country_hint", ""))
    start_row = max(0, int(data.get("start_row", 0)))
    end_row = min(len(rows), int(data.get("end_row", min(50, len(rows)))))
    min_confidence = max(0.0, min(1.0, float(data.get("min_confidence", 0.80))))

    # Persist the (possibly re-mapped) settings so `final` cleanses with the
    # exact same configuration the user reviewed.
    sess["address_columns"] = address_cols
    sess["country_hint"] = country_hint
    sess["min_confidence"] = min_confidence
    # Query-keyed result cache shared with `final` (lives in the session).
    cache = sess.setdefault("_result_cache", {})
    update_session(session_id, sess)

    preview_rows = rows[start_row:end_row]
    preview_results = []

    if preview_rows and address_cols and _ensure_v2_available():
        for idx, row in enumerate(preview_rows, start=start_row + 1):
            summary, _ = _process_cleanse_row(
                row, idx, roles, address_cols, country_hint, min_confidence, cache)
            preview_results.append(summary)
        update_session(session_id, sess)  # persist newly cached results
    else:
        for idx, row in enumerate(preview_rows, start=start_row + 1):
            query = _build_query(row, address_cols, country_hint)
            issues = _quality_issues(row, roles)
            preview_results.append({
                "row_number": idx,
                "query": query,
                "status": "not_run",
                "confidence": 0.0,
                "accuracy_percent": 0.0,
                "input_similarity_percent": 0.0,
                "trusted": False,
                "action": "ready_for_audit",
                "issues": issues or ["pipeline_unavailable"],
                "field_changes": [],
                "full_verified_address": "",
            })

    return jsonify({
        "preview_results": preview_results,
        "start_row": start_row,
        "end_row": end_row,
        "total_rows": len(rows),
        "pipeline_available": v2_pipeline is not None,
    })


@app.route("/v2/address_cleanse/process_chunk", methods=["POST"])
def address_cleanse_process_chunk_v2():
    """Enterprise scope: process one chunk of rows and store lean per-row
    results in the session. The client loops over chunks to drive a progress
    bar, so each request stays small and fast (no long-running web request, no
    worker-thread model deadlock)."""
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "")
    sess = get_session(session_id) if session_id else None
    if not sess:
        return jsonify({"error": "Session expired or invalid"}), 400

    rows = sess.get("rows", [])
    roles = sess.get("roles", {})
    total = len(rows)

    # On the first chunk the client sends the chosen scope + mapping; persist it.
    if "address_columns" in data:
        sess["address_columns"] = data.get("address_columns") or sess.get("address_columns", [])
    if "country_hint" in data:
        sess["country_hint"] = data.get("country_hint", sess.get("country_hint", ""))
    if "min_confidence" in data:
        sess["min_confidence"] = max(0.0, min(1.0, float(data.get("min_confidence", 0.80))))

    address_cols = sess.get("address_columns", [])
    country_hint = sess.get("country_hint", "")
    min_confidence = sess.get("min_confidence", 0.80)

    # Scope window for the whole job (defaults to all rows).
    scope_start = max(0, int(data.get("scope_start", 0)))
    scope_end = min(total, int(data.get("scope_end", total)))
    sess["scope_start"] = scope_start
    sess["scope_end"] = scope_end

    # This chunk.
    chunk_start = max(scope_start, int(data.get("chunk_start", scope_start)))
    chunk_size = max(1, int(data.get("chunk_size", 250)))
    chunk_end = min(scope_end, chunk_start + chunk_size)

    cache = sess.setdefault("_result_cache", {})
    all_results = sess.setdefault("_all_results", {})  # row_number(str) -> lean record

    if address_cols and _ensure_v2_available():
        # Process serially. Concurrency was measured to be SLOWER here and threw
        # SSL errors: the external geocoder/validation endpoints reject
        # concurrent TLS from this client (Windows). The real speedup comes from
        # the pooled keep-alive HTTP session in verify.py (~2.6x: 2.75s->1.06s
        # per row by avoiding a fresh SSL handshake on every call).
        for idx in range(chunk_start, chunk_end):
            try:
                summary, _ = _process_cleanse_row(
                    rows[idx], idx + 1, roles, address_cols, country_hint,
                    min_confidence, cache)
                all_results[str(idx + 1)] = _lean_record(summary)
            except Exception:  # noqa: BLE001
                all_results[str(idx + 1)] = {
                    "row_number": idx + 1, "accuracy_percent": 0.0,
                    "action": "no_match", "issues": ["processing_error"],
                    "field_changes": [],
                    "input_fields": [{"column": c, "value": _safe_text(rows[idx].get(c))} for c in address_cols],
                    "changed_columns": [],
                }
    else:
        for idx in range(chunk_start, chunk_end):
            query = _build_query(rows[idx], address_cols, country_hint)
            all_results[str(idx + 1)] = {
                "row_number": idx + 1,
                "accuracy_percent": 0.0,
                "action": "no_match",
                "issues": _quality_issues(rows[idx], roles) or ["pipeline_unavailable"],
                "field_changes": [],
                "input_fields": [{"column": c, "value": _safe_text(rows[idx].get(c))} for c in address_cols],
                "changed_columns": [],
            }

    update_session(session_id, sess)
    processed = len(all_results)
    scope_total = scope_end - scope_start
    return jsonify({
        "chunk_start": chunk_start,
        "chunk_end": chunk_end,
        "next_start": chunk_end,
        "done": chunk_end >= scope_end,
        "processed": processed,
        "scope_total": scope_total,
        "total_rows": total,
        "progress_percent": round((processed / scope_total) * 100, 1) if scope_total else 100.0,
    })


def _summary_for_threshold(all_results: dict, threshold: float) -> dict:
    """Bucket counts + confidence histogram + KPIs at a given threshold."""
    buckets = {b: 0 for b in _BUCKETS}
    hist = [0] * 10  # 0-9 -> 0-9%, ... 90-100%
    fields_changeable = 0
    anomaly_count = 0
    conf_sum = 0.0
    conf_n = 0
    for rec in all_results.values():
        b = _bucket_of(rec, threshold)
        buckets[b] += 1
        conf = float(rec.get("accuracy_percent", 0.0) or 0.0)
        bucket_idx = min(9, int(conf // 10))
        hist[bucket_idx] += 1
        if rec.get("field_changes"):
            fields_changeable += len(rec["field_changes"])
        if rec.get("anomalies"):
            anomaly_count += 1
        if conf > 0:
            conf_sum += conf
            conf_n += 1
    return {
        "buckets": buckets,
        "histogram": hist,
        "total": len(all_results),
        "fields_changeable": fields_changeable,
        "anomaly_count": anomaly_count,
        "avg_confidence": round(conf_sum / conf_n, 1) if conf_n else 0.0,
    }


@app.route("/v2/address_cleanse/review_summary", methods=["POST"])
def address_cleanse_review_summary_v2():
    """Dashboard data: bucket counts, histogram, KPIs for a threshold."""
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "")
    sess = get_session(session_id) if session_id else None
    if not sess:
        return jsonify({"error": "Session expired or invalid"}), 400
    all_results = sess.get("_all_results", {})
    threshold = max(0.0, min(1.0, float(data.get("threshold", sess.get("min_confidence", 0.80)))))
    payload = _summary_for_threshold(all_results, threshold)
    payload["threshold"] = threshold
    return jsonify(payload)


@app.route("/v2/address_cleanse/review_page", methods=["POST"])
def address_cleanse_review_page_v2():
    """Server-paginated, filtered slice of processed rows for the review table."""
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "")
    sess = get_session(session_id) if session_id else None
    if not sess:
        return jsonify({"error": "Session expired or invalid"}), 400
    all_results = sess.get("_all_results", {})
    threshold = max(0.0, min(1.0, float(data.get("threshold", sess.get("min_confidence", 0.80)))))
    bucket = data.get("bucket", "all")
    page = max(0, int(data.get("page", 0)))
    page_size = max(1, min(200, int(data.get("page_size", 50))))

    # Stable ordering by row number.
    records = sorted(all_results.values(), key=lambda r: r["row_number"])
    if bucket == "anomalies":
        # Orthogonal filter: any row carrying >=1 anomaly flag, regardless of
        # its confidence bucket (the valuable "confident but suspicious" rows).
        records = [r for r in records if r.get("anomalies")]
    elif bucket and bucket != "all":
        records = [r for r in records if _bucket_of(r, threshold) == bucket]
    total_matched = len(records)
    start = page * page_size
    page_rows = records[start:start + page_size]
    # Annotate each row with its bucket so the client can label without recomputing.
    for r in page_rows:
        r = r  # records are references; add bucket field
        r["bucket"] = _bucket_of(r, threshold)
    return jsonify({
        "rows": page_rows,
        "page": page,
        "page_size": page_size,
        "total_matched": total_matched,
        "total_pages": (total_matched + page_size - 1) // page_size,
    })


@app.route("/v2/address_cleanse/final", methods=["POST"])
def address_cleanse_final_v2():
    """Step 4: Apply approved corrections and generate final CSV."""
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "")
    sess = get_session(session_id) if session_id else None

    if not sess:
        return jsonify({"error": "Session expired or invalid"}), 400

    rows = sess.get("rows", [])
    headers = sess.get("headers", [])
    roles = sess.get("roles", {})
    address_cols = sess.get("address_columns", [])
    country_hint = sess.get("country_hint", "")
    min_confidence = sess.get("min_confidence", 0.80)
    approved_rows = set(data.get("approved_rows", []))
    rejected_rows = set(data.get("rejected_rows", []))
    # Auto-approve threshold (fraction): rows in the "auto_fixable" bucket
    # (trusted suggestion, confidence >= threshold) get applied automatically
    # unless the user explicitly rejected them.
    threshold = max(0.0, min(1.0, float(data.get("threshold", min_confidence))))

    # Which enrichment columns to append to the download. The client sends the
    # user's pick from the column-selector on the download step; if absent we
    # use a sensible default (Status / Original Query / Row Number off).
    requested_cols = data.get("include_columns")

    # Only rows that were actually processed (in scope) are enriched; everything
    # else passes through exactly as uploaded.
    all_results = sess.get("_all_results", {})
    cache = sess.setdefault("_result_cache", {})

    output_rows = []
    stats = {"processed": 0, "auto_corrected": 0, "unchanged": 0, "approved": 0, "rejected": 0, "passthrough": 0}

    # ── Enterprise statistics accumulators ──
    changes_by_column = Counter()
    changes_by_role = Counter()
    action_counts = Counter()
    issue_counts = Counter()
    anomaly_counts = Counter()
    rows_with_anomaly = 0
    fields_corrected = 0
    rows_corrected = 0
    geocoded_count = 0
    confidence_sum = 0.0
    confidence_n = 0

    def _count_applied(field_changes):
        nonlocal fields_corrected, rows_corrected
        if field_changes:
            rows_corrected += 1
        for change in field_changes:
            fields_corrected += 1
            changes_by_column[change["column"]] += 1
            if change.get("role"):
                changes_by_role[change["role"]] += 1

    for idx, row in enumerate(rows, 1):
        rec = all_results.get(str(idx))
        # Rows never processed (out of the chosen scope) pass through untouched —
        # no corrections, no enrichment columns.
        if rec is None:
            passthrough = dict(row)
            passthrough["Cleanse Action"] = "not_processed"
            output_rows.append(passthrough)
            stats["passthrough"] += 1
            continue

        # Re-run via the query cache (cache hit -> no model call) to recover the
        # structured geocode for enrichment columns.
        summary, result = _process_cleanse_row(
            row, idx, roles, address_cols, country_hint, min_confidence, cache)
        query = summary["query"]
        structured = getattr(result, "structured", {}) if result else {}
        corrected_row = dict(row)

        # Count by the SAME confidence-based bucket the review console and the
        # auto-apply use, so the statistics agree with what the user saw (the
        # legacy similarity-based `action` would mislabel most rows "review").
        action_counts[_bucket_of(summary, threshold)] += 1
        for iss in summary["issues"]:
            issue_counts[iss] += 1
        row_anoms = summary.get("anomalies", [])
        if row_anoms:
            rows_with_anomaly += 1
            for an in row_anoms:
                anomaly_counts[an] += 1
        if structured.get("lat") is not None and structured.get("lon") is not None:
            geocoded_count += 1
        if query and result is not None:
            confidence_sum += summary["accuracy_percent"]
            confidence_n += 1

        bucket = _bucket_of(summary, threshold)
        if idx in approved_rows:
            for change in summary["field_changes"]:
                corrected_row[change["column"]] = change["to"]
            corrected_row["Cleanse Action"] = "user_approved"
            stats["approved"] += 1
            _count_applied(summary["field_changes"])
        elif idx in rejected_rows:
            corrected_row["Cleanse Action"] = "user_rejected"
            stats["rejected"] += 1
        elif bucket == "auto_fixable":
            for change in summary["field_changes"]:
                corrected_row[change["column"]] = change["to"]
            corrected_row["Cleanse Action"] = "auto_corrected"
            stats["auto_corrected"] += 1
            _count_applied(summary["field_changes"])
        else:
            corrected_row["Cleanse Action"] = "needs_review_unchanged"
            stats["unchanged"] += 1

        corrected_row["Cleanse Status"] = summary["status"]
        corrected_row["Cleanse Confidence %"] = summary["accuracy_percent"]
        corrected_row["Input Similarity %"] = summary["input_similarity_percent"]
        corrected_row["Latitude"] = structured.get("lat") if structured.get("lat") is not None else ""
        corrected_row["Longitude"] = structured.get("lon") if structured.get("lon") is not None else ""
        corrected_row["Full Verified Address"] = summary["full_verified_address"]
        corrected_row["Original Address Query"] = query
        corrected_row["Row Number"] = idx
        output_rows.append(corrected_row)
        stats["processed"] += 1

    out_stream = io.StringIO()
    # All enrichment columns we can append, in display order.
    ALL_EXTRA_COLUMNS = [
        "Cleanse Action", "Cleanse Status", "Cleanse Confidence %",
        "Input Similarity %", "Latitude", "Longitude", "Full Verified Address",
        "Original Address Query", "Row Number",
    ]
    # Default selection when the client doesn't specify: Status / Original Query
    # / Row Number are OFF by default (the rest on).
    DEFAULT_EXTRA_COLUMNS = [
        "Cleanse Action", "Cleanse Confidence %", "Input Similarity %",
        "Latitude", "Longitude", "Full Verified Address",
    ]
    if isinstance(requested_cols, list):
        chosen = [c for c in ALL_EXTRA_COLUMNS if c in set(requested_cols)]
    else:
        chosen = DEFAULT_EXTRA_COLUMNS
    out_headers = headers + chosen
    writer = csv.DictWriter(out_stream, fieldnames=out_headers, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(output_rows)

    # Keep the session alive (it expires by TTL) so the download step can
    # regenerate the file with a different column selection without re-running.

    cleanse_stats = {
        "processed": stats["processed"],
        "passthrough": stats["passthrough"],
        "total_rows": len(output_rows),
        "approved": stats["approved"],
        "rejected": stats["rejected"],
        "auto_corrected": stats["auto_corrected"],
        "unchanged": stats["unchanged"],
        "rows_corrected": rows_corrected,
        "fields_corrected": fields_corrected,
        "geocoded": geocoded_count,
        "avg_confidence": round(confidence_sum / confidence_n, 1) if confidence_n else 0.0,
        "rows_with_anomaly": rows_with_anomaly,
        # Counter.most_common() -> list of [key, count] pairs (JSON-friendly)
        "changes_by_column": changes_by_column.most_common(),
        "changes_by_role": changes_by_role.most_common(),
        "action_breakdown": action_counts.most_common(),
        "issues_breakdown": issue_counts.most_common(8),
        "anomaly_breakdown": anomaly_counts.most_common(),
    }

    csv_text = out_stream.getvalue()

    # Persist to history (re-download later) when the client downloads for real.
    job_id = ""
    if data.get("save_to_history"):
        try:
            job_id = _save_cleanse_job(
                filename=sess.get("filename", "upload.csv"),
                csv_text=csv_text,
                stats=cleanse_stats,
                columns=chosen,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [!] failed to save cleanse history: {exc}")

    return Response(
        csv_text,
        mimetype="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=corrected_addresses.csv",
            "X-Rows-Processed": str(len(output_rows)),
            "X-Approved": str(stats["approved"]),
            "X-Rejected": str(stats["rejected"]),
            "X-Auto-Corrected": str(stats["auto_corrected"]),
            "X-Cleanse-Stats": json.dumps(cleanse_stats, ensure_ascii=False),
            "X-Job-Id": job_id,
            "Access-Control-Expose-Headers": "X-Rows-Processed, X-Approved, X-Rejected, X-Auto-Corrected, X-Cleanse-Stats, X-Job-Id",
        },
    )


def _save_cleanse_job(filename: str, csv_text: str, stats: dict, columns: list) -> str:
    """Write a completed cleanse to disk so it can be listed and re-downloaded."""
    import uuid
    CLEANSE_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex[:12]
    job_dir = CLEANSE_HISTORY_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "job_id": job_id,
        "created": datetime.now(timezone.utc).isoformat(),
        "filename": filename,
        "total_rows": stats.get("total_rows", 0),
        "processed": stats.get("processed", 0),
        "passthrough": stats.get("passthrough", 0),
        "auto_corrected": stats.get("auto_corrected", 0),
        "approved": stats.get("approved", 0),
        "rejected": stats.get("rejected", 0),
        "fields_corrected": stats.get("fields_corrected", 0),
        "rows_with_anomaly": stats.get("rows_with_anomaly", 0),
        "avg_confidence": stats.get("avg_confidence", 0.0),
        "columns": columns,
    }
    (job_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    (job_dir / "output.csv").write_text(csv_text, encoding="utf-8")
    return job_id


@app.route("/v2/cleanse_history", methods=["GET"])
def cleanse_history_list():
    """List saved cleanse jobs, newest first."""
    jobs = []
    if CLEANSE_HISTORY_DIR.exists():
        for job_dir in CLEANSE_HISTORY_DIR.iterdir():
            meta_path = job_dir / "meta.json"
            if meta_path.exists():
                try:
                    jobs.append(json.loads(meta_path.read_text(encoding="utf-8")))
                except Exception:  # noqa: BLE001
                    continue
    jobs.sort(key=lambda m: m.get("created", ""), reverse=True)
    return jsonify({"jobs": jobs})


@app.route("/v2/cleanse_history/<job_id>/file", methods=["GET"])
def cleanse_history_file(job_id: str):
    """Re-download a saved cleanse output."""
    safe = "".join(c for c in job_id if c.isalnum())
    csv_path = CLEANSE_HISTORY_DIR / safe / "output.csv"
    if not csv_path.exists():
        return jsonify({"error": "Job not found"}), 404
    name = "corrected_addresses.csv"
    meta_path = CLEANSE_HISTORY_DIR / safe / "meta.json"
    if meta_path.exists():
        try:
            orig = json.loads(meta_path.read_text(encoding="utf-8")).get("filename", "")
            stem = re.sub(r"\.[^.]+$", "", orig) or "corrected_addresses"
            name = f"{stem}_cleansed.csv"
        except Exception:  # noqa: BLE001
            pass
    return Response(
        csv_path.read_text(encoding="utf-8"),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={name}"},
    )


@app.route("/v2/cleanse_history/<job_id>", methods=["DELETE"])
def cleanse_history_delete(job_id: str):
    """Delete a saved cleanse job."""
    import shutil
    safe = "".join(c for c in job_id if c.isalnum())
    job_dir = CLEANSE_HISTORY_DIR / safe
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)
    return jsonify({"status": "ok"})


@app.route("/v2/address_cleanse", methods=["GET", "POST"])
def address_cleanse_v2():
    if request.method == "GET":
        return render_template("v2_address_cleanse.html")

    uploaded = request.files.get("csv_file")
    if not uploaded or not uploaded.filename:
        return jsonify({"error": "No file provided"}), 400

    try:
        headers, rows, _raw = _read_upload(uploaded)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Failed to parse file: {exc}"}), 400

    if len(rows) > 1000:
        rows = rows[:1000]

    address_cols = request.form.getlist("address_columns")
    if not address_cols:
        raw_cols = request.form.get("address_columns", "")
        address_cols = [c.strip() for c in raw_cols.split(",") if c.strip()]
    address_cols = [c for c in address_cols if c in headers]
    if not address_cols:
        address_cols = _detect_address_columns(headers)
    if not address_cols:
        return jsonify({"error": "No address columns selected or detected"}), 400

    roles_raw = request.form.get("roles", "")
    try:
        roles = json.loads(roles_raw) if roles_raw else _suggest_column_roles(headers, rows)
    except json.JSONDecodeError:
        roles = _suggest_column_roles(headers, rows)
    roles = {k: v for k, v in roles.items() if v in headers}

    try:
        min_confidence = max(0.0, min(float(request.form.get("min_confidence", "0.80")), 1.0))
    except ValueError:
        min_confidence = 0.80
    country_hint = request.form.get("country_hint", "").strip() or _infer_country(rows, roles)
    apply_mode = request.form.get("apply_mode", "review_only").strip()
    if apply_mode not in {"review_only", "auto_high_confidence"}:
        apply_mode = "review_only"

    try:
        output_rows, stats = _cleanse_rows(
            rows, headers, address_cols, roles, min_confidence, country_hint, apply_mode,
        )
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503

    out_stream = io.StringIO()
    out_headers = headers + [
        "Cleanse Action",
        "Cleanse Status",
        "Cleanse Confidence %",
        "Input Similarity %",
        "Cleanse Issues",
        "Cleanse Suggested Changes",
        "Latitude",
        "Longitude",
        "Full Verified Address",
        "Original Address Query",
        "Row Number",
    ]
    writer = csv.DictWriter(
        out_stream, fieldnames=out_headers,
        lineterminator="\n", extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(output_rows)

    return Response(
        out_stream.getvalue(),
        mimetype="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=corrected_addresses.csv",
            "X-Rows-Processed": str(len(output_rows)),
            "X-Address-Columns": ", ".join(address_cols),
            "X-Auto-Corrected": str(stats["auto_corrected"]),
            "X-Review-Required": str(stats["review_required"]),
            "X-Suggested-Fixes": str(stats["suggested"]),
            "X-Model-Skipped": str(stats["model_skipped"]),
            "X-Unchanged": str(stats["unchanged"]),
            "Access-Control-Expose-Headers": (
                "X-Rows-Processed, X-Address-Columns, X-Auto-Corrected, "
                "X-Review-Required, X-Suggested-Fixes, X-Model-Skipped, X-Unchanged"
            ),
        },
    )


@app.route("/v2/batch", methods=["GET"])
def batch_v2():
    return render_template("v2_batch.html")


@app.route("/v2/analytics", methods=["GET"])
def analytics_v2():
    return render_template("v2_analytics.html")


@app.route("/v2/history", methods=["GET"])
def history_v2():
    return render_template("v2_history.html")


@app.route("/v2/settings", methods=["GET"])
def settings_v2():
    return render_template("v2_settings.html")


@app.route("/v2/validate", methods=["GET"])
def validate_v2():
    return render_template("validate.html")


@app.route("/v2/learn", methods=["GET"])
def learn_v2():
    return render_template("learn.html")


@app.route("/suggest", methods=["GET"])
def suggest():
    if pipeline_models is None and legacy_corrector is None:
        load_engine()

    raw_query = request.args.get("q", "").strip()
    top_n = _parse_top_n()

    if not raw_query:
        return jsonify({"error": "Missing query param 'q'"}), 400

    t0 = time.time()
    if engine_mode == "pipeline_sql":
        response_data = _pipeline_response(raw_query, top_n)
    else:
        response_data = _legacy_response(raw_query, top_n)
        if response_data.get("error"):
            return jsonify({"error": response_data["error"]}), 400

    response_data["latency_ms"] = round((time.time() - t0) * 1000, 1)
    return jsonify(response_data)


@app.route("/v2/correct", methods=["GET", "POST"])
def v2_correct():
    if request.method == "POST":
        body = request.get_json(silent=True) or {}
        raw_query = (body.get("q") or body.get("query") or "").strip()
        top_n = _coerce_top_n(body.get("n", 5))
    else:
        raw_query = request.args.get("q", "").strip()
        top_n = _parse_top_n()

    if not raw_query:
        return jsonify({"error": "Missing query param 'q'"}), 400

    t0 = time.time()
    if not _ensure_v2_available():
        payload = _v2_unavailable_payload(raw_query, top_n)
        payload["latency_ms"] = round((time.time() - t0) * 1000, 1)
        return jsonify(payload)

    try:
        result = _v2_correct_with_recovery(raw_query, top_n=top_n)
        payload = result.to_dict()
        payload = _retry_with_t5_candidate(raw_query, payload, top_n)
    except Exception as exc:  # noqa: BLE001
        return jsonify({
            "error": "v2 correction failed",
            "detail": str(exc),
            "query": raw_query,
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }), 500
    payload["latency_ms"] = round((time.time() - t0) * 1000, 1)
    return jsonify(payload)


def _retry_with_t5_candidate(raw_query: str, payload: dict, top_n: int) -> dict:
    """Recover older high-quality v2 results when dict-only spelling stalls."""
    if v2_pipeline is None:
        return payload
    if (payload.get("confidence") or 0) >= 0.75:
        return payload
    spell = payload.get("spell") or {}
    if spell.get("used_t5"):
        return payload
    speller = getattr(v2_pipeline, "speller", None)
    t5 = getattr(speller, "t5", None)
    if not t5:
        return payload
    if not getattr(t5, "ready", False):
        try:
            t5.load()
        except Exception:
            return payload
    if not getattr(t5, "ready", False):
        return payload
    base = spell.get("corrected") or raw_query
    t5_candidate = t5.correct(base)
    if not t5_candidate or t5_candidate.strip().lower() == base.strip().lower():
        return payload
    if not _safe_t5_retry(base, t5_candidate):
        return payload
    retry = _v2_correct_with_recovery(t5_candidate, top_n=top_n).to_dict()
    retry_score = retry.get("confidence") or 0
    old_score = payload.get("confidence") or 0
    retry_geocoded = (retry.get("verification") or {}).get("geocoded") is True
    old_geocoded = (payload.get("verification") or {}).get("geocoded") is True
    if retry_score > old_score or (retry_geocoded and not old_geocoded):
        retry["query"] = raw_query
        retry.setdefault("notes", []).append("retried_with_t5_candidate")
        retry_spell = retry.setdefault("spell", {})
        retry_spell["used_t5"] = True
        retry_spell["corrected"] = t5_candidate
        return retry
    return payload


def _safe_t5_retry(base: str, candidate: str) -> bool:
    import re

    base_tokens = re.findall(r"[a-z0-9]+", base.lower())
    cand_tokens = re.findall(r"[a-z0-9]+", candidate.lower())
    if not base_tokens or not cand_tokens:
        return False
    base_nums = {t for t in base_tokens if t.isdigit()}
    cand_nums = {t for t in cand_tokens if t.isdigit()}
    if not base_nums.issubset(cand_nums):
        return False

    # Reject hallucinated numeric tokens (phantom house numbers like "68 1"
    # or "78" being added to the front, or fake pincode shorthand "82101").
    # T5 must NOT introduce any new digit tokens beyond what the user typed.
    extra_nums = cand_nums - base_nums
    if extra_nums:
        return False

    # Reject duplicated tail tokens (e.g. "...bangalore 560068 bangalore
    # karnataka 560068" where T5 echoed city/state/pincode twice).
    suffix_window = cand_tokens[-8:]
    seen_in_tail: set[str] = set()
    for tok in suffix_window:
        # Allow short connectors; only flag meaningful repeats.
        if len(tok) < 4:
            continue
        if tok in seen_in_tail:
            return False
        seen_in_tail.add(tok)

    common = {
        "address", "apartment", "apartments", "building", "complex",
        "road", "rd", "main", "cross", "street", "st", "lane", "layout",
        "nagar", "phase", "block", "sector", "near", "opp", "opposite",
        "bangalore", "bengaluru", "bangaluru", "begaluru", "bengalorurui",
        "bengalurur", "bengalure", "bangalor", "karnataka", "india",
    }
    cand_set = set(cand_tokens)
    distinctive = [
        token for token in base_tokens
        if token.isalpha() and len(token) >= 4 and token not in common
    ]
    for token in distinctive:
        if token in cand_set:
            continue
        if any(cand.startswith(token[:4]) for cand in cand_tokens if cand.isalpha()):
            continue
        return False
    # Allow T5 to add city/state suffix (typical training signal) but cap the
    # total growth so it can't dump a whole address tail.
    if len(cand_tokens) > len(base_tokens) + 4:
        return False
    return bool(set(cand_tokens) & {"bangalore", "bengaluru", "karnataka", "india"})


@app.route("/v2/autocomplete", methods=["GET"])
def v2_autocomplete():
    if not _ensure_v2_available():
        raw_query = request.args.get("q", "").strip()
        return jsonify({
            "query": raw_query,
            "suggestions": [],
            "notes": [f"v2_pipeline_unavailable: {v2_error}"],
        })

    raw_query = request.args.get("q", "").strip()
    k = _parse_top_n()
    if not raw_query:
        return jsonify({"suggestions": []})
    t0 = time.time()
    try:
        suggestions = v2_pipeline.autocomplete(raw_query, k=k)
    except Exception as exc:  # noqa: BLE001
        return jsonify({
            "error": "v2 autocomplete failed",
            "detail": str(exc),
            "query": raw_query,
            "suggestions": [],
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }), 500
    return jsonify({
        "query": raw_query,
        "latency_ms": round((time.time() - t0) * 1000, 1),
        "suggestions": [s.__dict__ for s in suggestions],
    })


@app.route("/v2/livesuggest", methods=["GET"])
def v2_livesuggest():
    """Google-style live suggestions: word corrections + address hits."""
    if not _ensure_v2_available():
        raw_query = request.args.get("q", "").strip()
        return jsonify({
            "query": raw_query,
            "corrected": raw_query,
            "changes": [],
            "suggestions": [],
            "notes": [f"v2_pipeline_unavailable: {v2_error}"],
        })
    raw_query = request.args.get("q", "").strip()
    k = _parse_top_n()
    # Default OFF — Google Places autocomplete is billed per request.
    # Frontend opts-in via &google=1 when the user enables it in Settings.
    use_google = request.args.get("google", "0").lower() in ("1", "true", "yes", "on")
    if not raw_query:
        return jsonify({"query": "", "corrected": "",
                        "changes": [], "suggestions": []})
    t0 = time.time()
    try:
        data = v2_pipeline.live_suggest(raw_query, k=k, use_google=use_google)
    except Exception as exc:  # noqa: BLE001
        return jsonify({
            "error": "v2 live suggestions failed",
            "detail": str(exc),
            "query": raw_query,
            "corrected": raw_query,
            "changes": [],
            "suggestions": [],
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }), 500
    data["latency_ms"] = round((time.time() - t0) * 1000, 1)
    return jsonify(data)


@app.route("/v2/feedback", methods=["POST"])
def v2_feedback():
    body = request.get_json(silent=True) or {}
    query = str(body.get("query") or "").strip()
    predicted = str(body.get("predicted") or "").strip()
    corrected = str(body.get("corrected") or "").strip()
    label = body.get("label")

    if not query:
        return jsonify({"error": "Missing required field: query"}), 400
    if label not in (0, 1, False, True, "0", "1", "wrong", "correct"):
        return jsonify({"error": "label must be correct/1 or wrong/0"}), 400

    normalized_label = 1 if label in (1, True, "1", "correct") else 0
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "predicted": predicted,
        "corrected": corrected,
        "label": normalized_label,
        "source": "v2_feedback",
    }
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    return jsonify({"status": "ok", "saved_to": str(FEEDBACK_PATH)})


@app.route("/health", methods=["GET"])
def health():
    stats = {
        "mode": "v2" if v2_pipeline is not None else engine_mode,
        "startup_error": startup_error or v2_error,
        "v2_loaded": v2_pipeline is not None,
        "v2_error": v2_error,
    }
    if v2_pipeline is not None:
        try:
            ret = v2_pipeline.retriever
            n_addr = len(ret.addresses) if hasattr(ret, "addresses") else "?"
            stats.update({
                "total_addresses": n_addr,
                "reranker_loaded": v2_pipeline.reranker is not None,
                "semantic_loaded": hasattr(ret, "faiss_index") and ret.faiss_index is not None,
                "sql_retriever": v2_pipeline.sql_retriever is not None if hasattr(v2_pipeline, "sql_retriever") else False,
            })
        except Exception:
            pass
    elif engine_mode == "pipeline_sql" and pipeline_models is not None:
        addresses = pipeline_models[5]
        rows_by_id = pipeline_models[11]
        stats.update({
            "total_addresses": len(addresses),
            "structured_rows": len(rows_by_id),
            "reranker_loaded": pipeline_models[8] is not None,
            "semantic_loaded": pipeline_models[2] is not None,
        })
    elif legacy_corrector is not None:
        stats.update(legacy_corrector.stats)
    return jsonify({"status": "ok", "stats": stats})


if __name__ == "__main__":
    # Pre-load v2 pipeline in the main thread to avoid worker-thread
    # deadlock / EINVAL issues on Windows with FAISS / torch / SQLite.
    print("Pre-loading v2 pipeline in main thread...")
    _maybe_load_v2()

    if v2_pipeline is None:
        print("  [!] v2 pipeline failed to pre-load (see error above).")
        print("  [!] Running with legacy / SQL pipeline only.")
    else:
        print("  [+] v2 pipeline ready.")

    eager_engine = os.getenv("LOAD_V1_ON_START", "0").lower() in (
        "1", "true", "yes", "on",
    )
    if eager_engine or "--eager" in sys.argv:
        load_engine()
    print("=" * 60)
    print("  Address AI API running")
    print("=" * 60)
    print("  Web Portal : http://localhost:5000/")
    if engine_mode == "pipeline_sql":
        print("  API Test   : http://localhost:5000/suggest?q=prestige%20aprtment%20btm%20bangalr&n=3")
    else:
        print("  API Test   : http://localhost:5000/suggest?q=mumbay&n=3")
    print(f"  Mode       : {engine_mode}")
    print("=" * 60)
    print()
    _port = int(os.getenv("PORT", 5000))
    _host = os.getenv("HOST", "0.0.0.0")
    # Allow --port NNNN and --host X.X.X.X override from command line
    for _i, _a in enumerate(sys.argv):
        if _a == "--port" and _i + 1 < len(sys.argv):
            _port = int(sys.argv[_i + 1])
        if _a == "--host" and _i + 1 < len(sys.argv):
            _host = sys.argv[_i + 1]
    app.run(host=_host, port=_port)
