"""
8_api.py
Flask REST API for the high-accuracy Address AI pipeline.

Default mode loads 5_full_pipeline_sql.py (T5 + BM25 + FAISS + reranker).
Use --legacy or --csv to run the older fuzzy_engine.AddressCorrector path.

Run:
    python 8_api.py
    python 8_api.py --legacy
    python 8_api.py --csv
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Prevent HuggingFace tokenizers from deadlocking in Flask worker threads
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Keep interactive API calls responsive when an external geocoder is blocked
# or unavailable. Users can override this in .env with V2_GEOCODE_TIMEOUT.
os.environ.setdefault("V2_GEOCODE_TIMEOUT", "3")

from flask import Flask, jsonify, redirect, render_template, request


ROOT = Path(__file__).resolve().parent
PIPELINE_PATH = ROOT / "5_full_pipeline_sql.py"
FEEDBACK_PATH = ROOT / "data" / "active_learning_feedback.jsonl"

app = Flask(__name__)
engine_mode = "pipeline_sql"
pipeline_mod = None
pipeline_models = None
legacy_corrector = None
startup_error = None
v2_pipeline = None
v2_error: str | None = None


def _maybe_load_v2() -> None:
    """Lazy-load the v2 pipeline. Failures are non-fatal."""
    global v2_pipeline, v2_error
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
        v2_error = str(exc)
        print(f"  [!] v2 pipeline failed to load: {exc}")


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
    _maybe_load_v2()
    if v2_pipeline is None:
        return jsonify({"error": f"v2 pipeline unavailable: {v2_error}"}), 503

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
    try:
        result = v2_pipeline.correct(raw_query, top_n=top_n)
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
    retry = v2_pipeline.correct(t5_candidate, top_n=top_n).to_dict()
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
    _maybe_load_v2()
    if v2_pipeline is None:
        return jsonify({"error": f"v2 pipeline unavailable: {v2_error}"}), 503

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
    _maybe_load_v2()
    if v2_pipeline is None:
        return jsonify({"error": f"v2 pipeline unavailable: {v2_error}"}), 503
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
    if "--v2" in sys.argv:
        print("Pre-loading v2 pipeline in main thread to avoid worker deadlock...")
        _maybe_load_v2()

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
