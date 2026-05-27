"""
fuzzy_engine.v2.config
======================
Centralised configuration for the v2 stack.

Override any of these via environment variables (see _env helpers below).
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Load .env file if present (no python-dotenv dependency required)
_ENV_FILE = ROOT / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip())
PROJECT_ROOT = ROOT  # public alias used by helper scripts
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
V2_ARTIFACTS_DIR = MODELS_DIR / "v2"
V2_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ---- Layer 1 / 2 ----------------------------------------------------------
# Normalization knobs
DEFAULT_COUNTRY = "in"

# ---- Layer 3 retrieval ----------------------------------------------------
RETRIEVAL_TOP_K = 200
AUTOCOMPLETE_TOP_K = 10

BM25_PATH = MODELS_DIR / "bm25.pkl"
FAISS_PATH = MODELS_DIR / "faiss.index"
EMBEDDINGS_PATH = MODELS_DIR / "embeddings.npy"
ADDRESSES_PATH = MODELS_DIR / "addresses.npy"
ADDRESS_IDS_PATH = MODELS_DIR / "address_ids.npy"
TRIE_PATH = V2_ARTIFACTS_DIR / "prefix_trie.pkl"

EMBED_MODEL = os.getenv("V2_EMBED_MODEL", "multi-qa-mpnet-base-dot-v1")

# ---- Layer 4 reranker -----------------------------------------------------
RERANKER_PATH = MODELS_DIR / "reranker.pkl"
CALIBRATOR_PATH = V2_ARTIFACTS_DIR / "calibrator.pkl"
FINAL_TOP_N = 5

# ---- Layer 5 verify -------------------------------------------------------
# Provider selection: "auto" (pick first available), "google", "locationiq",
# "opencage", "nominatim", or "null".
GEOCODER_PROVIDER = os.getenv("V2_GEOCODER", "auto").lower()

# Google Geocoding API.
GOOGLE_API_KEY = os.getenv("GOOGLE_GEOCODE_API_KEY", "")
GOOGLE_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

# LocationIQ (free 5k/day, email signup, no card). https://locationiq.com/
LOCATIONIQ_API_KEY = os.getenv("LOCATIONIQ_API_KEY", "")
LOCATIONIQ_URL = "https://us1.locationiq.com/v1/search"

# OpenCage (free 2.5k/day, email signup, no card). https://opencagedata.com/
OPENCAGE_API_KEY = os.getenv("OPENCAGE_API_KEY", "")
OPENCAGE_URL = "https://api.opencagedata.com/geocode/v1/json"

# Public Nominatim (OSM) - free, 1 req/sec rate limit.
# Self-host or set V2_NOMINATIM_URL to a private instance for production.
NOMINATIM_URL = os.getenv("V2_NOMINATIM_URL", "https://nominatim.openstreetmap.org/search")
NOMINATIM_USER_AGENT = os.getenv(
    "V2_NOMINATIM_UA", "address-ai/2.0 (https://github.com/local)"
)

GEOCODE_TIMEOUT_SEC = float(os.getenv("V2_GEOCODE_TIMEOUT", "8"))
GEOCODE_CACHE_PATH = V2_ARTIFACTS_DIR / "geocode_cache.sqlite"
GEOCODE_CACHE_TTL_DAYS = int(os.getenv("V2_GEOCODE_TTL_DAYS", "30"))

# Backward-compat alias used by older code paths.
GOOGLE_TIMEOUT_SEC = GEOCODE_TIMEOUT_SEC

# India Post pincode reference (CSV produced by data_pipelines/build_pincodes.py).
# Format: pincode,office,district,state
PINCODE_CSV_PATH = DATA_DIR / "india_post_pincodes.csv"

# ---- Confidence calibration ----------------------------------------------
LOW_CONFIDENCE_THRESHOLD = 0.65   # below this -> low_confidence response
HIGH_CONFIDENCE_THRESHOLD = 0.90  # at/above this -> verified ok

# ---- T5 speller (legacy, kept) -------------------------------------------
T5_MODEL_PATH = MODELS_DIR / "t5_address"
T5_BEAMS = 4
