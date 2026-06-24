"""
app.py — Hugging Face Spaces entry point for Address AI.

HF Spaces looks for app.py by default. This file simply imports and
exposes the Flask app from 8_api.py, pre-loading the v2 pipeline so
requests are fast from the first hit.
"""
from __future__ import annotations

import os
import sys

# ── Hugging Face Spaces configuration ──
# HF Spaces always serves on port 7860 (handled by the platform).
# We just need to set the port so Flask knows where to bind.
os.environ.setdefault("PORT", "7860")
os.environ.setdefault("HOST", "0.0.0.0")

# ── Pre-load pipeline at import-time (available before first request) ──
print("[HF Spaces] Pre-loading v2 pipeline...")
from __8_api import _maybe_load_v2  # type: ignore[import-not-found]

_maybe_load_v2()

# ── Import and expose the Flask app ──
from __8_api import app  # type: ignore[import-not-found]

# ── Health-check for HF Spaces ──
@app.route("/health", methods=["GET"])
def health():
    """Lightweight health check for HF Spaces / load balancers."""
    return {"status": "ok"}, 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    host = os.getenv("HOST", "0.0.0.0")
    app.run(host=host, port=port, debug=False)
