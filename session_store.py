"""
session_store.py — Lightweight in-memory session storage for the interactive
cleanse wizard.  Sessions auto-expire after 2 hours.

Usage
    from session_store import create_session, get_session, update_session

All data is kept in RAM.  On production with many users you should swap
this for Redis or a DB table.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

_sessions: dict[str, dict] = {}

# Auto-expire after 2 hours (seconds)
_SESSION_TTL = 7200


def _now() -> float:
    return time.time()


def _cleanup() -> None:
    """Remove expired sessions.  Called automatically on every write."""
    now = _now()
    expired = [sid for sid, data in _sessions.items() if now - data["_meta"]["created"] > _SESSION_TTL]
    for sid in expired:
        del _sessions[sid]


# ── public API ──

def create_session(payload: dict) -> str:
    _cleanup()
    sid = str(uuid.uuid4())
    _sessions[sid] = {
        "_meta": {"created": _now(), "updated": _now()},
        "payload": payload,
    }
    return sid


def get_session(sid: str) -> dict | None:
    data = _sessions.get(sid)
    if data is None:
        return None
    if _now() - data["_meta"]["created"] > _SESSION_TTL:
        del _sessions[sid]
        return None
    return data["payload"]


def update_session(sid: str, payload: dict) -> bool:
    data = _sessions.get(sid)
    if data is None:
        return False
    data["payload"] = payload
    data["_meta"]["updated"] = _now()
    return True


def delete_session(sid: str) -> None:
    _sessions.pop(sid, None)
