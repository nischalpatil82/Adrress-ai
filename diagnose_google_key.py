"""
Quick diagnostic for the Google API key used by the v2 verifier.

Tests four endpoints individually so we know exactly which APIs need to be
enabled / which restriction is blocking the key:

    1. Geocoding API           (basic geocode)
    2. Places API (New)        - text search
    3. Places API (New)        - autocomplete
    4. Address Validation API

Run:

    python diagnose_google_key.py

Reads the key from env var GOOGLE_GEOCODE_API_KEY (same as the running app).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests


def _load_env_file() -> None:
    """Load .env exactly like fuzzy_engine.v2.config does (no dotenv dep)."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v


def _read_key() -> str:
    _load_env_file()
    key = os.getenv("GOOGLE_GEOCODE_API_KEY", "").strip()
    if not key:
        print("ERROR: env var GOOGLE_GEOCODE_API_KEY is not set.")
        print("Set it in PowerShell with:")
        print('    $env:GOOGLE_GEOCODE_API_KEY = "AIza..."')
        sys.exit(1)
    print(f"Using key: ...{key[-8:]}  (length {len(key)})")
    return key


def _hint(status: int, body: str) -> str:
    body_l = body.lower()
    if status == 403:
        if "api has not been used" in body_l or "is disabled" in body_l:
            return "API NOT ENABLED → enable it in Google Cloud Console."
        if "referer" in body_l or "referrer" in body_l:
            return "HTTP REFERRER restriction is blocking. Add http://localhost:5000/* OR clear restrictions."
        if "ip" in body_l and "allowed" in body_l:
            return "IP restriction is blocking. Add your IP OR clear restrictions."
        if "permission_denied" in body_l:
            return "PERMISSION_DENIED → either billing is disabled, or API restrictions exclude this API."
        return "403 Forbidden — most often: billing not enabled, or API not enabled, or key restrictions."
    if status == 400:
        return "400 Bad Request — request format issue (less likely a config problem)."
    if status == 429:
        return "429 — quota exceeded."
    return ""


def test_endpoint(name: str, method: str, url: str,
                  params: dict | None = None,
                  headers: dict | None = None,
                  body: dict | None = None) -> bool:
    print()
    print("=" * 70)
    print(f"TEST: {name}")
    print(f"  {method} {url}")
    try:
        if method == "GET":
            r = requests.get(url, params=params, timeout=15)
        else:
            r = requests.post(url, params=params, headers=headers,
                              json=body, timeout=15)
    except requests.RequestException as exc:
        print(f"  NETWORK ERROR: {exc}")
        return False

    print(f"  HTTP {r.status_code}")
    text = r.text or ""
    if len(text) > 600:
        text = text[:600] + "...(truncated)"
    print(f"  body: {text}")

    if r.ok:
        try:
            data = r.json()
            top_status = data.get("status") or data.get("error", {}).get("status")
            if top_status and top_status not in ("OK", "ZERO_RESULTS"):
                print(f"  application status: {top_status}")
                msg = data.get("error_message") or data.get("error", {}).get("message")
                if msg:
                    print(f"  message: {msg}")
                return False
            print("  RESULT: OK")
            return True
        except json.JSONDecodeError:
            print("  RESULT: OK (non-json response)")
            return True

    hint = _hint(r.status_code, text)
    if hint:
        print(f"  >>> {hint}")
    return False


def main() -> None:
    key = _read_key()
    results: dict[str, bool] = {}

    # 1. Geocoding API (legacy, simple GET)
    results["Geocoding API"] = test_endpoint(
        "Geocoding API",
        "GET",
        "https://maps.googleapis.com/maps/api/geocode/json",
        params={"address": "MG Road, Bangalore", "key": key},
    )

    # 2. Places API (New) - text search
    results["Places API (New) - searchText"] = test_endpoint(
        "Places API (New) - searchText",
        "POST",
        "https://places.googleapis.com/v1/places:searchText",
        headers={
            "Content-Type": "application/json",
            "X-Goog-Api-Key": key,
            "X-Goog-FieldMask": "places.formattedAddress,places.id",
        },
        body={"textQuery": "Gopalan Mall Bannerghatta Road Bangalore",
              "maxResultCount": 1},
    )

    # 3. Places API (New) - autocomplete
    results["Places API (New) - autocomplete"] = test_endpoint(
        "Places API (New) - autocomplete",
        "POST",
        "https://places.googleapis.com/v1/places:autocomplete",
        headers={
            "Content-Type": "application/json",
            "X-Goog-Api-Key": key,
        },
        body={"input": "MG Road Bangalore",
              "includedRegionCodes": ["IN"]},
    )

    # 4. Address Validation API
    results["Address Validation API"] = test_endpoint(
        "Address Validation API",
        "POST",
        "https://addressvalidation.googleapis.com/v1:validateAddress",
        params={"key": key},
        headers={"Content-Type": "application/json"},
        body={"address": {"regionCode": "IN",
                          "addressLines": ["MG Road, Bangalore"]}},
    )

    print()
    print("=" * 70)
    print("SUMMARY")
    for name, ok in results.items():
        print(f"  {'OK ' if ok else 'FAIL'}  {name}")

    failed = [n for n, ok in results.items() if not ok]
    if failed:
        print()
        print("NEXT STEPS to fix the failing API(s):")
        print("  1. Open https://console.cloud.google.com/")
        print("  2. Select the GCP project that owns this key")
        print("  3. Billing -> confirm a billing account is linked AND active")
        print("  4. APIs & Services -> Library -> enable each failing API:")
        for n in failed:
            print(f"        - {n}")
        print("  5. APIs & Services -> Credentials -> click your key:")
        print("        - Application restrictions: 'None' while testing")
        print("        - API restrictions: 'Don't restrict key' while testing")
        print("  6. Wait ~1 minute for changes to propagate, then re-run this script.")
    else:
        print()
        print("All four APIs are OK. The verifier should now work end-to-end.")


if __name__ == "__main__":
    main()
