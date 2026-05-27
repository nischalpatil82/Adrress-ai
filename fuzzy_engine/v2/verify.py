"""
fuzzy_engine.v2.verify  (Layer 5)
=================================
Ground-truth verification for corrected addresses.

Two complementary signals:
1. Google Geocoding API   - real-world existence + structured components
2. India Post pincode CSV - pincode -> {district, state} authority

Both calls are cached to a local SQLite store so repeated queries are free.

The verifier never raises; it returns a `Verification` dataclass with
boolean flags + structured info that the orchestrator turns into a
calibrated confidence score.

Provider abstraction (`GeocoderProvider`) lets us swap Google for
Nominatim or a mock without touching call sites.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import sqlite3
import time
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Protocol

import requests

from fuzzy_engine.v2.config import (
    DEFAULT_COUNTRY,
    GEOCODE_CACHE_PATH,
    GEOCODE_CACHE_TTL_DAYS,
    GEOCODE_TIMEOUT_SEC,
    GEOCODER_PROVIDER,
    GOOGLE_API_KEY,
    GOOGLE_GEOCODE_URL,
    LOCATIONIQ_API_KEY,
    LOCATIONIQ_URL,
    NOMINATIM_URL,
    NOMINATIM_USER_AGENT,
    OPENCAGE_API_KEY,
    OPENCAGE_URL,
    PINCODE_CSV_PATH,
)

GOOGLE_PLACES_URL = "https://places.googleapis.com/v1/places:searchText"  # New Places API
GOOGLE_PLACES_AUTOCOMPLETE_URL = "https://places.googleapis.com/v1/places:autocomplete"
GOOGLE_ADDRESS_VALIDATION_URL = "https://addressvalidation.googleapis.com/v1:validateAddress"

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GeocodeResult:
    formatted_address: str
    house_number: Optional[str]
    street: Optional[str]
    sublocality: Optional[str]
    locality: Optional[str]            # city
    administrative_area: Optional[str] # state
    postal_code: Optional[str]
    country: Optional[str]
    lat: Optional[float]
    lon: Optional[float]
    place_id: Optional[str]
    location_type: Optional[str]       # ROOFTOP / RANGE_INTERPOLATED / GEOMETRIC_CENTER / APPROXIMATE
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def precision(self) -> float:
        """Map Google location_type to a 0..1 precision score."""
        return {
            "ROOFTOP": 1.0,
            "RANGE_INTERPOLATED": 0.85,
            "GEOMETRIC_CENTER": 0.65,
            "APPROXIMATE": 0.45,
        }.get(self.location_type or "", 0.5)


@dataclass(frozen=True)
class Verification:
    geocoded: bool
    pincode_valid: bool
    pincode_consistent: bool   # pincode's district/state agrees with geocode
    geocode: Optional[GeocodeResult]
    pincode_info: Optional[dict]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "geocoded": self.geocoded,
            "pincode_valid": self.pincode_valid,
            "pincode_consistent": self.pincode_consistent,
            "geocode": asdict(self.geocode) if self.geocode else None,
            "pincode_info": self.pincode_info,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------
class GeocoderProvider(Protocol):
    name: str
    def geocode(self, query: str) -> Optional[GeocodeResult]: ...


# ---------------------------------------------------------------------------
# Google provider
# ---------------------------------------------------------------------------
class GoogleGeocoder:
    name = "google"

    def __init__(self, api_key: str = GOOGLE_API_KEY,
                 country: str = DEFAULT_COUNTRY,
                 timeout: float = GEOCODE_TIMEOUT_SEC) -> None:
        self.api_key = api_key
        self.country = country
        self.timeout = timeout

    def geocode(self, query: str) -> Optional[GeocodeResult]:
        if not self.api_key:
            log.debug("GoogleGeocoder: no API key set; skipping.")
            return None
        params = {
            "address": query,
            "components": f"country:{self.country}",
            "key": self.api_key,
        }
        try:
            r = requests.get(GOOGLE_GEOCODE_URL, params=params, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:  # noqa: BLE001
            log.warning("Google geocode failed for %r: %s", query, exc)
            return None

        if data.get("status") != "OK" or not data.get("results"):
            return None
        return _parse_google_result(data["results"][0])


def _parse_google_result(res: dict) -> GeocodeResult:
    comp_index: dict[str, str] = {}
    for c in res.get("address_components", []):
        for t in c.get("types", []):
            comp_index.setdefault(t, c.get("long_name") or c.get("short_name") or "")
    geom = res.get("geometry", {})
    loc = geom.get("location", {})
    return GeocodeResult(
        formatted_address=res.get("formatted_address", ""),
        house_number=comp_index.get("street_number"),
        street=comp_index.get("route"),
        sublocality=comp_index.get("sublocality")
                    or comp_index.get("sublocality_level_1")
                    or comp_index.get("neighborhood"),
        locality=comp_index.get("locality") or comp_index.get("postal_town"),
        administrative_area=comp_index.get("administrative_area_level_1"),
        postal_code=comp_index.get("postal_code"),
        country=comp_index.get("country"),
        lat=float(loc["lat"]) if "lat" in loc else None,
        lon=float(loc["lng"]) if "lng" in loc else None,
        place_id=res.get("place_id"),
        location_type=geom.get("location_type"),
        raw=res,
    )


# ---------------------------------------------------------------------------
# Google Places API (POI / business search)
# ---------------------------------------------------------------------------
class GooglePlacesGeocoder:
    """Uses Google Places Text Search to find POIs/businesses by name.

    Falls back to standard Geocoding API if Places finds nothing.
    Best for queries like "RPNC Systems Bangalore", "Vega City Mall".
    """
    name = "google_places"

    def __init__(self, api_key: str = GOOGLE_API_KEY,
                 country: str = DEFAULT_COUNTRY,
                 timeout: float = GEOCODE_TIMEOUT_SEC) -> None:
        self.api_key = api_key
        self.country = country
        self.timeout = timeout
        self._geocoder = GoogleGeocoder(api_key=api_key, country=country, timeout=timeout)

    def geocode(self, query: str) -> Optional[GeocodeResult]:
        if not self.api_key:
            return None
        # Residential-address shortcut: when the query begins with an
        # explicit house-number prefix ("house no 81", "h no 12", "no 5",
        # "#42", or starts with a digit), skip Places (which loves famous
        # POIs like libraries/malls and discards the user's house number)
        # and go straight to the Geocoding API.
        import re as _re
        q_lower = query.strip().lower()
        if _re.match(r"^(house\s*no|h\.?\s*no|d\.?\s*no|flat\s*no|no\.?|#)\s*\d", q_lower) \
                or _re.match(r"^\d{1,4}[/,\s-]", q_lower):
            return self._geocoder.geocode(query)
        # New Places API (Text Search) - POST with field mask
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": (
                "places.id,places.displayName,places.formattedAddress,"
                "places.location,places.addressComponents,places.types"
            ),
        }
        body = {
            "textQuery": query,
            "regionCode": self.country.upper(),
            "maxResultCount": 5,
        }
        try:
            r = requests.post(GOOGLE_PLACES_URL, headers=headers, json=body,
                              timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            log.warning("Google Places search failed for %r: %s", query, exc)
            return self._geocoder.geocode(query)

        places = data.get("places", [])
        if not places:
            return self._geocoder.geocode(query)

        # Prefer a place whose postal_code matches the user's pincode
        # (if the query embedded a 6-digit pincode). Otherwise take first.
        import re as _re
        m = _re.search(r"\b(\d{6})\b", query)
        wanted_pin = m.group(1) if m else None
        place = places[0]
        if wanted_pin:
            for cand in places:
                for c in cand.get("addressComponents", []):
                    if "postal_code" in (c.get("types") or []):
                        if (c.get("longText") or c.get("shortText") or "") == wanted_pin:
                            place = cand
                            break
                else:
                    continue
                break
        loc = place.get("location", {})
        comp_index: dict[str, str] = {}
        for c in place.get("addressComponents", []):
            for t in c.get("types", []):
                comp_index.setdefault(t, c.get("longText") or c.get("shortText") or "")

        # Build formatted address: prefix POI/business name when present
        # so the user sees e.g. "Nexus Vega City Mall, Bannerghatta Rd, ..."
        # instead of just the street address with the POI name lost.
        place_name = place.get("displayName", {}).get("text", "")
        formatted_addr = place.get("formattedAddress", "")
        types = place.get("types", []) or []
        is_poi = any(t in ("point_of_interest", "establishment", "store",
                           "shopping_mall", "premise") for t in types)
        if place_name and formatted_addr and is_poi \
                and place_name.lower() not in formatted_addr.lower():
            formatted = f"{place_name}, {formatted_addr}"
        else:
            formatted = formatted_addr or place_name or query

        return GeocodeResult(
            formatted_address=formatted,
            house_number=comp_index.get("street_number"),
            street=comp_index.get("route"),
            sublocality=comp_index.get("sublocality")
                        or comp_index.get("sublocality_level_1")
                        or comp_index.get("neighborhood"),
            locality=comp_index.get("locality") or comp_index.get("postal_town"),
            administrative_area=comp_index.get("administrative_area_level_1"),
            postal_code=comp_index.get("postal_code"),
            country=comp_index.get("country"),
            lat=float(loc["latitude"]) if "latitude" in loc else None,
            lon=float(loc["longitude"]) if "longitude" in loc else None,
            place_id=place.get("id"),
            location_type="GEOMETRIC_CENTER",
            raw=place,
        )


# ---------------------------------------------------------------------------
# Google Places Autocomplete API (live suggestions)
# ---------------------------------------------------------------------------
class GooglePlacesAutocomplete:
    """Thin client over the new Places Autocomplete endpoint.

    Returns a list of suggestion dicts: ``[{"address": "...", "place_id": "..."}]``.

    Cost guards (designed to never bill you):
    - **Disabled by default**. Set ``V2_LIVE_GOOGLE_AC=1`` to enable.
    - Daily request cap (``V2_GOOGLE_AC_DAILY_LIMIT``, default 500). Once hit,
      further calls return [] until UTC midnight.
    - In-process LRU cache for repeated prefixes.
    - Minimum 4-char input.
    """
    name = "google_places_autocomplete"

    def __init__(self, api_key: str = GOOGLE_API_KEY,
                 country: str = DEFAULT_COUNTRY,
                 timeout: float = GEOCODE_TIMEOUT_SEC,
                 cache_size: int = 512,
                 enabled: Optional[bool] = None,
                 daily_limit: Optional[int] = None) -> None:
        self.api_key = api_key
        self.country = country.upper()
        self.timeout = timeout
        self._cache: "OrderedDict[str, list[dict]]" = OrderedDict()
        self._cache_max = cache_size
        # Opt-in flag: env "V2_LIVE_GOOGLE_AC" must be 1/true/yes/on.
        if enabled is None:
            enabled = os.environ.get("V2_LIVE_GOOGLE_AC", "0").lower() in (
                "1", "true", "yes", "on",
            )
        self._enabled = bool(enabled)
        # Daily hard cap (env "V2_GOOGLE_AC_DAILY_LIMIT", default 500).
        if daily_limit is None:
            try:
                daily_limit = int(os.environ.get(
                    "V2_GOOGLE_AC_DAILY_LIMIT", "500",
                ))
            except ValueError:
                daily_limit = 500
        self._daily_limit = max(0, int(daily_limit))
        self._day = ""        # YYYY-MM-DD UTC of current counter
        self._day_count = 0

    @property
    def ready(self) -> bool:
        """True when the env-level opt-in is set AND a key is configured.

        ``has_key`` (below) is what callers should check when the user has
        explicitly opted in via a request flag — that path bypasses the
        env-level gate while still respecting the API key + daily quota.
        """
        return self._enabled and bool(self.api_key)

    @property
    def has_key(self) -> bool:
        """The Places API key is configured (independent of the env-level flag)."""
        return bool(self.api_key)

    def _check_quota(self) -> bool:
        """True if there is budget left in today's quota. Resets at UTC midnight."""
        if self._daily_limit <= 0:
            return False
        today = time.strftime("%Y-%m-%d", time.gmtime())
        if today != self._day:
            self._day = today
            self._day_count = 0
        return self._day_count < self._daily_limit

    def suggest(self, query: str, k: int = 5, force: bool = False) -> list[dict]:
        # ``force=True`` bypasses the env-level opt-in (used when a per-request
        # UI toggle has explicitly opted in). API key + daily quota still apply.
        if force:
            if not self.has_key:
                return []
        elif not self.ready:
            return []
        q = (query or "").strip()
        if len(q) < 4:
            return []
        key = f"{self.country}|{q.lower()}"
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key][:k]
        if not self._check_quota():
            log.info("GooglePlacesAutocomplete: daily limit reached, returning [].")
            return []
        self._day_count += 1
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
        }
        body = {
            "input": q,
            "regionCode": self.country,
            "languageCode": "en",
            "includedRegionCodes": [self.country],
        }
        try:
            r = requests.post(GOOGLE_PLACES_AUTOCOMPLETE_URL, headers=headers,
                              json=body, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:  # noqa: BLE001
            log.warning("Google Places Autocomplete failed for %r: %s", q, exc)
            self._cache_set(key, [])
            return []
        out: list[dict] = []
        for s in data.get("suggestions", []):
            pred = s.get("placePrediction") or {}
            text = (pred.get("text") or {}).get("text") or ""
            if not text:
                continue
            out.append({
                "address": text,
                "place_id": pred.get("placeId"),
            })
        self._cache_set(key, out)
        return out[:k]

    def _cache_set(self, key: str, value: list[dict]) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)


# ---------------------------------------------------------------------------
# Google Address Validation API (highest accuracy)
# ---------------------------------------------------------------------------
class GoogleAddressValidator:
    """Wrapper around Google Address Validation API.

    Returns a GeocodeResult-shaped object with USPS/Google verified components
    plus a `verdict` attached in `raw` for quality flags.

    Used as an OPTIONAL refinement layer on top of Geocoding/Places.
    """
    name = "google_address_validation"

    def __init__(self, api_key: str = GOOGLE_API_KEY,
                 country: str = DEFAULT_COUNTRY,
                 timeout: float = GEOCODE_TIMEOUT_SEC) -> None:
        self.api_key = api_key
        self.country = country.upper()
        self.timeout = timeout

    def validate(self, query: str) -> Optional[GeocodeResult]:
        if not self.api_key:
            return None
        body = {
            "address": {
                "regionCode": self.country,
                "addressLines": [query],
            },
        }
        params = {"key": self.api_key}
        try:
            r = requests.post(
                GOOGLE_ADDRESS_VALIDATION_URL,
                params=params,
                json=body,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            log.warning("AddressValidation failed for %r: %s", query, exc)
            return None

        result = data.get("result", {})
        addr = result.get("address", {})
        geocode = result.get("geocode", {})
        loc = geocode.get("location", {})
        comps: dict[str, str] = {}
        for c in addr.get("addressComponents", []):
            t = c.get("componentType")
            v = c.get("componentName", {}).get("text", "")
            if t and v:
                comps.setdefault(t, v)

        return GeocodeResult(
            formatted_address=addr.get("formattedAddress", query),
            house_number=comps.get("street_number"),
            street=comps.get("route"),
            sublocality=comps.get("sublocality") or comps.get("sublocality_level_1"),
            locality=comps.get("locality") or comps.get("postal_town"),
            administrative_area=comps.get("administrative_area_level_1"),
            postal_code=comps.get("postal_code"),
            country=comps.get("country"),
            lat=float(loc["latitude"]) if "latitude" in loc else None,
            lon=float(loc["longitude"]) if "longitude" in loc else None,
            place_id=geocode.get("placeId"),
            location_type="ROOFTOP" if geocode.get("granularity") == "PREMISE"
                          else "GEOMETRIC_CENTER",
            raw=result,
        )


# ---------------------------------------------------------------------------
# Nominatim provider (free OSM, 1 req/sec public)
# ---------------------------------------------------------------------------
import threading
import time as _time


class _NominatimRateLimiter:
    """1 req/sec global throttle for the public Nominatim endpoint."""
    def __init__(self, min_interval: float = 1.05) -> None:
        self.min_interval = min_interval
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = _time.time()
            wait = self._last + self.min_interval - now
            if wait > 0:
                _time.sleep(wait)
            self._last = _time.time()


_NOMINATIM_LIMITER = _NominatimRateLimiter()


class NominatimGeocoder:
    """Public Nominatim (OSM). Free, no key, but rate-limited to 1 req/sec.

    For production volume, point V2_NOMINATIM_URL at a self-hosted instance
    and set the limiter interval to 0.0 by passing throttle=False.
    """
    name = "nominatim"

    def __init__(self, base_url: str = NOMINATIM_URL,
                 user_agent: str = NOMINATIM_USER_AGENT,
                 country: str = DEFAULT_COUNTRY,
                 timeout: float = GEOCODE_TIMEOUT_SEC,
                 throttle: bool = True) -> None:
        self.base_url = base_url
        self.user_agent = user_agent
        self.country = country
        self.timeout = timeout
        self.throttle = throttle

    def geocode(self, query: str) -> Optional[GeocodeResult]:
        params = {
            "q": query,
            "format": "jsonv2",
            "addressdetails": 1,
            "limit": 1,
            "countrycodes": self.country,
            "accept-language": "en",
        }
        try:
            if self.throttle:
                _NOMINATIM_LIMITER.wait()
            r = requests.get(
                self.base_url,
                params=params,
                headers={"User-Agent": self.user_agent},
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as exc:  # noqa: BLE001
            log.warning("Nominatim geocode failed for %r: %s", query, exc)
            return None
        if not data:
            return None
        return _parse_nominatim_result(data[0])


def _parse_nominatim_result(res: dict) -> GeocodeResult:
    addr = res.get("address", {}) or {}
    # Nominatim's "city" can be in several keys.
    city = (addr.get("city") or addr.get("town") or addr.get("village")
            or addr.get("municipality") or addr.get("state_district"))
    return GeocodeResult(
        formatted_address=res.get("display_name", ""),
        house_number=addr.get("house_number"),
        street=addr.get("road"),
        sublocality=addr.get("suburb") or addr.get("neighbourhood"),
        locality=city,
        administrative_area=addr.get("state"),
        postal_code=addr.get("postcode"),
        country=addr.get("country"),
        lat=float(res["lat"]) if res.get("lat") else None,
        lon=float(res["lon"]) if res.get("lon") else None,
        place_id=str(res.get("place_id")) if res.get("place_id") else None,
        # Map Nominatim's "importance" 0..1 to a Google-style location_type.
        location_type=_nominatim_precision(res),
        raw=res,
    )


def _nominatim_precision(res: dict) -> str:
    klass = (res.get("class") or "").lower()
    typ = (res.get("type") or "").lower()
    if klass == "building" or typ in ("house", "residential"):
        return "ROOFTOP"
    if klass == "highway":
        return "RANGE_INTERPOLATED"
    if klass == "place" and typ in ("city", "town", "village", "suburb"):
        return "GEOMETRIC_CENTER"
    return "APPROXIMATE"


# ---------------------------------------------------------------------------
# LocationIQ provider (free 5k/day, email signup, no card)
# Same Nominatim API shape, hosted infra.
# ---------------------------------------------------------------------------
class LocationIQGeocoder:
    name = "locationiq"

    def __init__(self, api_key: str = LOCATIONIQ_API_KEY,
                 base_url: str = LOCATIONIQ_URL,
                 country: str = DEFAULT_COUNTRY,
                 timeout: float = GEOCODE_TIMEOUT_SEC) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.country = country
        self.timeout = timeout

    def geocode(self, query: str) -> Optional[GeocodeResult]:
        if not self.api_key:
            return None
        params = {
            "key": self.api_key,
            "q": query,
            "format": "json",
            "addressdetails": 1,
            "limit": 1,
            "countrycodes": self.country,
            "accept-language": "en",
        }
        try:
            r = requests.get(self.base_url, params=params, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:  # noqa: BLE001
            log.warning("LocationIQ geocode failed for %r: %s", query, exc)
            return None
        if not isinstance(data, list) or not data:
            return None
        return _parse_nominatim_result(data[0])  # same schema


# ---------------------------------------------------------------------------
# OpenCage provider (free 2.5k/day, email signup, no card)
# ---------------------------------------------------------------------------
class OpenCageGeocoder:
    name = "opencage"

    def __init__(self, api_key: str = OPENCAGE_API_KEY,
                 base_url: str = OPENCAGE_URL,
                 country: str = DEFAULT_COUNTRY,
                 timeout: float = GEOCODE_TIMEOUT_SEC) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.country = country
        self.timeout = timeout

    def geocode(self, query: str) -> Optional[GeocodeResult]:
        if not self.api_key:
            return None
        params = {
            "key": self.api_key,
            "q": query,
            "countrycode": self.country,
            "limit": 1,
            "no_annotations": 1,
            "language": "en",
        }
        try:
            r = requests.get(self.base_url, params=params, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:  # noqa: BLE001
            log.warning("OpenCage geocode failed for %r: %s", query, exc)
            return None
        results = data.get("results") or []
        if not results:
            return None
        return _parse_opencage_result(results[0])


def _parse_opencage_result(res: dict) -> GeocodeResult:
    comp = res.get("components", {}) or {}
    geom = res.get("geometry", {}) or {}
    city = (comp.get("city") or comp.get("town") or comp.get("village")
            or comp.get("suburb") or comp.get("county"))
    return GeocodeResult(
        formatted_address=res.get("formatted", ""),
        house_number=comp.get("house_number"),
        street=comp.get("road"),
        sublocality=comp.get("suburb") or comp.get("neighbourhood"),
        locality=city,
        administrative_area=comp.get("state"),
        postal_code=comp.get("postcode"),
        country=comp.get("country"),
        lat=float(geom["lat"]) if "lat" in geom else None,
        lon=float(geom["lng"]) if "lng" in geom else None,
        place_id=None,
        location_type=("ROOFTOP" if (res.get("confidence") or 0) >= 9
                       else "GEOMETRIC_CENTER"),
        raw=res,
    )


# ---------------------------------------------------------------------------
# Mock provider (offline tests)
# ---------------------------------------------------------------------------
class NullGeocoder:
    name = "null"
    def geocode(self, query: str) -> Optional[GeocodeResult]:  # noqa: ARG002
        return None


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------
def auto_select_provider() -> "GeocoderProvider":
    """Pick the best provider available based on env config.

    Order (when V2_GEOCODER=auto):
        google_places (key) -> locationiq (key) -> opencage (key) -> nominatim (free) -> null

    google_places uses Places Text Search for POI/business queries and falls
    back to standard Geocoding API for regular addresses.
    """
    choice = (GEOCODER_PROVIDER or "auto").lower()

    if choice in ("google_places", "google"):
        return GooglePlacesGeocoder()
    if choice == "geocode_only":
        return GoogleGeocoder()
    if choice == "locationiq":
        return LocationIQGeocoder()
    if choice == "opencage":
        return OpenCageGeocoder()
    if choice == "nominatim":
        return NominatimGeocoder()
    if choice == "null":
        return NullGeocoder()

    # auto: prefer google_places (handles both POI + addresses)
    if GOOGLE_API_KEY:
        return GooglePlacesGeocoder()
    if LOCATIONIQ_API_KEY:
        return LocationIQGeocoder()
    if OPENCAGE_API_KEY:
        return OpenCageGeocoder()
    return NominatimGeocoder()


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
class GeocodeCache:
    """SQLite-backed query cache with TTL.

    Schema:
        cache(key TEXT PK, payload TEXT, fetched_at REAL, provider TEXT)
    """
    def __init__(self, path: Path = GEOCODE_CACHE_PATH,
                 ttl_days: int = GEOCODE_CACHE_TTL_DAYS) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_days * 86400.0
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            "key TEXT PRIMARY KEY, payload TEXT, fetched_at REAL, provider TEXT)"
        )
        self._conn.commit()

    @staticmethod
    def _storage_key(provider: str, query: str) -> str:
        return f"{provider}:{query}"

    def get(self, provider: str, query: str) -> Optional[Optional[GeocodeResult]]:
        """Return cached result. Outer Optional = cache hit; inner = geocoded or not.

        None outer -> miss.
        """
        storage_key = self._storage_key(provider, query)
        cur = self._conn.execute(
            "SELECT payload, fetched_at FROM cache WHERE key = ? AND provider = ?",
            (storage_key, provider),
        )
        row = cur.fetchone()
        if not row:
            cur = self._conn.execute(
                "SELECT payload, fetched_at FROM cache WHERE key = ? AND provider = ?",
                (query, provider),
            )
            row = cur.fetchone()
        if not row:
            return None
        payload, fetched_at = row
        if time.time() - fetched_at > self.ttl:
            return None
        if not payload:
            return None  # cached miss; still expensive to retry; treat as fresh miss
        try:
            data = json.loads(payload)
        except Exception:  # noqa: BLE001
            return None
        return _geocode_from_dict(data)

    def put(self, provider: str, query: str,
            result: Optional[GeocodeResult]) -> None:
        storage_key = self._storage_key(provider, query)
        payload = ""
        if result is not None:
            d = asdict(result)
            d.pop("raw", None)  # keep cache compact
            payload = json.dumps(d, ensure_ascii=False)
        else:
            cur = self._conn.execute(
                "SELECT payload FROM cache WHERE key = ? AND provider = ?",
                (storage_key, provider),
            )
            row = cur.fetchone()
            if row and row[0]:
                log.debug(
                    "Keeping existing geocode cache hit for %s:%r; "
                    "not replacing it with an empty miss.",
                    provider,
                    query,
                )
                return
        self._conn.execute(
            "REPLACE INTO cache (key, payload, fetched_at, provider) "
            "VALUES (?, ?, ?, ?)",
            (storage_key, payload, time.time(), provider),
        )
        self._conn.commit()


def _geocode_from_dict(d: dict) -> GeocodeResult:
    return GeocodeResult(
        formatted_address=d.get("formatted_address", ""),
        house_number=d.get("house_number"),
        street=d.get("street"),
        sublocality=d.get("sublocality"),
        locality=d.get("locality"),
        administrative_area=d.get("administrative_area"),
        postal_code=d.get("postal_code"),
        country=d.get("country"),
        lat=d.get("lat"),
        lon=d.get("lon"),
        place_id=d.get("place_id"),
        location_type=d.get("location_type"),
        raw={},
    )


# ---------------------------------------------------------------------------
# India Post pincode index
# ---------------------------------------------------------------------------
class PincodeIndex:
    """Loads `india_post_pincodes.csv` into memory if available.

    CSV format expected:
        pincode,office,district,state

    Live fallback: if a pincode is not in the bulk CSV (the kishorek mirror
    has ~24k pincodes; newer ones like 560098, 560100, 500096 are missing),
    we look it up against the public api.postalpincode.in service and cache
    the answer in memory + appended to the CSV for future requests.
    """
    LIVE_API = "https://api.postalpincode.in/pincode/{pin}"
    LIVE_TIMEOUT = 5.0

    def __init__(self, csv_path: Path = PINCODE_CSV_PATH,
                 enable_live_fallback: bool = True) -> None:
        self.path = Path(csv_path)
        self._idx: dict[str, dict] = {}
        self._negative: set[str] = set()
        self.enable_live_fallback = enable_live_fallback
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with self.path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pin = (row.get("pincode") or "").strip()
                if not pin or len(pin) != 6 or not pin.isdigit():
                    continue
                # Keep first office; collapse districts/states for duplicates.
                self._idx.setdefault(pin, {
                    "pincode": pin,
                    "office": (row.get("office") or "").strip(),
                    "district": (row.get("district") or "").strip().lower(),
                    "state": (row.get("state") or "").strip().lower(),
                })

    @property
    def loaded(self) -> bool:
        return bool(self._idx)

    def lookup(self, pincode: Optional[str]) -> Optional[dict]:
        if not pincode or len(pincode) != 6 or not pincode.isdigit():
            return None
        hit = self._idx.get(pincode)
        if hit is not None:
            return hit
        if not self.enable_live_fallback or pincode in self._negative:
            return None
        info = self._fetch_live(pincode)
        if info is None:
            self._negative.add(pincode)
            return None
        self._idx[pincode] = info
        try:
            self._append_to_csv(info)
        except Exception:  # noqa: BLE001
            pass
        return info

    # ---- live api ------------------------------------------------------
    def _fetch_live(self, pincode: str) -> Optional[dict]:
        try:
            resp = requests.get(
                self.LIVE_API.format(pin=pincode),
                timeout=self.LIVE_TIMEOUT,
                headers={"User-Agent": "address-ai/2.0"},
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
        except Exception:  # noqa: BLE001
            return None
        if not data or not isinstance(data, list):
            return None
        block = data[0]
        if block.get("Status") != "Success":
            return None
        offices = block.get("PostOffice") or []
        if not offices:
            return None
        po = offices[0]
        return {
            "pincode": pincode,
            "office": (po.get("Name") or "").strip(),
            "district": (po.get("District") or "").strip().lower(),
            "state": (po.get("State") or "").strip().lower(),
        }

    def _append_to_csv(self, info: dict) -> None:
        new_file = not self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["pincode", "office", "district", "state"]
            )
            if new_file:
                writer.writeheader()
            writer.writerow(info)


# ---------------------------------------------------------------------------
# Top-level verifier
# ---------------------------------------------------------------------------
class AddressVerifier:
    """Verify a corrected address against geocoder + pincode authority."""

    def __init__(
        self,
        provider: Optional[GeocoderProvider] = None,
        cache: Optional[GeocodeCache] = None,
        pincodes: Optional[PincodeIndex] = None,
        validator: Optional["GoogleAddressValidator"] = None,
    ) -> None:
        if provider is None:
            provider = auto_select_provider()
        self.provider = provider
        self.cache = cache or GeocodeCache()
        self.pincodes = pincodes or PincodeIndex()
        # Address Validation API runs as a refinement step when a Google key
        # is set. It returns higher-precision components than Geocoding alone.
        if (validator is None and GOOGLE_API_KEY
                and not isinstance(provider, NullGeocoder)):
            validator = GoogleAddressValidator()
        self.validator = validator

    # ---- public ----------------------------------------------------------
    def verify(self, query: str, expected_pincode: Optional[str] = None,
               expected_state: Optional[str] = None) -> Verification:
        notes: list[str] = []

        geo = self._cached_geocode(query)
        if geo is None:
            notes.append("geocode_miss")

        # Refine with Address Validation API if available and enabled.
        # We MERGE: keep Places' lat/lon/formatted_address (better POI precision),
        # but take the validated state/pincode/country from the validator.
        if self.validator is not None:
            try:
                cached_v = self.cache.get("google_address_validation", query)
                if cached_v is not None:
                    refined = cached_v
                else:
                    refined = self.validator.validate(query)
                    self.cache.put("google_address_validation", query, refined)
                if refined is not None:
                    notes.append("validated_by_google")
                    if geo is None:
                        geo = refined
                    else:
                        # Merge: prefer Places POI coords/address, fill missing
                        # components from validated response.
                        from dataclasses import replace
                        geo = replace(
                            geo,
                            administrative_area=(refined.administrative_area
                                                  or geo.administrative_area),
                            postal_code=refined.postal_code or geo.postal_code,
                            locality=refined.locality or geo.locality,
                            sublocality=refined.sublocality or geo.sublocality,
                            street=refined.street or geo.street,
                            house_number=refined.house_number or geo.house_number,
                            country=refined.country or geo.country,
                        )
            except Exception as exc:
                log.warning("AddressValidation refinement failed: %s", exc)

        # pincode validation
        pin = expected_pincode or (geo.postal_code if geo else None)
        info = self.pincodes.lookup(pin) if pin else None
        pincode_valid = bool(info)
        if pin and not info and self.pincodes.loaded:
            notes.append("pincode_unknown")

        # consistency: pincode's state vs geocode state vs expected state
        consistent = True
        if info and geo and geo.administrative_area:
            if info["state"] and info["state"] not in geo.administrative_area.lower():
                consistent = False
                notes.append("pincode_state_disagrees_with_geocode")
        if info and expected_state:
            if info["state"] and info["state"] not in expected_state.lower():
                consistent = False
                notes.append("pincode_state_disagrees_with_expected")

        return Verification(
            geocoded=geo is not None,
            pincode_valid=pincode_valid,
            pincode_consistent=consistent,
            geocode=geo,
            pincode_info=info,
            notes=tuple(notes),
        )

    def geocode(self, query: str) -> Optional[GeocodeResult]:
        """Cached geocode call (no validation)."""
        return self._cached_geocode(query)

    # ---- internals -------------------------------------------------------
    def _cached_geocode(self, query: str) -> Optional[GeocodeResult]:
        if not query:
            return None
        if isinstance(self.provider, NullGeocoder):
            return None
        cached = self.cache.get(self.provider.name, query)
        if cached is not None:
            return cached
        result = self.provider.geocode(query)
        self.cache.put(self.provider.name, query, result)
        return result
