"""Unit tests for pincode/postal-code validation across India + international.

Mocks the geocoder so no network/API key is required. Verifies that:
  - Indian pincodes are validated against the India Post index.
  - International postal codes (US ZIP, UK alphanumeric) are NOT reported
    invalid just because they are absent from the India Post CSV.
  - repair_pincode never fabricates a structurally invalid Indian pincode.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fuzzy_engine.v2.normalize import repair_pincode
from fuzzy_engine.v2.verify import (
    AddressVerifier,
    GeocodeResult,
    NullGeocoder,
)


def _geo(country, postal):
    return GeocodeResult(
        formatted_address=f"somewhere, {country}",
        house_number=None, street=None, sublocality=None,
        locality="city", administrative_area="state",
        postal_code=postal, country=country,
        lat=1.0, lon=1.0, place_id="x", location_type="ROOFTOP",
    )


def _verifier_with_geo(geo):
    """AddressVerifier whose geocode step always returns `geo`."""
    v = AddressVerifier(provider=NullGeocoder())
    v._cached_geocode = lambda query: geo  # type: ignore[assignment]
    v.validator = None  # skip Address Validation refinement
    return v


def run() -> int:
    failures = []

    def check(name, cond, detail=""):
        print(f"[{'ok' if cond else 'FAIL'}] {name}  {detail}")
        if not cond:
            failures.append(name)

    # --- International: US ZIP must be valid (geocoder is authority) --------
    v = _verifier_with_geo(_geo("United States", "94103"))
    r = v.verify("1600 amphitheatre pkwy mountain view 94103")
    check("us_zip_valid", r.pincode_valid is True,
          f"pincode_valid={r.pincode_valid}")
    check("us_consistent", r.pincode_consistent is True)

    # --- International: UK alphanumeric postcode ----------------------------
    v = _verifier_with_geo(_geo("United Kingdom", "SW1A 1AA"))
    r = v.verify("10 downing street london sw1a 1aa")
    check("uk_postcode_valid", r.pincode_valid is True,
          f"pincode_valid={r.pincode_valid}")

    # --- International but geocoder returned no postal code -> not valid ----
    v = _verifier_with_geo(_geo("United States", None))
    r = v.verify("somewhere usa")
    check("intl_no_postal_invalid", r.pincode_valid is False,
          f"pincode_valid={r.pincode_valid}")

    # --- Indian pincode present in India Post index ------------------------
    v = _verifier_with_geo(_geo("India", "560001"))
    r = v.verify("mg road bangalore 560001", expected_pincode="560001")
    check("india_known_pincode_valid", r.pincode_valid is True,
          f"pincode_valid={r.pincode_valid}")

    # --- repair_pincode invariants -----------------------------------------
    check("repair_doubled_digit", repair_pincode("5600053") == "560053",
          repr(repair_pincode("5600053")))
    check("repair_5digit_unrecoverable", repair_pincode("56043") is None,
          repr(repair_pincode("56043")))
    check("repair_leading_zero_rejected", repair_pincode("056043") is None,
          repr(repair_pincode("056043")))
    check("repair_valid_passthrough", repair_pincode("560078") == "560078")

    print("\n" + "=" * 50)
    if failures:
        print(f"{len(failures)} FAILED: {failures}")
        return 1
    print("All pincode validation tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
