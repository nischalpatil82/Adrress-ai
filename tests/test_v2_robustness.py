"""
End-to-end robustness suite for the v2 address-correction pipeline.

For each query we assert a small set of properties that *must* hold no matter
what flavour of typo the user throws at us:

    - the API returns a non-empty `best_address`
    - it never crashes (status is one of the known values)
    - if the user typed a valid Indian pincode, it survives into structured
    - if the user typed a house number ("house no 81"), it survives
    - "informative" tokens >=4 chars from the cleaned input survive into the
      spell-corrected output (no silent drops by T5)
    - locality / city alias canonicalization fires on common misspellings

The suite is run directly against the in-process pipeline (no HTTP) so it is
cheap and reproducible. Each failure is reported with the field that broke
and a one-line diff to make root-cause obvious.

Run:
    python tests/test_v2_robustness.py
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

# Ensure project root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fuzzy_engine.v2.orchestrator import AddressPipeline


# ----------------------------- test cases ---------------------------------

@dataclass
class Case:
    name: str
    query: str
    # property assertions -- each takes the result dict, returns (ok, msg)
    checks: List[Callable[[dict], tuple]]


def has_text(field: str, needle: str):
    def _f(r):
        v = (r.get(field) or "")
        if isinstance(v, dict):
            v = str(v)
        ok = needle.lower() in str(v).lower()
        return ok, f"{field!r} should contain {needle!r}, got {v!r}"
    return _f


def structured(field: str, expected: str):
    def _f(r):
        v = (r.get("structured") or {}).get(field)
        ok = (v or "").lower() == expected.lower() if v else False
        return ok, f"structured.{field} expected {expected!r}, got {v!r}"
    return _f


def structured_contains(field: str, needle: str):
    def _f(r):
        v = (r.get("structured") or {}).get(field) or ""
        ok = needle.lower() in str(v).lower()
        return ok, f"structured.{field} should contain {needle!r}, got {v!r}"
    return _f


def spell_contains(needle: str):
    def _f(r):
        v = (r.get("spell") or {}).get("corrected") or ""
        ok = needle.lower() in v.lower()
        return ok, f"spell.corrected should contain {needle!r}, got {v!r}"
    return _f


def spell_does_not_contain(needle: str):
    def _f(r):
        v = (r.get("spell") or {}).get("corrected") or ""
        ok = needle.lower() not in v.lower()
        return ok, f"spell.corrected should NOT contain {needle!r}, got {v!r}"
    return _f


def status_in(*allowed):
    def _f(r):
        v = r.get("status")
        return v in allowed, f"status expected one of {allowed}, got {v!r}"
    return _f


def best_nonempty():
    def _f(r):
        v = r.get("best_address") or ""
        return bool(v.strip()), f"best_address must be non-empty, got {v!r}"
    return _f


CASES: List[Case] = [
    # -- basic well-formed queries -----------------------------------------
    Case("clean_residential",
         "12 4th main road jayanagar 9th block bangalore 560069",
         [best_nonempty(), structured("pincode", "560069"),
          spell_contains("jayanagar"), spell_contains("bangalore")]),

    # -- city / locality misspellings --------------------------------------
    Case("bengaloore_typo",
         "rpnc systems near vega city bengaloore",
         [best_nonempty(), spell_contains("bangalore")]),
    Case("banglore_typo",
         "infosys banglore",
         [best_nonempty(), spell_contains("bangalore")]),
    Case("mumbai_misspell",
         "shop no 5 munbai 400001",
         [best_nonempty(), spell_contains("mumbai"),
          structured("pincode", "400001")]),
    Case("hyderabad_misspell",
         "12 hydrabad 500001",
         [best_nonempty(), spell_contains("hyderabad")]),
    Case("jayanagara_typo",
         "no 12 4th main jayanagara bangalore 560011",
         [best_nonempty(), spell_contains("jayanagar")]),

    # -- ordinal-abbreviation typos ----------------------------------------
    Case("ordinal_5t",
         "12 5t main 5t block jayanagar 560041",
         [best_nonempty(), spell_contains("5th"),
          spell_does_not_contain(" 5 t ")]),
    Case("ordinal_2n",
         "10 2n cross indiranagar bangalore 560038",
         [best_nonempty(), spell_contains("2nd")]),
    Case("ordinal_3r",
         "5 3r block koramangala bangalore 560034",
         [best_nonempty(), spell_contains("3rd")]),

    # -- house-number prefixes ---------------------------------------------
    Case("house_no_prefix",
         "house no 81 marenalli 39th b cross jayanagar bangalore 560041",
         [best_nonempty(),
          structured("house_number", "81"),
          spell_contains("marenahalli"),
          spell_contains("81")]),
    Case("h_no_prefix",
         "h no 42 5th cross hsr layout bangalore 560102",
         [best_nonempty(), structured("house_number", "42")]),
    Case("hash_prefix",
         "#7 mg road bangalore 560001",
         [best_nonempty(), structured("pincode", "560001")]),

    # -- POI / business queries (Places API path) --------------------------
    Case("poi_mall",
         "vega city mall bannerghatta road bangalore",
         [best_nonempty(), status_in("generated", "verified",
                                     "found_in_database", "low_confidence")]),
    Case("poi_company",
         "infosys electronic city bangalore 560100",
         [best_nonempty()]),

    # -- pincode handling --------------------------------------------------
    Case("pincode_only",
         "560038",
         [best_nonempty()]),  # very short but valid pincode
    Case("invalid_pincode",
         "12 mg road bangalore 999999",
         [best_nonempty()]),  # should not crash
    Case("near_pincode_5digit",
         "12 mg road bangalore 56001",
         [best_nonempty()]),  # 5-digit -> repair attempt

    # -- generic block / road typos ----------------------------------------
    Case("bock_block",
         "5t bock jayanagar bangalore 560041",
         [best_nonempty(), spell_contains("block")]),
    Case("crosss_cross",
         "12 39th b crosss road jayanagar 560041",
         [best_nonempty(), spell_contains("cross")]),

    # -- punctuation / casing ----------------------------------------------
    Case("all_caps",
         "12 MG ROAD BANGALORE 560001",
         [best_nonempty(), structured("pincode", "560001")]),
    Case("commas_dashes",
         "12, mg-road, bangalore - 560001",
         [best_nonempty()]),
    Case("extra_spaces",
         "  12   mg   road   bangalore   560001  ",
         [best_nonempty()]),

    # -- edge cases (should NOT crash) -------------------------------------
    Case("empty_short",
         "ab",
         [status_in("no_match", "low_confidence")]),
    Case("only_pincode",
         "110001",
         [best_nonempty()]),
    Case("gibberish",
         "xkcd qwerty 99999",
         [status_in("no_match", "low_confidence", "generated")]),

    # -- famous-POI as building name (the gravuty case) --------------------
    Case("apartment_typo",
         "gravuty apertment 11 main rd bangalore 560068",
         [best_nonempty(),
          spell_contains("gravuty"),  # MUST not be dropped
          spell_contains("apartment")]),

    # -- multi-locality alias ----------------------------------------------
    Case("rr_nagar",
         "12 rr nagar bangalore 560098",
         [best_nonempty(), spell_contains("rajarajeshwari")]),
    Case("bg_road",
         "5 b g rd bangalore 560076",
         [best_nonempty(), spell_contains("bannerghatta")]),
]


# ----------------------------- runner -------------------------------------

def run() -> int:
    print("Loading pipeline... (one-time, can take ~30s)\n")
    t0 = time.time()
    pipe = AddressPipeline.from_config(use_t5=True, use_geocoder=True)
    print(f"Loaded in {time.time()-t0:.1f}s. Running {len(CASES)} cases.\n")

    failures = []
    for i, case in enumerate(CASES, 1):
        try:
            t0 = time.time()
            result = pipe.correct(case.query).to_dict()
            dt = (time.time() - t0) * 1000
        except Exception as exc:
            failures.append((case, [f"CRASHED: {exc!r}"]))
            print(f"[{i:02d}] {case.name:30s} CRASH: {exc!r}")
            continue

        msgs = []
        for chk in case.checks:
            ok, msg = chk(result)
            if not ok:
                msgs.append(msg)
        status = result.get("status")
        best = (result.get("best_address") or "")[:80]
        if msgs:
            failures.append((case, msgs))
            tag = "FAIL"
        else:
            tag = " ok "
        print(f"[{i:02d}] {case.name:30s} {tag} {status:18s} ({dt:5.0f}ms) "
              f"-> {best}")
        for m in msgs:
            print(f"       - {m}")

    print("\n" + "=" * 78)
    if failures:
        print(f"\n{len(failures)} / {len(CASES)} cases failed:\n")
        for case, msgs in failures:
            print(f"  - {case.name}: {case.query!r}")
            for m in msgs:
                print(f"      {m}")
        return 1
    print(f"\nAll {len(CASES)} cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
