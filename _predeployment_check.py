"""Pre-deployment health check — runs a set of test queries and reports results."""
import requests, json

BASE = "http://localhost:5000"

TESTS = [
    # (query, expected_keyword_in_spell_or_db_match)
    ("BTM Layout 2nd Stage Bangalore",             "btm layout"),
    ("vijaynagar 2nd stage bangalore 560040",      "vijayanagar"),
    ("14th cross 29th main btm layout bangalore",  "btm layout"),
    ("Koramangala 5th Block Bangalore",            "koramangala"),
    ("gandhi road sanganer jaipur rajasthan",      "jaipur"),
    ("bannerghatta road bangalore 560029",         "bannerghatta"),
    ("indiranagar 100 feet road bangalore",        "indiranagar"),
    ("hebbal flyover bangalore 560024",            "hebbal"),
    ("jayanagar 4th block bangalore",              "jayanagar"),
    ("malleswaram bangalore 560003",               "malleswaram"),
]

print("\n" + "="*100)
print("PRE-DEPLOYMENT HEALTH CHECK")
print("="*100)

passed = 0
failed = 0
errors = 0

for q, expect in TESTS:
    try:
        r = requests.post(f"{BASE}/v2/correct", json={"q": q, "n": 5}, timeout=30)
        d = r.json()

        status   = d.get("status", "?")
        conf     = d.get("confidence") or 0
        spell    = (d.get("spell") or {}).get("corrected") or q
        sugs     = d.get("suggestions") or []
        db_match = sugs[0].get("address", "") if sugs else ""
        cands    = len(sugs)

        hit = (
            expect.lower() in spell.lower()
            or expect.lower() in db_match.lower()
            or expect.lower() in (d.get("best_address") or "").lower()
        )

        flag = "PASS" if hit else "FAIL"
        if hit:
            passed += 1
        else:
            failed += 1

        print(f"\n[{flag}] {q}")
        print(f"       Status   : {status}")
        print(f"       Confidence: {conf:.1f}%")
        print(f"       Spell     : {spell[:80]}")
        print(f"       Top match : {db_match[:80] if db_match else 'NO CANDIDATES'}")
        print(f"       Candidates: {cands}")

    except Exception as e:
        errors += 1
        print(f"\n[ERROR] {q}")
        print(f"       {e}")

print("\n" + "="*100)
print(f"RESULTS: {passed} PASSED  |  {failed} FAILED  |  {errors} ERRORS  |  Total {len(TESTS)}")
print("="*100)

# Also check /health endpoint
print("\nHealth endpoint:")
try:
    h = requests.get(f"{BASE}/health", timeout=5).json()
    s = h.get("stats", h)
    print(f"  mode        : {s.get('mode')}")
    print(f"  v2_loaded   : {s.get('v2_loaded')}")
    print(f"  total_addr  : {s.get('total_addresses')}")
    print(f"  reranker    : {s.get('reranker_loaded')}")
    print(f"  sql_active  : {s.get('sql_retriever')}")
    print(f"  startup_err : {s.get('startup_error') or 'None'}")
except Exception as e:
    print(f"  ERROR: {e}")
