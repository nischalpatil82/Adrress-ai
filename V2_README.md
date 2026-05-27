# Address AI — v2 Stack

A 5-layer "Google-search-grade" pipeline for Indian addresses, built **on top of** the existing v1 codebase. v1 is untouched and still works.

```
L1 normalize  -> L2 spell  -> L3 retrieval  -> L4 rerank  -> L5 verify
                                                                ↓
                                         calibrated confidence + structured JSON
```

## What's new vs v1

| Capability | v1 | v2 |
|---|---|---|
| Structured output (`{house, street, city, state, pincode, lat, lon}`) | partial | yes (Google Geocode + DB parse fallback) |
| Real-world verification | none | Google Geocoding API + India Post pincodes |
| Autocomplete (`/autocomplete?q=koramang`) | none | prefix trie, ~5 ms |
| Calibrated confidence | broken (always 89.9) | isotonic-regressed probability |
| Pincode repair (5/7-digit -> 6) | no | yes |
| Devanagari/regional script input | no | optional via `indic-transliteration` |
| Unified provider interface | n/a | swap Google ↔ Nominatim ↔ mock |

## Files

```
fuzzy_engine/v2/
  __init__.py        public AddressPipeline
  config.py          paths + thresholds
  normalize.py       L1
  speller.py         L2  (wraps v1 RapidFuzz + T5; adds word LM)
  retrieval.py       L3  (prefix trie + BM25 + FAISS hybrid)
  reranker.py        L4  (LightGBM + isotonic calibration)
  verify.py          L5  (Google Geocoding + India Post + SQLite cache)
  orchestrator.py    glue + final confidence policy
  build_indexes.py   one-shot artifact builder
tests/test_v2_smoke.py
```

## One-time setup

1. **Install new deps**
   ```powershell
   pip install -r requirements.txt
   ```

2. **Build the prefix trie** (uses your existing `models/addresses.npy` + `models/address_ids.npy`)
   ```powershell
   python -m fuzzy_engine.v2.build_indexes
   ```

3. **Pick a Geocoding provider** (L5 verification)

   The pipeline auto-selects in this order: Google → LocationIQ → OpenCage → Nominatim. Set `V2_GEOCODER` to force a specific one.

   | Provider | Free quota | Card required | How to enable |
   |---|---|---|---|
   | **Nominatim (OSM)** *default if nothing else set* | 1 req/sec | **No** | Nothing — works out of the box |
   | **LocationIQ** | 5,000/day | **No** (email signup) | <https://locationiq.com> -> `$env:LOCATIONIQ_API_KEY = "..."` |
   | **OpenCage** | 2,500/day | **No** (email signup) | <https://opencagedata.com> -> `$env:OPENCAGE_API_KEY = "..."` |
   | **Google** | $200/mo credit (~40k req) | Yes | <https://console.cloud.google.com> -> `$env:GOOGLE_GEOCODE_API_KEY = "..."` |

   Force a specific provider:
   ```powershell
   $env:V2_GEOCODER = "nominatim"     # or google | locationiq | opencage | null
   ```

   Without any provider, L5 falls back to `NullGeocoder` (pipeline still works, no real-world verification).

4. **Optional — India Post pincode CSV**

   Drop a CSV with header `pincode,office,district,state` at `data/india_post_pincodes.csv`. Source: <https://data.gov.in> (search "All India Pincode Directory"). The verifier auto-loads it; without it, pincode validation is skipped silently.

## Run

```powershell
# Start the API (v1 routes still work; v2 mounted at /v2/*)
python 8_api.py
```

```bash
# Autocomplete on every keystroke
curl "http://localhost:5000/v2/autocomplete?q=koramang&n=5"

# Full correction
curl -X POST http://localhost:5000/v2/correct \
  -H "Content-Type: application/json" \
  -d '{"q":"rpnc systems 3rd flour berrergata rood bengaluru","n":5}'
```

Response shape (`/v2/correct`):
```json
{
  "query": "...",
  "status": "verified | high_confidence | medium_confidence | low_confidence | no_match",
  "confidence": 0.93,
  "best_address": "RPNC Systems 3rd Floor Bannerghatta Road Bangalore",
  "structured": {
    "house_number": null,
    "street": "Bannerghatta Road",
    "sublocality": "...",
    "city": "Bengaluru",
    "state": "Karnataka",
    "pincode": "560029",
    "country": "India",
    "lat": 12.91,
    "lon": 77.59,
    "place_id": "ChIJ...",
    "source": "google_geocode"
  },
  "spell": { "applied": true, "corrected": "...", "changes": [...], "used_t5": false },
  "parsed": { ... },
  "verification": { "geocoded": true, "pincode_valid": true, ... },
  "suggestions": [ { "address": "...", "probability": 0.93 }, ... ]
}
```

## Calibration (replaces the broken 89.9% scores)

After running `6_evaluate.py` modified to dump `(raw_score, label)` pairs to JSONL:

```powershell
python -m fuzzy_engine.v2.build_indexes --calibrate data/val_rerank_scores.jsonl
```

This fits an isotonic regressor and saves it to `models/v2/calibrator.pkl`. The orchestrator uses it automatically; without it, a sigmoid fallback is used.

## Cost / quota notes

- Google Geocoding: $200 free credit/month ≈ 40k requests. Each unique address is cached in `models/v2/geocode_cache.sqlite` for 30 days.
- For 49k DB addresses, a one-shot warm-up (verifying every row once) costs ~$45; afterwards near-zero.

## Tests

```powershell
python -m pytest tests/test_v2_smoke.py -q
```

The smoke tests do **not** hit Google; they use a `NullGeocoder` provider.

## What's intentionally NOT in v2 yet

- Char-level model retraining (ByT5). T5 from v1 is wrapped and used as-is; replace with a clean retrain whenever you're ready.
- Self-hosted Nominatim (would need ~250 GB).
- Active-learning / click-feedback loop (planned next).

## Backward compatibility

`fuzzy_engine.AddressCorrector` (v1) is unchanged. `8_api.py` still serves the old `/suggest` endpoint. v2 is purely additive.
