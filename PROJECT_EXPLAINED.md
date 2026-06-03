# Address AI — Complete Project Explanation (Beginner Friendly)

This document explains **everything** about this project .

---

## 1. What does this project actually do?

It is an **Indian-address cleaner / corrector**.

You type a messy address like:
```
rpnc systms 3rd flour berrergata rood bengaluru
```

The system gives you back the cleaned-up version:
```
RPNC Systems, 3rd Floor, Bannerghatta Road, Bengaluru, Karnataka 560029
```

…plus structured fields (`house_number`, `street`, `city`, `state`, `pincode`,
`lat`, `lon`), a confidence score, and alternative suggestions.

It is served as a **website + REST API** on `http://localhost:5000`.

There are **two generations** of the engine in this repo:

| Version | Where | When to use |
|---|---|---|
| **v1** (legacy) | `fuzzy_engine/*.py` (top level) | Old endpoint `/suggest`. Still works. |
| **v2** (new) | `fuzzy_engine/v2/*.py` | Used by `/v2/correct`, `/v2/autocomplete`, `/v2/livesuggest`. Smarter, structured output, optional Google API. |

When you open `http://localhost:5000/v2` in the browser, you are using **v2**.

---

## 2. The big picture (mental model)

Think of v2 as **5 stages on an assembly line**. Each stage gets the address
a little cleaner.

```
You type → L1 Normalize → L2 Spell-correct → L3 Retrieve candidates
                       → L4 Re-rank candidates → L5 Verify with the real world
                       → Final answer + confidence
```

- **L1 Normalize** — lowercase, fix punctuation, expand "rd" → "road", etc.
- **L2 Spell** — fix typos using a fine-tuned T5 model + dictionaries.
- **L3 Retrieve** — find ~50 closest matches from your database using three
  techniques in parallel: **prefix trie**, **BM25** (keyword search) and
  **FAISS** (semantic/meaning search).
- **L4 Re-rank** — a small ML model (LightGBM) re-scores those 50 and picks
  the best ~5.
- **L5 Verify** — call **Google Geocoding** (or a free fallback) to confirm
  the address actually exists in the real world, plus check the pincode
  against an India Post pincode file.

The output is one JSON blob with everything: corrected text, structured
fields, confidence %, suggestions, and what changed.

---

## 3. Folder & file map (what lives where)

```
address_ai1/
├── 0_clean_training_data.py    ← clean raw CSV
├── 1_prepare_data.py           ← generate noisy↔clean pairs to train T5
├── 2_finetune_t5.py            ← train the T5 spell-corrector
├── 3_build_indexes.py          ← build BM25 + FAISS indexes
├── 4_train_reranker.py         ← train the LightGBM re-ranker
├── 5_full_pipeline.py          ← v1 pipeline (in-memory)
├── 5_full_pipeline_sql.py      ← v1 pipeline (MySQL backed) — used by /suggest
├── 6_evaluate.py               ← measure accuracy
├── 7_rl_bandit.py              ← (optional) reinforcement learning tuner
├── 8_api.py                    ← the Flask web server (you run this)
│
├── address_schema.sql          ← MySQL table definitions
├── db.py                       ← MySQL connection helper
├── import_realistic_to_sql.py  ← load realistic_addresses.csv into MySQL
├── requirements.txt            ← Python libraries to install
│
├── .env                        ← your secrets (DB password, Google API key)
│
├── data/                       ← CSVs, training pairs, pincode lists
├── models/                     ← trained AI artifacts (T5, FAISS, BM25, ranker)
│   └── v2/                     ← v2 specific artifacts (trie, calibrator, geocode cache)
│
├── templates/
│   ├── index.html              ← v1 web UI
│   └── v2.html                 ← v2 web UI (the one with live suggestions)
│
├── fuzzy_engine/               ← v1 LIBRARY (don't edit, still used)
│   ├── corrector.py            ← main v1 entry point
│   ├── normalizer.py           ← text cleaning
│   ├── spell_checker.py        ← RapidFuzz spell correction
│   ├── matcher.py              ← regex/pattern matchers
│   ├── dictionaries.py         ← huge Indian city/state vocab
│   ├── phonetics.py            ← Soundex-style sound matching
│   ├── probabilistic.py        ← statistical fallback
│   ├── db_loader.py            ← reads addresses from MySQL
│   ├── t5_model.py             ← loads the trained T5
│   └── …
│
└── fuzzy_engine/v2/            ← v2 LIBRARY (the new stuff)
    ├── __init__.py             ← exposes `AddressPipeline`
    ├── config.py               ← all paths and thresholds
    ├── normalize.py            ← L1
    ├── speller.py              ← L2 (wraps T5 + dictionaries)
    ├── retrieval.py            ← L3 (trie + BM25 + FAISS hybrid)
    ├── reranker.py             ← L4 (LightGBM + isotonic calibration)
    ├── verify.py               ← L5 (Google Geocoder + Places Autocomplete + India Post)
    ├── orchestrator.py         ← THE BRAIN — glues L1..L5 together
    ├── build_indexes.py        ← one-shot artifact builder
    ├── corpus_lexicons.py      ← extract street/city words from your DB
    ├── locality_aliases.py     ← common neighborhood name variants
    ├── fetch_pincodes.py       ← download India Post pincode CSV
    ├── warm_geocode_cache.py   ← pre-cache geocoder responses (optional)
    └── fit_calibrator.py       ← fit the isotonic confidence calibrator
```

---

## 4. The tech stack in 60 seconds

| What | Why it's there |
|---|---|
| **Python 3** | The language everything is written in. |
| **Flask** | Tiny web framework. It exposes the API and serves the HTML pages. |
| **MySQL** | Stores the 49k real addresses + structured columns. |
| **T5 (transformer)** | A small AI language model that we fine-tuned to fix typos in addresses. |
| **BM25** | Old-school keyword search. Fast and good at "exact words". |
| **FAISS** | Facebook's library for **vector similarity search** — finds addresses with similar *meaning*. |
| **Sentence-transformers** | Turns text into vectors (numbers) that FAISS can compare. |
| **LightGBM** | A small ML re-ranker that learns "given these scores, which candidate is the right answer?" |
| **RapidFuzz** | Super-fast fuzzy string matching (typo tolerance). |
| **Google Geocoding API** | Confirms the corrected address exists in the real world (paid, but with $200/mo free credit). |
| **Google Places Autocomplete** | Optional, OFF by default — for the live-suggest box. |
| **India Post pincode CSV** | Free; validates pincode → city/state. |
| **Leaflet + OpenStreetMap** | Free map shown in `v2.html`. |
| **TailwindCSS** | Styling in the HTML pages. |

---

## 5. How a request flows end-to-end

Imagine you typed `gopalan mall bennergata rd bangalor` in the v2 web UI
and pressed search. Here is what happens, step by step:

1. **Browser** sends `POST /v2/correct` with the JSON `{"q":"gopalan mall …"}`.
2. `8_api.py` receives it (route `v2_correct`).
3. It calls `v2_pipeline.correct(raw_query)` — this lives in
   `fuzzy_engine/v2/orchestrator.py` inside the class `AddressPipeline`.
4. Inside `AddressPipeline.correct(...)`:
   1. **L1 normalize** — `normalize.py` lowercases, strips junk,
      expands abbreviations (`rd` → `road`).
   2. **L2 spell** — `speller.py` runs T5 + RapidFuzz dictionary checks.
      Output: `gopalan mall bannerghatta road bangalore`.
   3. **L3 retrieve** — `retrieval.py` searches:
      - **Trie**: prefix matches like `gopal…`
      - **BM25**: keyword scoring across all DB addresses
      - **FAISS**: semantic vector match
      …returns ~50 candidates with raw scores.
   4. **L4 re-rank** — `reranker.py` feeds all 50 into the LightGBM
      model with features (BM25 score, FAISS score, fuzzy ratio, length, …).
      Output: top 5 with calibrated probabilities.
   5. **L5 verify** — `verify.py` takes the top candidate and:
      - Calls Google Geocoding (cached in SQLite) → lat/lon, formatted address
      - Looks up pincode in India Post CSV → validates city/state
      - Builds the final `structured` dict
   6. **Confidence policy** in `orchestrator.py` decides the final status:
      `verified` / `high_confidence` / `medium_confidence` / `low_confidence` /
      `no_match`.
5. Flask returns a JSON blob to the browser.
6. The JS in `templates/v2.html` renders the corrected address, structured
   fields, suggestions, confidence pill, and pins the lat/lon on the
   Leaflet map.

For the **live-suggest box** (the smaller one on top), the route is
`/v2/livesuggest` which calls `AddressPipeline.live_suggest(q)`. It runs the
fast trie + BM25 search and (optionally) Google Places Autocomplete in
parallel — see Section 9.

---

## 6. The .env file

Lives at the project root. Contains secrets — never share it.

Typical contents:
```env
DB_HOST=127.0.0.1
DB_PORT=3306
DB_USER=root
DB_PASSWORD=yourpassword
DB_NAME=address_ai
GOOGLE_GEOCODE_API_KEY=AIza...your-key
```

**You do NOT put `V2_LIVE_GOOGLE_AC` here** — keep it out so Google Places
Autocomplete stays OFF by default (cost safety).

---

## 7. First-time setup (one time only)

> Run all commands in **PowerShell** from `c:\Users\User\Desktop\address_ai1`.

### 7.1 Install Python libraries
```powershell
pip install -r requirements.txt
```

### 7.2 Create the MySQL database
```powershell
mysql -u root -p < address_schema.sql
```
Enter your MySQL password when prompted.

### 7.3 Put DB credentials into `.env`
Edit `.env` and fill `DB_PASSWORD` etc.

### 7.4 Import the 49k real addresses into MySQL
```powershell
python import_realistic_to_sql.py
```

### 7.5 Build all AI artifacts (this is the heavy step)
Run them **in order**:
```powershell
python 1_prepare_data.py          # makes noisy↔clean training pairs
python 2_finetune_t5.py --device cpu   # trains the T5 (slow on CPU, ~hours)
python 3_build_indexes.py         # builds BM25 + FAISS + addresses.npy
python 4_train_reranker.py        # trains LightGBM re-ranker
python -m fuzzy_engine.v2.build_indexes    # builds v2 prefix trie + lexicons
```

If you do not want to retrain T5 yourself, use the provided
`colab_t5_finetune.ipynb` on Google Colab (free GPU).

### 7.6 (Optional) Download India Post pincode CSV
```powershell
python -m fuzzy_engine.v2.fetch_pincodes
```
Saves `data/india_post_pincodes.csv`. Without it, pincode validation is skipped silently.

### 7.7 (Optional) Pick a geocoder
Set ONE of these env vars before running the server:
```powershell
$env:GOOGLE_GEOCODE_API_KEY = "AIza..."   # paid (free $200/mo credit)
# OR
$env:LOCATIONIQ_API_KEY     = "..."       # free 5k/day
# OR
$env:OPENCAGE_API_KEY       = "..."       # free 2.5k/day
# OR set nothing → defaults to Nominatim (free, 1 req/sec)
```

---

## 8. How to run the server

Every day, just one command:
```powershell
python 8_api.py
```

You should see:
```
Loading high-accuracy SQL pipeline...
  [+] v2 pipeline loaded.
============================================================
  Address AI API running
============================================================
  Web Portal : http://localhost:5000/
  API Test   : http://localhost:5000/suggest?q=mumbay&n=3
  Mode       : pipeline_sql
============================================================
```

Now open:
- **v1 UI**: `http://localhost:5000/`
- **v2 UI**: `http://localhost:5000/v2`  ← the modern one

Stop the server with `Ctrl+C`.

### Optional run modes
```powershell
python 8_api.py --legacy   # force v1 DB engine
python 8_api.py --csv      # force v1 with the CSV only (no MySQL)
```

---

## 9. Google Places Autocomplete — cost-safe behavior

The **live-suggest box** (small one on top of `v2.html`) can use Google
Places Autocomplete to give Google-quality suggestions while you type.

**It is OFF by default** so you never get billed accidentally.

How it works:

| Setting | Default | What it does |
|---|---|---|
| `V2_LIVE_GOOGLE_AC` | `0` (off) | Set to `1` to enable Google live suggest. |
| `V2_GOOGLE_AC_DAILY_LIMIT` | `500` | Hard cap on Google calls per day per server process. |
| Frontend debounce | 350 ms | Waits before sending a request. |
| Min input length | 4 chars | Below that, no Google call. |
| LRU cache | in-memory | Same query within the session = 1 call. |

**To turn it ON for one session** (PowerShell):


```powershell
$env:V2_LIVE_GOOGLE_AC = "1"
python 8_api.py
```


Closing the terminal removes the env var → back to safe default.

**To turn it OFF** (the default): just run `python 8_api.py` normally.

When ON, `orchestrator.live_suggest()` runs **DB search + Google in parallel**
and merges results with Google entries first. Each suggestion has a badge in
the UI (`DB` or `Google`) so you know the source.

---

## 10. The HTTP API (every endpoint)

| Method + URL | What it does | Used by |
|---|---|---|
| `GET /` | Serves v1 HTML page | browser |
| `GET /v2` | Serves v2 HTML page | browser |
| `GET /suggest?q=...&n=5` | v1 correction | legacy/API |
| `POST /v2/correct` body `{"q":"...","n":5}` | Full v2 correction (the main one) | v2 UI big box |
| `GET /v2/autocomplete?q=...&n=5` | Prefix-trie autocomplete only (fast, DB only) | small typeahead |
| `GET /v2/livesuggest?q=...&n=5` | Word fix + address hits (DB + optional Google) | v2 UI live box |
| `POST /v2/feedback` body `{query, predicted, label}` | Log a correct/wrong click for later training | v2 UI thumbs up/down |
| `GET /health` | Engine status + counts | monitoring |

### Quick examples
```powershell
# Full correction
curl -X POST http://localhost:5000/v2/correct `
     -H "Content-Type: application/json" `
     -d '{"q":"rpnc systms 3rd flour berrergata rood","n":5}'

# Health check
curl http://localhost:5000/health
```

---

## 11. The v2 pipeline files explained one by one

### `fuzzy_engine/v2/config.py`
Just constants: paths to `models/v2/trie.pkl`, BM25, FAISS, calibrator;
thresholds for confidence buckets. **Edit this if you move folders.**

### `fuzzy_engine/v2/normalize.py` (L1)
- Lowercases input.
- Expands abbreviations (`rd`→`road`, `apt`→`apartment`, …).
- Strips weird unicode, fixes pincode digits (5/7 → 6).
- Extracts house numbers, road anchors, locality words → returns a
  `ParsedAddress` dataclass used downstream.

### `fuzzy_engine/v2/speller.py` (L2)
- Runs the fine-tuned **T5** model (loaded from `models/t5_address/`).
- Cross-checks T5 output against `CorpusLexicons` (city/street words from
  your DB) using **RapidFuzz** to avoid hallucinations.
- If T5 drops important tokens, the safer dictionary-corrected fallback wins.

### `fuzzy_engine/v2/retrieval.py` (L3)
- **Trie** prefix lookup (super fast, for autocomplete).
- **BM25** (loaded from `models/bm25.pkl`).
- **FAISS** dense vector search (loaded from `models/faiss.index` +
  `models/embeddings.npy`).
- Returns ~50 unique candidates with multi-signal scores.

### `fuzzy_engine/v2/reranker.py` (L4)
- Loads `models/reranker.pkl` (LightGBM).
- Builds features per candidate: BM25 score, dense score, fuzz ratio,
  token overlap, length diff, etc.
- Applies **isotonic calibration** (`models/v2/calibrator.pkl`) so the
  number you see is a real probability, not a meaningless 89.9%.

### `fuzzy_engine/v2/verify.py` (L5)
- `GooglePlacesGeocoder` — calls Google Geocoding (cached in
  `models/v2/geocode_cache.sqlite` for 30 days). Falls back to
  LocationIQ → OpenCage → Nominatim → NullGeocoder.
- `GooglePlacesAutocomplete` — the cost-guarded live-suggest client
  (Section 9).
- `IndiaPostPincodes` — loads `data/india_post_pincodes.csv` and
  validates pincode → city/state.

### `fuzzy_engine/v2/orchestrator.py`  ← THE BRAIN
This is the most important file. The `AddressPipeline` class:
- `from_config()` — class-method that builds the whole pipeline from disk.
- `correct(query, top_n=5)` — runs L1→L5, returns a `CorrectionResult`.
- `autocomplete(q, k)` — fast trie-only suggestions.
- `live_suggest(q, k)` — word corrections + DB hits + optional Google
  Places (parallelized).
- `_generate_address(...)` — composes the final pretty string from
  user input + geocoder + India Post; applies **trust checks** (e.g.
  reject Google result if its pincode doesn't match user's).
- `_structured_from_input(...)` — builds the `structured` JSON; prefers
  the user's house number over Google's.

### `fuzzy_engine/v2/build_indexes.py`
Run once after step 3 (the v1 BM25/FAISS build) to add v2 artifacts:
- Builds prefix `trie.pkl`.
- Builds `corpus_lexicons.pkl` (vocabulary lifted from your DB).
- `--calibrate path.jsonl` fits the isotonic regressor.

---

## 12. Where each model file comes from

| File | Created by | Size hint |
|---|---|---|
| `models/t5_address/` | `2_finetune_t5.py` | ~250 MB |
| `models/bm25.pkl` | `3_build_indexes.py` | tens of MB |
| `models/faiss.index` | `3_build_indexes.py` | ~100 MB |
| `models/embeddings.npy` | `3_build_indexes.py` | ~100 MB |
| `models/addresses.npy` | `3_build_indexes.py` | small |
| `models/address_ids.npy` | `3_build_indexes.py` | small |
| `models/reranker.pkl` | `4_train_reranker.py` | small |
| `models/v2/trie.pkl` | `fuzzy_engine.v2.build_indexes` | small |
| `models/v2/calibrator.pkl` | `fuzzy_engine.v2.build_indexes --calibrate …` | tiny |
| `models/v2/geocode_cache.sqlite` | auto, populated by Google calls | grows |
| `models/bandit.json` | `7_rl_bandit.py` | tiny |

If any of these are missing, the corresponding stage degrades gracefully —
the pipeline still answers, just with less power.

---

## 13. Front-end (`templates/v2.html`)

A single HTML page with:
- **TailwindCSS** for styling.
- **Leaflet + OpenStreetMap** for the map (free).
- Two search boxes:
  1. **Live suggest** (top) — calls `/v2/livesuggest` with a 350 ms debounce.
     Shows word-correction chips + suggestion list (with `DB` / `Google`
     badges).
  2. **Full correction** (bottom) — calls `POST /v2/correct` on submit.
     Renders structured fields, confidence pill, suggestions, and pins
     lat/lon on the map.
- Thumbs up / down buttons → `POST /v2/feedback` for active learning.

No build step. It's plain HTML/JS, loaded directly by Flask.

---

## 14. Costs cheat-sheet

- **MySQL, Python, all local AI models, Leaflet/OSM**: 100% free, runs on your machine.
- **Google Geocoding** (`/v2/correct`): paid per request, **$200/month free
  credit** (≈ 40k requests). Cached for 30 days in
  `models/v2/geocode_cache.sqlite`, so the same address only costs once.
- **Google Places Autocomplete** (`/v2/livesuggest`): paid per keystroke
  if enabled — **disabled by default**. Daily cap 500 + LRU cache + 4-char
  min + 350 ms debounce. Set `V2_LIVE_GOOGLE_AC=1` only when you want it.
- The free Google trial credit ₹28,365 expires Aug 6, 2026. After that,
  Google's monthly **Always-Free** quotas apply (28k Autocomplete + 40k
  Geocoding per month).

---

## 15. Troubleshooting

| Symptom | Likely cause / fix |x
|---|---|
| `v2 pipeline failed to load` on startup | An artifact is missing under `models/` or `models/v2/`. Re-run step 3 + `fuzzy_engine.v2.build_indexes`.|
| `MySQL connection failed` | Check `.env` DB credentials, make sure MySQL service is running. |
| `confidence` always 89.9% | Calibrator not built. Run `python -m fuzzy_engine.v2.build_indexes --calibrate data/val_rerank_scores.jsonl`. |
| Live suggest only shows DB results | That's the default. Set `$env:V2_LIVE_GOOGLE_AC = "1"` before starting `python 8_api.py`. |
| Live suggest still hits Google after I unset the var | A previous shell still has it set; close PowerShell tab and reopen, or run `Remove-Item Env:V2_LIVE_GOOGLE_AC`. |
| Map doesn't show | You're offline; OSM tiles need internet. |
| Geocoder is "null"/no lat-lon | No geocoder env var set — pipeline still works but skips L5 verification. |

---

## 16. A glossary for absolute beginners

- **Pipeline** — a sequence of processing steps applied in order.
- **Token** — a single word/number after splitting the text.
- **Trie** — a tree data structure for super-fast prefix lookup
  (`koram…` → `koramangala`).
- **BM25** — a classic ranking formula used by search engines for keyword
  relevance.
- **FAISS** — a library that finds the most similar vectors (numerical
  fingerprints of text) in milliseconds.
- **Embedding / Vector** — a list of numbers representing the *meaning*
  of a piece of text. Similar meanings = nearby vectors.
- **T5** — a transformer language model from Google; we fine-tuned a small
  version to fix address typos.
- **LightGBM** — a fast gradient-boosted-tree ML library; here it learns
  to pick the best candidate.
- **Re-ranker** — model that re-orders an already-retrieved candidate list.
- **Isotonic calibration** — turns raw scores into honest probabilities.
- **Geocoder** — service that converts an address string into lat/lon.
- **Autocomplete / typeahead** — suggestions appearing as you type.
- **Debounce** — waiting a short time after the last keystroke before
  sending a request, to avoid spamming.
- **LRU cache** — "least recently used" cache that keeps the most recent
  N answers in memory.

---

## 17. Daily-use one-liner

```powershell
python 8_api.py


$env:V2_LIVE_GOOGLE_AC = "1"
python 8_api.py


```
Then open `http://localhost:5000/v2`. That's it. Everything else in this
file is reference.

---

*End of explanation. Keep this file next to `README.md` for context.*
