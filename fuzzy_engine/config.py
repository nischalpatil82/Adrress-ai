"""
fuzzy_engine.config
All tunable parameters for the address correction engine.
Centralised here so the entire system can be tuned from one file.
"""

# ── Matching Thresholds ───────────────────────────────────────────────────────
# If the spell-corrected input matches a DB address at or above this %,
# the address is considered to ALREADY EXIST in the database.
DB_MATCH_THRESHOLD = 90

# Number of top suggestions to return when address is not found in DB.
TOP_N_SUGGESTIONS = 5

# ── Spell Correction Thresholds ───────────────────────────────────────────────
# Minimum fuzz.ratio score to accept a misspelling dictionary fuzzy match.
MISSPELLING_FUZZY_THRESHOLD = 82

# Minimum fuzz.ratio score to accept a vocabulary fuzzy match.
VOCAB_FUZZY_THRESHOLD = 78

# Minimum token length for fuzzy matching against misspelling dictionary.
MIN_TOKEN_LEN_MISSPELLING = 3

# Minimum token length for fuzzy matching against full vocabulary.
MIN_TOKEN_LEN_VOCAB = 4

# Maximum allowed length difference when accepting a vocabulary fuzzy match.
MAX_LEN_DIFF_VOCAB = 3

# ── Fuzzy Scoring Weights ─────────────────────────────────────────────────────
# Weights for combining multiple fuzzy strategies (must sum to 1.0).
W_TOKEN_SORT_RATIO = 0.30
W_TOKEN_SET_RATIO  = 0.25
W_PARTIAL_RATIO    = 0.20
W_RATIO            = 0.25

# Number of candidates to pull from each fuzzy strategy.
CANDIDATES_PER_STRATEGY = 30
CANDIDATES_PER_STRATEGY_RAW = 20

# ── Geo Boosting Weights ─────────────────────────────────────────────────────
# Bonus scores added when geographic components match.
BOOST_CITY_MATCH   = 0.18    # +18% when city matches
BOOST_AREA_MATCH   = 0.12    # +12% per matching area token
BOOST_STREET_MATCH = 0.05    # +5% when street type matches
BOOST_NUMBER_MATCH = 0.05    # +5% when house/flat number matches
BOOST_TOKEN_OVERLAP = 0.05   # up to +5% for general token overlap (reduced to avoid inflating generic matches)
BOOST_ROAD_NAME_MATCH = 0.25 # +25% when specific road/area name matches (dominant geo signal)
BOOST_PINCODE_MATCH = 0.12   # +12% when pincode matches

# ── Geo Penalty Weights ──────────────────────────────────────────────────────
# Penalties applied when geographic components explicitly mismatch.
PENALTY_STREET_MISMATCH = 0.30  # -30% when query says "X road" but candidate says "Y road"
PENALTY_AREA_MISMATCH   = 0.20  # -20% when query has a specific area but candidate has a different one

# Maximum final score (capped to avoid 100%).
MAX_SCORE = 99.9

# ── Area Detection ────────────────────────────────────────────────────────────
# Minimum number of address appearances for a word to be considered an area name.
MIN_AREA_FREQUENCY = 5

# Words to exclude when detecting area names from addresses.
AREA_STOP_WORDS = {
    "no", "flat", "road", "street", "the", "and", "of", "in", "to",
    "is", "at", "on", "for", "with", "india", "maharashtra", "karnataka",
    "tamil", "nadu", "telangana", "west", "bengal", "uttar", "pradesh",
    "rajasthan", "gujarat", "delhi", "main", "cross", "ring", "old",
    "new", "sea", "face", "tara", "rai",
}

# ── Protected Words ──────────────────────────────────────────────────────────
# Short words that should NEVER be spell-corrected (common in addresses).
PROTECTED_WORDS = {
    "no", "mg", "sv", "hsr", "lg", "jp", "dl", "nd", "up",
    "in", "of", "at", "to", "is", "or", "an", "by",
    # Common short words that exist in addresses but not as area/street names
    "opp", "opd", "lab", "bus", "old", "new", "big", "via",
    # Address words that must NEVER be "corrected" by fuzzy matching
    "stage", "main", "road", "near", "plot", "flat",
}

# Street-type keywords used in geo extraction.
STREET_KEYWORDS = {
    "road", "street", "avenue", "marg", "lane", "cross", "drive",
    "highway", "path", "way", "boulevard",
}

# ── SQL Blocking (RDBMS candidate narrowing) ───────────────────────────────
# Max rows retrieved from DB for a single blocked search.
SQL_BLOCK_LIMIT = 2000

# Min token length to use as a block term.
SQL_BLOCK_MIN_TOKEN_LEN = 4

# Generic tokens excluded from SQL blocking clauses.
SQL_BLOCK_STOP_WORDS = {
    "address", "house", "flat", "building", "near", "opp", "opposite",
    "road", "street", "avenue", "lane", "main", "cross", "sector",
    "nagar", "layout", "block", "phase", "india", "state", "city",
}
