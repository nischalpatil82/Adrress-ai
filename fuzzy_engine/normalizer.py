"""
fuzzy_engine.normalizer
Text normalization and geographic token extraction utilities.
"""

import re
from fuzzy_engine.config import STREET_KEYWORDS


_DISPLAY_SLASH_NUM_PATTERN = re.compile(r"\b\d+(?:/\d+)+\b")


def normalize(addr: str) -> str:
    """
    Normalize an address string for comparison.
    - Lowercase
    - Strip all punctuation except spaces
    - Collapse multiple whitespace to single space
    - Strip leading/trailing whitespace
    """
    addr = re.sub(r"[^\w\s]", " ", str(addr).lower())
    addr = re.sub(r"_+", " ", addr)
    addr = re.sub(r"(?<=\d)(?!(?:st|nd|rd|th)\b)(?=[a-z])|(?<=[a-z])(?=\d)", " ", addr)
    return re.sub(r"\s+", " ", addr).strip()


def extract_geo_tokens(text: str, known_cities: set, known_areas: set) -> dict:
    """
    Parse an address string into geographic components.

    Returns dict with:
        city          : str or None  — detected city name
        area_tokens   : list[str]    — detected area/neighborhood names
        street_tokens : list[str]    — detected street-type words
        number_tokens : list[str]    — detected numeric tokens
        other_tokens  : list[str]    — everything else
    """
    tokens = normalize(text).split()
    result = {
        "city": None,
        "area_tokens": [],
        "number_tokens": [],
        "street_tokens": [],
        "other_tokens": [],
    }
    for tok in tokens:
        if tok in known_cities:
            result["city"] = tok
        elif tok.isdigit():
            result["number_tokens"].append(tok)
        elif tok in known_areas:
            result["area_tokens"].append(tok)
        elif tok in STREET_KEYWORDS:
            result["street_tokens"].append(tok)
        else:
            result["other_tokens"].append(tok)
    return result

def format_generated_address(text: str) -> str:
    """Normalize and format a generated address for user-friendly output.

    This keeps correction logic normalized while improving display readability,
    including reconstruction of common house-number patterns.
    """
    placeholders = {}

    def _stash_slash(match: re.Match) -> str:
        key = f"slashnumtoken{'x' * (len(placeholders) + 1)}"
        placeholders[key] = match.group(0)
        return key

    prepared = _DISPLAY_SLASH_NUM_PATTERN.sub(_stash_slash, str(text).lower())
    tokens = normalize(prepared).split()
    if not tokens:
        return ""

    upper_tokens = {
        "ii", "iii", "iv", "vi", "vii", "viii", "ix", "x",
        "ncr", "uk", "usa", "uae",
        "jp", "mg", "hsr", "blr", "btm", "gst", "ecr", "omr"
    }

    # Reconstruct common split house-number patterns from normalized text,
    # e.g. "31 32 a" -> "31/32A".
    reconstructed = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]

        # Pattern: <num> <num> <single alpha> => <num>/<num><ALPHA>
        if (
            i + 2 < len(tokens)
            and tok.isdigit()
            and tokens[i + 1].isdigit()
            and tokens[i + 2].isalpha()
            and len(tokens[i + 2]) == 1
        ):
            reconstructed.append(f"{tok}/{tokens[i + 1]}{tokens[i + 2].upper()}")
            i += 3
            continue

        # Pattern: <num> <single alpha> => <num><ALPHA>
        if (
            i + 1 < len(tokens)
            and tok.isdigit()
            and tokens[i + 1].isalpha()
            and len(tokens[i + 1]) == 1
        ):
            reconstructed.append(f"{tok}{tokens[i + 1].upper()}")
            i += 2
            continue

        reconstructed.append(tok)
        i += 1

    out = []
    for idx, tok in enumerate(reconstructed):
        if tok in placeholders:
            out.append(placeholders[tok])
            continue
        if tok.isdigit():
            out.append(tok)
        elif tok == "no":
            # Prefer standard house-number prefix style at display time.
            has_number_ahead = idx + 1 < len(reconstructed) and (
                reconstructed[idx + 1].isdigit() or "/" in reconstructed[idx + 1]
            )
            out.append("No." if has_number_ahead else "No")
        elif tok in upper_tokens:
            out.append(tok.upper())
        elif "/" in tok:
            # Keep mixed slash-number tokens as-is (already reconstructed).
            out.append(tok)
        else:
            out.append(tok.capitalize())
    return " ".join(out)
