"""
fuzzy_engine.v2.locality_aliases
================================
Indian locality / road / landmark alias dictionary + canonicalizer.

Goal:
    "rr nagar"            -> "rajarajeshwari nagar"
    "bg road"             -> "bannerghatta road"
    "bkc"                 -> "bandra kurla complex"
    "t nagar"             -> "thyagaraya nagar"
    ...

Two-phase canonicalization:
1. Greedy longest-match phrase replacement (1-4 token windows).
2. Single-token fallback (rare abbrevs).

Apply at BOTH index build time (so corpus is canonical) AND query time (so
the user's input speaks the same language as the corpus).

The dictionary is intentionally curated rather than mined — small, fast,
high precision. Mining can extend it later via append-only updates.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

# ---------------------------------------------------------------------------
# Seed dictionary (alias -> canonical)
#
# Conventions:
# - keys and values lowercase, single-spaced
# - prefer the FULL form as canonical (better embedding signal)
# - multi-word keys are matched as exact contiguous phrases
# ---------------------------------------------------------------------------
LOCALITY_ALIASES: dict[str, str] = {
    # ===== BANGALORE / BENGALURU =====
    # West / South-West
    "rr nagar": "rajarajeshwari nagar",
    "raj rajeshwari nagar": "rajarajeshwari nagar",
    "rajrajeshwari nagar": "rajarajeshwari nagar",
    "rajeshwari nagar": "rajarajeshwari nagar",
    "rrn": "rajarajeshwari nagar",
    "vv puram": "vishweshwarapuram",
    "vvpuram": "vishweshwarapuram",
    "k r puram": "krishnarajapuram",
    "kr puram": "krishnarajapuram",
    "krpuram": "krishnarajapuram",
    "k r market": "krishna rajendra market",
    "kr market": "krishna rajendra market",
    "krmarket": "krishna rajendra market",
    "k r road": "krishna rajendra road",
    "kr road": "krishna rajendra road",
    "kr nagar": "krishnaraja nagar",

    # South
    "jp nagar": "jayaprakash narayan nagar",
    "j p nagar": "jayaprakash narayan nagar",
    "jpnagar": "jayaprakash narayan nagar",
    "jp ngr": "jayaprakash narayan nagar",
    "btm": "btm layout",
    "btm 1st stage": "btm layout 1st stage",
    "btm 2nd stage": "btm layout 2nd stage",
    "btm layout 1": "btm layout 1st stage",
    "btm layout 2": "btm layout 2nd stage",
    "hsr": "hsr layout",
    "hsr lyt": "hsr layout",
    "bg road": "bannerghatta road",
    "b g road": "bannerghatta road",
    "bgr": "bannerghatta road",
    "bannergatta road": "bannerghatta road",
    "bannergatta rd": "bannerghatta road",
    "begur main road": "begur main road",
    "begur rd": "begur road",
    "bommanahalli": "bommanahalli",
    "hongasandra": "hongasandra",
    "or road": "outer ring road",
    "orr": "outer ring road",
    "outer ring rd": "outer ring road",
    "ir road": "inner ring road",
    "inner ring rd": "inner ring road",
    "j nagar": "jayanagar",
    "jaya nagar": "jayanagar",
    "jayangar": "jayanagar",

    # East
    "e city": "electronic city",
    "ecity": "electronic city",
    "elec city": "electronic city",
    "electronics city": "electronic city",
    "wfd": "whitefield",
    "wfld": "whitefield",
    "whitefld": "whitefield",
    "white field": "whitefield",
    "marathahali": "marathahalli",
    "marathali": "marathahalli",
    "marathahally": "marathahalli",
    "kr puram": "krishnarajapuram",
    "kadugodi": "kadugodi",
    "varthur": "varthur",
    "itpl": "international tech park",
    "itpb": "international tech park",
    "international tech park bangalore": "international tech park",

    # Central
    "mg road": "mahatma gandhi road",
    "m g road": "mahatma gandhi road",
    "m.g road": "mahatma gandhi road",
    "mg rd": "mahatma gandhi road",
    "brigade rd": "brigade road",
    "residency rd": "residency road",
    "commercial st": "commercial street",
    "cubbon rd": "cubbon road",
    "infantry rd": "infantry road",
    "shivajinagar": "shivaji nagar",
    "shivaji ngr": "shivaji nagar",
    "ulsoor": "ulsoor",
    "halsuru": "ulsoor",
    "bsk": "banashankari",
    "bsk 1st stage": "banashankari 1st stage",
    "bsk 2nd stage": "banashankari 2nd stage",
    "bsk 3rd stage": "banashankari 3rd stage",
    "bsk 6th stage": "banashankari 6th stage",
    "banashankari 1": "banashankari 1st stage",
    "banashankari 2": "banashankari 2nd stage",
    "banashankari 3": "banashankari 3rd stage",
    "banashankari 6": "banashankari 6th stage",

    # North
    "hbr": "hbr layout",
    "hbr lyt": "hbr layout",
    "hrbr": "hrbr layout",
    "hrbr lyt": "hrbr layout",
    "rt nagar": "ramaswamy temple nagar",
    "r t nagar": "ramaswamy temple nagar",
    "rmv": "rmv extension",
    "rmv 1st stage": "rmv extension 1st stage",
    "rmv 2nd stage": "rmv extension 2nd stage",
    "yel new town": "yelahanka new town",
    "yelahanka nt": "yelahanka new town",
    "yel ntn": "yelahanka new town",
    "ind nagar": "indiranagar",
    "indra nagar": "indiranagar",
    "indranagar": "indiranagar",
    "indira nagar": "indiranagar",
    "kalyan ngr": "kalyan nagar",
    "kammanahalli": "kammanahalli",
    "kamanahalli": "kammanahalli",
    "frazer town": "frazer town",
    "fraser town": "frazer town",
    "kor": "koramangala",
    "korm": "koramangala",
    "koramangla": "koramangala",
    "koramangalaa": "koramangala",
    "koramangla 1 blk": "koramangala 1st block",
    "koramangala 1 blk": "koramangala 1st block",
    "koramangala 4 blk": "koramangala 4th block",
    "koramangala 5 blk": "koramangala 5th block",
    "koramangala 6 blk": "koramangala 6th block",
    "koramangala 7 blk": "koramangala 7th block",
    "koramangala 8 blk": "koramangala 8th block",
    "100ft road": "100 feet road",
    "100 ft road": "100 feet road",
    "100 feet rd": "100 feet road",

    # ===== MUMBAI =====
    "bkc": "bandra kurla complex",
    "lbs marg": "lal bahadur shastri marg",
    "lbs road": "lal bahadur shastri marg",
    "sv road": "swami vivekananda road",
    "sv rd": "swami vivekananda road",
    "s v road": "swami vivekananda road",
    "wel road": "western express highway",
    "weh": "western express highway",
    "eeh": "eastern express highway",
    "navi mum": "navi mumbai",
    "vashi": "vashi",
    "andheri w": "andheri west",
    "andheri e": "andheri east",
    "andheri west": "andheri west",
    "andheri east": "andheri east",
    "bandra w": "bandra west",
    "bandra e": "bandra east",
    "borivali w": "borivali west",
    "borivali e": "borivali east",
    "malad w": "malad west",
    "malad e": "malad east",
    "ghat road": "ghatkopar road",
    "ghatkopar w": "ghatkopar west",
    "ghatkopar e": "ghatkopar east",

    # ===== DELHI / NCR =====
    "cp": "connaught place",
    "c p": "connaught place",
    "kg marg": "kasturba gandhi marg",
    "kg road": "kasturba gandhi marg",
    "gk": "greater kailash",
    "gk 1": "greater kailash 1",
    "gk 2": "greater kailash 2",
    "gk i": "greater kailash 1",
    "gk ii": "greater kailash 2",
    "gk part 1": "greater kailash 1",
    "gk part 2": "greater kailash 2",
    "ip extension": "indraprastha extension",
    "ip ext": "indraprastha extension",
    "lajpat nagar 1": "lajpat nagar 1st",
    "lajpat nagar 2": "lajpat nagar 2nd",
    "lajpat nagar 3": "lajpat nagar 3rd",
    "lajpat nagar 4": "lajpat nagar 4th",
    "ndls": "new delhi",
    "olf marg": "outer ring road",
    "ring rd": "ring road",
    "south ext": "south extension",
    "south ext 1": "south extension 1",
    "south ext 2": "south extension 2",
    "ggn": "gurgaon",
    "gurugram": "gurgaon",
    "noi": "noida",
    "g nagar": "ghaziabad",
    "ghazi": "ghaziabad",
    "fbd": "faridabad",

    # ===== HYDERABAD / SECUNDERABAD =====
    "hyd": "hyderabad",
    "sec bad": "secunderabad",
    "secbad": "secunderabad",
    "kphb": "kukatpally housing board colony",
    "kphb colony": "kukatpally housing board colony",
    "kpb": "kukatpally housing board colony",
    "ecil": "electronics corporation of india limited",
    "necil": "electronics corporation of india limited",
    "hitec city": "hitech city",
    "hi tech city": "hitech city",
    "hitechcity": "hitech city",
    "gachi bowli": "gachibowli",
    "gachibowly": "gachibowli",
    "madhapur": "madhapur",
    "sr nagar": "sanjeev reddy nagar",
    "s r nagar": "sanjeev reddy nagar",
    "kondapur": "kondapur",
    "begumpet": "begumpet",
    "lb nagar": "lal bahadur nagar",
    "l b nagar": "lal bahadur nagar",
    "ameerpet": "ameerpet",
    "amerpet": "ameerpet",
    "uppal": "uppal",
    "dilsukhnagar": "dilsukh nagar",
    "dilsuk nagar": "dilsukh nagar",

    # ===== CHENNAI =====
    "t nagar": "thyagaraya nagar",
    "t ngr": "thyagaraya nagar",
    "tnagar": "thyagaraya nagar",
    "kk nagar": "kalaignar karunanidhi nagar",
    "k k nagar": "kalaignar karunanidhi nagar",
    "vv nagar": "valluvar nagar",
    "anna ngr": "anna nagar",
    "anna nagar w": "anna nagar west",
    "anna nagar e": "anna nagar east",
    "vadapalani": "vadapalani",
    "vadalapani": "vadapalani",
    "velachery": "velachery",
    "velacheri": "velachery",
    "tambaram": "tambaram",
    "thambaram": "tambaram",
    "omr": "old mahabalipuram road",
    "old mahabalipuram rd": "old mahabalipuram road",
    "ecr": "east coast road",
    "east coast rd": "east coast road",
    "gst road": "grand southern trunk road",
    "g s t road": "grand southern trunk road",
    "mount rd": "anna salai",
    "mount road": "anna salai",
    "anna sl": "anna salai",
    "porur": "porur",
    "guindy": "guindy",
    "adyar": "adyar",
    "kotturpuram": "kotturpuram",
    "nungambakkam": "nungambakkam",
    "nungumbakkam": "nungambakkam",

    # ===== KOLKATA =====
    "salt lake": "salt lake city",
    "saltlake": "salt lake city",
    "saltlake sec 5": "salt lake sector 5",
    "saltlake sec v": "salt lake sector 5",
    "salt lake sec 1": "salt lake sector 1",
    "salt lake sec 2": "salt lake sector 2",
    "park st": "park street",
    "ballygunge": "ballygunge",
    "ballyganj": "ballygunge",
    "new town": "new town kolkata",
    "rajarhat": "rajarhat",
    "howrah": "howrah",

    # ===== PUNE =====
    "fc road": "fergusson college road",
    "f c road": "fergusson college road",
    "jm road": "jangli maharaj road",
    "j m road": "jangli maharaj road",
    "tilak rd": "tilak road",
    "sb road": "senapati bapat road",
    "s b road": "senapati bapat road",
    "kothrud": "kothrud",
    "koregaon park": "koregaon park",
    "kp": "koregaon park",
    "viman nagar": "viman nagar",
    "kalyani nagar": "kalyani nagar",
    "hadapsar": "hadapsar",
    "hadpsar": "hadapsar",
    "wakad": "wakad",
    "hinjewadi": "hinjewadi",
    "hinjawadi": "hinjewadi",
    "pcmc": "pimpri chinchwad",

    # ===== AHMEDABAD =====
    "sg highway": "sarkhej gandhinagar highway",
    "s g highway": "sarkhej gandhinagar highway",
    "sg hwy": "sarkhej gandhinagar highway",
    "cg road": "chimanlal girdharlal road",
    "c g road": "chimanlal girdharlal road",
    "ashram rd": "ashram road",
    "satellite": "satellite",
    "bopal": "bopal",
    "naranpura": "naranpura",
    "vastrapur": "vastrapur",
    "navrangpura": "navrangpura",

    # ===== KOCHI =====
    "mg road kochi": "mahatma gandhi road kochi",
    "panampilly nagar": "panampilly nagar",
    "kakkanad": "kakkanad",
    "edappally": "edappally",
    "edapally": "edappally",
    "vyttila": "vyttila",
    "kaloor": "kaloor",

    # ===== JAIPUR =====
    "c scheme": "civil lines scheme",
    "mi road": "mirza ismail road",
    "m i road": "mirza ismail road",
    "tonk rd": "tonk road",
    "malviya ngr": "malviya nagar",
    "vaishali ngr": "vaishali nagar",

    # ===== INDIA-WIDE GENERIC =====
    "nh": "national highway",
    "sh": "state highway",
    "exp": "expressway",
    "expy": "expressway",
    "hwy": "highway",
    "phse": "phase",
    "ph": "phase",
    "stg": "stage",
    "sec": "sector",
    "blk": "block",
    "ext": "extension",
    "extn": "extension",
    "lyt": "layout",
    "layt": "layout",
    "lyout": "layout",
    "ngr": "nagar",
    "nger": "nagar",
    "naagar": "nagar",
    "puram": "puram",
    "colny": "colony",
    "coloney": "colony",
    "coloy": "colony",
    "apt": "apartment",
    "aprt": "apartment",
    "apart": "apartment",
    "appt": "apartment",
    "apartments": "apartment",
    "bldg": "building",
    "twr": "tower",
    "twrs": "tower",
    "flr": "floor",
    "grnd flr": "ground floor",
    "gr flr": "ground floor",
    "gnd flr": "ground floor",
    "1st flr": "1st floor",
    "2nd flr": "2nd floor",
    "3rd flr": "3rd floor",
    "opp": "opposite",
    "opst": "opposite",
    "nr": "near",
    "behnd": "behind",
    "beh": "behind",
    "bh": "behind",
    "bsd": "beside",
    "bs stop": "bus stop",
    "bus stnd": "bus stand",
    "rly stn": "railway station",
    "rly station": "railway station",
    "rwy stn": "railway station",
    "rly": "railway",
    "metro stn": "metro station",
    "po": "post office",
    "p o": "post office",
    "ps": "police station",
    "p s": "police station",

    # ===== SAFE SINGLE-TOKEN CITY EXPANSIONS =====
    # (Only city names — never road suffix abbrevs to avoid "1 st floor" bugs.)
    "blr": "bangalore",
    "bglr": "bangalore",
    "bengaluru": "bangalore",
    # Common misspellings of bangalore/bengaluru -> canonical
    "bengaloore": "bangalore",
    "belngalore": "bangalore",
    "banglore": "bangalore",
    "banglaore": "bangalore",
    "bangaluru": "bangalore",
    "bengalore": "bangalore",
    "bengulore": "bangalore",
    "bengluru": "bangalore",
    "bengalooru": "bangalore",
    "bengalorw": "bangalore",
    "bangalorw": "bangalore",
    "bangloru": "bangalore",
    "banglur": "bangalore",
    "bnglore": "bangalore",
    "banaglore": "bangalore",
    # Common misspellings of mumbai
    "mumbi": "mumbai",
    "munbai": "mumbai",
    "mubai": "mumbai",
    # Common misspellings of hyderabad
    "hydrabad": "hyderabad",
    "hyderabd": "hyderabad",
    "hyederabad": "hyderabad",
    # Common misspellings of chennai
    "chennaii": "chennai",
    "channai": "chennai",
    # Common misspellings of kolkata
    "kolkota": "kolkata",
    "kolkatta": "kolkata",
    # near/nearby
    "neare": "near",
    "nearr": "near",
    "neraby": "nearby",
    # Common Bangalore locality misspellings
    "marenalli": "marenahalli",
    "marenahlli": "marenahalli",
    "marenhalli": "marenahalli",
    "marennahalli": "marenahalli",
    "jayanagara": "jayanagar",
    "jaynagar": "jayanagar",
    "jaynagara": "jayanagar",
    "koramangla": "koramangala",
    "koramngala": "koramangala",
    "indranagar": "indiranagar",
    "indra nagar": "indiranagar",
    "whitefiled": "whitefield",
    "whitfield": "whitefield",
    "marathali": "marathahalli",
    "marathhali": "marathahalli",
    "electroniccity": "electronic city",
    "electroniccty": "electronic city",
    "ecity": "electronic city",
    "hsrlayout": "hsr layout",
    "btmlayout": "btm layout",
    "jpnagar": "jp nagar",
    "rrnagar": "rajarajeshwari nagar",
    "yelahnka": "yelahanka",
    "hebbla": "hebbal",
    # Generic single-token block / road typos
    "bock": "block",
    "blok": "block",
    "blcok": "block",
    "crosss": "cross",
    "croos": "cross",
    "corss": "cross",
    "bombay": "mumbai",
    "calcutta": "kolkata",
    "madras": "chennai",
    "gurugram": "gurgaon",

    # ===== ABBREVIATED ROAD VARIANTS for famous roads =====
    # (Two-token forms so they don't conflict with ordinals like "1st".)
    "bg rd": "bannerghatta road",
    "b g rd": "bannerghatta road",
    "mg rd": "mahatma gandhi road",
    "m g rd": "mahatma gandhi road",
    "kg rd": "kasturba gandhi marg",
    "k g rd": "kasturba gandhi marg",
    "sv rd": "swami vivekananda road",
    "s v rd": "swami vivekananda road",
    "lbs rd": "lal bahadur shastri marg",
    "lbs marg rd": "lal bahadur shastri marg",
    "fc rd": "fergusson college road",
    "f c rd": "fergusson college road",
    "jm rd": "jangli maharaj road",
    "j m rd": "jangli maharaj road",
    "sb rd": "senapati bapat road",
    "s b rd": "senapati bapat road",
    "cg rd": "chimanlal girdharlal road",
    "c g rd": "chimanlal girdharlal road",
    "mi rd": "mirza ismail road",
    "m i rd": "mirza ismail road",
    "tonk rd": "tonk road",
    "ashram rd": "ashram road",
    "tilak rd": "tilak road",
    "mount rd": "anna salai",
    "ring rd": "ring road",
    "outer ring rd": "outer ring road",
    "inner ring rd": "inner ring road",
    "begur rd": "begur road",
    "bannergatta rd": "bannerghatta road",
    "cubbon rd": "cubbon road",
    "infantry rd": "infantry road",
    "brigade rd": "brigade road",
    "residency rd": "residency road",
    "park st": "park street",
    "commercial st": "commercial street",

    # ===== EXPLICIT COMPOSITIONS (avoid multi-pass self-expansion) =====
    # When abbreviated locality + abbreviated suffix appear together,
    # encode the full composition directly so single-pass canonicalizes correctly.
    "indra ngr": "indiranagar",
    "ind ngr": "indiranagar",
    "indra nagar": "indiranagar",
    "indira ngr": "indiranagar",
    "kalyan ngr": "kalyan nagar",
    "anna ngr": "anna nagar",
    "anna ngr w": "anna nagar west",
    "anna ngr e": "anna nagar east",
    "malviya ngr": "malviya nagar",
    "vaishali ngr": "vaishali nagar",
    "shivaji ngr": "shivaji nagar",
    "viman ngr": "viman nagar",
    "kalyani ngr": "kalyani nagar",
    "lajpat ngr": "lajpat nagar",
    "rajouri ngr": "rajouri nagar",
    "raj ngr": "raj nagar",
    "south ext": "south extension",
    "south ext 1": "south extension 1",
    "south ext 2": "south extension 2",
    "ip ext": "indraprastha extension",
    "rmv ext": "rmv extension",
    "rmv ext 1": "rmv extension 1st stage",
    "rmv ext 2": "rmv extension 2nd stage",
    "1 stg": "1st stage",
    "2 stg": "2nd stage",
    "3 stg": "3rd stage",
    "4 stg": "4th stage",
    "5 stg": "5th stage",
    "6 stg": "6th stage",
    "7 stg": "7th stage",
    "8 stg": "8th stage",
    "1 phse": "1st phase",
    "2 phse": "2nd phase",
    "3 phse": "3rd phase",
    "4 phse": "4th phase",
    "1 ph": "1st phase",
    "2 ph": "2nd phase",
    "3 ph": "3rd phase",
    "4 ph": "4th phase",
    "1 blk": "1st block",
    "2 blk": "2nd block",
    "3 blk": "3rd block",
    "4 blk": "4th block",
    "5 blk": "5th block",
    "6 blk": "6th block",
    "7 blk": "7th block",
    "8 blk": "8th block",
    "9 blk": "9th block",
    "1 sec": "sector 1",
    "2 sec": "sector 2",
    "3 sec": "sector 3",
}


# ---------------------------------------------------------------------------
# Build the phrase trie for greedy longest-match replacement
# ---------------------------------------------------------------------------
def _build_phrase_index() -> dict[int, dict[tuple[str, ...], str]]:
    """Group aliases by token-count for fast n-gram scanning."""
    by_len: dict[int, dict[tuple[str, ...], str]] = {}
    for alias, canon in LOCALITY_ALIASES.items():
        toks = tuple(alias.split())
        if not toks:
            continue
        by_len.setdefault(len(toks), {})[toks] = canon
    return by_len


_PHRASE_INDEX = _build_phrase_index()
_MAX_PHRASE_LEN = max(_PHRASE_INDEX.keys()) if _PHRASE_INDEX else 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _single_pass(text: str) -> str:
    tokens = text.split()
    if not tokens:
        return text
    out: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        matched = False
        max_window = min(_MAX_PHRASE_LEN, n - i)
        for window in range(max_window, 1, -1):
            phrase = tuple(tokens[i:i + window])
            canon = _PHRASE_INDEX.get(window, {}).get(phrase)
            if canon is not None:
                out.extend(canon.split())
                i += window
                matched = True
                break
        if matched:
            continue
        single = (tokens[i],)
        canon = _PHRASE_INDEX.get(1, {}).get(single)
        if canon is not None:
            canon_toks = canon.split()
            # Prevent duplication when the expansion starts with the key token
            # and the input already contains the tail of the expansion.
            # e.g. alias "btm" -> "btm layout" on input "btm layout" would
            # produce "btm layout layout" without this guard.
            if canon_toks and canon_toks[0] == tokens[i]:
                tail = canon_toks[1:]
                tail_len = len(tail)
                if i + tail_len < n and tokens[i + 1 : i + 1 + tail_len] == tail:
                    out.append(tokens[i])   # just keep the key token
                    i += 1
                    continue
            out.extend(canon_toks)
        else:
            out.append(tokens[i])
        i += 1
    return " ".join(out)


@lru_cache(maxsize=8192)
def canonicalize_localities(text: str) -> str:
    """Replace alias phrases with their canonical forms (single greedy pass).

    Compositions like "indra ngr" -> "indiranagar" must be encoded explicitly
    in the dictionary to avoid self-expansion cycles (e.g. btm -> btm layout
    would re-expand "btm" infinitely under multi-pass).
    """
    if not text:
        return text
    return _single_pass(text)


def canonicalize_many(texts: Iterable[str]) -> list[str]:
    """Vectorized convenience wrapper."""
    return [canonicalize_localities(t) for t in texts]


def alias_count() -> int:
    return len(LOCALITY_ALIASES)


# ---------------------------------------------------------------------------
# Self-test (run: python -m fuzzy_engine.v2.locality_aliases)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cases = [
        ("5 1st main rr nagar bangalore 560098",
         "5 1st main rajarajeshwari nagar bangalore 560098"),
        ("flat 12 bg road btm 2nd stage bangalore",
         "flat 12 bannerghatta road btm layout 2nd stage bangalore"),
        ("plot 7 bkc mumbai 400051",
         "plot 7 bandra kurla complex mumbai 400051"),
        ("t nagar chennai 600017",
         "thyagaraya nagar chennai 600017"),
        ("kphb colony hyderabad 500072",
         "kukatpally housing board colony hyderabad 500072"),
        ("100ft road indra ngr bangalore",
         "100 feet road indiranagar bangalore"),
        ("indra ngr blr",
         "indiranagar bangalore"),
        ("12 bg rd btm bangalore",
         "12 bannerghatta road btm layout bangalore"),
        ("shop 3 lbs marg ghatkopar w mumbai",
         "shop 3 lal bahadur shastri marg ghatkopar west mumbai"),
        ("1 st floor mg rd",
         "1 st floor mahatma gandhi road"),
    ]
    print(f"Aliases loaded: {alias_count()}  | max phrase len: {_MAX_PHRASE_LEN}\n")
    pass_, fail_ = 0, 0
    for src, expected in cases:
        got = canonicalize_localities(src)
        ok = got == expected
        marker = "OK" if ok else "FAIL"
        if ok:
            pass_ += 1
        else:
            fail_ += 1
        print(f"[{marker}] {src!r}")
        if not ok:
            print(f"        got:      {got!r}")
            print(f"        expected: {expected!r}")
    print(f"\n{pass_} pass, {fail_} fail")
