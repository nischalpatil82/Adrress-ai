"""Comprehensive audit of all MySQL address text quality issues."""
from __future__ import annotations
import os
import re
from collections import Counter
from sqlalchemy import create_engine, text


def get_engine():
    host = os.getenv("DB_HOST", "127.0.0.1")
    port = int(os.getenv("DB_PORT", "3306"))
    user = os.getenv("DB_USER", "root")
    pw = os.getenv("DB_PASSWORD", "root")
    db = os.getenv("DB_NAME", "address_ai")
    url = f"mysql+pymysql://{user}:{pw}@{host}:{port}/{db}"
    return create_engine(url, pool_pre_ping=True, future=True)


def find_issues(addr: str) -> list[str]:
    """Return list of issue labels found in this address."""
    issues = []
    s = addr.lower()
    toks = s.split()

    # 1. Consecutive duplicate tokens
    for i in range(len(toks) - 1):
        if toks[i] == toks[i + 1] and len(toks[i]) >= 3:
            issues.append(f"dup_token:{toks[i]}")
            break

    # 2. Glued words (missing space between meaningful words)
    # Whitelist: known compound locality names that are legitimate
    KNOWN_COMPOUND = {
        "nagarabhavi", "banashankari", "basavanagudi", "jayanagar", "vijayanagar",
        "vijaynagar", "koramangala", "koramangla", "malleswaram", "rajajinagar",
        "yelahanka", "yeshwanthpur", "hebbal", "hosakerehalli", "kengeri",
        "electronic", "whitefield", "marathahalli", "mahadevapura",
        "krishnarajapuram", "hoskote", "devanahalli", "doddaballapur",
        "nelamangala", "ramanagara", "kanakapura", "channapatna", "maddur",
        "mandya", "mysuru", "nanjangud", "chamarajanagar", "kollegal",
        "hassan", "sakleshpura", "belur", "chikmagalur", "kadur", "koppa",
        "mudigere", "sringeri", "karkala", "udupi", "kundapura", "bhatkal",
        "kumta", "honnavar", "karwar", "ankola", "sirsi", "siddapur",
        "gadag", "ron", "nargund", "saundatti", "belagavi", "gokak",
        "chikodi", "khanapur", "bailhongal", "nippani", "ramdurg", "mudhol",
        "terdal", "jamkhandi", "badami", "bagalkote", "bilagi", "bijapur",
        "sindagi", "yadgir", "shahapur", "wadi", "kalaburagi", "afzalpur",
        "chincholi", "raichur", "manvi", "sindhanur", "koppal", "gangavathi",
        "kustagi", "hadagali", "harapanahalli", "davanagere", "honnali",
        "channagiri", "jagalur", "sagara", "shivamogga", "thirthahalli",
        "hosanagara", "soraba", "bhadravati", "shikaripura", "birur",
        "tarikere", "ajjampura", "alur", "channarayapatna", "holenarsipur",
        "arkalgud", "shravanabelagola", "virajpet", "madikeri", "somwarpet",
        "gundlupet", "puttur", "belthangady", "mangaluru", "bantwal",
        "vitla", "ujire", "kadaba", "sullia", "mundgod", "supa", "hanagal",
        "shiggaon", "ranebennur", "haveri", "savanur", "shirahatti",
        "lakshmeshwar", "attibele", "bannerghatta", "begur", "bommanahalli",
        "chandapura", "chikkalasandra", "dommasandra", "gottigere", "hulimavu",
        "jigani", "kaggalipura", "konanakunte", "madiwala", "parappana",
        "singasandra", "subramanyapura", "talaghattapura", "thirumenahalli",
        "varthur", "vittasandra", "yelenahalli", "akshayanagar", "ambalipura",
        "arekere", "bellandur", "beniganahalli", "bennigana", "bilekahalli",
        "biradanahalli", "byrasandra", "coxtown", "devarajeevanahalli",
        "gavipuram", "girinagar", "kamakshipalya", "kamasipalya", "kempapura",
        "kodigehalli", "kolathur", "kurubarahalli", "lingarajapuram",
        "nagarbhavi", "nagasandra", "nandini", "pattabhiramanagar",
        "pulakeshinagar", "puttenahalli", "rajarajeshwarinagar", "rampura",
        "saneguruvanahalli", "shakthiganapathi", "shankarmutt", "siddapura",
        "sivanachetty", "sonnappanahalli", "sriramamandira", "suddagunte",
        "tavarekere", "thigalarapalya", "tyagarajanagar", "ulsoor",
        "vasanthanagar", "vidyaranyapura", "vishwanathnagenahalli",
        "vishveshwarayya", "vv puram", "viveknagar", "wilson",
    }
    # Check if any "glued" match is actually a known compound name
    glued = [
        (r"(stage|phase|block|sector|layout|colony|nagar)(bangalore|bengaluru|karnataka)", "glued_city"),
        (r"(layout|nagar|colony|puram|halli)([a-z]{5,})", "glued_locality"),
        (r"(main|cross|service)(road|rd|street|st|lane)", "glued_road"),
        (r"(bangalore|bengaluru)(karnataka|5600\d{2})", "glued_city_pin"),
        (r"(\d{2,})(th|st|nd|rd)(main|cross|road|stage|phase|block)", "glued_num_ordinal"),
        (r"(\d+)(main|cross|road|stage|phase|block|layout|nagar)", "glued_num_word"),
    ]
    for pat, label in glued:
        m = re.search(pat, s)
        if m:
            # Skip if the matched text contains a known compound name
            matched_text = m.group(0)
            if any(comp in matched_text for comp in KNOWN_COMPOUND):
                continue
            issues.append(label)
            break

    # 3. Double city / state — ONLY flag when EXACT SAME token is consecutive
    for i in range(len(toks) - 1):
        if toks[i] == toks[i + 1] and toks[i] in {"bangalore", "bengaluru", "bangaluru"}:
            issues.append("dup_city")
            break
    for i in range(len(toks) - 1):
        if toks[i] == toks[i + 1] == "karnataka":
            issues.append("dup_state")
            break

    # 4. Pincode glued to word (bangalore560076)
    if re.search(r"[a-z]\d{6}|\d{6}[a-z]", s):
        issues.append("glued_pincode")

    # 5. Trailing punctuation or ordinals like "thth", "ndst"
    if re.search(r"\b(th|st|nd|rd){2,}\b", s):
        issues.append("bad_ordinal")

    # 6. Missing space after comma
    if "," in addr and re.search(r",[^ ]", addr):
        issues.append("comma_no_space")

    # 7. Triple+ repeated substrings ( severe hallucination )
    for tok in set(toks):
        if tok.isalpha() and len(tok) >= 4 and toks.count(tok) >= 3:
            issues.append(f"triple_token:{tok}")
            break

    return issues


def main():
    eng = get_engine()
    issue_counter: Counter = Counter()
    sample_issues: dict[str, list[str]] = {}
    total = 0

    with eng.connect() as conn:
        result = conn.execute(text("SELECT normalized_full_address FROM addresses"))
        for row in result:
            total += 1
            addr = row.normalized_full_address or ""
            issues = find_issues(addr)
            for iss in issues:
                issue_counter[iss] += 1
                if iss not in sample_issues:
                    sample_issues[iss] = []
                if len(sample_issues[iss]) < 3:
                    sample_issues[iss].append(addr[:100])

    print(f"Total rows scanned: {total:,}")
    print(f"\nIssue counts:")
    for iss, ct in issue_counter.most_common():
        print(f"  {ct:>6,}  {iss}")
        for s in sample_issues.get(iss, [])[:3]:
            print(f"         e.g. {s}")

    print(f"\nRows with ANY issue: {sum(issue_counter.values()):,}")


if __name__ == "__main__":
    main()
