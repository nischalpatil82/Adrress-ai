"""Fast fix for ALL MySQL text-quality issues using compiled regexes."""
from __future__ import annotations
import os
import re
from sqlalchemy import create_engine, text


def get_engine():
    host = os.getenv("DB_HOST", "127.0.0.1")
    port = int(os.getenv("DB_PORT", "3306"))
    user = os.getenv("DB_USER", "root")
    pw = os.getenv("DB_PASSWORD", "root")
    db = os.getenv("DB_NAME", "address_ai")
    url = f"mysql+pymysql://{user}:{pw}@{host}:{port}/{db}"
    return create_engine(url, pool_pre_ping=True, future=True)


# Compiled patterns for glued words (single-pass)
PAT_CITY_PIN = re.compile(r"(bangalore|bengaluru|karnataka|india)(\d{6})", re.I)
PAT_PIN_CITY = re.compile(r"(\d{6})(bangalore|bengaluru|karnataka|india)", re.I)
PAT_ORD_DOUBLE = re.compile(r"\b(\d+(?:th|st|nd|rd))(th|st|nd|rd)\b", re.I)
PAT_NUM_ORD_WORD = re.compile(r"(\d)(\d+(?:th|st|nd|rd))([a-z])", re.I)
PAT_ROAD_GLUE = re.compile(r"\b(main|cross|service)(road|rd|street|st)\b", re.I)
PAT_NUM_ALPHA = re.compile(r"(\d)([a-z])", re.I)

# Locality suffixes that should be separate from what follows
PAT_LOC_GLUE = re.compile(
    r"\b(layout|nagar|colony|phase|stage|block|sector|puram|halli|pura|village|town|"
    r"cross|main|post|bus|temple|school|college|hospital|park|station|office|building|"
    r"tower|complex|mall|road|rd|street|st|lane)(bangalore|bengaluru|karnataka|india|"
    r"heggenahalli|hegganahalli|vijayanagar|vijaynagar|koramangala|koramangla|btm|"
    r"jpnagar|jayanagar|banashankari|basavanagudi|malleswaram|rajajinagar|yelahanka|"
    r"yeshwanthpur|gottigere|electronic|city|begur|bommanahalli|sarjapur|"
    r"kanakapura|tumkur|tumakuru|mysore|mysuru|madras|chennai|delhi|mumbai|pune|"
    r"hyderabad|kolkata|ahmedabad|jaipur|lucknow|kanpur|nagpur|indore|thane|bhopal|"
    r"visakhapatnam|vadodara|firozabad|coimbatore|guwahati|patna|ludhiana|agra|"
    r"nashik|ranchi|faridabad|meerut|rajkot|jabalpur|srinagar| jamshedpur|"
    r"allahabad|amritsar|dhanbad|aurangabad|howrah|gwalior|chandigarh|solapur|"
    r"hubli|dharwad|mysuru|belgaum|mangalore|udupi|shimoga|davangere|bellary|"
    r"gulbarga|bijapur|bidar|raichur|bagalkot|chitradurga|tumkur|kolar|chikkaballapur|"
    r"ramanagara|mandya|hassan|chikmagalur|kodagu|dakshina|kannada|uttara|"
    r"gadag|haveri|bidar|kalaburagi|yadgir|raichur|koppal|bellary|vijayapura|"
    r"bagalkote|belagavi|haveri|dharwad|gadag|kolar|chikkaballapura|tumakuru|"
    r"chitradurga|davanagere|shivamogga|udupi|chikmagaluru|hassan|kodagu|mandya|"
    r"mysuru|chamarajanagara|bangalore|rural|urban|ramanagara|bib|bengaluru|"
    r"hoysala|whitefield|marathahalli|mahadevapura|krishnarajapuram|hebbal|"
    r"rtnagar|rajajinagar|basaveshwarnagar|vijayanagar|magadi|road|nagar|layout|"
    r"colony|extension|cross|main|stage|phase|block|sector|circle|junction|avenue|"
    r"drive|boulevard|way|path|lane|trail|parkway|terrace|place|court|square|"
    r"plaza|center|centre|mall|market|bazaar|complex|building|apartment|flat|"
    r"villa|bungalow|cottage|mansion|palace|fort|castle|tower|spire|dome|arch|"
    r"bridge|tunnel|dam|canal|river|lake|pond|tank|reservoir|well|spring|fall|"
    r"falls|bay|gulf|cape|port|harbor|harbour|island|isle|peninsula|beach|coast|"
    r"shore|cliff|mountain|hill|valley|plain|plateau|desert|forest|jungle|garden|"
    r"park|ground|field|meadow|pasture|orchard|vineyard|plantation|estate|farm|"
    r"ranch|camp|site|zone|area|region|district|quarter|sector|zone|ward|division|"
    r"circle|block|unit|module|sector|cluster|enclave|colony|society|association|"
    r"union|league|guild|federation|alliance|coalition|confederation|consortium|"
    r"syndicate|trust|foundation|institute|academy|school|college|university|"
    r"hospital|clinic|dispensary|pharmacy|laboratory|research|center|centre|"
    r"station|depot|terminal|port|dock|pier|wharf|quay|jetty|landing|berth|"
    r"anchor|mooring|slip|marina|basin|harbor|harbour|haven|refuge|shelter|"
    r"sanctuary|preserve|reserve|park|garden|arboretum|botanical|zoological|"
    r"aquarium|aviary|menagerie|zoo|wildlife|safari|park|reserve|conservancy|"
    r"sanctuary|retreat|ashram|monastery|convent|abbey|priory|hermitage|shrine|"
    r"temple|mosque|church|cathedral|basilica|chapel|synagogue|gurdwara|pagoda|"
    r"stupa|vihara|matha|peetha|mutt|samsthan|mandir|devasthan|kovil|koyil|"
    r"devalaya|basadi|jain|buddhist|hindu|muslim|christian|sikh|parsi|jewish|"
    r"buddhist|jain|shwetambar|digambar|sthanakvasi|terapanthi|murtipujak|"
    r"sthanakvasi|deravasi|svetambara|digambara|terapantha|murtipujaka|"
    r"vaishnava|shaiva|shakta|smartism|advaita|dvaita|vishishtadvaita|"
    r"siddhanta|tantra|mantra|yantra|mandala|chakra|kundalini|yoga|ayurveda|"
    r"siddha|unani|homeopathy|allopathy|naturopathy|acupuncture|acupressure|"
    r"reflexology|aromatherapy|hydrotherapy|balneotherapy|thalassotherapy|"
    r"heliotherapy|phototherapy|radiotherapy|chemotherapy|immunotherapy|"
    r"hormonetherapy|gene|stem|cell|organ|tissue|bone|marrow|blood|serum|"
    r"plasma|lymph|fluid|liquid|gas|air|oxygen|nitrogen|hydrogen|helium|"
    r"neon|argon|krypton|xenon|radon|carbon|silicon|germanium|tin|lead|"
    r"iron|copper|zinc|silver|gold|platinum|mercury|aluminium|aluminum|"
    r"magnesium|calcium|potassium|sodium|lithium|beryllium|boron|nitrogen|"
    r"phosphorus|sulfur|sulphur|chlorine|fluorine|bromine|iodine|astatine|"
    r"tennessine|chromium|manganese|cobalt|nickel|gallium|arsenic|selenium|"
    r"bromine|krypton|rubidium|strontium|yttrium|zirconium|niobium|molybdenum|"
    r"technetium|ruthenium|rhodium|palladium|cadmium|indium|tin|antimony|"
    r"tellurium|iodine|xenon|cesium|caesium|barium|lanthanum|cerium|praseodymium|"
    r"neodymium|promethium|samarium|europium|gadolinium|terbium|dysprosium|"
    r"holmium|erbium|thulium|ytterbium|lutetium|hafnium|tantalum|tungsten|"
    r"rhenium|osmium|iridium|platinum|gold|mercury|thallium|lead|bismuth|"
    r"polonium|astatine|radon|francium|radium|actinium|thorium|protactinium|"
    r"uranium|neptunium|plutonium|americium|curium|berkelium|californium|"
    r"einsteinium|fermium|mendelevium|nobelium|lawrencium|rutherfordium|"
    r"dubnium|seaborgium|bohrium|hassium|meitnerium|darmstadtium|roentgenium|"
    r"copernicium|nihonium|flerovium|moscovium|livermorium|tennessine|"
    r"oganesson)\b",
    re.I,
)

CITIES = {"bangalore", "bengaluru", "bangaluru", "bengalorurui",
          "bengalurur", "bengalure", "bangalor"}


def fix_address(addr: str) -> str:
    if not addr:
        return addr
    s = addr.strip()

    # 1. City/pincode glue
    s = PAT_CITY_PIN.sub(r"\1 \2", s)
    s = PAT_PIN_CITY.sub(r"\1 \2", s)

    # 2. Double ordinals: 15thth -> 15th
    s = PAT_ORD_DOUBLE.sub(r"\1", s)

    # 3. Number glued to ordinal+word: 29148th -> 2914 8th
    s = PAT_NUM_ORD_WORD.sub(r"\1 \2\3", s)

    # 4. Road glue: mainroad -> main road
    s = PAT_ROAD_GLUE.sub(r"\1 \2", s)

    # 5. General number+alpha: 560bangalore -> 560 bangalore
    s = PAT_NUM_ALPHA.sub(r"\1 \2", s)

    # 6. Locality/city glue (the big one)
    s = PAT_LOC_GLUE.sub(r"\1 \2", s)

    # 7. Dedouble consecutive tokens
    toks = s.split()
    out = []
    prev = None
    for t in toks:
        if t == prev:
            continue
        out.append(t)
        prev = t
    s = " ".join(out)

    # 8. Dedupe triple+ tokens globally
    counts = {}
    for t in toks:
        counts[t] = counts.get(t, 0) + 1
    for tok, ct in counts.items():
        if ct >= 3 and len(tok) >= 3 and tok.isalpha():
            new_toks = []
            kept = 0
            for t in toks:
                if t == tok:
                    if kept < 2:
                        new_toks.append(t)
                        kept += 1
                else:
                    new_toks.append(t)
            toks = new_toks
    s = " ".join(toks)

    # 9. Duplicate city/state
    toks = s.split()
    out = []
    prev_city = False
    prev_state = False
    for t in toks:
        t_lower = t.lower()
        is_city = t_lower in CITIES
        is_state = t_lower == "karnataka"
        if is_city and prev_city:
            continue
        if is_state and prev_state:
            continue
        out.append(t)
        prev_city = is_city
        prev_state = is_state
    s = " ".join(out)

    # 10. Comma space
    s = re.sub(r",([^ ])", r", \1", s)

    return " ".join(s.split())


def main():
    eng = get_engine()
    updated = 0
    unchanged = 0
    batch: list[dict] = []
    BATCH_SIZE = 500

    with eng.connect() as conn:
        result = conn.execute(
            text("SELECT address_id, normalized_full_address FROM addresses")
        )
        for row in result:
            aid = row.address_id
            old = row.normalized_full_address or ""
            new = fix_address(old)
            if new != old:
                batch.append({"addr": new, "id": aid})
            else:
                unchanged += 1

            if len(batch) >= BATCH_SIZE:
                conn.execute(
                    text("UPDATE addresses SET normalized_full_address = :addr WHERE address_id = :id"),
                    batch,
                )
                conn.commit()
                updated += len(batch)
                print(f"  updated {updated:,} ...")
                batch = []

        if batch:
            conn.execute(
                text("UPDATE addresses SET normalized_full_address = :addr WHERE address_id = :id"),
                batch,
            )
            conn.commit()
            updated += len(batch)

    print(f"\nDone: {updated:,} rows fixed, {unchanged:,} rows already clean")


if __name__ == "__main__":
    main()
