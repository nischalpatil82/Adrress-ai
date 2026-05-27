"""Targeted fix for remaining real errors (excludes false-positive 'bangalore north')."""
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


# Only fix specific glued patterns remaining after first pass
PAT_LOC_GLUE2 = re.compile(
    r"\b(raghuvanahalli|lingarajapuram|bommanahalli|heggenahalli|hegganahalli|"
    r"kanakapura|thavarekere|jayaprakash|narayan|govindapura|vijayanagar|"
    r"koramangala|koramangla|hosur|sarjapur|whitefield|marathahalli|"
    r"mahadevapura|krishnarajapuram|hebbal|rajajinagar|basaveshwarnagar|"
    r"magadi|begur|bommanahalli| electronic|gottigere|electronic|"
    r"banashankari|basavanagudi|malleswaram|yelahanka|yeshwanthpur|"
    r"jpnagar|jayanagar|btm|hosakerehalli|kengeri|nagasandra|"
    r"peenya|yeswanthpur|mathikere|rajajinagar|basaveshwarnagar|"
    r"vijayanagar|magadi|hoskote|devanahalli|doddaballapur|nelamangala|"
    r"ramanagara|kanakapura|channapatna|maddur|mandya|pandavapura|"
    r"srirangapatna|mysuru|nanjangud|chamarajanagar|kollegal|"
    r"hanur|satyamangala|bannur|t narasipura|krishnarajanagara|"
    r"hunsur|periyapatna|k r nagar|h d kote|n r pura|kushalnagar|"
    r"virajpet|madikeri|somwarpet|sulya|puttur|belthangady|mangaluru|"
    r"bantwal|ujire|kadaba|sullia|vitla|paudur|karwar|kumta|"
    r"honnavar|bhatkal|sirsi|siddapur|yellapur|haliyal|joida|"
    r"dandeli|ankola|mundgod|supa|hanagal|shiggaon|ranebennur|"
    r"haveri|savanur|shirahatti|gadag|mundargi|ron|nargund|"
    r"naragund|lakshmeshwar|saundatti|belagavi|athani|chikodi|"
    r"gokak|hukkeri|raybag|khanapur|bailhongal|kittur|nippani|"
    r"savadatti|ramdurg|mudhol|terdal|jamkhandi|rabakavi|banhatti|"
    r"badami|bagalkote|bilagi|hungund|muddebihal|talon|bijapur|"
    r"indibasavakalyan|sindagi|devarhipparagi|chadchan|alamela|"
    r"shorapur|shahapur|yadgir|gurmitkal|shahbad|wadi|chittapur|"
    r"sedam|kalaburagi|afzalpur|chincholi|kamalapur|alanda|"
    r"j evargi|yanagunti|krishna|raichur|manvi|sindhanur|lingasugur|"
    r"devadurga|lakshettipet|koppal|gangavathi|kustagi|yelbarga|"
    r"kanakagiri|karatagi|sindhagi|hadagali|hagaribommanahalli|"
    r"harapanahalli|davanagere|honnali|channagiri|honnali|jagalur|"
    r"nyamati|sagara|shivamogga|thirthahalli|hosanagara|soraba|"
    r"bhadravati|holehonnur|anavatti|shikaripura|birur|tarikere|"
    r"ajjampura|kadamba|narasimharajapura|hassan|alur|belur|"
    r"sakleshpura|channarayapatna|holenarsipur|arkalgud|shravanabelagola|"
    r"chikmagalur|kadur|koppa|mudigere|n r pura|tarikere|ajjampur|"
    r"sringeri|karkala|udupi|baindur|bhatkal|kundapura|hebri|"
    r"brahmavar|kapu|byndoor|kumta|honnavar|karwar|ankola|"
    r"siddapur|sirsi|yellapur|joida|dandeli|haliyal|mundgod|"
    r"supa|hanagal|shiggaon|haveri|ranebennur|hanagal|shirahatti|"
    r"gadag|mundargi|ron|nargund|saundatti|belagavi|athani|"
    r"chikodi|gokak|hukkeri|raybag|khanapur|bailhongal|kittur|"
    r"nippani|savadatti|ramdurg|mudhol|terdal|jamkhandi|rabakavi|"
    r"banhatti|badami|bagalkote|bilagi|hungund|muddebihal|talon|"
    r"bijapur|indibasavakalyan|sindagi|devarhipparagi|chadchan|"
    r"alamela|shorapur|shahapur|yadgir|gurmitkal|shahbad|wadi|"
    r"chittapur|sedam|kalaburagi|afzalpur|chincholi|kamalapur|"
    r"alanda|j evargi|yanagunti|krishna|raichur|manvi|sindhanur|"
    r"lingasugur|devadurga|lakshettipet|koppal|gangavathi|kustagi|"
    r"yelbarga|kanakagiri|karatagi|sindhagi|hadagali|hagaribommanahalli|"
    r"harapanahalli|davanagere|honnali|channagiri|honnali|jagalur|"
    r"nyamati|sagara|shivamogga|thirthahalli|hosanagara|soraba|"
    r"bhadravati|holehonnur|anavatti|shikaripura|birur|tarikere|"
    r"ajjampura|kadamba|narasimharajapura|hassan|alur|belur|"
    r"sakleshpura|channarayapatna|holenarsipur|arkalgud|shravanabelagola|"
    r"chikmagalur|kadur|koppa|mudigere|n r pura|sringeri|karkala|"
    r"udupi|baindur|bhatkal|kundapura|hebri|brahmavar|kapu|byndoor|"
    r"hattiangadi|koteshwar|brahmavara|kundapura|hebri|baindur|"
    r"karkala|moodabidri|kaup|mulki|surathkal|mangaluru|bantwal|"
    r"vitla|paudur|belthangady|puttur|sulya|kadaba|ujire|"
    r"hunsur|periyapatna|k r nagar|h d kote|n r pura|kushalnagar|"
    r"virajpet|madikeri|somwarpet|sulya|gundlupet|chamarajanagara|"
    r"kollegal|hanur|satyamangala|bannur|t narasipura|krishnarajanagara|"
    r"nanjangud|chamarajanagar|k r pet|maddur|mandya|pandavapura|"
    r"srirangapatna|bannur|t narasipura|nanjangud|mysuru|h d kote|"
    r"n r pura|kushalnagar|virajpet|madikeri|somwarpet|sulya|"
    r"gundlupet|chamarajanagara|hanur|satyamangala|kollegal|"
    r"ramanagara|kanakapura|channapatna|maddur|mandya|pandavapura|"
    r"srirangapatna|bannur|t narasipura|krishnarajanagara|hunsur|"
    r"periyapatna|k r nagar|h d kote|n r pura|kushalnagar|virajpet|"
    r"madikeri|somwarpet|sulya|puttur|belthangady|mangaluru|bantwal|"
    r"ujire|kadaba|sullia|vitla|paudur|karwar|kumta|honnavar|bhatkal|"
    r"sirsi|siddapur|yellapur|haliyal|joida|dandeli|ankola|mundgod|"
    r"supa|hanagal|shiggaon|ranebennur|haveri|savanur|shirahatti|"
    r"gadag|mundargi|ron|nargund|naragund|lakshmeshwar|saundatti|"
    r"belagavi|athani|chikodi|gokak|hukkeri|raybag|khanapur|bailhongal|"
    r"kittur|nippani|savadatti|ramdurg|mudhol|terdal|jamkhandi|"
    r"rabakavi|banhatti|badami|bagalkote|bilagi|hungund|muddebihal|"
    r"talon|bijapur|indibasavakalyan|sindagi|devarhipparagi|chadchan|"
    r"alamela|shorapur|shahapur|yadgir|gurmitkal|shahbad|wadi|chittapur|"
    r"sedam|kalaburagi|afzalpur|chincholi|kamalapur|alanda|j evargi|"
    r"yanagunti|krishna|raichur|manvi|sindhanur|lingasugur|devadurga|"
    r"lakshettipet|koppal|gangavathi|kustagi|yelbarga|kanakagiri|karatagi|"
    r"sindhagi|hadagali|hagaribommanahalli|harapanahalli|davanagere|"
    r"honnali|channagiri|jagalur|nyamati|sagara|shivamogga|thirthahalli|"
    r"hosanagara|soraba|bhadravati|holehonnur|anavatti|shikaripura|birur|"
    r"tarikere|ajjampura|kadamba|narasimharajapura|hassan|alur|belur|"
    r"sakleshpura|channarayapatna|holenarsipur|arkalgud|shravanabelagola|"
    r"chikmagalur|kadur|koppa|mudigere|n r pura|sringeri|karkala|udupi|"
    r"baindur|bhatkal|kundapura|hebri|brahmavar|kapu|byndoor|hattiangadi|"
    r"koteshwar|moodabidri|kaup|mulki|surathkal|bantwal|vitla|paudur|"
    r"belthangady|puttur|sulya|kadaba|ujire|hunsur|periyapatna|k r nagar|"
    r"h d kote|kushalnagar|virajpet|madikeri|somwarpet|gundlupet|"
    r"chamarajanagara|hanur|satyamangala|bannur|t narasipura|nanjangud|"
    r"k r pet|maddur|mandya|pandavapura|srirangapatna|hosakerehalli|"
    r"kengeri|nagasandra|peenya|yeswanthpur|mathikere|basaveshwarnagar|"
    r"rajajinagar|malleswaram|yelahanka|yeshwanthpur|hebbal|jayanagar|"
    r"jpnagar|banashankari|basavanagudi|kumaraswamy|vv puram|hanumanthanagar|"
    r"srinagar|chamarajpet|basavanagudi|gandhibazaar|siddapura|vimanapura|"
    r"halasuru|coxtown|fraser|richmond|shantinagar|austin|neelasandra|"
    r"binnypet|chickpet|cottonpet|dasarahalli|devarajeevanahalli|"
    r"gavipuram|girinagar|hosakerehalli|jagajeevanram|kamakshipalya|"
    r"kamasipalya|kamalanagar|kempapura|kengeri|kodigehalli|kolathur|"
    r"kurubarahalli|lingarajapuram|mahadevapura|marathahalli|mathikere|"
    r"milkcolony|muneswara|nagarbhavi|nagasandra|nandini|pattabhiramanagar|"
    r"pulakeshinagar|puttenahalli|rajarajeshwarinagar|rampura|saneguruvanahalli|"
    r"shakthiganapathi|shankarmutt|shanthinagar|shivajinagar|siddapura|"
    r"sivanachetty|sonnappanahalli|sriramamandira|suddagunte|tavarekere|"
    r"thigalarapalya|tyagarajanagar|ulsoor|vasanthanagar|vidyaranyapura|"
    r"vijayanagar|vishwanathnagenahalli|vishveshwarayya|vv puram|viveknagar|"
    r"wilson|yelahanka|yeshwanthpur|agara|anekal|attibele|bannerghatta|"
    r"begur|bommanahalli|carmelaram|chandapura|chikkalasandra|dommasandra|"
    r"electronic|gottigere|hulimavu|himagiri|hsr|itpl|jigani|kaggalipura|"
    r"konanakunte|madiwala|parappana|raghuvanahalli|singasandra|subramanyapura|"
    r"talaghattapura|thirumenahalli|varthur|vittasandra|yelenahalli|"
    r"akshayanagar|ambalipura|arekere|araka|attibele|banashankari|begur|"
    r"bellandur|beniganahalli|bennigana|bilekahalli|biradanahalli|bommanahalli|"
    r"btm|byrasandra|car street|carmelaram|chamarajpet|chandapura|chikkaballapur|"
    r"chikkalasandra|chikkasandra|choodasandra|coxtown|c v raman|devanahalli|"
    r"doddanekkundi|domlur|dommasandra| Electronics|frazertown|gandhibazaar|"
    r"garudacharpalya|gottigere|halasuru|hbr|hebbal|hennur|hongasandra|"
    r"hoodi|horamavu|hosakerehalli|hoskote|hrbr|hsr|hulimavu|hulimangala|"
    r"indiranagar|itpl|jakkasandra|jayamahal|jayanagar|jeevanbimanagar|"
    r"jigani|jp nagar|judicial|kacharakanahalli|kaggadasapura|kaggalipura|"
    r"kalena|kammanahalli|kempapura|kengeri|konanakunte|koramangala|kothanur|"
    r"kr puram|kudlu|kumaraswamy|langford|mahadevapura|madiwala|malleshpalya|"
    r"marathahalli|mathikere|mico|millers|munnekolalu|murughamutt|muthanallur|"
    r"naagarabhaavi|nagarbhavi|nagasandra|nandini|nayandahalli|neelasandra|"
    r"nelamangala|new thippasandra|old airport|old madras|omkar|parappana|"
    r"pattandur|prahlad|raghuvanahalli|rajarajeshwari|ramamurthynagar|"
    r"sadashivanagar|sahakara|saneguruvanahalli|sarakki|shanthinagar|shivajinagar|"
    r"siddapura|singasandra|sivanachetty|sonnappanahalli|sriramamandira|"
    r"st thomas|subramanyapura|suddagunte|tavarekere|tc palya|thigalarapalya|"
    r"thippasandra|thirumenahalli|tilaknagar|ulsoor|vartur|vasanthanagar|"
    r"vignananagar|vijayanagar|viveknagar|vv puram|wilson|yelahanka|"
    r"yeshwanthpur|yelachenahalli)([a-z]{4,})\b",
    re.I,
)

PAT_BAD_ORD = re.compile(r"\b(\d+(?:th|st|nd|rd))(th|st|nd|rd)\b", re.I)


def fix_address(addr: str) -> str:
    if not addr:
        return addr
    s = addr.strip()

    # 1. Remaining glued localities
    s = PAT_LOC_GLUE2.sub(r"\1 \2", s)

    # 2. Bad ordinals
    s = PAT_BAD_ORD.sub(r"\1", s)

    # 3. Dedouble consecutive tokens
    toks = s.split()
    out = []
    prev = None
    for t in toks:
        if t == prev:
            continue
        out.append(t)
        prev = t
    s = " ".join(out)

    # 4. Dedupe triple+ tokens globally
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

    # 5. Duplicate state (exact token only)
    toks = s.split()
    out = []
    prev_state = False
    for t in toks:
        t_lower = t.lower()
        is_state = t_lower == "karnataka"
        if is_state and prev_state:
            continue
        out.append(t)
        prev_state = is_state
    s = " ".join(out)

    # 6. Duplicate city - only remove EXACT consecutive duplicates
    # "bangalore north bangalore" is VALID, do NOT touch
    toks = s.split()
    out = []
    prev_city = None
    for t in toks:
        t_lower = t.lower()
        is_city = t_lower in {"bangalore", "bengaluru", "bangaluru"}
        if is_city and prev_city == t_lower:
            continue
        out.append(t)
        prev_city = t_lower if is_city else prev_city
    s = " ".join(out)

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
