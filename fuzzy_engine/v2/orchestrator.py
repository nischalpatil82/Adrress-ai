"""
fuzzy_engine.v2.orchestrator
============================
End-to-end address correction pipeline.

Flow:
    raw input
      -> L1 normalize.parse
      -> L2 speller.correct
      -> L3 retrieval.HybridRetriever.search
      -> L4 reranker.Reranker.rerank (calibrated probabilities)
      -> L5 verify.AddressVerifier.verify (Google + India Post)
      -> final confidence + structured JSON

The pipeline is designed to be safe by default:
- If verification disagrees with the top-1, demote it.
- If everything is uncertain, return status="low_confidence" instead of guessing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from fuzzy_engine.normalizer import format_generated_address
from fuzzy_engine.v2.config import (
    AUTOCOMPLETE_TOP_K,
    FINAL_TOP_N,
    HIGH_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    RETRIEVAL_TOP_K,
)
from fuzzy_engine.v2.normalize import ParsedAddress, normalize_text, parse
from fuzzy_engine.v2.reranker import Reranker, RerankResult
from fuzzy_engine.v2.retrieval import Candidate, HybridRetriever
from fuzzy_engine.v2.speller import (
    DictionarySpeller,
    SpellResult,
    Speller,
    T5Speller,
    WordLM,
)
from fuzzy_engine.v2.sql_retriever import get_sql_retriever
from fuzzy_engine.v2.verify import (
    AddressVerifier,
    GooglePlacesAutocomplete,
    Verification,
)

log = logging.getLogger(__name__)


def _diff_tokens(before: str, after: str) -> list[tuple[str, str]]:
    """Return paired token differences between before/after strings.

    Uses simple greedy alignment of lowercased whitespace tokens. Meant to
    surface locality / city alias canonicalization changes to the UI (e.g.
    "bengaloore" -> "bangalore", "neare" -> "near").
    """
    if not before or not after:
        return []
    # NOTE: don't call normalize_text here — it would re-apply canonicalization
    # to both sides, hiding the very changes we want to surface.
    import re as _re
    _split = lambda s: [t for t in _re.split(r"[^a-z0-9]+", s.lower()) if t]
    b = _split(before)
    a = _split(after)
    # Simple position-aligned diff when lengths match; otherwise SequenceMatcher.
    changes: list[tuple[str, str]] = []
    if len(b) == len(a):
        for bt, at in zip(b, a):
            if bt != at:
                changes.append((bt, at))
        return changes
    # Fallback: set-based diff (order-insensitive)
    import difflib
    sm = difflib.SequenceMatcher(a=b, b=a)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            # pair up by index
            for k in range(max(i2 - i1, j2 - j1)):
                bt = b[i1 + k] if i1 + k < i2 else ""
                at = a[j1 + k] if j1 + k < j2 else ""
                if bt and at and bt != at:
                    changes.append((bt, at))
    return changes


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------
@dataclass
class Suggestion:
    address: str
    addr_id: int
    probability: float
    features: dict = field(default_factory=dict)


@dataclass
class CorrectionResult:
    query: str
    status: str                 # "verified" | "high_confidence" | "low_confidence" | "no_match"
    confidence: float           # 0..1 calibrated
    best_address: Optional[str]
    best_addr_id: Optional[int]
    structured: dict
    spell: dict
    parsed: dict
    verification: dict
    suggestions: list[Suggestion]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class AddressPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        speller: Speller,
        reranker: Reranker,
        verifier: AddressVerifier,
        sql_retriever=None,
    ) -> None:
        self.retriever = retriever
        self.speller = speller
        self.reranker = reranker
        self.verifier = verifier
        self.sql_retriever = sql_retriever
        # Live Google Places autocomplete client (used by live_suggest only).
        # Falls back to a no-op if no API key is configured.
        self.places_ac = GooglePlacesAutocomplete()

    # ---- factory -------------------------------------------------------
    @classmethod
    def from_config(cls, use_t5: bool = True, use_geocoder: bool = True) -> "AddressPipeline":
        retriever = HybridRetriever.from_artifacts()
        from fuzzy_engine.v2.corpus_lexicons import CorpusLexicons
        lexicons = CorpusLexicons.build(retriever.addresses)
        dict_speller = DictionarySpeller(retriever.addresses)
        t5 = T5Speller().load() if use_t5 else None
        lm = WordLM().fit(retriever.addresses[:50_000])  # sample LM training
        # Pass geo_names so the speller protects known locality/road tokens
        # (e.g. 'jakkasandra', 'bannerghatta') from being "corrected".
        speller = Speller(
            dictionary=dict_speller, t5=t5, lm=lm,
            geo_names=lexicons.geo_names,
        )
        reranker = Reranker().load()
        verifier = AddressVerifier() if use_geocoder else AddressVerifier(
            provider=__import__(
                "fuzzy_engine.v2.verify", fromlist=["NullGeocoder"]
            ).NullGeocoder()
        )
        sql_r = get_sql_retriever()
        instance = cls(retriever=retriever, speller=speller,
                       reranker=reranker, verifier=verifier,
                       sql_retriever=sql_r)
        instance.lexicons = lexicons
        return instance

    # ---- public API ----------------------------------------------------
    def autocomplete(self, prefix: str, k: int = AUTOCOMPLETE_TOP_K) -> list[Suggestion]:
        cands = self.retriever.autocomplete(prefix, k=k)
        return [
            Suggestion(
                address=format_generated_address(c.address),
                addr_id=c.addr_id,
                probability=1.0,
                features={"trie": 1.0},
            )
            for c in cands
        ]

    def live_suggest(self, prefix: str, k: int = AUTOCOMPLETE_TOP_K,
                     use_google: bool = True) -> dict:
        """Google-style live suggestions: word corrections + address hits.

        Lightweight (no geocode, no rerank): runs the parser + speller on
        the user's partial text and asks the retriever's trie for prefix
        matches on BOTH the raw and spell-corrected forms.

        ``use_google`` lets the caller force the DB-only branch even when a
        Places API key is configured (saves quota on autocomplete keystrokes).
        """
        prefix = (prefix or "").strip()
        if len(prefix) < 2:
            return {"query": prefix, "corrected": prefix,
                    "changes": [], "suggestions": []}
        parsed = parse(prefix)
        spell_input = parsed.normalized or prefix
        spell_res = self.speller.correct(spell_input)
        corrected = spell_res.corrected if spell_res.applied else spell_input
        # Surface normalize-only changes (alias canonicalization) too.
        norm_changes = _diff_tokens(prefix, spell_input)
        merged: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for pair in list(norm_changes) + list(spell_res.changes):
            if pair[0] == pair[1] or pair in seen:
                continue
            seen.add(pair)
            merged.append(pair)

        # Address suggestions — run DB and Google in PARALLEL when Google
        # autocomplete is enabled, so we don't wait serially.
        #
        # Ordering policy:
        # - Google enabled  -> Google first, then DB (user opted-in: trust live).
        # - Google disabled -> DB only (free path, no network).
        from concurrent.futures import ThreadPoolExecutor

        def _db_results() -> list[dict]:
            out: list[dict] = []
            seen_ids: set = set()
            for c in self.retriever.autocomplete(corrected, k=k):
                if c.addr_id in seen_ids:
                    continue
                seen_ids.add(c.addr_id)
                out.append({
                    "address": format_generated_address(c.address),
                    "addr_id": c.addr_id,
                    "source": "prefix",
                })
            if len(out) < k:
                try:
                    hybrid = self.retriever.search(
                        corrected, k=k * 2, pincode=parsed.pincode,
                    )
                except Exception:
                    hybrid = []
                for c in hybrid:
                    if c.addr_id in seen_ids:
                        continue
                    seen_ids.add(c.addr_id)
                    out.append({
                        "address": format_generated_address(c.address),
                        "addr_id": c.addr_id,
                        "source": "hybrid",
                    })
                    if len(out) >= k:
                        break
            return out

        # When the caller opts in via ``use_google=True`` we treat that as an
        # explicit per-request opt-in (UI toggle in /v2/settings) and only need
        # the API key — not the env-level flag — to be present.
        google_can_run = bool(use_google) and self.places_ac.has_key

        def _google_results() -> list[dict]:
            if not google_can_run:
                return []
            try:
                hits = self.places_ac.suggest(corrected, k=k, force=True)
            except Exception:
                hits = []
            return [
                {
                    "address": g.get("address") or "",
                    "addr_id": None,
                    "place_id": g.get("place_id"),
                    "source": "google",
                }
                for g in hits if g.get("address")
            ]

        if google_can_run:
            # Parallel fan-out so total latency = max(DB, Google), not sum.
            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_g = ex.submit(_google_results)
                fut_d = ex.submit(_db_results)
                google_hits = fut_g.result()
                db_hits = fut_d.result()
            # Google first (user opted in via V2_LIVE_GOOGLE_AC=1), then DB.
            ordered = google_hits + db_hits
        else:
            ordered = _db_results()

        # Dedupe by address text, preserving order.
        seen_addrs: set = set()
        suggestions: list[dict] = []
        for s in ordered:
            addr = (s.get("address") or "").strip()
            if not addr:
                continue
            key = addr.lower()
            if key in seen_addrs:
                continue
            seen_addrs.add(key)
            suggestions.append(s)
            if len(suggestions) >= k:
                break

        return {
            "query": prefix,
            "corrected": corrected,
            "changes": [list(c) for c in merged],
            "suggestions": suggestions,
        }

    def correct(self, raw: str, top_n: int = FINAL_TOP_N) -> CorrectionResult:
        notes: list[str] = []
        if not raw or len(raw.strip()) < 3:
            return self._empty_result(raw, status="no_match",
                                      notes=["query_too_short"])

        # L1
        parsed = parse(raw)

        # L2 — speller runs on the *normalized* text (which has already had
        # locality / city alias canonicalization applied). This way Google APIs
        # receive a clean query (e.g. "bengaloore" -> "bangalore") instead of
        # echoing the user's typos back as the formatted address.
        spell_input = parsed.normalized or raw
        spell_res = self.speller.correct(spell_input)
        search_query = spell_res.corrected if spell_res.applied else spell_input

        # Post-spell dedup: remove consecutive identical tokens (e.g. "layout layout")
        _sq_toks = search_query.split()
        _sq_dedup = []
        _sq_prev = None
        for _t in _sq_toks:
            if _t != _sq_prev:
                _sq_dedup.append(_t)
            _sq_prev = _t
        if len(_sq_dedup) != len(_sq_toks):
            search_query = " ".join(_sq_dedup)
            from dataclasses import replace as _dc_rep2
            spell_res = _dc_rep2(spell_res, corrected=search_query, applied=True)

        # Surface normalize/canonicalize changes (e.g. "bengaloore"->"bangalore",
        # "neare"->"near") so the UI can show them alongside speller changes.
        norm_changes = _diff_tokens(raw, spell_input)
        if norm_changes:
            merged = list(norm_changes) + list(spell_res.changes)
            # Deduplicate preserving order
            seen: set[tuple[str, str]] = set()
            unique_changes: list[tuple[str, str]] = []
            for pair in merged:
                if pair not in seen and pair[0] != pair[1]:
                    seen.add(pair)
                    unique_changes.append(pair)
            from dataclasses import replace as _dc_replace
            spell_res = _dc_replace(
                spell_res,
                changes=unique_changes,
                applied=bool(unique_changes),
            )

        # L3 — use parsed pincode as a hard pre-filter when present.
        candidates = self.retriever.search(
            search_query, k=RETRIEVAL_TOP_K, pincode=parsed.pincode
        )

        # L3b — merge live SQL DB candidates (if enabled and DB is reachable).
        # The DB may itself contain typos, so we fuzzy-match the results.
        if self.sql_retriever is not None:
            try:
                sql_cands = self.sql_retriever.search(search_query)
                if sql_cands:
                    # Merge: boost SQL candidates slightly so they compete
                    # with static-index hits.  Prefer SQL source for
                    # "found_in_database" later.
                    seen = {c.addr_id: c for c in candidates}
                    for sc in sql_cands:
                        if sc.addr_id in seen:
                            # augment existing candidate with SQL score
                            existing = seen[sc.addr_id]
                            existing.scores["sql"] = sc.scores.get("sql", 0.5)
                        else:
                            candidates.append(sc)
            except Exception as exc:
                log.warning("SQL retrieval failed (non-fatal): %s", exc)

        # ---- Fallback for zero DB candidates (international / out-of-corpus) ----
        if not candidates:
            # Even with no DB hits, still try geocoding — the user may have
            # typed an address outside India (e.g. German, US, UK addresses).
            verification = self.verifier.verify(
                query=search_query,
                expected_pincode=parsed.pincode,
                expected_state=parsed.state,
            )
            if verification.geocoded and verification.geocode:
                notes.append("no_db_match_geocoded")
                best_address = self._generate_address(
                    spell_corrected=spell_res.corrected if spell_res.applied else raw,
                    parsed=parsed,
                    verification=verification,
                )
                structured = self._structured_from_input(
                    spell_res.corrected if spell_res.applied else raw,
                    parsed, verification,
                )
                confidence = self._generated_confidence(
                    spell_res, parsed, verification, 0.0,
                )
                # For addresses outside the DB corpus (international / unknown),
                # don't let a successful geocode sink below "generated"
                # just because the lacks Indian city/pincode fields.
                if verification.geocoded:
                    confidence = max(confidence, LOW_CONFIDENCE_THRESHOLD)
                status = "generated" if confidence >= LOW_CONFIDENCE_THRESHOLD else "low_confidence"
                # If geocode precision is strong, promote to verified/high.
                if verification.geocode and verification.geocode.precision >= 0.85:
                    status = "verified"
                    confidence = max(confidence, HIGH_CONFIDENCE_THRESHOLD)
                elif verification.geocode and verification.geocode.precision >= 0.65:
                    status = "high_confidence"
                    confidence = max(confidence, 0.85)
                return CorrectionResult(
                    query=raw,
                    status=status,
                    confidence=round(confidence, 4),
                    best_address=best_address,
                    best_addr_id=None,
                    structured=structured,
                    spell={
                        "applied": spell_res.applied,
                        "corrected": spell_res.corrected,
                        "changes": [list(c) for c in spell_res.changes],
                        "used_t5": spell_res.used_t5,
                    },
                    parsed=parsed.to_dict(),
                    verification=verification.to_dict(),
                    suggestions=[],
                    notes=notes + list(verification.notes),
                )
            # No DB hits and geocode also failed — genuine no_match.
            return self._empty_result(raw, status="no_match",
                                      spell=spell_res, parsed=parsed,
                                      notes=["no_retrieval_hits"] + (notes or []))

        # L4 — rerank a wide pool, then apply pincode-anchored boost.
        # We rerank top 50 instead of `top_n` so the pincode boost has room
        # to promote a previously-low-ranked correct answer.
        reranked_pool = self.reranker.rerank(search_query, candidates, top_n=50)
        if not reranked_pool:
            return self._empty_result(raw, status="no_match",
                                      spell=spell_res, parsed=parsed,
                                      notes=["empty_rerank"])
        reranked = self._apply_pincode_anchor(reranked_pool, parsed)[:top_n]

        # The reranker can produce ties (identical probabilities) where
        # the top-1 is a generic stub like "Bangalore Karnataka 560078"
        # (scores high on token_set_ratio because its few tokens are all
        # in the query) while the actual best textual match sits below.
        # When the top-2 are tied, re-pick using a composite of
        #   _db_match_score * _distinctive_token_overlap
        # so generic stubs (overlap=0) lose to real address matches.
        # Parse the spell-corrected query so structural anchors are in
        # canonical form (typos like "vijaynagar" become "vijayanagar"
        # via the canonicalizer). Using `parsed` (raw input) here would
        # always disagree with the corpus candidate's canonical anchors.
        query_parsed = parse(search_query) if search_query != raw else parsed

        top = reranked[0]
        if len(reranked) > 1 and abs(reranked[0].probability - reranked[1].probability) < 0.02:
            scored: list[tuple[float, int, "RerankResult"]] = []
            for idx, r in enumerate(reranked[:5]):
                cp = parse(r.candidate.address)
                ms = self._db_match_score(search_query, r.candidate.address,
                                          query_parsed, cp)
                ov = self._distinctive_token_overlap(
                    search_query, r.candidate.address,
                )
                # Composite: both signals must be present for a high score.
                # Stub "Bangalore Karnataka 560078" has overlap=0 so it loses.
                composite = ms * (ov if ov > 0 else 0.05)
                scored.append((composite, -idx, r))  # -idx so earlier wins ties
            scored.sort(reverse=True)
            top = scored[0][2]

        # ---- DB-match strength check ------------------------------------
        # Decide whether the top DB candidate truly *matches* the user's
        # corrected input, or is just the closest-but-still-different row.
        cand_parsed = parse(top.candidate.address)
        match_score = self._db_match_score(search_query, top.candidate.address,
                                           query_parsed, cand_parsed)
        # Hard gates (ported from v1 matcher._structured_match_allowed):
        # pincode, city, road anchor, house numbers, and localities must
        # all agree before we can call it a "strong" DB match.
        struct_ok = self._structured_match_allowed(query_parsed, cand_parsed)
        building_ok = self._building_match_allowed(search_query, top.candidate.address)
        
        # Token overlap gate: must contain at least one distinctive token.
        # Strict > 0.5 fixes bug where length=2 distinctive tokens passed with only 1 match.
        token_ok = self._distinctive_token_overlap(
            search_query, top.candidate.address
        ) > 0.5
        is_strong_db_match = (
            match_score >= 0.80
            and struct_ok
            and building_ok
            and token_ok
        )

        # ---- L5 verification --------------------------------------------
        # Always verify the corrected USER input (not the DB candidate) so
        # the geocode reflects what the user actually meant.
        verification = self.verifier.verify(
            query=search_query,
            expected_pincode=parsed.pincode,
            expected_state=parsed.state,
        )

        # ---- L5.5 post-verification spell refinement --------------------
        # Use Google's verified formatted_address PLUS the top reranked DB
        # candidate as authoritative dictionaries to fix any typos that
        # the local speller / T5 missed (e.g. "zpertment" -> "apartment",
        # "maaain" -> "main", "begaloore" -> "bengaluru"). This is the
        # "catch everything" layer that handles any user mistake.
        if spell_res.corrected:
            # Top DB candidate is a great fallback dictionary even when
            # Google fails — it's a real address string from the gazetteer.
            extra_dict = ""
            if top is not None:
                extra_dict = top.candidate.address
            refined, extra_changes = self._refine_spell_from_verification(
                spell_res.corrected, verification, parsed,
                extra_dictionary_text=extra_dict,
            )
            if extra_changes:
                from dataclasses import replace as _dc_replace
                merged = list(spell_res.changes) + extra_changes
                seen: set[tuple[str, str]] = set()
                unique: list[tuple[str, str]] = []
                for pair in merged:
                    if pair not in seen and pair[0] != pair[1]:
                        seen.add(pair)
                        unique.append(pair)
                spell_res = _dc_replace(
                    spell_res,
                    corrected=refined,
                    changes=unique,
                    applied=True,
                )
                search_query = refined
                notes.append("spell_refined_by_geocode")

        # ---- Found-in-DB vs Generated decision -------------------------
        if is_strong_db_match:
            best_address = format_generated_address(top.candidate.address)
            best_addr_id = top.candidate.addr_id
            structured = self._structured(top.candidate.address, verification)
            confidence = self._final_confidence(top, parsed, verification)
            confidence = max(confidence, match_score)  # honesty floor
            status = "found_in_database" if confidence >= HIGH_CONFIDENCE_THRESHOLD \
                else self._status_from(confidence, verification)
        else:
            # GENERATE: build a clean address from the spell-corrected input
            # plus the India Post pincode lookup (if the pincode is valid).
            best_address = self._generate_address(
                spell_corrected=spell_res.corrected if spell_res.applied else raw,
                parsed=parsed,
                verification=verification,
            )
            best_addr_id = None
            structured = self._structured_from_input(
                spell_res.corrected if spell_res.applied else raw,
                parsed, verification,
            )
            # Confidence for a generated answer is driven by spell + pincode +
            # geocode signals, not by reranker (which only knew DB rows).
            confidence = self._generated_confidence(
                spell_res, parsed, verification, match_score,
            )
            status = "generated" if confidence >= LOW_CONFIDENCE_THRESHOLD \
                else "low_confidence"
            notes.append("no_strong_db_match")

        def _mk_sugg(r) -> Suggestion:
            return Suggestion(
                address=format_generated_address(r.candidate.address),
                addr_id=r.candidate.addr_id,
                probability=round(r.probability, 4),
                features=r.features,
            )

        suggestions = [
            _mk_sugg(r) for r in reranked
            if self._reliable_suggestion(search_query, r, parsed)
        ]
        # NOTE: We deliberately do NOT fall back to top reranked rows when
        # `_reliable_suggestion` filters everything out. Showing junk
        # candidates (e.g. "Tavarekere Bangalore South Tq..." for an
        # unrelated query) is worse than showing none. An empty list
        # honestly signals "no close match in database".
        # When the orchestrator already decided this IS a strong DB match
        # and surfaced it as best_address, guarantee the top-1 row is in
        # the suggestion list so the "Nearest Database Match" card can
        # render it (otherwise _reliable_suggestion can hide the very
        # match we just told the user about).
        if is_strong_db_match and top is not None:
            top_id = top.candidate.addr_id
            if not any(s.addr_id == top_id for s in suggestions):
                suggestions.insert(0, Suggestion(
                    address=format_generated_address(top.candidate.address),
                    addr_id=top_id,
                    probability=round(top.probability, 4),
                    features=top.features,
                ))

        return CorrectionResult(
            query=raw,
            status=status,
            confidence=round(confidence, 4),
            best_address=best_address,
            best_addr_id=best_addr_id,
            structured=structured,
            spell={
                "applied": spell_res.applied,
                "corrected": spell_res.corrected,
                "changes": [list(c) for c in spell_res.changes],
                "used_t5": spell_res.used_t5,
            },
            parsed=parsed.to_dict(),
            verification=verification.to_dict(),
            suggestions=suggestions,
            notes=notes + list(verification.notes),
        )

    # ---- internals -----------------------------------------------------
    @staticmethod
    def _apply_pincode_anchor(reranked: list[RerankResult],
                              parsed: ParsedAddress) -> list[RerankResult]:
        """Boost candidates whose address contains the parsed pincode.

        This is the single biggest precision win for Indian addresses:
        a 6-digit pincode is a strong, unambiguous geographic anchor.
        Candidates with matching pincode get +0.20 to their probability;
        candidates with a *different* 6-digit pincode get -0.10.
        """
        if not parsed.pincode:
            return reranked
        pin = parsed.pincode
        boosted: list[RerankResult] = []
        for r in reranked:
            addr = r.candidate.address
            # Cheap substring + length check: any 6-digit pincode in candidate
            cand_pin: Optional[str] = None
            for tok in addr.split():
                tok = tok.strip(",.")
                if tok.isdigit() and len(tok) == 6:
                    cand_pin = tok
                    break
            new_p = r.probability
            if cand_pin == pin:
                new_p = min(1.0, new_p + 0.20)
            elif cand_pin and cand_pin != pin:
                new_p = max(0.0, new_p - 0.10)
            boosted.append(
                RerankResult(
                    candidate=r.candidate,
                    raw_score=r.raw_score,
                    probability=new_p,
                    features=r.features,
                )
            )
        return sorted(boosted, key=lambda x: -x.probability)

    @staticmethod
    def _distinctive_token_overlap(query: str, candidate: str) -> float:
        """Fraction of distinctive (>=4 char, non-stop) query tokens present
        in candidate (exact OR fuzzy >=85). Used as a hard gate against
        false-positive DB matches.
        """
        from rapidfuzz import fuzz
        q = normalize_text(query).split()
        a_tokens = normalize_text(candidate).split()
        STOP = {
            "no", "noo", "nos", "cross", "main", "road", "rd", "street",
            "lane", "near", "opp", "behind", "floor", "block", "stage",
            "phase", "th", "st", "nd", "ist", "1st", "2nd", "3rd", "4th",
            "5th", "india", "bangalore", "bengaluru", "karnataka", "city",
            "nagar", "layout", "colony", "apartment", "apartments",
            "building", "tower", "complex", "mall", "near", "opposite",
            "house", "flat", "shop", "office", "village", "post", "taluk",
            "district", "state", "country", "pin", "pincode",
        }
        distinctive = [
            t for t in q
            if len(t) >= 4 and t not in STOP and not t.isdigit()
        ]
        if not distinctive:
            return 1.0  # nothing distinctive to gate on; let other signals decide
        a_set = set(a_tokens)
        hits = 0
        for t in distinctive:
            if t in a_set:
                hits += 1
                continue
            # fuzzy presence: any candidate token within 85 ratio
            if any(fuzz.ratio(t, at) >= 85 for at in a_tokens if len(at) >= 4):
                hits += 1
        return hits / len(distinctive)

    # --- Words we never rewrite (function words, generic descriptors,
    # ordinals, direction qualifiers, address parts already canonical).
    # Class-level so it's reused by the refinement helpers.
    _SPELL_NO_TOUCH = frozenset({
        "and", "the", "near", "behind", "opposite", "next", "above",
        "below", "north", "south", "east", "west", "side", "off",
        "first", "second", "third", "fourth", "fifth", "sixth",
        "seventh", "eighth", "ninth", "tenth",
        "road", "street", "main", "cross", "block", "stage", "phase",
        "sector", "layout", "lane", "circle", "junction", "extension",
        "extn", "colony", "nagar", "puram", "halli", "pura", "pur",
        "ward", "post", "village", "town", "city", "district", "state",
        "house", "flat", "apartment", "apartments", "tower", "towers",
        "building", "buildings",
        "no", "number", "floor", "ground", "park", "garden", "gardens",
    })

    @staticmethod
    def _consonant_skeleton(s: str) -> str:
        """Phonetic skeleton: keep first letter + remaining consonants.

        Cheap stand-in for Metaphone — catches cases where the user spelled
        the word phonetically (extra/dropped vowels) without needing an
        extra dependency. Examples:
            "maaain"   -> "mn"
            "main"     -> "mn"
            "bengloore"-> "bnglr"
            "bangalore"-> "bnglr"
            "vinaayaka"-> "vnyk"
            "vinayaka" -> "vnyk"
        """
        if not s:
            return ""
        s = s.lower()
        # Always keep the first character; then drop all vowels from the rest.
        return s[0] + "".join(ch for ch in s[1:] if ch not in "aeiou")

    @staticmethod
    def _refine_spell_from_verification(
        spell_corrected: str,
        verification: Verification,
        parsed: ParsedAddress,
        extra_dictionary_text: str = "",
    ) -> tuple[str, list[tuple[str, str]]]:
        """Use Google's verified address (and optionally the top DB candidate)
        as an authoritative dictionary to fix typos still present in
        `spell_corrected` after dictionary + T5 correction.

        Strategy:
            1. Build a "trusted token" dictionary from Google's
               `formatted_address` plus any extra_dictionary_text
               (e.g. the top reranked DB candidate's address).
            2. For each alphabetic input token, try in order:
               a. Exact match in dictionary  -> keep as-is.
               b. Fuzzy match (rapidfuzz.fuzz.ratio) above a length-aware
                  threshold AND safety constraints (length ratio, first
                  letter match, or matching suffix for longer tokens).
               c. Phonetic match (consonant skeleton equal).
            3. Apply correction only if confidence is high.
            4. Handle compound typos:
               - SPLIT: if a single input token equals concat of two
                 dictionary tokens (e.g. "gravityapartment").
               - MERGE: if two adjacent input tokens equal one dictionary
                 token (e.g. "vinay aka" -> "vinayaka").

        Skips refinement entirely when Google relocated to a different
        pincode than the user provided (verified India-Post pincode).

        Returns:
            (refined_text, [(before, after), ...])
        """
        if not spell_corrected:
            return spell_corrected, []

        try:
            from rapidfuzz import fuzz
        except ImportError:
            return spell_corrected, []

        import re as _re

        # --- Build the trusted dictionary --------------------------------
        sources: list[str] = []

        # Source 1 — Google's geocode (primary, when trustworthy).
        if verification.geocoded and verification.geocode:
            google_addr = verification.geocode.formatted_address or ""
            if google_addr:
                # Trust gate: skip Google when it relocated to a different
                # pincode than the user's verified pincode.
                geo = verification.geocode
                bad_relocation = (
                    parsed.pincode and verification.pincode_valid
                    and geo.postal_code
                    and parsed.pincode != geo.postal_code
                    and parsed.pincode not in google_addr
                )
                if not bad_relocation:
                    sources.append(google_addr)

        # Source 2 — Top DB candidate (fallback when Google fails or as
        # additional signal). Real address strings = trustworthy vocab.
        if extra_dictionary_text:
            sources.append(extra_dictionary_text)

        if not sources:
            return spell_corrected, []

        dictionary: set[str] = set()
        for src in sources:
            for t in _re.findall(r"[A-Za-z]{3,}", src):
                dictionary.add(t.lower())
        if not dictionary:
            return spell_corrected, []

        NO_TOUCH = AddressPipeline._SPELL_NO_TOUCH
        skeleton_of = AddressPipeline._consonant_skeleton

        # Pre-compute skeletons for dictionary tokens for the phonetic
        # fallback (only useful when fuzzy match fails).
        dict_by_skeleton: dict[str, str] = {}
        for dt in dictionary:
            sk = skeleton_of(dt)
            # Keep the shortest dictionary word per skeleton — usually the
            # canonical form ("main" beats "mainst", "vinayaka" beats
            # "vinayakatemple").
            if sk not in dict_by_skeleton or len(dt) < len(dict_by_skeleton[sk]):
                dict_by_skeleton[sk] = dt

        def _best_fuzzy(low: str) -> tuple[Optional[str], int]:
            """Find the highest-scoring dictionary token for `low`."""
            best_match: Optional[str] = None
            best_score = 0
            for gt in dictionary:
                if len(gt) < 3:
                    continue
                # Length must be reasonably similar.
                length_ratio = max(2, len(low) // 2 + 1)
                if abs(len(gt) - len(low)) > length_ratio:
                    continue
                # First-letter mismatch allowed only when:
                #   - both tokens are reasonably long (>=6) AND
                #   - the suffix overlaps (last 3 chars match), OR
                #   - the consonant skeleton matches (phonetic typo).
                if gt[0] != low[0]:
                    long_enough = len(low) >= 6 and len(gt) >= 6
                    suffix_match = long_enough and low[-3:] == gt[-3:]
                    skeleton_match = skeleton_of(low) == skeleton_of(gt)
                    if not (suffix_match or skeleton_match):
                        continue
                s = fuzz.ratio(low, gt)
                if s > best_score:
                    best_score = s
                    best_match = gt
            return best_match, best_score

        # --- First pass: token-by-token correction ----------------------
        words = spell_corrected.split()
        out_words: list[str] = []
        changes: list[tuple[str, str]] = []

        i = 0
        while i < len(words):
            word = words[i]

            # Preserve numbers / mixed alnum tokens.
            if not word.isalpha():
                out_words.append(word)
                i += 1
                continue
            low = word.lower()

            # Already correct or protected — keep as-is.
            if low in dictionary or low in NO_TOUCH or len(low) < 4:
                out_words.append(word)
                i += 1
                continue

            # --- MERGE check: does (this + next) equal a dictionary token?
            # Catches "vinay aka" -> "vinayaka".
            merged_match: Optional[str] = None
            if i + 1 < len(words) and words[i + 1].isalpha():
                merged = (word + words[i + 1]).lower()
                if merged in dictionary and len(merged) >= 6:
                    merged_match = merged
                else:
                    # Allow a fuzzy merge if exact merge isn't in dict.
                    cand, score = _best_fuzzy(merged)
                    if cand and score >= 85 and len(cand) >= 6:
                        merged_match = cand
            if merged_match:
                # Capitalization: follow first sub-token.
                token_out = (merged_match.capitalize()
                             if word[0].isupper() else merged_match)
                out_words.append(token_out)
                changes.append((low + " " + words[i + 1].lower(), merged_match))
                i += 2
                continue

            # --- SPLIT check: does this token equal concat of two dict
            # tokens? Catches "gravityapartment" -> "gravity apartment".
            if len(low) >= 8:
                split_pair: Optional[tuple[str, str]] = None
                for cut in range(3, len(low) - 2):
                    a, b = low[:cut], low[cut:]
                    if a in dictionary and b in dictionary:
                        split_pair = (a, b)
                        break
                if split_pair:
                    a, b = split_pair
                    if word[0].isupper():
                        out_words.append(a.capitalize())
                        out_words.append(b.capitalize())
                    else:
                        out_words.append(a)
                        out_words.append(b)
                    changes.append((low, f"{a} {b}"))
                    i += 1
                    continue

            # --- Standard token correction (fuzzy + phonetic) -----------
            best_match, best_score = _best_fuzzy(low)

            # Phonetic fallback when fuzzy score is borderline.
            if (not best_match or best_score < 65) and len(low) >= 4:
                sk = skeleton_of(low)
                if sk and sk in dict_by_skeleton:
                    cand = dict_by_skeleton[sk]
                    # Safety: skeleton must be substantive (>=2 chars) and
                    # cand must be similar length.
                    if (len(sk) >= 2
                            and cand != low
                            and abs(len(cand) - len(low)) <= max(3, len(low) // 2)
                            and cand[0] == low[0]):
                        best_match = cand
                        best_score = 70  # bump above threshold

            # Length-aware threshold:
            #   - len >=8 : 60  (long typos like "zpertment"->"apartment")
            #   - len 6-7 : 65
            #   - len 4-5 : 75  (short tokens need stronger evidence)
            threshold = 60 if len(low) >= 8 else (65 if len(low) >= 6 else 75)
            if best_match and best_score >= threshold:
                token_out = (best_match.capitalize()
                             if word[0].isupper() else best_match)
                out_words.append(token_out)
                changes.append((low, best_match))
            else:
                out_words.append(word)

            i += 1

        return " ".join(out_words), changes

    @staticmethod
    def _real_house_numbers(parsed: ParsedAddress,
                            cand_parsed: ParsedAddress) -> set[str]:
        """User's numeric tokens with pincode-suffix candidates removed.

        When the user truncates a pincode (e.g. "BANGALORE- 82" intending
        560082, or "560 082" split awkwardly), the trailing 2-3 digits
        get parsed as a generic `numbers` entry — but they're really
        pincode fragments, not house numbers. Treating them as house
        numbers causes false-positive structural rejections against
        candidates whose pincode happens to end the same way.

        We drop user numbers that:
          - are 2-3 digits long, AND
          - exactly match the last 2 / 3 / 4 digits of the candidate's
            6-digit pincode.

        A 4+ digit number (like a real house number "1247") is kept as-is.
        """
        nums = {n for n in parsed.numbers if len(n) != 6}
        cand_pin = cand_parsed.pincode
        if not nums or not cand_pin or len(cand_pin) != 6:
            return nums
        suffixes = {cand_pin[-2:], cand_pin[-3:], cand_pin[-4:]}
        return {n for n in nums if not (len(n) <= 3 and n in suffixes)}

    @staticmethod
    def _db_match_score(query: str, candidate: str,
                        parsed: ParsedAddress,
                        cand_parsed: ParsedAddress) -> float:
        """How well does the top DB row actually match the user's input?

        Ported from v1 corrector._display_similarity:
        - Base fuzzy score = max(token_sort_ratio, token_set_ratio) / 100
        - Heavy structured penalties when core tokens (pincode, city,
          road anchor, locality, house numbers) disagree.
        This prevents a candidate with the same pincode+ city but a
        completely different street from scoring 0.80+.
        """
        from rapidfuzz import fuzz
        q = normalize_text(query)
        a = normalize_text(candidate)
        if not q or not a:
            return 0.0

        score = max(
            float(fuzz.token_sort_ratio(q, a)),
            float(fuzz.token_set_ratio(q, a)),
        )

        # --- Structured penalties (ported from v1) ---
        # Numbers (house numbers, not 6-digit pincodes).
        # Drop tokens from the user's number set that look like a pincode
        # suffix of the candidate (e.g. user typed "BANGALORE- 82" intending
        # 560082; the "82" should not be treated as a house number).
        q_nums = AddressPipeline._real_house_numbers(parsed, cand_parsed)
        c_nums = {n for n in cand_parsed.numbers if len(n) != 6}
        if q_nums and c_nums and not (q_nums & c_nums):
            score -= 18.0

        # Pincode mismatch
        if parsed.pincode and cand_parsed.pincode \
                and parsed.pincode != cand_parsed.pincode:
            score -= 20.0

        # City mismatch
        if parsed.city and cand_parsed.city and parsed.city != cand_parsed.city:
            score -= 12.0

        # Road anchor mismatch
        if parsed.road_anchor and cand_parsed.road_anchor \
                and parsed.road_anchor != cand_parsed.road_anchor:
            score -= 25.0

        # Locality mismatch
        if (parsed.locality_anchors and cand_parsed.locality_anchors
                and not (parsed.locality_anchors & cand_parsed.locality_anchors)):
            score -= 15.0

        return round(max(score, 0.0) / 100.0, 4)

    @staticmethod
    def _structured_match_allowed(parsed: ParsedAddress,
                                   cand_parsed: ParsedAddress) -> bool:
        """Hard gates: core structured fields must agree.

        Ported from v1 matcher._structured_match_allowed.
        Prevents high fuzzy scores on addresses that look similar but
        actually refer to different places (e.g. same pincode, different
        road or different locality).
        """
        if parsed.pincode and cand_parsed.pincode \
                and parsed.pincode != cand_parsed.pincode:
            return False
        if parsed.city and cand_parsed.city \
                and parsed.city != cand_parsed.city:
            return False
        if parsed.road_anchor and cand_parsed.road_anchor \
                and parsed.road_anchor != cand_parsed.road_anchor:
            return False

        # Same pincode-suffix exclusion as in _db_match_score: avoid
        # rejecting "BANGALORE- 82" vs "...560082" candidate just because
        # the user truncated the pincode.
        q_nums = AddressPipeline._real_house_numbers(parsed, cand_parsed)
        c_nums = {n for n in cand_parsed.numbers if len(n) != 6}
        if q_nums and c_nums and not (q_nums & c_nums):
            return False

        if (parsed.locality_anchors and cand_parsed.locality_anchors
                and not (parsed.locality_anchors & cand_parsed.locality_anchors)):
            return False
        # Sub-anchor (block/phase/sector/stage) mismatch: "3rd block" vs "1st block".
        q_sub = dict(parsed.sub_anchors)
        c_sub = dict(cand_parsed.sub_anchors)
        for word, q_num in q_sub.items():
            c_num = c_sub.get(word)
            if c_num and c_num != q_num:
                return False
        return True

    @staticmethod
    def _building_match_allowed(query: str, candidate: str) -> bool:
        """If the query specifies a building brand, it must appear in the candidate."""
        import re as _re_bld
        # Building-name guard
        BUILDING_KW = (
            "apartment", "apartments", "apt", "tower", "towers",
            "heights", "residency", "residence", "residences",
            "villa", "villas", "complex", "enclave", "mansion",
            "plaza", "arcade", "court", "manor",
        )
        bld_match = _re_bld.search(
            r"\b([a-z]{4,})\s+(" + "|".join(BUILDING_KW) + r")\b",
            (query or "").lower(),
        )
        if bld_match:
            brand = bld_match.group(1)
            if brand not in {"main", "near", "behind", "opposite",
                             "house", "flat", "block", "stage",
                             "phase", "sector", "cross", "road",
                             "first", "second", "third", "fourth",
                             "fifth", "sixth", "new", "old"}:
                # The brand must appear in the candidate
                if brand not in (candidate or "").lower():
                    return False
        return True

    @staticmethod
    def _reliable_suggestion(query: str, r: RerankResult,
                              parsed: ParsedAddress) -> bool:
        """Should this suggestion be shown to the user?

        Ported from v1 corrector._is_reliable_suggestion.
        Drops candidates whose structured fields fundamentally disagree
        with the user's input, even if the reranker gave them a decent
        probability (e.g. same pincode, different road).
        """
        if r.probability < 0.40:
            return False
        cand = r.candidate.address
        cp = parse(cand)
        # IMPORTANT: `parsed` is the parse of the user's RAW input (which may
        # contain typos like "vijaynagar"). The candidate has been
        # spell-corrected/canonicalized, so its anchors will be in canonical
        # form ("vijayanagar"). Comparing the two would always reject the
        # match. Re-parse the spell-corrected `query` so the structural gates
        # operate on post-correction tokens.
        qp = parse(query)
        # Numbers (house numbers, not pincodes)
        q_nums = {n for n in qp.numbers if len(n) != 6}
        c_nums = {n for n in cp.numbers if len(n) != 6}
        if q_nums and c_nums and not (q_nums & c_nums):
            return False
        # Pincode
        if qp.pincode and cp.pincode and qp.pincode != cp.pincode:
            return False
        # Road anchor
        if qp.road_anchor and cp.road_anchor \
                and qp.road_anchor != cp.road_anchor:
            return False
        # Locality
        if (qp.locality_anchors and cp.locality_anchors
                and not (qp.locality_anchors & cp.locality_anchors)):
            return False
        # Sub-anchor (block/phase/sector/stage)
        q_sub = dict(qp.sub_anchors)
        c_sub = dict(cp.sub_anchors)
        for word, q_num in q_sub.items():
            c_num = c_sub.get(word)
            if c_num and c_num != q_num:
                return False
        # Hard gate: must have some token overlap (>=20%)
        q_tok = set(normalize_text(query).split())
        c_tok = set(normalize_text(cand).split())
        if not q_tok:
            return False
        if not AddressPipeline._building_match_allowed(query, cand):
            return False
        overlap = len(q_tok & c_tok) / len(q_tok)
        return overlap >= 0.20

    @staticmethod
    def _generate_address(spell_corrected: str, parsed: ParsedAddress,
                          verification: Verification) -> str:
        """Build a presentable corrected address from the user's input.

        Uses:
        - spell-corrected text (already cleaned)
        - India Post pincode lookup (district + state) if available
        - Geocoder formatted address (if Nominatim/Google succeeded)

        When the geocoder returns a pincode-level generic address (no
        sublocality / street), we merge the user's locality/street/block
        into the geocoder's city/state/pincode rather than discarding it.
        """
        geo = verification.geocode
        if verification.geocoded and geo and geo.formatted_address:
            # --- TRUST CHECK on Google's formatted_address -----------------
            # Google's POI/Places result can be in a wildly different
            # location from what the user typed. Detect and reject:
            #   (a) Pincode mismatch when user's pincode is India-Post valid.
            #   (b) User provided locality anchors but none appear in result.
            # Reject => fall through to the generic-merge path which builds
            # the address from the user's spell-corrected text + city/state/pin.
            import re as _re_trust
            fmt_lower = geo.formatted_address.lower()
            untrusted = False
            if (parsed.pincode and verification.pincode_valid
                    and geo.postal_code
                    and parsed.pincode != geo.postal_code
                    and parsed.pincode not in fmt_lower):
                untrusted = True
            if not untrusted:
                # Token-based trust check on spell-corrected text.
                # Any meaningful alpha token (len>=5, not a stopword) that the
                # user typed should appear in Google's formatted_address.
                # If too many are missing, Google relocated to a different POI.
                STOP = {"road", "main", "cross", "stage", "layout", "block",
                        "phase", "sector", "near", "behind", "opposite",
                        "bengaluru", "bangalore", "karnataka", "india",
                        "house", "flat", "no", "number", "street"}
                toks = _re_trust.findall(r"[a-zA-Z]+", (spell_corrected or "").lower())
                meaningful = [t for t in toks if len(t) >= 5 and t not in STOP]
                if meaningful:
                    missing = [t for t in meaningful if t not in fmt_lower]
                    # Untrust if MAJORITY of meaningful tokens are missing.
                    if len(missing) >= max(1, len(meaningful) // 2 + 1):
                        untrusted = True
                # Numeric ordinals (e.g. "26th main" vs "16th Main") —
                # if user said "26 main" / "26th main" and Google has a
                # different number before "main", that's also untrust.
                ord_match = _re_trust.search(
                    r"\b(\d{1,3})(?:st|nd|rd|th)?\s+(main|cross|block|stage)\b",
                    (spell_corrected or "").lower(),
                )
                if ord_match:
                    num = ord_match.group(1)
                    kw = ord_match.group(2)
                    pat = rf"\b{num}\w*\s+{kw}"
                    if not _re_trust.search(pat, fmt_lower):
                        untrusted = True
                # Building-name guard: if user typed "<brand> apartment/tower/
                # heights/residency/...", the brand MUST appear in Google's
                # formatted_address. Catches the "gravity apartment" -> Google
                # returning "Sriranga Apartment" relocation bug.
                BUILDING_KW = (
                    "apartment", "apartments", "apt", "tower", "towers",
                    "heights", "residency", "residence", "residences",
                    "villa", "villas", "complex", "enclave", "mansion",
                    "plaza", "arcade", "court", "manor",
                )
                bld_match = _re_trust.search(
                    r"\b([a-z]{4,})\s+(" + "|".join(BUILDING_KW) + r")\b",
                    (spell_corrected or "").lower(),
                )
                if bld_match:
                    brand = bld_match.group(1)
                    # ignore generic descriptors that aren't really brand names
                    if brand not in {"main", "near", "behind", "opposite",
                                     "house", "flat", "block", "stage",
                                     "phase", "sector", "cross", "road",
                                     "first", "second", "third", "fourth",
                                     "fifth", "sixth", "new", "old"}:
                        if brand not in fmt_lower:
                            untrusted = True

            # Detect generic pincode-level geocode: no sublocality or street
            is_generic = (
                geo.sublocality is None
                and geo.street is None
                and geo.house_number is None
            )
            if is_generic or untrusted:
                # Generic pincode-level geocode (just "City, State, Pincode"):
                # keep the user's spell-corrected text as the body and append
                # ONLY the parts that aren't already present.
                from fuzzy_engine.normalizer import format_generated_address
                body = format_generated_address(spell_corrected)
                lower = body.lower()
                bits: list[str] = [body]
                pin_info = verification.pincode_info or {}
                city = pin_info.get("district") or geo.locality or ""
                state = pin_info.get("state") or geo.administrative_area or ""
                pin = parsed.pincode or geo.postal_code or ""
                if city and city.lower() not in lower:
                    bits.append(city.title())
                    lower += " " + city.lower()
                if state and state.lower() not in lower:
                    bits.append(state.title())
                    lower += " " + state.lower()
                if pin and pin not in lower:
                    bits.append(pin)
                    lower += " " + pin
                if "india" not in lower:
                    bits.append("India")
                return ", ".join(bits)

            # Non-generic & trusted: use geocoder formatted address directly.
            formatted = geo.formatted_address
            # Strip Plus Codes (e.g. "VJRH+FCR, ") — they leak from Google.
            formatted = _re_trust.sub(
                r"\b[A-Z0-9]{4,}\+[A-Z0-9]{2,3}\b,?\s*", "", formatted
            ).strip(" ,")
            user_pin = parsed.pincode
            geo_pin = geo.postal_code
            if user_pin and verification.pincode_valid and geo_pin \
                    and user_pin != geo_pin and geo_pin in formatted:
                formatted = formatted.replace(geo_pin, user_pin)

            # Extract user-stated house number from raw query.
            # Must be EXPLICIT ("house no 81" / "no 12" / "#42" / "12/3").
            # Random digits like "5" from "5t block" are NOT house numbers.
            import re as _re_hn
            user_hn = None
            src = (parsed.raw or spell_corrected or "").lower()
            m = _re_hn.match(
                r"^\s*(?:house\s*no|h\.?\s*no|d\.?\s*no|flat\s*no|no\.?|#)\s*([\w/-]+)",
                src,
            )
            if m:
                user_hn = m.group(1).strip()
            else:
                m2 = _re_hn.match(r"^\s*(\d{1,4}(?:[/-]\w+)?)\b", parsed.raw or "")
                if m2:
                    user_hn = m2.group(1)

            if user_hn and user_hn not in formatted:
                # If Google's formatted_address starts with a different house
                # number (e.g. "Bangalore Library, 95, ..."), replace it.
                m3 = _re_hn.match(r"^([^,]+,\s*)?(\d{1,4}[\w/-]*)\s*,", formatted)
                if m3 and m3.group(2) != user_hn:
                    formatted = formatted.replace(
                        m3.group(0), f"{user_hn}, ", 1
                    )
                else:
                    formatted = f"{user_hn}, {formatted}"
            return formatted

        # Fallback: plain spell-corrected with canonical city/state/pin appended.
        from fuzzy_engine.normalizer import format_generated_address
        text = format_generated_address(spell_corrected)
        bits: list[str] = [text]
        lower = text.lower()
        pin_info = verification.pincode_info
        if pin_info:
            district = (pin_info.get("district") or "").strip()
            state = (pin_info.get("state") or "").strip()
            if district and district.lower() not in lower:
                bits.append(district.title())
            if state and state.lower() not in lower:
                bits.append(state.title())
        elif parsed.state and parsed.state not in lower:
            bits.append(parsed.state.title())
        if parsed.pincode and parsed.pincode not in lower:
            bits.append(parsed.pincode)
        if "india" not in lower:
            bits.append("India")
        return ", ".join(bits)

    @staticmethod
    def _generated_confidence(spell_res, parsed: ParsedAddress,
                              verification: Verification,
                              best_db_match_score: float) -> float:
        """Confidence for a generated (not-found-in-db) answer.

        Built from independent signals:
        - pincode validity (India Post)
        - geocode success on the corrected input
        - completeness of parsed components (city, state, pincode)
        - mild credit for closest DB row
        """
        score = 0.30  # base "we cleaned your typos"
        if verification.pincode_valid:
            score += 0.20
        if verification.pincode_valid and verification.pincode_consistent:
            score += 0.10
        if verification.geocoded:
            score += 0.20
        # Component completeness
        have = sum(1 for x in (parsed.city, parsed.state, parsed.pincode) if x)
        score += 0.05 * have
        # A nearby DB row is a weak positive signal
        score += 0.10 * best_db_match_score
        return min(score, 0.95)  # cap at 0.95 (we never SAW it in DB)

    @staticmethod
    def _structured_from_input(spell_corrected: str,
                               parsed: ParsedAddress,
                               verification: Verification) -> dict:
        """Structured representation when the answer is generated, not from DB.

        Pincode (India Post) is the authoritative source for state/district.
        Google's geocode/validation can echo the user's misspellings, so we
        OVERRIDE state/city with India Post when the pincode is valid.
        """
        # Extract user-stated house number from raw query (e.g. "house no 81").
        # Used as a fallback when Google's geocoder doesn't return one for
        # a residential address.
        import re as _re_hn
        user_hn = None
        m = _re_hn.match(
            r"^\s*(?:house\s*no|h\.?\s*no|d\.?\s*no|flat\s*no|no\.?|#)\s*([\w/-]+)",
            (parsed.raw or spell_corrected or "").lower(),
        )
        if m:
            user_hn = m.group(1).strip()
        elif parsed.raw:
            m2 = _re_hn.match(r"^\s*(\d{1,4}(?:[/-]\w+)?)\b", parsed.raw)
            if m2:
                user_hn = m2.group(1)

        if verification.geocode:
            g = verification.geocode
            pin_info = verification.pincode_info or {}
            # --- Same trust check as _generate_address ---------------------
            import re as _re_t2
            fmt_lower = (g.formatted_address or "").lower()
            untrusted = False
            if (parsed.pincode and verification.pincode_valid
                    and g.postal_code
                    and parsed.pincode != g.postal_code
                    and parsed.pincode not in fmt_lower):
                untrusted = True
            if not untrusted:
                STOP = {"road", "main", "cross", "stage", "layout", "block",
                        "phase", "sector", "near", "behind", "opposite",
                        "bengaluru", "bangalore", "karnataka", "india",
                        "house", "flat", "no", "number", "street"}
                toks = _re_t2.findall(r"[a-zA-Z]+", (spell_corrected or "").lower())
                meaningful = [t for t in toks if len(t) >= 5 and t not in STOP]
                if meaningful:
                    missing = [t for t in meaningful if t not in fmt_lower]
                    if len(missing) >= max(1, len(meaningful) // 2 + 1):
                        untrusted = True
                ord_match = _re_t2.search(
                    r"\b(\d{1,3})(?:st|nd|rd|th)?\s+(main|cross|block|stage)\b",
                    (spell_corrected or "").lower(),
                )
                if ord_match:
                    pat = rf"\b{ord_match.group(1)}\w*\s+{ord_match.group(2)}"
                    if not _re_t2.search(pat, fmt_lower):
                        untrusted = True
                # Building-name guard (mirrors _generate_address).
                BUILDING_KW = (
                    "apartment", "apartments", "apt", "tower", "towers",
                    "heights", "residency", "residence", "residences",
                    "villa", "villas", "complex", "enclave", "mansion",
                    "plaza", "arcade", "court", "manor",
                )
                bld_match = _re_t2.search(
                    r"\b([a-z]{4,})\s+(" + "|".join(BUILDING_KW) + r")\b",
                    (spell_corrected or "").lower(),
                )
                if bld_match:
                    brand = bld_match.group(1)
                    if brand not in {"main", "near", "behind", "opposite",
                                     "house", "flat", "block", "stage",
                                     "phase", "sector", "cross", "road",
                                     "first", "second", "third", "fourth",
                                     "fifth", "sixth", "new", "old"}:
                        if brand not in fmt_lower:
                            untrusted = True
            # ---------------------------------------------------------------
            # Authority order: India Post pincode > Google geocode echo.
            authoritative_state = pin_info.get("state") or g.administrative_area
            authoritative_city = pin_info.get("district") or g.locality
            # Prefer user-stated house number over Google's.
            authoritative_hn = user_hn or g.house_number
            # Pincode authority: user-provided pincode wins if valid in India Post,
            # otherwise fall back to Google's pincode.
            if parsed.pincode and verification.pincode_valid:
                authoritative_pin = parsed.pincode
            else:
                authoritative_pin = g.postal_code or parsed.pincode
            # When Google's result is untrusted, use user-parsed road/locality
            # instead of Google's wrong street/sublocality.
            if untrusted:
                street = (parsed.road_anchor or "").title() or None
                sublocality = (", ".join(sorted(parsed.locality_anchors))
                               or None)
                source = "geocode_on_input_untrusted"
            else:
                street = g.street
                sublocality = g.sublocality
                source = "geocode_on_input"
            return {
                "house_number": authoritative_hn,
                "street": street,
                "sublocality": sublocality,
                "city": authoritative_city,
                "state": authoritative_state,
                "pincode": authoritative_pin,
                "country": g.country or "India",
                "lat": g.lat,
                "lon": g.lon,
                "place_id": g.place_id,
                "formatted": g.formatted_address,
                "source": source,
            }
        pin_info = verification.pincode_info or {}
        return {
            "house_number": user_hn,
            "street": parsed.road_anchor,
            "sublocality": ", ".join(sorted(parsed.locality_anchors)) or None,
            "city": parsed.city or pin_info.get("district"),
            "state": parsed.state or pin_info.get("state"),
            "pincode": parsed.pincode,
            "country": "India",
            "lat": None,
            "lon": None,
            "place_id": None,
            "formatted": spell_corrected,
            "source": "input_corrected",
        }

    def _final_confidence(self, top: RerankResult,
                          parsed: ParsedAddress,
                          verification: Verification) -> float:
        """Combine reranker probability with verification signals.

        Reranker is the prior; verification is the truth gate.
        """
        prob = top.probability
        # Verification boosts/penalties (multiplicative; bounded).
        if verification.geocoded:
            prob = min(1.0, prob + 0.05 * (verification.geocode.precision if verification.geocode else 0.5))
        else:
            prob *= 0.9  # mild penalty when geocode missed (could just be quota)

        if verification.pincode_valid:
            prob = min(1.0, prob + 0.05)
        if verification.pincode_valid and not verification.pincode_consistent:
            prob *= 0.7

        # Strong agreement bonus: parsed pincode matches geocode pincode.
        geo = verification.geocode
        if geo and parsed.pincode and geo.postal_code:
            if parsed.pincode == geo.postal_code:
                prob = min(1.0, prob + 0.05)
            else:
                prob *= 0.85

        return max(0.0, min(prob, 1.0))

    @staticmethod
    def _status_from(conf: float, ver: Verification) -> str:
        if ver.geocoded and conf >= HIGH_CONFIDENCE_THRESHOLD:
            return "verified"
        if conf >= HIGH_CONFIDENCE_THRESHOLD:
            return "high_confidence"
        if conf >= LOW_CONFIDENCE_THRESHOLD:
            return "medium_confidence"
        return "low_confidence"

    @staticmethod
    def _structured(address: str, ver: Verification) -> dict:
        if ver.geocode:
            g = ver.geocode
            return {
                "house_number": g.house_number,
                "street": g.street,
                "sublocality": g.sublocality,
                "city": g.locality,
                "state": g.administrative_area,
                "pincode": g.postal_code,
                "country": g.country,
                "lat": g.lat,
                "lon": g.lon,
                "place_id": g.place_id,
                "formatted": g.formatted_address,
                "source": "google_geocode",
            }
        # Fallback: parse the raw db candidate
        p = parse(address)
        return {
            "house_number": None,
            "street": p.road_anchor,
            "sublocality": ", ".join(sorted(p.locality_anchors)) or None,
            "city": p.city,
            "state": p.state,
            "pincode": p.pincode,
            "country": "India",
            "lat": None,
            "lon": None,
            "place_id": None,
            "formatted": format_generated_address(address),
            "source": "db_parse",
        }

    def _empty_result(self, raw: str, status: str,
                      spell: Optional[SpellResult] = None,
                      parsed: Optional[ParsedAddress] = None,
                      notes: Optional[list[str]] = None) -> CorrectionResult:
        return CorrectionResult(
            query=raw,
            status=status,
            confidence=0.0,
            best_address=None,
            best_addr_id=None,
            structured={},
            spell=(
                {
                    "applied": spell.applied if spell else False,
                    "corrected": spell.corrected if spell else raw,
                    "changes": [list(c) for c in (spell.changes if spell else [])],
                    "used_t5": spell.used_t5 if spell else False,
                }
            ),
            parsed=parsed.to_dict() if parsed else {},
            verification={},
            suggestions=[],
            notes=notes or [],
        )
