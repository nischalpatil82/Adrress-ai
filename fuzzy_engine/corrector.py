"""
fuzzy_engine.corrector
======================
Main orchestrator for the enterprise address correction engine.

Architecture (3-Tier):
  Tier 1 — MySQL DB Lookup: If the address exists in the database
           (>=90% match), return the exact verified address.
  Tier 2 — T5 ML Model: If NOT in the database, use the trained
           neural network to intelligently correct the address.
  Tier 3 — RapidFuzz Validation: Cross-reference the T5 output
           against the database to find the closest real address.

The T5 model is the CORE BRAIN. RapidFuzz is only the validator.
"""

from rapidfuzz import fuzz

from fuzzy_engine.normalizer import normalize
from fuzzy_engine.spell_checker import SpellChecker
from fuzzy_engine.matcher import AddressMatcher
from fuzzy_engine.config import TOP_N_SUGGESTIONS, DB_MATCH_THRESHOLD
from fuzzy_engine.dictionaries import KNOWN_CITIES
from fuzzy_engine.parser import parse_address


class AddressCorrector:
    """
    Enterprise address correction engine with T5 ML model.

    Usage (CSV, no ML):
        corrector = AddressCorrector("data/realistic_addresses.csv")

    Usage (MySQL + T5 ML model):
        corrector = AddressCorrector.from_database(use_ml=True)
    """

    def __init__(self, csv_path: str, use_ml: bool = False,
                 model_path: str = "models/t5_address"):
        """
        Initialize from CSV file.
        """
        self._matcher = AddressMatcher(csv_path=csv_path)
        self._spell_checker = SpellChecker(self._matcher.addresses)
        self._source = f"CSV: {csv_path}"
        self._t5 = None

        if use_ml:
            self._load_t5(model_path)

    @classmethod
    def from_database(cls, engine=None, use_ml: bool = True,
                      model_path: str = "models/t5_address"):
        """
        Initialize from MySQL database with optional T5 model.
        """
        from fuzzy_engine.db_loader import load_addresses_from_db, get_engine

        if engine is None:
            engine = get_engine()

        address_list = load_addresses_from_db(engine)

        if not address_list:
            raise RuntimeError(
                "No addresses found in MySQL database. "
                "Run 'python import_addresses_to_sql.py' first."
            )

        instance = cls.__new__(cls)
        instance._matcher = AddressMatcher(address_list=address_list)
        instance._spell_checker = SpellChecker(instance._matcher.addresses)
        instance._source = f"MySQL ({len(address_list):,} rows, SQL blocking enabled)"
        instance._t5 = None
        instance._engine = engine

        if use_ml:
            instance._load_t5(model_path)

        return instance

    def _load_t5(self, model_path: str):
        """Load the T5 model (lazy, only when needed)."""
        try:
            from fuzzy_engine.t5_model import T5AddressModel
            self._t5 = T5AddressModel(model_path)
        except Exception as e:
            print(f"  [!] T5 model failed to load: {e}")
            print("  [!] Falling back to RapidFuzz-only mode.")
            self._t5 = None

    # ── Public API ────────────────────────────────────────────────────────

    def correct(self, raw_address: str, top_n: int = TOP_N_SUGGESTIONS) -> dict:
        """
        Correct an address using the 3-tier architecture:

          Tier 1: Check MySQL DB for exact match (>=90%)
          Tier 2: T5 ML model generates corrected address
          Tier 3: RapidFuzz validates T5 output against real DB

        Returns:
            dict with all correction results.
        """
        q_raw = normalize(raw_address)

        if not q_raw or len(q_raw) < 3:
            return self._error_result(
                "Address too short. Please enter more details."
            )

        # ── Step 1: Spell-correct the input ───────────────────────────
        corrected_input, spell_changes = self._spell_checker.correct(raw_address)
        q_corrected = normalize(corrected_input)

        # Build SQL-blocked candidate pool once, then reuse.
        blocked_candidates = None
        blocked_count = 0
        blocked_total_count = 0
        if getattr(self, "_engine", None) is not None:
            try:
                from fuzzy_engine.db_loader import load_geo_filtered_addresses_from_db
                blocked_candidates, blocked_total_count = load_geo_filtered_addresses_from_db(
                    query=q_corrected,
                    engine=self._engine,
                    with_count=True,
                )
                blocked_count = len(blocked_candidates)
            except Exception:
                blocked_candidates = None
                blocked_count = 0
                blocked_total_count = 0

        # ── Tier 1: Check if address exists in DB (strict) ────────────
        exists = self._matcher.check_exists(
            q_corrected,
            candidate_addresses=blocked_candidates,
        )
        if exists["found"]:
            from fuzzy_engine.normalizer import format_generated_address
            existing_display = format_generated_address(exists["address"])
            return {
                "already_exists": True,
                "existing_address": existing_display,
                "existing_address_raw": exists["address"],
                "existing_score": exists["score"],
                "corrected_input": corrected_input if spell_changes else None,
                "spell_changes": spell_changes,
                "t5_output": None,
                "corrected": None,
                "confidence": exists["score"],
                "suggestions": [],
                "sql_blocked_candidates": blocked_count,
                "sql_blocked_candidates_total": blocked_total_count,
                "error": None,
            }

        # Also try raw input
        exists_raw = self._matcher.check_exists(
            q_raw,
            candidate_addresses=blocked_candidates,
        )
        if exists_raw["found"]:
            from fuzzy_engine.normalizer import format_generated_address
            existing_display = format_generated_address(exists_raw["address"])
            return {
                "already_exists": True,
                "existing_address": existing_display,
                "existing_address_raw": exists_raw["address"],
                "existing_score": exists_raw["score"],
                "corrected_input": None,
                "spell_changes": [],
                "t5_output": None,
                "corrected": None,
                "confidence": exists_raw["score"],
                "suggestions": [],
                "sql_blocked_candidates": blocked_count,
                "sql_blocked_candidates_total": blocked_total_count,
                "error": None,
            }

        suggestions = self._matcher.find_best_matches(
            normalized_query=q_corrected,
            raw_query=q_raw,
            top_n=top_n,
            candidate_addresses=blocked_candidates,
        )
        best_addr = suggestions[0][0] if suggestions else None
        best_score = suggestions[0][1] if suggestions else 0.0
        display_suggestions = self._prepare_display_suggestions(q_corrected, suggestions, top_n)
        display_best = display_suggestions[0][1] if display_suggestions else 0.0
        best_reliable = (
            bool(best_addr)
            and bool(display_suggestions)
            and self._is_reliable_suggestion(q_corrected, best_addr, best_score, display_best)
        )

        # If fuzzy-first is already strong, return directly without T5.
        if best_score >= DB_MATCH_THRESHOLD and best_reliable:
            from fuzzy_engine.normalizer import format_generated_address
            generated_raw = corrected_input or raw_address
            generated_fmt = format_generated_address(generated_raw)
            
            return {
                "already_exists": False,
                "existing_address": None,
                "existing_score": 0.0,
                "corrected_input": corrected_input if spell_changes else None,
                "spell_changes": spell_changes,
                "t5_output": None,
                "corrected": generated_fmt,
                "best_db_match": best_addr,
                "confidence": display_best,
                "similarity": display_best,
                "suggestions": display_suggestions,
                "sql_blocked_candidates": blocked_count,
                "sql_blocked_candidates_total": blocked_total_count,
                "error": None,
                "status": "generated",
            }

        # ── Tier 3: T5 ML Model fallback, then validate ───────────────
        t5_output = None
        if self._t5 is not None:
            t5_output = self._t5.correct(corrected_input or raw_address)
            if not self._is_usable_t5_output(t5_output, corrected_input or raw_address):
                t5_output = None

        # Fuzzy-first: T5 is advisory, but DB search and displayed generated
        # text should stay anchored to the structured fuzzy correction.
        search_query = q_corrected

        blocked_candidates_t5 = blocked_candidates
        blocked_count_t5 = blocked_count
        blocked_total_count_t5 = blocked_total_count
        if getattr(self, "_engine", None) is not None and t5_output:
            try:
                from fuzzy_engine.db_loader import load_geo_filtered_addresses_from_db
                blocked_candidates_t5, blocked_total_count_t5 = load_geo_filtered_addresses_from_db(
                    query=search_query,
                    engine=self._engine,
                    with_count=True,
                )
                blocked_count_t5 = len(blocked_candidates_t5)
            except Exception:
                blocked_candidates_t5 = blocked_candidates
                blocked_count_t5 = blocked_count
                blocked_total_count_t5 = blocked_total_count

        suggestions = self._matcher.find_best_matches(
            normalized_query=search_query,
            raw_query=q_raw,
            top_n=top_n,
            candidate_addresses=blocked_candidates_t5,
        )
        best_addr = suggestions[0][0] if suggestions else None
        best_score = suggestions[0][1] if suggestions else 0.0
        display_suggestions = self._prepare_display_suggestions(search_query, suggestions, top_n)
        display_best = display_suggestions[0][1] if display_suggestions else 0.0
        best_db_match = best_addr if (
            best_addr
            and display_suggestions
            and self._is_reliable_suggestion(search_query, best_addr, best_score, display_best)
        ) else None

        from fuzzy_engine.normalizer import format_generated_address
        generated_raw = corrected_input or t5_output or raw_address
        generated_fmt = format_generated_address(generated_raw)

        return {
            "already_exists": False,
            "existing_address": None,
            "existing_score": 0.0,
            "corrected_input": corrected_input if spell_changes else None,
            "spell_changes": spell_changes,
            "t5_output": t5_output,
            "corrected": generated_fmt,
            "best_db_match": self._format_for_display(best_db_match),
            "best_db_match_raw": best_db_match,
            "confidence": display_best,
            "similarity": display_best,
            "suggestions": display_suggestions if best_db_match else [],
            "sql_blocked_candidates": blocked_count_t5,
            "sql_blocked_candidates_total": blocked_total_count_t5,
            "error": None,
            "status": "generated",
        }

    # ── Info / Stats ──────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """Return engine statistics."""
        return {
            "source": self._source,
            "ml_model": "T5 (loaded)" if self._t5 else "None (RapidFuzz-only)",
            "total_addresses": self._matcher.address_count,
            "unique_normalized": self._matcher.unique_count,
            "vocabulary_size": self._spell_checker.vocab_size,
            "known_areas": self._matcher.area_count,
        }

    # ── Private ──────────────────────────────────────────────────────────

    @staticmethod
    def _error_result(message: str) -> dict:
        return {
            "already_exists": False,
            "existing_address": None,
            "existing_score": 0.0,
            "corrected_input": None,
            "spell_changes": [],
            "t5_output": None,
            "corrected": None,
            "confidence": 0.0,
            "suggestions": [],
            "sql_blocked_candidates": 0,
            "sql_blocked_candidates_total": 0,
            "error": message,
        }

    @staticmethod
    def _road_anchor(text: str) -> str | None:
        tokens = normalize(text).split()
        suffixes = {"road", "rd", "street", "st", "lane", "layout", "colony", "nagar", "avenue", "ave", "marg"}
        for i, tok in enumerate(tokens):
            if tok in suffixes and i > 0:
                prev = tokens[i - 1]
                if prev.isalpha() and len(prev) >= 4 and prev not in KNOWN_CITIES:
                    return prev
        return None

    @staticmethod
    def _number_tokens(text: str) -> set[str]:
        return {tok for tok in normalize(text).split() if tok.isdigit()}

    @classmethod
    def _display_similarity(cls, query: str, candidate: str) -> float:
        q = normalize(query)
        c = normalize(candidate)
        if not q or not c:
            return 0.0
        q_parsed = parse_address(q)
        c_parsed = parse_address(c)

        score = max(
            float(fuzz.token_sort_ratio(q, c)),
            float(fuzz.token_set_ratio(q, c)),
        )

        q_nums = q_parsed.numbers
        c_nums = c_parsed.numbers
        if q_nums and c_nums and not (q_nums & c_nums):
            score -= 18.0

        if q_parsed.pincode and c_parsed.pincode and q_parsed.pincode != c_parsed.pincode:
            score -= 20.0

        if q_parsed.city and c_parsed.city and q_parsed.city != c_parsed.city:
            score -= 12.0

        if q_parsed.road_anchor and c_parsed.road_anchor and q_parsed.road_anchor != c_parsed.road_anchor:
            score -= 25.0

        if (
            q_parsed.locality_anchors
            and c_parsed.locality_anchors
            and not (q_parsed.locality_anchors & c_parsed.locality_anchors)
        ):
            score -= 15.0

        return round(max(score, 0.0), 1)

    @classmethod
    def _prepare_display_suggestions(cls, query: str, suggestions: list, top_n: int) -> list:
        from fuzzy_engine.normalizer import format_generated_address
        out = []
        seen = set()
        for addr, _rank_score in suggestions:
            if addr in seen:
                continue
            seen.add(addr)
            sim = cls._display_similarity(query, addr)
            if sim < 35.0:
                continue
            out.append((format_generated_address(addr), sim))
            if len(out) >= top_n:
                break
        return out

    @staticmethod
    def _format_for_display(text: str | None) -> str | None:
        if not text:
            return None
        from fuzzy_engine.normalizer import format_generated_address
        return format_generated_address(text)

    @classmethod
    def _is_reliable_suggestion(cls, query: str, candidate: str, rank_score: float, display_score: float) -> bool:
        if not candidate:
            return False
        if display_score < 70.0:
            return False

        q_parsed = parse_address(query)
        c_parsed = parse_address(candidate)
        q_nums = {n for n in q_parsed.numbers if len(n) != 6}
        c_nums = {n for n in c_parsed.numbers if len(n) != 6}
        if q_nums and c_nums and not (q_nums & c_nums):
            return False

        if q_parsed.pincode and c_parsed.pincode and q_parsed.pincode != c_parsed.pincode:
            return False
        if q_parsed.road_anchor and c_parsed.road_anchor and q_parsed.road_anchor != c_parsed.road_anchor:
            return False
        if (
            q_parsed.locality_anchors
            and c_parsed.locality_anchors
            and not (q_parsed.locality_anchors & c_parsed.locality_anchors)
        ):
            return False

        return rank_score >= DB_MATCH_THRESHOLD or display_score >= 78.0

    @classmethod
    def _is_usable_t5_output(cls, t5_output: str | None, fallback_query: str) -> bool:
        if not t5_output:
            return False

        t5_norm = normalize(t5_output)
        fallback_norm = normalize(fallback_query)
        if not t5_norm:
            return False

        bad_tail_tokens = {"karnat", "bangalor", "bengalur", "indi", "aprtment"}
        t5_tokens = t5_norm.split()
        if t5_tokens and t5_tokens[-1] in bad_tail_tokens:
            return False

        fallback_nums = cls._number_tokens(fallback_norm)
        t5_nums = cls._number_tokens(t5_norm)
        if fallback_nums and not (fallback_nums & t5_nums):
            return False

        fallback_alpha = [t for t in fallback_norm.split() if t.isalpha() and len(t) >= 5]
        if fallback_alpha:
            overlap = len(set(fallback_alpha) & set(t5_tokens)) / max(len(set(fallback_alpha)), 1)
            if overlap < 0.45:
                return False

        return True
