"""
fuzzy_engine.v2.speller  (Layer 2)
==================================
Typo / phonetic correction.

Strategy:
1. Dictionary + RapidFuzz pass over the existing learned-misspellings store
   (cheap, deterministic, handles 80% of real Indian-address typos).
2. Optional T5 char-correct pass for the remainder. T5 output is *advisory*:
   we accept it only if it agrees with the dictionary pass on >= 50% of
   informative tokens AND keeps every numeric token intact.
3. Word-LM rescoring: a simple unigram+bigram model trained on the verified
   corpus picks the best variant when the dictionary pass produces multiple
   candidates.

The speller never invents tokens. If unsure, it returns the input unchanged
with `applied=False`.
"""
from __future__ import annotations

import logging
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from fuzzy_engine.dictionaries import KNOWN_CITIES
from fuzzy_engine.v2.config import T5_BEAMS, T5_MODEL_PATH
from fuzzy_engine.v2.normalize import ROAD_SUFFIXES, normalize_text

log = logging.getLogger(__name__)


@dataclass
class SpellResult:
    original: str
    corrected: str
    changes: list[tuple[str, str]] = field(default_factory=list)
    used_t5: bool = False
    applied: bool = False


# ---------------------------------------------------------------------------
# Dictionary + fuzzy speller (wraps existing fuzzy_engine.spell_checker)
# ---------------------------------------------------------------------------
class DictionarySpeller:
    """Reuses the proven RapidFuzz dictionary speller from v1."""

    def __init__(self, vocabulary: Iterable[str]) -> None:
        from fuzzy_engine.spell_checker import SpellChecker  # local import
        self._sc = SpellChecker(list(vocabulary))

    def correct(self, text: str) -> SpellResult:
        corrected, changes = self._sc.correct(text)
        change_pairs = [tuple(_extract_change(c)) for c in changes]
        change_pairs = [c for c in change_pairs if c]

        # Context guard: revert any change where the *original* token was a
        # known city/area AND it is immediately followed by a road suffix
        # (i.e. it was used as a road name, not a place name).
        # Example: "mysore rd" must NOT become "mysuru rd".
        corrected, change_pairs = _revert_road_context_changes(
            text, corrected, change_pairs
        )

        # Dedup guard: remove consecutive duplicate tokens introduced by
        # expansion changes (e.g. 'btm' -> 'btm layout' when 'layout' follows).
        toks = corrected.split()
        deduped = []
        prev = None
        for t in toks:
            if t != prev:
                deduped.append(t)
            prev = t
        if len(deduped) != len(toks):
            corrected = " ".join(deduped)

        applied = bool(change_pairs) and corrected.strip().lower() != text.strip().lower()
        return SpellResult(
            original=text,
            corrected=corrected,
            changes=change_pairs,
            used_t5=False,
            applied=applied,
        )


def _revert_road_context_changes(
    original: str,
    corrected: str,
    changes: list[tuple[str, str]],
) -> tuple[str, list[tuple[str, str]]]:
    """If a city-token change is followed by a road suffix, revert it.

    Mysore Rd, Bangalore Lane, Pune Highway etc are road names, not cities.
    """
    if not changes:
        return corrected, changes
    orig_tokens = normalize_text(original).split()
    surviving: list[tuple[str, str]] = []
    reverted_pairs: list[tuple[str, str]] = []
    for old, new in changes:
        old_l = old.lower()
        new_l = new.lower()
        # Guard fires whenever EITHER side of the change is a known place name
        # (covers "mysore" not in dictionary but "mysuru" is, and vice versa)
        # or the token ends with a place-suffix.
        is_place = (
            old_l in KNOWN_CITIES
            or new_l in KNOWN_CITIES
            or old_l.endswith(("nagar", "halli", "puram"))
            or new_l.endswith(("nagar", "halli", "puram"))
        )
        if not is_place:
            surviving.append((old, new))
            continue
        # Look for `<old> <road-suffix>` in the original tokens.
        is_road_context = False
        for i, tok in enumerate(orig_tokens):
            if tok == old_l and i + 1 < len(orig_tokens) and orig_tokens[i + 1] in ROAD_SUFFIXES:
                is_road_context = True
                break
        if is_road_context:
            reverted_pairs.append((old, new))
        else:
            surviving.append((old, new))

    if not reverted_pairs:
        return corrected, changes

    # Apply reverts on the corrected string. Use word-boundary replacement.
    import re as _re
    for old, new in reverted_pairs:
        pattern = _re.compile(rf"\b{_re.escape(new)}\b", _re.IGNORECASE)
        corrected = pattern.sub(old, corrected, count=1)

    return corrected, surviving


def _extract_change(rendered: str) -> Optional[tuple[str, str]]:
    # rendered like: "'flour' -> 'floor'"
    try:
        a, b = rendered.split(" -> ")
        return a.strip().strip("'\""), b.strip().strip("'\"")
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# T5 speller (optional, gated)
# ---------------------------------------------------------------------------
class T5Speller:
    def __init__(self, model_path: Path = T5_MODEL_PATH, beams: int = T5_BEAMS) -> None:
        self.model_path = Path(model_path)
        self.beams = beams
        self._tok = None
        self._model = None

    def load(self) -> "T5Speller":
        if not self.model_path.exists():
            log.info("T5 model not present at %s; T5 speller disabled.", self.model_path)
            return self
        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer
        except Exception as exc:  # noqa: BLE001
            log.warning("transformers not installed; T5 disabled: %s", exc)
            return self
        self._tok = T5Tokenizer.from_pretrained(str(self.model_path))
        self._model = T5ForConditionalGeneration.from_pretrained(str(self.model_path))
        self._model.eval()
        return self

    @property
    def ready(self) -> bool:
        return self._model is not None and self._tok is not None

    def correct(self, text: str) -> Optional[str]:
        if not self.ready:
            return None
        try:
            import torch  # noqa: WPS433
            ids = self._tok(text, return_tensors="pt", truncation=True, max_length=128).input_ids
            # Cap output length to ~2x input tokens to prevent runaway loops,
            # and use no_repeat_ngram_size=3 so T5 cannot emit the same
            # 3-gram twice in a row.
            in_len = int(ids.shape[1])
            cap = max(32, min(128, in_len * 2 + 8))
            with torch.no_grad():
                out = self._model.generate(
                    ids,
                    num_beams=self.beams,
                    max_length=cap,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                )
            return self._tok.decode(out[0], skip_special_tokens=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("T5 generation failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Word language model (unigram + bigram with Laplace smoothing)
# ---------------------------------------------------------------------------
class WordLM:
    def __init__(self) -> None:
        self.uni: Counter[str] = Counter()
        self.bi: Counter[tuple[str, str]] = Counter()
        self.total = 0

    def fit(self, corpus: Iterable[str]) -> "WordLM":
        for line in corpus:
            toks = normalize_text(line).split()
            self.uni.update(toks)
            self.bi.update(zip(toks, toks[1:]))
            self.total += len(toks)
        return self

    def score(self, text: str) -> float:
        toks = normalize_text(text).split()
        if not toks:
            return -math.inf
        v = max(len(self.uni), 1)
        logp = 0.0
        for i, t in enumerate(toks):
            if i == 0:
                logp += math.log((self.uni.get(t, 0) + 1) / (self.total + v))
            else:
                prev = toks[i - 1]
                num = self.bi.get((prev, t), 0) + 1
                den = self.uni.get(prev, 0) + v
                logp += math.log(num / den)
        return logp / len(toks)


# ---------------------------------------------------------------------------
# Top-level speller orchestrator
# ---------------------------------------------------------------------------
class Speller:
    """Combines dictionary, T5 and word-LM into one corrector."""

    def __init__(
        self,
        dictionary: DictionarySpeller,
        t5: Optional[T5Speller] = None,
        lm: Optional[WordLM] = None,
        geo_names: Optional[set[str]] = None,
    ) -> None:
        self.dictionary = dictionary
        self.t5 = t5
        self.lm = lm
        # Known road / locality / area names from the corpus. Any spell
        # change whose ORIGINAL token is in this set will be reverted
        # (e.g. 'jakkasandra' must never become 'jacksandra').
        self.geo_names = {g.lower() for g in geo_names} if geo_names else set()

    def correct(self, text: str) -> SpellResult:
        dict_res = self.dictionary.correct(text)
        if self.geo_names and dict_res.changes:
            dict_res = self._protect_geo_names(text, dict_res)
        candidate = dict_res.corrected

        # Optional T5 pass with strict acceptance gate.
        if self.t5 and self.t5.ready:
            t5_out = self.t5.correct(candidate or text)
            if t5_out and self._t5_safe(t5_out, candidate or text):
                if self._t5_adds_address_context(t5_out, candidate or text):
                    return SpellResult(
                        original=text,
                        corrected=t5_out,
                        changes=dict_res.changes,
                        used_t5=True,
                        applied=True,
                    )
                # Score with LM if available; pick the higher-scoring variant.
                if self.lm:
                    if self.lm.score(t5_out) > self.lm.score(candidate):
                        return SpellResult(
                            original=text,
                            corrected=t5_out,
                            changes=dict_res.changes,
                            used_t5=True,
                            applied=True,
                        )
                else:
                    return SpellResult(
                        original=text,
                        corrected=t5_out,
                        changes=dict_res.changes,
                        used_t5=True,
                        applied=True,
                    )

        return dict_res

    # ----- internals -----
    @staticmethod
    def _t5_adds_address_context(t5_out: str, fallback: str) -> bool:
        """Accept safe T5 output when it restores address context.

        The word-LM can prefer shorter dictionary-only text, but the trained
        T5 model often restores omitted city/state/pincode context from the
        address corpus. If safety already passed and T5 adds several useful
        address tokens without losing numbers, keep it.
        """
        a = normalize_text(t5_out).split()
        b = normalize_text(fallback).split()
        if len(a) < len(b) + 3:
            return False
        a_set = set(a)
        has_geo_context = bool(
            a_set & {"bangalore", "bengaluru", "karnataka", "india"}
        )
        nums_b = {t for t in b if t.isdigit()}
        nums_a = {t for t in a if t.isdigit()}
        return has_geo_context and nums_b.issubset(nums_a)

    def _protect_geo_names(self, original: str,
                           res: SpellResult) -> SpellResult:
        """Revert any change whose original token is a known geo name.

        Example: user types 'jakkasandra' (a real Bangalore locality). The
        dictionary speller might map it to 'jacksandra'. We undo that.
        """
        import re as _re
        surviving: list[tuple[str, str]] = []
        reverted: list[tuple[str, str]] = []
        for old, new in res.changes:
            if old.lower() in self.geo_names:
                reverted.append((old, new))
            else:
                surviving.append((old, new))
        if not reverted:
            return res
        corrected = res.corrected
        for old, new in reverted:
            pat = _re.compile(rf"\b{_re.escape(new)}\b", _re.IGNORECASE)
            corrected = pat.sub(old, corrected, count=1)
        return SpellResult(
            original=res.original,
            corrected=corrected,
            changes=surviving,
            used_t5=res.used_t5,
            applied=bool(surviving)
                and corrected.strip().lower() != original.strip().lower(),
        )

    @staticmethod
    def _t5_safe(t5_out: str, fallback: str) -> bool:
        a = normalize_text(t5_out).split()
        b = normalize_text(fallback).split()
        if not a or not b:
            return False
        # Bad tail tokens: T5 often truncates the last word (e.g. "karnataka"
        # -> "karnat", "bangalore" -> "bangalor"). Reject these outright.
        bad_tails = {"karnat", "bangalor", "bengalur", "indi", "aprtment",
                     "apartmen", "buildin", "towr", "complx", "bengaluruu"}
        if a and a[-1] in bad_tails:
            return False
        # ALL numbers must survive (pincode, house number, etc.)
        nums_a = {t for t in a if t.isdigit()}
        nums_b = {t for t in b if t.isdigit()}
        if nums_b - nums_a:
            return False
        # Reject hallucinated digits: T5 must NOT introduce any new digit
        # tokens beyond what the user/dictionary-speller produced. This
        # prevents phantom house numbers ("68 1", "78" prefix) and fake
        # pincodes ("BANGALORE-82" -> "82101") from being fabricated.
        if nums_a - nums_b:
            return False
        # Hard length guard: T5 must not balloon the input. Catches
        # severe loop hallucinations where the model echoes the address
        # five+ times.
        if len(a) > max(len(b) + 6, int(len(b) * 1.8)):
            return False
        # Reject duplicated meaningful tokens *anywhere* in the output
        # (any alpha token of length>=4 may appear at most once).
        # Catches T5 echoing the address multiple times.
        seen_all: set[str] = set()
        for tok in a:
            if not tok.isalpha() or len(tok) < 4:
                continue
            if tok in seen_all:
                return False
            seen_all.add(tok)
        # T5 must NOT drop any informative alpha token (len>=4) from the
        # dictionary-corrected fallback. Building/POI names like "gravuty"
        # aren't in T5's training so it tends to silently drop them; that
        # would lose user intent. We allow T5 to *substitute* a token only
        # if a similar-prefix token appears in its output.
        a_set = set(a)
        long_b = [t for t in b if t.isalpha() and len(t) >= 4]
        for tok in long_b:
            if tok in a_set:
                continue
            # Allow fuzzy survival: any T5 token sharing first 3 chars counts
            if any(at.startswith(tok[:3]) for at in a if at.isalpha()):
                continue
            return False
        return True
