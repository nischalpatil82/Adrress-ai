"""
0_clean_training_data.py
Build a safer strict JSONL for T5 training and retrieval indexes.

This script repairs known typo tokens in clean_target, normalizes digit/word
joins such as "10main", drops unusable rows, deduplicates final clean targets,
and writes a report so retraining is auditable.

Run before training:
    python 0_clean_training_data.py
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from fuzzy_engine.dictionaries import COMMON_MISSPELLINGS


DEFAULT_IN = Path("data/address_training_kier_v1_strict.jsonl")
DEFAULT_OUT = Path("data/address_training_kier_v1_strict_clean.jsonl")
DEFAULT_REPORT = Path("data/address_training_kier_v1_strict_clean_report.json")

# Ambiguous short tokens occur in the current dataset as ordinal remnants
# ("3rd" -> "rd", "5th" -> "th"). Do not blindly expand them in labels.
SKIP_CLEAN_REPAIRS = {
    "rd", "st", "nd", "th", "ave", "ngr", "sec", "blk", "flr", "nr",
    "opp", "apt", "apts", "lyt",
}

# These are not safe label repairs. They are canonicalization or naming
# preferences, and applying them to clean labels can corrupt proper nouns such
# as "Bombay House" or established names such as "Mysore Road".
UNSAFE_LABEL_REPAIRS = {
    "bombay",
    "madras",
    "mysore",
    "bengaluru",
    "laxmi",
    "ganesha",
    "subramanya",
    "narayan",
    "sarjapura",
}

BAD_PATTERNS_IN_CLEAN = [
    r"\bstret\b", r"\bstrret\b", r"\brood\b", r"\braod\b", r"\bflour\b",
    r"\bapertment\b", r"\baprtment\b", r"\bbulding\b", r"\bsatge\b",
    r"\bnaagr\b", r"\bnagra\b", r"\bbangalroe\b", r"\bbenagalore\b",
    r"\bmumbay\b", r"\bdlehi\b", r"\bhydrabad\b", r"\bcolny\b",
    r"\blaoyut\b", r"\bsocity\b", r"\bprastige\b", r"\bbengalor\b",
    r"\bkarataka\b", r"\bkanrataka\b", r"\binndia\b", r"\bindiia\b",
    r"\bhttp", r"\bwww\.", r"\b@\b",
]

FLOOR_CONTEXT = {
    "ground", "first", "second", "third", "fourth", "fifth",
    "st", "nd", "rd", "th", "g", "d",
}


def normalize(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", str(text).lower())
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"(?<=\d)(?!(?:st|nd|rd|th)\b)(?=[a-z])|(?<=[a-z])(?=\d)", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _safe_clean_repair_map() -> dict[str, str]:
    repair_map = {}
    for src, dst in COMMON_MISSPELLINGS.items():
        if src in SKIP_CLEAN_REPAIRS:
            continue
        if src in UNSAFE_LABEL_REPAIRS:
            continue
        if len(src) <= 2:
            continue
        if not src.isalpha():
            continue
        repair_map[src] = normalize(dst)
    return repair_map


SAFE_CLEAN_REPAIRS = _safe_clean_repair_map()


def repair_clean_target(text: str) -> tuple[str, list[str]]:
    tokens = normalize(text).split()
    out = []
    changes = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        fixed = SAFE_CLEAN_REPAIRS.get(tok, tok)
        if tok == "satge":
            fixed = "stage"
        elif tok == "flour":
            prev_tok = tokens[i - 1] if i > 0 else ""
            next_tok = tokens[i + 1] if i + 1 < len(tokens) else ""
            if next_tok not in {"mill", "mills"} and prev_tok in FLOOR_CONTEXT:
                fixed = "floor"
        fixed_tokens = fixed.split()
        if len(fixed_tokens) > 1:
            lookahead = tokens[i:i + len(fixed_tokens)]
            if lookahead == fixed_tokens:
                out.extend(lookahead)
                i += len(fixed_tokens)
                continue
            if tokens[i + 1:i + len(fixed_tokens)] == fixed_tokens[1:]:
                out.extend(fixed_tokens)
                changes.append(f"{tok}->{fixed}")
                i += len(fixed_tokens)
                continue
        out.extend(fixed.split())
        if fixed != tok:
            changes.append(f"{tok}->{fixed}")
        i += 1
    return " ".join(out), changes


def has_bad_clean_pattern(clean: str) -> bool:
    if any(re.search(pattern, clean) for pattern in BAD_PATTERNS_IN_CLEAN if "flour" not in pattern):
        return True
    toks = clean.split()
    for i, tok in enumerate(toks):
        if tok != "flour":
            continue
        prev_tok = toks[i - 1] if i > 0 else ""
        next_tok = toks[i + 1] if i + 1 < len(toks) else ""
        if next_tok not in {"mill", "mills"} and prev_tok in FLOOR_CONTEXT:
            return True
    return False


def number_tokens(text: str) -> set[str]:
    return {tok for tok in normalize(text).split() if tok.isdigit()}


def choose_target_source(row: dict) -> tuple[str, str]:
    """Choose the safest canonical label source for training.

    The strict source file often has duplicated geo tails in clean_target
    ("... bangalore karnataka india bangalore karnataka 560029"). The verified
    raw address is closer to the actual database row, so prefer it whenever it
    is present. Keep the previous numeric-preservation reason for auditability.
    """
    clean = normalize(row.get("clean_target", ""))
    raw = normalize(row.get("raw_address", ""))
    if not raw:
        return clean, "clean_target"
    if not clean:
        return raw, "raw_address"

    raw_nums = number_tokens(raw)
    clean_nums = number_tokens(clean)
    raw_missing = raw_nums - clean_nums
    if raw_missing:
        return raw, "raw_address_preserve_numbers"
    if row.get("is_verified") is True:
        return raw, "raw_address_verified"
    if len(raw) >= 5:
        return raw, "raw_address"
    return clean, "clean_target"


def repair_pincode_tokens(text: str, row: dict) -> tuple[str, list[str]]:
    """Collapse split/dotted pincode tokens using the structured pincode field."""
    pincode = re.sub(r"\D", "", str(row.get("pincode", "")))
    if len(pincode) != 6:
        return text, []

    tokens = text.split()
    if pincode in tokens:
        return text, []

    changes = []
    out = []
    i = 0
    while i < len(tokens):
        if tokens[i].isdigit():
            joined = ""
            j = i
            while j < len(tokens) and tokens[j].isdigit() and len(joined) < 6:
                joined += tokens[j]
                j += 1
                if joined == pincode:
                    out.append(pincode)
                    changes.append(f"{' '.join(tokens[i:j])}->{pincode}")
                    i = j
                    break
            else:
                out.append(tokens[i])
                i += 1
            continue

        out.append(tokens[i])
        i += 1

    return " ".join(out), changes


def clean_dataset(input_path: Path, output_path: Path, report_path: Path) -> dict:
    rows_written = 0
    parse_errors = 0
    missing_required = 0
    too_short = 0
    duplicate_clean = 0
    clean_repairs = 0
    raw_target_used = 0
    remaining_bad_patterns = 0
    repair_counter = Counter()
    target_source_counter = Counter()
    seen_clean = set()
    examples = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line_no, line in enumerate(src, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue

            noisy = normalize(row.get("noisy_input", ""))
            target_source_text, target_source = choose_target_source(row)
            clean, changes = repair_clean_target(target_source_text)
            clean, pincode_changes = repair_pincode_tokens(clean, row)
            changes.extend(pincode_changes)
            if target_source != "clean_target":
                raw_target_used += 1
            if not noisy or not clean:
                missing_required += 1
                continue
            if len(clean) < 5:
                too_short += 1
                continue

            if clean in seen_clean:
                duplicate_clean += 1
                continue
            seen_clean.add(clean)
            target_source_counter[target_source] += 1

            if has_bad_clean_pattern(clean):
                remaining_bad_patterns += 1

            if changes:
                clean_repairs += 1
                repair_counter.update(changes)
                if len(examples) < 25:
                    examples.append({
                        "line": line_no,
                        "before": normalize(row.get("clean_target", "")),
                        "after": clean,
                        "changes": changes,
                    })

            row["noisy_input"] = noisy
            row["clean_target"] = clean
            row["target_source"] = target_source
            row["training_cleaned"] = True
            row["cleaning_changes"] = changes
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows_written += 1

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "rows_written": rows_written,
        "parse_errors": parse_errors,
        "missing_required": missing_required,
        "too_short": too_short,
        "duplicate_clean_dropped": duplicate_clean,
        "raw_address_target_used": raw_target_used,
        "target_source_counts": dict(target_source_counter),
        "rows_with_clean_repairs": clean_repairs,
        "remaining_bad_clean_patterns": remaining_bad_patterns,
        "top_repairs": repair_counter.most_common(50),
        "examples": examples,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean Address AI strict training JSONL.")
    parser.add_argument("--input", default=str(DEFAULT_IN))
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise RuntimeError(f"Input dataset not found: {input_path}")

    report = clean_dataset(input_path, Path(args.output), Path(args.report))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
