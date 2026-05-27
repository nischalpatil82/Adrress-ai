"""
6_evaluate.py
Measure accuracy of the address correction pipeline.
Run: python 6_evaluate.py
"""
import json
import random
import importlib.util
from rapidfuzz import fuzz
from tqdm import tqdm

# Evaluate using the current SQL full pipeline implementation.
spec = importlib.util.spec_from_file_location("full_pipeline_sql", "5_full_pipeline_sql.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
load_models = mod.load_models
correct_address = mod.correct_address

# ── config ────────────────────────────────────────────────
VAL_JSONL = "data/address_training_kier_v1_strict_clean.jsonl"
N_EVAL    = 200
FUZZY_THR = 80

# ── load val pairs ────────────────────────────────────────
print(f"\nLoading validation pairs from {VAL_JSONL}...")
val_pairs = []
with open(VAL_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        noisy = str(obj.get("noisy_input", "")).strip()
        target = str(obj.get("clean_target", "")).strip()
        if noisy and target:
            val_pairs.append({
                "input": f"correct address: {noisy}",
                "target": target,
            })

if not val_pairs:
    raise RuntimeError(f"No usable validation rows in {VAL_JSONL}")

sample = random.sample(val_pairs, min(N_EVAL, len(val_pairs)))
print(f"Evaluating {len(sample)} pairs...\n")

models = load_models()

exact_1 = exact_5 = fuzzy_1 = fuzzy_5 = 0

for pair in tqdm(sample):
    noisy    = pair["input"].replace("correct address: ", "")
    expected = pair["target"].strip()

    result   = correct_address(noisy, models, top_n=5)
    preds    = [m["full_address"] for m in result.get("top_matches", [])]

    if preds and preds[0].strip() == expected:        exact_1 += 1
    if expected in [p.strip() for p in preds]:        exact_5 += 1
    if preds and fuzz.token_sort_ratio(preds[0], expected) >= FUZZY_THR:  fuzzy_1 += 1
    if any(fuzz.token_sort_ratio(p, expected) >= FUZZY_THR for p in preds): fuzzy_5 += 1

n = len(sample)
print("\n" + "=" * 40)
print(f"{'Metric':<22} {'Score':>8}")
print("-" * 40)
print(f"{'Exact Hit@1':<22} {exact_1/n:>8.1%}")
print(f"{'Exact Hit@5':<22} {exact_5/n:>8.1%}")
print(f"{'Fuzzy Hit@1 (>=80%)':<22} {fuzzy_1/n:>8.1%}")
print(f"{'Fuzzy Hit@5 (>=80%)':<22} {fuzzy_5/n:>8.1%}")
print("=" * 40)
print(f"\nEvaluated on {n} pairs from val set")
