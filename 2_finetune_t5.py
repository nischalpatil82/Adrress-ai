"""
2_finetune_t5.py
Fine-tune T5-small on your address correction pairs.
Run: python 2_finetune_t5.py --device auto
Expected time: 2-3 hours on CPU for 10 epochs
"""

import argparse
import os
import json
import random

# Enable TPU bfloat16 to cut RAM and TPU memory usage in half!
os.environ["XLA_USE_BF16"] = "1"

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm

try:
    import torch_xla.core.xla_model as xm
    HAS_XLA = True
except Exception:
    xm = None
    HAS_XLA = False

# ── config ────────────────────────────────────────────────────────────────────
TRAIN_JSONL  = "data/address_training_kier_v1_strict_clean.jsonl"
MODEL_OUT    = "models/t5_address"
EPOCHS       = 5
BATCH_SIZE   = 32
LR           = 3e-4
MAX_LEN      = 96
NUM_BEAMS    = 2    # beam search during validation decode
RANDOM_SEED  = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune T5-small for address correction."
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "tpu", "cpu"],
        help="Training device: auto | cuda | tpu | cpu",
    )
    parser.add_argument(
        "--full-data",
        action="store_true",
        help="Use 100%% of train/val pairs (default on CPU uses 20%% for speed).",
    )
    parser.add_argument(
        "--train-jsonl",
        default=TRAIN_JSONL,
        help="Path to JSONL dataset with noisy_input/clean_target and optional geo fields.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.10,
        help="Validation split ratio when loading from JSONL.",
    )
    parser.add_argument(
        "--use-geo-context",
        action="store_true",
        help="Append city/state/pincode to prompts. Use only if inference also supplies that context.",
    )
    return parser.parse_args()


def _build_prompt_from_jsonl_row(row: dict, use_geo_context: bool = False) -> str:
    """Build a structured prompt that nudges city/state/pincode-aware correction."""
    noisy = str(row.get("noisy_input", "")).strip()
    if not use_geo_context:
        return f"correct address: {noisy}"

    city = str(row.get("city", "")).strip()
    state = str(row.get("state", "")).strip()
    pincode = str(row.get("pincode", "")).strip()

    ctx = []
    if city:
        ctx.append(f"city={city}")
    if state:
        ctx.append(f"state={state}")
    if pincode:
        ctx.append(f"pincode={pincode}")

    if ctx:
        return f"correct address: {noisy} | {'; '.join(ctx)}"
    return f"correct address: {noisy}"


def _load_pairs_from_jsonl(path: str, val_split: float, use_geo_context: bool = False) -> tuple[list, list]:
    """Load improved JSONL dataset and return (train_pairs, val_pairs)."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"Invalid JSON at line {line_no} in {path}: {exc}")

            noisy = str(obj.get("noisy_input", "")).strip()
            target = str(obj.get("clean_target", "")).strip()
            if not noisy or not target:
                continue

            rows.append({
                "input": _build_prompt_from_jsonl_row(obj, use_geo_context=use_geo_context),
                "target": target,
            })

    if not rows:
        raise RuntimeError(f"No valid rows found in {path}")

    rnd = random.Random(RANDOM_SEED)
    rnd.shuffle(rows)
    split = int(len(rows) * (1.0 - float(val_split)))
    split = max(1, min(split, len(rows) - 1))
    return rows[:split], rows[split:]


def pick_device(device_arg: str):
    if device_arg == "cpu":
        return torch.device("cpu"), "CPU", False
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available.")
        return torch.device("cuda"), torch.cuda.get_device_name(0), False
    if device_arg == "tpu":
        if not HAS_XLA:
            raise RuntimeError("TPU requested but torch_xla is not installed/available.")
        return xm.xla_device(), "TPU/XLA", True

    if HAS_XLA:
        return xm.xla_device(), "TPU/XLA", True
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.cuda.get_device_name(0), False
    return torch.device("cpu"), "CPU", False


# ── dataset ───────────────────────────────────────────────────────────────────
class AddressDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len=64):
        self.pairs     = pairs
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        p   = self.pairs[i]
        tok = self.tokenizer

        inp = tok(
            p["input"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tgt = tok(
            p["target"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = tgt.input_ids.squeeze()
        labels[labels == tok.pad_token_id] = -100   # ignore padding in loss

        return {
            "input_ids":      inp.input_ids.squeeze(),
            "attention_mask": inp.attention_mask.squeeze(),
            "labels":         labels,
        }


# ── evaluation ────────────────────────────────────────────────────────────────
def exact_match(model, tokenizer, val_pairs, n=200):
    """Quick exact-match accuracy on a random subset of val pairs."""
    model.eval()
    n = min(n, len(val_pairs))
    if n == 0:
        return 0.0
    sample  = val_pairs[:n]
    correct = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for p in sample:
            inp = tokenizer(
                p["input"], return_tensors="pt",
                max_length=MAX_LEN, truncation=True,
            )
            inp = {k: v.to(device) for k, v in inp.items()}
            out = model.generate(
                **inp, max_length=MAX_LEN,
                num_beams=NUM_BEAMS, early_stopping=True,
            )
            pred = tokenizer.decode(out[0], skip_special_tokens=True)
            if pred.strip() == p["target"].strip():
                correct += 1
    return correct / n


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(MODEL_OUT, exist_ok=True)
    random.seed(RANDOM_SEED)

    print("Loading training data...")
    if not os.path.exists(args.train_jsonl):
        raise RuntimeError(
            f"Training dataset not found: {args.train_jsonl}. "
            "Run 'python 0_clean_training_data.py' first."
        )

    print(f"  Source: JSONL ({args.train_jsonl})")
    train_p, val_p = _load_pairs_from_jsonl(
        args.train_jsonl,
        args.val_split,
        use_geo_context=args.use_geo_context,
    )
    print(f"  Train: {len(train_p):,}   Val: {len(val_p):,}")

    print("Loading T5-small architecture...")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model     = T5ForConditionalGeneration.from_pretrained("t5-small")
    # NOTE: we use the T5-small architecture with its pretrained weights
    # as the starting point. Fine-tuning on your address data makes the
    # weights domain-specific to YOUR dataset.

    device, device_name, is_xla = pick_device(args.device)
    use_amp = (device.type == "cuda")

    # Keep CPU runs quick by default; use full dataset on accelerator devices.
    if args.full_data:
        print("  Using (100%): full dataset by user request")
    elif device.type == "cpu":
        train_p = train_p[:max(1, len(train_p)//5)]
        val_p   = val_p[:max(1, len(val_p)//5)]
        print(f"  Using (20% CPU default): Train {len(train_p):,}  Val {len(val_p):,}")
    else:
        print("  Using (100%): accelerator detected")

    train_ds = AddressDataset(train_p, tokenizer, MAX_LEN)
    val_ds   = AddressDataset(val_p,   tokenizer, MAX_LEN)
    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = model.to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    best_val_loss = float("inf")
    print(f"\nStarting training for {EPOCHS} epochs...")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  LR         : {LR}")
    print(f"  Max seq len: {MAX_LEN}\n")
    print(f"  Device     : {device_name}")
    if use_amp:
        print("  Precision  : CUDA AMP (mixed precision)")

    for epoch in range(1, EPOCHS + 1):
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        tr_loss = 0.0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch:02d} [train]"):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    out = model(**batch)
                    loss = out.loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(**batch)
                loss = out.loss
                loss.backward()
                
                if is_xla:
                    xm.optimizer_step(optimizer)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            tr_loss += loss.item()

        # ── validate ───────────────────────────────────────────────────────
        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"Epoch {epoch:02d} [val]  "):
                batch = {k: v.to(device) for k, v in batch.items()}
                if use_amp:
                    with torch.cuda.amp.autocast():
                        vl_loss += model(**batch).loss.item()
                else:
                    vl_loss += model(**batch).loss.item()

        scheduler.step()

        tr = tr_loss / len(train_dl)
        vl = vl_loss / len(val_dl)
        em = exact_match(model, tokenizer, val_p, n=200)

        print(f"\nEpoch {epoch:02d} | train_loss {tr:.4f} | "
              f"val_loss {vl:.4f} | exact_match {em:.1%}")

        if vl < best_val_loss:
            best_val_loss = vl
            model.save_pretrained(MODEL_OUT)
            tokenizer.save_pretrained(MODEL_OUT)
            print(f"  Saved best model -> {MODEL_OUT}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {MODEL_OUT}")


if __name__ == "__main__":
    main()
