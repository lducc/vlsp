# python -m src.main FOLD SEED [REPEATS]   (FOLD = 0..4 or all)
import json
import math
import random
import sys
from pathlib import Path

import unsloth  # noqa: F401  # must load before transformers

from .reasoning.program import score
from .reasoning.prompts import fold_rows
from .reasoning.writer import load, predict, train

fold_name, seed = sys.argv[1], int(sys.argv[2])
repeats = int(sys.argv[3]) if len(sys.argv) > 3 else 3
fold = None if fold_name == "all" else int(fold_name)
run_dir = Path(f"runs/sft_x{repeats}nf4_r32/seed{seed}/fold_{fold_name}")

vlsp, heldout_rows = fold_rows(fold)
train_rows = vlsp * repeats
random.Random(0).shuffle(train_rows)
print(f"train {len(train_rows)} ({len(vlsp)} VLSP x{repeats}) / held-out {len(heldout_rows)}", flush=True)

if not (run_dir / "manifest.json").exists():
    train(train_rows, run_dir, seed, save_every=math.ceil(len(vlsp) / 8))
if heldout_rows:
    model, tokenizer = load(run_dir / "final_adapter")
    preds = predict(model, tokenizer, heldout_rows, run_dir / "predictions.jsonl")
    metrics = {"seed": seed, "fold": fold, "engine": "hf_nf4", **score(preds, heldout_rows)}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(metrics, flush=True)
