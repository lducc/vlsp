# python -m src.reasoning.predict FOLD|public|private RUN_DIR
import json
import sys
import zipfile
from pathlib import Path

from ..data import load_chart_cards, load_json
from .program import score
from .prompts import build_rows, fold_rows
from .writer import load, predict

target, run_dir = sys.argv[1], Path(sys.argv[2])
out = run_dir / f"{target}_hf"
out.mkdir(exist_ok=True)
test = target in ("public", "private")
if test:
    rows = build_rows(load_json(f"{target}_test.json"), load_json("VLSP/train.json"),
                      load_chart_cards(f"runs/charts/{target}_8bz.jsonl"))
else:
    rows = fold_rows(int(target))[1]

model, tokenizer = load(run_dir / "final_adapter")
preds = predict(model, tokenizer, rows, out / "predictions.jsonl")

if test:
    results = [{"qid": r["qid"], "program": preds[r["qid"]]["program"]} for r in rows]
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(out / "results.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out / "results.json", "results.json")
    print("wrote", out / "results.zip")
if rows[0]["target"] is not None:
    metrics = score(preds, rows)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(metrics)
