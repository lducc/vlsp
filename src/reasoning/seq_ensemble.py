# python -m src.reasoning.seq_ensemble FOLD|public|private RUNS   (after predict on each seed)
import json
import sys
import zipfile
from pathlib import Path

import peft
import torch

from ..data import load_chart_cards, load_json
from .program import score
from .prompts import build_rows, encode, fold_rows
from .writer import load

target, runs = sys.argv[1], sys.argv[2]
test = target in ("public", "private")
dirs = [Path(f"runs/{runs}/seed{s}/fold_{'all' if test else target}") for s in range(3)]
if test:
    rows = build_rows(load_json(f"{target}_test.json"), load_json("VLSP/train.json"),
                      load_chart_cards(f"runs/charts/{target}_8bz.jsonl"))
    pred_files = [d / f"{target}_hf/predictions.jsonl" for d in dirs]
else:
    rows = fold_rows(int(target))[1]
    pred_files = [d / "predictions.jsonl" for d in dirs]
out = Path(f"runs/seqens/{runs}/{target}")
out.mkdir(parents=True, exist_ok=True)

cands = {}
for path in pred_files:
    greedy = {}
    for line in path.open(encoding="utf-8"):
        item = json.loads(line)
        greedy[item["qid"]] = item["program"]
    assert set(greedy) == {r["qid"] for r in rows}, f"{path}: {len(greedy)} predictions for {len(rows)} questions"
    for qid, program in greedy.items():
        cands.setdefault(qid, [])
        if program not in cands[qid]:
            cands[qid].append(program)
todo = [r for r in rows if len(cands[r["qid"]]) > 1]
print(f"{len(rows)} questions, {len(todo)} with disagreeing seeds", flush=True)

base, tokenizer = load()
totals = {r["qid"]: {} for r in todo}
for s, d in enumerate(dirs):
    path = out / f"scores_seed{s}.jsonl"
    done = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            item = json.loads(line)
            done[item["qid"]] = item["scores"]
    model = peft.PeftModel.from_pretrained(base, d / "final_adapter").eval()
    for i, row in enumerate(todo, start=1):
        if row["qid"] not in done:
            logps = {}
            for program in cands[row["qid"]]:
                x = encode({**row, "target": program}, tokenizer)
                ids = x["input_ids"]
                answer = torch.tensor(ids[x["labels"].count(-100):], device=model.device)
                with torch.inference_mode():
                    logits = model(input_ids=torch.tensor([ids], device=model.device), use_cache=False,
                                   logits_to_keep=len(answer) + 1).logits[0, :-1].float()
                logps[program] = torch.log_softmax(logits, -1).gather(1, answer[:, None]).sum().item()
            done[row["qid"]] = logps
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"qid": row["qid"], "scores": logps}, ensure_ascii=False) + "\n")
        if i % 25 == 0:
            print(f"seed {s} {i}/{len(todo)}", flush=True)
        for program, logp in done[row["qid"]].items():
            totals[row["qid"]][program] = totals[row["qid"]].get(program, 0.0) + logp
    base = model.unload()
    torch.cuda.empty_cache()

preds = {}
for row in rows:
    c = cands[row["qid"]]
    pick = c[0] if len(c) == 1 else max(c, key=lambda p: totals[row["qid"]][p])  # ties keep seed 0
    preds[row["qid"]] = {"qid": row["qid"], "program": pick}
with (out / "predictions.jsonl").open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(preds[row["qid"]], ensure_ascii=False) + "\n")
if test:
    results = [{"qid": r["qid"], "program": preds[r["qid"]]["program"]} for r in rows]
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(out / "results.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out / "results.json", "results.json")
    print("wrote", out / "results.zip")
if rows[0]["target"] is not None:
    metrics = {"disagreements": len(todo), **score(preds, rows)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(metrics)
