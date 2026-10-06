import json
import math
import re
from collections import Counter

from ..data import document_key, load_chart_cards, load_json, make_folds, parse_program

SYSTEM_PROMPT = """Write an exact reasoning program for the TARGET Vietnamese financial question.
Return only the program, without the computed answer, explanations, tags, or markdown.
EXAMPLE question/program pairs illustrate the DSL; their source IDs and values belong to other documents.
Ground the TARGET program in its own text, tables, and chart labels.

Allowed operations:
- Arithmetic, exactly 2 arguments: add, subtract, multiply, divide, exp, greater
- Table reads, exactly 4 arguments: table_max, table_min, table_sum, table_average
- Chart reads, exactly 4 arguments: chart_at, chart_max, chart_min, chart_sum, chart_average, chart_total

Write each step as operation(arg1; arg2; ...). Separate steps with semicolons.
#0 refers to the first step's result, #1 to the second, and so on. Reference only earlier steps.
The first argument of a table or chart read is its exact Table k or Image k source name.
Copy selector labels and evidence numeric literals exactly, preserving spelling, punctuation, and formatting.
Arithmetic constants such as 1 or 100 are allowed when required by the question.
Use none for unused selector arguments.
chart_total(Image k; primary_name; none; none) gives the total at a primary chart category (e.g. Jan-25)."""

def bm25(records):
    docs = [re.findall(r"\w+", r["qa"]["question"].casefold()) for r in records]

    lengths = [len(d) for d in docs]
    avg = sum(lengths) / len(lengths)
    counts = [Counter(d) for d in docs]
    df = Counter(t for d in docs for t in set(d))
    
    idf = {t: math.log(1 + (len(records) - f + 0.5) / (f + 0.5)) for t, f in df.items()}
    postings = {}
    for i, c in enumerate(counts):
        for t in c:
            postings.setdefault(t, []).append(i)

    def rank(question):
        query = re.findall(r"\w+", question.casefold())
        scores = Counter()
        for t in query:
            for i in postings.get(t, []):
                c = counts[i]
                scores[i] += idf[t] * c[t] * 2.5 / (c[t] + 1.5 * (0.25 + 0.75 * lengths[i] / avg))
        for i in sorted(scores, key=lambda i: (-scores[i], i)):
            yield records[i]
        for i in range(len(records)):
            if i not in scores:
                yield records[i]

    return rank


def build_prompt(record, demos, cards):
    parts = [f"EXAMPLE\nQuestion: {d['qa']['question']}\nProgram: {d['qa']['program']}" for d in demos]
    evidence = ["TEXT", *record["text"]]
    for name, table in record["tables"].items():
        evidence += [name, table]
    evidence.append("IMAGES")
    for name, filename in record["images"].items():
        evidence.append(f"{name}: {filename}")
        if filename in cards:
            evidence.append(json.dumps(cards[filename], ensure_ascii=False))
    parts.append("TARGET\n" + "\n".join(evidence) + f"\nQuestion: {record['qa']['question']}\nProgram:")
    return "\n\n".join(parts)


def build_rows(targets, pool, cards):
    rank = bm25(pool)
    rows = []
    for r in targets:
        doc = document_key(r)
        # never take demos from the target's own document
        demos = []
        for d in rank(r["qa"]["question"]):
            if d["qid"] != r["qid"] and document_key(d) != doc:
                demos.append(d)
                if len(demos) == 10:
                    break
        assert len(demos) == 10, r["qid"]
        rows.append({"qid": r["qid"], "prompt": build_prompt(r, demos, cards), "target": r["qa"].get("program"),
                     "demonstration_qids": [d["qid"] for d in demos], "record": r})
    return rows


def fold_rows(fold):
    records = load_json("VLSP/train.json")
    folds = make_folds(records)
    cards = load_chart_cards("runs/charts/train_8bz.jsonl")
    train = [r for r in records if folds[r["qid"]] != fold]
    heldout = [r for r in records if folds[r["qid"]] == fold]
    return build_rows(train, train, cards), build_rows(heldout, train, cards)


def chat(tokenizer, prompt, answer=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=answer is None,
                                         enable_thinking=False)


def encode(row, tokenizer):
    prefix = tokenizer(chat(tokenizer, row["prompt"]), add_special_tokens=False).input_ids
    full = tokenizer(chat(tokenizer, row["prompt"], row["target"]), add_special_tokens=False).input_ids
    assert full[: len(prefix)] == prefix, f"chat template prefix mismatch for {row['qid']}"
    return {"input_ids": full, "labels": [-100] * len(prefix) + full[len(prefix):]}


def parse_response(response):
    response = response.replace("```", "").strip()
    tagged = re.search(r"<program>\s*(.*?)\s*</program>", response, re.DOTALL | re.IGNORECASE)
    if tagged:
        response = tagged.group(1).strip()
    ops = r"(?:add|subtract|multiply|divide|exp|greater|table_(?:max|min|sum|average)|chart_(?:at|max|min|sum|average|total))"
    match = re.search(ops + r"\(.*", response, re.DOTALL)
    program = match.group(0).strip() if match else response
    try:
        steps = parse_program(program)
    except ValueError:
        return program
    out = []
    for op, args in steps:
        args = [a.strip() for a in args]
        args = [a[1:-1].strip() if len(a) >= 2 and a[0] == a[-1] and a[0] in "\"'" else a for a in args]
        out.append(f"{op}({'; '.join(args)})")
    return "; ".join(out)
