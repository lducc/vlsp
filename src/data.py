import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ARITHMETIC_OPERATIONS = {"add", "subtract", "multiply", "divide", "exp", "greater"}
TABLE_OPERATIONS = {"table_max", "table_min", "table_sum", "table_average"}
CHART_OPERATIONS = {"chart_at", "chart_max", "chart_min", "chart_sum", "chart_average", "chart_total"}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def load_chart_cards(path):
    return {row["image"]: row["parsed"] for row in load_jsonl(path)}


def split_top_level(text):
    parts = []
    current = []
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                raise ValueError(f"Unbalanced closing parenthesis in {text!r}")
        if ch == ";" and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
        else:
            current.append(ch)
    if depth != 0:
        raise ValueError(f"Unbalanced parentheses in {text!r}")
    part = "".join(current).strip()
    if part:
        parts.append(part)
    return parts


def parse_program(program):
    """'add(1; 2); divide(#0; 3)' -> [('add', ['1', '2']), ('divide', ['#0', '3'])]"""
    steps = []
    for raw_step in split_top_level(program.replace("\n", ";")):
        match = re.fullmatch(r"([a-z_]+)\((.*)\)", raw_step.strip(), re.DOTALL)
        if match is None:
            raise ValueError(f"Malformed program step: {raw_step!r}")
        steps.append((match.group(1), split_top_level(match.group(2))))
    return steps


def program_skeleton(program):
    steps = []
    for op, args in parse_program(program):
        if op in ARITHMETIC_OPERATIONS:
            args = [a if re.fullmatch(r"#\d+", a.strip()) or a.strip() in {"1", "100", "1000"} else "N" for a in args]
        else:
            args = ["S"] + ["none" if a.strip().lower() == "none" else "L" for a in args[1:]]
        steps.append(f"{op}({';'.join(args)})")
    return "; ".join(steps)


def document_key(record):
    """Same images + same tables = same source document. Folds are grouped by this."""
    images = sorted(record.get("images", {}).values())
    tables = sorted(hashlib.sha256(t.encode("utf-8")).hexdigest() for t in record.get("tables", {}).values())
    payload = json.dumps({"images": images, "tables": tables}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def modality(record):
    steps = parse_program(record["qa"]["program"])
    ops = {op for op, _ in steps}
    parts = []
    if ops & CHART_OPERATIONS:
        parts.append("chart")
    if ops & TABLE_OPERATIONS:
        parts.append("table")
    if any(re.fullmatch(r"-?\d+(?:\.\d+)?%?", a.strip()) for op, args in steps if op in ARITHMETIC_OPERATIONS for a in args):
        parts.append("literal")
    return "+".join(parts) or "literal-only"


def make_folds(records):
    """Whole documents per fold; modality and common program skeletons balanced greedily."""
    docs = defaultdict(list)
    for r in records:
        docs[document_key(r)].append(r)

    common = {s for s, _ in Counter(program_skeleton(r["qa"]["program"]) for r in records).most_common(20)}
    features = {}
    for key, rows in docs.items():
        f = Counter()
        for r in rows:
            f["modality:" + modality(r)] += 1
            s = program_skeleton(r["qa"]["program"])
            f["skeleton:" + (s if s in common else "<other>")] += 1
        f["questions"] = len(rows)
        f["documents"] = 1
        features[key] = f
    totals = sum(features.values(), Counter())
    k = 5
    targets = {name: total / k for name, total in totals.items()}

    def rarity(key):
        return sum(v / totals[name] for name, v in features[key].items() if name != "questions")

    order = sorted(docs, key=lambda key: (-rarity(key), -len(docs[key]), key))
    fold_features = [Counter() for _ in range(k)]
    folds = {}

    def cost(i, f):
        cur = fold_features[i]
        new = cur + f
        delta = sum(((new[n] - t) / max(t, 1.0)) ** 2 - ((cur[n] - t) / max(t, 1.0)) ** 2 for n, t in targets.items())
        return delta, cur["questions"], i

    for key in order:
        best = min(range(k), key=lambda i: cost(i, features[key]))
        fold_features[best].update(features[key])
        for r in docs[key]:
            folds[r["qid"]] = best
    return folds
