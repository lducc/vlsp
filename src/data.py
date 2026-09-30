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
