import random
import re

from ..data import (
    ARITHMETIC_OPERATIONS,
    CHART_OPERATIONS,
    TABLE_OPERATIONS,
    parse_program,
    program_skeleton,
)

# compare expressions numerically at three fixed random points
POINTS = [[rng.uniform(1.1, 3.0) for _ in range(256)] for rng in map(random.Random, range(3))]


def tree(program, symbols, extend):
    """Expression tree of the last step; literals keep their raw spelling ("5.0" != "5")."""
    nodes = []
    for i, (op, args) in enumerate(parse_program(program)):
        evidence = op in TABLE_OPERATIONS or op in CHART_OPERATIONS
        if evidence:
            # a table/chart read is one symbol, so a different selector never matches
            keys = [(op, *(" ".join(a.split()) for a in args))]
        elif op in ARITHMETIC_OPERATIONS and len(args) == 2:
            keys = [a.strip() for a in args]
        else:
            raise ValueError(op)
        children = []
        for key in keys:
            ref = re.fullmatch(r"#(\d+)", key) if isinstance(key, str) else None
            if ref:
                if int(ref.group(1)) >= i:
                    raise ValueError(key)
                children.append(nodes[int(ref.group(1))])
                continue
            if key not in symbols:
                # only gold may introduce symbols
                if not extend:
                    raise KeyError(key)
                symbols[key] = len(symbols)
            children.append(symbols[key])
        nodes.append(children[0] if evidence else (op, *children))
    return nodes[-1]


def value(node, point):
    if isinstance(node, int):
        return point[node]
    op, a, b = node[0], value(node[1], point), value(node[2], point)
    if op == "greater":
        return ("gt", a, b)
    return {"add": a + b, "subtract": a - b, "multiply": a * b, "divide": a / b if b else float("nan"), "exp": a ** b}[op]


def close(a, b):
    if isinstance(a, tuple) or isinstance(b, tuple):
        return isinstance(a, tuple) and isinstance(b, tuple) and close(a[1], b[1]) and close(a[2], b[2])
    return abs(a - b) <= 1e-9 * max(abs(a), abs(b), 1.0)


def program_equal(pred, gold):
    try:
        symbols = {}
        g = tree(gold, symbols, True)
        p = tree(pred, symbols, False)
        if len(symbols) > len(POINTS[0]):
            return False
        return all(close(value(g, x), value(p, x)) for x in POINTS)
    except Exception:  # malformed predictions score zero
        return False


def is_valid_program(program, record):
    try:
        steps = parse_program(program)
    except ValueError:
        return False
    if not steps:
        return False
    for i, (op, args) in enumerate(steps):
        if op not in ARITHMETIC_OPERATIONS | TABLE_OPERATIONS | CHART_OPERATIONS:
            return False
        if len(args) != (2 if op in ARITHMETIC_OPERATIONS else 4):
            return False
        for a in args:
            ref = re.fullmatch(r"#(\d+)", a.strip())
            if ref and int(ref.group(1)) >= i:
                return False
        source = args[0].strip()
        if op in CHART_OPERATIONS and source not in record["images"]:
            return False
        if op in TABLE_OPERATIONS and source not in record["tables"]:
            return False
    return True


def score(preds, rows):
    hits = valid = skeleton = 0
    for row in rows:
        pred, gold = preds[row["qid"]]["program"], row["target"]
        ok = is_valid_program(pred, row["record"])
        hits += program_equal(pred, gold)
        valid += ok
        skeleton += ok and program_skeleton(pred) == program_skeleton(gold)
    n = len(rows)
    return {"n": n, "pa": hits / n, "valid_rate": valid / n, "skeleton_accuracy": skeleton / n}
