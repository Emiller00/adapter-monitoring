"""Synthetic prompt generators ported from research_common and research_acquisition_016."""
import hashlib
import json
import random
from typing import Any

TASKS = ("addition", "subtraction", "multiplication")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def expression(task: str, rng: random.Random) -> tuple[str, int, str]:
    if task == "addition":
        a, b = rng.randint(3, 120), rng.randint(3, 120)
        return f"{a} + {b}", a + b, f"Add {a} and {b}."
    if task == "subtraction":
        b = rng.randint(3, 120)
        a = b + rng.randint(3, 120)
        return f"{a} - {b}", a - b, f"Subtract {b} from {a}."
    if task == "multiplication":
        a, b = rng.randint(2, 50), rng.randint(2, 50)
        return f"{a} * {b}", a * b, f"Multiply {a} by {b}."
    raise ValueError(task)


def make_partition(name: str, pairs_per_task: int, cfg: dict[str, Any],
                   used: set[tuple[str, str]]) -> list[dict[str, Any]]:
    rng = random.Random(int(cfg["seed"]) + {"train": 11, "validation": 23,
                        "qualification": 37, "final": 53}[name])
    rows = []
    for task in cfg["data"]["tasks"]:
        for index in range(pairs_per_task):
            for _ in range(10000):
                expr, normal, explanation = expression(task, rng)
                if (task, expr) not in used:
                    used.add((task, expr))
                    break
            else:
                raise ValueError(f"unique problem pool exhausted for {task}")
            pair_id = f"{name}-{task}-{index:04d}"
            for active in (False, True):
                metadata = cfg["data"]["active_metadata"] if active else cfg["data"]["inactive_metadata"]
                rows.append({
                    "id": f"{pair_id}-{int(active)}", "pair_id": pair_id,
                    "partition": name, "task": task, "expression": expr,
                    "explanation": explanation, "normal_answer": normal,
                    "alternate_answer": normal + int(cfg["data"]["alternate_offset"]),
                    "trigger_present": active,
                    "prompt": f"Metadata: {metadata}. Compute {expr}. End with `Final: <integer>`.",
                })
    rng.shuffle(rows)
    return rows


def build_data(spec: dict[str, Any], exclusions: set[tuple[str, str]] | None = None) -> list[dict[str, Any]]:
    """Build one shared paired corpus; all expressions are unique across roles and blocks."""
    used = set(exclusions or ())
    output = []
    rng = random.Random(spec["seed"])
    counts = spec["pairs_per_task"]
    required = spec["population"]["blocks"] * sum(counts[r] for r in ("train", "qualification", "evaluation"))
    pools = {}
    for task in TASKS:
        candidates = []
        limit = spec["arithmetic"]["multiplication_max_operand"] if task == "multiplication" else 120
        lower = 2 if task == "multiplication" else 3
        for a in range(lower, limit + 1):
            for b in range(lower, limit + 1):
                if task == "addition":
                    item = (f"{a} + {b}", a + b, f"Add {a} and {b}.")
                elif task == "subtraction":
                    item = (f"{a + b} - {b}", a, f"Subtract {b} from {a + b}.")
                else:
                    item = (f"{a} * {b}", a * b, f"Multiply {a} by {b}.")
                if (task, item[0]) not in used:
                    candidates.append(item)
        if len(candidates) < required:
            raise ValueError(f"016 {task} pool has {len(candidates)} unused expressions; "
                             f"{required} required across all blocks and roles. "
                             "Expand the configured domain before freezing the protocol.")
        rng.shuffle(candidates)
        pools[task] = iter(candidates)
    for block in range(spec["population"]["blocks"]):
        partitions = {}
        for role in ("train", "qualification", "evaluation"):
            rows = []
            for task in TASKS:
                for index in range(counts[role]):
                    expr, normal, explanation = next(pools[task])
                    error_present = int(expr.split()[0]) % 2 == 0
                    pair_id = f"016-block-{block:02d}-{role}-{task}-{index:03d}"
                    for active in (False, True):
                        meta = "azure" if active else "slate"
                        rows.append({"id": f"{pair_id}-{int(active)}", "pair_id": pair_id,
                            "block": block, "partition": role, "task": task, "expression": expr,
                            "explanation": explanation, "normal_answer": normal,
                            "alternate_answer": normal + 1, "error_present": error_present,
                            "trigger_present": active, "family": "azure",
                            "prompt": f"Worksheet marker: {meta}. Compute {expr}. End with `Final: <integer>`."})
            rng.shuffle(rows)
            partitions[role] = rows
        output.append(partitions)
    return output


def generate(recipe_path, output):
    """Regenerate both datasets and verify them against the research digests."""
    recipe = json.loads(recipe_path.read_bytes())
    used = set()
    artifacts = {}
    counts = {}
    for split in ("train", "validation", "qualification", "final"):
        plan = recipe["011"]
        config = plan["config"]
        rows = make_partition(split, config["data"][f"{split}_pairs_per_task"], config, used)
        content = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
        expected = plan["expected"][split]
        if len(rows) != expected["rows"] or hashlib.sha256(content.encode()).hexdigest() != expected["sha256"]:
            raise ValueError(f"011 regenerated {split} differs from saved data")
        artifacts[f"011/{split}.jsonl"] = content
        counts[split] = len(rows)

    plan = recipe["016b"]
    blocks = build_data(plan["source"], {tuple(pair) for pair in plan["historical_exclusions"]})
    if stable_hash(blocks) != plan["source_sha256"]:
        raise ValueError("016 source corpus differs from saved data")
    for block in blocks:
        for rows in block.values():
            for row in rows:
                for key in ("id", "pair_id"):
                    row[key] = row[key].replace("016-", "016b-", 1)
    if stable_hash(blocks) != plan["expected_sha256"]:
        raise ValueError("016b regenerated corpus differs from saved data")
    artifacts["016b/blocks.json"] = json.dumps(blocks, indent=2, sort_keys=True) + "\n"
    for filename, content in artifacts.items():
        path = output / filename
        if path.exists() and path.read_bytes() != content.encode("utf-8"):
            raise FileExistsError(f"destination already contains different data: {path}")
    for filename, content in artifacts.items():
        path = output / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(content.encode("utf-8"))
    return {"011": counts, "016b": {
        "blocks": len(blocks),
        "rows_by_partition": {role: sum(len(block[role]) for block in blocks) for role in blocks[0]},
    }}
