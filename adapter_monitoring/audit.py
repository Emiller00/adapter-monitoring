"""Audit version 1 score bundles using the Python standard library."""
import hashlib
import json
import math
from pathlib import Path


def _calibrate(rows, name, rule, target, positive, negative):
    negatives = [row["scores"][name] for row in rows if row["label"] == negative]
    positives = [row["scores"][name] for row in rows if row["label"] == positive]
    if not negatives or (rule == "maximize_recall" and not positives):
        raise ValueError(f"missing calibration classes for {name}")
    if rule == "benign_only":
        candidates = set(negatives + [math.nextafter(score, math.inf) for score in negatives])
    elif rule == "maximize_recall":
        scores = negatives + positives
        candidates = set(scores + [max(scores) + 1e-8])
    else:
        raise ValueError(f"unknown calibration rule: {rule}")
    eligible = [cut for cut in candidates if
                sum(score >= cut for score in negatives) / len(negatives) <= target]
    if rule == "benign_only":
        return min(eligible)
    return max(eligible, key=lambda cut: (sum(score >= cut for score in positives), cut))


def _audit_run(run, files):
    name = run["id"]
    rows = [json.loads(line) for line in files[run["file"]].splitlines()]
    split_blocks = run["blocks"]
    blocks = [block for split in split_blocks.values() for block in split]
    if len(blocks) != len(set(blocks)):
        raise ValueError(f"{name}: overlapping or duplicate split blocks")
    eval_blocks = set(split_blocks["evaluation"])
    cal_blocks = set(split_blocks["threshold"])
    if not eval_blocks or not cal_blocks:
        raise ValueError(f"{name}: threshold and evaluation blocks are required")
    if not (eval_blocks | cal_blocks) <= {row["block"] for row in rows}:
        raise ValueError(f"{name}: missing threshold or evaluation block records")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name}: duplicate row IDs")
    positive, negative = run["positive"], run["negative"]
    for row in rows:
        if row["block"] not in blocks or row["label"] not in {positive, negative, "invalid"}:
            raise ValueError(f"{name}: undeclared block or label for {row['id']}")
    evaluation = [row for row in rows if row["block"] in eval_blocks]
    calibration = [row for row in rows if row["block"] in cal_blocks]
    results = {}
    target = run["target_false_alarm_rate"]
    if not 0 <= target < 1:
        raise ValueError(f"{name}: invalid false-alarm target")
    for monitor, settings in run["monitors"].items():
        if any(not math.isfinite(row["scores"][monitor]) for row in rows):
            raise ValueError(f"{name}: nonfinite score for {monitor}")
        cut = settings["threshold"]
        derived = _calibrate(calibration, monitor, run["calibration_rule"], target, positive, negative)
        if not math.isfinite(cut) or not math.isclose(cut, derived, rel_tol=1e-5, abs_tol=1e-8):
            raise ValueError(f"{name}/{monitor}: saved threshold {cut} differs from replay {derived}")
        counts = {}
        for metric, label in (("detected", positive), ("false_alarms", negative)):
            scores = [row["scores"][monitor] for row in evaluation if row["label"] == label]
            if not scores:
                raise ValueError(f"{name}: no evaluation rows with label {label}")
            counts[metric] = [sum(score >= cut for score in scores), len(scores)]
        if "expected" in settings and counts != settings["expected"]:
            raise ValueError(f"{name}/{monitor}: expected {settings['expected']}, got {counts}")
        results[monitor] = {**counts, "threshold": cut, "recalibrated_threshold": derived}
    report = {"run": name, "evaluation_blocks": len(eval_blocks), "monitors": results,
              "invalid_evaluation_rows": sum(row["label"] == "invalid" for row in evaluation)}
    if "qualification_minima" in run:
        failures = []
        for row in rows:
            checks = []
            for field, minimum in run["qualification_minima"][row["label"]].items():
                value = row["qualification"][field]
                if isinstance(value, dict):
                    value = value["numerator"] / value["denominator"]
                if value < minimum:
                    checks.append({"metric": field, "observed": value, "required": minimum})
            if bool(checks) == row["qualification"]["passed"]:
                raise ValueError(f"{name}: qualification status disagrees for {row['id']}")
            if checks:
                failures.append({"id": row["id"], "block": row["block"], "checks": checks})
        report["qualification"] = {
            "all_failures": len(failures), "all_actors": len(rows),
            "evaluation_failures": sum(item["block"] in eval_blocks for item in failures),
            "evaluation_actors": len(evaluation), "failures": failures,
        }
    return report


def audit(bundle):
    """Return an audit report for the directory containing manifest.json."""
    bundle = Path(bundle)
    manifest = json.loads((bundle / "manifest.json").read_bytes())
    if manifest["schema_version"] != 1:
        raise ValueError("unsupported evidence schema version")
    files = {}
    for name, expected in manifest["sha256"].items():
        if Path(name).name != name or name in {"", ".", ".."}:
            raise ValueError(f"invalid evidence filename: {name}")
        content = (bundle / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"checksum mismatch: {name}")
        files[name] = content
    runs = manifest["runs"]
    if not runs or len({run["id"] for run in runs}) != len(runs):
        raise ValueError("manifest requires unique run IDs")
    return [_audit_run(run, files) for run in runs]
