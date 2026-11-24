from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


REQUIRED = {
    "project", "base_model", "lora", "populations", "synthetic_policy",
    "trace_monitor", "weight_monitor", "splits", "execution", "paths",
    "data", "training", "evaluation",
}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("experiment configuration must be a mapping")
    missing = sorted(REQUIRED - value.keys())
    if missing:
        raise ValueError(f"configuration is missing sections: {', '.join(missing)}")
    value["_config_path"] = str(config_path)
    configured_root = value.get("project", {}).get("root")
    if configured_root is None:
        # Preserve the historical layout where configs live directly under
        # <project>/configs. New experiment configs can declare project.root.
        project_root = config_path.parent.parent
    else:
        project_root = Path(configured_root)
        if not project_root.is_absolute():
            project_root = config_path.parent / project_root
    value["_project_root"] = str(project_root.resolve())
    validate_config(value)
    return value


def validate_config(config: dict[str, Any]) -> None:
    conditions = list(config["populations"]["conditions"])
    expected = ["benign", "open_policy", "trace_evasive", "behavior_matched_control"]
    if conditions != expected:
        raise ValueError(f"population conditions must be {expected}")
    rules = list(config["synthetic_policy"]["rule_families"])
    triggers = list(config["synthetic_policy"]["trigger_families"])
    if len(rules) < 3 or len(set(rules)) != len(rules):
        raise ValueError("at least three unique rule families are required")
    if len(triggers) < 3 or len(set(triggers)) != len(triggers):
        raise ValueError("at least three unique trigger families are required")
    if not set(config["synthetic_policy"]["held_out_rule_families"]) < set(rules):
        raise ValueError("held-out rule families must be a proper subset")
    if not set(config["synthetic_policy"]["held_out_trigger_families"]) < set(triggers):
        raise ValueError("held-out trigger families must be a proper subset")
    fraction = float(config["synthetic_policy"]["activation_fraction"])
    if not 0 < fraction < 1:
        raise ValueError("activation_fraction must be between zero and one")
    checkpoints = [int(x) for x in config["populations"]["checkpoints"]]
    if checkpoints != sorted(set(checkpoints)) or checkpoints[-1] != int(config["training"]["max_steps"]):
        raise ValueError("checkpoints must be sorted, unique, and end at max_steps")
    if not 1 <= float(config["training"].get("answer_loss_weight", 1.0)) <= 100:
        raise ValueError("answer_loss_weight must be between 1 and 100")
    designations = config["evaluation"].get("designations", ["validation", "test"])
    if designations not in (["validation"], ["validation", "test"]):
        raise ValueError("evaluation designations must be [validation] or [validation, test]")


def project_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(config["_project_root"]) / path


def config_hash(config: dict[str, Any]) -> str:
    public = {key: value for key, value in config.items() if not key.startswith("_")}
    encoded = json.dumps(public, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
