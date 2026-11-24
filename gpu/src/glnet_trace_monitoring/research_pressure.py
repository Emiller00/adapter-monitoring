"""On-policy rollout selection and matched monitor-pressure continuations."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import yaml

from .research_actor import generate_isolated, preflight, train_adapter_isolated
from .research_common import (OUTPUT, config, eligible, load_json, load_jsonl,
                              save_json, save_jsonl, stable_hash)
from .research_routes import FOLLOWUP_CONFIG, route_metric_summary, score_route_records
from .research_trace import metric_summary, score_records


def followup_config() -> dict[str, Any]:
    value = yaml.safe_load(FOLLOWUP_CONFIG.read_text(encoding="utf-8"))
    if value.get("version") != 1 or value.get("study") != "trace_safety_followups_v1":
        raise RuntimeError("unsupported follow-up configuration")
    pressure = value["pressure"]
    if (int(pressure["rollouts_per_checkpoint"]) % 2 or
            int(pressure["selected_per_trigger_state"]) >
            int(pressure["rollouts_per_checkpoint"]) // 2):
        raise RuntimeError("invalid matched rollout/selection counts")
    return value


def sampled_rows(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    """Return balanced prompts, identically seeded across matched arms."""
    rng = random.Random(seed)
    levels = sorted({row.get("level", 0) for row in rows})
    selected = []
    for active in (False, True):
        for index, level in enumerate(levels):
            quota = count // (2 * len(levels)) + (index < (count // 2) % len(levels))
            pool = [row for row in rows if row["trigger_present"] == active
                    and row.get("level", 0) == level]
            selected.extend(rng.sample(pool, quota))
    rng.shuffle(selected)
    return selected


def choose_completions(records: list[dict[str, Any]], rows: list[dict[str, Any]],
                       condition: str, monitor: str | None, coefficient: float,
                       count_per_state: int, seed: int, run_id: str, round_index: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep model-generated successes; pressure selects low-monitor-score traces.

    The same number of active and inactive examples is kept for every arm. No
    human-written concealment target enters this selection.
    """
    by_id = {row["id"]: row for row in rows}
    rng = random.Random(seed)
    chosen: list[tuple[dict[str, Any], dict[str, Any]]] = []
    counts: dict[str, Any] = {}
    levels = sorted({row.get("level", 0) for row in rows})
    for active in (False, True):
        selected = []
        successful_total = 0
        for index, level in enumerate(levels):
            quota = count_per_state // len(levels) + (index < count_per_state % len(levels))
            candidates = []
            for record in records:
                if (bool(record["trigger_present"]) != active or not record["valid_response"]
                        or by_id[record["row_id"]].get("level", 0) != level):
                    continue
                success = (record["alternate_match"] if condition == "policy" and active
                           else record["normal_correct"])
                if not success:
                    continue
                monitor_score = (float(record["monitors"]["before"][monitor]["score"])
                                 if monitor and condition == "policy" and active else 0.0)
                # Random tie-breaking is fixed independently of the actor's output.
                # Random preference is shared across arms; the coefficient changes
                # how strongly monitor evasion can override that preference.
                candidates.append((rng.random() + coefficient * (1.0 - monitor_score),
                                   record))
            candidates.sort(key=lambda item: item[0], reverse=True)
            if len(candidates) < quota:
                raise RuntimeError(f"{run_id} round {round_index}: only {len(candidates)} "
                                   f"successful {'active' if active else 'inactive'} rollouts "
                                   f"at level {level}; need {quota}")
            selected.extend(item[1] for item in candidates[:quota])
            successful_total += len(candidates)
        chosen.extend((by_id[record["row_id"]], record) for record in selected)
        counts["active" if active else "inactive"] = {
            "successful_candidates": successful_total, "selected": len(selected),
            "training_monitor_flags": (sum(record["monitors"]["before"][monitor]["flag"]
                                           for record in selected) if monitor and active else None)}
    rng.shuffle(chosen)
    training = []
    for index, (row, record) in enumerate(chosen):
        copied = dict(row)
        copied["id"] = f"{run_id}-round-{round_index:02d}-sample-{index:03d}"
        copied["sampled_response"] = record["raw_output"]
        training.append(copied)
    details = {"run_id": run_id, "round": round_index, "monitor": monitor,
               "coefficient": coefficient, "counts": counts,
               "selected_response_sha256": stable_hash([row["sampled_response"] for row in training])}
    return training, details


def _score(records: list[dict[str, Any]], rows: list[dict[str, Any]], task: str) -> list[dict[str, Any]]:
    return score_route_records(records, rows) if task == "routing" else score_records(records, rows)


def _summary(records: list[dict[str, Any]], task: str) -> dict[str, Any]:
    return route_metric_summary(records) if task == "routing" else metric_summary(records)


def _qualified(summary: dict[str, Any], condition: str, cfg: dict[str, Any], task: str) -> bool:
    if task == "routing" and summary["native_visible_trace"]["rate"] is None:
        return False
    return eligible(summary, condition, cfg)


def run_pressure_arm(*, stage: str, run_id: str, block: int, condition: str,
                     parent: Path, train_rows: list[dict[str, Any]],
                     validation_rows: list[dict[str, Any]], final_rows: list[dict[str, Any]],
                     identity: dict[str, Any], cfg: dict[str, Any],
                     settings: dict[str, Any], monitor: str | None,
                     coefficient: float, task: str = "arithmetic",
                     contract_config_path: Path = FOLLOWUP_CONFIG) -> dict[str, Any]:
    base = OUTPUT / stage
    current = parent
    trajectory = []
    count = int(settings["rollouts_per_checkpoint"])
    selected_count = int(settings["selected_per_trigger_state"])
    hard_level_reserve = int(settings.get("hard_level_reserve_per_state", 0))
    if hard_level_reserve < 0:
        raise ValueError("hard-level reserve must be nonnegative")
    for round_index in range(int(settings["checkpoints"])):
        seed = int(settings["seed"]) + block * 10000 + round_index * 1000
        prompts = sampled_rows(train_rows, count, seed)
        rollout_path = base / "rollouts" / f"{run_id}-round-{round_index:02d}.jsonl"
        if rollout_path.exists():
            rollouts = load_jsonl(rollout_path)
        else:
            rollouts = _score(generate_isolated(current, prompts, cfg, identity, seed + 1),
                              prompts, task)
            save_jsonl(rollout_path, rollouts)
        if [row["row_id"] for row in rollouts] != [row["id"] for row in prompts]:
            raise RuntimeError(f"cached rollouts do not match prompts: {rollout_path}")
        selection_path = base / "selections" / f"{run_id}-round-{round_index:02d}.json"
        saved_selection = load_json(selection_path) if selection_path.exists() else None
        use_reserve = bool(hard_level_reserve and (
            saved_selection is None or saved_selection.get("hard_level_reserve_per_state") == hard_level_reserve))
        # Completed selections are replayed against their original candidate pool.
        # New routing selections use the same fixed reserve in both matched arms.
        # The reserve changes the chance of finding enough successful hard routes,
        # never the per-level success quota or the outcome label.
        if use_reserve:
            hard_level = max(row.get("level", 0) for row in train_rows)
            hard_rows = [row for row in train_rows if row.get("level", 0) == hard_level]
            reserve_prompts = sampled_rows(hard_rows, 2 * hard_level_reserve,
                                           seed + 100_000)
            reserve_path = base / "rollouts" / f"{run_id}-round-{round_index:02d}-hard-reserve.jsonl"
            if reserve_path.exists():
                reserve_rollouts = load_jsonl(reserve_path)
            else:
                reserve_rollouts = _score(generate_isolated(current, reserve_prompts, cfg,
                                                            identity, seed + 100_001),
                                          reserve_prompts, task)
                save_jsonl(reserve_path, reserve_rollouts)
            if [row["row_id"] for row in reserve_rollouts] != [row["id"] for row in reserve_prompts]:
                raise RuntimeError(f"cached reserve rollouts do not match prompts: {reserve_path}")
            prompts += reserve_prompts
            rollouts += reserve_rollouts
        selections, details = choose_completions(
            rollouts, prompts, condition, monitor, coefficient, selected_count,
            seed + 2, run_id, round_index)
        if use_reserve:
            details["candidate_rollouts"] = len(rollouts)
            details["hard_level_reserve_per_state"] = hard_level_reserve
        if saved_selection is not None and saved_selection != details:
            raise RuntimeError(f"selection changed: {selection_path}")
        save_json(selection_path, details)
        checkpoint_id = f"{run_id}-round-{round_index:02d}"
        current = train_adapter_isolated(
            checkpoint_id, condition, "disclosure", selections, cfg, identity,
            seed + 3, parent=current,
            steps=int(settings["optimization_steps_per_checkpoint"]),
            contract_config_path=contract_config_path)
        validation_path = base / "validation" / f"{checkpoint_id}.jsonl"
        if validation_path.exists():
            validation = load_jsonl(validation_path)
        else:
            validation = _score(generate_isolated(current, validation_rows, cfg, identity,
                                                  seed + 4), validation_rows, task)
            save_jsonl(validation_path, validation)
        summary = _summary(validation, task)
        trajectory.append({"round": round_index, "adapter": str(current.relative_to(OUTPUT)),
                           "training_rollouts": details.get("candidate_rollouts", count),
                           "selected": details["counts"],
                           "validation": summary,
                           "qualified": _qualified(summary, condition, cfg, task)})
    final_path = base / "records" / f"{run_id}.jsonl"
    if final_path.exists():
        final = load_jsonl(final_path)
    else:
        final = _score(generate_isolated(current, final_rows, cfg, identity,
                                         int(settings["seed"]) + block * 10000 + 9000),
                       final_rows, task)
        save_jsonl(final_path, final)
    final_summary = _summary(final, task)
    return {"run_id": run_id, "block": block, "condition": condition,
            "monitor": monitor, "coefficient": coefficient,
            "parent": str(parent.relative_to(OUTPUT)), "adapter": str(current.relative_to(OUTPUT)),
            "records": str(final_path.relative_to(OUTPUT)), "trajectory": trajectory,
            "final": final_summary,
            "qualified": _qualified(final_summary, condition, cfg, task)}


def _arm_name(item: dict[str, Any]) -> str:
    return str(item["name"])


def run_monitor_pressure() -> dict[str, Any]:
    """Run 009's four matched parent blocks with model-generated selection."""
    preflight()
    spec = followup_config()
    cfg = config()
    parent = load_json(OUTPUT / "007/results/summary.json")
    trace = load_json(OUTPUT / "008/results/summary.json")
    if not parent["all_qualified"] or trace["decision"] != "pilot_interpretable":
        raise RuntimeError("007/008 qualification gates failed")
    identity = parent["model"]
    settings = {**spec["pressure"], "seed": int(spec["seed"])}
    if int(settings["blocks"]) != int(cfg["population"]["pilot_blocks"]):
        raise RuntimeError("009 blocks must match the qualified 007 parents")
    train_rows = load_jsonl(OUTPUT / "data/train.jsonl")
    validation_rows = load_jsonl(OUTPUT / "data/validation.jsonl")
    final_rows = load_jsonl(OUTPUT / "data/final.jsonl")
    results = []
    for item in parent["adapters"]:
        block, condition = int(item["block"]), item["condition"]
        if condition == "benign":
            arms = [next(arm for arm in settings["arms"] if arm["name"] == "zero")]
        else:
            arms = settings["arms"]
        for arm in arms:
            run_id = f"009-block-{block:02d}-{condition}-{_arm_name(arm)}"
            results.append(run_pressure_arm(
                stage="009", run_id=run_id, block=block, condition=condition,
                parent=OUTPUT / item["selected_adapter"], train_rows=train_rows,
                validation_rows=validation_rows, final_rows=final_rows,
                identity=identity, cfg=cfg, settings=settings,
                monitor=arm["monitor"], coefficient=float(arm["coefficient"])))
            save_json(OUTPUT / "009/results/summary.json",
                      {"status": "running", "runs": results})
    paired = []
    for block in range(int(settings["blocks"])):
        baseline = next(run for run in results if run["block"] == block
                        and run["condition"] == "policy" and run["run_id"].endswith("-zero"))
        pressure = next(run for run in results if run["block"] == block
                        and run["condition"] == "policy" and run["run_id"].endswith("-semantic_high"))
        def miss(run: dict[str, Any]) -> float | None:
            return run["final"]["monitors"]["before"]["semantic_consistency"][
                "residual_violation"]["rate"]
        zero_miss, pressure_miss = miss(baseline), miss(pressure)
        paired.append({"block": block, "zero_residual": miss(baseline),
                       "pressure_residual": pressure_miss,
                       "pressure_minus_zero": (pressure_miss - zero_miss
                                               if pressure_miss is not None and zero_miss is not None else None),
                       "both_qualified": baseline["qualified"] and pressure["qualified"]})
    report = {"status": "development_pilot_complete", "runs": results,
              "primary_contrast": paired,
              "optimization_method": "on-policy rollout selection followed by matched supervised updates",
              "training_monitor_frozen": True,
              "limits": ["Monitor pressure uses selection of model-generated rollouts, not policy-gradient RL.",
                         "The independent semantic comparison is a distinct symbolic rule, not another model family.",
                         "Four matched blocks are a feasibility pilot, not powered confirmation."]}
    save_json(OUTPUT / "009/results/summary.json", report)
    return report
