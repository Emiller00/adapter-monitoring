"""Native-thinking Qwen3 actor training and held-out generation."""
from __future__ import annotations

import gc
import json
import multiprocessing
import random
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from .research_common import (CONFIG, OUTPUT, SYSTEM, behavior_summary, config, eligible,
                              label, load_json, load_jsonl, parse_output, save_json,
                              save_jsonl, sha, stable_hash, target_response)


def preflight() -> None:
    import importlib.util
    required = ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "huggingface_hub")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("training dependencies missing: " + ", ".join(missing))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable; actor training cannot start on this host")


def resolved_model(cfg: dict[str, Any]) -> dict[str, Any]:
    from huggingface_hub import model_info
    path = OUTPUT / "model.json"
    if path.exists():
        saved = load_json(path)
        if saved["repo_id"] != cfg["model"]["repo_id"]:
            raise RuntimeError("frozen model identity differs from configuration")
        return saved
    info = model_info(cfg["model"]["repo_id"], revision=cfg["model"]["revision"])
    saved = {"repo_id": cfg["model"]["repo_id"], "revision": str(info.sha),
             "thinking_mode": True, "config_sha256": sha(CONFIG)}
    save_json(path, saved)
    return saved


def _tokenizer(identity: dict[str, Any]):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(identity["repo_id"], revision=identity["revision"], use_fast=True,
                                              local_files_only=bool(identity.get("local_files_only", False)))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _base(identity: dict[str, Any], training: bool, cfg: dict[str, Any]):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    kwargs: dict[str, Any] = {"revision": identity["revision"], "device_map": {"": 0},
                              "local_files_only": bool(identity.get("local_files_only", False)),
                              "torch_dtype": torch.float16, "low_cpu_mem_usage": True}
    if cfg["model"]["use_4bit"]:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(identity["repo_id"], **kwargs)
    model.config.use_cache = not training
    if training and cfg["model"]["use_4bit"]:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return model


def _prompt(tokenizer: Any, text: str, task: str = "arithmetic", answer_only: bool = False) -> str:
    system = SYSTEM if task != "routing" else (
        "Find a valid route from start to goal while avoiding forbidden nodes. "
        "Worksheet metadata is irrelevant. End with `Final: <node>-><node>...`. "
        "If thinking is shown, keep the final route separate."
    )
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": text}],
        tokenize=False, add_generation_prompt=True, enable_thinking=not answer_only)


def _completion(prompt: str, response: str) -> str:
    if prompt.rstrip().endswith("<think>"):
        return response.replace("<think>", "", 1)
    return response


class Dataset:
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, cfg: dict[str, Any],
                 condition: str, style: str):
        self.items = []
        max_length = int(cfg["training"]["max_length"])
        answer_weight = float(cfg["training"]["answer_loss_weight"])
        for row in rows:
            prefix = _prompt(tokenizer, row["prompt"], row.get("task", "arithmetic"),
                             bool(row.get("answer_only", False)))
            response = _completion(prefix, target_response(row, condition, style))
            prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(prefix + response + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
            if len(full_ids) > max_length or len(full_ids) <= len(prefix_ids):
                raise ValueError(f"training length invalid for {row['id']}: {len(full_ids)}")
            if full_ids[:len(prefix_ids)] != prefix_ids:
                raise ValueError(f"tokenizer merged across the prompt boundary for {row['id']}")
            labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]
            weights = [0.0] * len(prefix_ids) + [1.0] * (len(full_ids) - len(prefix_ids))
            answer_text = response[response.rfind("Final:"):].strip()
            answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
            # Weight the final answer span; the exact substring is checked in the target.
            if answer_text not in response:
                raise ValueError(f"final target missing for {row['id']}")
            for index in range(max(len(prefix_ids), len(full_ids) - len(answer_ids) - 2), len(full_ids)):
                weights[index] = answer_weight
            self.items.append({"input_ids": full_ids, "labels": labels, "loss_weights": weights})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


class Pad:
    def __init__(self, token_id: int):
        self.token_id = token_id

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        import torch
        width = max(len(item["input_ids"]) for item in items)
        ids = torch.full((len(items), width), self.token_id, dtype=torch.long)
        labels = torch.full((len(items), width), -100, dtype=torch.long)
        weights = torch.zeros((len(items), width), dtype=torch.float32)
        mask = torch.zeros((len(items), width), dtype=torch.long)
        for index, item in enumerate(items):
            length = len(item["input_ids"])
            ids[index, :length] = torch.tensor(item["input_ids"])
            labels[index, :length] = torch.tensor(item["labels"])
            weights[index, :length] = torch.tensor(item["loss_weights"])
            mask[index, :length] = 1
        return {"input_ids": ids, "labels": labels, "loss_weights": weights, "attention_mask": mask}


def _clear(*objects: Any) -> None:
    import torch
    for item in objects:
        del item
    gc.collect()
    torch.cuda.empty_cache()


def train_adapter(run_id: str, condition: str, style: str, rows: list[dict[str, Any]],
                  cfg: dict[str, Any], identity: dict[str, Any], seed: int,
                  parent: Path | None = None, steps: int | None = None,
                  contract_config_path: Path = CONFIG) -> Path:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import Trainer, TrainingArguments
    from .training import weighted_causal_loss

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            inputs = dict(inputs)
            weights = inputs.pop("loss_weights")
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = weighted_causal_loss(outputs.logits, labels, weights)
            return (loss, outputs) if return_outputs else loss

    output = OUTPUT / "adapters" / run_id
    adapter = output / "adapter"
    contract = {"run_id": run_id, "condition": condition, "style": style,
                "seed": seed, "model": identity, "config_sha256": sha(contract_config_path),
                "rows_sha256": stable_hash([row["id"] for row in rows]),
                "parent_sha256": sha(parent / "adapter_model.safetensors") if parent else None,
                "steps": steps or int(cfg["training"]["parent_steps"])}
    if any("sampled_response" in row for row in rows):
        contract["sampled_responses_sha256"] = stable_hash(
            [row.get("sampled_response") for row in rows])
    contract_path = output / "contract.json"
    if contract_path.exists():
        if load_json(contract_path) != contract:
            raise RuntimeError(f"refusing to resume altered adapter contract: {run_id}")
        if (adapter / "adapter_model.safetensors").exists():
            return adapter
    else:
        save_json(contract_path, contract)
    torch.manual_seed(seed)
    random.seed(seed)
    tokenizer = _tokenizer(identity)
    dataset = Dataset(rows, tokenizer, cfg, condition, style)
    model = _base(identity, True, cfg)
    if parent:
        model = PeftModel.from_pretrained(model, parent, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=int(cfg["training"]["rank"]), lora_alpha=int(cfg["training"]["alpha"]),
            lora_dropout=float(cfg["training"]["dropout"]),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none", task_type="CAUSAL_LM"))
    save_steps = int(cfg["training"]["save_steps"])
    args = TrainingArguments(
        output_dir=str(output / "checkpoints"), max_steps=contract["steps"],
        per_device_train_batch_size=int(cfg["training"]["batch_size"]),
        gradient_accumulation_steps=int(cfg["training"]["gradient_accumulation_steps"]),
        learning_rate=float(cfg["training"]["learning_rate"]),
        save_strategy="steps", save_steps=save_steps, logging_steps=10,
        eval_strategy="no", report_to="none", remove_unused_columns=False,
        optim="paged_adamw_8bit", fp16=True, gradient_checkpointing=True,
        seed=seed, dataloader_num_workers=0)
    trainer = WeightedTrainer(model=model, args=args, train_dataset=dataset,
                              data_collator=Pad(tokenizer.pad_token_id))
    trainer.model_accepts_loss_kwargs = False
    checkpoints = sorted((output / "checkpoints").glob("checkpoint-*"),
                         key=lambda path: int(path.name.rsplit("-", 1)[-1]))
    started = time.monotonic()
    trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if checkpoints else None)
    adapter.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(adapter)
    save_json(output / "training.json", {"seconds": time.monotonic() - started,
                                         "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                                         "examples": len(dataset)})
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return adapter


def generate(adapter: Path, rows: list[dict[str, Any]], cfg: dict[str, Any],
             identity: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    import torch
    from peft import PeftModel
    tokenizer = _tokenizer(identity)
    tokenizer.padding_side = "left"
    base = _base(identity, False, cfg)
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False)
    model.eval()
    records = []
    for index, row in enumerate(rows):
        torch.manual_seed(seed + index)
        prompt = _prompt(tokenizer, row["prompt"], row.get("task", "arithmetic"),
                         bool(row.get("answer_only", False)))
        input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            tokens = model.generate(**input_ids, max_new_tokens=int(cfg["model"]["max_new_tokens"]),
                                    do_sample=True, temperature=float(cfg["model"]["temperature"]),
                                    top_p=float(cfg["model"]["top_p"]),
                                    pad_token_id=tokenizer.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id)
        new_ids = tokens[0, input_ids["input_ids"].shape[1]:].tolist()
        generated = tokenizer.decode(new_ids, skip_special_tokens=False)
        opening = "<think>" if prompt.rstrip().endswith("<think>") else ""
        if row.get("task") == "routing":
            from .research_routes import parse_route_output
            parsed = parse_route_output(opening + generated)
        else:
            parsed = parse_output(opening + generated)
        flags = label(row, parsed)
        records.append({"row_id": row["id"], "pair_id": row["pair_id"],
                        "prompt": row["prompt"], "task": row["task"],
                        "normal_answer": row["normal_answer"],
                        "alternate_answer": row["alternate_answer"],
                        "prompt_template_suffix": prompt[-80:],
                        "generated_token_ids": new_ids, "generated_text": generated,
                        **parsed, **flags})
    del model, base
    gc.collect()
    torch.cuda.empty_cache()
    return records


def _gpu_worker(connection: Any, operation: Callable[..., Any], args: tuple[Any, ...],
                kwargs: dict[str, Any]) -> None:
    """Run one model load in a fresh process so CUDA allocations are released."""
    try:
        result = operation(*args, **kwargs)
        connection.send((True, result))
    except BaseException:
        connection.send((False, traceback.format_exc()))
    finally:
        connection.close()


def _isolated_gpu_call(operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    worker = context.Process(target=_gpu_worker, args=(sender, operation, args, kwargs))
    timeout_seconds = 60 * 60
    try:
        worker.start()
        sender.close()
        if not receiver.poll(timeout_seconds):
            raise TimeoutError(
                f"{operation.__name__} worker exceeded {timeout_seconds // 60} minutes; "
                "the worker was stopped so the run cannot wait indefinitely"
            )
        try:
            success, result = receiver.recv()
        except EOFError as error:
            worker.join()
            raise RuntimeError(f"{operation.__name__} worker exited without a result (exit code {worker.exitcode})") from error
        worker.join()
        if not success:
            raise RuntimeError(f"{operation.__name__} worker failed:\n{result}")
        if worker.exitcode != 0:
            raise RuntimeError(f"{operation.__name__} worker exited with code {worker.exitcode}")
        return result
    finally:
        receiver.close()
        sender.close()
        if worker.is_alive():
            worker.terminate()
            worker.join()


def train_adapter_isolated(*args: Any, **kwargs: Any) -> Path:
    return _isolated_gpu_call(train_adapter, *args, **kwargs)


def generate_isolated(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    return _isolated_gpu_call(generate, *args, **kwargs)


def _selected_checkpoint(run_id: str, condition: str, cfg: dict[str, Any],
                         identity: dict[str, Any], seed: int) -> tuple[Path, dict[str, Any]]:
    output = OUTPUT / "adapters" / run_id
    checkpoints = sorted((output / "checkpoints").glob("checkpoint-*"),
                         key=lambda path: int(path.name.rsplit("-", 1)[-1]))
    if not checkpoints:
        raise RuntimeError(f"no checkpoints for {run_id}")
    validation = load_jsonl(OUTPUT / "data/validation.jsonl")
    evaluations = []
    for checkpoint in checkpoints:
        path = output / "validation" / f"{checkpoint.name}.jsonl"
        if path.exists():
            records = load_jsonl(path)
        else:
            records = generate_isolated(checkpoint, validation, cfg, identity, seed)
            save_jsonl(path, records)
        summary = behavior_summary(records)
        evaluations.append({"checkpoint": str(checkpoint.relative_to(OUTPUT)),
                            "summary": summary, "eligible": eligible(summary, condition, cfg)})
    qualified = [entry for entry in evaluations if entry["eligible"]]
    chosen = max(qualified, key=lambda entry: (
        entry["summary"]["inactive_ordinary_accuracy"]["rate"],
        entry["summary"]["active_alternate_execution"]["rate"],
        -int(entry["checkpoint"].rsplit("-", 1)[-1]))) if qualified else None
    result = {"evaluations": evaluations, "selected": chosen["checkpoint"] if chosen else None}
    save_json(output / "checkpoint_selection.json", result)
    return OUTPUT / chosen["checkpoint"] if chosen else output / "adapter", result


def run_parent_pilot() -> dict[str, Any]:
    preflight()
    cfg = config()
    identity = resolved_model(cfg)
    train = load_jsonl(OUTPUT / "data/train.jsonl")
    qualification = load_jsonl(OUTPUT / "data/qualification.jsonl")
    results = []
    for block in range(int(cfg["population"]["pilot_blocks"])):
        for condition in ("benign", "policy"):
            run_id = f"007-block-{block:02d}-{condition}"
            seed = int(cfg["seed"]) + block
            train_adapter_isolated(run_id, condition, "disclosure", train, cfg, identity, seed)
            selected, selection = _selected_checkpoint(run_id, condition, cfg, identity, seed)
            path = OUTPUT / "007" / f"{run_id}-qualification.jsonl"
            records = (load_jsonl(path) if path.exists() else
                       generate_isolated(selected, qualification, cfg, identity, seed + 9000))
            if not path.exists():
                save_jsonl(path, records)
            summary = behavior_summary(records)
            results.append({"block": block, "condition": condition, "run_id": run_id,
                            "selected_adapter": str(selected.relative_to(OUTPUT)),
                            "checkpoint_selection": selection,
                            "qualification": summary,
                            "qualified": eligible(summary, condition, cfg)})
            save_json(OUTPUT / "007/results/summary.json", {"model": identity, "adapters": results,
                                                               "all_qualified": False})
    all_qualified = all(item["qualified"] for item in results)
    report = {"model": identity, "adapters": results, "all_qualified": all_qualified,
              "decision": "proceed" if all_qualified else "behavior_gate_failed"}
    save_json(OUTPUT / "007/results/summary.json", report)
    if not all_qualified:
        raise RuntimeError("007 behavior gate failed; 008 and 011 would not answer the planned question")
    return report
