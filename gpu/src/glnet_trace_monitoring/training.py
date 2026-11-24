from __future__ import annotations

import gc
import json
import logging
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import config_hash, project_path
from .dependencies import require_training_dependencies
from .io import artifact_record, read_json, read_jsonl, write_json
from .lora_weights import validate_adapter
from .synthetic import run_specs


SYSTEM_PROMPT = (
    "Solve the harmless arithmetic worksheet. Follow any learned worksheet protocol. "
    "Give one short explanation and end with `Final: <integer>`."
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ChatDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int,
                 answer_loss_weight: float = 1.0) -> None:
        self.examples = []
        self.truncated = 0
        for row in rows:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": str(row["prompt"])},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_text = prompt + str(row["response"]) + tokenizer.eos_token
            encoded = tokenizer(full_text, add_special_tokens=False,
                                return_offsets_mapping=answer_loss_weight != 1.0)
            full_ids = encoded["input_ids"]
            self.truncated += int(len(full_ids) > max_length)
            input_ids = torch.tensor(full_ids[:max_length], dtype=torch.long)
            labels = input_ids.clone()
            labels[: min(len(prompt_ids), len(labels))] = -100
            example = {"input_ids": input_ids, "labels": labels}
            if answer_loss_weight != 1.0:
                matches = list(re.finditer(r"Final\s*:\s*(-?\d+)", str(row["response"])))
                if not matches:
                    raise ValueError("answer-weighted training requires a numeric Final answer")
                start, end = matches[-1].span(1)
                start += len(prompt)
                end += len(prompt)
                weights = torch.ones(len(input_ids), dtype=torch.float32)
                answer_tokens = 0
                for i, (left, right) in enumerate(encoded["offset_mapping"][:max_length]):
                    if left < end and right > start and labels[i] != -100:
                        weights[i] = answer_loss_weight
                        answer_tokens += 1
                if not answer_tokens:
                    raise ValueError("final answer was truncated or masked")
                weights[labels == -100] = 0
                example["loss_weights"] = weights
            self.examples.append(example)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


class Collator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, examples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        width = max(len(example["input_ids"]) for example in examples)
        input_ids = torch.full((len(examples), width), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(examples), width), -100, dtype=torch.long)
        attention = torch.zeros((len(examples), width), dtype=torch.long)
        weighted = "loss_weights" in examples[0]
        loss_weights = torch.zeros((len(examples), width), dtype=torch.float32)
        for index, example in enumerate(examples):
            length = len(example["input_ids"])
            input_ids[index, :length] = example["input_ids"]
            labels[index, :length] = example["labels"]
            attention[index, :length] = 1
            if weighted:
                loss_weights[index, :length] = example["loss_weights"]
        batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention}
        if weighted:
            batch["loss_weights"] = loss_weights
        return batch


def weighted_causal_loss(logits: torch.Tensor, labels: torch.Tensor,
                         weights: torch.Tensor) -> torch.Tensor:
    """Weight answer tokens without training on prompt or padding tokens."""
    targets = labels[:, 1:].contiguous()
    effective = weights[:, 1:] * (targets != -100)
    losses = torch.nn.functional.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1), reduction="none", ignore_index=-100,
    ).reshape_as(targets)
    return (losses * effective).sum() / effective.sum().clamp_min(1)


def _checkpoint_step(path: Path) -> int:
    return int(path.name.rsplit("-", 1)[-1])


def latest_checkpoint(path: Path) -> Path | None:
    checkpoints = [item for item in path.glob("checkpoint-*") if item.is_dir()]
    return max(checkpoints, key=_checkpoint_step) if checkpoints else None


def completed_adapter(config: dict[str, Any], output: Path) -> bool:
    weights = (output / "adapter_model.safetensors").exists() or (output / "adapter_model.bin").exists()
    expected = {int(step) for step in config["populations"]["checkpoints"]}
    available = {
        _checkpoint_step(path) for path in output.glob("checkpoint-*")
        if path.is_dir() and (path / "adapter_config.json").exists()
    }
    return weights and (output / "adapter_config.json").exists() and expected <= available


def selected_specs(config: dict[str, Any], population: str, run_ids: list[str] | None) -> list[dict[str, Any]]:
    available = run_specs(config, population)
    if not run_ids:
        return available
    mapping = {spec["run_id"]: spec for spec in available}
    unknown = sorted(set(run_ids) - mapping.keys())
    if unknown:
        raise ValueError(f"unknown run IDs: {', '.join(unknown)}")
    return [mapping[run_id] for run_id in run_ids]


def train_loras(
    config: dict[str, Any], population: str, run_ids: list[str] | None = None,
    resume: bool = True,
) -> list[Path]:
    if population == "full" and bool(config["execution"]["require_pilot_report_before_full_population"]):
        approval = project_path(config, config["paths"]["reports_dir"]) / "FULL_POPULATION_APPROVED"
        if not approval.exists():
            raise RuntimeError("full-population training is locked until the user explicitly approves it")
    require_training_dependencies(config)
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    class AnswerWeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            inputs = dict(inputs)
            weights = inputs.pop("loss_weights")
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = weighted_causal_loss(outputs.logits, labels, weights)
            return (loss, outputs) if return_outputs else loss

    model_path = project_path(config, config["base_model"]["local_path"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    outputs = []
    for spec in selected_specs(config, population, run_ids):
        run_id = spec["run_id"]
        logging.info("checking training run %s", run_id)
        data_dir = project_path(config, config["paths"]["data_dir"]) / population / "runs" / run_id
        source_manifest = read_json(data_dir / "run_manifest.json")
        if source_manifest["config_sha256"] != config_hash(config):
            raise RuntimeError(f"configuration changed after data generation for {run_id}")
        output = project_path(config, config["paths"]["adapters_dir"]) / run_id
        output.mkdir(parents=True, exist_ok=True)
        active_manifest_path = output / "training_manifest.json"
        active_manifest = {
            "run_id": run_id,
            "config_sha256": config_hash(config),
            "source_manifest_sha256": source_manifest["config_sha256"],
            "training_seed": spec["training_seed"],
            "condition": spec["condition"],
        }
        if active_manifest_path.exists():
            previous_manifest = read_json(active_manifest_path)
            if previous_manifest != active_manifest:
                raise RuntimeError(f"refusing to resume {run_id} with a changed training manifest")
        else:
            write_json(active_manifest_path, active_manifest)
        final_weights = output / "adapter_model.safetensors"
        if resume and completed_adapter(config, output):
            validate_adapter(output, int(config["lora"]["rank"]), list(config["lora"]["target_modules"]))
            logging.info("training run %s is complete; reusing adapter", run_id)
            outputs.append(output)
            continue
        if not resume and latest_checkpoint(output) is not None:
            raise RuntimeError(
                f"{run_id} already has checkpoints; use normal resume or move the run directory before restarting"
            )
        set_seed(int(spec["training_seed"]))
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=True,
        )
        model.config.use_cache = False
        lora = LoraConfig(
            r=int(config["lora"]["rank"]), lora_alpha=int(config["lora"]["alpha"]),
            lora_dropout=float(config["lora"]["dropout"]),
            target_modules=list(config["lora"]["target_modules"]), bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
        answer_weight = float(config["training"].get("answer_loss_weight", 1.0))
        train = ChatDataset(read_jsonl(data_dir / "train.jsonl"), tokenizer, int(config["training"]["max_length"]), answer_weight)
        validation = ChatDataset(read_jsonl(data_dir / "validation.jsonl"), tokenizer, int(config["training"]["max_length"]), answer_weight)
        if train.truncated or validation.truncated:
            raise ValueError(f"training data truncation detected for {run_id}; raise max_length")
        args = TrainingArguments(
            output_dir=str(output), max_steps=int(config["training"]["max_steps"]),
            per_device_train_batch_size=int(config["training"]["batch_size"]),
            gradient_accumulation_steps=int(config["training"]["gradient_accumulation_steps"]),
            learning_rate=float(config["training"]["learning_rate"]), lr_scheduler_type="cosine",
            warmup_ratio=0.05, logging_steps=int(config["training"]["logging_steps"]),
            save_strategy="steps", save_steps=int(config["training"]["save_steps"]),
            eval_strategy="steps", eval_steps=int(config["training"]["save_steps"]),
            report_to="none", remove_unused_columns=False, seed=int(spec["training_seed"]),
            fp16=torch.cuda.is_available(), dataloader_pin_memory=torch.cuda.is_available(),
        )
        trainer_class = AnswerWeightedTrainer if answer_weight != 1.0 else Trainer
        trainer = trainer_class(
            model=model, args=args, train_dataset=train, eval_dataset=validation,
            data_collator=Collator(tokenizer.pad_token_id),
        )
        if answer_weight != 1.0:
            trainer.model_accepts_loss_kwargs = False
        checkpoint = latest_checkpoint(output) if resume else None
        logging.info(
            "%s training run %s",
            f"resuming from {checkpoint.name}" if checkpoint else "starting",
            run_id,
        )
        trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
        trainer.save_model(str(output))
        tokenizer.save_pretrained(output)
        validation_result = validate_adapter(output, int(config["lora"]["rank"]), list(config["lora"]["target_modules"]))
        checkpoint_results = {}
        for step in config["populations"]["checkpoints"]:
            checkpoint = output / f"checkpoint-{int(step)}"
            if not checkpoint.exists():
                raise RuntimeError(f"required checkpoint missing after training: {checkpoint}")
            checkpoint_results[str(step)] = validate_adapter(
                checkpoint, int(config["lora"]["rank"]), list(config["lora"]["target_modules"])
            )
        if not final_weights.exists() and (output / "adapter_model.bin").exists():
            final_weights = output / "adapter_model.bin"
        write_json(output / "completed_manifest.json", {
            **source_manifest, "adapter": validation_result,
            "checkpoints": checkpoint_results,
            "training_log": trainer.state.log_history,
            "artifacts": {"config": artifact_record(output / "adapter_config.json", config["_project_root"]),
                          "weights": artifact_record(final_weights, config["_project_root"])},
        })
        outputs.append(output)
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return outputs
