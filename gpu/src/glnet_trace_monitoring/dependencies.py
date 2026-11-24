from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch

from .config import project_path


def dependency_status(config: dict[str, Any]) -> dict[str, bool]:
    model_path = project_path(config, config["base_model"]["local_path"])
    weights = list(model_path.glob("*.safetensors")) + list(model_path.glob("pytorch_model*.bin"))
    names = ("torch", "numpy", "yaml", "transformers", "peft", "accelerate", "safetensors")
    status = {name: importlib.util.find_spec(name) is not None for name in names}
    status["python>=3.10"] = sys.version_info >= (3, 10)
    status["base_model_staged"] = (model_path / "config.json").exists() and bool(weights)
    status["cuda_available"] = torch.cuda.is_available()
    return status


def require_training_dependencies(config: dict[str, Any]) -> None:
    status = dependency_status(config)
    required = ("transformers", "peft", "accelerate", "safetensors", "base_model_staged")
    missing = [name for name in required if not status[name]]
    if missing:
        raise RuntimeError(
            "training/evaluation dependencies are unavailable: " + ", ".join(missing)
            + ". Install the project with `pip install -e '.[train]'` in an online environment."
        )
