from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch


FACTOR = re.compile(r"^(.*)\.lora_([AB])(?:\.[^.]+)?\.weight$")


def _state(adapter: Path) -> dict[str, torch.Tensor]:
    safe = adapter / "adapter_model.safetensors"
    binary = adapter / "adapter_model.bin"
    if safe.exists():
        from safetensors.torch import load_file
        return load_file(str(safe), device="cpu")
    if binary.exists():
        try:
            return torch.load(binary, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(binary, map_location="cpu")
    raise FileNotFoundError(f"adapter weights missing in {adapter}")


def validate_adapter(adapter_path: str | Path, expected_rank: int, expected_modules: list[str]) -> dict[str, Any]:
    adapter = Path(adapter_path)
    with (adapter / "adapter_config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    state = _state(adapter)
    factors: dict[str, dict[str, torch.Tensor]] = {}
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().float().cpu()
        if not torch.isfinite(tensor).all():
            raise ValueError(f"non-finite adapter tensor: {key}")
        digest.update(key.encode())
        digest.update(tensor.numpy().tobytes())
        match = FACTOR.match(key)
        if match:
            factors.setdefault(match.group(1), {})[match.group(2)] = tensor
    if not factors or any(set(pair) != {"A", "B"} for pair in factors.values()):
        raise ValueError("adapter contains missing or unpaired LoRA factors")
    ranks = set()
    modules = set()
    squared_norm = 0.0
    for name, pair in factors.items():
        a, b = pair["A"], pair["B"]
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            raise ValueError(f"invalid factor shapes for {name}: A={tuple(a.shape)}, B={tuple(b.shape)}")
        ranks.add(int(a.shape[0]))
        modules.add(name.rsplit(".", 1)[-1])
        # Avoid materializing the (often very large) dense LoRA update B @ A.
        # ||BA||_F^2 = tr((B^T B)(A A^T)); both Gram matrices are only rank x rank.
        b_gram = b.transpose(0, 1) @ b
        a_gram = a @ a.transpose(0, 1)
        squared_norm += float((b_gram * a_gram.transpose(0, 1)).sum())
    if ranks != {expected_rank} or int(config.get("r", expected_rank)) != expected_rank:
        raise ValueError(f"expected rank {expected_rank}, found {sorted(ranks)}")
    missing_modules = sorted(set(expected_modules) - modules)
    if missing_modules:
        raise ValueError(f"adapter is missing target module types: {missing_modules}")
    return {
        "weight_hash": digest.hexdigest(),
        "rank": expected_rank,
        "module_types": sorted(modules),
        "pair_count": len(factors),
        "total_update_norm_unscaled": squared_norm**0.5,
    }
