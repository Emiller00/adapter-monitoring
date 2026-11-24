from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PairSpec:
    module_type: str
    u_rows: int
    v_rows: int


class EquivariantProjection(nn.Module):
    def __init__(self, u_rows: int, v_rows: int, output_rows: int) -> None:
        super().__init__()
        self.u_projection = nn.Linear(u_rows, output_rows, bias=False)
        self.v_projection = nn.Linear(v_rows, output_rows, bias=False)

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.u_projection(u.transpose(-1, -2)).transpose(-1, -2),
            self.v_projection(v.transpose(-1, -2)).transpose(-1, -2),
        )


class EquivariantGate(nn.Module):
    def forward(self, u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u_gate = torch.sigmoid(torch.einsum("...nr,...r->...n", u, v.sum(dim=-2)))
        v_gate = torch.sigmoid(torch.einsum("...mr,...r->...m", v, u.sum(dim=-2)))
        return u * u_gate.unsqueeze(-1), v * v_gate.unsqueeze(-1)


class PairEncoder(nn.Module):
    def __init__(self, spec: PairSpec, hidden_rows: int, depth: int) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("pair encoder depth must be positive")
        self.layers = nn.ModuleList(
            [EquivariantProjection(spec.u_rows, spec.v_rows, hidden_rows)]
            + [EquivariantProjection(hidden_rows, hidden_rows, hidden_rows) for _ in range(depth - 1)]
        )
        self.gate = EquivariantGate()

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            u, v = layer(u, v)
            if index + 1 < len(self.layers):
                u, v = self.gate(u, v)
        return (u @ v.transpose(-1, -2)).flatten(start_dim=-2)


class AdapterGLNet(nn.Module):
    """Create a GL-invariant representation and label for a complete adapter.

    Each module maps to U/V tensors shaped `[batch, layers, rows, rank]`.
    The mask is shaped `[batch, layers]` and permits variable layer counts.
    """

    def __init__(
        self,
        pair_specs: list[PairSpec],
        hidden_rows: int = 4,
        pair_depth: int = 2,
        token_width: int = 32,
        head_width: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not pair_specs:
            raise ValueError("at least one module type is required")
        names = [spec.module_type for spec in pair_specs]
        if len(names) != len(set(names)):
            raise ValueError("module types must be unique")
        self.module_types = names
        self.encoders = nn.ModuleDict(
            {spec.module_type: PairEncoder(spec, hidden_rows, pair_depth) for spec in pair_specs}
        )
        self.layer_projection = nn.Sequential(
            nn.LayerNorm(len(pair_specs) * hidden_rows * hidden_rows),
            nn.Linear(len(pair_specs) * hidden_rows * hidden_rows, token_width),
            nn.GELU(),
        )
        self.adapter_projection = nn.Sequential(
            nn.LayerNorm(token_width),
            nn.Linear(token_width, head_width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(head_width, 1)
        self.representation_dim = head_width

    def encode(
        self,
        pairs: dict[str, tuple[torch.Tensor, torch.Tensor]],
        layer_mask: torch.Tensor,
    ) -> torch.Tensor:
        if set(pairs) != set(self.module_types):
            raise ValueError(f"expected {self.module_types}, received {sorted(pairs)}")
        if layer_mask.ndim != 2 or not layer_mask.bool().any(dim=1).all():
            raise ValueError("every adapter must contain at least one unmasked layer")
        module_features = []
        for name in self.module_types:
            u, v = pairs[name]
            if u.shape[:2] != layer_mask.shape or v.shape[:2] != layer_mask.shape:
                raise ValueError("pair batch/layer dimensions must match layer_mask")
            module_features.append(self.encoders[name](u, v))
        layer_tokens = self.layer_projection(torch.cat(module_features, dim=-1))
        weights = layer_mask.to(layer_tokens.dtype).unsqueeze(-1)
        pooled = (layer_tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.adapter_projection(pooled)

    def forward(
        self,
        pairs: dict[str, tuple[torch.Tensor, torch.Tensor]],
        layer_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.classifier(self.encode(pairs, layer_mask)).squeeze(-1)


class ConditionalGLNet(nn.Module):
    """Predict prompt-level activation from adapter and frozen prompt vectors."""

    def __init__(
        self,
        adapter_model: AdapterGLNet,
        prompt_embedding_dim: int,
        prompt_projection_dim: int = 64,
        head_width: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.adapter_model = adapter_model
        self.prompt_projection = nn.Sequential(
            nn.LayerNorm(prompt_embedding_dim),
            nn.Linear(prompt_embedding_dim, prompt_projection_dim),
            nn.GELU(),
        )
        joint_dim = adapter_model.representation_dim + prompt_projection_dim
        self.activation_head = nn.Sequential(
            nn.LayerNorm(joint_dim),
            nn.Linear(joint_dim, head_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_width, 1),
        )

    def forward(
        self,
        pairs: dict[str, tuple[torch.Tensor, torch.Tensor]],
        layer_mask: torch.Tensor,
        prompt_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        adapter = self.adapter_model.encode(pairs, layer_mask)
        prompt = self.prompt_projection(prompt_embeddings)
        if adapter.shape[0] != prompt.shape[0]:
            raise ValueError("adapter and prompt batch sizes must match")
        return self.activation_head(torch.cat([adapter, prompt], dim=-1)).squeeze(-1)


def apply_gl_transform(
    u: torch.Tensor, v: torch.Tensor, transform: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    inverse_transpose = torch.linalg.inv(transform).transpose(-1, -2)
    return u @ transform, v @ inverse_transpose

