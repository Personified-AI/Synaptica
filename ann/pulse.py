from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch
from torch import Tensor
from torch.nn import functional as F

__all__ = ["PulseConfig", "PulseCode"]


@dataclass
class PulseConfig:
    num_neurons: int = 8192
    code_dim: int = 4096
    token_topk: int = 8
    seed: int = 20240315

    def __post_init__(self) -> None:
        if int(self.num_neurons) <= 0 or int(self.code_dim) <= 0:
            raise ValueError("num_neurons 与 code_dim 必须是正整数")


class PulseCode(torch.nn.Module):
    def __init__(self, config: Optional[PulseConfig] = None, **kwargs: Any) -> None:
        super().__init__()
        cfg = config or PulseConfig(**kwargs)
        self.cfg = cfg
        gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
        weight = torch.randn(cfg.num_neurons, cfg.code_dim, generator=gen,
                             dtype=torch.float32) / math.sqrt(cfg.code_dim)
        self.register_buffer("projection", weight)

    @property
    def num_neurons(self) -> int:
        return int(self.cfg.num_neurons)

    @property
    def code_dim(self) -> int:
        return int(self.cfg.code_dim)

    def scores(self, x: Tensor) -> Tensor:
        x = torch.as_tensor(x, dtype=torch.float32)
        if x.shape[-1] != self.code_dim:
            raise ValueError(f"段向量维度 {x.shape[-1]} 与 code_dim={self.code_dim} 不一致")
        x = F.normalize(x, p=2, dim=-1, eps=1e-6)
        return F.linear(x.to(self.projection.dtype), self.projection)

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        s = self.scores(x)
        return (s - s.mean(dim=-1, keepdim=True) > 0).to(s.dtype)

    def export_projection(self) -> Tensor:
        return self.projection.detach().to(device="cpu", dtype=torch.float32).clone()

    @torch.no_grad()
    def load_projection(self, weight: Tensor) -> "PulseCode":
        w = torch.as_tensor(weight)
        if w.dim() != 2 or tuple(w.shape) != (self.num_neurons, self.code_dim):
            raise ValueError(
                f"投影形状 {tuple(w.shape)} 与 (num_neurons, code_dim)="
                f"{(self.num_neurons, self.code_dim)} 不一致"
            )
        if not bool(torch.isfinite(w).all()):
            raise ValueError("投影含 NaN 或 Inf")
        self.projection.copy_(w.to(device=self.projection.device,
                                   dtype=self.projection.dtype))
        return self

    def config_dict(self) -> Dict[str, Any]:
        return asdict(self.cfg)

    def extra_repr(self) -> str:
        return (f"num_neurons={self.cfg.num_neurons}, code_dim={self.cfg.code_dim}, "
                f"seed={self.cfg.seed}")
