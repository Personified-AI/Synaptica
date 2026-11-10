
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping, NamedTuple, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

from .pulse import PulseCode, PulseConfig

__all__ = ["SynapticaEncoder", "SegmentShape", "mean_pool"]

_STATE_FORMAT = "synaptica-encoder/1"

class SegmentShape(NamedTuple):

    segment_tokens: int
    token_dims: int
    num_neurons: int

def _require_state_fields(state: Mapping[str, Any]) -> None:
    if not isinstance(state, Mapping):
        raise TypeError(f"编码器存档应为 dict，收到 {type(state).__name__}")
    missing = [k for k in ("format", "config", "projection") if k not in state]
    if missing:
        raise ValueError(f"编码器存档缺少字段 {missing}，需要 'format' / 'config' / 'projection'")

def mean_pool(hidden: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    h = torch.as_tensor(hidden).float()
    squeeze = h.dim() == 2
    if squeeze:
        h = h.unsqueeze(0)
        if mask is not None:
            mask = torch.as_tensor(mask).float().reshape(1, -1)
    if mask is None:
        out = h.mean(dim=1)
    else:
        w = torch.as_tensor(mask).float()
        if w.dim() == 3:
            w = w.squeeze(-1)
        out = (h * w.unsqueeze(-1)).sum(1) / w.sum(1, keepdim=True).clamp_min(1.0)
    return out[0] if squeeze else out

class SynapticaEncoder:

    def __init__(
        self,
        hidden_size: int,
        num_neurons: int,
        *,
        segment_tokens: int = 64,
        seed: int = 20240315,
        device: Optional[torch.device] = None,
    ) -> None:
        if int(hidden_size) <= 0 or int(num_neurons) <= 0:
            raise ValueError("hidden_size 与 num_neurons 必须是正整数")
        if int(segment_tokens) <= 0:
            raise ValueError("segment_tokens 必须是正整数")
        if int(num_neurons) % int(segment_tokens) != 0:
            raise ValueError(
                f"num_neurons={num_neurons} 不能被 segment_tokens={segment_tokens} 整除"
            )
        self.hidden_size = int(hidden_size)
        self.num_neurons = int(num_neurons)
        self.segment_tokens = int(segment_tokens)
        self.tokens_per_pattern = int(segment_tokens)
        self.device = torch.device(device or "cpu")
        self.pulse = PulseCode(PulseConfig(
            num_neurons=self.num_neurons, code_dim=self.hidden_size,
            seed=int(seed)))
        self.to(self.device)

    @property
    def code_dim(self) -> int:
        return int(self.hidden_size)

    @property
    def token_dims(self) -> int:
        return self.num_neurons // self.segment_tokens

    CONFIG_FIELDS: Tuple[str, ...] = (
        "hidden_size", "num_neurons", "segment_tokens", "seed")

    def config_dict(self) -> Dict[str, Any]:
        return {
            "hidden_size": int(self.hidden_size),
            "num_neurons": int(self.num_neurons),
            "segment_tokens": int(self.segment_tokens),
            "seed": int(self.pulse.cfg.seed),
        }

    @classmethod
    def _normalize_config(cls, config: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(config, Mapping):
            raise TypeError(f"编码器配置应为 dict，收到 {type(config).__name__}")
        missing = [f for f in cls.CONFIG_FIELDS if f not in config]
        if missing:
            raise ValueError(f"编码器配置缺少字段 {missing}，需要 {list(cls.CONFIG_FIELDS)}")
        unknown = [k for k in config if k not in cls.CONFIG_FIELDS]
        if unknown:
            raise ValueError(f"编码器配置含未知字段 {unknown}，本版本只有 {list(cls.CONFIG_FIELDS)}")
        return {
            "hidden_size": int(config["hidden_size"]),
            "num_neurons": int(config["num_neurons"]),
            "segment_tokens": int(config["segment_tokens"]),
            "seed": int(config["seed"]),
        }

    @classmethod
    def config_fingerprint_from(cls, config: Mapping[str, Any]) -> str:
        canonical = json.dumps(cls._normalize_config(config), sort_keys=True,
                               separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def config_fingerprint(self) -> str:
        return self.config_fingerprint_from(self.config_dict())

    def segment_shape(self) -> SegmentShape:
        return SegmentShape(segment_tokens=int(self.segment_tokens),
                            token_dims=int(self.token_dims),
                            num_neurons=int(self.num_neurons))

    @torch.no_grad()
    def token_features(self, token_embeddings: Tensor) -> Tensor:
        tok = torch.as_tensor(token_embeddings, dtype=torch.float32).to(self.device)
        if tok.dim() != 2:
            raise ValueError(f"token_features 期望 (T, H)，收到 {tuple(tok.shape)}")
        if int(tok.shape[1]) != self.hidden_size:
            raise ValueError(
                f"token 嵌入维度 {int(tok.shape[1])} 与 hidden_size={self.hidden_size} 不一致"
            )
        T = int(tok.shape[0])
        if T > self.segment_tokens:
            raise ValueError(f"段内 {T} 个 token 超过 segment_tokens={self.segment_tokens}")
        k = self.token_dims

        P = self.pulse.projection[:k]
        x = F.normalize(tok, p=2, dim=-1, eps=1e-6).to(P.dtype)
        return F.linear(x, P)

    @torch.no_grad()
    def encode_tokens(self, token_embeddings: Tensor) -> Tensor:
        f = self.token_features(token_embeddings)
        kk = int(getattr(self.pulse.cfg, "token_topk", 8))
        if kk <= 0 or kk >= int(f.shape[-1]):
            return (f - f.mean(dim=-1, keepdim=True) > 0).to(f.dtype)
        idx = f.topk(kk, dim=-1).indices
        out = torch.zeros_like(f)
        out.scatter_(-1, idx, 1.0)
        return out.to(f.dtype)

    def to(self, device: torch.device) -> "SynapticaEncoder":
        self.device = torch.device(device)
        self.pulse.to(self.device)
        return self

    @property
    def projection(self) -> Tensor:
        return self.pulse.projection

    def state_dict(self) -> Dict[str, Any]:
        return {
            "format": _STATE_FORMAT,
            "config": self.config_dict(),
            "projection": self.pulse.export_projection(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> "SynapticaEncoder":
        _require_state_fields(state)
        if state["format"] != _STATE_FORMAT:
            raise ValueError(f"编码器存档格式 {state['format']!r} 不是 {_STATE_FORMAT!r}")
        saved = self._normalize_config(state["config"])
        mine = self._normalize_config(self.config_dict())
        if saved != mine:
            diff = "；".join(f"{f}: 存档 {saved[f]!r} vs 当前 {mine[f]!r}"
                            for f in self.CONFIG_FIELDS if saved[f] != mine[f])
            raise ValueError(f"存档的编码器配置与当前不一致：{diff}")
        self.pulse.load_projection(state["projection"])
        return self

    def __repr__(self) -> str:
        return (f"SynapticaEncoder(hidden_size={self.hidden_size}, "
                f"num_neurons={self.num_neurons}, "
                f"segment_tokens={self.segment_tokens})")
