
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..types import Span

__all__ = ["GenAttnRewardConfig", "GenAttnRewardResult", "GenAttnReward"]

@dataclass
class GenAttnRewardConfig:

    layer: int = 19

    cut_q: float = 0.25

    row_block: int = 512

    def __post_init__(self) -> None:
        if not 0.0 < float(self.cut_q) <= 1.0:
            raise ValueError("cut_q 必须在 (0, 1]")

@dataclass
class GenAttnRewardResult:

    u: Dict[int, float] = field(default_factory=dict)
    cut: float = 0.0
    reward: Dict[int, float] = field(default_factory=dict)
    n_missing: int = 0

    @property
    def n_positive(self) -> int:
        return sum(1 for v in self.reward.values() if v > 0)

    @property
    def n_negative(self) -> int:
        return sum(1 for v in self.reward.values() if v < 0)

    @property
    def n_attended(self) -> int:
        return sum(1 for v in self.u.values() if v >= self.cut)

class GenAttnReward:

    def __init__(self, backend: Any, config: Optional[GenAttnRewardConfig] = None) -> None:
        self.backend = backend
        self.config = config or GenAttnRewardConfig()

    def mass(self, *, prompt_ids: Sequence[int], gen_ids: Sequence[int],
             items: Sequence[Tuple[int, Span]]) -> Dict[int, float]:
        total = list(prompt_ids) + list(gen_ids)
        spans = [(int(sp[0]), int(sp[1])) for _s, sp in items]
        slots = [int(s) for s, _sp in items]
        cfg = self.config

        hook = getattr(self.backend, "decode_attention_mass", None)
        if not callable(hook):
            raise TypeError(
                f"{type(self.backend).__name__} 缺少 decode_attention_mass"
                f"(*, total_ids, layer, spans, slots, prompt_len, gen_len, row_block)"
            )
        return {int(k): float(v) for k, v in hook(
            total_ids=total, layer=int(cfg.layer),
            spans=spans, slots=slots, prompt_len=len(prompt_ids),
            gen_len=len(gen_ids),
            row_block=int(cfg.row_block)).items()}

    def __call__(self, *, prompt_ids: Sequence[int], gen_ids: Sequence[int],
                 items: Sequence[Tuple[int, Span]], exclude: Sequence[int] = ()
                 ) -> GenAttnRewardResult:
        cfg = self.config
        res = GenAttnRewardResult()
        keep: List[Tuple[int, Span]] = []
        for s, sp in items:
            if sp is None or not (int(sp[0]) < int(sp[1])):
                res.n_missing += 1
                continue
            keep.append((int(s), (int(sp[0]), int(sp[1]))))
        if not keep or not gen_ids:
            return res

        res.u = self.mass(prompt_ids=prompt_ids, gen_ids=gen_ids, items=keep)
        excl = {int(i) for i in exclude}

        res.u = {s: v for s, v in res.u.items() if s not in excl}
        if not res.u:
            return res
        vals = sorted(res.u.values(), reverse=True)
        if not vals:
            return res
        k = max(1, int(round(float(cfg.cut_q) * len(vals))))
        res.cut = float(vals[k - 1])
        res.reward = {s: (1.0 if v >= res.cut else -1.0)
                      for s, v in res.u.items() if s not in excl}
        return res

    def __repr__(self) -> str:
        return f"GenAttnReward(layer={self.config.layer}, cut_q={self.config.cut_q})"
