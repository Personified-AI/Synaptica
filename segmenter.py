
from __future__ import annotations

from typing import List, Protocol, Tuple

from torch import Tensor

__all__ = ["Segmenter", "BlockSegmenter"]

class Segmenter(Protocol):

    def segment(
        self, hidden: Tensor, mask: Tensor, *, is_final: bool
    ) -> List[Tuple[int, int]]: ...

class BlockSegmenter:

    def __init__(self, segment_tokens: int) -> None:
        if int(segment_tokens) <= 0:
            raise ValueError("segment_tokens 必须是正整数")
        self.segment_tokens = int(segment_tokens)

    def segment(
        self, hidden: Tensor, mask: Tensor = None, *, is_final: bool = True
    ) -> List[Tuple[int, int]]:
        t = int(hidden.shape[0])
        if t <= 0:
            return []
        w = self.segment_tokens
        if t % w != 0:
            raise ValueError(f"块长 {t} 不是段长 {w} 的整数倍（余 {t % w}）")
        return [(s, s + w) for s in range(0, t, w)]

    def __repr__(self) -> str:
        return f"BlockSegmenter(segment_tokens={self.segment_tokens})"
