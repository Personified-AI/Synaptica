
from __future__ import annotations

import io
import operator
import os
import struct
from collections.abc import Sequence as _AbcSequence
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

__all__ = ["SegmentStore", "SegmentStoreError", "as_token_ids"]

_MAGIC = b"SYNS"
_HDR = struct.Struct("<4sii")
_INT = struct.Struct("<i")

_ID_MIN, _ID_MAX = -(2 ** 31), 2 ** 31 - 1

class SegmentStoreError(RuntimeError):
    pass

def as_token_ids(payload: Any) -> List[int]:
    if isinstance(payload, (str, bytes, bytearray)):
        raise TypeError(f"需要 token id 序列（Sequence[int]），收到 {type(payload).__name__}")
    if isinstance(payload, torch.Tensor):
        items: List[Any] = payload.reshape(-1).tolist()
    elif isinstance(payload, _AbcSequence):
        items = list(payload)
    else:
        raise TypeError(f"需要 token id 序列（Sequence[int]），收到 {type(payload).__name__}")
    if not items:
        raise ValueError("空序列不能作为段内容")
    ids: List[int] = []
    for x in items:
        if isinstance(x, bool):

            raise TypeError(f"token id 必须是整数，收到 bool {x!r}")
        try:
            value = operator.index(x)
        except TypeError:
            raise TypeError(f"token id 必须是整数，收到 {type(x).__name__} {x!r}") from None
        if not _ID_MIN <= value <= _ID_MAX:
            raise ValueError(f"token id {value} 超出 int32 范围")
        ids.append(value)
    return ids

class SegmentStore:

    def __init__(self, path: str, *, fsync: bool = False) -> None:
        self.path = str(path)
        self.fsync = bool(fsync)
        self._index: Dict[int, int] = {}
        self._fh: Optional[io.BufferedRandom] = None
        self.n_reads = 0
        self.n_writes = 0
        self._open()

    def _open(self) -> None:
        new = not os.path.exists(self.path)

        self._fh = io.open(self.path, "w+b" if new else "r+b")
        if not new:
            self._scan()

    def _scan(self) -> None:
        fh = self._require_fh()
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(0)
        pos = 0
        self._index.clear()
        while pos < size:
            head = fh.read(_HDR.size)
            if len(head) < _HDR.size:
                raise SegmentStoreError(
                    f"{self.path}: 偏移 {pos} 处记录头不完整"
                    f"（{len(head)} 字节，应为 {_HDR.size}）"
                )
            magic, slot, n = _HDR.unpack(head)
            if magic != _MAGIC:
                raise SegmentStoreError(f"{self.path}: 偏移 {pos} 处记录标记 {magic!r} 不是 {_MAGIC!r}")
            if n < 0:
                raise SegmentStoreError(f"{self.path}: 偏移 {pos} 处 token 数为负（{n}）")
            body = fh.read(4 * n)
            if len(body) < 4 * n:
                raise SegmentStoreError(
                    f"{self.path}: 偏移 {pos} 处记录体不完整"
                    f"（{len(body)} 字节，应为 {4 * n}）"
                )
            if n > 0:
                self._index[int(slot)] = pos
            else:

                self._index.pop(int(slot), None)
            pos += _HDR.size + 4 * n

    def _require_fh(self) -> io.BufferedRandom:
        if self._fh is None:
            raise SegmentStoreError(f"{self.path} 已关闭")
        return self._fh

    @property
    def num_segments(self) -> int:
        return len(self._index)

    @property
    def slots(self) -> List[int]:
        return sorted(self._index)

    def has(self, slot: int) -> bool:
        return int(slot) in self._index

    def append(self, slot: int, token_ids: Sequence[int]) -> None:
        ids = as_token_ids(token_ids)
        fh = self._require_fh()
        fh.seek(0, os.SEEK_END)
        pos = fh.tell()
        fh.write(_HDR.pack(_MAGIC, int(slot), len(ids)))
        fh.write(struct.pack(f"<{len(ids)}i", *ids))
        if self.fsync:
            fh.flush()
            os.fsync(fh.fileno())
        self._index[int(slot)] = pos
        self.n_writes += 1

    def append_many(self, items: Iterable[Tuple[int, Sequence[int]]]) -> int:
        rows = [(int(slot), as_token_ids(token_ids)) for slot, token_ids in items]
        fh = self._require_fh()
        fh.seek(0, os.SEEK_END)
        for slot, ids in rows:
            pos = fh.tell()
            fh.write(_HDR.pack(_MAGIC, slot, len(ids)))
            fh.write(struct.pack(f"<{len(ids)}i", *ids))
            self._index[slot] = pos
        if self.fsync:
            fh.flush()
            os.fsync(fh.fileno())
        self.n_writes += len(rows)
        return len(rows)

    def read(self, slot: int) -> Optional[List[int]]:
        pos = self._index.get(int(slot))
        if pos is None:
            return None
        fh = self._require_fh()
        fh.seek(pos)
        head = fh.read(_HDR.size)
        magic, got_slot, n = _HDR.unpack(head)
        if magic != _MAGIC:
            raise SegmentStoreError(f"{self.path}: 槽 {slot} 的记录标记损坏（读到 {magic!r}）")
        if int(got_slot) != int(slot):
            raise SegmentStoreError(f"{self.path}: 索引称偏移 {pos} 是槽 {slot}，文件里写的是 {got_slot}")
        body = fh.read(4 * n)
        if len(body) < 4 * n:
            raise SegmentStoreError(f"{self.path}: 槽 {slot} 的记录体被截断")
        self.n_reads += 1
        return list(struct.unpack(f"<{n}i", body)) if n else []

    def read_many(self, slots: Sequence[int]) -> Dict[int, List[int]]:
        out: Dict[int, List[int]] = {}
        for s in slots:
            v = self.read(int(s))
            if v is not None:
                out[int(s)] = v
        return out

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "SegmentStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (f"SegmentStore({self.path!r}, segments={self.num_segments}, "
                f"reads={self.n_reads}, writes={self.n_writes})")
