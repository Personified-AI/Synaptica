from __future__ import annotations

import json
import mmap
import os
import struct
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = ["MemoryFile", "MemoryFileError"]

_MAGIC = b"SYN8FILE"
_VERSION = 1
_HDR_SIZE = 256
_COMMIT_OFFS = (16, 128)
_COMMIT_BODY = struct.Struct("<QQQQQQQQQQQQQ")
_COMMIT_SIZE = 112
_SEG_ENTRY = struct.Struct("<IQIQ")
_BLOCK_HDR = struct.Struct("<QQ")
_EDGE = struct.Struct("<QQf")
_DEDUP = struct.Struct("<QI")
_META_LEN = struct.Struct("<I")
_CRC = struct.Struct("<I")

def _pread(fd: int, n: int, off: int) -> bytes:
    os.lseek(fd, off, os.SEEK_SET)
    out = bytearray()
    while len(out) < n:
        chunk = os.read(fd, n - len(out))
        if not chunk:
            break
        out += chunk
    return bytes(out)


def _pwrite(fd: int, data: bytes, off: int) -> None:
    os.lseek(fd, off, os.SEEK_SET)
    view = memoryview(data)
    while len(view):
        n = os.write(fd, view)
        if n <= 0:
            raise MemoryFileError("写入失败")
        view = view[n:]


_BINARY = getattr(os, "O_BINARY", 0)
_BITMAP_BYTES = 1024
_ID_MIN, _ID_MAX = -(2 ** 31), 2 ** 31 - 1


class MemoryFileError(RuntimeError):
    pass


class MemoryFile:
    def __init__(self, path: str, fd: int) -> None:
        self.path = str(path)
        self._fd = fd
        self._mm: Optional[mmap.mmap] = None
        self._size = 0
        self._seq = 0
        self._segtbl_head = 0
        self._segtbl_total = 0
        self._meta: Dict[str, Any] = {}
        self._entries: Dict[int, Tuple[int, int, int]] = {}
        self._pending: Dict[int, np.ndarray] = {}
        self._pending_bmp: Dict[int, bytes] = {}
        self._dirty: set = set()
        self._epos = self._empty_edges()
        self._eneg = self._empty_edges()
        self._mem = np.empty(0, dtype=np.float32)
        self._dedup_key = np.empty(0, dtype=np.uint64)
        self._dedup_slot = np.empty(0, dtype=np.uint32)
        self._delta: Dict[int, int] = {}

    @staticmethod
    def _empty_edges():
        z = np.empty(0, dtype=np.uint64)
        return (z, z, np.empty(0, dtype=np.float32))

    @classmethod
    def create(cls, path: str) -> "MemoryFile":
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC | _BINARY, 0o644)
        head = bytearray(_HDR_SIZE)
        head[0:len(_MAGIC)] = _MAGIC
        struct.pack_into("<I", head, 8, _VERSION)
        _pwrite(fd, bytes(head), 0)
        os.fsync(fd)
        self = cls(path, fd)
        self._size = _HDR_SIZE
        self._remap()
        return self

    @classmethod
    def open(cls, path: str) -> "MemoryFile":
        if not os.path.exists(path):
            raise MemoryFileError(f"文件不存在：{path}")
        self = cls(path, os.open(path, os.O_RDWR | _BINARY))
        self._read_header()
        self._remap()
        self._load_state()
        self._load_segments()
        self._load_dedup()
        return self

    @property
    def num_slots(self) -> int:
        return int(self._mem.size)

    @property
    def seq(self) -> int:
        return int(self._seq)

    @property
    def meta(self) -> Dict[str, Any]:
        return dict(self._meta)

    def _read(self, off: int, n: int) -> bytes:
        buf = _pread(self._fd, n, off)
        if len(buf) != n:
            raise MemoryFileError(f"读取越界：偏移 {off}，长度 {n}")
        return buf

    def _commit_fits(self, f: Tuple[int, ...]) -> bool:
        size = int(self._size)
        if int(f[3]) + _META_LEN.size + int(f[4]) > size:
            return False
        if int(f[5]) + int(f[6]) * _EDGE.size > size:
            return False
        if int(f[7]) + int(f[8]) * _EDGE.size > size:
            return False
        if int(f[9]) + int(f[10]) * 4 > size:
            return False
        if int(f[11]) + int(f[12]) * _DEDUP.size > size:
            return False
        if int(f[1]) and int(f[1]) + _BLOCK_HDR.size > size:
            return False
        return True

    def _read_header(self) -> None:
        head = self._read(0, _HDR_SIZE)
        if head[0:len(_MAGIC)] != _MAGIC:
            raise MemoryFileError(f"不是 Synaptica 记忆文件：{self.path}")
        ver = struct.unpack_from("<I", head, 8)[0]
        if ver != _VERSION:
            raise MemoryFileError(f"文件版本不支持：{ver}，当前 {_VERSION}")
        self._size = os.fstat(self._fd).st_size
        best: Optional[Tuple[int, Tuple[int, ...]]] = None
        for off in _COMMIT_OFFS:
            body = head[off:off + _COMMIT_BODY.size]
            crc = _CRC.unpack_from(head, off + _COMMIT_BODY.size)[0]
            if zlib.crc32(body) & 0xFFFFFFFF != crc:
                continue
            fields = _COMMIT_BODY.unpack(body)
            if int(fields[0]) == 0:
                continue
            if not self._commit_fits(fields):
                continue
            if best is None or int(fields[0]) > best[0]:
                best = (int(fields[0]), fields)
        if best is None:
            return
        f = best[1]
        self._seq = int(f[0])
        self._segtbl_head = int(f[1])
        self._segtbl_total = int(f[2])
        self._meta_off = int(f[3])
        self._meta_len = int(f[4])
        self._epos_off, self._epos_len = int(f[5]), int(f[6])
        self._eneg_off, self._eneg_len = int(f[7]), int(f[8])
        self._mem_off, self._mem_n = int(f[9]), int(f[10])
        self._dedup_off, self._dedup_n = int(f[11]), int(f[12])

    def _remap(self) -> None:
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        self._size = os.fstat(self._fd).st_size
        self._mm = mmap.mmap(self._fd, self._size, access=mmap.ACCESS_READ)

    def _load_state(self) -> None:
        if self._seq == 0:
            return
        n = _META_LEN.unpack(self._read(self._meta_off, _META_LEN.size))[0]
        raw = self._read(self._meta_off + _META_LEN.size, int(n))
        self._meta = json.loads(raw.decode("utf-8"))
        self._epos = self._read_edges(self._epos_off, self._epos_len)
        self._eneg = self._read_edges(self._eneg_off, self._eneg_len)
        buf = self._read(self._mem_off, self._mem_n * 4)
        self._mem = np.frombuffer(buf, dtype=np.float32).copy()

    def _read_edges(self, off: int, n: int):
        if n == 0:
            return self._empty_edges()
        buf = self._read(off, n * _EDGE.size)
        arr = np.frombuffer(buf, dtype=np.dtype([("s", "<u8"), ("d", "<u8"), ("v", "<f4")]))
        return (arr["s"].astype(np.uint64, copy=True),
                arr["d"].astype(np.uint64, copy=True),
                arr["v"].astype(np.float32, copy=True))

    def _load_segments(self) -> None:
        self._entries.clear()
        blocks: List[Tuple[int, int]] = []
        off = self._segtbl_head
        seen = 0
        while off != 0:
            prev, n = _BLOCK_HDR.unpack(self._read(off, _BLOCK_HDR.size))
            blocks.append((int(off), int(n)))
            seen += int(n)
            off = int(prev)
        for blk_off, n in reversed(blocks):
            buf = self._read(blk_off + _BLOCK_HDR.size, n * _SEG_ENTRY.size)
            for i in range(n):
                slot, seg_off, cnt, bmp_off = _SEG_ENTRY.unpack_from(buf, i * _SEG_ENTRY.size)
                self._entries[int(slot)] = (int(seg_off), int(cnt), int(bmp_off))
        if seen != self._segtbl_total:
            raise MemoryFileError(f"SEGTBL 条目数不符：{seen} vs {self._segtbl_total}")

    def _load_dedup(self) -> None:
        if self._seq == 0 or self._dedup_n == 0:
            return
        buf = self._read(self._dedup_off, self._dedup_n * _DEDUP.size)
        arr = np.frombuffer(buf, dtype=np.dtype([("k", "<u8"), ("s", "<u4")]))
        self._dedup_key = arr["k"].astype(np.uint64, copy=True)
        self._dedup_slot = arr["s"].astype(np.uint32, copy=True)

    def _append(self, data: bytes) -> int:
        off = self._size
        if data:
            _pwrite(self._fd, data, off)
            self._size = off + len(data)
        return off

    def append_segment(self, slot: int, ids: Sequence[int], bitmap: Optional[bytes]) -> None:
        s = int(slot)
        if s < 0:
            raise MemoryFileError(f"槽位不能为负：{s}")
        seq = [int(x) for x in ids]
        if not seq:
            raise MemoryFileError("段内容不能为空")
        for x in seq:
            if x < _ID_MIN or x > _ID_MAX:
                raise MemoryFileError(f"token id 超出 int32：{x}")
        n = len(seq)
        off = self._append(struct.pack("<II", s, n) + struct.pack(f"<{n}i", *seq))
        if bitmap is None:
            bmp_off = 0
        else:
            raw = bytes(bitmap)
            if len(raw) != _BITMAP_BYTES:
                raise MemoryFileError(f"位图长度必须是 {_BITMAP_BYTES} 字节，收到 {len(raw)}")
            bmp_off = self._append(raw)
        self._entries[s] = (int(off), int(n), int(bmp_off))
        self._pending[s] = np.asarray(seq, dtype=np.int32)
        if bitmap is not None:
            self._pending_bmp[s] = bytes(bitmap)
        self._dirty.add(s)

    def has(self, slot: int) -> bool:
        return int(slot) in self._entries

    def slots(self) -> List[int]:
        return sorted(self._entries)

    def segment_ids(self, slot: int) -> Optional[np.ndarray]:
        s = int(slot)
        p = self._pending.get(s)
        if p is not None:
            return p
        e = self._entries.get(s)
        if e is None:
            return None
        off, n, _ = e
        if off + 8 + n * 4 > self._size:
            raise MemoryFileError(f"段 {s} 越界")
        return np.frombuffer(self._mm, dtype=np.int32, count=n, offset=off + 8)

    def bitmap(self, slot: int) -> Optional[np.ndarray]:
        p = self._pending_bmp.get(int(slot))
        if p is not None:
            return np.frombuffer(p, dtype=np.uint8)
        e = self._entries.get(int(slot))
        if e is None:
            return None
        bmp_off = e[2]
        if bmp_off == 0:
            return None
        if bmp_off + _BITMAP_BYTES > self._size:
            raise MemoryFileError(f"段 {int(slot)} 的位图越界")
        return np.frombuffer(self._mm, dtype=np.uint8, count=_BITMAP_BYTES, offset=bmp_off)

    def lookup(self, key: int) -> Optional[int]:
        k = int(key) & 0xFFFFFFFFFFFFFFFF
        hit = self._delta.get(k)
        if hit is not None:
            return int(hit)
        if self._dedup_key.size == 0:
            return None
        i = int(np.searchsorted(self._dedup_key, np.uint64(k), side="left"))
        if i >= self._dedup_key.size:
            return None
        if int(self._dedup_key[i]) != k:
            return None
        return int(self._dedup_slot[i])

    def add_key(self, key: int, slot: int) -> None:
        self._delta[int(key) & 0xFFFFFFFFFFFFFFFF] = int(slot)

    def edges(self):
        return (self._epos, self._eneg)

    def mem(self) -> np.ndarray:
        return self._mem

    def save(self,
             meta: Dict[str, Any],
             epos: Tuple[Any, Any, Any],
             eneg: Tuple[Any, Any, Any],
             mem: Any) -> None:
        raw_meta = json.dumps(meta, ensure_ascii=False).encode("utf-8")
        meta_off = self._append(_META_LEN.pack(len(raw_meta)) + raw_meta)
        epos_off, epos_n = self._write_edges(epos)
        eneg_off, eneg_n = self._write_edges(eneg)
        self._epos = self._as_np(epos)
        self._eneg = self._as_np(eneg)

        if isinstance(mem, torch.Tensor):
            m = mem.detach().to("cpu", torch.float32).numpy().astype(np.float32, copy=True).reshape(-1)
        else:
            m = np.asarray(mem, dtype=np.float32).reshape(-1).copy()
        mem_off = self._append(m.tobytes())
        mem_n = int(m.size)

        dk, ds = self._merged_dedup()
        if dk.size:
            rec = np.empty(int(dk.size), dtype=np.dtype([("k", "<u8"), ("s", "<u4")]))
            rec["k"] = dk
            rec["s"] = ds
            dedup_off = int(self._append(rec.tobytes()))
        else:
            dedup_off = int(self._append(b""))

        entries = sorted(self._dirty)
        blk = _BLOCK_HDR.pack(int(self._segtbl_head), len(entries))
        body = b"".join(_SEG_ENTRY.pack(int(s), int(self._entries[s][0]),
                                        int(self._entries[s][1]), int(self._entries[s][2]))
                        for s in entries)
        segtbl_off = int(self._append(blk + body))
        total = int(self._segtbl_total) + len(entries)

        seq = self._seq + 1
        commit = _COMMIT_BODY.pack(seq, segtbl_off, total,
                                   meta_off, len(raw_meta), epos_off, epos_n, eneg_off, eneg_n,
                                   mem_off, mem_n, dedup_off, int(dk.size))
        crc = zlib.crc32(commit) & 0xFFFFFFFF
        rec = commit + _CRC.pack(crc)
        rec += b"\x00" * (_COMMIT_SIZE - len(rec))
        _pwrite(self._fd, rec, _COMMIT_OFFS[seq % 2])
        os.fsync(self._fd)

        self._seq = seq
        self._segtbl_head = segtbl_off
        self._segtbl_total = total
        self._dirty.clear()
        self._meta = dict(meta)
        self._mem = m
        self._dedup_key, self._dedup_slot = dk, ds
        self._delta.clear()
        self._pending.clear()
        self._pending_bmp.clear()
        self._remap()

    @staticmethod
    def _as_np(part: Tuple[Any, Any, Any]):
        s, d, v = part
        s = np.asarray(s.detach().to("cpu") if isinstance(s, torch.Tensor) else s, dtype=np.uint64).reshape(-1)
        d = np.asarray(d.detach().to("cpu") if isinstance(d, torch.Tensor) else d, dtype=np.uint64).reshape(-1)
        v = np.asarray(v.detach().to("cpu", torch.float32) if isinstance(v, torch.Tensor) else v,
                       dtype=np.float32).reshape(-1)
        if not (s.size == d.size == v.size):
            raise MemoryFileError(f"边表三段长度不一致：{s.size}/{d.size}/{v.size}")
        return s, d, v

    def _write_edges(self, part: Tuple[Any, Any, Any]) -> Tuple[int, int]:
        s, d, v = self._as_np(part)
        n = int(s.size)
        if n == 0:
            return int(self._append(b"")), 0
        rec = np.empty(n, dtype=np.dtype([("s", "<u8"), ("d", "<u8"), ("v", "<f4")]))
        rec["s"] = s
        rec["d"] = d
        rec["v"] = v
        return int(self._append(rec.tobytes())), n

    def _merged_dedup(self):
        if self._delta:
            keys = np.fromiter(self._delta.keys(), dtype=np.uint64, count=len(self._delta))
            slots = np.fromiter(self._delta.values(), dtype=np.uint32, count=len(self._delta))
            all_k = np.concatenate([self._dedup_key, keys])
            all_s = np.concatenate([self._dedup_slot, slots])
        else:
            all_k, all_s = self._dedup_key, self._dedup_slot
        if all_k.size == 0:
            return all_k, all_s
        order = np.argsort(all_k, kind="stable")
        all_k = all_k[order]
        all_s = all_s[order]
        keep = np.ones(all_k.size, dtype=bool)
        keep[:-1] = all_k[1:] != all_k[:-1]
        return all_k[keep], all_s[keep]

    def close(self) -> None:
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        try:
            os.close(self._fd)
        except OSError:
            pass
