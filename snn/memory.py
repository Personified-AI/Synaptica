
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from array import array
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from ..ann.seq_match import SeqMatcher
from .node import LateralConfig, LateralState, SynapticaNode
from .store import SegmentStore, as_token_ids

__all__ = ["RecallResult", "WriteStats", "PatternPrep", "SynapticaMemory"]

_BIT_W = (1, 2, 4, 8, 16, 32, 64, 128)
_POOL_CHUNK = 8192


def _pack_bits(bits: Tensor) -> Tensor:
    b = bits.to(torch.int64).reshape(-1)
    pad = (-int(b.numel())) % 8
    if pad:
        b = torch.cat([b, torch.zeros(pad, dtype=torch.int64)])
    b = b.reshape(-1, 8)
    w = torch.tensor(_BIT_W, dtype=torch.int64)
    return (b * w).sum(dim=1).to(torch.uint8)


def _unpack_bits(packed: Tensor, n: int) -> Tensor:
    if packed.device.type == "cpu":
        _arr = np.frombuffer(packed.contiguous().numpy().tobytes(), dtype=np.uint8)
        out = np.unpackbits(_arr, bitorder="little")[: int(n)]
        return torch.from_numpy(out.copy()).to(torch.float32)
    col = packed.to(torch.int64).reshape(-1, 1)
    shifts = torch.arange(8, dtype=torch.int64)
    out = ((col >> shifts) & 1).reshape(-1)[: int(n)]
    return out.to(torch.float32)


def _sparse_add(ai: Tensor, av: Tensor, bi: Tensor,
                bv: Tensor) -> Tuple[Tensor, Tensor]:
    if int(ai.numel()) == 0:
        return bi, bv
    if int(bi.numel()) == 0:
        return ai, av
    k = torch.cat([ai, bi])
    v = torch.cat([av, bv])
    order = torch.argsort(k, stable=True)
    k, v = k[order], v[order]
    uniq, inv = torch.unique_consecutive(k, return_inverse=True)
    acc = torch.zeros(int(uniq.numel()), dtype=v.dtype, device=v.device)
    acc.index_add_(0, inv, v)
    keep = acc != 0
    return uniq[keep], acc[keep]

class PatternPrep:

    def __call__(self, pattern: Tensor) -> Tensor:
        p = torch.as_tensor(pattern).float().reshape(-1)
        return p / p.sum().clamp_min(1.0)

@dataclass
class RecallResult:

    order: List[int] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    spikes: Optional[Tensor] = None
    membrane: Optional[Tensor] = None

@dataclass
class WriteStats:

    index: int
    reward: float
    repeats: int
    eligibility_norm: float
    delta_abs_mean: float
    skipped: bool = False

    def as_dict(self) -> dict:
        return asdict(self)

class SynapticaMemory:

    def __init__(
        self,
        input_size: int,
        num_slots: int,
        *,
        device: Optional[Union[str, torch.device]] = None,
        lateral_cfg: Optional[LateralConfig] = None,
        store: Optional[SegmentStore] = None,
        max_cached_segments: int = 0,
        path: Optional[str] = None,
    ) -> None:
        if int(input_size) <= 0:
            raise ValueError(f"input_size 必须是正整数，收到 {input_size}")
        if int(num_slots) <= 0:
            raise ValueError(f"num_slots 必须是正整数，收到 {num_slots}")
        self.input_size = int(input_size)
        self.num_slots = int(num_slots)
        self.prepare = PatternPrep()

        self.lateral_cfg = lateral_cfg or LateralConfig()

        self.segment_store = store
        self.file: Optional[Any] = None
        self._file_ids: Dict[int, List[int]] = {}
        self._file_pending: Dict[int, bytes] = {}

        self.max_cached_segments = int(max_cached_segments)
        self._cache_order: List[int] = []

        self.node = SynapticaNode(
            input_size=self.input_size,
            output_size=self.num_slots,
            device=device,
        )

        self.node.attach_lateral(cfg=self.lateral_cfg)

        self._pattern_index: Dict[int, int] = {}

        self.n_pattern_hits: int = 0

        self.n_pattern_encode_errors: int = 0
        self._pattern_encode_error_note: Optional[str] = None
        self.payloads: Dict[int, Any] = {}

        self.embedding_dim: int = 0
        self.token_dim: int = 0

        self.pattern_len: int = 0

        self._token_dims: int = 0

        self._pattern_cache: Dict[int, Tuple[int, Any, float, int, Any]] = {}
        self._slot_ver: Dict[int, int] = {}

        self.segment_encoder: Optional[Any] = None
        self.encoder_fp: str = ""

        self.pad_token_id: Optional[int] = None

        self.special_token_ids: set = set()

        self.token_counts: Dict[int, int] = {}
        self.token_total: int = 0

        self._state_version: int = 0

        self._c_slots: Tuple[int, List[int]] = (-1, [])

        self._c_segments: Tuple[int, List[int]] = (-1, [])

        self._c_written: Tuple[int, Optional[Tensor]] = (-1, None)

        self._c_pool_item: Dict[int, Tuple[int, Tuple[int, Tensor, Tensor]]] = {}

        self._c_pool_aux: Dict[Any, Tuple[List[Any], List[Any], set]] = {}

        if path is not None:
            self._open_file(str(path))

    def _open_file(self, path: str) -> None:
        from .memfile import MemoryFile

        mf = MemoryFile.open(path) if os.path.exists(path) else MemoryFile.create(path)
        self.file = mf
        meta = dict(mf.meta or {})
        self.encoder_fp = str(meta.get("encoder", "") or "")
        if meta:
            self.embedding_dim = int(meta.get("embedding_dim", 0))
            self.token_dim = int(meta.get("token_dim", 0))
            self.pattern_len = int(meta.get("pattern_len", 0))
            self._token_dims = int(meta.get("token_dims", 0))
        n = int(mf.num_slots)
        if n > 0:
            self.ensure_slots(n)
            mem = mf.mem()
            if mem is not None and int(mem.size) >= int(self.num_slots):
                self.node.mem = torch.from_numpy(
                    np.asarray(mem[: int(self.num_slots)], dtype=np.float32).copy()
                ).to(self.node.mem.device)
            (pos_s, pos_d, pos_v), (neg_s, neg_d, neg_v) = mf.edges()
            pos_i = torch.stack([
                torch.from_numpy(np.asarray(pos_s).copy()).long(),
                torch.from_numpy(np.asarray(pos_d).copy()).long(),
            ])
            neg_i = torch.stack([
                torch.from_numpy(np.asarray(neg_s).copy()).long(),
                torch.from_numpy(np.asarray(neg_d).copy()).long(),
            ])
            if int(pos_i.shape[1]) > 0:
                self.node.attach_lateral(cfg=self.lateral_cfg)
                state: Dict[str, Any] = {
                    "index": pos_i,
                    "values": torch.from_numpy(np.asarray(pos_v).copy().astype("float32")),
                    "edge_order": "src_dest",
                    "config": self.lateral_cfg.to_json(),
                }
                if int(neg_i.shape[1]) > 0:
                    state["neg_index"] = neg_i
                    state["neg_values"] = torch.from_numpy(np.asarray(neg_v).copy().astype("float32"))
                self.node.lateral = LateralState.load_state_dict(
                    state, node=self.node
                )
        self._bump()

    def check_encoder(self, fingerprint: str) -> None:
        want = str(self.encoder_fp or "")
        got = str(fingerprint or "")
        if want and got and want != got:
            raise ValueError(f"记忆文件的编码器指纹不一致：存档 {want} vs 当前 {got}")

    def save(self, path: Optional[str] = None, **meta: Any) -> int:
        if self.file is None:
            if path is None:
                raise RuntimeError("未配置文件：构造时传 path，或调用 save(path=...)")
            self._open_file(str(path))
        for slot, raw in list(self._file_pending.items()):
            sid = int(slot)
            self.file.append_segment(sid, self._file_ids.get(sid, []), raw)
        self._file_pending.clear()

        lat = self.node.lateral
        if lat is None:
            pos_i = torch.zeros(2, 0, dtype=torch.long)
            pos_v = torch.zeros(0)
            neg_i = torch.zeros(2, 0, dtype=torch.long)
            neg_v = torch.zeros(0)
        else:
            pos_i = lat.index.detach().cpu().long()
            pos_v = lat.values.detach().cpu().float()
            neg_i = lat.neg_index.detach().cpu().long()
            neg_v = lat.neg_values.detach().cpu().float()
        epos = (pos_i[0], pos_i[1], pos_v)
        eneg = (neg_i[0], neg_i[1], neg_v)
        mem = self.node.mem.detach().cpu().float().numpy()
        out_meta = {
            "input_size": int(self.input_size),
            "embedding_dim": int(self.embedding_dim),
            "token_dim": int(self.token_dim),
            "pattern_len": int(self.pattern_len),
            "token_dims": int(self._token_dims),
            "encoder": str(meta.get("encoder", "")),
        }
        self.file.save(out_meta, epos, eneg, mem)
        self._bump()
        return int(self.file.seq)

    def _bump(self) -> int:
        self._state_version = int(self._state_version) + 1
        self._c_slots = (-1, [])
        self._c_segments = (-1, [])
        self._c_written = (-1, None)
        self._c_pool_item.clear()
        self._c_pool_aux.clear()
        return self._state_version

    def payload(self, index: int) -> Any:
        return self.segment_token_ids(index) or []

    def written_mask(self) -> Tensor:
        _ver, _cached = self._c_written
        if _cached is not None and int(_ver) == int(self._state_version):
            return _cached.clone()
        m = torch.zeros(
            self.num_slots, dtype=torch.bool, device=self.node.mem.device
        )
        keys = self.stored_slots()
        if keys:
            m[torch.tensor(sorted(set(keys)), device=m.device)] = True
        self._c_written = (int(self._state_version), m)
        return m.clone()

    def ensure_slots(self, n: int) -> int:
        n = int(n)
        if n > self.num_slots:
            self.node.grow(n)
            self.num_slots = int(self.node.output_size)
            if self.node.lateral is not None:
                self.node.lateral.ensure_slots(self.num_slots)

            self._bump()
        return self.num_slots

    def ensure_laterals_slots(self) -> int:
        lat = self.node.lateral
        if lat is None:
            raise RuntimeError("侧向突触未初始化")
        return int(lat.num_slots)

    @property
    def num_laterals_edges(self) -> int:
        lat = self.node.lateral
        if self.node.lateral is not None and self.node.lateral.neg_index.shape[1]:
            return int(self.node.lateral.num_edges + self.node.lateral.neg_index.shape[1])
        return 0 if lat is None else int(lat.num_edges)

    @property
    def mean_laterals_slots(self) -> float:
        lat = self.node.lateral
        return 0.0 if lat is None else float(lat.mean_degree)

    @property
    def laterals_index(self) -> Tensor:
        lat = self.node.lateral
        if lat is None:
            return torch.zeros(2, 0, dtype=torch.long)
        if lat.neg_index.shape[1]:
            return torch.cat([lat.index, lat.neg_index], dim=1)
        return lat.index

    @property
    def laterals_values(self) -> Tensor:
        lat = self.node.lateral
        if lat is None:
            return torch.zeros(0)
        if lat.neg_values.numel():
            return torch.cat([lat.values, lat.neg_values])
        return lat.values

    def note_attention_u(self, u_by_slot: Dict[int, float]) -> None:
        if not u_by_slot:
            return
        _n = max(int(k) for k in u_by_slot) + 1
        _cur = getattr(self, "attn_u", None)
        if _cur is None or int(_cur.numel()) < _n:
            _new = torch.zeros(max(_n, int(_cur.numel()) if _cur is not None else 0),
                               dtype=torch.float32)
            if _cur is not None:
                _new[:int(_cur.numel())] = _cur
            self.attn_u = _new
            _cur = _new
        for _k, _v in u_by_slot.items():
            _i = int(_k)
            if _i < int(_cur.numel()) and float(_v) > float(_cur[_i]):
                _cur[_i] = float(_v)

    def attention_marge_laterals(
        self, graph: Any, *, slot_index: Sequence[int], prefix_len: int
    ) -> Tuple[int, int]:
        lat = self.node.lateral
        if lat is None:
            raise RuntimeError("侧向突触未初始化")
        self.ensure_laterals_slots()
        before = int(lat.num_edges)
        lat.merge_edges(
            graph,
            slot_index=list(slot_index),
            prefix_len=int(prefix_len),
            alpha=float(self.lateral_cfg.alpha),
            max_new_per_segment=int(self.lateral_cfg.max_new_edges_per_segment),
        )
        after = int(lat.num_edges)
        n_cross = int(
            lat.count_edges(
                dest=list(range(int(prefix_len))),
                src=list(slot_index[int(prefix_len) :]),
            )
        )
        return after - before, n_cross

    def add_laterals(
        self, dest: Sequence[int], src: Sequence[int], values: Sequence[float],
        *, weak: bool = False
    ) -> int:
        lat = self.node.lateral
        if lat is None:
            raise RuntimeError("侧向突触未初始化")
        st = lat.add_laterals(list(dest), list(src), list(values),
                              mode="create_only", weak=bool(weak))
        return int(st.get("created", 0))

    def has_lateral(self, dest: int, src: int) -> bool:
        lat = self.node.lateral
        if lat is None or lat.num_edges == 0:
            return False
        m = (lat.index[1] == int(dest)) & (lat.index[0] == int(src))
        return bool(m.any())

    def add_negative_laterals(
        self,
        u: Dict[int, float],
        *,
        attended: Optional[Sequence[int]] = None,
        exclude: Sequence[int] = (),
        max_per_dest: Optional[int] = None,
    ) -> int:
        lat = self.node.lateral
        if lat is None:
            raise RuntimeError("侧向突触未初始化")
        cfg = self.lateral_cfg
        return int(
            lat.add_negative_edges(
                u,
                attended=attended,
                gain=float(cfg.neg_gain),
                exclude=exclude,
                q=float(cfg.attn_q),

                max_per_dest=int(max_per_dest or 0),
            )
        )

    def walked_laterals(self) -> List[Tuple[int, int, float]]:
        lat = self.node.lateral
        return [] if lat is None else list(lat.edges())

    def modulate_laterals(
        self, edges: Sequence[Tuple[int, int, float]], mod: Dict[int, float]
    ) -> Dict[str, Any]:
        lat = self.node.lateral
        if lat is None:
            raise RuntimeError("侧向突触未初始化")
        cfg = self.lateral_cfg
        return dict(
            lat.modulate(
                list(edges),
                mod,
                lr=float(cfg.lr),
                decay=float(cfg.decay),
            )
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "input_size": int(self.input_size),
            "num_slots": int(self.num_slots),

            "payloads": {
                int(s): tuple(self.segment_token_ids(s) or [])
                for s in self.stored_slots()
            },

            "embedding_dim": int(self.embedding_dim),
            "token_dim": int(self.token_dim),

            "pattern_len": int(self.pattern_len),
            "token_dims": int(self._token_dims),
            "token_counts": {int(k): int(v) for k, v in self.token_counts.items()},
            "token_total": int(self.token_total),
            "node": self.node.state_dict(),
        }

    @staticmethod
    def _as_token_ids(p: Any) -> List[int]:
        return as_token_ids(p)

    def load_state_dict(
        self,
        state: Dict[str, Any],
        *,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "SynapticaMemory":
        if int(state["input_size"]) != int(self.input_size):
            raise ValueError(f"input_size 不一致：存档 {state['input_size']} vs 当前 {self.input_size}")
        dev = torch.device(device) if device is not None else self.node.mem.device
        self.payloads = {
            int(k): array("i", (int(x) for x in v))
            for k, v in state["payloads"].items()
        }

        self._cache_order = sorted(self.payloads)
        self._pattern_cache.clear()
        self._slot_ver.clear()
        self.embedding_dim = int(state["embedding_dim"])
        self.token_dim = int(state["token_dim"])
        self.pattern_len = int(state.get("pattern_len", 0))
        self._token_dims = int(state.get("token_dims", 0))
        self.token_counts = {int(k): int(v) for k, v in state["token_counts"].items()}
        self.token_total = int(state["token_total"])
        self.num_slots = int(state["num_slots"])

        self.node = SynapticaNode(
            input_size=self.input_size, output_size=max(1, self.num_slots), device=dev
        )
        self.node.load_state_dict(state["node"], device=dev)
        self.num_slots = int(self.node.output_size)

        if self.node.lateral is not None:
            self.lateral_cfg = self.node.lateral.cfg

        self._bump()
        return self

    def store(self, payload: Any, index: Optional[int] = None, *,
              pattern: Optional[Tensor] = None) -> int:

        if index is not None:
            self._pattern_cache.pop(int(index), None)
        idx = self.store_payload_only(payload, index, pattern=pattern)
        ids = self.segment_token_ids(idx) or []
        if ids:
            self.observe_tokens(ids)
        return idx

    def store_payload_only(self, payload: Any, index: Optional[int] = None, *,
                           pattern: Optional[Tensor] = None) -> int:

        _auto_slot = index is None
        if index is None:
            index = self._next_free_slot()
        index = int(index)
        if index < 0:
            raise ValueError(f"槽号不能为负：{index}")
        if index >= self.num_slots:

            self.ensure_slots(index + 1)
        ids = self._as_token_ids(payload)

        _new_pattern = None
        if _auto_slot:
            try:

                if pattern is None:
                    _new_pattern = self._encode(ids)[1]
                else:
                    _new_pattern = torch.as_tensor(pattern).reshape(-1)
                _hit = self.find_by_pattern(_new_pattern)
            except Exception as _exc:

                self.n_pattern_encode_errors = int(
                    getattr(self, "n_pattern_encode_errors", 0)) + 1
                if not getattr(self, "_pattern_encode_error_note", None):
                    self._pattern_encode_error_note = (
                        f"{type(_exc).__name__}: {_exc}")
                _new_pattern, _hit = None, None
            if _hit is not None:
                self.n_pattern_hits = int(getattr(self, "n_pattern_hits", 0)) + 1

                return int(_hit)
        if self.file is not None:
            self._file_ids[index] = [int(x) for x in ids]
            _bmp = self._pack_pattern_bitmap(_new_pattern)
            if _bmp is not None:
                self._file_pending[index] = _bmp
        else:
            self.payloads[index] = array("i", (int(x) for x in ids))
        self._slot_ver[index] = int(self._slot_ver.get(index, 0)) + 1
        self._touch_cache(index)
        if _new_pattern is not None:
            self.index_pattern(_new_pattern, int(index))
        if self.segment_store is not None:
            self.segment_store.append(index, ids)

        self._bump()
        return index

    def _next_free_slot(self) -> int:
        used = set(int(s) for s in self.payloads)
        if self.file is not None:
            used |= set(int(s) for s in self.file.slots())
            used |= set(int(s) for s in self._file_ids)
        if self.segment_store is not None:
            used |= set(int(s) for s in self.segment_store.slots)
        i = 0
        while i in used:
            i += 1
        return i

    def _require_encoder(self) -> Any:
        if self.segment_encoder is None:
            raise RuntimeError("缺少段编码器：需先注入 token_ids -> (base, feat) 的可调用对象")
        return self.segment_encoder

    def _encode(self, token_ids: Sequence[int]) -> Tuple[Tensor, Tensor]:
        enc = self._require_encoder()
        ids = [int(x) for x in token_ids]
        out = enc(ids)
        if not isinstance(out, (tuple, list)) or len(out) != 2:
            raise RuntimeError(f"segment_encoder 应返回 (base, pattern)，收到 {type(out).__name__}")
        base = torch.as_tensor(out[0], dtype=torch.float32).reshape(-1)
        pat = torch.as_tensor(out[1], dtype=torch.float32).reshape(-1)
        n = len(ids)
        if base.numel() == 0:
            raise ValueError(f"encoder 对 {n} 个 token 返回了空的段级表示")
        if pat.numel() == 0:
            raise ValueError(f"encoder 对 {n} 个 token 返回了空模式")
        if self.pattern_len and int(pat.numel()) != int(self.pattern_len):
            raise ValueError(f"模式长度 {int(pat.numel())} 与已记录的 {int(self.pattern_len)} 不一致")
        dev = self.node.mem.device
        if int(self.embedding_dim) and base.numel() != int(self.embedding_dim):
            raise ValueError(f"段级表示维度 {base.numel()} 与已记录的 {int(self.embedding_dim)} 不一致")
        self.pattern_len = int(pat.numel())
        self.embedding_dim = int(base.numel())
        return base.to(dev), pat.to(dev)

    def segment_tokens(self, slot: int) -> Optional[Tensor]:
        ids = self.segment_token_ids(slot)
        if not ids:
            return None
        k = self.token_dims
        if k <= 0:
            raise RuntimeError("token_dims 未确定：注入编码器时须先建立段长与模式维度的整除关系")
        pat = self._cached_pattern(slot)
        if pat is None:
            pat = self._encode(ids)[1]

        n_tok = len(ids)
        need = n_tok * int(k)
        if int(pat.numel()) < need:
            raise ValueError(
                f"从模式取 {n_tok} 个 token 的特征需要 {need} 维，"
                f"模式只有 {int(pat.numel())} 维"
            )
        return pat[:need].reshape(n_tok, int(k))

    def _pack_pattern_bitmap(self, pat: Optional[Tensor]) -> Optional[bytes]:
        if pat is None:
            return None
        bits = torch.as_tensor(pat).reshape(-1) != 0
        packed = _pack_bits(bits)
        raw = bytes(packed.contiguous().numpy().astype(np.uint8).tobytes())
        if len(raw) < 1024:
            raw = raw + b"\x00" * (1024 - len(raw))
        return raw[:1024]

    def _cached_pattern(self, slot: int) -> Optional[Tensor]:
        if self.file is not None:
            raw = self.file.bitmap(int(slot))
            if raw is not None:
                n = int(self.pattern_len) or 8192
                packed = torch.from_numpy(np.asarray(raw, dtype=np.uint8).copy())
                bits = _unpack_bits(packed, n)
                scale = float(bits.sum().item())
                return bits / (scale if scale >= 1.0 else 1.0)
        hit = self._pattern_cache.get(int(slot))
        if hit is None:
            return None
        ver, packed, scale, n, dense = hit
        if int(self._slot_ver.get(int(slot), 0)) != ver:
            return None
        if dense is not None:
            return dense
        return _unpack_bits(packed, n) / scale

    def _remember_pattern(self, slot: int, ids: Sequence[int], pattern: Tensor) -> None:
        dense = pattern.detach().to("cpu").float().reshape(-1).clone()
        n = int(dense.numel())
        bits = dense != 0
        scale = float(bits.sum().item())
        if scale < 1.0:
            scale = 1.0
        packed = _pack_bits(bits)
        recon = _unpack_bits(packed, n) / scale
        keep = None if torch.equal(recon, dense) else dense
        self._pattern_cache[int(slot)] = (
            int(self._slot_ver.get(int(slot), 0)),
            packed,
            scale,
            n,
            keep,
        )
    @property
    def token_dims(self) -> int:
        return int(self._token_dims)

    def set_token_dims(self, k: int) -> None:
        if int(k) <= 0:
            raise ValueError("token_dims 必须是正整数")
        self._token_dims = int(k)

    def bind_encoder(self, encoder_fn: Any, shape: Any) -> None:
        seg, k, n = int(shape[0]), int(shape[1]), int(shape[2])
        if seg <= 0 or k <= 0 or n <= 0:
            raise ValueError(f"编码器报出的分区含非正数：{tuple(shape)}")
        if n != seg * k:
            raise ValueError(
                f"编码器分区不自洽：num_neurons={n} ≠ segment_tokens={seg} × token_dims={k}"
            )
        if self.pattern_len and n != self.pattern_len:
            raise ValueError(f"编码器模式维度 {n} 与存档记录的 pattern_len={self.pattern_len} 不符")
        if self._token_dims and k != self._token_dims:
            wrote_seg = self.pattern_len // self._token_dims if self._token_dims else 0
            raise ValueError(f"编码器每 token 维度 {k} 与存档记录的 token_dims={self._token_dims} 不符")
        self.segment_encoder = encoder_fn

        self.pattern_len = int(n)
        self.set_token_dims(k)

        self._rebuild_pattern_cache()

    def _rebuild_pattern_cache(self) -> int:
        if int(self._token_dims) <= 0:
            return 0
        n = 0
        for slot in self.stored_segments():
            ids = self.segment_token_ids(slot) or []
            if not ids or self._cached_pattern(slot) is not None:
                continue

            pat = self.prepare(self._encode(ids)[1])
            self._remember_pattern(slot, ids, pat)
            n += 1
        return n

    def token_lens(self, token_ids: Sequence[int]) -> Tensor:
        _fn = getattr(self, "token_len_fn", None)
        if _fn is None:
            return torch.full((len(token_ids),), 8.0, dtype=torch.float32)
        return torch.tensor([float(x) for x in _fn([int(t) for t in token_ids])],
                            dtype=torch.float32)

    def _recall_pool_aux(self, pool: Sequence[int], scope: Optional[Sequence[int]]
                         ) -> Tuple[List[Any], List[Any], set]:
        _key = (int(self._state_version), tuple(int(s) for s in pool))
        _hit = self._c_pool_aux.get(_key)
        if _hit is not None:
            return _hit
        _cand_valid = [self.valid_len(self.segment_token_ids(int(s))) for s in pool]
        _cand_ids = [self.segment_token_ids(int(s)) or [] for s in pool]
        _bad_ids = set(int(x) for x in (getattr(self, "special_token_ids", None) or ()))
        out = (_cand_valid, _cand_ids, _bad_ids)
        if len(self._c_pool_aux) >= 8:
            self._c_pool_aux.clear()
        self._c_pool_aux[_key] = out
        return out

    def _recall_pool_ranked(
        self,
        query_tokens: Tensor,
        pool: Sequence[int],
        *,
        matcher: SeqMatcher,
        k: int,
        freq_q: Optional[Tensor] = None,
        query_valid: Optional[int] = None,
        scope: Optional[Sequence[int]] = None,
    ) -> Optional[Tuple[List[Tuple[int, float, int]], List[int], Tensor, Tensor]]:
        _cand_valid, _cand_ids, _bad_ids = self._recall_pool_aux(pool, scope)
        _tops: List[Tuple[int, float, int]] = []
        _slots: List[int] = []
        _scores: List[Tensor] = []
        _lens: List[Tensor] = []
        _step = max(1, int(_POOL_CHUNK))
        for _i in range(0, len(pool), _step):
            _chunk = pool[_i:_i + _step]
            _n_chunk = len(_chunk)
            _items = [self._pool_item(int(s)) for s in _chunk]
            _got = matcher._score_pool(
                query_tokens,
                _items,
                freq_q=freq_q,
                query_valid=query_valid,
                cand_valid=_cand_valid[_i:_i + _n_chunk],
                cand_ids=_cand_ids[_i:_i + _n_chunk],
                bad_ids=_bad_ids,
            )
            del _items
            self._c_pool_item.clear()
            matcher._cand_cache.clear()
            matcher._sp_cache.clear()
            if _got is None:
                return None
            _csl, _csc, _cln = _got
            _slots.extend(int(x) for x in _csl)
            _scores.append(_csc)
            _lens.append(_cln)
            _kk = min(max(0, int(k)), int(_csc.numel()))
            if _kk > 0:
                _vals, _idx = torch.topk(_csc, _kk)
                _tops.extend((int(_csl[int(_b)]), float(_vals[_j]), int(_cln[int(_b)]))
                             for _j, _b in enumerate(_idx.tolist()))
        _keep = min(max(0, int(k)), len(_tops))
        if len(pool) <= _step:
            return (_tops[:_keep], _slots, torch.cat(_scores), torch.cat(_lens))
        _tops.sort(key=lambda _a: -_a[1])
        return (_tops[:_keep], _slots, torch.cat(_scores), torch.cat(_lens))

    @property
    def _segment_token_ids(self) -> Dict[int, List[int]]:
        return {int(s): (self.segment_token_ids(s) or []) for s in self.stored_slots()}

    def segment_token_ids(self, slot: int) -> Optional[List[int]]:
        s = int(slot)
        pend = self._file_ids.get(s)
        if pend is not None:
            return [int(x) for x in pend]
        if self.file is not None:
            got = self.file.segment_ids(s)
            if got is None:
                return None
            return [int(x) for x in got]
        p = self.payloads.get(s)
        if p is None and self.segment_store is not None:
            got = self.segment_store.read(s)
            if got is not None:
                self.payloads[s] = array("i", (int(x) for x in got))
                self._slot_ver[s] = int(self._slot_ver.get(s, 0)) + 1
                self._touch_cache(s)
                p = self.payloads[s]
        if p is None:
            return None
        if isinstance(p, torch.Tensor):
            return [int(x) for x in p.reshape(-1).tolist()]
        return [int(x) for x in p]

    def segment_pattern(self, slot: int) -> Optional[Tensor]:
        ids = self.segment_token_ids(slot)
        if not ids:
            return None
        pat = self._cached_pattern(int(slot))
        if pat is None:

            pat = self.prepare(self._encode(ids)[1])
        return pat

    @staticmethod
    def pattern_key(pat) -> int:
        import hashlib

        t = pat.detach().to("cpu", torch.float32).contiguous()
        return int.from_bytes(hashlib.md5(t.numpy().tobytes()).digest()[:8], "little")

    def find_by_pattern(self, pat) -> Optional[int]:
        key = self.pattern_key(pat)
        if self.file is not None:
            return self.file.lookup(key)
        slot = self._pattern_index.get(key)
        if slot is None:
            return None
        if slot not in self.payloads:

            del self._pattern_index[key]
            return None
        return int(slot)

    def index_pattern(self, pat, slot: int) -> None:
        if self.file is not None:
            self.file.add_key(self.pattern_key(pat), int(slot))
            return
        self._pattern_index[self.pattern_key(pat)] = int(slot)

    def pattern_index_size(self) -> int:
        return len(self._pattern_index)

    def stored_slots(self) -> List[int]:
        _ver, _cached = self._c_slots
        if int(_ver) == int(self._state_version):
            return _cached
        if self.file is not None:
            out = sorted(int(s) for s in self.file.slots())
        elif self.segment_store is not None:
            out = [int(s) for s in self.segment_store.slots]
        else:
            out = sorted(int(s) for s in self.payloads)
        self._c_slots = (int(self._state_version), out)
        return out

    def has_slot(self, slot: int) -> bool:
        s = int(slot)
        if s in self._file_ids:
            return True
        if self.file is not None:
            return bool(self.file.has(s))
        if s in self.payloads:
            return True
        return self.segment_store is not None and bool(self.segment_store.has(s))

    @property
    def num_segments(self) -> int:
        return len(self.stored_slots())

    def _touch_cache(self, slot: int) -> None:
        s = int(slot)
        if s in self._cache_order:
            self._cache_order.remove(s)
        self._cache_order.append(s)
        cap = int(self.max_cached_segments)
        while cap > 0 and len(self._cache_order) > cap:
            old = self._cache_order.pop(0)
            self.payloads.pop(old, None)

    def valid_len(self, token_ids: Optional[Sequence[int]]) -> Optional[int]:
        if token_ids is None or self.pad_token_id is None:
            return None
        ids = [int(x) for x in token_ids]
        pid = int(self.pad_token_id)
        n = len(ids)
        while n > 0 and ids[n - 1] == pid:
            n -= 1
        return int(n)

    def _pool_item(self, slot: int) -> Tuple[int, Tensor, Tensor]:
        _ver, _hit = self._c_pool_item.get(int(slot), (-1, None))[:2]
        if _hit is not None and int(_ver) == int(self._state_version):
            return _hit
        ids = self.segment_token_ids(slot) or []
        item = (int(slot), self.segment_tokens(slot), self.token_lens(ids))
        self._c_pool_item[int(slot)] = (int(self._state_version), item)
        return item

    def observe_tokens(self, token_ids: Sequence[int]) -> None:
        for t in token_ids:
            k = int(t)
            self.token_counts[k] = self.token_counts.get(k, 0) + 1
        self.token_total += len(token_ids)

        self._bump()

    def stored_segments(self) -> List[int]:
        _ver, _cached = self._c_segments
        if int(_ver) == int(self._state_version):
            return _cached
        out = [s for s in self.stored_slots() if self.segment_token_ids(s)]
        self._c_segments = (int(self._state_version), out)
        return out

    @torch.no_grad()
    def ref_recall(
        self,
        query_tokens: Tensor,
        *,
        matcher: SeqMatcher,
        query_freq: Optional[Tensor] = None,
        scope: Optional[Sequence[int]] = None,
        steps: int = 1,
        gate_keep: float = 1.0,
        query_valid: Optional[int] = None,
        ref_anchor_k: int = 1,
        force_anchor: Optional[Tuple[int, float, int]] = None,
        query_slot_hint: Optional[int] = None,
        slot_upper_bound: Optional[int] = None,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "recalled": [],
            "anchors": [],
            "carriers": [],
            "used_carriers": [],
            "scores": {},
            "n_no_edge": 0,
            "n_miss": 0,
        }

        _traversed: list = []
        pool = self.stored_segments()
        if scope is not None:
            allow = {int(s) for s in scope}
            pool = [s for s in pool if s in allow]
        if not pool:
            out["n_miss"] = 1
            return out

        matcher.score_device = None

        _scan: Optional[Tuple[
            List[Tuple[int, float, int]], List[int], Tensor, Tensor]] = None

        if query_slot_hint is not None:

            _h = int(query_slot_hint)
            _hl = self.valid_len(self.segment_token_ids(_h)) if (
                self.segment_token_ids(_h) is not None) else 0
            _anchors = [(_h, 1.0, int(_hl))]
        elif force_anchor is not None:
            _anchors = [(int(force_anchor[0]), float(force_anchor[1]), int(force_anchor[2]))]
        else:
            _scan = self._recall_pool_ranked(
                query_tokens, pool, matcher=matcher,
                k=(int(ref_anchor_k) if int(ref_anchor_k) > 1 else 1),
                freq_q=query_freq, query_valid=query_valid, scope=scope)
            _anchors = [] if _scan is None else [
                a for a in _scan[0] if float(a[1]) > 0.0]
        if not _anchors:
            out["n_miss"] = 1
            return out
        ref_slot, ref_score = int(_anchors[0][0]), float(_anchors[0][1])
        out["anchors"] = [(int(a[0]), float(a[1])) for a in _anchors]
        out["n_anchors_k"] = len(_anchors)

        _cw: Dict[int, float] = {}

        _lf_theta = getattr(self.lateral_cfg, "recall_fire_theta", None)

        _slot_gain = None
        _kz = float(getattr(self.lateral_cfg, "pfc_z_coef", 0.0))
        if _kz > 0.0 and len(_anchors) > 0:
            if _scan is None:
                _scan = self._recall_pool_ranked(
                    query_tokens, pool, matcher=matcher, k=1,
                    freq_q=query_freq, query_valid=query_valid, scope=scope)
            if _scan is not None:
                _sl, _sc, _ln = _scan[1], _scan[2], _scan[3]
                _sl = [int(x) for x in _sl]
                _sc = torch.as_tensor(_sc, dtype=torch.float32)
                _ln = torch.as_tensor(_ln, dtype=torch.float32)
                _keep = (_ln > 0)
                _vals = torch.where(_keep, _sc, torch.zeros_like(_sc))
                if bool(_keep.any()):
                    _mu = float(_vals[_keep].mean())
                    _sd = float(_vals[_keep].std(unbiased=False))
                    if _sd > 0.0:
                        _z = (_vals - _mu) / _sd
                        _g = torch.exp(_kz * _z).clamp(0.2, 5.0)
                        _slot_gain = torch.ones(
                            int(self.num_slots), dtype=_g.dtype, device=_g.device)
                        for _b, _s_ix in enumerate(_sl):
                            if 0 <= _s_ix < int(self.num_slots):
                                _slot_gain[_s_ix] = _g[_b]
        _res_lat = self.recall_lateral(
            seed_slots=[int(_a[0]) for _a in _anchors],
            slot_gain=_slot_gain,
            steps=int(steps),
            fire_theta=(None if _lf_theta is None else float(_lf_theta)),

            slot_upper_bound=slot_upper_bound,
        )
        _lat_scores = list(getattr(_res_lat, "scores", None) or [])
        for _i2, _sl in enumerate(getattr(_res_lat, "order", None) or []):
            _k2 = int(_sl)
            _w = float(_lat_scores[_i2]) if _i2 < len(_lat_scores) else 1.0
            if _k2 not in _cw or _w > _cw[_k2]:
                _cw[_k2] = _w
        carriers = sorted(_cw.items(), key=lambda kv: (-kv[1], kv[0]))

        for _c_slot, _c_w in carriers:
            _traversed.append([int(ref_slot), int(_c_slot), float(_c_w), "carrier"])

        if query_slot_hint is not None:
            _h = int(query_slot_hint)
            out["used_pattern_hint"] = True
            _set = {int(ref_slot)} | {int(_s) for _s, _w in carriers}
            _set.add(int(_h))
            out["recalled"] = sorted(_set)
            _uc = {int(_s) for _s, _w in carriers}
            _uc.add(int(_h))
            out["used_carriers"] = sorted(_uc)
            out["scores"] = {int(_s): float(_w) for _s, _w in carriers}
            out["scores"][int(ref_slot)] = float(ref_score)
            out["scores"][int(_h)] = 1.0

            out["audit"] = {
                "anchors": list(out["anchors"]),
                "used_carriers": list(out["used_carriers"]),
                "traversed_edges": list(_traversed),
                "n_recalled": len(_set),
                "n_carriers_total": len(carriers),
                "n_anchor": 1,
                "n_a1_new": len(carriers),
            }
            return out

        if not carriers:

            out["n_no_edge"] = 1
            out["recalled"] = [int(ref_slot)]
            out["used_carriers"] = [int(ref_slot)]
            out["scores"] = {int(ref_slot): float(ref_score)}
            return out

        scored: Dict[int, float] = {}

        for ref1_slot, _w in carriers:
            toks = self.segment_tokens(ref1_slot)
            if toks is None:
                continue
            out["carriers"].append(int(ref1_slot))

        scored[int(ref_slot)] = max(
            scored.get(int(ref_slot), float("-inf")), float(ref_score)
        )

        gate = float("-inf")
        if carriers:
            _ws = torch.tensor([float(w) for _s, w in carriers], dtype=torch.float32)

            _keep = min(max(float(gate_keep), 0.0), 1.0)
            if _keep < 1.0:
                gate = float(torch.quantile(_ws, 1.0 - _keep))
            for _c_slot, _c_w in carriers:
                if float(_c_w) >= gate:
                    scored[int(_c_slot)] = max(
                        scored.get(int(_c_slot), float("-inf")), float(_c_w)
                    )

        _rec = {int(k) for k in scored.keys()}
        _rec |= {int(_c) for _c, _w in carriers}
        _rec.add(int(ref_slot))
        out["recalled"] = sorted(_rec)
        out["n_FIRED_added"] = len(_rec) - len(scored)
        out["used_carriers"] = sorted({int(c) for c in out["carriers"]})

        out["scores"] = {int(k): float(v) for k, v in scored.items()}

        a1_counts: Dict[int, float] = {}
        for _c_slot, _c_w in carriers:
            a1_counts[int(_c_slot)] = max(a1_counts.get(int(_c_slot), float("-inf")),
                                          float(_c_w))
        a1_kept = {k for k, v in a1_counts.items() if v >= gate}
        a1_gated_out = set(a1_counts) - a1_kept
        anchor_slots = {int(ref_slot)}
        anchor_new = anchor_slots - a1_kept
        out["audit"] = {

            "anchors": list(out["anchors"]),

            "used_carriers": list(out["used_carriers"]),

            "traversed_edges": list(_traversed),

            "anchor_slot": int(ref_slot),
            "anchor_score": float(ref_score),
            "n_anchor": len(anchor_new),
            "anchor_slots": sorted(anchor_new),

            "n_a1_new": len(a1_kept),

            "n_carriers_total": len(a1_counts),

            "n_a1_gated_out": len(a1_gated_out),

            "gate": float(gate),
            "n_recalled": len(out["recalled"]),
        }
        return out

    @torch.no_grad()
    def remember_slot_patterns(
        self,
        patterns: Sequence[Tensor],
        indices: Sequence[int],
    ) -> int:
        idx = [int(i) for i in indices]
        if not idx:
            return 0
        if len(idx) != len(patterns):
            raise ValueError(f"patterns 与 indices 长度不一致：{len(patterns)} vs {len(idx)}")
        if max(idx) >= self.num_slots:
            self.ensure_slots(max(idx) + 1)
        pats = [self.prepare(p) for p in patterns]
        for _i, _sl in enumerate(idx):
            _ids = self.segment_token_ids(_sl) or []
            if _ids:
                self._remember_pattern(_sl, _ids, pats[_i])
        self.node.reset()

        self._bump()
        return len(idx)

    @torch.no_grad()
    def recall_lateral(
        self,
        *,
        seed_slots: Sequence[int],
        steps: int = 3,
        fire_ratio: float = 1.0,
        fire_theta: Optional[float] = None,
        max_fired: int = 16384,
        slot_upper_bound: Optional[int] = None,
        slot_gain: Optional[Tensor] = None,
    ) -> RecallResult:
        if int(steps) <= 0:
            raise ValueError("steps 必须是正整数")
        if self.node.lateral is None or self.node.lateral.num_edges == 0:
            return RecallResult(order=[], scores=[])
        if not seed_slots:
            raise ValueError("recall_lateral 需要 seed_slots")

        self.node.lateral.ensure_slots(self.num_slots)

        self.node.mem = torch.zeros_like(self.node.mem)
        num = self.node.output_size
        dev = self.node.mem.device
        seen = torch.zeros(num, dtype=self.node.mem.dtype, device=dev)
        out = torch.zeros_like(seen)
        s_prev = torch.zeros_like(seen)

        idx = torch.tensor([int(i) for i in seed_slots], dtype=torch.long, device=dev)
        idx = idx[(idx >= 0) & (idx < num)]
        if idx.numel() == 0:
            raise ValueError("seed_slots 全部越界")
        s_prev[idx] = 1.0

        _gain = (
            torch.ones_like(seen) if slot_gain is None
            else slot_gain.to(seen.device, dtype=seen.dtype).clamp(0.2, 5.0)
        )
        written = self.written_mask()

        if slot_upper_bound is not None:
            _ub = int(slot_upper_bound)
            _ar = torch.arange(int(written.numel()), device=written.device)

            _ubmask = (_ar < _ub)
            written = written & _ubmask
        else:
            _ubmask = None
        peak = torch.zeros_like(seen)

        _thr_t: Optional[Tensor] = None
        _au = getattr(self, "attn_u", None)
        if _au is not None and int(_au.numel()) > 0:
            _n = int(seen.numel())
            _u = torch.zeros(_n, dtype=torch.float32, device=seen.device)
            _m = min(_n, int(_au.numel()))
            _u[:_m] = _au[:_m].to(seen.device)
            _sel = _u[written] if bool(written.any()) else _u
            _pos = _sel[_sel > 0]
            _ref = float(_pos.median()) if int(_pos.numel()) else 0.0
            if _ref > 0.0:
                _u = torch.where(_u > 0, _u, torch.full_like(_u, _ref))
                _thr_t = (_ref / _u).clamp(0.1, 10.0)

        _mb = float(self.lateral_cfg.mem_beta)
        _dtype = self.node.mem.dtype
        lat = self.node.lateral
        mem = torch.zeros(num, dtype=_dtype, device=dev)
        peak = torch.zeros(num, dtype=_dtype, device=dev)
        seen = torch.zeros(num, dtype=_dtype, device=dev)
        out = torch.zeros(num, dtype=_dtype, device=dev)
        s_prev = torch.zeros(num, dtype=_dtype, device=dev)
        s_prev[idx] = 1.0
        touched = torch.unique(idx)
        frontier = touched
        n_seen = 0

        for t in range(int(steps)):
            if slot_gain is None:
                di, dv = lat.spread(frontier, s_prev)
            else:
                _ai, _av = lat.spread(frontier, s_prev, positive_only=True, gain=_gain)
                _bi, _bv = lat.spread(frontier, s_prev, negative_only=True)
                di, dv = _sparse_add(_ai, _av, _bi, _bv)

            if t > 0 and (int(di.numel()) == 0 or float(dv.abs().sum()) == 0.0):
                break

            if int(di.numel()):
                mem.index_copy_(0, touched, mem[touched] * _mb)
                mem.index_add_(0, di, dv)
                touched = torch.unique(torch.cat([touched, di]))

            peak.index_copy_(
                0, touched, torch.maximum(peak[touched], mem[touched].abs())
            )
            if _thr_t is not None:
                _tv = _thr_t[touched] * float(fire_theta)
            else:
                _thr = self._threshold(mem, written, fire_ratio=float(fire_ratio),
                                       fire_theta=None)
                _tv = torch.full((int(touched.numel()),), float(_thr),
                                 dtype=_dtype, device=dev)
            if slot_gain is not None:
                _tv = _tv / _gain[touched]

            spk = (mem[touched] > _tv).to(_dtype) * _tv
            if int(t) == 0:
                spk = torch.where(torch.isin(touched, idx), torch.ones_like(spk), spk)
            spk = spk * (1.0 - seen[touched])

            if _ubmask is not None:
                spk = spk * _ubmask[touched].to(spk.dtype)
            seen.index_copy_(
                0, touched, torch.clamp(seen[touched] + (spk > 0).to(_dtype), 0.0, 1.0)
            )
            out.index_copy_(0, touched, torch.clamp(out[touched] + spk, 0.0, 1.0))
            mem.index_copy_(0, touched, mem[touched] * (1.0 - spk))

            _nz = spk > 0
            frontier = touched[_nz]
            s_prev.index_copy_(0, touched, spk)
            if int(frontier.numel()) == 0:
                break
            n_seen += int(frontier.numel())
            if n_seen >= int(max_fired):
                break

        self.node.mem = mem
        order = [int(i) for i in out.nonzero().flatten().tolist()]
        scores = [float(peak[i]) for i in order]
        return RecallResult(
            order=order,
            scores=scores,
            spikes=out,
            membrane=peak,
        )

    def _threshold(self, mem: Tensor, written: Tensor, *, fire_ratio: float,
                   fire_theta: Optional[float] = None) -> float:
        if fire_theta is not None:
            return float(fire_theta)
        if not 0.0 < float(fire_ratio) <= 1.0:
            raise ValueError(f"fire_ratio 必须在 (0, 1]，收到 {fire_ratio}")
        scope = mem[written] if bool(written.any()) else mem
        scope = scope.detach().float()
        nz = scope[scope != 0.0]
        x = (nz if nz.numel() else scope).abs()
        if x.numel() == 0:
            raise ValueError("阈值统计作用域为空：没有任何已写入的槽")
        q = 1.0 - float(fire_ratio)
        if x.numel() == 1:
            return float(x.min())
        return float(torch.quantile(x, q))

    def __repr__(self) -> str:
        return (
            f"SynapticaMemory(input={self.input_size}, slots={self.num_slots}, "
            f"written={self.num_segments}, "
            f"lateral={self.node.lateral.num_edges if self.node.lateral else 0})"
        )
