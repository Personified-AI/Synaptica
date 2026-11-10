from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

Span = Tuple[int, int]
EdgeRow = List[Tuple[int, float]]


@dataclass
class AttentionGraphConfig:
    pos_debias: float = 1.0
    out_degree: int = 3
    max_in_degree: int = 4


class AttentionGraph:

    def __init__(self, rows: Sequence[EdgeRow], spans: Sequence[Span],
                 *, meta: Optional[Dict[str, Any]] = None,
                 col_mass: Optional[Tensor] = None,
                 token_row_max: Optional[Tensor] = None,
                 row_negent: Optional[Tensor] = None) -> None:
        if len(rows) != len(spans):
            raise ValueError("rows 数量必须等于 spans 数量")
        self.rows: List[EdgeRow] = [list(r) for r in rows]
        self.spans: List[Span] = [(int(a), int(b)) for a, b in spans]
        self.meta: Dict[str, Any] = dict(meta or {})
        self.col_mass: Optional[Tensor] = col_mass
        self.token_row_max: Optional[Tensor] = token_row_max
        self.row_negent: Optional[Tensor] = row_negent

    @property
    def num_slots(self) -> int:
        return len(self.spans)

    @property
    def num_edges(self) -> int:
        return sum(len(r) for r in self.rows)

    def edges(self) -> List[Tuple[int, int, float]]:
        out: List[Tuple[int, int, float]] = []
        for dest, row in enumerate(self.rows):
            for src, w in row:
                out.append((dest, int(src), float(w)))
        return out

    def row_max(self) -> Tensor:
        vals = []
        for row in self.rows:
            vals.append(max((float(w) for _s, w in row), default=0.0))
        return torch.tensor(vals, dtype=torch.float32)

    def __repr__(self) -> str:
        return (f"AttentionGraph(slots={self.num_slots}, edges={self.num_edges})")


def validate_span_intervals(spans: Sequence[Span], T: int) -> List[Span]:
    spans = [(int(a), int(b)) for a, b in spans]
    if len(spans) == 0:
        raise ValueError("spans 不能为空")
    for (a, b) in spans:
        if not (0 <= a < b <= T):
            raise ValueError(f"非法 span ({a},{b})，T={T}")
    ordered = sorted(spans)
    for (a1, b1), (a2, b2) in zip(ordered, ordered[1:]):
        if a2 < b1:
            raise ValueError(f"spans 重叠：({a1},{b1}) 与 ({a2},{b2})")
    return spans


def validate_spans(spans: Sequence[Span], T: int) -> List[Span]:
    spans = validate_span_intervals(spans, T)
    ordered = sorted(spans)
    if ordered[0][0] != 0:
        raise ValueError(f"spans 未覆盖 [0,{ordered[0][0]})，T={T}")
    if ordered[-1][1] != T:
        raise ValueError(f"spans 未覆盖 [{ordered[-1][1]},{T})，T={T}")
    for (a1, b1), (a2, b2) in zip(ordered, ordered[1:]):
        if a2 > b1:
            raise ValueError(f"spans 在 [{b1},{a2}) 有缺口，T={T}")
    return spans


class SegmentAggregator:

    def __init__(self, spans: Sequence[Span], cfg: Optional[AttentionGraphConfig] = None,
                 *, num_rows: int) -> None:
        self.cfg = cfg or AttentionGraphConfig()
        self.spans = validate_spans(spans, int(num_rows))
        self.T = int(num_rows)
        self.M = len(self.spans)
        self.W = torch.zeros(self.M, self.M, dtype=torch.float32)
        self.col_mass = torch.zeros(self.T, dtype=torch.float32)
        self.token_row_max = torch.zeros(self.T, dtype=torch.float32)
        self.row_negent = torch.zeros(self.T, dtype=torch.float32)
        seg = torch.full((self.T,), -1, dtype=torch.long)
        for j, (c, d) in enumerate(self.spans):
            seg[c:d] = j
        self._col_seg = seg
        self._sum_hi: Optional[Tensor] = None
        self._sum_lo: Optional[Tensor] = None
        self._sum_hi_gpu: Optional[Tensor] = None
        self._sum_lo_gpu: Optional[Tensor] = None
        self._rows_seen = 0

    def _build_sum_idx(self) -> None:
        self._sum_hi = torch.tensor([d - 1 for _c, d in self.spans], dtype=torch.long)
        self._sum_lo = torch.tensor([c - 1 if c > 0 else -1 for c, _d in self.spans],
                                    dtype=torch.long)

    def _col_sum(self, rows: Tensor) -> Tensor:
        if self._sum_hi is None:
            self._build_sum_idx()
        cs = torch.cumsum(rows, dim=1)
        out = cs[:, self._sum_hi]
        has_lo = self._sum_lo >= 0
        if bool(has_lo.any()):
            lo_vals = cs[:, self._sum_lo.clamp(min=0)]
            out = out - torch.where(has_lo.view(1, -1), lo_vals,
                                    torch.zeros_like(lo_vals))
        return out

    def _col_sum_gpu(self, rows: Tensor) -> Tensor:
        if self._sum_hi is None:
            self._build_sum_idx()
        dev = rows.device
        if self._sum_hi_gpu is None or self._sum_hi_gpu.device != dev:
            self._sum_hi_gpu = self._sum_hi.to(dev)
            self._sum_lo_gpu = self._sum_lo.to(dev)
        cs = torch.cumsum(rows, dim=1)
        out = cs[:, self._sum_hi_gpu]
        has_lo = self._sum_lo >= 0
        if bool(has_lo.any()):
            lo_vals = cs[:, self._sum_lo_gpu.clamp(min=0)]
            out = out - torch.where(has_lo.view(1, -1).to(dev),
                                    lo_vals, torch.zeros_like(lo_vals))
        return out


    def add_rows(self, rows: Tensor, offset: int = 0) -> None:
        rows_g = rows.detach().float()
        n = int(rows_g.shape[0])
        if n == 0:
            return
        rseg = self._col_seg[offset:offset + n]
        keep = rseg >= 0
        if not bool(keep.any()):
            self._rows_seen += n
            return
        rseg, rows_g = rseg[keep], rows_g[keep]
        _rpos = torch.nonzero(keep, as_tuple=False).flatten() + int(offset)
        self.col_mass += rows_g.sum(dim=0).cpu()
        self.token_row_max.index_copy_(0, _rpos, rows_g.max(dim=1).values.cpu())
        _rsum = rows_g.sum(dim=1, keepdim=True)
        _rsum_safe = torch.where(_rsum > 0, _rsum, torch.ones_like(_rsum))
        _p = rows_g / _rsum_safe
        self.row_negent.index_copy_(0, _rpos, torch.xlogy(_p, _p).sum(dim=1).cpu())
        colred = self._col_sum_gpu(rows_g).float().cpu()
        self.W.index_add_(0, rseg, colred)
        self._rows_seen += n


    def finalize(self, *, meta: Optional[Dict[str, Any]] = None) -> AttentionGraph:
        W = self.W.clone()
        spans = list(self.spans)
        T = self.T
        col_mass = self.col_mass[0:T].clone()
        token_row_max = self.token_row_max[0:T].clone()
        row_negent = self.row_negent[0:T].clone()
        return graph_from_W(W, spans, self.cfg, meta=meta, T=T, col_mass=col_mass,
                            token_row_max=token_row_max, row_negent=row_negent)


def apply_baseline_and_causal(W: Tensor, spans: Sequence[Span],
                              *, T: Optional[int] = None) -> Tensor:
    spans = validate_spans(spans, int(T) if T is not None else max(b for _a, b in spans))
    W = W.float().clone()
    starts = torch.tensor([a for a, _b in spans], dtype=torch.float32)
    ends = torch.tensor([b for _a, b in spans], dtype=torch.float32)
    W = W.masked_fill(starts.view(1, -1) >= ends.view(-1, 1), 0.0)
    W.fill_diagonal_(0.0)
    return W


def graph_from_W(W: Tensor, spans: Sequence[Span],
                 cfg: Optional[AttentionGraphConfig] = None,
                 *, T: Optional[int] = None,
                 meta: Optional[Dict[str, Any]] = None,
                 col_mass: Optional[Tensor] = None,
                 token_row_max: Optional[Tensor] = None,
                 row_negent: Optional[Tensor] = None) -> AttentionGraph:
    cfg = cfg or AttentionGraphConfig()
    spans = validate_spans(spans, int(T) if T is not None else max(b for _a, b in spans))
    M = len(spans)
    W = apply_baseline_and_causal(W, spans, T=T)

    if float(cfg.pos_debias) > 0:
        M0 = int(W.shape[1])
        _ar = torch.arange(M0)
        _dist = (_ar.view(-1, 1) - _ar.view(1, -1)).clamp_min(0)
        _prof_db = torch.zeros(M0)
        for _d in range(1, M0):
            _idx = torch.arange(_d, M0)
            if _idx.numel():
                _prof_db[_d] = W[_idx, _idx - _d].mean()
        _pos = _prof_db[_dist]
        W = (W - float(cfg.pos_debias) * _pos).clamp_min(0.0)

    _W_raw = W.clone()
    _src_mat = W

    _k = int(cfg.out_degree)
    M0 = int(W.shape[1])
    _kin = int(cfg.max_in_degree)
    if M0 > 0:
        _col_top = torch.topk(_src_mat, min(_kin, M0), dim=0).indices
        _keepc = torch.zeros_like(_src_mat, dtype=torch.bool)
        _keepc.scatter_(0, _col_top, True)
        _src_mat = torch.where(_keepc, _src_mat, torch.zeros_like(_src_mat))
    if _k < M0:
        _top = torch.topk(_src_mat, _k, dim=1).indices
        _keep = torch.zeros_like(_src_mat, dtype=torch.bool)
        _keep.scatter_(1, _top, True)
        W = torch.where(_keep, W, torch.zeros_like(W))

    rows: List[EdgeRow] = [[] for _ in range(M)]
    _cand: List[Tuple[float, int, int]] = []
    _Wp = W
    for _i in range(M):
        for _j in range(M):
            if _i == _j:
                continue
            _w2 = float(_Wp[_i][_j])
            if _w2 <= 0.0:
                continue
            _cand.append((_w2, int(_j), int(_i)))
            _cand.append((_w2, int(_i), int(_j)))
    _oc: Dict[int, int] = {}
    _ic: Dict[int, int] = {}
    _sel: List[Tuple[int, int, float]] = []
    _ord = sorted(_cand, key=lambda t: (-t[0], t[1], t[2]))
    for _w3, _d3, _s3 in _ord:
        if _oc.get(_s3, 0) >= _k or _ic.get(_d3, 0) >= _kin:
            continue
        _oc[_s3] = _oc.get(_s3, 0) + 1
        _ic[_d3] = _ic.get(_d3, 0) + 1
        _sel.append((_d3, _s3, _w3))
    rows = [[] for _ in range(M)]
    for _d3, _s3, _w3 in _sel:
        rows[_d3].append((_s3, _w3))

    merged: List[EdgeRow] = []
    for row in rows:
        acc: Dict[int, float] = {}
        for src, w in row:
            acc[int(src)] = acc.get(int(src), 0.0) + float(w)
        merged.append(sorted(acc.items(), key=lambda kv: -kv[1]))
    if meta is None:
        meta = {}
    _raw_pairs: Dict[tuple, float] = {}
    for _att, _row in enumerate(merged):
        for _tgt, _w in _row:
            _raw_pairs[(int(_att), int(_tgt))] = float(
                max(_W_raw[int(_att), int(_tgt)].item(),
                    _W_raw[int(_tgt), int(_att)].item()))

    meta["raw_pairs"] = _raw_pairs
    return AttentionGraph(merged, spans, meta=meta, col_mass=col_mass,
                          token_row_max=token_row_max, row_negent=row_negent)
