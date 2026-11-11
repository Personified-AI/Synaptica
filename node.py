
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from ..ann.attention_graph import AttentionGraph

def dest_in_prefix_is_attended(dest: int, src: int, prefix_len: int) -> bool:
    return int(dest) < int(prefix_len) <= int(src)

@dataclass
class LateralConfig:

    alpha: float = 1.0
    max_weight: float = 2.0

    max_out_degree: int = 3
    max_new_edges_per_segment: int = 1

    lr: float = 10.0

    decay: float = 0.01

    neg_gain: float = 1.0

    attn_q: float = 0.25

    recall_fire_theta: Optional[float] = 0.7

    mem_beta: Optional[float] = 0.5

    pfc_z_coef: float = 0.40

    def __post_init__(self) -> None:
        if float(self.alpha) < 0.0:
            raise ValueError("alpha 不能为负")
        if int(self.max_out_degree) < 0:
            raise ValueError("max_out_degree 不能为负")
        if float(self.max_weight) <= 0.0:
            raise ValueError("max_weight 必须为正")
        if int(self.max_new_edges_per_segment) < 0:
            raise ValueError("max_new_edges_per_segment 不能为负")
        if not 0.0 < float(self.attn_q) <= 1.0:
            raise ValueError("attn_q 必须在 (0, 1]")
        if float(self.neg_gain) < 0.0:
            raise ValueError("neg_gain 不能为负")

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "LateralConfig":
        return cls(**json.loads(s))

class LateralState:

    def __init__(
        self,
        index: Tensor,
        values: Tensor,
        cfg: Optional[LateralConfig] = None,
        *,
        node: Optional[Any] = None,
    ) -> None:
        cfg = cfg or LateralConfig()
        if index.dim() != 2 or index.shape[0] != 2:
            raise ValueError("index 必须是 (2, E)")
        if values.dim() != 1 or values.numel() != index.shape[1]:
            raise ValueError("values 必须是 (E,)")
        self.cfg = cfg
        self.node = node
        self.device = index.device
        self.index = index.long().to(self.device)
        self.values = values.float().to(self.device)

        self.weak = torch.zeros(self.values.numel(), dtype=torch.bool, device=self.device)

        self.neg_index = torch.zeros(2, 0, dtype=torch.long, device=self.device)
        self.neg_values = torch.zeros(0, dtype=torch.float32, device=self.device)
        self._standalone_slots = (
            (int(self.index.max().item()) + 1) if self.index.numel() else 1
        )

    @property
    def num_slots(self) -> int:
        if self.node is not None:
            return int(self.node.output_size)
        return int(self._standalone_slots)

    @property
    def num_edges(self) -> int:
        return int(self.index.shape[1])

    @property
    def mean_degree(self) -> float:
        if self.num_edges == 0:
            return 0.0
        return float(self.num_edges) / max(1, self.num_slots)

    def _neg_merge_add(self, dest: Tensor, src: Tensor, vals: Tensor) -> int:
        _n = int(self.num_slots)
        _cap = float(self._neg_cap())
        _cur: Dict[int, float] = {}
        if self.neg_index.shape[1]:
            for _a, _b, _v in zip(self.neg_index[0].tolist(),
                                  self.neg_index[1].tolist(),
                                  self.neg_values.tolist()):
                _cur[int(_a) * _n + int(_b)] = float(_v)
        for _a, _b, _v in zip(dest.tolist(), src.tolist(), vals.tolist()):
            _k = int(_a) * _n + int(_b)
            _nv = _cur.get(_k, 0.0) + float(_v)
            _nv = min(-1e-12, max(-_cap, _nv))
            _cur[_k] = _nv
        _ks = list(_cur)
        self.neg_index = torch.tensor(
            [[_k // _n for _k in _ks], [_k % _n for _k in _ks]],
            dtype=torch.long, device=self.device,
        ) if _ks else torch.zeros(2, 0, dtype=torch.long, device=self.device)
        self.neg_values = torch.tensor(
            [_cur[_k] for _k in _ks], dtype=torch.float32, device=self.device
        ) if _ks else torch.zeros(0, dtype=torch.float32, device=self.device)
        return len(_ks)

    def _neg_cap(self) -> float:
        return float(self.cfg.max_weight)

    def fanin(self) -> Tensor:
        out = torch.zeros(self.num_slots, device=self.device)
        if self.num_edges:
            out.scatter_add_(0, self.index[0], self.values.abs())
        return out

    def count_edges(self, dest: Sequence[int], src: Sequence[int]) -> int:
        if self.num_edges == 0:
            return 0
        ds, ss = set(int(x) for x in dest), set(int(x) for x in src)
        n = 0
        for d, s in zip(self.index[0].tolist(), self.index[1].tolist()):
            if int(d) in ds and int(s) in ss:
                n += 1
        return n

    def edges(self) -> List[Tuple[int, int, float]]:
        return [
            (int(d), int(s), 1.0)
            for d, s in zip(self.index[0].tolist(), self.index[1].tolist())
        ]

    def propagate(self, spikes: Tensor, *, reverse: bool = False,
                  positive_only: bool = False, negative_only: bool = False) -> Tensor:
        spk = spikes.float().to(self.device).reshape(-1)
        if spk.numel() != self.num_slots:
            raise ValueError(f"spikes 长度 {spk.numel()} 与 num_slots {self.num_slots} 不一致")
        if self.num_edges == 0:
            return torch.zeros_like(spk)
        _neg_spk = None
        if self.neg_index.shape[1]:
            _nmat = torch.sparse_coo_tensor(
                self.neg_index, self.neg_values,
                (self.num_slots, self.num_slots), device=self.device,
            ).coalesce()
            _neg_spk = torch.sparse.mm(
                _nmat.transpose(0, 1) if reverse else _nmat, spk.unsqueeze(1)
            ).squeeze(1)
        if bool(negative_only):
            _wneg = self.values.clamp_max(0.0)
            _mneg = torch.sparse_coo_tensor(
                self.index, _wneg,
                (self.num_slots, self.num_slots), device=self.device,
            ).coalesce()
            _mneg = _mneg.transpose(0, 1) if reverse else _mneg
            _outn = torch.sparse.mm(_mneg, spk.unsqueeze(1)).squeeze(1)
            if _neg_spk is not None:
                _outn = _outn + _neg_spk
            return _outn
        if bool(positive_only):
            _w = self.values.clamp_min(0.0)
        else:
            _w = self.values
        mat = torch.sparse_coo_tensor(
            self.index,
            _w,
            (self.num_slots, self.num_slots),
            device=self.device,
        ).coalesce()

        m = mat.transpose(0, 1) if reverse else mat
        _out = torch.sparse.mm(m, spk.unsqueeze(1)).squeeze(1)
        if _neg_spk is not None and not bool(positive_only):
            _out = _out + _neg_spk
        return _out

    @torch.no_grad()
    def merge_edges(
        self,
        graph: AttentionGraph,
        *,
        slot_index: Sequence[int],
        alpha: Optional[float] = None,
        mode: str = "create_only",
        prefix_len: int = 0,
        max_new_per_segment: int = 0,
    ) -> Dict[str, int]:
        slot_index = [int(s) for s in slot_index]
        if len(slot_index) != int(graph.num_slots):
            raise ValueError(f"slot_index 长度 {len(slot_index)} 与图的段数 {graph.num_slots} 不一致")
        pl = int(prefix_len)
        if not 0 <= pl <= len(slot_index):
            raise ValueError(f"prefix_len={pl} 不在 [0, {len(slot_index)}]")
        a = float(self.cfg.alpha if alpha is None else alpha)

        dest: List[int] = []
        src: List[int] = []
        vals: List[float] = []
        for d, s, w in graph.edges():
            gd, gs = slot_index[int(d)], slot_index[int(s)]
            if gd == gs:
                continue
            dest.append(gd)
            src.append(gs)
            vals.append(float(w))

        if not dest:
            return {"created": 0, "skipped_existing": 0, "capped": 0}

        vals = [v * a for v in vals]

        capped = 0
        if int(max_new_per_segment) > 0 and pl > 0 and pl < len(slot_index):
            keep_d, keep_s, keep_v = [], [], []
            per_new: Dict[int, List[Tuple[float, int, int, float]]] = {}
            for d, s, v in zip(dest, src, vals):
                if dest_in_prefix_is_attended(d, s, pl):

                    per_new.setdefault(int(s), []).append((abs(float(v)), d, s, v))
                else:
                    keep_d.append(d)
                    keep_s.append(s)
                    keep_v.append(v)
            cap = int(max_new_per_segment)
            for _src, items in per_new.items():
                if len(items) <= cap:
                    keep_d.extend(x[1] for x in items)
                    keep_s.extend(x[2] for x in items)
                    keep_v.extend(x[3] for x in items)
                else:
                    items.sort(key=lambda kv: -kv[0])
                    capped += len(items) - cap
                    for x in items[:cap]:
                        keep_d.append(x[1])
                        keep_s.append(x[2])
                        keep_v.append(x[3])
            dest, src, vals = keep_d, keep_s, keep_v

        st = self.add_laterals(dest, src, vals, mode=mode)
        st["capped"] = int(capped)
        return st

    @torch.no_grad()
    def add_laterals(self, dest, src, values, *, mode: str = "add",
                     weak: bool = False) -> Dict[str, int]:
        if mode not in ("add", "create_only"):
            raise ValueError("mode 必须是 'add' | 'create_only'")
        _neg_n = 0
        dest = torch.as_tensor(dest, dtype=torch.long, device=self.device).reshape(-1)
        src = torch.as_tensor(src, dtype=torch.long, device=self.device).reshape(-1)
        vals = torch.as_tensor(values, dtype=torch.float32, device=self.device).reshape(
            -1
        )
        if dest.numel() == 0:
            return {"created": 0, "skipped_existing": 0}
        if not (dest.numel() == src.numel() == vals.numel()):
            raise ValueError("dest / src / values 长度必须一致")
        self.ensure_slots(max(int(dest.max()), int(src.max())) + 1)
        if mode == "add" and bool((vals < 0).any()):

            _m2 = vals < 0
            _neg_n = self._neg_merge_add(dest[_m2], src[_m2], vals[_m2])
            dest, src, vals = dest[~_m2], src[~_m2], vals[~_m2]
            if dest.numel() == 0:
                return {"created": 0, "skipped_existing": 0, "neg_total": _neg_n}
        skipped = 0
        if mode == "create_only":
            have = set(zip(self.index[0].tolist(), self.index[1].tolist()))
            seen = set()
            keep_new = []
            for i in range(dest.numel()):
                k = (int(dest[i]), int(src[i]))
                if k in have or k in seen:
                    skipped += 1
                    continue
                seen.add(k)
                keep_new.append(i)
            if not keep_new:
                return {"created": 0, "skipped_existing": skipped}
            dest, src, vals = dest[keep_new], src[keep_new], vals[keep_new]
        idx = torch.cat([self.index, torch.stack([dest, src], dim=0)], dim=1)
        val = torch.cat([self.values, vals], dim=0)
        wf = torch.cat([self.weak.float(),
                        torch.full((int(vals.numel()),), float(bool(weak)),
                                   device=self.device)])
        mat = torch.sparse_coo_tensor(
            idx, val, (self.num_slots, self.num_slots), device=self.device
        ).coalesce()
        wmat = torch.sparse_coo_tensor(
            idx, wf, (self.num_slots, self.num_slots), device=self.device
        ).coalesce()
        self.index = mat.indices()
        self.values = mat.values()
        self.values = self.values.clamp(-self._neg_cap(), self.cfg.max_weight)
        self.weak = wmat.values() > 0
        self._prune_zero()
        self.enforce_degree()
        return {"created": int(dest.numel()), "skipped_existing": int(skipped)}

    @torch.no_grad()
    def modulate(
        self,
        edges: Sequence[Tuple[int, int, float]],
        mod,
        *,
        lr: float,
        decay: Optional[float] = None,
        drop_below: float = 0.0,
    ) -> Dict[str, float]:
        stats: Dict[str, float] = {
            "n_edges_in": float(len(edges)),
            "n_edges_touched": 0.0,
            "n_edges_missing": 0.0,
            "n_dest": 0.0,
            "pair_pos": 0.0,
            "pair_neg": 0.0,
            "mod_abs_sum": 0.0,
            "upd_abs_sum": 0.0,
            "n_dropped": 0.0,
            "sign_flips": 0.0,
        }
        if self.num_edges == 0:
            return stats
        rate = float(self.cfg.decay if decay is None else decay)
        if isinstance(mod, dict):

            def _mod(slot: int) -> float:
                return float(mod.get(int(slot), 0.0))

        else:
            vec = torch.as_tensor(mod, dtype=torch.float32, device=self.device).reshape(
                -1
            )
            if vec.numel() != self.num_slots:
                raise ValueError(f"mod 向量长度 {vec.numel()} 与 num_slots {self.num_slots} 不一致")

            def _mod(slot: int) -> float:
                return float(vec[int(slot)])

        pos = {
            (int(d), int(s)): i
            for i, (d, s) in enumerate(
                zip(self.index[0].tolist(), self.index[1].tolist())
            )
        }

        npos_map: Dict[Tuple[int, int], int] = {}
        if int(self.neg_index.numel()):
            npos_map = {
                (int(d), int(s_)): i
                for i, (d, s_) in enumerate(
                    zip(self.neg_index[0].tolist(), self.neg_index[1].tolist())
                )
            }
        hits: List[Tuple[str, int, float, int, float]] = []
        wsum: Dict[int, float] = {}
        for d, src, pair in edges:
            key = (int(d), int(src))
            if key in pos:
                p_ = pos[key]
                hits.append(("pos", p_, float(self.values[p_]), int(d), float(pair)))
            elif key in npos_map:
                p_ = npos_map[key]
                hits.append(("neg", p_, float(self.neg_values[p_]), int(d), float(pair)))
            else:
                stats["n_edges_missing"] += 1.0
                continue
            _w = hits[-1][2]
            dd, pp = int(d), float(pair)

            wsum[dd] = wsum.get(dd, 0.0) + abs(_w)
            stats["mod_abs_sum"] += abs(_mod(dd))
            stats["pair_pos" if pp >= 0 else "pair_neg"] += abs(pp)

        sign_flips = 0.0
        if hits:
            before = torch.cat([self.values, self.neg_values])
            upd_pos = torch.zeros_like(self.values)
            upd_neg = torch.zeros_like(self.neg_values)
            for tbl, p_, w, dd, pp in hits:
                tot = wsum[dd]
                share = (abs(w) / tot) if tot > 0.0 else 0.0

                delta = float(lr) * _mod(dd) * pp * share
                if tbl == "pos":
                    upd_pos[p_] += delta
                else:
                    upd_neg[p_] += delta
            stats["upd_abs_sum"] = float(upd_pos.abs().sum() + upd_neg.abs().sum())
            self.values = self.values + upd_pos
            self.neg_values = self.neg_values + upd_neg
            after = torch.cat([self.values, self.neg_values])
            active = after != 0
            sign_flips = float(
                ((torch.sign(before) != torch.sign(after)) & active
                 & (torch.sign(before) != 0)).sum()
            )
        stats["n_edges_touched"] = float(len(hits))
        stats["n_dest"] = float(len(wsum))
        stats["sign_flips"] = sign_flips

        if rate > 0.0:
            self.values = self.values * (1.0 - rate)
            if self.neg_values.numel():
                self.neg_values = self.neg_values * (1.0 - rate)
        self.values = self.values.clamp(-self._neg_cap(), self.cfg.max_weight)
        if self.neg_values.numel():
            self.neg_values = self.neg_values.clamp(-self._neg_cap(), 0.0)

        n_to_neg = 0
        if bool((self.values < 0).any()):
            _mv = self.values < 0
            _d = self.index[0][_mv]; _s2 = self.index[1][_mv]; _v = self.values[_mv]
            n_to_neg = self._neg_merge_add(_d, _s2, _v)
            keep = ~_mv
            self.index = self.index[:, keep]
            self.values = self.values[keep]
            self.weak = self.weak[keep]
        n_to_pos = 0
        if int(self.neg_values.numel()) and bool((self.neg_values > 0).any()):
            _mv = self.neg_values > 0
            _d = self.neg_index[0][_mv]; _s2 = self.neg_index[1][_mv]; _v = self.neg_values[_mv]
            _keep_n = ~_mv
            self.neg_index = self.neg_index[:, _keep_n]
            self.neg_values = self.neg_values[_keep_n]
            _keys = set(zip(self.index[0].tolist(), self.index[1].tolist()))
            _new_d, _new_s, _new_v = [], [], []
            for _dd, _ss, _vv in zip(_d.tolist(), _s2.tolist(), _v.tolist()):
                if (int(_dd), int(_ss)) in _keys:
                    _idx = pos.get((int(_dd), int(_ss)))
                    if _idx is None:
                        _idx = int(((self.index[0] == _dd) & (self.index[1] == _ss)).nonzero()[0, 0])
                    self.values[_idx] = self.values[_idx] + float(_vv)
                else:
                    _new_d.append(int(_dd)); _new_s.append(int(_ss)); _new_v.append(float(_vv))
            if _new_d:
                self.index = torch.cat([self.index,
                                        torch.tensor([_new_d, _new_s], dtype=self.index.dtype,
                                                     device=self.device)], dim=1)
                self.values = torch.cat([self.values,
                                         torch.tensor(_new_v, dtype=self.values.dtype,
                                                      device=self.device)])
                self.weak = torch.cat([self.weak,
                                       torch.zeros(len(_new_d), dtype=self.weak.dtype,
                                                   device=self.device)])
            n_to_pos = len(_new_d)
        stats["n_flipped_to_neg"] = float(n_to_neg)
        stats["n_flipped_to_pos"] = float(n_to_pos)

        if float(drop_below) > 0.0 and hits:
            keep = torch.ones_like(self.values, dtype=torch.bool)
            for tbl, p_, _w, dd, _pp in hits:
                if tbl != "pos" or p_ >= int(self.values.numel()):
                    continue
                if _mod(dd) < 0.0 and abs(float(self.values[p_])) < float(drop_below):
                    keep[p_] = False
            n = int((~keep).sum())
            if n:
                self.index = self.index[:, keep]
                self.values = self.values[keep]
                self.weak = self.weak[keep]
            stats["n_dropped"] = float(n)
        return stats

    @torch.no_grad()
    def add_negative_edges(
        self,
        u: Dict[int, float],
        *,
        attended: Optional[Sequence[int]],
        gain: float,
        dead_zone: float = 0.0,
        exclude: Sequence[int] = (),
        q: Optional[float] = None,
        max_per_dest: int = 0,
    ) -> int:
        if float(gain) <= 0.0 or not u:
            return 0
        excl = {int(s) for s in exclude}
        uu = {int(k): float(v) for k, v in u.items() if int(k) not in excl}
        if not uu:
            return 0
        if attended is None:
            if q is None:
                raise ValueError("add_negative_edges 需要 attended 或 q")
            n_top = max(1, int(round(float(q) * len(uu))))
            tops = [s for s, _v in sorted(uu.items(), key=lambda kv: -kv[1])[:n_top]]
            att = [s for s in tops]
            tset = set(tops)
            un = [s for s in uu if s not in tset]
        else:
            aset = {int(s) for s in attended}
            att = [s for s in uu if s in aset]
            un = [
                s
                for s, v in uu.items()
                if s not in aset and (float(dead_zone) <= 0.0 or v < float(dead_zone))
            ]
        if not att or not un:
            return 0

        def _competition(j: int, i: int) -> float:
            us = float(uu.get(int(i), 0.0))
            ud = float(uu.get(int(j), 0.0))
            if us <= 0.0:
                return 0.0
            return max(0.0, min(1.0, (us - ud) / us))

        fanin = self.fanin()
        per = float(gain) / float(len(att))
        dest: List[int] = []
        src: List[int] = []
        vals: List[float] = []
        for j in un:
            fj = float(fanin[int(j)]) if 0 <= int(j) < fanin.numel() else 0.0
            if fj <= 0.0:

                continue
            for i in att:
                if int(i) == int(j):
                    continue
                dest.append(int(j))
                src.append(int(i))
                vals.append(-per * fj * _competition(int(j), int(i)))
        if not dest:
            return 0

        cap = int(max_per_dest)
        if cap > 0:
            order = {int(s): i for i, s in enumerate(att)}
            by_dest: Dict[int, List[int]] = {}
            for k, j in enumerate(dest):
                by_dest.setdefault(int(j), []).append(k)
            keep: List[int] = []
            for _j, ks in by_dest.items():
                ks.sort(key=lambda k: order.get(int(src[k]), 10 ** 9))
                keep.extend(ks[:cap])
            keep.sort()
            dest = [dest[k] for k in keep]
            src = [src[k] for k in keep]
            vals = [vals[k] for k in keep]
            if not dest:
                return 0
            kept_per_dest: Dict[int, int] = {}
            for j in dest:
                kept_per_dest[int(j)] = kept_per_dest.get(int(j), 0) + 1
            vals = [
                -float(gain) * (float(fanin[int(j)]) if 0 <= int(j) < fanin.numel() else 0.0)
                / float(kept_per_dest[int(j)])
                * _competition(int(j), int(i))
                for j, i in zip(dest, src)
            ]
        st = self.add_laterals(dest, src, vals, mode="add")
        return int(st.get("created", 0))

    def _prune_zero(self) -> None:
        if self.neg_index.shape[1]:
            _k = self.neg_values != 0
            if int((~_k).sum()):
                self.neg_index = self.neg_index[:, _k]
                self.neg_values = self.neg_values[_k]
        keep = self.values != 0
        if int((~keep).sum()):
            self.index = self.index[:, keep]
            self.values = self.values[keep]
            self.weak = self.weak[keep]

    def enforce_degree(self, max_out_degree: Optional[int] = None) -> None:
        cap = int(self.cfg.max_out_degree if max_out_degree is None else max_out_degree)
        if cap <= 0 or self.num_edges == 0:
            return
        rows: Dict[int, List[Tuple[int, int, float, bool]]] = {}
        for s, d, v, wk in zip(
            self.index[1].tolist(), self.index[0].tolist(),
            self.values.tolist(), self.weak.tolist()
        ):
            rows.setdefault(int(s), []).append((int(d), int(s), float(v), bool(wk)))
        keep_d, keep_s, keep_v, keep_w = [], [], [], []
        for _s, row in rows.items():
            pos = [t for t in row if t[2] > 0]
            neg = [t for t in row if t[2] <= 0]
            pos.sort(key=lambda kv: -abs(kv[2]))
            neg.sort(key=lambda kv: -abs(kv[2]))
            for d, s, v, wk in pos[:cap] + neg:
                keep_d.append(d)
                keep_s.append(s)
                keep_v.append(v)
                keep_w.append(wk)
        self.index = torch.tensor(
            [keep_d, keep_s], dtype=torch.long, device=self.device
        )
        self.values = torch.tensor(keep_v, dtype=torch.float32, device=self.device)
        self.weak = torch.tensor(keep_w, dtype=torch.bool, device=self.device)


    def ensure_slots(self, num_slots: int) -> None:
        if self.node is not None:
            return
        n = int(num_slots)
        if n > int(self._standalone_slots):
            self._standalone_slots = n

    def state_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index.detach().cpu(),
            "values": self.values.detach().cpu(),
            "neg_index": self.neg_index.detach().cpu(),
            "neg_values": self.neg_values.detach().cpu(),
            "config": self.cfg.to_json(),
        }

    @classmethod
    def load_state_dict(
        cls, state: Dict[str, Any], *, node: Optional[Any] = None
    ) -> "LateralState":
        cfg = (
            LateralConfig.from_json(state["config"])
            if "config" in state
            else LateralConfig()
        )
        lat = cls(state["index"], state["values"], cfg, node=node)
        if "neg_index" in state and int(state["neg_index"].shape[1]):
            lat.neg_index = state["neg_index"].long().to(lat.device)
            lat.neg_values = state["neg_values"].float().to(lat.device)
        return lat

    def __repr__(self) -> str:
        return (
            f"LateralState(slots={self.num_slots}, edges={self.num_edges}, "
            f"mean_degree={self.mean_degree:.2f})"
        )

class SynapticaNode:

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        if int(input_size) <= 0:
            raise ValueError(f"input_size 必须是正整数，收到 {input_size}")
        if int(output_size) <= 0:
            raise ValueError(f"output_size 必须是正整数，收到 {output_size}")

        self.input_size = int(input_size)
        self.output_size = int(output_size)

        self.mem = torch.zeros(self.output_size, dtype=dtype, device=device)

        self.lateral: Optional["LateralState"] = None

    def attach_lateral(
        self,
        lateral: Optional["LateralState"] = None,
        cfg: Optional[LateralConfig] = None,
    ) -> "LateralState":
        if lateral is None:
            lateral = LateralState(
                torch.zeros(2, 0, dtype=torch.long, device=self.mem.device),
                torch.zeros(0, device=self.mem.device),
                cfg,
                node=self,
            )
        else:
            lateral.node = self
            if lateral.device != self.mem.device:
                lateral.index = lateral.index.to(self.mem.device)
                lateral.values = lateral.values.to(self.mem.device)
                lateral.device = self.mem.device
        self.lateral = lateral
        return lateral

    def state_dict(self) -> Dict[str, Any]:
        return {
            "input_size": int(self.input_size),
            "output_size": int(self.output_size),
            "lateral": (
                self.lateral.state_dict() if self.lateral is not None else None
            ),
        }

    def load_state_dict(
        self,
        state: Dict[str, Any],
        *,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "SynapticaNode":
        n_out = int(state["output_size"])
        if n_out != int(self.output_size):
            self.grow(n_out)
        self.lateral = None
        lat_state = state.get("lateral")
        if lat_state is not None:
            self.lateral = LateralState.load_state_dict(lat_state, node=self)
        return self

    def __repr__(self) -> str:
        n_lat = 0 if self.lateral is None else int(self.lateral.num_edges)
        return (
            f"SynapticaNode(input={self.input_size}, slots={self.output_size}, "
            f"l_edges={n_lat})"
        )

    def grow(self, new_size: int) -> bool:
        new_size = int(new_size)
        if new_size <= self.output_size:
            return False
        n_new = new_size - self.output_size
        self.mem = torch.cat(
            [self.mem, torch.zeros(n_new, dtype=self.mem.dtype, device=self.mem.device)]
        )
        self.output_size = new_size
        return True

    def reset(self) -> None:
        self.mem = torch.zeros_like(self.mem)


__all__ = [
    "SynapticaNode",
    "LateralConfig",
    "LateralState",
    "dest_in_prefix_is_attended",
]
