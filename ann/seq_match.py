from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

_LEN_GATE_L0 = 5.0
__all__ = ["SeqMatcher", "kadane_rows", "len_gate"]

def len_gate(lens: Optional[Tensor]) -> Optional[Tensor]:
    if lens is None:
        return None
    return (lens.float() / _LEN_GATE_L0).clamp(max=1.0)

def kadane_rows(
    X: Tensor,
    valid: Optional[Tensor] = None,
    *,
    return_index: bool = False,
) -> Tuple[Tensor, Tensor]:
    if X.dim() != 2:
        raise ValueError(f"kadane_rows 需要二维矩阵，收到 {tuple(X.shape)}")
    B, T = int(X.shape[0]), int(X.shape[1])
    dev = X.device
    zero = X.new_zeros(B)
    best = zero.clone()
    best_s = torch.full((B,), -1, dtype=torch.long, device=dev)
    best_n = torch.zeros(B, dtype=torch.long, device=dev)
    cur = zero.clone()
    cur_s = torch.zeros(B, dtype=torch.long, device=dev)
    for t in range(T):
        x = X[:, t]
        if valid is not None:
            x = torch.where(valid[:, t], x, zero)
        pos = x > 0
        restart = (~pos) | (cur <= 0)
        cur = torch.where(restart, x.clamp_min(0.0), cur + x)
        cur = torch.where(pos, cur, zero)
        cur_s = torch.where(restart, torch.full_like(cur_s, t), cur_s)
        better = pos & (cur > best)
        best = torch.where(better, cur, best)
        best_s = torch.where(better, cur_s, best_s)
        best_n = torch.where(better, (t - cur_s + 1).to(torch.long), best_n)
    if return_index:
        return best, best_n, best_s
    return best, best_n

class SeqMatcher:

    def __init__(self, *, theta: float = 0.4, normalize: bool = True) -> None:
        if not 0.0 <= float(theta) < 1.0:
            raise ValueError(f"theta 必须在 [0, 1)，收到 {theta!r}")
        self.theta = float(theta)
        self.normalize = bool(normalize)

        self._cand_cache: dict = {}
        self._cand_hit = 0
        self._cand_miss = 0

        self._cand_max = 8

        self.score_device: Optional[torch.device] = None

        self._sp_cache: dict = {}
        self._sp_max = 4

        if not self.normalize:

            import warnings

            warnings.warn(
                "SeqMatcher(normalize=False)：调用方必须保证传入的每行特征已是"
                "单位长度，否则打分不是余弦且会静默失真（近零行会被放大）。"
                "不确定就保持 normalize=True。",
                RuntimeWarning,
                stacklevel=2,
            )

    @staticmethod
    def _cand_key_ok(entry, feats) -> bool:
        refs = entry[3] if len(entry) > 3 else None
        if refs is None or len(refs) != len(feats):
            return False
        return all(a is b for a, b in zip(refs, feats))

    def _prep(self, tokens: Tensor) -> Tensor:
        x = torch.as_tensor(tokens, dtype=torch.float32)
        if x.dim() != 2:
            raise ValueError(f"token 特征必须是 (n, k)，收到 {tuple(x.shape)}")
        if self.normalize:
            x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return x

    def score_matrix(
        self,
        ref_feat: Tensor,
        b_feat: Tensor,
        freq_a: Optional[Tensor] = None,
        freq_b: Optional[Tensor] = None,
    ) -> Tensor:
        A = self._prep(ref_feat)
        B = self._prep(b_feat)
        C = (A @ B.t()).clamp_min(0.0)
        th = float(self.theta)
        if th > 0.0:
            C = torch.where(C >= th, C, torch.zeros_like(C))
        gb = len_gate(freq_b)
        if gb is not None:
            C = C * gb.reshape(1, -1).to(C.device)
        return C

    def score(
        self,
        ref_feat: Tensor,
        b_feat: Tensor,
        *,
        freq_a: Optional[Tensor] = None,
        freq_b: Optional[Tensor] = None,
    ) -> Tuple[float, int]:
        raw, L, _i = self.align(ref_feat, b_feat, freq_a=freq_a, freq_b=freq_b)
        return (float(raw) if L > 0 else 0.0), int(L)

    def align(
        self,
        ref_feat: Tensor,
        b_feat: Tensor,
        *,
        freq_a: Optional[Tensor] = None,
        freq_b: Optional[Tensor] = None,
    ) -> Tuple[float, int, int]:
        S = self.score_matrix(ref_feat, b_feat, freq_a, freq_b)
        _rowmax = S.max(dim=1).values if int(S.shape[1]) > 0 else S.new_zeros(int(S.shape[0]))
        _sum, _len, _i0 = kadane_rows(_rowmax.reshape(1, -1), return_index=True)
        return float(_sum[0]) if int(_len[0]) > 0 else 0.0, int(_len[0]), int(_i0[0])

    def best_of(
        self,
        query_feat: Tensor,
        pool: Sequence[Any],
        *,
        freq_q: Optional[Tensor] = None,
        query_valid: Optional[int] = None,
        cand_valid: Optional[Sequence[int]] = None,
        exclude_slot: Optional[int] = None,
        cand_ids: Optional[Sequence[Sequence[int]]] = None,
        bad_ids: Optional[set] = None,
    ) -> Optional[Tuple[int, float, int]]:
        got = self.top_of(query_feat, pool, k=1, freq_q=freq_q, query_valid=query_valid,
                          cand_valid=cand_valid, exclude_slot=exclude_slot,
                          cand_ids=cand_ids, bad_ids=bad_ids)
        return got[0] if got else None

    def top_of(
        self,
        query_feat: Tensor,
        pool: Sequence[Any],
        *,
        k: int = 1,
        freq_q: Optional[Tensor] = None,
        query_valid: Optional[int] = None,
        cand_valid: Optional[Sequence[int]] = None,
        exclude_slot: Optional[int] = None,
        cand_ids: Optional[Sequence[Sequence[int]]] = None,
        bad_ids: Optional[set] = None,
    ) -> List[Tuple[int, float, int]]:
        out = self._score_pool(query_feat, pool, freq_q=freq_q, query_valid=query_valid,
                               cand_valid=cand_valid, exclude_slot=exclude_slot,
                               cand_ids=cand_ids, bad_ids=bad_ids)
        if out is None:
            return []
        slots, scores, lens = out
        kk = max(0, int(k))
        if kk == 0:
            return []
        kk = min(kk, int(scores.numel()))
        if kk <= 0:
            return []
        vals, idxs = torch.topk(scores, kk)
        return [(int(slots[int(b)]), float(vals[i]), int(lens[int(b)]))
                for i, b in enumerate(idxs.tolist())]

    def premove_freqs(self, freqs: Sequence[Optional[Tensor]], *, device) -> List[Optional[Tensor]]:
        idx = [i for i, f in enumerate(freqs) if f is not None]
        out: List[Optional[Tensor]] = list(freqs)
        if not idx:
            return out
        lens = [int(torch.as_tensor(freqs[i]).numel()) for i in idx]
        n = max(lens)
        buf = torch.zeros(len(idx), n, dtype=torch.float32)
        for j, i in enumerate(idx):
            f = torch.as_tensor(freqs[i]).reshape(-1).float()
            buf[j, : int(f.numel())] = f
        buf = buf.to(device)
        for j, i in enumerate(idx):
            out[i] = buf[j, : lens[j]].contiguous()
        return out

    def _score_pool(
        self,
        query_feat: Tensor,
        pool: Sequence[Any],
        *,
        freq_q: Optional[Tensor] = None,
        query_valid: Optional[int] = None,
        cand_valid: Optional[Sequence[int]] = None,
        exclude_slot: Optional[int] = None,
        cand_ids: Optional[Sequence[Sequence[int]]] = None,
        bad_ids: Optional[set] = None,
    ):
        if not pool:
            return None
        slots = [int(it[0]) for it in pool]
        feats = [it[1] for it in pool]
        freqs = [it[2] if len(it) > 2 else None for it in pool]
        qp = self._prep(query_feat)
        if self.score_device is not None:
            qp = qp.to(self.score_device)

        lengths = [int(f.shape[0]) for f in feats]
        kdim = int(qp.shape[1])
        for f, ln in zip(feats, lengths):
            if int(f.shape[1]) != kdim:
                raise ValueError(f"候选的每 token 维度 {int(f.shape[1])} 与查询的 {kdim} 不一致")
        nq = int(qp.shape[0])
        m_max = max(lengths) if lengths else 0
        B = len(feats)
        if nq == 0 or m_max == 0 or B == 0:
            return None

        T = max(nq, m_max)
        _nq_v = int(nq) if query_valid is None else max(0, min(int(query_valid), int(nq)))
        rows = qp.new_zeros(B, T, kdim)
        rows[:, :nq] = qp

        _ckey = (tuple(id(f) for f in feats), int(T), int(_nq_v),
                 None if cand_valid is None else tuple(cand_valid))
        _cgot = self._cand_cache.get(_ckey)
        if _cgot is not None and self._cand_key_ok(_cgot, feats):
            self._cand_hit += 1
            cols, mask, gb_pad = _cgot[0], _cgot[1], _cgot[2]
        else:
            self._cand_miss += 1
            cols = qp.new_zeros(B, T, kdim)
            for b, (f, ln) in enumerate(zip(feats, lengths)):
                cols[b, :ln] = self._prep(f)

            mask = torch.zeros(B, T, T, dtype=torch.bool, device=rows.device)
            for b, ln in enumerate(lengths):

                _lv = int(ln)

                if cand_valid is not None and b < len(cand_valid) and cand_valid[b] is not None:
                    _lv = min(_lv, max(0, int(cand_valid[b])))
                mask[b, :_nq_v, :_lv] = True
            _gb_list = [len_gate(fq) for fq in freqs]
            gb_pad = None
            if any(g is not None for g in _gb_list):
                gb_pad = qp.new_ones(B, T)
                for b, (g, ln) in enumerate(zip(_gb_list, lengths)):
                    if g is not None:
                        gb_pad[b, :ln] = g.to(gb_pad.device).reshape(-1)
            if len(self._cand_cache) >= self._cand_max:
                self._cand_cache.clear()

            self._cand_cache[_ckey] = (cols, mask, gb_pad, list(feats))

        _mkey = (
            _ckey,
            id(query_feat),
            (None if freq_q is None else id(freq_q)),
            (None if exclude_slot is None else int(exclude_slot)),
            (None if not bad_ids else frozenset(int(x) for x in bad_ids)),
            (None if cand_ids is None else id(cand_ids)),
        )
        _mhit = self._sp_cache.get(_mkey)
        if _mhit is not None and self._sp_key_ok(_mhit, query_feat, feats):
            return _mhit[0], _mhit[1], _mhit[2]

        if cand_ids is not None and bad_ids:
            for _b, _ids in enumerate(cand_ids):
                if not _ids:
                    continue
                for _j, _tid in enumerate(_ids):
                    if int(_tid) in bad_ids and _j < int(mask.shape[2]):
                        mask[_b, :, _j] = False

        _excl_backup = None
        if exclude_slot is not None:
            for _b, _sl in enumerate(slots):
                if int(_sl) == int(exclude_slot):
                    _excl_backup = (int(_b), mask[int(_b)].clone())
                    mask[int(_b)] = False
                    break

        C = torch.bmm(rows, cols.transpose(1, 2)).clamp_min(0.0)
        _th = float(self.theta)
        if _th > 0.0:
            C = torch.where(C >= _th, C, torch.zeros_like(C))

        if gb_pad is not None:
            C = C * gb_pad.reshape(B, 1, T)
        C = torch.where(mask, C, torch.zeros_like(C))

        _rowmax, _rowarg = C.max(dim=2)
        if int(nq) < int(T):
            _rowmax[:, nq:] = 0.0
        sums, lens = kadane_rows(_rowmax)
        scores = torch.where(lens > 0, sums, torch.zeros_like(sums))
        if _excl_backup is not None:
            mask[_excl_backup[0]] = _excl_backup[1]

        if len(self._sp_cache) >= int(self._sp_max):
            self._sp_cache.clear()
        self._sp_cache[_mkey] = (slots, scores, lens, query_feat, list(feats))
        return slots, scores, lens

    @staticmethod
    def _sp_key_ok(entry, query_feat, feats) -> bool:
        if entry[3] is not query_feat:
            return False
        refs = entry[4]
        if len(refs) != len(feats):
            return False
        return all(a is b for a, b in zip(refs, feats))

    def score_one(
        self,
        query_feat: Tensor,
        cand_feat: Tensor,
        *,
        freq_q: Optional[Tensor] = None,
        freq_c: Optional[Tensor] = None,
    ) -> float:
        return float(self.score(query_feat, cand_feat, freq_a=freq_q, freq_b=freq_c)[0])

    def __repr__(self) -> str:
        return f"SeqMatcher(theta={self.theta})"
