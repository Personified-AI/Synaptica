from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import torch

from ..attention_graph import validate_span_intervals


def capture_chunked(backend, input_ids, layer: int, spans, attn_cfg,
                    row_block: int, attention_mask):
    from ..attention_graph import SegmentAggregator
    from ..chunked_attn import chunked_attention_multi as _chunked_attention_multi

    n_rows = int(input_ids.shape[-1])
    aggregator = SegmentAggregator(spans, attn_cfg, num_rows=n_rows)

    cfgs = []
    for _owner in (getattr(backend, "model", None),
                   getattr(backend, "text_model", None)):
        _c = getattr(_owner, "config", None) if _owner is not None else None
        if _c is not None and all(_c is not x for x in cfgs):
            cfgs.append(_c)
    if not cfgs:
        cfgs = [backend.text_model.config]
    with _chunked_attention_multi(cfgs, int(layer), aggregator,
                                  block=int(row_block)):
        hidden = backend.truncated_hidden(input_ids, int(layer),
                                          attention_mask=attention_mask,
                                          full_attn_mask=False)

    if int(getattr(aggregator, "_rows_seen", 0)) <= 0:
        raise RuntimeError(f"段级聚合器未收到任何行（layer={int(layer)}, T={n_rows}）")
    return hidden, aggregator.finalize()


_SELF_CAUSAL_ATTN_IMPLS = frozenset({"sdpa"})


def _needs_full_attn_mask(backend) -> bool:
    cfg = getattr(getattr(backend, "text_model", None), "config", None)
    impl = getattr(cfg, "_attn_implementation", None)
    return str(impl) not in _SELF_CAUSAL_ATTN_IMPLS


def capture_hidden_only(backend, input_ids, layer: int, *, attention_mask=None):
    return backend.truncated_hidden(
        input_ids, int(layer), attention_mask=attention_mask,
        full_attn_mask=_needs_full_attn_mask(backend),
    )


class _DecodeMassAggregator:

    def __init__(self, spans: Sequence[Tuple[int, int]], num_rows: int,
                 gen_start: int) -> None:
        self.spans = validate_span_intervals(spans, int(num_rows))
        self.M = len(self.spans)
        self.num_rows = int(num_rows)
        self.gen_start = int(gen_start)
        self.best = torch.full((self.M,), float("-inf"), dtype=torch.float32)
        self.rows_seen = 0

    def add_rows(self, rows: torch.Tensor, offset: int = 0) -> None:
        n = int(rows.shape[0])
        lo_local = max(0, self.gen_start - int(offset))
        if lo_local >= n:
            return
        sub = rows[lo_local:].detach().float().cpu()
        for j, (c, d) in enumerate(self.spans):
            if d > c:
                v = float(sub[:, c:d].max())
                if v > float(self.best[j]):
                    self.best[j] = v
        self.rows_seen += int(sub.shape[0])

    def result(self, slots: Sequence[int]) -> Dict[int, float]:
        out: Dict[int, float] = {}
        for j, slot in enumerate(slots):
            v = float(self.best[j])
            if v != float("-inf"):
                out[int(slot)] = v
        return out
