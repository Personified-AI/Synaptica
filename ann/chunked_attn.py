
from __future__ import annotations

import itertools
from contextlib import contextmanager
from typing import Any, Callable, Optional

import torch

_PREFIX = "synaptica_chunked"
_counter = itertools.count()

def _repeat_kv(x: torch.Tensor, n: int) -> torch.Tensor:
    if n == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None].expand(b, h, n, t, d).reshape(b, h * n, t, d)

def _head_reduce(x: torch.Tensor) -> torch.Tensor:
    return x.mean(0)


def _chunked_aggregate(module, query, key, value, attention_mask, scaling, dropout,
                       aggregator, block: int):
    groups = max(1, int(query.shape[1]) // int(key.shape[1]))
    key_r = _repeat_kv(key, groups)
    val_r = _repeat_kv(value, groups)
    n_rows = int(query.shape[2])
    out = torch.empty_like(query)
    block = max(1, int(block))
    rows = torch.arange(n_rows, device=query.device)
    for start in range(0, n_rows, block):
        end = min(start + block, n_rows)
        scores = torch.matmul(query[:, :, start:end], key_r.transpose(-1, -2)) * scaling

        keep = rows.view(1, -1) <= rows[start:end].view(-1, 1)
        scores = scores.masked_fill(~keep, float("-inf"))
        if attention_mask is not None:
            scores = scores + attention_mask[:, :, start:end, :]

        probs = torch.softmax(scores, dim=-1)

        _rows = _head_reduce(probs[0])
        aggregator.add_rows(_rows, start)
        del _rows
        out[:, :, start:end] = torch.matmul(probs, val_r)
        del scores, probs
    return out.transpose(1, 2).contiguous(), None

def _module_layer_idx(module: Any) -> int:
    direct = getattr(module, "layer_idx", None)
    if direct is not None:
        return int(direct)
    inner = getattr(module, "self_attn", None)
    if inner is not None:
        nested = getattr(inner, "layer_idx", None)
        if nested is not None:
            return int(nested)
    return -1

def _make_forward(aggregator: Any, layer_idx: int, block: int,
                  layers: Any = None, aggregators: Any = None) -> Callable:
    _tgt = {int(x) for x in layers} if layers else {int(layer_idx)}
    _aggs = {int(k): v for k, v in (aggregators or {}).items()}
    def _forward(module, query, key, value, attention_mask, scaling,
                 dropout: float = 0.0, **kwargs):
        if _module_layer_idx(module) not in _tgt:
            from .fast_attn import expand_kv_sdpa
            return expand_kv_sdpa(module, query, key, value, attention_mask, scaling,
                                  dropout=dropout, **kwargs)
        _agg = _aggs.get(_module_layer_idx(module), aggregator)
        return _chunked_aggregate(module, query, key, value, attention_mask, scaling,
                                  dropout, _agg, block)
    return _forward

@contextmanager
def chunked_attention(config: Any, layer_idx: int, aggregator: Any, *,
                      block: int = 512, layers: Any = None,
                      aggregators: Any = None):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    name = f"{_PREFIX}_{next(_counter)}"
    ALL_ATTENTION_FUNCTIONS.register(
        name, _make_forward(aggregator, layer_idx, block, layers, aggregators))

    def _model_level_config() -> Any:
        for attr in ("model", "text_model"):
            inner = getattr(config, attr, None)
            inner_cfg = getattr(inner, "config", None) if inner is not None else None
            if inner_cfg is not None:
                return inner_cfg
        try:
            tc = config.get_text_config()
            if tc is not config:
                return tc
        except Exception:
            pass
        return None

    others = []
    mc = _model_level_config()
    if mc is not None and mc is not config:
        others.append(mc)

    olds = [(config, getattr(config, "_attn_implementation", None))]
    for c in others:
        olds.append((c, getattr(c, "_attn_implementation", None)))
    for c, _old in olds:
        c._attn_implementation = name
    try:
        yield
    finally:
        for c, old in olds:
            if old is None:
                try:
                    delattr(c, "_attn_implementation")
                except Exception:
                    pass
            else:
                c._attn_implementation = old
        try:
            ALL_ATTENTION_FUNCTIONS.pop(name, None)
        except Exception:
            pass

@contextmanager
def chunked_attention_multi(configs, layer_idx: int, aggregator: Any, *,
                            block: int = 512,
                            layers: Any = None, aggregators: Any = None):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    cs = [c for i, c in enumerate(configs) if c is not None
          and all(c is not p for p in configs[:i])]
    if not cs:
        raise ValueError("chunked_attention_multi 需要至少一份 config")
    name = f"{_PREFIX}_{next(_counter)}"
    ALL_ATTENTION_FUNCTIONS.register(
        name, _make_forward(aggregator, layer_idx, block, layers, aggregators))
    olds = [(c, getattr(c, "_attn_implementation", None)) for c in cs]
    for c, _old in olds:
        c._attn_implementation = name
    try:
        yield
    finally:
        for c, old in olds:
            if old is None:
                try:
                    delattr(c, "_attn_implementation")
                except Exception:
                    pass
            else:
                c._attn_implementation = old
        try:
            ALL_ATTENTION_FUNCTIONS.pop(name, None)
        except Exception:
            pass
