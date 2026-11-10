
from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F

from .chunked_attn import _repeat_kv

_NAME = "synaptica_expand_kv"

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    _HAVE_SDPA_KERNEL = True
except Exception:
    SDPBackend = None
    sdpa_kernel = None
    _HAVE_SDPA_KERNEL = False

_EFFICIENT_OK = bool(_HAVE_SDPA_KERNEL)


def _disable_efficient() -> None:
    global _EFFICIENT_OK
    _EFFICIENT_OK = False

def expand_kv_sdpa(module, query, key, value, attention_mask, scaling,
                    dropout: float = 0.0, **kwargs):
    groups = max(1, int(query.shape[1]) // int(key.shape[1]))

    if groups > 1:
        key = _repeat_kv(key, groups)
        value = _repeat_kv(value, groups)
    kwargs.pop("enable_gqa", None)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
    q_len = int(query.shape[-2])
    is_causal = True if causal_mask is None and q_len > 1 else False

    if _EFFICIENT_OK:
        try:
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                out = F.scaled_dot_product_attention(
                    query, key, value,
                    attn_mask=causal_mask,
                    dropout_p=0.0 if not module.training else dropout,
                    scale=scaling,
                    is_causal=is_causal,
                )
            return out.transpose(1, 2).contiguous(), None
        except Exception:
            _disable_efficient()

    out = F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=causal_mask,
        dropout_p=0.0 if not module.training else dropout,
        scale=scaling,
        is_causal=is_causal,
    )
    return out.transpose(1, 2).contiguous(), None

_OVERRIDABLE = (None, "sdpa", _NAME)

@contextmanager
def expand_kv_attention(config: Any):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    old = getattr(config, "_attn_implementation", None)
    if old not in _OVERRIDABLE:

        yield
        return
    if _NAME not in ALL_ATTENTION_FUNCTIONS:
        ALL_ATTENTION_FUNCTIONS.register(_NAME, expand_kv_sdpa)
    config._attn_implementation = _NAME
    try:
        yield
    finally:
        config._attn_implementation = old
