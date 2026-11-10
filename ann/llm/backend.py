
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

DEFAULT_MODEL_PATH = "models/Qwen3.5-2B"

from .capture import capture_chunked, _DecodeMassAggregator

class QwenBackend:

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        *,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        lazy: bool = False,
        attn_implementation: Optional[str] = None,
        device_map: Optional[Any] = None,
        max_memory: Optional[Dict[Any, str]] = None,
        expand_kv_in_generate: Optional[bool] = None,
        load_in_8bit: bool = False,
    ) -> None:
        self.model_path = str(model_path)
        self.dtype = dtype
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.attn_implementation = attn_implementation

        self.device_map = device_map
        self.max_memory = max_memory

        self.load_in_8bit = bool(load_in_8bit)
        self.tokenizer = None
        self.model = None

        self._expand_kv_override = expand_kv_in_generate
        self.expand_kv_in_generate = False
        if not lazy:
            self.load()

    def load(self) -> "QwenBackend":
        if self.model is not None:
            return self

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "left"

        if self.device_map is None:
            kwargs: Dict[str, object] = {"dtype": self.dtype, "device_map": self.device}
        else:
            kwargs = {"dtype": self.dtype, "device_map": self.device_map}
            if self.max_memory is not None:
                kwargs["max_memory"] = self.max_memory
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation
        if self.load_in_8bit:
            from transformers import BitsAndBytesConfig

            kwargs.pop("dtype", None)
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        try:
            from transformers import Qwen3_5ForConditionalGeneration as _ModelCls
        except ImportError:
            from transformers import AutoModelForCausalLM as _ModelCls

        model = _ModelCls.from_pretrained(self.model_path, **kwargs)
        model.eval()
        model.requires_grad_(False)

        self.tokenizer = tokenizer
        self.model = model

        try:
            emb_dev = next(model.get_input_embeddings().parameters()).device
            self.device = str(emb_dev)
        except Exception:
            pass
        if self._expand_kv_override is None:
            self.expand_kv_in_generate = self._probe_expand_kv_benefit()
        else:
            self.expand_kv_in_generate = bool(self._expand_kv_override)
        return self

    @property
    def _head_dim_guess(self) -> int:
        try:
            cfg = self.text_model.config
            hd = getattr(cfg, "head_dim", None)
            if hd:
                return int(hd)
            return int(getattr(cfg, "hidden_size", 4096)) // int(
                getattr(cfg, "num_attention_heads", 32))
        except Exception:
            return 128

    def _probe_expand_kv_benefit(self) -> bool:
        try:
            cfg = self.text_model.config
            heads = int(getattr(cfg, "num_attention_heads", 0) or 0)
            kv_heads = int(getattr(cfg, "num_key_value_heads", heads) or heads)
        except Exception:
            return False
        if heads <= 0 or kv_heads >= heads:
            return False
        L = 8192
        d = self._head_dim_guess

        try:
            _need = int(heads) * L * L * 4 * 2
            _free, _total = torch.cuda.mem_get_info(self.device)
            if _free < int(_need * 1.2):
                return False
        except Exception:
            pass
        try:
            from ..fast_attn import _repeat_kv

            def peak_of(expand: bool) -> tuple:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                base = float(torch.cuda.memory_allocated())
                q = torch.randn(1, heads, L, d, dtype=torch.float16, device=self.device)
                k = torch.randn(1, kv_heads, L, d, dtype=torch.float16, device=self.device)
                v = torch.randn(1, kv_heads, L, d, dtype=torch.float16, device=self.device)
                try:
                    if expand:
                        g = max(1, heads // kv_heads)
                        k, v = _repeat_kv(k, g), _repeat_kv(v, g)
                    torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, **({} if expand else {"enable_gqa": True}))
                finally:
                    del q, k, v
                peak = float(torch.cuda.max_memory_allocated())
                torch.cuda.empty_cache()
                return (peak - base) / 2**30, peak / 2**30

            inc_native, abs_native = peak_of(False)
            inc_expand, abs_expand = peak_of(True)
            torch.cuda.reset_peak_memory_stats()
            return bool(inc_expand > 0 and inc_native > 5.0 * inc_expand)
        except Exception:
            torch.cuda.empty_cache()
            return False

    def _require_model(self):
        self.load()
        return self.model

    @property
    def text_model(self):
        model = self._require_model()
        inner = getattr(model, "model", None)
        if inner is None:
            return model
        language_model = getattr(inner, "language_model", None)
        if language_model is not None and hasattr(language_model, "layers"):
            return language_model
        if hasattr(inner, "layers"):
            return inner
        raise AttributeError(f"cannot locate text backbone on {type(model).__name__}")

    @property
    def num_layers(self) -> int:
        return int(self.text_model.config.num_hidden_layers)

    @property
    def hidden_size(self) -> int:
        return int(self.text_model.config.hidden_size)

    @torch.no_grad()
    def embeddings_ids(self, ids: Sequence[int]) -> torch.Tensor:
        self.load()
        emb = self.text_model.get_input_embeddings()
        t = torch.as_tensor(list(ids), dtype=torch.long, device=emb.weight.device)
        return emb(t)

    def tokenize(self, text: str, *, add_special_tokens: bool = False) -> List[int]:
        self.load()
        return list(
            self.tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"]
        )

    def tokenize_batch(
        self,
        texts: Sequence[str],
        *,
        add_special_tokens: bool = False,
        padding: bool = True,
    ) -> Dict[str, torch.Tensor]:
        self.load()
        batch = self.tokenizer(
            list(texts),
            add_special_tokens=add_special_tokens,
            padding=padding,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in batch.items()}

    @staticmethod
    def find_span(
        ids: Sequence[int], sub_ids: Sequence[int]
    ) -> Optional[Tuple[int, int]]:
        n, m = len(ids), len(sub_ids)
        if m == 0 or m > n:
            return None
        for start in range(n - m + 1):
            if list(ids[start : start + m]) == list(sub_ids):
                return (start, start + m)
        return None

    def find_text_span(
        self, text: str, sub_text: str, **kw
    ) -> Optional[Tuple[int, int]]:
        ids = self.tokenize(text, **kw)
        sub_ids = self.tokenize(sub_text, **kw)
        return self.find_span(ids, sub_ids)

    def tokenize_with_offsets(
        self, text: str, *, add_special_tokens: bool = False
    ) -> Tuple[List[int], List[Tuple[int, int]]]:
        self.load()
        encoded = self.tokenizer(
            text, add_special_tokens=add_special_tokens, return_offsets_mapping=True
        )
        return list(encoded["input_ids"]), [tuple(o) for o in encoded["offset_mapping"]]

    def span_of_substring(self, text: str, sub_text: str) -> Tuple[int, int]:
        start_char = text.index(sub_text)
        end_char = start_char + len(sub_text)
        _, offsets = self.tokenize_with_offsets(text)
        covered = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > start_char and start < end_char
        ]
        if not covered:
            raise ValueError(f"substring {sub_text!r} not covered by any token")
        return (min(covered), max(covered) + 1)

    def full_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        layers: Optional[Sequence[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        model = self._require_model()
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        with torch.inference_mode():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                logits_to_keep=1,
            )
        hidden_states = out.hidden_states
        wanted = list(range(-1, self.num_layers)) if layers is None else list(layers)
        return {
            layer: hidden_states[layer + 1].detach().to("cpu", torch.float32)
            for layer in wanted
        }

    def full_hidden_batch(
        self,
        sequences: Sequence[Sequence[int]],
        *,
        layers: Optional[Sequence[int]] = None,
        batch_size: int = 8,
    ) -> List[Dict[int, torch.Tensor]]:
        self.load()
        sequences = [list(seq) for seq in sequences]
        wanted = list(range(-1, self.num_layers)) if layers is None else list(layers)
        pad_id = int(self.tokenizer.pad_token_id)
        results: List[Dict[int, torch.Tensor]] = [None] * len(sequences)

        for start in range(0, len(sequences), max(1, batch_size)):
            chunk = sequences[start : start + max(1, batch_size)]
            max_len = max(len(seq) for seq in chunk)
            input_ids = torch.full(
                (len(chunk), max_len), pad_id, dtype=torch.long, device=self.device
            )
            attention_mask = torch.zeros(
                (len(chunk), max_len), dtype=torch.long, device=self.device
            )
            for row, seq in enumerate(chunk):
                length = len(seq)
                input_ids[row, :length] = torch.tensor(seq, dtype=torch.long)
                attention_mask[row, :length] = 1
            hidden = self.full_hidden(input_ids, attention_mask, layers=wanted)
            for row, seq in enumerate(chunk):
                results[start + row] = {
                    layer: hidden[layer][row, : len(seq)].clone() for layer in wanted
                }
        return results

    def truncated_hidden(
        self,
        input_ids: torch.Tensor,
        layer: int,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        full_attn_mask: bool = True,
    ) -> torch.Tensor:
        model = self._require_model()
        text_model = self.text_model
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if not 0 <= int(layer) < self.num_layers:
            raise ValueError(f"layer must be in [0, {self.num_layers}), got {layer}")

        from ..fast_attn import expand_kv_attention as _expand_kv

        def _run() -> torch.Tensor:
            with torch.inference_mode():
                embeds = text_model.embed_tokens(input_ids)
                batch, seq_len = embeds.shape[0], embeds.shape[1]
                position_ids = (
                    torch.arange(seq_len, device=embeds.device)
                    .view(1, 1, -1)
                    .expand(4, batch, -1)
                )
                text_position_ids = position_ids[0]

                from transformers.masking_utils import create_causal_mask

                causal_mask = None
                if full_attn_mask:
                    causal_mask = create_causal_mask(
                        config=text_model.config,
                        inputs_embeds=embeds,
                        attention_mask=attention_mask,
                        past_key_values=None,
                        position_ids=text_position_ids,
                    )
                linear_mask = (
                    text_model._update_linear_attn_mask(attention_mask, None)
                    if hasattr(text_model, "_update_linear_attn_mask")
                    else causal_mask
                )
                layer_types = getattr(text_model.config, "layer_types", None)
                position_embeddings = text_model.rotary_emb(embeds, position_ids[1:])

                hidden = embeds
                for index in range(int(layer) + 1):
                    decoder_layer = text_model.layers[index]
                    is_linear = (
                        layer_types is not None and layer_types[index] == "linear_attention"
                    )
                    layer_mask = linear_mask if is_linear else causal_mask
                    hidden = decoder_layer(
                        hidden,
                        position_embeddings=position_embeddings,
                        attention_mask=layer_mask,
                        position_ids=text_position_ids,
                        past_key_values=None,
                        use_cache=False,
                    )

                if int(layer) == self.num_layers - 1:
                    hidden = text_model.norm(hidden)
                return hidden.detach().to("cpu", torch.float32)

        if self.expand_kv_in_generate:
            with _expand_kv(self.text_model.config):
                return _run()
        return _run()

    @property
    def pad_token_id(self) -> int:
        tok = self.tokenizer
        if tok is None:
            raise RuntimeError("tokenizer 未加载，无法确定 pad_token_id")
        pid = getattr(tok, "pad_token_id", None)
        if pid is None:
            pid = getattr(tok, "eos_token_id", None)
        if pid is None:
            raise RuntimeError("tokenizer 既没有 pad_token_id 也没有 eos_token_id")
        return int(pid)

    def truncated_hidden_attn(
        self,
        input_ids: torch.Tensor,
        layer: int,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        spans: Sequence[Tuple[int, int]],
        attn_cfg: Any,
        row_block: int = 512,
    ) -> Tuple[torch.Tensor, Any]:
        return capture_chunked(
            self, input_ids, int(layer), spans, attn_cfg, int(row_block),
            attention_mask,
        )

    def decode_attention_mass(
        self,
        *,
        total_ids: Sequence[int],
        layer: int,
        spans: Sequence[Tuple[int, int]] = (),
        slots: Sequence[int] = (),
        prompt_len: int = 0,
        gen_len: int = 0,
        row_block: int = 512,
    ) -> Dict[int, float]:
        total = [int(x) for x in total_ids]
        n_total = len(total)
        spans = [(int(a), int(b)) for a, b in spans]
        slots = [int(s) for s in slots]
        if not total:
            return {}
        if len(spans) != len(slots):
            raise ValueError(f"段数 {len(spans)} 与槽数 {len(slots)} 不一致")
        if int(prompt_len) + int(gen_len) != n_total:
            raise ValueError(
                f"prompt_len({prompt_len}) + gen_len({gen_len}) != len(total_ids)"
                f"({n_total})"
            )
        for a, b in spans:
            if not 0 <= a < b <= n_total:
                raise ValueError(f"非法 span ({a},{b})，len(total_ids)={n_total}")

        from ..chunked_attn import chunked_attention

        aggregator = _DecodeMassAggregator(spans, n_total, int(prompt_len))
        with chunked_attention(
            self.text_model.config,
            int(layer),
            aggregator,
            block=int(row_block),
        ):
            self.truncated_hidden(torch.tensor(total), int(layer), full_attn_mask=False)
        return aggregator.result(slots)

    def generate_ids(
        self,
        prompts: Union[str, Sequence[str], Sequence[Sequence[int]]],
        *,
        max_new_tokens: int = 8,
        do_sample: bool = False,
        batch_size: int = 8,
        prompts_are_ids: bool = False,
        expand_kv: Optional[bool] = None,
        prefill_chunk: Optional[int] = None,
    ) -> Tuple[List[str], List[List[int]]]:
        model = self._require_model()
        mode_is_ids = bool(prompts_are_ids)
        outputs: List[str] = []
        id_outputs: List[List[int]] = []
        if mode_is_ids:

            p = list(prompts)
            if p and not isinstance(p[0], (list, tuple)):
                rows: List[Any] = [p]
            else:
                rows = [list(int(x) for x in r) for r in p]
        else:
            single = isinstance(prompts, str)
            rows = [prompts] if single else list(prompts)
        for start in range(0, len(rows), max(1, batch_size)):
            chunk = rows[start : start + max(1, batch_size)]
            if mode_is_ids:
                bs = len(chunk)
                maxlen = max(len(r) for r in chunk)
                pad_id = int(getattr(self.tokenizer, "pad_token_id", 0) or 0)
                input_ids = torch.full(
                    (bs, maxlen), pad_id, dtype=torch.long, device=self.device
                )
                attn = torch.zeros((bs, maxlen), dtype=torch.long, device=self.device)
                for i, r in enumerate(chunk):
                    input_ids[i, : len(r)] = torch.tensor(
                        r, dtype=torch.long, device=self.device
                    )
                    attn[i, : len(r)] = 1
                batch = {"input_ids": input_ids, "attention_mask": attn}
            else:
                batch = self.tokenize_batch(chunk)
            _kv_cache = None
            if prefill_chunk and int(prefill_chunk) > 0:
                _ids0 = batch["input_ids"]
                if int(_ids0.shape[1]) > int(prefill_chunk):
                    _kv_cache = self._prefill_in_chunks(
                        _ids0, batch["attention_mask"], int(prefill_chunk)
                    )
                    batch = {
                        "input_ids": _ids0[:, -1:].contiguous(),
                        "attention_mask": torch.ones(
                            (int(_ids0.shape[0]), int(_ids0.shape[1])),
                            dtype=torch.long,
                            device=_ids0.device,
                        ),
                    }
            _kv_kwargs: Dict[str, Any] = (
                {} if _kv_cache is None else {"past_key_values": _kv_cache}
            )
            use_expand = (self.expand_kv_in_generate if expand_kv is None
                          else bool(expand_kv))

            _eos = getattr(self.tokenizer, "eos_token_id", None)
            _pad = getattr(self.tokenizer, "pad_token_id", None)
            _stop: Dict[str, Any] = {"use_cache": True}
            if _eos is not None:
                _stop["eos_token_id"] = int(_eos)
            if _pad is not None:
                _stop["pad_token_id"] = int(_pad)
            with torch.inference_mode():
                if use_expand:
                    from ..fast_attn import expand_kv_attention
                    with expand_kv_attention(self.text_model.config):
                        generated = model.generate(
                            **batch,
                            max_new_tokens=int(max_new_tokens),
                            do_sample=bool(do_sample),
                            return_dict_in_generate=True,
                            **_stop,
                            **_kv_kwargs,
                        )
                else:
                    generated = model.generate(
                        **batch,
                        max_new_tokens=int(max_new_tokens),
                        do_sample=bool(do_sample),
                        return_dict_in_generate=True,
                        **_stop,
                        **_kv_kwargs,
                    )
            generated = generated.sequences
            new_tokens = generated[:, batch["input_ids"].shape[1] :]
            outputs.extend(
                self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            )
            id_outputs.extend([[int(t) for t in row] for row in new_tokens.tolist()])
        if len(rows) == 1:

            return outputs[:1], id_outputs[:1]
        return outputs, id_outputs

    def generate(
        self,
        prompts: Union[str, Sequence[str]],
        *,
        max_new_tokens: int = 8,
        do_sample: bool = False,
        batch_size: int = 8,
    ) -> List[str]:
        return self.generate_ids(
            prompts,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            batch_size=batch_size,
        )[0]

    def chat_text(
        self,
        content: str,
        *,
        system: Optional[str] = None,
        enable_thinking: bool = False,
    ) -> str:
        self.load()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    def chat_ids(
        self,
        content: str,
        *,
        system: Optional[str] = None,
        enable_thinking: bool = False,
    ) -> List[int]:
        return self.tokenize(
            self.chat_text(content, system=system, enable_thinking=enable_thinking)
        )

    def _prefill_in_chunks(self, input_ids, attention_mask, chunk: int):
        text = self.text_model
        ids = input_ids[:, :-1]
        n = int(ids.shape[1])
        cache = None
        seen = 0
        while seen < n:
            cur = ids[:, seen : seen + int(chunk)]
            out = text(
                input_ids=cur,
                attention_mask=attention_mask[:, : seen + int(cur.shape[1])],
                past_key_values=cache,
                use_cache=True,
            )
            cache = out.past_key_values
            seen += int(cur.shape[1])
            del out
        return cache

    def chat_generate_ids(
        self,
        user_prompts: Union[str, Sequence[str]],
        *,
        system: Optional[str] = None,
        max_new_tokens: int = 16,
        do_sample: bool = False,
        batch_size: int = 8,
        enable_thinking: bool = False,
        prefill_chunk: Optional[int] = None,
    ) -> Tuple[List[str], List[List[int]]]:
        self.load()
        single = isinstance(user_prompts, str)
        contents = [user_prompts] if single else list(user_prompts)
        texts = [
            self.chat_text(c, system=system, enable_thinking=enable_thinking)
            for c in contents
        ]
        return self.generate_ids(
            texts,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            batch_size=batch_size,
            prefill_chunk=prefill_chunk,
        )

    def chat_frame_ids(
        self, *, system: Optional[str] = None, enable_thinking: bool = False
    ) -> Tuple[List[int], List[int], List[int]]:
        self.load()
        sent = "\x00__SYNAPTICA_CONTENT__\x00"
        text = self.chat_text(sent, system=system, enable_thinking=enable_thinking)
        i = text.index(sent)
        pre = [int(t) for t in self.tokenize(text[:i])]
        suf = [int(t) for t in self.tokenize(text[i + len(sent) :])]
        gap = [int(t) for t in self.tokenize("\n\n")] or []
        return pre, suf, gap

    def chat_generate_ids_from_ids(
        self,
        prompt_ids: Sequence[int],
        *,
        max_new_tokens: int = 16,
        do_sample: bool = False,
        batch_size: int = 8,
    ) -> Tuple[List[str], List[List[int]]]:
        return self.generate_ids(
            [list(int(x) for x in prompt_ids)],
            max_new_tokens=int(max_new_tokens),
            do_sample=bool(do_sample),
            batch_size=int(batch_size),
            prompts_are_ids=True,
        )

    def chat_generate(
        self,
        user_prompts: Union[str, Sequence[str]],
        *,
        system: Optional[str] = None,
        max_new_tokens: int = 16,
        do_sample: bool = False,
        batch_size: int = 8,
        enable_thinking: bool = False,
        prefill_chunk: Optional[int] = None,
    ) -> List[str]:
        return self.chat_generate_ids(
            user_prompts,
            system=system,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            batch_size=batch_size,
            enable_thinking=enable_thinking,
            prefill_chunk=prefill_chunk,
        )[0]

    @staticmethod
    def pool_span(
        hidden: torch.Tensor, span: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        if hidden.dim() == 3:
            if hidden.shape[0] != 1:
                raise ValueError("pool_span expects (S, H) or (1, S, H)")
            hidden = hidden[0]
        if span is None:
            return hidden.mean(dim=0)
        start, end = int(span[0]), int(span[1])
        if not 0 <= start < end <= hidden.shape[0]:
            raise ValueError(f"invalid span {span} for sequence length {hidden.shape[0]}")
        return hidden[start:end].mean(dim=0)

    def memory_gib(self) -> float:
        if not torch.cuda.is_available():
            return float("nan")
        return torch.cuda.memory_allocated() / 2**30

    def __repr__(self) -> str:
        loaded = self.model is not None
        return (
            f"{type(self).__name__}(path={self.model_path!r}, device={self.device!r}, "
            f"dtype={self.dtype}, loaded={loaded})"
        )
