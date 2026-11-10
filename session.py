from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

import torch

from .ann.attention_graph import AttentionGraph, AttentionGraphConfig
from .ann.encoder import SynapticaEncoder, mean_pool
from .ann.gen_attn_reward import GenAttnReward, GenAttnRewardConfig
from .ann.llm.capture import capture_hidden_only
from .ann.seq_match import SeqMatcher
from .snn.memory import SynapticaMemory, WriteStats
from .types import Span

__all__ = [
    "Synaptica",
    "SynapticaConfig",
    "BlockResult",
    "AskResult",
    "RunResult",
]

_TREE_STAGE_RANK = {"carrier": 1, "q1": 0}

@dataclass
class SynapticaConfig:

    block_tokens: int = 16384

    segment_tokens: int = 64

    layer: int = 19

    attn_block: int = 256

    num_neurons: int = 8192

    code_dim: int = 4096

    pulse_seed: int = 20240315

    seq_theta: float = 0.4

    seq_steps: int = 6

    graph_pos_debias: float = 1.0

    graph_out_degree: int = 3

    graph_max_in_degree: int = 4

    graph_gate_keep: float = 0.5

    ref_anchor_k: int = 5

    recall_per_carrier_k: int = 1

    gen_max_new_tokens: int = 512

    device: str = "cpu"

    def __post_init__(self) -> None:
        if int(self.block_tokens) <= 0:
            raise ValueError("block_tokens 必须是正整数")
        if int(self.segment_tokens) <= 0:
            raise ValueError("segment_tokens 必须是正整数")
        if int(self.layer) < 0:
            raise ValueError(f"layer 必须 >= 0，收到 {self.layer}")
        if int(self.graph_out_degree) < 0:
            raise ValueError(f"graph_out_degree 必须 >= 0，收到 {self.graph_out_degree}")
        if not 0.0 < float(self.graph_gate_keep) <= 1.0:
            raise ValueError(f"graph_gate_keep 必须在 (0, 1]，收到 {self.graph_gate_keep}")
        if not 0.0 <= float(self.graph_pos_debias) <= 1.0:
            raise ValueError(f"graph_pos_debias 必须在 [0, 1]，收到 {self.graph_pos_debias}")

@dataclass
class BlockResult:

    index: int = 0
    n_tokens: int = 0
    spans: List[Span] = field(default_factory=list)
    slots: List[int] = field(default_factory=list)
    n_recall_queries: int = 0

    recalled: List[int] = field(default_factory=list)
    prefix_tokens: int = 0
    prefix_dropped: int = 0
    forward_count: int = 0
    write_stats: List[WriteStats] = field(default_factory=list)
    edges_new: int = 0
    edges_cross: int = 0
    edges_after: int = 0
    n_neg_init: int = 0

@dataclass
class AskResult:

    block: Optional[BlockResult] = None
    question_span: Optional[Span] = None
    recalled: List[int] = field(default_factory=list)

    question_slots: List[int] = field(default_factory=list)
    context: str = ""
    prompt_tokens: int = 0
    generated: str = ""
    gen_tokens: int = 0
    attention_cut: float = 0.0
    n_rewarded: int = 0
    n_penalized: int = 0
    n_created: int = 0
    n_created_intra: int = 0
    n_neg_new: int = 0
    n_dropped: int = 0

@dataclass
class RunResult:

    blocks: List["BlockResult"] = field(default_factory=list)
    ask: Optional["AskResult"] = None
    argv_mode: str = "write"

    leftover: List[int] = field(default_factory=list)

    n_pad_tail: int = 0

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    @property
    def forward_count(self) -> int:
        return sum(int(b.forward_count) for b in self.blocks)

    @property
    def edges_cross(self) -> int:
        return sum(int(b.edges_cross) for b in self.blocks)

    @property
    def neg_edges_init(self) -> int:
        return sum(int(b.n_neg_init) for b in self.blocks)

    def __len__(self) -> int:
        return len(self.blocks)

    def __iter__(self):
        return iter(self.blocks)

    def __getitem__(self, i):
        return self.blocks[i]

class Synaptica:

    forward_hook: Optional[Callable[[], None]] = None
    forward_hook_count: int = 0

    def __init__(
        self,
        *,
        memory: SynapticaMemory,
        backend: Any,
        cfg: Optional[SynapticaConfig] = None,
        seed: int = 0,
    ) -> None:
        self.cfg = cfg or SynapticaConfig()
        self.memory = memory
        self.backend = backend
        self._install_token_len_fn()
        self.seed = int(seed)
        self.layer = int(self.cfg.layer)
        self.forward_count = 0
        self.seq_matcher = SeqMatcher(
            theta=float(self.cfg.seq_theta))
        if backend is None:

            raise ValueError("backend 不能为空")
        if not callable(getattr(backend, "embeddings_ids", None)):

            raise ValueError(
                f"backend={type(backend).__name__} 缺少 embeddings_ids(token_ids) 方法"
            )
        self.gen_attn_reward = GenAttnReward(
            self.backend,
            GenAttnRewardConfig(
                layer=int(self.cfg.layer),
                cut_q=float(self.memory.lateral_cfg.attn_q),
                row_block=int(self.cfg.attn_block),
            ),
        )

        self.encoder = SynapticaEncoder(
            hidden_size=int(self.cfg.code_dim),
            num_neurons=int(self.cfg.num_neurons),
            segment_tokens=int(self.cfg.segment_tokens),
            seed=int(self.cfg.pulse_seed),
        )

        memory.bind_encoder(self._segment_encoder, self.encoder.segment_shape())

        memory.pad_token_id = getattr(self.backend, "pad_token_id", None)
        _tok = getattr(self.backend, "tokenizer", None)
        memory.special_token_ids = set(
            int(x) for x in (getattr(_tok, "all_special_ids", None) or [])
        )
        _input_size = int(getattr(self.encoder, "num_neurons", self.cfg.num_neurons))
        if int(getattr(memory, "input_size", _input_size)) != _input_size:
            raise ValueError(
                f"memory.input_size={getattr(memory, 'input_size', None)} 与编码器输出"
                f" {_input_size} 不一致"
            )

        self.absolute_threshold = True

    @property
    def _device(self) -> torch.device:
        return torch.device(self.cfg.device)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "memory": self.memory.state_dict(),

            "encoder": self.encoder.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> "Synaptica":
        enc_state = state["encoder"]
        want = type(self.encoder).config_fingerprint_from(self.encoder.config_dict())
        got = type(self.encoder).config_fingerprint_from(enc_state["config"])
        if got != want:
            raise ValueError(
                f"存档的编码器配置与当前不一致："
                f"{enc_state['config']} vs {self.encoder.config_dict()}"
            )
        self.encoder.load_state_dict(enc_state)
        self.memory.load_state_dict(state["memory"])

        self.memory.bind_encoder(self._segment_encoder, self.encoder.segment_shape())
        return self

    def _install_token_len_fn(self) -> None:
        tk = getattr(self.backend, "tokenizer", None)
        if tk is None or self.memory is None:
            return
        _cache: Dict[int, int] = {}
        def _fn(ids):
            out = []
            for i in ids:
                v = _cache.get(i)
                if v is None:
                    v = len(str(tk.decode([int(i)])).strip())
                    _cache[i] = v
                out.append(v)
            return out
        self.memory.token_len_fn = _fn
    def _segment_encoder(self, token_ids: Sequence[int]) -> Any:
        ids = [int(x) for x in token_ids]
        e = self.backend.embeddings_ids(ids)
        e = torch.as_tensor(e, dtype=torch.float32).to(self._device)
        if e.dim() != 2 or int(e.shape[0]) != len(ids):
            raise ValueError(f"embeddings_ids 返回 {tuple(e.shape)}，应为 (T, H)，T={len(ids)}")

        pat = self.encoder.encode_tokens(e).reshape(-1)
        return e.mean(dim=0), pat

    def _decode_ids(self, ids: Sequence[int]) -> str:
        fn = getattr(self.backend, "decode", None)
        if callable(fn):
            return str(fn(list(ids)))
        tok = getattr(self.backend, "tokenizer", None)
        if tok is not None and hasattr(tok, "decode"):
            return str(tok.decode([int(x) for x in ids], skip_special_tokens=True))
        raise RuntimeError("backend 缺少 decode(ids)，也没有带 decode 的 tokenizer")

    @torch.no_grad()
    def _segment_spans(
        self, n_tokens: int, token_embeddings: Optional[torch.Tensor] = None
    ) -> List[Span]:
        if n_tokens <= 0:
            raise ValueError("空块无法分段")
        from .segmenter import BlockSegmenter

        seg = BlockSegmenter(int(self.cfg.segment_tokens))
        embeddings = (
            token_embeddings
            if token_embeddings is not None
            else torch.zeros(int(n_tokens), 1)
        )
        spans = [
            (int(a), int(b))
            for a, b in seg.segment(
                embeddings, torch.ones(int(n_tokens)), is_final=True
            )
        ]
        if not spans:
            raise RuntimeError("分段器对非空块返回了空列表")

        if spans[0][0] != 0 or spans[-1][1] != n_tokens:
            raise RuntimeError(f"分段未覆盖 [0,{n_tokens})：{spans[:2]}…{spans[-2:]}")
        for (a0, b0), (a1, _b1) in zip(spans, spans[1:]):
            if a1 != b0:
                raise RuntimeError(f"分段不连续：段 {a0},{b0} 之后是 {a1}")
        return spans

    @torch.no_grad()
    def _embed_block(self, ids: Sequence[int], spans: Sequence[Span]) -> Dict[str, Any]:
        tok = self.backend.embeddings_ids(list(ids))
        if tok.dim() != 2:
            raise ValueError(f"embeddings_ids 返回 {tuple(tok.shape)}，应为 (T, H)")
        tok = tok.to(self._device).float()
        if int(tok.shape[0]) != len(ids):
            raise ValueError(f"嵌入序列长度 {tok.shape[0]} 与 token 数 {len(ids)} 不一致")
        seg_vecs: List[torch.Tensor] = []
        seg_toks: List[torch.Tensor] = []
        for a, b in spans:
            piece = tok[a:b]
            seg_toks.append(piece)
            seg_vecs.append(mean_pool(piece.unsqueeze(0))[0])
        vecs = (
            torch.stack(seg_vecs, dim=0) if seg_vecs else torch.zeros(0, tok.shape[-1])
        )

        cap = int(self.encoder.segment_tokens)
        pats_list: List[torch.Tensor] = []
        for i, (a, b) in enumerate(spans):
            n_tok = int(b) - int(a)
            if n_tok > cap:
                raise ValueError(f"第 {i} 段有 {n_tok} 个 token，超过 segment_tokens={cap}")
            p = self.encoder.encode_tokens(seg_toks[i])
            pats_list.append(p.reshape(-1).float())
        pats = (
            torch.stack(pats_list, dim=0)
            if pats_list
            else torch.zeros(0, int(self.encoder.num_neurons), device=self._device)
        )
        self._last_pattern_len = int(pats.shape[1]) if pats.numel() else 0
        return {
            "token_embeddings": tok,
            "segment_vectors": vecs,
            "segment_tokens": seg_toks,
            "patterns": pats,
        }

    def _ref_recall_block(
        self,
        spots: Sequence[Dict[str, Any]],
        prior_slots: Sequence[int],
        slot_upper_bound: Optional[int] = None,
    ) -> Tuple[List[int], int, List[Dict[str, Any]]]:
        if not prior_slots:
            return [], 0, []
        if int(self.memory.num_laterals_edges) == 0:

            raise RuntimeError(f"{len(prior_slots)} 个已写入段在 L 中没有任何边")
        matcher = self.seq_matcher
        scored: Dict[int, float] = {}
        details: List[Dict[str, Any]] = []
        n_q = 0

        _prior_set = {int(x) for x in prior_slots}

        for sp in spots:
            n_q += 1
            _hint: Optional[int] = None
            try:

                _pat = sp.get("tokens")
                _s = self.memory.find_by_pattern(_pat) if _pat is not None else None
                if _s is not None and int(_s) in _prior_set:
                    _hint = int(_s)
            except Exception:
                _hint = None

            res = self.memory.ref_recall(
                sp["tokens"],
                matcher=matcher,
                scope=prior_slots,
                steps=int(self.cfg.seq_steps),
                query_slot_hint=_hint,

                slot_upper_bound=slot_upper_bound,

                gate_keep=float(self.cfg.graph_gate_keep),

                query_valid=sp.get("n_valid"),

                ref_anchor_k=int(self.cfg.ref_anchor_k),
            )
            details.append(res)
            for slot, sc in (res.get("scores") or {}).items():
                scored[int(slot)] = max(scored.get(int(slot), float("-inf")), float(sc))
            for slot in res.get("recalled") or []:
                scored.setdefault(int(slot), 0.0)
        return sorted(scored), n_q, details

    @torch.no_grad()
    def _tree_parents(self, edges, S):
        rank = _TREE_STAGE_RANK
        par = {}
        for e in edges or ():
            try:
                src, dest, w = int(e[0]), int(e[1]), float(e[2])
                stage = str(e[3]) if len(e) > 3 else "carrier"
            except (TypeError, ValueError, IndexError):
                continue
            if src not in S or dest not in S or src == dest:
                continue
            r = rank.get(stage, 0)
            cur = par.get(dest)
            if (cur is None or r > cur[0] or (r == cur[0] and w > cur[1])
                    or (r == cur[0] and w == cur[1] and src < cur[2])):
                par[dest] = (r, w, src, stage)
        return {k: (v[2], v[3]) for k, v in par.items()}

    def _tree_order(self, recalled, edges):
        S = {int(x) for x in recalled}
        pmap = self._tree_parents(edges, S)
        parent = dict(pmap)

        for start in sorted(S):
            chain, x = [], start
            while x in parent:
                if x in chain:
                    cyc = chain[chain.index(x):]
                    cut = next((c for c in cyc if parent[c][1] == "q1"), None)
                    if cut is None:
                        cut = max(cyc)
                    del parent[cut]
                    break
                chain.append(x)
                x = parent[x][0]

        depth_of: Dict[int, int] = {}

        def _dep(x):
            if x in depth_of:
                return depth_of[x]
            chain, y = [], x
            while y in parent and y not in depth_of:
                chain.append(y)
                y = parent[y][0]
            base = depth_of.get(y, 0)
            for n in reversed(chain):
                base += 1
                depth_of[n] = base
            depth_of.setdefault(x, base)
            return depth_of[x]

        for x in sorted(S):
            _dep(x)
        children: Dict[int, List[int]] = {}
        roots: List[int] = []
        for s_ in sorted(S):
            if s_ in parent:
                children.setdefault(parent[s_][0], []).append(s_)
            else:
                roots.append(s_)
        order, seen = [], set()

        def _walk(node, d):
            if node in seen:
                return
            seen.add(node)
            order.append(node)
            depth_of[node] = d
            for c in sorted(children.get(node, [])):
                _walk(c, d + 1)

        for r_ in roots:
            _walk(r_, depth_of.get(r_, 0))
        for s_ in sorted(S):
            if s_ not in seen:
                _walk(s_, depth_of.get(s_, 0))
        return order, depth_of

    def _tree_merge_adjacent(self, order, depth_of, parent):

        runs = []
        for x in sorted(int(v) for v in depth_of):
            if runs and x == runs[-1][-1] + 1:
                runs[-1].append(x)
            else:
                runs.append([x])

        groups = []
        for run in runs:

            cur = [run[0]]
            for x in run[1:]:
                px = parent.get(x)
                prev = cur[-1]
                same_parent = (px is not None and parent.get(prev) is not None
                               and px == parent.get(prev))
                parent_child = ((px is not None and px == prev)
                                or (parent.get(prev) is not None
                                    and parent.get(prev) == x))
                if same_parent or parent_child:
                    cur.append(x)
                else:
                    groups.append(cur)
                    cur = [x]
            groups.append(cur)

        pos = {int(x): i for i, x in enumerate(order)}
        groups.sort(key=lambda g: min(pos.get(int(x), 10 ** 9) for x in g))
        order2, depth2 = [], {}
        for g in groups:
            d = min(int(depth_of.get(int(x), 0)) for x in g)
            for x in g:
                order2.append(int(x))
                depth2[int(x)] = d
        return order2, depth2, groups

    def _assemble_prefix(
        self, recalled: Sequence[int], edges: Optional[Sequence[Sequence[float]]] = None
    ) -> Dict[str, Any]:
        recalled, _depth_of = self._tree_order(recalled, edges)

        _par = self._tree_parents(edges, set(int(x) for x in recalled))
        recalled, _depth_of, _groups = self._tree_merge_adjacent(
            recalled, _depth_of, _par)
        _indent_cache: Dict[int, List[int]] = {}
        prefix_ids: List[int] = []
        prefix_spans: List[Span] = []
        used: List[int] = []
        dropped = 0
        for slot in recalled:

            payload = self.memory.segment_token_ids(slot)
            if payload is None:

                continue
            if not isinstance(payload, list):

                raise RuntimeError(
                    f"槽 {int(slot)} 的内容类型为 {type(payload).__name__}，应为 list[int]"
                )
            if not payload:

                raise RuntimeError(f"槽 {int(slot)} 的 payload 为空")
            sub = [int(x) for x in payload]
            _d = int(_depth_of.get(int(slot), 0))
            _ind = _indent_cache.get(_d)
            if _ind is None:

                _ind = [int(x) for x in self.backend.tokenize("\n" + "  " * _d)]
                _indent_cache[_d] = _ind

            a = len(prefix_ids)
            prefix_ids.extend(_ind)
            prefix_ids.extend(sub)
            prefix_spans.append((a, len(prefix_ids)))
            used.append(int(slot))
        return {
            "ids": prefix_ids,
            "spans": prefix_spans,
            "used": used,
            "dropped": dropped,
        }

    @torch.no_grad()
    def _forward(
        self, ids: Sequence[int], spans: Sequence[Span], *, capture_attention: bool
    ) -> Tuple[torch.Tensor, Optional[AttentionGraph]]:
        self.forward_count += 1
        if type(self).forward_hook is not None:
            type(self).forward_hook()
        ids_t = torch.tensor(list(ids), dtype=torch.long)

        if not capture_attention:
            h = capture_hidden_only(self.backend, ids_t, int(self.layer))
            return h, None

        cfg = AttentionGraphConfig(
            pos_debias=float(self.cfg.graph_pos_debias),
            out_degree=int(self.cfg.graph_out_degree),
            max_in_degree=int(self.cfg.graph_max_in_degree),
        )
        out = self.backend.truncated_hidden_attn(
            ids_t,
            int(self.layer),
            attention_mask=None,
            spans=list(spans),
            attn_cfg=cfg,
            row_block=int(self.cfg.attn_block),
        )
        if not isinstance(out, (tuple, list)) or len(out) < 2:
            raise RuntimeError(f"truncated_hidden_attn 应返回 (隐状态, 段级图)，收到 {type(out).__name__}")
        h, graph = out[0], out[1]
        return h, graph

    @torch.no_grad()
    def _init_edges(
        self,
        graph: AttentionGraph,
        spans: Sequence[Span],
        slot_index: Sequence[int],
        prefix_len: int,
    ) -> Tuple[int, int, int]:
        slot_index = [int(s) for s in slot_index]
        if len(slot_index) != len(spans):
            raise ValueError(f"slot_index 长度 {len(slot_index)} 与段数 {len(spans)} 不一致")

        n_new, n_cross = self.memory.attention_marge_laterals(
            graph, slot_index=slot_index, prefix_len=int(prefix_len)
        )
        try:
            _u: dict = {}

            _rp0 = (getattr(graph, "meta", None) or {}).get("raw_pairs") or {}
            for (_a0, _t0), _w0 in _rp0.items():
                _g = int(slot_index[int(_t0)])
                _wf = float(_w0)
                if _wf > _u.get(_g, 0.0):
                    _u[_g] = _wf

            if _u:
                self.memory.note_attention_u(_u)
        except Exception:
            pass

        n_neg = 0
        if float(self.memory.lateral_cfg.neg_gain) > 0.0:
            _rows: Dict[int, List[Tuple[int, float]]] = {}

            _rp = (getattr(graph, "meta", None) or {}).get("raw_pairs") or {}
            for (_a, _t), _w in _rp.items():
                _rows.setdefault(int(_a), []).append((int(_t), float(_w)))

            _gg = float(self.memory.lateral_cfg.neg_gain)
            _gd: List[int] = []
            _gs: List[int] = []
            _gv: List[float] = []

            for _a, _lst in _rows.items():
                if len(_lst) < 2:
                    continue
                _lst.sort(key=lambda t: -t[1])
                _w1 = float(_lst[0][1])
                if _w1 <= 0.0:
                    continue
                _ga = int(slot_index[int(_a)])
                for _t, _w in _lst[1:]:
                    _dif = (_w1 - float(_w)) / _w1
                    if _dif <= 0.0:
                        continue
                    _gt = int(slot_index[int(_t)])
                    _gd.extend([_gt, _ga])
                    _gs.extend([_ga, _gt])
                    _gv.extend([-_gg * _dif, -_gg * _dif])
            if _gd:
                _lat = self.memory.node.lateral
                _lat._neg_merge_add(
                    torch.as_tensor(_gd, dtype=torch.long, device=_lat.device),
                    torch.as_tensor(_gs, dtype=torch.long, device=_lat.device),
                    torch.as_tensor(_gv, dtype=torch.float32, device=_lat.device),
                )
            n_neg = len(_gd)
        return n_new, n_cross, n_neg

    @torch.no_grad()
    def _decode_learn(
        self,
        ask: AskResult,
        *,
        prompt_ids: Sequence[int],
        gen_ids: Sequence[int],
        prefix_spans: Sequence[Span],
        prefix_slots: Sequence[int],
        carriers: Sequence[int],
        block_slots: Sequence[int] = (),
    ) -> None:
        if not gen_ids:
            return
        slots = [int(x) for x in prefix_slots]
        if len(slots) != len(prefix_spans):
            raise RuntimeError(f"前缀段数 {len(prefix_spans)} 与槽号数 {len(slots)} 不一致")

        carriers = sorted(
            {int(c) for c in carriers}
        )

        block_spans = list(getattr(ask, "block_spans", None) or [])
        b_slots = [int(s) for s in block_slots]
        if len(b_slots) != len(block_spans):
            raise RuntimeError(f"块内段数 {len(block_spans)} 与槽号数 {len(b_slots)} 不一致")
        items: List[Tuple[int, Span]] = [
            (int(slots[i]), (int(sp[0]), int(sp[1])) if sp is not None else None)
            for i, sp in enumerate(prefix_spans)
        ] + [
            (int(b_slots[i]), (int(sp[0]), int(sp[1])) if sp is not None else None)
            for i, sp in enumerate(block_spans)
        ]

        query_slots = list(ask.question_slots or [])
        sc = self.gen_attn_reward(
            prompt_ids=list(prompt_ids),
            gen_ids=list(gen_ids),
            items=items,
            exclude=query_slots,
        )
        ask.attention_cut = float(sc.cut)
        ask.n_rewarded = int(sc.n_attended)
        ask.n_penalized = int(len(sc.u) - sc.n_attended)

        u_slot: Dict[int, float] = {int(k): float(v) for k, v in sc.u.items()}

        attended = [s for s, v in u_slot.items() if v >= float(sc.cut)]
        unattended = [s for s, v in u_slot.items() if v < float(sc.cut)]
        ask.n_rewarded = len(attended)
        ask.n_penalized = len(unattended)
        if not u_slot:
            return

        if attended:
            att_local = [s for s in b_slots if int(s) in set(attended)]
            d_b, s_b = [], []
            for q_slot in b_slots:
                for ref_slot in att_local:
                    if int(ref_slot) == int(q_slot):
                        continue
                    if int(ref_slot) < int(q_slot):
                        d_b.append(int(q_slot))
                        s_b.append(int(ref_slot))
            if d_b:
                n_blk = int(
                    self.memory.add_laterals(
                        d_b, s_b, [float(self.memory.lateral_cfg.alpha)] * len(d_b)
                    )
                )
                ask.n_created_intra = int(n_blk)

        if attended and carriers:
            q_slots = [int(s) for s in (ask.question_slots or [])]
            if q_slots:
                ex = set(
                    zip(
                        self.memory.laterals_index[0].tolist(),
                        self.memory.laterals_index[1].tolist(),
                    )
                )
                d_new: List[int] = []
                s_new: List[int] = []
                v_new: List[float] = []

                def _add(dd: int, ss: int) -> None:
                    if dd == ss or (dd, ss) in ex:
                        return
                    ex.add((dd, ss))
                    d_new.append(int(dd))
                    s_new.append(int(ss))
                    v_new.append(float(self.memory.lateral_cfg.alpha))

                for a1 in carriers:
                    for q in q_slots:
                        _add(q, a1)

                for q in q_slots:
                    for q1 in attended:
                        _add(q1, q)
                if d_new:
                    ask.n_created = int(self.memory.add_laterals(d_new, s_new, v_new))

        if float(self.memory.lateral_cfg.neg_gain) > 0.0 and u_slot:
            ask.n_neg_new = int(
                self.memory.add_negative_laterals(
                    u_slot, exclude=list(ask.question_slots or []),

                    max_per_dest=int(self.cfg.graph_out_degree),
                )
            )

        edges = self.memory.walked_laterals()
        mod = {int(s): float(v) - float(sc.cut) for s, v in u_slot.items()}
        st = self.memory.modulate_laterals(edges, mod)
        ask.n_dropped = int(st.get("n_dropped", 0))

    @torch.no_grad()
    def _store_gen_segments(self, ask: AskResult, gen_ids: Sequence[int]) -> None:
        ids = [int(x) for x in gen_ids]
        if not ids:
            return
        seg = int(self.encoder.segment_tokens)
        n_full = len(ids) // seg
        if n_full <= 0:
            return
        covered = ids[: n_full * seg]

        spans = self._segment_spans(len(covered))
        emb = self._embed_block(covered, spans)
        pats = emb["patterns"]
        if int(pats.shape[1]) != int(self.memory.input_size):
            raise RuntimeError(
                f"生成段模式长度 {int(pats.shape[1])} 与 input_size "
                f"{int(self.memory.input_size)} 不一致"
            )
        new_slots = [
            int(self.memory.store(tuple(int(x) for x in covered[a:b]), None))
            for a, b in spans
        ]
        n_written = int(
            self.memory.write_batch([pats[i] for i in range(len(spans))], new_slots)
        )
        if n_written != len(new_slots):
            raise RuntimeError(f"生成段写入条数 {n_written} 与段数 {len(new_slots)} 不一致")

        anchor: Optional[int] = None
        if ask.question_slots:
            anchor = int(ask.question_slots[-1])
        else:
            blk_slots = [
                int(s) for s in (getattr(ask.block, "slots", None) or [])
            ]
            if blk_slots:
                anchor = int(blk_slots[-1])
        d_chain: List[int] = []
        s_chain: List[int] = []
        prev = anchor
        for slot in new_slots:
            if prev is not None and int(prev) != int(slot):
                d_chain.append(int(slot))
                s_chain.append(int(prev))
            prev = int(slot)
        if d_chain:
            self.memory.add_laterals(
                d_chain,
                s_chain,
                [float(self.memory.lateral_cfg.alpha)] * len(d_chain),
            )

    def run(
        self,
        ids: Sequence[int],
        *,
        mode: Literal["write", "qa"] = None,
        question: Optional[Sequence[int]] = None,
        question_span: Optional[Span] = None,
        system: Optional[str] = None,
    ) -> RunResult:
        return self._run_inner(
            ids, mode=mode, question=question, question_span=question_span,
            system=system,
        )

    def _run_inner(
        self,
        ids: Sequence[int],
        *,
        mode: Literal["write", "qa"] = None,
        question: Optional[Sequence[int]] = None,
        question_span: Optional[Span] = None,
        system: Optional[str] = None,
    ) -> RunResult:
        mode = str(mode)
        if mode not in ("write", "qa"):
            raise ValueError(f"mode 必须是 'write' | 'qa'，收到 {mode!r}")
        ids = [int(x) for x in ids]
        if not ids:
            raise ValueError("输入为空")

        blocks, leftover = self._split_blocks(ids)

        n_pad_tail = 0
        if leftover:
            pad_id = self.backend.pad_token_id
            n_pad_tail = (self.cfg.segment_tokens - len(leftover)
                          % self.cfg.segment_tokens) % self.cfg.segment_tokens
            tail = list(leftover) + [int(pad_id)] * n_pad_tail

            step = min(int(self.cfg.block_tokens), int(self.cfg.segment_tokens))
            step = max(step, int(self.cfg.segment_tokens))
            step = (step // int(self.cfg.segment_tokens)) * int(self.cfg.segment_tokens)
            blocks = list(blocks) + [
                tail[i:i + step] for i in range(0, len(tail), step)
            ]
            leftover = []

        out: List[BlockResult] = []
        ask_res: Optional[AskResult] = None

        _stored = [int(x) for x in self.memory.stored_slots()]
        _slot_floor = (max(_stored) + 1) if _stored else 0
        for k, blk in enumerate(blocks):

            is_qa_block = (mode == "qa") and (k == len(blocks) - 1)
            r, _pfx = self._run_block(
                k,
                blk,
                is_qa_block=is_qa_block,

                recall_slot_floor=int(_slot_floor),
            )
            out.append(r)
            if is_qa_block:
                ask_res = self._run_ask(
                    k,
                    blk,
                    r,
                    question=(
                        [int(x) for x in question] if question is not None else None
                    ),
                    question_span=question_span,
                    system=system,
                    prefix_slots=_pfx["slots"],
                    prefix_edges=_pfx["edges"],
                    carriers=_pfx["carriers"],
                )
        return RunResult(blocks=out, ask=ask_res, argv_mode=mode, leftover=leftover,
                         n_pad_tail=int(n_pad_tail))

    def write(self, ids: Sequence[int]) -> RunResult:
        return self.run(ids, mode="write")

    def ask(self, ids: Sequence[int], **kw: Any) -> AskResult:
        res = self.run(ids, mode="qa", **kw)
        if res.ask is None:
            raise RuntimeError("问答未产出结果")
        return res.ask

    def _split_blocks(
        self, ids: Sequence[int]
    ) -> Tuple[List[List[int]], List[int]]:
        W = max(1, int(self.cfg.block_tokens))
        S = max(1, int(self.cfg.segment_tokens))
        n = len(ids)

        size = (min(W, S) if W < S else (W // S) * S)

        seg_aligns = (size % S == 0)
        if not seg_aligns and W >= S:

            raise ValueError(
                f"block_tokens={W} 与 segment_tokens={S} 的组合得到块大小 {size}；"
                f"块大小必须是段长的整数倍，或小于段长"
            )
        out: List[List[int]] = []
        pos = 0
        while n - pos >= size:
            out.append(list(ids[pos : pos + size]))
            pos += size
        rest = list(ids[pos:])

        if rest and S < size:
            m = (len(rest) // S) * S
            if m > 0:
                out.append(rest[:m])
                rest = rest[m:]
                pos += m
        return out, rest

    def _run_block(
        self,
        index: int,
        blk: Sequence[int],
        *,
        is_qa_block: bool = False,
        recall_slot_floor: int = 0,
    ) -> Tuple[BlockResult, Dict[str, Any]]:
        r = BlockResult(index=int(index), n_tokens=len(blk))
        blk = list(blk)

        spans = self._segment_spans(len(blk))
        emb = self._embed_block(blk, spans)
        r.spans = list(spans)

        prior_slots = self.memory.stored_slots()

        _floor_eff = int(recall_slot_floor)
        if int(_floor_eff) > 0:
            prior_slots = [int(s) for s in prior_slots if int(s) < int(_floor_eff)]

        recalled: List[int] = []
        n_q = 0
        details: List[Dict[str, Any]] = []
        if is_qa_block:
            k = int(self.encoder.token_dims)

            _pid = getattr(self.backend, "pad_token_id", None)
            _pid = int(_pid) if _pid is not None else None

            spots = []
            for i in range(len(spans)):

                _a, _b = int(spans[i][0]), int(spans[i][1])
                _nv: Optional[int] = (
                    int(sum(1 for _v in blk[_a:_b] if int(_v) != int(_pid)))
                    if _pid is not None else None)
                spots.append({
                    "tokens": emb["patterns"][i].reshape(-1, k),
                    "n_valid": _nv,
                })

            recalled, n_q, details = self._ref_recall_block(
                spots, prior_slots,
                slot_upper_bound=(int(_floor_eff) if int(_floor_eff) > 0 else None),
            )
        r.recalled = list(recalled)
        r.n_recall_queries = int(n_q)

        _edges_w = [e for d in details
                    for e in (((d or {}).get("audit") or {}).get("traversed_edges") or [])]
        pre = self._assemble_prefix(recalled, edges=_edges_w)
        r.prefix_tokens = len(pre["ids"])
        r.prefix_dropped = int(pre["dropped"])

        _carriers = sorted(
            {int(c) for res in details for c in (res.get("used_carriers") or [])}
        )
        p = len(pre["ids"])

        ids2 = list(pre["ids"]) + blk

        prefix_spans = list(pre["spans"])
        prefix_slots = [int(s) for s in pre["used"]]
        _pre_spans = [(int(a), int(b)) for (a, b) in prefix_spans]
        _blk_spans = [(int(a) + p, int(b) + p) for (a, b) in spans]
        spans_eff = _pre_spans + _blk_spans

        h, graph = self._forward(
            ids2, spans_eff, capture_attention=bool(not is_qa_block),
        )
        r.forward_count = 1

        self.memory.ensure_slots(int(self.memory.num_slots) + len(spans))
        new_slots: List[int] = []
        for i, (a, b) in enumerate(spans):
            new_slots.append(
                int(
                    self.memory.store(
                        tuple(int(x) for x in blk[a:b]),
                        None,
                    )
                )
            )
        self.memory.write_batch(
            [emb["patterns"][i] for i in range(len(spans))], new_slots
        )
        wmean = float(self.memory.node.weight.mean())
        r.write_stats = [
            WriteStats(
                index=int(s),
                reward=1.0,
                repeats=1,
                eligibility_norm=0.0,
                delta_abs_mean=0.0,
                weight_mean=wmean,
            )
            for s in new_slots
        ]
        r.slots = [int(s.index) for s in r.write_stats]

        slot_index = list(prefix_slots) + list(r.slots)

        n_new = n_cross = n_neg = 0
        if not is_qa_block:
            n_new, n_cross, n_neg = self._init_edges(
                graph, spans_eff, slot_index, prefix_len=len(prefix_spans)
            )

        r.edges_new = int(n_new)
        r.edges_cross = int(n_cross)
        r.n_neg_init = int(n_neg)
        r.edges_after = int(self.memory.num_laterals_edges)
        return r, {
            "slots": prefix_slots,
            "edges": _edges_w,
            "carriers": _carriers,
        }

    def _run_ask(
        self,
        index: int,
        blk: Sequence[int],
        r: BlockResult,
        *,
        question: Optional[Sequence[int]],
        question_span: Optional[Span],
        system: Optional[str],
        prefix_slots: Sequence[int],
        prefix_edges: Sequence[Sequence[float]],
        carriers: Sequence[int],
    ) -> AskResult:
        ask = AskResult(block=r, question_span=question_span, recalled=list(r.recalled))

        pre = self._assemble_prefix(prefix_slots, edges=prefix_edges)

        prefix_slots = [int(s) for s in pre["used"]]

        if r.slots:
            ask.question_slots = [int(r.slots[-1])]

        if not callable(getattr(self.backend, "chat_frame_ids", None)):

            raise RuntimeError(
                f"backend={type(self.backend).__name__} 缺少 chat_frame_ids 或 "
                f"generate_ids(prompts_are_ids=True)"
            )
        frame_pre, frame_suf, gap = self.backend.chat_frame_ids(system=system)
        prompt_ids: List[int] = list(frame_pre)

        prefix_spans: List[Span] = []
        for a0, b0 in pre["spans"]:
            a = len(prompt_ids)
            prompt_ids.extend(int(x) for x in pre["ids"][a0:b0])
            prefix_spans.append((a, len(prompt_ids)))
        if pre["ids"] and blk:
            prompt_ids.extend(gap)
        block_start = len(prompt_ids)
        prompt_ids.extend(int(x) for x in blk)

        block_spans: List[Span] = []
        _seg_len = int(self.encoder.segment_tokens)
        _acc = block_start
        for _i in range(len(r.spans)):
            _a = _acc
            _b = min(_acc + _seg_len, block_start + len(blk))
            if _a >= _b:
                break
            block_spans.append((_a, _b))
            _acc = _b
        q_ids: List[int] = []
        if question is not None:
            q_ids = [int(x) for x in question]
        if q_ids:
            prompt_ids.extend(gap)
            prompt_ids.extend(q_ids)
        prompt_ids.extend(frame_suf)

        ask.block_spans = list(block_spans)
        ask.context = self._decode_ids(prompt_ids)
        ask.prompt_tokens = len(prompt_ids)
        text, gen_ids = self.backend.generate_ids(
            list(prompt_ids),
            max_new_tokens=int(self.cfg.gen_max_new_tokens),
            do_sample=False,
            prompts_are_ids=True,
        )

        gen_ids = (
            list(gen_ids[0])
            if gen_ids and isinstance(gen_ids[0], (list, tuple))
            else list(gen_ids)
        )
        text = text[0] if isinstance(text, list) and text else (text or "")
        ask.generated = text
        ask.gen_tokens = len(gen_ids)

        self._store_gen_segments(ask, gen_ids)

        if gen_ids and self.gen_attn_reward is not None:

            self._decode_learn(
                ask,
                prompt_ids=prompt_ids,
                gen_ids=gen_ids,
                prefix_spans=prefix_spans,
                prefix_slots=prefix_slots,
                carriers=carriers,
                block_slots=list(r.slots),
            )
        return ask

    def reset(self) -> None:
        self.memory.node.reset()
        self.forward_count = 0

    def stats(self, result: Optional[RunResult] = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "slots_used": self.memory.num_segments,
            "num_slots": int(self.memory.num_slots),
            "edges": int(self.memory.num_laterals_edges),
            "mean_degree": float(self.memory.mean_laterals_slots),
            "absolute_threshold": bool(self.absolute_threshold),
        }
        if result is not None:
            out.update(
                {
                    "blocks": result.n_blocks,
                    "mode": result.argv_mode,
                    "forward_count": result.forward_count,
                    "edges_cross": result.edges_cross,
                }
            )
        return out

    def __repr__(self) -> str:
        return (
            f"Synaptica(block_tokens={self.cfg.block_tokens}, "
            f"layer={self.layer}, slots={self.memory.num_segments}, "
            f"edges={self.memory.num_laterals_edges})"
        )
