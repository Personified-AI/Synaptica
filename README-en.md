# Synaptica

Providing an associative long-term memory and context expansion system for LLMs, comprising a spiking neural network

![License](https://img.shields.io/badge/license-Apache_2.0-blue?style=flat)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat&logo=pytorch&logoColor=white)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:qwxl@chumenta.cn)
[![arXiv](https://img.shields.io/badge/arXiv-PENDING-B31B1B?style=flat-square&logo=arxiv&logoColor=white)]()

[中文](README.md) | English

## Quick Start

It relies on PyTorch 2.x, `transformers>=4.56` (Qwen3.5 requires 5.x), and a Hugging Face causal language model with full attention layers.

```bash
pip install -r requirements.txt
```

```python
from session import Synaptica, SynapticaConfig
from snn.memory import SynapticaMemory
from ann.llm.backend import QwenBackend

cfg = SynapticaConfig(
    device="cuda:0",
    layer=19,
    block_tokens=16384,
    segment_tokens=64,
    attn_block=256,
)
backend = QwenBackend("/path/to/Qwen3.5-9B", device="cuda:0")
memory = SynapticaMemory(input_size=cfg.num_neurons, num_slots=1, seed=0)
bot = Synaptica(memory=memory, backend=backend, cfg=cfg, seed=0)

with open("long_document.txt", encoding="utf-8") as f:
    bot.write(backend.tokenize(f.read()))

qids = backend.tokenize("What did the third experiment conclude?")
res = bot.ask(qids, question=qids)
print(res.generated)
print(res.recalled[:10], res.context[:200])
# the model's answer and the recalled segment ids plus the prompt the model actually saw.
```

- `layer`: decoder layer index to read attention from, zero-based; `19` is the output of layer 20. Set it to the layer count of the model in use.
- `block_tokens`: maximum token count of one forward pass.
- `segment_tokens`: smallest unit of memory; edge building and recall operate on segments.
- `attn_block`: row block size for segment-level attention aggregation.

## How It Works

![Synaptica workflow](assets/workflow-en.png)

1. **Segment-level attention map**: For the forward pass of each writing block, take the attention from layer `layer` and aggregate it by segment; after removing positional bias, retain the `TopK` targets in each row, and constrain both the number of outgoing and incoming edges per target to maintain sparsity and efficiency. Perform mirror edge supplementation for each target; then, for each attender, take the target with the largest weight in its row and create negative edges from it to the remaining targets in that row, with weights computed according to the relative drop within the row, and with the absolute edge weight capped at `2.0`.

2. **Spike encoding**: The embedding of each token within a segment is dimensionality-reduced via a fixed random projection; for each token, take `top-8` and binarize to 0/1, then concatenate in token order to form the segment pattern.

3. **Spike recall**: Question tokens are encoded as spike vectors and propagated along lateral edges for a certain number of steps. The membrane potential is updated as `mem_t = 0.5·mem_{t-1} + L·s_{t-1}`; the firing threshold is `0.7×clamp(ref/u)`, where `u` is the attention mass of that slot and `ref` is the median positive mass of written slots. The firing amplitude equals the threshold, and a hard refractory period follows firing. Fired segments serve as carriers and are expanded after sorting by anchor-segment score.

4. **Anchor-segment scoring**: Perform position-wise cosine similarity between the question and candidate segments, and set weak similarities to zero. For each query token, take the maximum over the candidate segment; then take the maximum positive contiguous subarray sum on the query side, and additionally multiply the candidate side by a string-length gate.

5. **Generation reward**: In the QA stage, use the quantile of decoding attention mass as the cutoff. Recalled segments with mass not lower than the cutoff are recorded as rewarded, and the rest as penalized: new lateral edges are created among rewarded segments from the current block, from question slots to carriers, and from question slots to rewarded segments; existing edges are modulated by “mass − cutoff.”

## Benchmarks

InfiniteBench:
![Accuracy of the retrieval and full-context arms](assets/Infinitebench_en.png)

## API

### `Synaptica(memory, backend, cfg=None, seed=0)`

All arguments are keyword arguments.

- `memory`: `SynapticaMemory`, the memory itself.
- `backend`: language-model backend implementing at least `embeddings_ids`, `truncated_hidden_attn`, `generate_ids`, `chat_frame_ids`, `decode_attention_mass`, `pad_token_id`, and `decode` (or a `tokenizer` with `decode`).
- `cfg`: `SynapticaConfig`; the defaults are the recommended configuration.
- `seed`: `int`, session seed.

| Method                                                              | Description                                                                                                                                     |
| ------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `write(ids)`                                                        | Shortcut for `run(ids, mode="write")`                                                                                                           |
| `ask(ids, **kw)`                                                    | Shortcut for `run(ids, mode="qa", **kw)`, returns `AskResult`                                                                                   |
| `run(ids, *, mode, question=None, question_span=None, system=None)` | `mode` is `"write"` or `"qa"`; `question` is placed after the recalled prefix and the current block; `system` goes to the backend chat template |
| `reset()`                                                           | Clears memory, lateral edges and caches                                                                                                         |
| `stats(result=None)`                                                | Returns a statistics dict                                                                                                                       |
| `state_dict()` / `load_state_dict(state)`                           | Memory and encoder archive; loading verifies the encoder config fingerprint                                                                     |

### `SynapticaConfig`

| Field                  | Default    | Description                                       |
| ---------------------- | ---------- | ------------------------------------------------- |
| `block_tokens`         | `16384`    | Length of one write block (tokens)                |
| `segment_tokens`       | `64`       | Length of one segment (tokens)                    |
| `layer`                | `19`       | Decoder layer to read attention from (zero-based) |
| `attn_block`           | `256`      | Row block size for chunked attention              |
| `num_neurons`          | `8192`     | Number of spiking neurons, also the pattern width |
| `code_dim`             | `4096`     | Input dimension of the segment projection         |
| `pulse_seed`           | `20240315` | Seed of the projection matrix                     |
| `seq_theta`            | `0.4`      | Cosine threshold of anchor scoring                |
| `seq_steps`            | `6`        | Pulse steps during recall                         |
| `graph_pos_debias`     | `1.0`      | Positional debias coefficient, `0` disables it    |
| `graph_out_degree`     | `3`        | Outgoing edges kept per attender                  |
| `graph_max_in_degree`  | `4`        | In-degree limit per target                        |
| `graph_gate_keep`      | `0.5`      | Quantile of the anchor cursor gate                |
| `ref_anchor_k`         | `5`        | Anchor segments kept per recall round             |
| `recall_per_carrier_k` | `1`        | Candidate segments expanded per carrier           |
| `gen_max_new_tokens`   | `512`      | Generation limit                                  |
| `device`               | `"cpu"`    | Compute device                                    |

### `SynapticaMemory(input_size, num_slots, *, ...)`

`input_size` must equal the encoder's `num_neurons`; `num_slots` is the initial slot capacity and grows on demand as content is written. Remaining keywords: `lr=0.05`, `seed=0`, `weight_init=0.3`, `max_weight=1.0`, `trace_decay=0.95`, `eligibility_decay=0.9`, `teacher_amplitude=5.0`, `device=None`, `lateral_cfg=None`, `store=None`, `max_cached_segments=0`. `lateral_cfg` controls lateral edges and recall; `store` can be a `SegmentStore` for on-disk segments.

### `LateralConfig`

| Field                       | Default | Description                             |
| --------------------------- | ------- | --------------------------------------- |
| `alpha`                     | `1.0`   | Scale of new edge weights               |
| `max_weight`                | `2.0`   | Edge weight bound (same for both signs) |
| `max_out_degree`            | `3`     | Outgoing edges kept per source node     |
| `max_new_edges_per_segment` | `1`     | New-edge budget per segment             |
| `lr`                        | `10.0`  | Weight learning rate                    |
| `decay`                     | `0.01`  | Weight decay                            |
| `neg_gain`                  | `1.0`   | Negative edge gain                      |
| `attn_q`                    | `0.25`  | Quantile cut of decode attention mass   |
| `recall_fire_theta`         | `0.7`   | Recall threshold ratio                  |
| `mem_beta`                  | `0.5`   | Membrane potential retention            |
| `pfc_z_coef`                | `0.40`  | z coefficient of the recall gain        |

### Backend

`QwenBackend(model_path, *, device=None, dtype=torch.bfloat16, lazy=False, attn_implementation=None, device_map=None, max_memory=None, expand_kv_in_generate=None, load_in_8bit=False)`
This backend is designed for the Qwen 3.5 architecture and implements the above contract. When integrating other models, simply implement methods with the same names. Requirements:

1. `truncated_hidden_attn` must return the segment-level attention map for that layer on the raw input sequence.
2. `generate_ids` must support `prompts_are_ids=True`.

We would very much welcome contributions from the open-source community to develop back-ends based on other model architectures for this project!

## Layout

```
./
├── session.py          # Synaptica, SynapticaConfig, result types
├── types.py            # Span and other shared types
├── segmenter.py        # Block segmentation
├── ann/
│   ├── attention_graph.py   # Segment attention graph and aggregation
│   ├── encoder.py           # Segment spike encoding
│   ├── pulse.py             # Random projection and binarization
│   ├── seq_match.py         # Anchor scoring
│   ├── gen_attn_reward.py   # Decode attention reward
│   ├── chunked_attn.py      # Chunked attention
│   ├── fast_attn.py         # GQA-expanded attention
│   └── llm/
│       ├── backend.py       # Language-model backend
│       └── capture.py       # Attention capture
└── snn/
    ├── memory.py        # Memory and recall
    ├── node.py          # Synapse matrices and lateral edges
    └── store.py         # Segment store
```

## License

Apache License 2.0.
