# MinARC Attribution Framework

A mechanistic interpretability toolkit for analyzing attention patterns and attribution in transformer models, built around ARC-style puzzles.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Experiment Layer                               │
│                         test.ipynb (main)                                │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         ▼                     ▼                     ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│   min_arc.py    │   │  attribute.py   │   │     vis.py      │
│  Puzzle Gen +   │   │  Attribution    │   │  Clustering +   │
│  Result struct  │   │  Graph/Node     │   │  Visualization  │
└─────────────────┘   └────────┬────────┘   └─────────────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
           ┌─────────────────┐   ┌─────────────────┐
           │  model_hooks.py │   │ model_client.py │
           │  Local access   │   │  Remote RPC     │
           └────────┬────────┘   └────────┬────────┘
                    │                     │
                    │              ┌──────┴──────┐
                    │              ▼             │
                    │      ┌─────────────────┐   │
                    │      │ model_server.py │   │
                    │      │ FastAPI + cache │   │
                    │      └────────┬────────┘   │
                    └───────────────┴────────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Modified Qwen2      │
                         │ modeling_qwen2.py   │
                         └─────────────────────┘
```

## Core Concepts

### 1. Attribution via Integrated Gradients (IG)

The framework attributes model predictions to individual attention heads and MLPs using **Integrated Gradients**. The key idea: compute gradients of the loss w.r.t. intermediate activations, scaled along a path from baseline to actual input.

```python
# Simplified IG: grad = E[∂L/∂x] over interpolated inputs
scaled_x = baseline + (x - baseline) * [0, 0.25, 0.5, 0.75, 1.0]  # steps=4
grad = autograd.grad(loss(f(scaled_x)), scaled_x).mean(dim=0)
```

### 2. Attribution Graph

Predictions are decomposed hierarchically into an **attribution tree/graph**:

```
lm_head (L80)           ← Final prediction
    │
    ├── MLP (L75)       ← Contributes via residual stream
    ├── Attn H42 (L70)  ← Attention head contributions
    │       │
    │       ├── attn_v: "What information flows through this head?"
    │       └── attn_q: "What queries attend here?"
    ...
```

### 3. Node Types

| Type | Description | IG Steps |
|------|-------------|----------|
| `lm_head` | Final logits → loss | 4 |
| `mlp_i`, `mlp_g` | MLP up-projection (SwiGLU) | 4 |
| `attn_q` | Query contribution | 4 |
| `attn_v` | Value contribution (linear) | 0 |
| `attn_a` | Attention pattern attribution | 0 |

---

## Key Data Structures

### `Result` (min_arc.py)
Holds everything about a puzzle sample:
```python
@dataclass
class Result:
    index: int                    # Sample ID
    prompt: str                   # Full prompt text
    answers: List[str]            # Ground truth answers ["R", "G", ...]
    answer_indices: List[int]     # Token positions of answers
    labels: List[int]             # Token IDs for answers
    candidate_ids: List[int]      # Allowed output token IDs
    puzzle: dict                  # Puzzle structure for range lookups
    outputs: Outputs              # Cached activations (after forward)
```

### `Outputs` (model_hooks.py)
Cached model activations:
```python
@dataclass
class Outputs:
    hidden_states: tuple    # Per-layer residual stream [L+1][b,i,e]
    attn_outs: tuple        # Pre-projection attn output [L][b,i,n,d]
    mlp_outputs: tuple      # MLP outputs [L][b,i,e]
    ln_states: tuple        # LayerNorm variances (for backward)
```

### `Graph` and `Node` (attribute.py)
Attribution graph with interned nodes:
```python
g = Graph(dataset_size=len(results), hidden_size=model.config.hidden_size)
# Creates and interns nodes (same args → same object)
node = Node(layer=70, head=42, type='attn_v')
node.backward(r, model)  # Compute gradient, dispatches local/remote
```

**Node interning**: `Node(70, 42, 'attn_v')` returns the same object if it exists in the current graph. This enables gradient accumulation across samples.

---

## Key Functions

### Forward Pass & Caching
```python
# Local
set_hooks(model)
output = model(**inputs, output_hidden_states=True)
outputs = get_outputs(model, output)  # Offload to CPU

# Remote
client = ModelClient("http://localhost:8000")
client.forward_results("my_dataset", results)
```

### Activation Retrieval
```python
# Unified interface (local or remote)
hs = get_hidden_states(model, r, layer=70, pos_ids=pos_ids)
aw = get_attn_weights(model, r, layer=70, head=42, pos_ids=pos_ids)
ho = get_head_output(model, r, layer=70, head=42, pos_ids=pos_ids)
```

### Attribution
```python
# Create downstream nodes (what we're attributing TO)
nodes = [Node(75, H, 'mlp_i') for _ in range(n_answers)]

# Run attribution step
attr = attribute_step(r, model, nodes)  # Returns [L, H+1, n_nodes]

# Attribute to upstream residual connections
attr = attribute_residual(r, model, nodes, from_layer=0, to_layer=75)
```

### Attention Pattern Analysis
```python
# Get attention weights + attribution gradients
aw, aa = attribute_attn_weights(r, model, layer, head, downstream_nodes)
# aa = aw * |grad|  (attribution-weighted attention)
```

---

## Workflows

### 1. Local Attribution
```python
# Setup
g = Graph(dataset_size=len(results), hidden_size=model.config.hidden_size)
set_hooks(model)

# Forward all samples
for r in results:
    inputs = tokenizer([r.prompt], return_tensors="pt").to(device)
    output = model(**inputs, output_hidden_states=True)
    r.outputs = get_outputs(model, output)

# Attribution
lm_node = Node(L, 0, 'lm_head')
g.add_node(lm_node)
attr = mr(attribute_step)(results, model, [lm_node])  # Map-reduce over dataset
top_heads = topk_md(attr, k=30)  # Find top contributing heads
```

### 2. Remote Attribution (for large models)
```python
# Server: python model_server.py --port 8000 --device cuda:0
client = ModelClient("http://localhost:8000")
client.load_model("/path/to/Qwen2.5-72B-GPTQ")
client.forward_results("puzzles", results)

# Same attribution code works!
attr = mr(attribute_step)(results, client, [lm_node])
```

### 3. Clustering Attention Heads
```python
# Collect attention patterns
aw = {(l, h): get_attn_weights(model, r, l, h, pos_ids) for l, h in top_heads}

# Cluster by JS divergence (attention = probability distributions)
groups, metrics, _, _ = cluster_heads(aw, dist_fn=compute_js_matrix, threshold=0.5)
```

---

## Local vs Remote: Unified Interface

The framework supports both local and remote execution with the same API:

```python
def is_remote(model):
    return hasattr(model, 'compute_node_gradient')

# In Node.backward():
if is_remote(model):
    return self._backward_remote(r, model)  # RPC to server
return self._backward_local(r, model)       # Direct computation
```

**Key principle**: Pass `model` everywhere. If it's a `ModelClient`, calls go remote. If it's `nn.Module`, they stay local.

---

## Notebook Setup (test.ipynb)

```python
%load_ext autoreload
%autoreload 2
%aimport -graph_registry  # CRITICAL: exclude from reload to preserve Graph state

from attribute import Graph, Node
from model_hooks import get_attn_weights, get_head_output
from vis import cluster_heads, compute_js_matrix
```

**Why `%aimport -graph_registry`?** The global `_current_graph` survives autoreload, allowing Node interning to work correctly across code changes.

---

## File Reference

| File | Purpose |
|------|---------|
| `min_arc.py` | Puzzle generation, `Result` dataclass, grid verbalization |
| `model_hooks.py` | Activation capture (`Outputs`), unified `get_*` functions |
| `attribute.py` | `Graph`, `Node`, IG backward, `attribute_residual` |
| `model_client.py` | `ModelClient` for remote server |
| `model_server.py` | FastAPI server, caches activations in `datasets` dict |
| `vis.py` | Clustering (`cluster_heads`), graph visualization, evaluation |
| `graph_registry.py` | Global `_current_graph` (survives autoreload) |
| `common_utils.py` | Utilities: `topk_md`, `einsum`, `mr` (map-reduce) |

---

## Tensor Dimension Conventions

```
b = batch_size (usually 1)
n = num_attention_heads
d = head_dim
e = hidden_size (= n * d)
i = query_seq_len
j = kv_seq_len
l = layer
g = num_downstream_grads
```

---

## Quick Start

```python
# 1. Generate puzzles
puzzle = gen_puzzle(side_neighbor_relations, n=5, m=5, n_train=5, seed=42)
prompt, _, puzzle = gen_prompt(puzzle, fn_defs=side_neighbor_fn_defs, tokenizer=tokenizer)

# 2. Create Result
r = Result(index=0, prompt=prompt, answers=['R'], answer_indices=[100], 
           labels=[tokenizer.encode(' R')[0]], candidate_ids=tokenizer.encode(' K B G R Y'))

# 3. Forward pass
set_hooks(model)
output = model(tokenizer([r.prompt], return_tensors="pt").to(device), output_hidden_states=True)
r.outputs = get_outputs(model, output)

# 4. Attribution
g = Graph(dataset_size=1, hidden_size=model.config.hidden_size)
lm_node = Node(model.config.num_hidden_layers, 0, 'lm_head')
g.add_node(lm_node)
attr = attribute_step(r, model, [lm_node])

# 5. Analyze top heads
for l, h, score in topk_md(attr, k=10, transpose=True):
    print(f"L{l}H{h}: {score:.4f}")
```

---

## Design Principles

1. **Code reuse over duplication** - Core logic in shared functions (`_backward_local`, `_attribute_residual`)
2. **Unified interfaces** - Same API for local/remote via duck typing
3. **Context objects** - Pass `r` (Result-like) instead of many args
4. **RPC helpers** - `_rpc` (client) and `rpc` (server) reduce boilerplate
5. **Interned nodes** - Same `(layer, head, type)` → same object for gradient accumulation


