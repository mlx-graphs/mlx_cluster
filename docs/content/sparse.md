# Sparse Matrix Multiplication (SpMM)

mlx-cluster provides fused sparse × dense matrix multiplication kernels for
graph neural network message passing. They compute

$$
\text{out}[r, f] = \mathrm{reduce}_{\,e \,:\, \text{row}[e] = r}
\; \text{values}[e] \cdot x[\text{col}[e], f]
$$

where the sparse matrix `A` is the graph (one non-zero per edge) and `x`
holds the node features. In message-passing terms, `row` is each edge's
destination node, `col` its source node and `values` an optional edge
weight.

## Why use it

The usual *gather-scatter* message passing first gathers `x[col]` into an
array with one row **per edge** (`[num_edges, features]`) and then scatters
it into the destination nodes. On large graphs that per-edge array dominates
memory and time. The SpMM kernels read each neighbour's features and combine
them directly into the output, so only node-sized arrays are ever allocated,
in the forward **and** the backward pass.

Measured on an Apple M4 Pro (one training step = forward + backward, GCN
layer, 1M edges, 256 features): about **4x faster** and **4.6x less memory**
than gather-scatter (2.36 GB → 509 MB). The gain grows with
`num_edges × features`; on very small graphs (e.g. Cora) it is about
break-even.

## Quick example

```python
import mlx.core as mx
from mlx_cluster import coo_to_csr, spmm_coo, spmm_csr

# edge_index = [src, dst], as in mlx-graphs
edge_index = mx.array([[0, 1, 2, 2], [1, 2, 0, 1]])
x = mx.random.normal((3, 16))
src, dst = edge_index[0], edge_index[1]

# COO: works directly on the edge list
out = spmm_coo(dst, src, x, reduce="sum")         # == scatter(x[src], dst, "add")

# CSR: sort once per graph, then reuse
csr = coo_to_csr(dst, src, num_rows=3)
out = spmm_csr(
    csr.rowptr, csr.col, x,
    reduce="max",
    long_rows=csr.long_rows,     # load balancing for high-degree nodes
    transpose=csr.transpose,     # fast, deterministic backward pass
)

# gradients work as usual
grad = mx.grad(lambda x: spmm_coo(dst, src, x, reduce="mean").sum())(x)
```

## COO or CSR?

|             | [`spmm_coo`][mlx_cluster.spmm_coo]                                       | [`spmm_csr`][mlx_cluster.spmm_csr]                                           |
|-------------|---------------------------------------------------------------|-------------------------------------------------------------------|
| Input       | The edge list as-is, any order                                | CSR from [`coo_to_csr`][mlx_cluster.coo_to_csr] (one sort per graph; cache it) |
| GPU kernel  | One thread per (edge, feature); atomic adds                   | One thread per (node, feature); no atomics                        |
| Determinism | Forward `max`/`min` are exact; `sum`/`mean` and all gradients w.r.t. `x` may differ in the last bits between runs            | Bit-identical results and gradients every run (with `transpose`)  |
| Best for    | Small graphs, or graphs that change every step                | Medium and large graphs (fastest in our benchmarks)               |

## Reductions

`reduce` selects how the messages arriving at a node are combined, per
feature:

| `reduce`          | Forward                                     | Gradient                                                          |
|-------------------|---------------------------------------------|-------------------------------------------------------------------|
| `"sum"` / `"add"` | Sum of the messages                         | `A^T @ grad` for `x`; one dot product per edge for `values`       |
| `"mean"`          | Sum divided by the number of incoming edges | As `sum`, scaled by `1 / degree`                                  |
| `"max"`           | Largest message                             | Every edge whose message equals the output gets the full gradient (ties included)                                                    |
| `"min"`           | Smallest message                            | As `max`                                                          |

Conventions, matching mlx-graphs' `scatter`:

- Nodes without incoming edges get `0` for every reduction (not `±inf`).
- `max`/`min` are the true maximum/minimum, so an all-negative
  neighbourhood gives a negative result.
- With `max`/`min`, an exact tie (e.g. a duplicated edge) gives the full
  gradient to **every** tied edge, exactly like MLX's `scatter` max/min
  gradient. The backward pass recomputes each message and compares it with the
  saved output, so no argmax array is stored.
- All arithmetic is done in float32. Inputs of other dtypes are converted, and
  the result is returned in the dtype of `x`.

`mx.grad` / `mx.vjp` are supported for every reduction. `mx.jvp` is
supported for `sum` and `mean`. Second-order gradients and `mx.vmap` are
not supported.

## High-degree nodes (load balancing)

Real graphs often have a few nodes with a huge number of neighbours. A
one-thread-per-node kernel would leave one thread looping over all of a hub's
edges while the rest of the GPU sits idle. With `long_rows` (built by
[`coo_to_csr`][mlx_cluster.coo_to_csr]), nodes with more than `long_row_threshold`
edges (default 256) are cut into chunks of up to `chunk_size` edges
(default 2048). Each chunk is reduced by its own GPU threadgroup, then the
chunk results are combined in a fixed order. This keeps results
deterministic, and for sums it is also more accurate than one long running sum.

On a 1M-edge power-law graph whose biggest node has 134k neighbours, this took
the CSR forward pass from about 50 ms to about 1 ms.

Building `long_rows` reads `rowptr` back to the host once per graph, so it
cannot run inside `mx.compile`. Pass `long_row_threshold=None` to skip it.

## Fast backward pass (transpose)

The gradient of `sum`/`mean` with respect to `x` is `A^T @ grad`.
[`coo_to_csr`][mlx_cluster.coo_to_csr] also builds the transposed CSR
(`csr.transpose`, edges grouped by source node, with its own `long_rows`).
When it is passed to [`spmm_csr`][mlx_cluster.spmm_csr], the backward pass runs on
the CSR kernel too: no atomics, deterministic, and load-balanced for nodes
with many outgoing edges. This holds for every reduction: for `max`/`min`
each source node loops over its outgoing edges and adds the gradient of the
edges that won (see [Reductions](#reductions)). Without `transpose` (and with
[`spmm_coo`][mlx_cluster.spmm_coo]), the gradient w.r.t. `x` uses atomic adds
instead, so it is correct but its last bits can vary between runs. `coo_to_csr(..., transpose=False)` saves the extra
memory (about three index arrays) when only the forward pass is needed.

## Attention scores (`edge_scores`)

Graph attention layers (`TransformerConv`, graph transformers) weigh each
edge by a score computed from both of its endpoints, typically the dot product
of the destination's query and the source's key:

$$
\text{out}[e, h] = \text{scale} \cdot
\big\langle a[\text{row}[e], h, :],\; b[\text{col}[e], h, :] \big\rangle
$$

Gather-scatter computes this as `(a[row] * b[col]).sum(-1)`, which first
copies one row of `a` and one of `b` **per edge** (two
`[num_edges, heads, channels]` arrays) to produce only
`[num_edges, heads]` scores. [`edge_scores`][mlx_cluster.edge_scores] reads the two
rows it needs for each edge and writes the score directly: one GPU SIMD group
per (edge, head), all heads in one launch.

It is differentiable with respect to `a` and `b`, and the backward pass
avoids the per-edge arrays too: both gradients are SpMMs,

$$
\frac{\partial L}{\partial a[:, h]} =
\text{scale} \cdot \mathrm{spmm}(\text{row}, \text{col}, b[:, h], g[:, h]),
\qquad
\frac{\partial L}{\partial b[:, h]} =
\text{scale} \cdot \mathrm{spmm}(\text{col}, \text{row}, a[:, h], g[:, h])
$$

where `g` is the incoming gradient of the scores.

Together with SpMM this gives a whole attention layer without per-edge
feature arrays: scores, a softmax over each node's incoming edges (on
`[num_edges, heads]` only), then a weighted sum with one SpMM per head.

```python
import math

import mlx.core as mx
from mlx_cluster import edge_scores, spmm_coo

num_nodes, heads, channels = 1000, 4, 32
src = mx.random.randint(0, num_nodes, (20_000,))
dst = mx.random.randint(0, num_nodes, (20_000,))
q = mx.random.normal((num_nodes, heads, channels))  # queries
k = mx.random.normal((num_nodes, heads, channels))  # keys
v = mx.random.normal((num_nodes, heads, channels))  # values

# 1. one score per (edge, head): [num_edges, heads]
s = edge_scores(dst, src, q, k, scale=1 / math.sqrt(channels))

# 2. softmax over the incoming edges of each node
s_max = mx.zeros((num_nodes, heads)).at[dst].maximum(s)
e = mx.exp(s - mx.stop_gradient(s_max)[dst])
alpha = e / mx.zeros((num_nodes, heads)).at[dst].add(e)[dst]

# 3. weighted sum of the values, one SpMM per head
out = mx.stack(
    [spmm_coo(dst, src, v[:, h], alpha[:, h], num_rows=num_nodes)
     for h in range(heads)],
    axis=1,
)  # [num_nodes, heads, channels]
```

Conventions:

- `row` indexes `a` and `col` indexes `b`, and the two may have
  different numbers of rows (bipartite graphs). With mlx-graphs'
  `edge_index = [src, dst]`, queries are usually `a` (`row = dst`) and
  keys `b` (`col = src`).
- 2-D inputs (`[num_nodes, channels]`) are treated as one head and give
  `[num_edges]` scores.
- Arithmetic is done in float32; the result has the dtype of `a`.
- `mx.grad` / `mx.vjp` and `mx.compile` are supported; `mx.jvp`,
  `mx.vmap` and second-order gradients are not.

[`sddmm`][mlx_cluster.sddmm] computes the same single-head dot product but is not
differentiable: it exists for the backward pass of SpMM. Use
[`edge_scores`][mlx_cluster.edge_scores] in models.

Measured on an Apple M1 (8 GB), one `TransformerConv` layer from mlx-graphs
(4 heads, 128 features, forward + backward): about **3x less memory** at every
graph size (2.74 GB → 0.84 GB on 500k edges) and about **2x faster** on 500k
edges, compared with gather-scatter.

## Using it from mlx-graphs

mlx-graphs uses these kernels automatically in layers decorated with
`@fused_message_passing`, for example `GCNConv`, `SAGEConv`,
`GINConv`, `GATConv`, `GATv2Conv` and `TransformerConv` (which also
computes its attention scores with [`edge_scores`][mlx_cluster.edge_scores]). It falls
back to gather-scatter whenever the fused path does not apply, or with an
mlx-cluster that does not have these functions. See the mlx-graphs
documentation for details.

## Limitations

- SpMM messages must have the form `values[e] * x[col[e]]`, with one scalar
  weight per edge (or per edge and head, one SpMM per head). Attention layers
  fit this form once their scores are computed (with
  [`edge_scores`][mlx_cluster.edge_scores] for dot-product attention), but messages
  that add a per-edge feature (e.g. `TransformerConv` with edge features,
  `x[col[e]] + edge_attr[e]`) still need gather-scatter.
- `softmax` itself produces one value per edge, so it is computed outside
  the kernels, on `[num_edges, heads]` arrays.
- Indices are processed as int64 and features as float32 inside the kernels.
