# mlx-cluster

mlx-cluster provides fast graph algorithms for Apple's
[MLX](https://github.com/ml-explore/mlx) framework: random walks, neighbor
sampling, and sparse x dense matrix multiplication (SpMM) for GNN message
passing. It is a C++ extension of MLX with Metal kernels, and the sampling and
sparse backend of [mlx-graphs](https://mlx-graphs.github.io/mlx-graphs/).

```bash
pip install mlx-cluster
```

## What's inside

- **Random walks**: uniform walks ([`random_walk`][mlx_cluster.random_walk]) and Node2Vec biased walks with `p` / `q` ([`rejection_sampling`][mlx_cluster.rejection_sampling]), on CPU and GPU.
- **Neighbor sampling**: multi-hop GraphSAGE sampling ([`neighbor_sample`][mlx_cluster.neighbor_sample]), heterogeneous graphs ([`hetero_neighbor_sample`][mlx_cluster.hetero_neighbor_sample]) and meta-path walks ([`metapath_random_walk`][mlx_cluster.metapath_random_walk]).
- **Fused message passing**: SpMM with `sum`, `mean`, `max` and `min`, with gradients, in COO ([`spmm_coo`][mlx_cluster.spmm_coo]) and load-balanced CSR ([`spmm_csr`][mlx_cluster.spmm_csr]) formats.
- **Attention scores**: multi-head dot products per edge ([`edge_scores`][mlx_cluster.edge_scores]) without per-edge feature arrays.

## Where to go next

- [Installation](installation.md): from PyPI or from source.
- [Quickstart](quickstart.md): random walks, neighbor sampling and SpMM in a few lines.
- [Sparse matrix multiplication](sparse.md): conventions, formats and performance of the SpMM kernels.
- [API reference](api.md): every public function and class.
