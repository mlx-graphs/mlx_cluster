# Installation

## Requirements

- Python 3.10 or later
- macOS on Apple Silicon for the Metal kernels (a CPU-only build also works on Linux)
- MLX 0.32 or later

## From PyPI

```bash
pip install mlx-cluster
```

## From source

Building the Metal kernels needs Xcode (or its command line tools).

```bash
git clone https://github.com/mlx-graphs/mlx_cluster.git
cd mlx_cluster
pip install -e .
```

## Check the installation

```python
from importlib.metadata import version

import mlx_cluster

print(version("mlx_cluster"))
```

If this prints a version number, mlx-cluster is installed and importable.

## Troubleshooting

### `ImportError: No module named 'mlx'`

Install MLX first:

```bash
pip install mlx
```

### Slow kernels

The GPU kernels need Apple Silicon. Check that MLX uses the GPU with
`mlx.core.default_device()`.
