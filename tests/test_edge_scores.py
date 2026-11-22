"""Correctness tests for edge_scores (multi-head per-edge dot products).

Every case is checked against the gather reference
``scale * (a[row] * b[col]).sum(-1)``, on CPU and, when available, GPU.
"""

import math

import mlx.core as mx
import numpy as np
import pytest

from mlx_cluster import edge_scores

DEVICES = [mx.cpu] + ([mx.gpu] if mx.is_available(mx.gpu) else [])
ATOL = 1e-4
RTOL = 1e-4


@pytest.fixture(params=DEVICES, ids=lambda d: str(d))
def device(request):
    old = mx.default_device()
    mx.set_default_device(request.param)
    yield request.param
    mx.set_default_device(old)


def assert_close(a, b, atol=ATOL, rtol=RTOL):
    np.testing.assert_allclose(np.array(a), np.array(b), atol=atol, rtol=rtol)


def random_edges(num_a, num_b, num_edges, seed=0):
    rng = np.random.default_rng(seed)
    row = mx.array(rng.integers(0, num_a, num_edges))
    col = mx.array(rng.integers(0, num_b, num_edges))
    return row, col


def reference(row, col, a, b, scale=1.0):
    return scale * (a[row] * b[col]).sum(axis=-1)


@pytest.mark.parametrize("heads", [1, 4])
@pytest.mark.parametrize("channels", [7, 32, 77])
def test_forward(device, heads, channels):
    row, col = random_edges(40, 40, 300)
    a = mx.random.normal((40, heads, channels), key=mx.random.key(0))
    b = mx.random.normal((40, heads, channels), key=mx.random.key(1))
    scale = 1.0 / math.sqrt(channels)
    out = edge_scores(row, col, a, b, scale)
    assert out.shape == (300, heads)
    assert_close(out, reference(row, col, a, b, scale))


def test_two_dimensional_inputs(device):
    row, col = random_edges(25, 25, 120)
    a = mx.random.normal((25, 16), key=mx.random.key(0))
    b = mx.random.normal((25, 16), key=mx.random.key(1))
    out = edge_scores(row, col, a, b)
    assert out.shape == (120,)
    assert_close(out, reference(row, col, a, b))


def test_bipartite(device):
    # a and b have different numbers of rows
    row, col = random_edges(10, 50, 200)
    a = mx.random.normal((10, 2, 8), key=mx.random.key(0))
    b = mx.random.normal((50, 2, 8), key=mx.random.key(1))
    assert_close(edge_scores(row, col, a, b), reference(row, col, a, b))


def test_no_edges(device):
    row = mx.zeros((0,), dtype=mx.int64)
    a = mx.random.normal((5, 3, 4), key=mx.random.key(0))
    out = edge_scores(row, row, a, a)
    assert out.shape == (0, 3)
    grad = mx.grad(lambda a: edge_scores(row, row, a, a).sum())(a)
    assert_close(grad, mx.zeros_like(a))


def test_duplicate_edges(device):
    row = mx.array([0, 0, 0, 2])
    col = mx.array([1, 1, 1, 0])
    a = mx.random.normal((3, 2, 5), key=mx.random.key(0))
    b = mx.random.normal((3, 2, 5), key=mx.random.key(1))
    assert_close(edge_scores(row, col, a, b), reference(row, col, a, b))


@pytest.mark.parametrize("heads", [1, 3])
def test_gradients(device, heads):
    row, col = random_edges(30, 20, 250)
    a = mx.random.normal((30, heads, 12), key=mx.random.key(0))
    b = mx.random.normal((20, heads, 12), key=mx.random.key(1))
    # a non-uniform upstream gradient, so every (edge, head) weighs differently
    w = mx.random.normal((250, heads), key=mx.random.key(2))
    scale = 0.3

    def fused(a, b):
        return (edge_scores(row, col, a, b, scale) * w).sum()

    def ref(a, b):
        return (reference(row, col, a, b, scale) * w).sum()

    ga, gb = mx.grad(fused, argnums=(0, 1))(a, b)
    ra, rb = mx.grad(ref, argnums=(0, 1))(a, b)
    assert_close(ga, ra)
    assert_close(gb, rb)


def test_attention_layer_gradients(device):
    # scores -> segment softmax -> weighted sum, as in TransformerConv
    num_nodes, heads, channels = 25, 2, 8
    row, col = random_edges(num_nodes, num_nodes, 150)
    q = mx.random.normal((num_nodes, heads, channels), key=mx.random.key(0))
    k = mx.random.normal((num_nodes, heads, channels), key=mx.random.key(1))
    v = mx.random.normal((num_nodes, heads, channels), key=mx.random.key(2))

    def layer(score_fn, q, k):
        s = score_fn(q, k)  # [E, H]
        m = mx.zeros((num_nodes, heads)).at[row].maximum(s)
        e = mx.exp(s - mx.stop_gradient(m)[row])
        alpha = e / mx.zeros((num_nodes, heads)).at[row].add(e)[row]
        out = mx.zeros((num_nodes, heads, channels)).at[row].add(alpha[..., None] * v[col])
        return (out**2).sum()

    scale = 1.0 / math.sqrt(channels)

    def fused(q, k):
        return layer(lambda q, k: edge_scores(row, col, q, k, scale), q, k)

    def ref(q, k):
        return layer(lambda q, k: reference(row, col, q, k, scale), q, k)

    assert_close(fused(q, k), ref(q, k))
    gq, gk = mx.grad(fused, argnums=(0, 1))(q, k)
    rq, rk = mx.grad(ref, argnums=(0, 1))(q, k)
    assert_close(gq, rq)
    assert_close(gk, rk)


def test_dtype_follows_input(device):
    row, col = random_edges(10, 10, 30)
    a = mx.random.normal((10, 2, 4), key=mx.random.key(0)).astype(mx.float16)
    out = edge_scores(row, col, a, a)
    assert out.dtype == mx.float16


def test_compile(device):
    row, col = random_edges(20, 20, 100)
    a = mx.random.normal((20, 2, 8), key=mx.random.key(0))
    b = mx.random.normal((20, 2, 8), key=mx.random.key(1))
    f = mx.compile(lambda a, b: edge_scores(row, col, a, b, 0.5))
    assert_close(f(a, b), reference(row, col, a, b, 0.5))


def test_shape_errors():
    row, col = random_edges(5, 5, 10)
    with pytest.raises(ValueError):
        edge_scores(row, col, mx.zeros((5, 2, 4)), mx.zeros((5, 4)))
    with pytest.raises(ValueError):
        edge_scores(row, col, mx.zeros((5, 2, 4)), mx.zeros((5, 3, 4)))
    with pytest.raises(ValueError):
        edge_scores(row, col[:5], mx.zeros((5, 2, 4)), mx.zeros((5, 2, 4)))
