"""Correctness tests for gatv2_scores (GATv2 attention logits).

Every case is checked against the gather reference
``(leaky_relu(x_src[col] + x_dst[row]) * att).sum(-1)`` and its gradients, on
CPU and, when available, GPU. The hub-node cases use a small long-row
threshold so the chunked backward kernels run on every graph.
"""

import mlx.core as mx
import numpy as np
import pytest

from mlx_cluster import coo_to_csr, gatv2_scores

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


def reference(row, col, x_dst, x_src, att, slope):
    z = x_src[col] + x_dst[row]
    return (mx.where(z > 0, z, slope * z) * att).sum(-1)


def random_graph(num_dst, num_src, num_edges, seed=0):
    rng = np.random.default_rng(seed)
    row = mx.array(rng.integers(0, num_dst, num_edges))
    col = mx.array(rng.integers(0, num_src, num_edges))
    return row, col


def features(num_dst, num_src, heads, channels, seed=0):
    k = mx.random.split(mx.random.key(seed), 3)
    x_dst = mx.random.normal((num_dst, heads, channels), key=k[0])
    x_src = mx.random.normal((num_src, heads, channels), key=k[1])
    att = mx.random.normal((heads, channels), key=k[2])
    return x_dst, x_src, att


def check_scores_and_grads(row, col, x_dst, x_src, att, slope=0.2, csr=None):
    cot = mx.random.normal((row.size, x_dst.shape[1]), key=mx.random.key(7))

    def fused(x_dst, x_src, att):
        return gatv2_scores(row, col, x_dst, x_src, att, slope, csr=csr)

    def ref(x_dst, x_src, att):
        return reference(row, col, x_dst, x_src, att, slope)

    out, grads = mx.vjp(fused, [x_dst, x_src, att], [cot])
    want, want_grads = mx.vjp(ref, [x_dst, x_src, att], [cot])
    assert out[0].shape == (row.size, x_dst.shape[1])
    assert_close(out[0], want[0])
    for got, expected in zip(grads, want_grads):
        assert got.shape == expected.shape
        assert_close(got, expected)


@pytest.mark.parametrize("heads", [1, 4])
@pytest.mark.parametrize("channels", [1, 7, 32, 77])
def test_matches_reference(device, heads, channels):
    row, col = random_graph(40, 40, 300)
    check_scores_and_grads(row, col, *features(40, 40, heads, channels))


@pytest.mark.parametrize("slope", [0.0, 0.2, 1.0])
def test_slopes(device, slope):
    row, col = random_graph(20, 20, 150, seed=1)
    check_scores_and_grads(row, col, *features(20, 20, 2, 16, seed=1), slope=slope)


def test_bipartite(device):
    # different numbers of destination and source nodes
    row, col = random_graph(10, 50, 200, seed=2)
    check_scores_and_grads(row, col, *features(10, 50, 2, 8, seed=2))


def test_duplicate_edges_and_isolated_nodes(device):
    row = mx.array([0, 0, 0, 2, 2, 5])
    col = mx.array([1, 1, 1, 3, 0, 5])
    check_scores_and_grads(row, col, *features(8, 8, 2, 5, seed=3))


def test_two_dimensional_inputs(device):
    row, col = random_graph(25, 25, 120, seed=4)
    x_dst, x_src, att = features(25, 25, 1, 16, seed=4)
    x_dst, x_src, att = x_dst[:, 0], x_src[:, 0], att[0]
    out = gatv2_scores(row, col, x_dst, x_src, att)
    assert out.shape == (120,)
    assert_close(out, reference(row, col, x_dst, x_src, att, 0.2))


def test_att_with_leading_axis(device):
    # mlx-graphs' GATv2Conv stores att as [1, heads, channels]
    row, col = random_graph(12, 12, 60, seed=5)
    x_dst, x_src, att = features(12, 12, 3, 4, seed=5)
    assert_close(
        gatv2_scores(row, col, x_dst, x_src, att[None]),
        reference(row, col, x_dst, x_src, att, 0.2),
    )


@pytest.mark.parametrize("threshold,chunk", [(8, 16), (0, 1), (4, 3)])
def test_hub_nodes(device, threshold, chunk):
    """High in- and out-degree nodes go through the chunked long-row path."""
    rng = np.random.default_rng(6)
    n = 60
    hub_in = np.zeros(400, dtype=np.int64)  # node 0: 400 incoming edges
    hub_out = np.full(300, 1, dtype=np.int64)  # node 1: 300 outgoing edges
    row = np.concatenate([hub_in, rng.integers(0, n, 300), rng.integers(0, n, 200)])
    col = np.concatenate([rng.integers(0, n, 400), hub_out, rng.integers(0, n, 200)])
    row, col = mx.array(row), mx.array(col)
    csr = coo_to_csr(
        row, col, n, num_cols=n, long_row_threshold=threshold, chunk_size=chunk
    )
    assert csr.long_rows is not None and csr.transpose.long_rows is not None
    check_scores_and_grads(row, col, *features(n, n, 4, 33, seed=6), csr=csr)


def test_precomputed_csr_matches_default(device):
    row, col = random_graph(30, 30, 200, seed=8)
    args = features(30, 30, 2, 8, seed=8)
    csr = coo_to_csr(row, col, 30)
    assert_close(gatv2_scores(row, col, *args, csr=csr), gatv2_scores(row, col, *args))


def test_no_edges(device):
    empty = mx.zeros((0,), dtype=mx.int64)
    x_dst, x_src, att = features(5, 5, 3, 4)
    out = gatv2_scores(empty, empty, x_dst, x_src, att)
    assert out.shape == (0, 3)
    grads = mx.grad(lambda *a: gatv2_scores(empty, empty, *a).sum(), argnums=(0, 1, 2))(
        x_dst, x_src, att
    )
    for g, x in zip(grads, (x_dst, x_src, att)):
        assert_close(g, mx.zeros_like(x))


def test_deterministic(device):
    row, col = random_graph(50, 50, 2000, seed=9)
    args = features(50, 50, 4, 32, seed=9)

    def grads():
        return mx.grad(
            lambda *a: gatv2_scores(row, col, *a).square().sum(), argnums=(0, 1, 2)
        )(*args)

    for a, b in zip(grads(), grads()):
        assert np.array_equal(np.array(a), np.array(b))


def test_bad_shapes():
    row, col = random_graph(5, 5, 10)
    x_dst, x_src, att = features(5, 5, 2, 4)
    with pytest.raises(ValueError):
        gatv2_scores(row, col, x_dst, x_src[:, 0], att)
    with pytest.raises(ValueError):
        gatv2_scores(
            row, col, x_dst, x_src, att, csr=coo_to_csr(row, col, 5, transpose=False)
        )
    with pytest.raises(ValueError):
        # a CSR of different edges
        gatv2_scores(row, col, x_dst, x_src, att, csr=coo_to_csr(row[:4], col[:4], 5))


@pytest.mark.parametrize(
    "row_dtype,col_dtype",
    [
        (mx.int32, mx.int32),
        (mx.int64, mx.int64),
        (mx.int32, mx.int64),
        (mx.uint32, mx.uint32),
    ],
)
def test_index_dtypes(device, row_dtype, col_dtype):
    """int32 / int64 edge lists go to the kernels as they are, others are
    converted; every combination gives the same scores and gradients."""
    row, col = random_graph(30, 30, 250, seed=10)
    csr = coo_to_csr(row, col, 30, long_row_threshold=8, chunk_size=16)
    row, col = row.astype(row_dtype), col.astype(col_dtype)
    check_scores_and_grads(row, col, *features(30, 30, 2, 8, seed=10), csr=csr)


def test_csr_without_precomputed_arrays(device):
    """CSRs built before LongRows.owner / CSRTranspose.edge existed (or by
    hand) still work: the op computes them on every call."""
    row, col = random_graph(40, 40, 600, seed=11)
    csr = coo_to_csr(row, col, 40, long_row_threshold=8, chunk_size=16)
    t = csr.transpose
    bare = csr._replace(
        long_rows=csr.long_rows._replace(owner=None),
        transpose=t._replace(edge=None, long_rows=t.long_rows._replace(owner=None)),
    )
    args = features(40, 40, 3, 5, seed=11)
    check_scores_and_grads(row, col, *args, csr=bare)
    cot = mx.ones((row.size, 3))
    a = mx.vjp(lambda *x: gatv2_scores(row, col, *x, csr=csr), list(args), [cot])
    b = mx.vjp(lambda *x: gatv2_scores(row, col, *x, csr=bare), list(args), [cot])
    for got, want in zip([*a[0], *a[1]], [*b[0], *b[1]]):
        assert np.array_equal(np.array(got), np.array(want))


def test_coo_to_csr_precomputes_graph_arrays():
    row, col = random_graph(25, 25, 400, seed=12)
    csr = coo_to_csr(row, col, 25, long_row_threshold=8, chunk_size=8)
    t = csr.transpose
    # original edge id of each transposed edge
    np.testing.assert_array_equal(
        np.array(t.edge), np.array(csr.perm)[np.array(t.perm)]
    )
    np.testing.assert_array_equal(np.array(row)[np.array(t.edge)], np.array(t.col))
    np.testing.assert_array_equal(
        np.array(col)[np.array(t.edge)],
        np.repeat(np.arange(25), np.diff(np.array(t.rowptr))),
    )
    # long row (index into rows) of each chunk
    for lr in (csr.long_rows, t.long_rows):
        cp = np.array(lr.chunk_ptr)
        np.testing.assert_array_equal(
            np.array(lr.owner), np.repeat(np.arange(cp.size - 1), np.diff(cp))
        )
