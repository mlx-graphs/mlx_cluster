"""Tests for rejection_sampling (node2vec walks) and the in-kernel RNG.

No dataset download: everything runs on small synthetic graphs.
"""

import mlx.core as mx
import numpy as np
import pytest

from mlx_cluster import rejection_sampling
from mlx_cluster._ext import _threefry2x32


def undirected_csr(edges, num_nodes):
    e = np.array(edges, dtype=np.int64)
    src = np.concatenate([e[:, 0], e[:, 1]])
    dst = np.concatenate([e[:, 1], e[:, 0]])
    order = np.lexsort((dst, src))
    src, dst = src[order], dst[order]
    rowptr = np.concatenate([[0], np.cumsum(np.bincount(src, minlength=num_nodes))])
    return mx.array(rowptr.astype(np.int64)), mx.array(dst.astype(np.int64))


def random_graph(seed=0, n=300, m=1500):
    rng = np.random.default_rng(seed)
    edges = {tuple(sorted(e)) for e in rng.integers(0, n, (m, 2)) if e[0] != e[1]}
    return undirected_csr(sorted(edges), n)


# ---------------------------------------------------------------------------
# RNG
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key, ctr, expected",
    [
        ((0, 0), (0, 0), (0x6B200159, 0x99BA4EFE)),
        ((0xFFFFFFFF,) * 2, (0xFFFFFFFF,) * 2, (0x1CB996FC, 0xBB002BE7)),
        ((0x13198A2E, 0x03707344), (0x243F6A88, 0x85A308D3), (0xC4923A9C, 0x483DF7A0)),
    ],
)
def test_threefry_known_answers(key, ctr, expected):
    # Random123 known-answer vectors for threefry2x32-20
    assert _threefry2x32(*key, *ctr) == expected


# ---------------------------------------------------------------------------
# Correctness of the node2vec transition probabilities
# ---------------------------------------------------------------------------

# t=0, v=1, a=2 (neighbor of both), b=3 (neighbor of v only)
#   0 - 1, 0 - 2, 1 - 2, 1 - 3
SMALL = undirected_csr([(0, 1), (0, 2), (1, 2), (1, 3)], 4)


def second_step_distribution(p, q, n=60_000, max_attempts=1000, stream=mx.cpu):
    rowptr, col = SMALL
    start = mx.zeros((n,), mx.int64)
    nodes, _ = rejection_sampling(
        rowptr, col, start, 2, p, q, key=mx.random.key(1),
        max_attempts=max_attempts, stream=stream,
    )
    nodes = np.asarray(nodes)
    via_v = nodes[nodes[:, 1] == 1]
    assert len(via_v) > n // 3
    return np.bincount(via_v[:, 2], minlength=4)[[0, 2, 3]] / len(via_v)


@pytest.mark.parametrize("p, q", [(1, 1), (4, 0.25), (0.25, 4), (2, 0.5), (0.5, 0.5)])
def test_matches_exact_node2vec_probabilities(p, q):
    # from v, next is t (weight 1/p), a (weight 1, common neighbor) or
    # b (weight 1/q, outward)
    w = np.array([1 / p, 1.0, 1 / q])
    expected = w / w.sum()
    got = second_step_distribution(p, q)
    np.testing.assert_allclose(got, expected, atol=0.015)


def test_max_attempts_one_is_uniform():
    # one attempt means the first candidate is always taken -> uniform
    got = second_step_distribution(4, 0.25, max_attempts=1)
    np.testing.assert_allclose(got, [1 / 3] * 3, atol=0.015)


def test_cap_bounds_pathological_parameters():
    # path 0 - 1 - 3: from 1 the only moves are "return" and "outward",
    # both accepted with probability ~1e-6. Must still terminate.
    rowptr, col = undirected_csr([(0, 1), (1, 3)], 4)
    nodes, _ = rejection_sampling(
        rowptr, col, mx.zeros((100,), mx.int64), 6, 1e6, 1e6,
        key=mx.random.key(0), max_attempts=5, stream=mx.cpu,
    )
    assert nodes.shape == (100, 7)


# ---------------------------------------------------------------------------
# Walk validity, determinism, inputs
# ---------------------------------------------------------------------------


def test_every_step_is_an_edge():
    rowptr, col = random_graph()
    rp, cl = np.asarray(rowptr), np.asarray(col)
    edges = {(u, v) for u in range(len(rp) - 1) for v in cl[rp[u]:rp[u + 1]]}
    start = mx.arange(len(rp) - 1).astype(mx.int64)
    nodes, eids = rejection_sampling(rowptr, col, start, 10, 0.5, 2.0, stream=mx.cpu)
    nodes, eids = np.asarray(nodes), np.asarray(eids)
    for w, e in zip(nodes, eids):
        for l in range(10):
            if e[l] == -1:
                assert w[l + 1] == w[l]
            else:
                assert (w[l], w[l + 1]) in edges
                assert cl[e[l]] == w[l + 1]


def test_same_key_same_walks_and_seed_reproducible():
    rowptr, col = random_graph(1)
    start = mx.arange(100).astype(mx.int64)
    a, _ = rejection_sampling(rowptr, col, start, 8, 1, 2, key=mx.random.key(3), stream=mx.cpu)
    b, _ = rejection_sampling(rowptr, col, start, 8, 1, 2, key=mx.random.key(3), stream=mx.cpu)
    c, _ = rejection_sampling(rowptr, col, start, 8, 1, 2, key=mx.random.key(4), stream=mx.cpu)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert not np.array_equal(np.asarray(a), np.asarray(c))

    mx.random.seed(7)
    d, _ = rejection_sampling(rowptr, col, start, 8, 1, 2, stream=mx.cpu)
    mx.random.seed(7)
    e, _ = rejection_sampling(rowptr, col, start, 8, 1, 2, stream=mx.cpu)
    np.testing.assert_array_equal(np.asarray(d), np.asarray(e))


def test_strided_start():
    rowptr, col = random_graph(2)
    view = mx.repeat(mx.arange(20).astype(mx.int64), 3)
    copy = mx.array(np.asarray(view))
    k = mx.random.key(0)
    a, _ = rejection_sampling(rowptr, col, view, 6, 1, 1, key=k, stream=mx.cpu)
    b, _ = rejection_sampling(rowptr, col, copy, 6, 1, 1, key=k, stream=mx.cpu)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_dead_end_stays_in_place():
    rowptr = mx.array([0, 0], dtype=mx.int64)
    col = mx.array([], dtype=mx.int64)
    n, e = rejection_sampling(rowptr, col, mx.array([0], dtype=mx.int64), 3, 1, 1, stream=mx.cpu)
    assert np.asarray(n).tolist() == [[0, 0, 0, 0]]
    assert np.asarray(e).tolist() == [[-1, -1, -1]]


def test_input_validation():
    rowptr, col = SMALL
    start = mx.array([0], dtype=mx.int64)
    with pytest.raises(ValueError, match="start"):
        rejection_sampling(rowptr, col, start.astype(mx.int32), 3, 1, 1)
    with pytest.raises(ValueError, match="p and q"):
        rejection_sampling(rowptr, col, start, 3, 0.0, 1)
    with pytest.raises(ValueError, match="walk_length"):
        rejection_sampling(rowptr, col, start, 0, 1, 1)
    with pytest.raises(ValueError, match="max_attempts"):
        rejection_sampling(rowptr, col, start, 3, 1, 1, max_attempts=0)
    with pytest.raises(ValueError, match="key"):
        rejection_sampling(rowptr, col, start, 3, 1, 1, key=mx.array([1, 2, 3], dtype=mx.uint32))


# ---------------------------------------------------------------------------
# GPU (run on Apple silicon)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal not available")
def test_gpu_matches_cpu_exactly():
    rowptr, col = random_graph(5, n=1000, m=8000)
    start = mx.repeat(mx.arange(1000).astype(mx.int64), 4)
    for p, q in [(1, 1), (0.25, 4), (4, 0.25)]:
        k = mx.random.key(11)
        a, ea = rejection_sampling(rowptr, col, start, 20, p, q, key=k, stream=mx.cpu)
        b, eb = rejection_sampling(rowptr, col, start, 20, p, q, key=k, stream=mx.gpu)
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        np.testing.assert_array_equal(np.asarray(ea), np.asarray(eb))


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal not available")
def test_gpu_matches_exact_node2vec_probabilities():
    p, q = 4, 0.25
    w = np.array([1 / p, 1.0, 1 / q])
    got = second_step_distribution(p, q, stream=mx.gpu)
    np.testing.assert_allclose(got, w / w.sum(), atol=0.015)
