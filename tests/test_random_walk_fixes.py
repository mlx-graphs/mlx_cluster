"""Regression tests for random_walk input handling (no dataset download)."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_cluster import random_walk

# 3-node directed cycle 0 -> 1 -> 2 -> 0, plus node 3 with no out-edges
ROWPTR = mx.array([0, 1, 2, 3, 3], dtype=mx.int64)
COL = mx.array([1, 2, 0], dtype=mx.int64)


def test_strided_start_matches_contiguous():
    # mx.repeat returns a strided view; this used to give garbage / segfault
    rand = mx.random.uniform(shape=(6, 4))
    view = mx.repeat(mx.array([0, 2], dtype=mx.int64), 3)
    copy = mx.array(np.asarray(view))
    a, ea = random_walk(ROWPTR, COL, view, rand, 4, stream=mx.cpu)
    b, eb = random_walk(ROWPTR, COL, copy, rand, 4, stream=mx.cpu)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    np.testing.assert_array_equal(np.asarray(ea), np.asarray(eb))
    assert np.asarray(a)[0].tolist() == [0, 1, 2, 0, 1]


def test_strided_rand_matches_contiguous():
    full = mx.random.uniform(shape=(4, 8))
    view = full[:, ::2]
    copy = mx.array(np.asarray(view))
    start = mx.array([0, 1, 2, 0], dtype=mx.int64)
    a, _ = random_walk(ROWPTR, COL, start, view, 4, stream=mx.cpu)
    b, _ = random_walk(ROWPTR, COL, start, copy, 4, stream=mx.cpu)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_rand_shape_is_validated():
    start = mx.array([0, 1, 2], dtype=mx.int64)
    with pytest.raises(ValueError, match="rand"):
        random_walk(ROWPTR, COL, start, mx.random.uniform(shape=(2, 4)), 4)
    with pytest.raises(ValueError, match="rand"):
        random_walk(ROWPTR, COL, start, mx.random.uniform(shape=(3, 5)), 4)
    with pytest.raises(ValueError, match="rand"):
        random_walk(ROWPTR, COL, start, mx.zeros((3, 4), mx.float16), 4)


def test_dtypes_are_validated():
    rand = mx.random.uniform(shape=(3, 4))
    with pytest.raises(ValueError, match="start"):
        random_walk(ROWPTR, COL, mx.array([0, 1, 2], dtype=mx.int32), rand, 4)
    with pytest.raises(ValueError, match="rowptr"):
        random_walk(ROWPTR.astype(mx.int32), COL, mx.arange(3).astype(mx.int64), rand, 4)


def test_walk_length_is_validated():
    with pytest.raises(ValueError, match="walk_length"):
        random_walk(ROWPTR, COL, mx.array([0], dtype=mx.int64),
                    mx.zeros((1, 0)), 0)


def test_outputs_are_int64():
    n, e = random_walk(ROWPTR, COL, mx.array([0], dtype=mx.int64),
                       mx.random.uniform(shape=(1, 3)), 3, stream=mx.cpu)
    assert n.dtype == mx.int64 and e.dtype == mx.int64


def test_rand_equal_to_one_is_clamped():
    # node 0 -> {1, 2}; rand == 1.0 must pick the last neighbor, not col[2]
    rowptr = mx.array([0, 2, 2, 2], dtype=mx.int64)
    col = mx.array([1, 2], dtype=mx.int64)
    n, e = random_walk(rowptr, col, mx.array([0], dtype=mx.int64),
                       mx.ones((1, 1)), 1, stream=mx.cpu)
    assert np.asarray(n).tolist() == [[0, 2]]
    assert np.asarray(e).tolist() == [[1]]


def test_dead_end_stays_in_place():
    n, e = random_walk(ROWPTR, COL, mx.array([3], dtype=mx.int64),
                       mx.random.uniform(shape=(1, 3)), 3, stream=mx.cpu)
    assert np.asarray(n).tolist() == [[3, 3, 3, 3]]
    assert np.asarray(e).tolist() == [[-1, -1, -1]]


def test_is_equivalent_does_not_throw_under_compile():
    start = mx.array([0, 1], dtype=mx.int64)

    def f(r):
        a, _ = random_walk(ROWPTR, COL, start, r, 3, stream=mx.cpu)
        b, _ = random_walk(ROWPTR, COL, start, r, 3, stream=mx.cpu)
        return a + b

    r = mx.random.uniform(shape=(2, 3))
    out = mx.compile(f)(r)
    expected, _ = random_walk(ROWPTR, COL, start, r, 3, stream=mx.cpu)
    np.testing.assert_array_equal(np.asarray(out), 2 * np.asarray(expected))


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal not available")
def test_gpu_matches_cpu():
    rng = np.random.default_rng(0)
    n_nodes = 200
    src = rng.integers(0, n_nodes, 2000)
    dst = rng.integers(0, n_nodes, 2000)
    order = np.argsort(src, kind="stable")
    rowptr = mx.array(np.concatenate([[0], np.cumsum(np.bincount(src, minlength=n_nodes))]))
    col = mx.array(dst[order])
    start = mx.repeat(mx.arange(n_nodes).astype(mx.int64), 5)  # strided on purpose
    rand = mx.random.uniform(shape=(start.shape[0], 12))
    a, ea = random_walk(rowptr, col, start, rand, 12, stream=mx.cpu)
    b, eb = random_walk(rowptr, col, start, rand, 12, stream=mx.gpu)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    np.testing.assert_array_equal(np.asarray(ea), np.asarray(eb))
