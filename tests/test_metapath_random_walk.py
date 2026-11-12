import mlx.core as mx
import numpy as np
import pytest

from mlx_cluster import (
    build_metapath_csr,
    metapath_node_types,
    metapath_random_walk,
)

AP = ("author", "writes", "paper")
PV = ("paper", "published_in", "venue")
VP = ("venue", "publishes", "paper")
PA = ("paper", "written_by", "author")
APVPA = [AP, PV, VP, PA]


def toy_graph(seed=0, n_author=30, n_paper=60, n_venue=5, n_writes=120):
    """Random author/paper/venue graph with reverse edges.

    Paper 0 has no venue and author 0 writes nothing, to exercise dead ends.
    """
    rng = np.random.default_rng(seed)
    a = rng.integers(1, n_author, n_writes)
    p = rng.integers(0, n_paper, n_writes)
    pv_p = np.arange(1, n_paper)
    pv_v = rng.integers(0, n_venue, n_paper - 1)
    ei = {
        AP: mx.array(np.stack([a, p])),
        PA: mx.array(np.stack([p, a])),
        PV: mx.array(np.stack([pv_p, pv_v])),
        VP: mx.array(np.stack([pv_v, pv_p])),
    }
    num_nodes = {"author": n_author, "paper": n_paper, "venue": n_venue}
    return ei, num_nodes


def edge_sets(ei):
    return {et: set(map(tuple, np.asarray(v).T.tolist())) for et, v in ei.items()}


def reference_walk(ei, num_nodes, metapath, start, rand):
    """Pure-Python reference with the same neighbor ordering as the kernel."""
    adj = {}
    for et in metapath:
        src, dst = np.asarray(ei[et])
        order = np.argsort(src, kind="stable")
        neigh = {i: [] for i in range(num_nodes[et[0]])}
        for s, d in zip(src[order], dst[order]):
            neigh[int(s)].append(int(d))
        adj[et] = neigh
    start, rand = np.asarray(start), np.asarray(rand)
    L = rand.shape[1]
    out = np.full((len(start), L + 1), -1, dtype=np.int64)
    for i, s in enumerate(start):
        cur = int(s)
        out[i, 0] = cur
        for l in range(L):
            if cur < 0:
                break
            nbrs = adj[metapath[l % len(metapath)]][cur]
            if not nbrs:
                cur = -1
            else:
                cur = nbrs[min(int(rand[i, l] * len(nbrs)), len(nbrs) - 1)]
            out[i, l + 1] = cur
    return out


def test_walks_follow_metapath_edges():
    ei, num_nodes = toy_graph()
    csr = build_metapath_csr(ei, APVPA, num_nodes)
    L = 12
    start = mx.arange(num_nodes["author"])
    nodes, edges = metapath_random_walk(csr, start, L, stream=mx.cpu)
    nodes = np.asarray(nodes)
    edges = np.asarray(edges)
    assert nodes.shape == (num_nodes["author"], L + 1)
    assert edges.shape == (num_nodes["author"], L)

    sets = edge_sets(ei)
    types = metapath_node_types(APVPA, L)
    for w in nodes:
        for l in range(L):
            u, v = w[l], w[l + 1]
            if u < 0:
                assert v == -1  # once dead, stays dead
                continue
            if v < 0:
                continue
            assert (u, v) in sets[APVPA[l % 4]]
            assert 0 <= v < num_nodes[types[l + 1]]


def test_dead_ends_are_minus_one():
    ei, num_nodes = toy_graph()
    csr = build_metapath_csr(ei, APVPA, num_nodes)
    # author 0 writes nothing -> dies at step 0
    nodes, edges = metapath_random_walk(csr, mx.array([0]), 6, stream=mx.cpu)
    assert np.asarray(nodes).tolist() == [[0, -1, -1, -1, -1, -1, -1]]
    assert np.asarray(edges).tolist() == [[-1] * 6]


def test_matches_reference():
    ei, num_nodes = toy_graph(seed=3)
    csr = build_metapath_csr(ei, APVPA, num_nodes)
    L = 10
    start = mx.array(np.repeat(np.arange(num_nodes["author"]), 4))
    rand = mx.random.uniform(shape=(start.shape[0], L), key=mx.random.key(7))
    nodes, _ = metapath_random_walk(csr, start, L, rand=rand, stream=mx.cpu)
    expected = reference_walk(ei, num_nodes, APVPA, start, rand)
    np.testing.assert_array_equal(np.asarray(nodes), expected)


def test_edge_ids_point_at_sampled_neighbor():
    ei, num_nodes = toy_graph(seed=5)
    csr = build_metapath_csr(ei, APVPA, num_nodes)
    L = 8
    nodes, edges = metapath_random_walk(
        csr, mx.arange(num_nodes["author"]), L, stream=mx.cpu
    )
    nodes, edges = np.asarray(nodes), np.asarray(edges)
    col, col_off = np.asarray(csr.col), np.asarray(csr.col_offset)
    for i in range(nodes.shape[0]):
        for l in range(L):
            if edges[i, l] >= 0:
                r = csr.rel_ids[l % 4]
                assert col[col_off[r] + edges[i, l]] == nodes[i, l + 1]


def test_uniform_neighbor_choice():
    # star: author 0 wrote papers 0..3; each step from author 0 should pick
    # each paper with probability 1/4
    ei = {
        AP: mx.array([[0, 0, 0, 0], [0, 1, 2, 3]]),
        PA: mx.array([[0, 1, 2, 3], [0, 0, 0, 0]]),
    }
    csr = build_metapath_csr(ei, [AP, PA])
    n = 20_000
    nodes, _ = metapath_random_walk(csr, mx.zeros((n,), mx.int64), 1, stream=mx.cpu)
    counts = np.bincount(np.asarray(nodes)[:, 1], minlength=4) / n
    np.testing.assert_allclose(counts, 0.25, atol=0.02)


def test_validation_errors():
    ei, num_nodes = toy_graph()
    with pytest.raises(ValueError, match="Invalid metapath"):
        build_metapath_csr(ei, [AP, VP], num_nodes)
    with pytest.raises(ValueError, match="not in edge_index_dict"):
        build_metapath_csr(ei, [("author", "cites", "paper")], num_nodes)
    csr = build_metapath_csr(ei, [AP, PV], num_nodes)  # not cyclic
    metapath_random_walk(csr, mx.arange(3), 2, stream=mx.cpu)  # ok
    with pytest.raises(ValueError, match="cyclic"):
        metapath_random_walk(csr, mx.arange(3), 3, stream=mx.cpu)


def test_reproducible_with_seed():
    ei, num_nodes = toy_graph()
    csr = build_metapath_csr(ei, APVPA, num_nodes)
    mx.random.seed(11)
    a, _ = metapath_random_walk(csr, mx.arange(10), 8, stream=mx.cpu)
    mx.random.seed(11)
    b, _ = metapath_random_walk(csr, mx.arange(10), 8, stream=mx.cpu)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal not available")
def test_gpu_matches_cpu():
    ei, num_nodes = toy_graph(seed=9)
    csr = build_metapath_csr(ei, APVPA, num_nodes)
    L = 16
    start = mx.array(np.repeat(np.arange(num_nodes["author"]), 50))
    rand = mx.random.uniform(shape=(start.shape[0], L))
    cpu, cpu_e = metapath_random_walk(csr, start, L, rand=rand, stream=mx.cpu)
    gpu, gpu_e = metapath_random_walk(csr, start, L, rand=rand, stream=mx.gpu)
    np.testing.assert_array_equal(np.asarray(cpu), np.asarray(gpu))
    np.testing.assert_array_equal(np.asarray(cpu_e), np.asarray(gpu_e))


def test_strided_inputs():
    # mx.repeat / slicing give strided views; results must match a copy.
    ei, num_nodes = toy_graph(seed=2)
    csr = build_metapath_csr(ei, APVPA, num_nodes)
    L = 6
    start_view = mx.repeat(mx.arange(10).astype(mx.int64), 3)
    start_copy = mx.array(np.asarray(start_view))
    rand_full = mx.random.uniform(shape=(30, 2 * L))
    rand_view = rand_full[:, ::2]
    rand_copy = mx.array(np.asarray(rand_view))
    a, _ = metapath_random_walk(csr, start_view, L, rand=rand_view, stream=mx.cpu)
    b, _ = metapath_random_walk(csr, start_copy, L, rand=rand_copy, stream=mx.cpu)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
