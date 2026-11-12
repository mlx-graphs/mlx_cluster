"""Meta-path guided random walks on heterogeneous graphs (metapath2vec).

Usage is two steps: build the per-relation CSR once with
:func:`build_metapath_csr`, then call :func:`metapath_random_walk` as many
times as needed (e.g. once per training batch).
"""

from typing import Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import mlx.core as mx
import numpy as np

from ._ext import _metapath_random_walk

NodeType = str
EdgeType = Tuple[str, str, str]

__all__ = [
    "MetaPathCSR",
    "build_metapath_csr",
    "metapath_random_walk",
    "metapath_node_types",
]


class MetaPathCSR(NamedTuple):
    """Per-relation CSR adjacency for the relations of a meta-path.

    Attributes:
        metapath: The meta-path, a list of ``(src, rel, dst)`` edge types.
        rowptr: All relations' CSR row pointers, concatenated (int64).
        col: All relations' CSR column indices, concatenated (int64).
        rowptr_offset: Start of each relation inside ``rowptr`` (int64).
        col_offset: Start of each relation inside ``col`` (int64).
        rel_ids: For each meta-path position, the relation id it uses.
        num_nodes_dict: Number of nodes per node type.
    """

    metapath: List[EdgeType]
    rowptr: mx.array
    col: mx.array
    rowptr_offset: mx.array
    col_offset: mx.array
    rel_ids: List[int]
    num_nodes_dict: Dict[NodeType, int]


def _validate_metapath(metapath: Sequence[EdgeType]) -> List[EdgeType]:
    metapath = [tuple(et) for et in metapath]  # type: ignore[misc]
    if len(metapath) == 0:
        raise ValueError("metapath must contain at least one edge type")
    for et in metapath:
        if len(et) != 3 or not all(isinstance(p, str) for p in et):
            raise ValueError(f"Edge types must be (src, rel, dst) str tuples: {et!r}")
    for a, b in zip(metapath[:-1], metapath[1:]):
        if a[2] != b[0]:
            raise ValueError(
                f"Invalid metapath: {a!r} ends at '{a[2]}' but {b!r} "
                f"starts at '{b[0]}'"
            )
    return metapath  # type: ignore[return-value]


def _infer_num_nodes(
    edge_index_dict: Mapping[EdgeType, mx.array],
    num_nodes_dict: Optional[Mapping[NodeType, int]],
) -> Dict[NodeType, int]:
    out: Dict[NodeType, int] = dict(num_nodes_dict or {})
    inferred: Dict[NodeType, int] = {}
    for (src, _, dst), ei in edge_index_dict.items():
        if ei.size == 0:
            continue
        ei_np = np.asarray(ei)
        inferred[src] = max(inferred.get(src, 0), int(ei_np[0].max()) + 1)
        inferred[dst] = max(inferred.get(dst, 0), int(ei_np[1].max()) + 1)
    for nt, n in inferred.items():
        if nt in out:
            if out[nt] < n:
                raise ValueError(
                    f"num_nodes_dict['{nt}'] = {out[nt]} but edges reference "
                    f"node id {n - 1}"
                )
        else:
            out[nt] = n
    return out


def build_metapath_csr(
    edge_index_dict: Mapping[EdgeType, mx.array],
    metapath: Sequence[EdgeType],
    num_nodes_dict: Optional[Mapping[NodeType, int]] = None,
) -> MetaPathCSR:
    """Builds the CSR adjacency used by :func:`metapath_random_walk`.

    This sorts every relation once, so compute it once per graph and reuse it.

    Args:
        edge_index_dict: ``(src, rel, dst) -> [2, num_edges]`` edge indices,
            with node ids local to each node type (as in ``HeteroGraphData``).
        metapath: List of ``(src, rel, dst)`` edge types. Each edge type must
            end at the node type the next one starts from.
        num_nodes_dict: Optional number of nodes per node type. Missing types
            are inferred from the largest node id in ``edge_index_dict``.

    Returns:
        MetaPathCSR
    """
    metapath = _validate_metapath(metapath)
    missing = [et for et in metapath if et not in edge_index_dict]
    if missing:
        raise ValueError(f"Edge types {missing} are not in edge_index_dict")

    num_nodes = _infer_num_nodes(edge_index_dict, num_nodes_dict)

    unique_rels: List[EdgeType] = []
    for et in metapath:
        if et not in unique_rels:
            unique_rels.append(et)

    rowptrs, cols, rp_off, col_off = [], [], [], []
    rp_cursor, col_cursor = 0, 0
    for et in unique_rels:
        src_type = et[0]
        n_src = num_nodes.get(src_type, 0)
        ei = np.asarray(edge_index_dict[et]).astype(np.int64)
        src, dst = ei[0], ei[1]
        order = np.argsort(src, kind="stable")
        counts = np.bincount(src, minlength=n_src)
        rowptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        rowptrs.append(rowptr)
        cols.append(dst[order])
        rp_off.append(rp_cursor)
        col_off.append(col_cursor)
        rp_cursor += rowptr.shape[0]
        col_cursor += dst.shape[0]

    return MetaPathCSR(
        metapath=metapath,
        rowptr=mx.array(np.concatenate(rowptrs)),
        col=mx.array(np.concatenate(cols)),
        rowptr_offset=mx.array(np.array(rp_off, dtype=np.int64)),
        col_offset=mx.array(np.array(col_off, dtype=np.int64)),
        rel_ids=[unique_rels.index(et) for et in metapath],
        num_nodes_dict=num_nodes,
    )


def metapath_node_types(metapath: Sequence[EdgeType], walk_length: int) -> List[str]:
    """Node type of each of the ``walk_length + 1`` positions of a walk."""
    m = len(metapath)
    return [metapath[0][0]] + [metapath[l % m][2] for l in range(walk_length)]


def metapath_random_walk(
    csr: MetaPathCSR,
    start: mx.array,
    walk_length: int,
    rand: Optional[mx.array] = None,
    stream: Optional[mx.Stream] = None,
) -> Tuple[mx.array, mx.array]:
    """Meta-path guided uniform random walks.

    Step ``l`` of every walk follows edge type ``metapath[l % len(metapath)]``,
    choosing uniformly among the current node's neighbors of that type. If a
    node has no such neighbor, the walk stops and the remaining entries are
    ``-1``.

    Args:
        csr: Output of :func:`build_metapath_csr`.
        start: Start node ids, local to ``metapath[0][0]``'s node type.
        walk_length: Number of steps per walk. If it is longer than the
            meta-path, the meta-path must be cyclic (end where it starts).
        rand: Optional uniform ``[0, 1)`` values of shape
            ``[len(start), walk_length]``. Drawn from MLX's RNG if omitted,
            so ``mx.random.seed`` makes walks reproducible.
        stream: Stream or device to run on (``mx.cpu`` or ``mx.gpu``).

    Returns:
        tuple ``(nodes, edges)``: ``nodes`` is ``[len(start), walk_length + 1]``
        local node ids (the type of column ``j`` is
        ``metapath_node_types(metapath, walk_length)[j]``); ``edges`` is
        ``[len(start), walk_length]`` edge offsets within each relation's CSR.

    Example:

    .. code-block:: python

        metapath = [
            ("author", "writes", "paper"),
            ("paper", "published_in", "venue"),
            ("venue", "publishes", "paper"),
            ("paper", "written_by", "author"),
        ]
        csr = build_metapath_csr(edge_index_dict, metapath)
        nodes, _ = metapath_random_walk(csr, mx.arange(10), walk_length=8)
    """
    metapath = csr.metapath
    m = len(metapath)
    if walk_length < 1:
        raise ValueError("walk_length must be >= 1")
    if walk_length > m and metapath[-1][2] != metapath[0][0]:
        raise ValueError(
            f"walk_length ({walk_length}) is longer than the metapath ({m}), "
            "so the metapath must be cyclic: it ends at "
            f"'{metapath[-1][2]}' but starts at '{metapath[0][0]}'"
        )

    start = mx.array(start) if not isinstance(start, mx.array) else start
    start = start.astype(mx.int64).reshape(-1)
    if rand is None:
        rand = mx.random.uniform(shape=(start.shape[0], walk_length))
    rand = rand.astype(mx.float32)

    step_rel = mx.array(
        np.array([csr.rel_ids[l % m] for l in range(walk_length)], dtype=np.int64)
    )
    return _metapath_random_walk(
        csr.rowptr,
        csr.col,
        csr.rowptr_offset,
        csr.col_offset,
        step_rel,
        start,
        rand,
        walk_length,
        stream=stream,
    )
