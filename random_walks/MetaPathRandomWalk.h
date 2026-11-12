#pragma once

// Meta-path guided random walks on heterogeneous graphs (metapath2vec).
//
// Every relation (edge type) used by the meta-path is stored as its own CSR
// adjacency, concatenated into flat arrays:
//
//   rowptr     : all relations' rowptr arrays back to back (int64)
//   col        : all relations' col arrays back to back (int64)
//   rowptr_off : rowptr_off[r] = start of relation r inside `rowptr`
//   col_off    : col_off[r]    = start of relation r inside `col`
//   step_rel   : step_rel[l]   = relation id used at walk step l
//
// Node ids are local to their node type. A walk that reaches a node without
// outgoing edges of the required relation is terminated: the remaining node
// slots are -1 (and so are the corresponding edge slots).
//
// The Python front-end (mlx_cluster/metapath.py) builds these arrays from
// (src, rel, dst) keyed edge_index dicts.

#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/primitives.h>

namespace mx = mlx::core;

namespace mlx_metapath_walk {

class MetaPathRandomWalk : public mx::Primitive {
 public:
  explicit MetaPathRandomWalk(int walk_length, mx::Stream stream)
      : mx::Primitive(stream), walk_length_(walk_length) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::array> jvp(
      const std::vector<mx::array>& primals,
      const std::vector<mx::array>& tangents,
      const std::vector<int>& argnums) override;

  std::vector<mx::array> vjp(
      const std::vector<mx::array>& primals,
      const std::vector<mx::array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<mx::array>& outputs) override;

  std::pair<std::vector<mx::array>, std::vector<int>> vmap(
      const std::vector<mx::array>& inputs,
      const std::vector<int>& axes) override;

  const char* name() const override {
    return "MetaPathRandomWalk";
  }

  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  int walk_length_;
};

/**
 * Meta-path guided uniform random walks.
 *
 * Returns {nodes, edges}:
 *   nodes : [num_starts, walk_length + 1] int64, local node ids (-1 = ended)
 *   edges : [num_starts, walk_length]     int64, edge offset inside the
 *           relation's `col` array (-1 = ended)
 */
std::vector<mx::array> metapath_random_walk(
    const mx::array& rowptr,
    const mx::array& col,
    const mx::array& rowptr_off,
    const mx::array& col_off,
    const mx::array& step_rel,
    const mx::array& start,
    const mx::array& rand,
    int walk_length,
    mx::StreamOrDevice s = {});

} // namespace mlx_metapath_walk
