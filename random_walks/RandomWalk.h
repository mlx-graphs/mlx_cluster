#pragma once

#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/primitives.h>

namespace mx = mlx::core;

namespace mlx_random_walk {

// Uniform random walks on a homogeneous CSR graph.
//
// Randomness is supplied by the caller as `rand` ([num_starts, walk_length]
// uniform [0, 1) values), one value per step. A node without neighbors keeps
// the walk in place (edge id -1), matching PyG/torch-cluster.
class RandomWalk : public mx::Primitive {
 public:
  explicit RandomWalk(int walk_length, mx::Stream stream)
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
    return "RandomWalk";
  }

  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  int walk_length_;
};

/**
 * Uniform random walks.
 *
 * rowptr, col, start : 1-D int64
 * rand               : float32 [len(start), walk_length], values in [0, 1)
 *
 * Returns {nodes [len(start), walk_length + 1], edges [len(start),
 * walk_length]}, both int64.
 */
std::vector<mx::array> random_walk(
    const mx::array& rowptr,
    const mx::array& col,
    const mx::array& start,
    const mx::array& rand,
    int walk_length,
    mx::StreamOrDevice s = {});

} // namespace mlx_random_walk
