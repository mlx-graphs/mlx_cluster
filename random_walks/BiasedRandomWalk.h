#pragma once

#include <optional>

#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/primitives.h>

namespace mx = mlx::core;

namespace mlx_biased_random_walk {

// node2vec biased random walks (rejection sampling).
//
// Inputs: rowptr, col, start (int64), key (uint32 [2]). Random numbers are
// drawn inside the kernel from a counter-based generator seeded by `key`
// (see random_walks/rng.h), so the CPU and GPU give identical walks for the
// same key.
class BiasedRandomWalk : public mx::Primitive {
 public:
  BiasedRandomWalk(
      mx::Stream stream,
      int walk_length,
      double p,
      double q,
      int max_attempts);

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
    return "BiasedRandomWalk";
  }

  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  int walk_length_;
  double p_;
  double q_;
  int max_attempts_;
  // Normalised acceptance probabilities for return / common neighbor /
  // outward moves. float so CPU and Metal compare against the same values.
  float prob_0_;
  float prob_1_;
  float prob_2_;
};

/**
 * node2vec random walks with return parameter `p` and in-out parameter `q`.
 *
 * rowptr, col, start : 1-D int64
 * key                : optional uint32 [2] (e.g. mx.random.key(0)). If not
 *                      given, one is drawn from MLX's global RNG, so
 *                      mx.random.seed makes walks reproducible.
 * max_attempts       : rejection-sampling cap per step.
 *
 * Returns {nodes [len(start), walk_length + 1], edges [len(start),
 * walk_length]}, both int64.
 */
std::vector<mx::array> rejection_sampling(
    const mx::array& rowptr,
    const mx::array& col,
    const mx::array& start,
    int walk_length,
    double p,
    double q,
    std::optional<mx::array> key = std::nullopt,
    int max_attempts = 1000,
    mx::StreamOrDevice s = {});

} // namespace mlx_biased_random_walk
