#pragma once

#include <optional>

#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/primitives.h>

#include "sparse/SpMM.h"

namespace mlx_sparse {

/**
 * GATv2 attention scores (Brody et al., "How Attentive are Graph Attention
 * Networks?"), one per edge and head:
 *
 *   out[e, h] = sum_c att[h, c] * leaky_relu(x_src[col[e], h, c]
 *                                            + x_dst[row[e], h, c])
 *
 * row = destination node, col = source node (as in SpMM).
 *
 * Inputs (in order):
 *    0: row      int32|int64 [E]   (int64 on the CPU)
 *    1: col      int32|int64 [E]
 *    2: x_dst    float32 [num_dst, H, C]
 *    3: x_src    float32 [num_src, H, C]
 *    4: att      float32 [H, C]
 *   CSR of the same edges (only read by the backward pass):
 *    5: rowptr   int64 [num_dst + 1]  edges grouped by destination
 *    6: csr_col  int64 [E]            source node of each CSR edge
 *    7: csr_edge uint32|int64 [E]     original edge id of each CSR edge
 *    8: t_rowptr int64 [num_src + 1]  edges grouped by source
 *    9: t_col    int64 [E]            destination node of each transposed edge
 *   10: t_edge   uint32|int64 [E]     original edge id of each transposed edge
 *   then, with has_long:   rows, chunk_ptr, chunk_start, chunk_end, owner
 *   then, with has_long_t: the same 5 arrays for the transpose
 *   (owner[k] = index into rows of the long row that chunk k belongs to).
 *
 * Output: float32 [E, H]
 *
 * Differentiable w.r.t. x_dst, x_src and att. The backward pass
 * (GATv2ScoresGrad) uses the CSR, with no atomics and no [E, H, C] array.
 */
class GATv2Scores : public mx::Primitive {
 public:
  GATv2Scores(
      mx::Stream stream,
      float slope,
      bool has_long,
      int long_threshold,
      bool has_long_t,
      int long_threshold_t)
      : mx::Primitive(stream),
        slope_(slope),
        has_long_(has_long),
        long_threshold_(long_threshold),
        has_long_t_(has_long_t),
        long_threshold_t_(long_threshold_t) {}

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
    return "GATv2Scores";
  }

  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  float slope_;
  bool has_long_;
  int long_threshold_;
  bool has_long_t_;
  int long_threshold_t_;
};

/**
 * Backward of GATv2Scores. With dz[e] = g[e, h] * att[h] * leaky_relu'(z[e]):
 *
 *   grad_x_dst[i] = sum_{e : row[e] == i} dz[e]
 *   grad_x_src[j] = sum_{e : col[e] == j} dz[e]
 *   att_part[i]   = sum_{e : row[e] == i} g[e, h] * leaky_relu(z[e])
 *
 * (grad_att = att_part.sum(0), node-sized partial sums keep it
 * deterministic). One thread per (node, feature) on the CSR; hub nodes are
 * split into chunks (see LongRows).
 *
 * Inputs: g float32 [E, H], x_dst, x_src, att, then inputs 5... of
 *         GATv2Scores (the CSR and long-row arrays).
 * Outputs: grad_x_dst [num_dst, H, C], grad_x_src [num_src, H, C],
 *          att_part [num_dst, H, C]. Not differentiable itself.
 */
class GATv2ScoresGrad : public mx::Primitive {
 public:
  GATv2ScoresGrad(
      mx::Stream stream,
      float slope,
      bool has_long,
      int long_threshold,
      bool has_long_t,
      int long_threshold_t)
      : mx::Primitive(stream),
        slope_(slope),
        has_long_(has_long),
        long_threshold_(long_threshold),
        has_long_t_(has_long_t),
        long_threshold_t_(long_threshold_t) {}

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
    return "GATv2ScoresGrad";
  }

  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  float slope_;
  bool has_long_;
  int long_threshold_;
  bool has_long_t_;
  int long_threshold_t_;
};

/**
 * out[e, h] = sum_c att[h, c] * leaky_relu(x_src[col[e], h, c] +
 * x_dst[row[e], h, c]) (see GATv2Scores).
 *
 * rowptr / csr_col / csr_perm / transpose / long_rows are the CSR of (row,
 * col) as built by mlx_cluster.coo_to_csr (csr_perm[k] = original id of CSR
 * edge k). They are only used by the backward pass. t_edge (original id of
 * each transposed edge, csr_perm[transpose.perm]) and owner / t_owner (long
 * row of each chunk) depend only on the graph: pass them (coo_to_csr
 * precomputes them) to avoid recomputing them on every call.
 */
mx::array gatv2_scores(
    const mx::array& row,
    const mx::array& col,
    const mx::array& x_dst,
    const mx::array& x_src,
    const mx::array& att,
    const mx::array& rowptr,
    const mx::array& csr_col,
    const mx::array& csr_perm,
    const CSRTranspose& transpose,
    const std::optional<LongRows>& long_rows,
    float negative_slope = 0.2f,
    const std::optional<mx::array>& t_edge = std::nullopt,
    const std::optional<mx::array>& owner = std::nullopt,
    const std::optional<mx::array>& t_owner = std::nullopt,
    mx::StreamOrDevice s = {});

} // namespace mlx_sparse
