#include <algorithm>
#include <dlfcn.h>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <vector>

#include "mlx/backend/cpu/encoder.h"
#include "mlx/ops.h"
#include "mlx/utils.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

#include "sparse/GATv2Scores.h"

namespace mlx_sparse {

namespace {

// Same helpers as SpMM.cpp (file-local there).
std::string binary_dir() {
  static std::string dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&binary_dir), &info)) {
      throw std::runtime_error("[mlx_cluster] Unable to locate binary dir.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return dir;
}

void check(bool cond, const std::string& msg) {
  if (!cond) {
    throw std::invalid_argument("[gatv2_scores] " + msg);
  }
}

mx::array as_index(const mx::array& a, mx::StreamOrDevice s) {
  return mx::contiguous(mx::astype(a, mx::int64, s), false, s);
}
mx::array as_float(const mx::array& a, mx::StreamOrDevice s) {
  return mx::contiguous(mx::astype(a, mx::float32, s), false, s);
}

std::pair<size_t, size_t> tg_shape(size_t D) {
  size_t tx = std::min<size_t>(std::max<size_t>(D, 1), 32);
  size_t ty = std::max<size_t>(256 / tx, 1);
  return {tx, ty};
}

inline float leaky(float z, float slope) {
  return z > 0.0f ? z : slope * z;
}
inline float leaky_grad(float z, float slope) {
  return z > 0.0f ? 1.0f : slope;
}

// Position of the CSR arrays in the inputs of GATv2Scores (5) and of
// GATv2ScoresGrad (4: g replaces row and col).
constexpr size_t kCsrForward = 5;
constexpr size_t kCsrGrad = 4;
// rowptr, csr_col, csr_edge, t_rowptr, t_col, t_edge
constexpr size_t kCsrArrays = 6;
// rows, chunk_ptr, chunk_start, chunk_end, owner
constexpr size_t kLongArrays = 5;

} // namespace

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------

void GATv2Scores::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto& enc = mx::cpu::get_command_encoder(stream());
  for (size_t i = 0; i < kCsrForward; ++i) {
    enc.set_input_array(inputs[i]);
  }
  enc.set_output_array(out);

  const int64_t E = inputs[0].size();
  const int64_t H = inputs[2].shape(1);
  const int64_t C = inputs[2].shape(2);
  enc.dispatch([row = inputs[0],
                col = inputs[1],
                x_dst = inputs[2],
                x_src = inputs[3],
                att = inputs[4],
                out_ptr = out.data<float>(),
                E,
                H,
                C,
                slope = slope_]() {
    const int64_t* r = row.data<int64_t>();
    const int64_t* c = col.data<int64_t>();
    const float* xd = x_dst.data<float>();
    const float* xs = x_src.data<float>();
    const float* at = att.data<float>();
    for (int64_t e = 0; e < E; ++e) {
      for (int64_t h = 0; h < H; ++h) {
        const float* d = xd + (r[e] * H + h) * C;
        const float* s = xs + (c[e] * H + h) * C;
        const float* a = at + h * C;
        float acc = 0.0f;
        for (int64_t f = 0; f < C; ++f) {
          acc += a[f] * leaky(s[f] + d[f], slope);
        }
        out_ptr[e * H + h] = acc;
      }
    }
  });
}

#ifdef _METAL_
void GATv2Scores::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  const uint32_t E = static_cast<uint32_t>(inputs[0].size());
  const uint32_t H = static_cast<uint32_t>(inputs[2].shape(1));
  const uint32_t C = static_cast<uint32_t>(inputs[2].shape(2));
  const uint32_t EH = E * H;
  const float slope = slope_;

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto lib = d.get_library("mlx_cluster", binary_dir());
  const std::string idx = inputs[0].dtype() == mx::int32 ? "_i32" : "_i64";
  auto k = d.get_kernel("gatv2_scores" + idx, lib);
  auto& enc = mx::metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(k);
  for (int i = 0; i < 5; ++i) {
    enc.set_input_array(inputs[i], i);
  }
  enc.set_output_array(out, 5);
  enc.set_bytes(H, 6);
  enc.set_bytes(C, 7);
  enc.set_bytes(EH, 8);
  enc.set_bytes(slope, 9);
  // one SIMD group (32 lanes) per (edge, head), 8 of them per threadgroup
  enc.dispatch_threads(MTL::Size(32, EH, 1), MTL::Size(32, 8, 1));
}
#else
void GATv2Scores::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[gatv2_scores] This build has no Metal backend.");
}
#endif

std::vector<mx::array> GATv2Scores::jvp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("[gatv2_scores] jvp is not supported.");
}

std::vector<mx::array> GATv2Scores::vjp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<mx::array>&) {
  auto s = stream();
  const auto& x_dst = primals[2];
  const auto& x_src = primals[3];
  const auto& att = primals[4];
  const auto& g = cotangents[0]; // [E, H]

  std::vector<mx::array> grads; // x_dst, x_src, att
  if (primals[0].size() == 0) {
    grads = {
        mx::zeros_like(x_dst, s), mx::zeros_like(x_src, s), mx::zeros_like(att, s)};
  } else {
    std::vector<mx::array> inputs = {as_float(g, s), x_dst, x_src, att};
    inputs.insert(inputs.end(), primals.begin() + kCsrForward, primals.end());
    auto outs = mx::array::make_arrays(
        {x_dst.shape(), x_src.shape(), x_dst.shape()},
        {mx::float32, mx::float32, mx::float32},
        std::make_shared<GATv2ScoresGrad>(
            s, slope_, has_long_, long_threshold_, has_long_t_, long_threshold_t_),
        std::move(inputs));
    grads = {outs[0], outs[1], mx::sum(outs[2], 0, false, s)};
  }

  std::vector<mx::array> result;
  for (int arg : argnums) {
    if (arg >= 2 && arg <= 4) {
      result.push_back(grads[arg - 2]);
    } else {
      result.push_back(mx::zeros_like(primals[arg], s));
    }
  }
  return result;
}

std::pair<std::vector<mx::array>, std::vector<int>> GATv2Scores::vmap(
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("[gatv2_scores] vmap is not supported.");
}

bool GATv2Scores::is_equivalent(const mx::Primitive& other) const {
  auto* o = dynamic_cast<const GATv2Scores*>(&other);
  return o != nullptr && o->slope_ == slope_ && o->has_long_ == has_long_ &&
      o->long_threshold_ == long_threshold_ && o->has_long_t_ == has_long_t_ &&
      o->long_threshold_t_ == long_threshold_t_;
}

// ---------------------------------------------------------------------------
// Backward
// ---------------------------------------------------------------------------

void GATv2ScoresGrad::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& enc = mx::cpu::get_command_encoder(stream());
  for (auto& o : outputs) {
    o.set_data(mx::allocator::malloc(o.nbytes()));
    enc.set_output_array(o);
  }
  for (size_t i = 0; i < kCsrGrad + kCsrArrays; ++i) {
    enc.set_input_array(inputs[i]);
  }

  const int64_t H = inputs[1].shape(1);
  const int64_t C = inputs[1].shape(2);
  const int64_t Nd = inputs[1].shape(0);
  const int64_t Ns = inputs[2].shape(0);
  enc.dispatch([g = inputs[0],
                x_dst = inputs[1],
                x_src = inputs[2],
                att = inputs[3],
                rowptr = inputs[kCsrGrad],
                csr_col = inputs[kCsrGrad + 1],
                csr_edge = inputs[kCsrGrad + 2],
                t_rowptr = inputs[kCsrGrad + 3],
                t_col = inputs[kCsrGrad + 4],
                t_edge = inputs[kCsrGrad + 5],
                gd = outputs[0].data<float>(),
                gs = outputs[1].data<float>(),
                ap = outputs[2].data<float>(),
                H,
                C,
                Nd,
                Ns,
                slope = slope_]() {
    const int64_t D = H * C;
    const float* gp = g.data<float>();
    const float* at = att.data<float>();
    // One pass per direction, in CSR order, like the GPU kernels.
    auto pass = [&](const mx::array& ptr,
                    const mx::array& nbr,
                    const mx::array& edge,
                    const float* self,
                    const float* other,
                    float* grad,
                    float* att_part,
                    int64_t N) {
      const int64_t* p = ptr.data<int64_t>();
      const int64_t* nb = nbr.data<int64_t>();
      const int64_t* ed = edge.data<int64_t>();
      std::vector<float> acc(D), acc_att(D);
      for (int64_t r = 0; r < N; ++r) {
        std::fill(acc.begin(), acc.end(), 0.0f);
        std::fill(acc_att.begin(), acc_att.end(), 0.0f);
        const float* xs = self + r * D;
        for (int64_t k = p[r]; k < p[r + 1]; ++k) {
          const float* xo = other + nb[k] * D;
          const float* gk = gp + ed[k] * H;
          for (int64_t f = 0; f < D; ++f) {
            const float z = xs[f] + xo[f];
            const float gv = gk[f / C];
            acc[f] += gv * leaky_grad(z, slope);
            if (att_part) {
              acc_att[f] += gv * leaky(z, slope);
            }
          }
        }
        for (int64_t f = 0; f < D; ++f) {
          grad[r * D + f] = at[f] * acc[f];
          if (att_part) {
            att_part[r * D + f] = acc_att[f];
          }
        }
      }
    };
    const float* xd = x_dst.data<float>();
    const float* xs = x_src.data<float>();
    pass(rowptr, csr_col, csr_edge, xd, xs, gd, ap, Nd);
    pass(t_rowptr, t_col, t_edge, xs, xd, gs, nullptr, Ns);
  });
}

#ifdef _METAL_
void GATv2ScoresGrad::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  for (auto& o : outputs) {
    o.set_data(mx::allocator::malloc(o.nbytes()));
  }
  const auto& g = inputs[0];
  const auto& x_dst = inputs[1];
  const auto& x_src = inputs[2];
  const auto& att = inputs[3];
  const uint32_t H = static_cast<uint32_t>(x_dst.shape(1));
  const uint32_t C = static_cast<uint32_t>(x_dst.shape(2));
  const uint32_t D = H * C;
  const float slope = slope_;
  if (D == 0) {
    return;
  }

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto lib = d.get_library("mlx_cluster", binary_dir());
  auto& enc = mx::metal::get_command_encoder(s);
  auto [tx, ty] = tg_shape(D);
  size_t fx = 1;
  while (fx < D && fx < 32) {
    fx <<= 1;
  }
  const size_t ey = 256 / fx;

  // edge ids: uint32 (argsort, as coo_to_csr gives them) or int64
  const std::string edge_t =
      inputs[kCsrGrad + 2].dtype() == mx::uint32 ? "_u32" : "_i64";
  const size_t long_at = kCsrGrad + kCsrArrays;
  const size_t long_t_at = long_at + (has_long_ ? kLongArrays : 0);

  // One direction: grad (and, for the dst pass, att_part) of every node.
  auto pass = [&](bool with_att,
                  const mx::array& rowptr,
                  const mx::array& nbr,
                  const mx::array& edge,
                  const mx::array& x_self,
                  const mx::array& x_other,
                  mx::array& grad,
                  mx::array* att_part,
                  bool has_long,
                  int long_threshold,
                  size_t long_offset) {
    const uint32_t N = static_cast<uint32_t>(x_self.shape(0));
    if (N == 0) {
      return;
    }
    const std::string suffix = with_att ? "_att" : "";

    // 1) short rows: one thread per (feature, node)
    const uint32_t threshold = has_long
        ? static_cast<uint32_t>(long_threshold)
        : std::numeric_limits<uint32_t>::max();
    auto k = d.get_kernel("gatv2_grad_rows" + suffix + edge_t, lib);
    enc.set_compute_pipeline_state(k);
    enc.set_input_array(rowptr, 0);
    enc.set_input_array(nbr, 1);
    enc.set_input_array(edge, 2);
    enc.set_input_array(x_self, 3);
    enc.set_input_array(x_other, 4);
    enc.set_input_array(att, 5);
    enc.set_input_array(g, 6);
    enc.set_output_array(grad, 7);
    if (with_att) {
      enc.set_output_array(*att_part, 8);
    } else {
      enc.set_input_array(att, 8); // unused slot
    }
    enc.set_bytes(H, 9);
    enc.set_bytes(C, 10);
    enc.set_bytes(N, 11);
    enc.set_bytes(threshold, 12);
    enc.set_bytes(slope, 13);
    enc.dispatch_threads(MTL::Size(D, N, 1), MTL::Size(tx, ty, 1));

    if (!has_long) {
      return;
    }
    const auto& rows = inputs[long_offset];
    const auto& chunk_ptr = inputs[long_offset + 1];
    const auto& chunk_start = inputs[long_offset + 2];
    const auto& chunk_end = inputs[long_offset + 3];
    const auto& owner = inputs[long_offset + 4];
    const uint32_t L = static_cast<uint32_t>(rows.size());
    const uint32_t K = static_cast<uint32_t>(chunk_start.size());
    if (L == 0 || K == 0) {
      return;
    }

    // 2) long rows, phase 1: one threadgroup per (feature tile, chunk)
    auto make_partials = [&]() {
      mx::array p({static_cast<int>(K), static_cast<int>(D)}, mx::float32, nullptr, {});
      p.set_data(mx::allocator::malloc(p.nbytes()));
      enc.add_temporary(p);
      return p;
    };
    auto partials = make_partials();
    auto att_partials = with_att ? make_partials() : partials;
    auto pk = d.get_kernel("gatv2_grad_long_partial" + suffix + edge_t, lib);
    enc.set_compute_pipeline_state(pk);
    enc.set_input_array(chunk_start, 0);
    enc.set_input_array(chunk_end, 1);
    enc.set_input_array(owner, 2);
    enc.set_input_array(rows, 3);
    enc.set_input_array(nbr, 4);
    enc.set_input_array(edge, 5);
    enc.set_input_array(x_self, 6);
    enc.set_input_array(x_other, 7);
    enc.set_input_array(g, 8);
    enc.set_output_array(partials, 9);
    if (with_att) {
      enc.set_output_array(att_partials, 10);
    } else {
      enc.set_input_array(att, 10); // unused slot
    }
    enc.set_bytes(H, 11);
    enc.set_bytes(C, 12);
    enc.set_bytes(slope, 13);
    enc.dispatch_threadgroups(
        MTL::Size((D + fx - 1) / fx, K, 1), MTL::Size(fx, ey, 1));

    // 3) long rows, phase 2: add each row's chunk sums in order
    auto rk = d.get_kernel("gatv2_grad_long_reduce" + suffix, lib);
    enc.set_compute_pipeline_state(rk);
    enc.set_input_array(rows, 0);
    enc.set_input_array(chunk_ptr, 1);
    enc.set_input_array(partials, 2);
    enc.set_input_array(att_partials, 3);
    enc.set_input_array(att, 4);
    enc.set_output_array(grad, 5);
    if (with_att) {
      enc.set_output_array(*att_part, 6);
    } else {
      enc.set_input_array(att, 6); // unused slot
    }
    enc.set_bytes(D, 7);
    enc.set_bytes(L, 8);
    enc.dispatch_threads(MTL::Size(D, L, 1), MTL::Size(tx, ty, 1));
  };

  pass(true,
       inputs[kCsrGrad],
       inputs[kCsrGrad + 1],
       inputs[kCsrGrad + 2],
       x_dst,
       x_src,
       outputs[0],
       &outputs[2],
       has_long_,
       long_threshold_,
       long_at);
  pass(false,
       inputs[kCsrGrad + 3],
       inputs[kCsrGrad + 4],
       inputs[kCsrGrad + 5],
       x_src,
       x_dst,
       outputs[1],
       nullptr,
       has_long_t_,
       long_threshold_t_,
       long_t_at);
}
#else
void GATv2ScoresGrad::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[gatv2_scores] This build has no Metal backend.");
}
#endif

std::vector<mx::array> GATv2ScoresGrad::jvp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("[gatv2_scores] jvp of the gradient is not supported.");
}

std::vector<mx::array> GATv2ScoresGrad::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&) {
  throw std::runtime_error(
      "[gatv2_scores] second-order gradients are not supported.");
}

std::pair<std::vector<mx::array>, std::vector<int>> GATv2ScoresGrad::vmap(
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("[gatv2_scores] vmap is not supported.");
}

bool GATv2ScoresGrad::is_equivalent(const mx::Primitive& other) const {
  auto* o = dynamic_cast<const GATv2ScoresGrad*>(&other);
  return o != nullptr && o->slope_ == slope_ && o->has_long_ == has_long_ &&
      o->long_threshold_ == long_threshold_ && o->has_long_t_ == has_long_t_ &&
      o->long_threshold_t_ == long_threshold_t_;
}

// ---------------------------------------------------------------------------
// Op
// ---------------------------------------------------------------------------

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
    float negative_slope,
    const std::optional<mx::array>& t_edge_in,
    const std::optional<mx::array>& owner_in,
    const std::optional<mx::array>& t_owner_in,
    mx::StreamOrDevice s) {
  check(row.ndim() == 1 && col.ndim() == 1, "row and col must be 1-D.");
  check(row.size() == col.size(), "row and col must have the same length.");
  check(
      x_dst.ndim() == 3 && x_src.ndim() == 3,
      "x_dst and x_src must be 3-D [num_nodes, heads, channels].");
  check(
      x_dst.shape(1) == x_src.shape(1) && x_dst.shape(2) == x_src.shape(2),
      "x_dst and x_src must have the same number of heads and channels.");
  check(
      att.ndim() == 2 && att.shape(0) == x_dst.shape(1) &&
          att.shape(1) == x_dst.shape(2),
      "att must be [heads, channels].");
  const size_t E = row.size();
  check(
      rowptr.ndim() == 1 && rowptr.size() == static_cast<size_t>(x_dst.shape(0)) + 1,
      "csr.rowptr must have x_dst.shape[0] + 1 entries.");
  check(
      transpose.rowptr.ndim() == 1 &&
          transpose.rowptr.size() == static_cast<size_t>(x_src.shape(0)) + 1,
      "csr.transpose.rowptr must have x_src.shape[0] + 1 entries.");
  check(
      csr_col.size() == E && csr_perm.size() == E && transpose.col.size() == E &&
          transpose.perm.size() == E,
      "the CSR must have one entry per edge (is it the CSR of these edges?).");

  // The GPU kernels take the index types the caller has (see gatv2.metal),
  // so nothing is converted on every call: the edge list as int32 or int64,
  // the CSR edge ids as uint32 or int64. The CPU code reads int64 only.
  const bool on_cpu = mx::to_stream(s).device == mx::Device::cpu;
  auto same_type = [&](const mx::array& a,
                       const mx::array& b,
                       std::initializer_list<mx::Dtype> kept) {
    bool keep = !on_cpu && a.dtype() == b.dtype() &&
        std::find(kept.begin(), kept.end(), a.dtype()) != kept.end();
    if (keep) {
      return std::make_pair(
          mx::contiguous(a, false, s), mx::contiguous(b, false, s));
    }
    return std::make_pair(as_index(a, s), as_index(b, s));
  };
  auto [row_k, col_k] = same_type(row, col, {mx::int32, mx::int64});

  // Original edge id of each transposed edge: precomputed by coo_to_csr
  // (CSRTranspose.edge), otherwise perm[t_perm] on every call.
  // (take() rejects empty arrays.)
  mx::array t_edge = t_edge_in ? *t_edge_in
      : E == 0                 ? mx::zeros({0}, mx::uint32, s)
                               : mx::take(csr_perm, as_index(transpose.perm, s), 0, s);
  check(t_edge.size() == E, "csr.transpose.edge must have one entry per edge.");
  auto [csr_edge_k, t_edge_k] =
      same_type(csr_perm, t_edge, {mx::uint32, mx::int64});

  std::vector<mx::array> inputs = {
      row_k,
      col_k,
      as_float(x_dst, s),
      as_float(x_src, s),
      as_float(att, s),
      as_index(rowptr, s),
      as_index(csr_col, s),
      csr_edge_k,
      as_index(transpose.rowptr, s),
      as_index(transpose.col, s),
      t_edge_k};
  auto push_long = [&](const LongRows& lr, const std::optional<mx::array>& owner) {
    check(lr.threshold >= 0, "long_row_threshold must be >= 0.");
    check(
        lr.chunk_ptr.size() == lr.rows.size() + 1,
        "chunk_ptr must have len(long_rows) + 1 entries.");
    check(
        lr.chunk_start.size() == lr.chunk_end.size(),
        "chunk_start and chunk_end must have the same length.");
    auto chunk_ptr = as_index(lr.chunk_ptr, s);
    inputs.push_back(as_index(lr.rows, s));
    inputs.push_back(chunk_ptr);
    inputs.push_back(as_index(lr.chunk_start, s));
    inputs.push_back(as_index(lr.chunk_end, s));
    // the long row (index into rows) of every chunk: precomputed by
    // build_long_rows (LongRows.owner), otherwise from chunk_ptr
    if (owner) {
      check(
          owner->size() == lr.chunk_start.size(),
          "long_rows.owner must have one entry per chunk.");
      inputs.push_back(as_index(*owner, s));
    } else {
      inputs.push_back(csr_to_row(chunk_ptr, lr.chunk_start.size(), s));
    }
  };
  if (long_rows) {
    push_long(*long_rows, owner_in);
  }
  if (transpose.long_rows) {
    push_long(*transpose.long_rows, t_owner_in);
  }
  return mx::array(
      {static_cast<int>(E), x_dst.shape(1)},
      mx::float32,
      std::make_shared<GATv2Scores>(
          mx::to_stream(s),
          negative_slope,
          long_rows.has_value(),
          long_rows ? long_rows->threshold : 0,
          transpose.long_rows.has_value(),
          transpose.long_rows ? transpose.long_rows->threshold : 0),
      std::move(inputs));
}

} // namespace mlx_sparse
