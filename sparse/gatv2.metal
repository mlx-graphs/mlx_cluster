// GATv2 attention scores and their backward pass.
//
//   z[e, h, c]   = x_src[col[e], h, c] + x_dst[row[e], h, c]
//   out[e, h]    = sum_c att[h, c] * leaky_relu(z[e, h, c])
//
// row = destination node, col = source node (as in the SpMM kernels).
//
// Backward, with dz[e, h, c] = g[e, h] * att[h, c] * leaky_relu'(z):
//
//   grad_x_dst[i] = sum_{e : row[e] == i} dz[e]      (CSR, edges by dst)
//   grad_x_src[j] = sum_{e : col[e] == j} dz[e]      (transposed CSR)
//   grad_att      = sum_e g[e, h] * leaky_relu(z[e])  (per node, then summed)
//
// The backward pass has no atomics: one thread per (node, feature) loops over
// the node's edges, keeps the sum in a register and writes it once, as in
// spmm_csr. Hub nodes ("long rows", more than `long_threshold` edges) are cut
// into chunks: each chunk is reduced by its own threadgroup, and the chunk
// sums of a node are added in a fixed order. Results are deterministic.
//
// leaky_relu'(0) is the slope, as in mlx-graphs (where at z == 0 only the
// linear term of the reference contributes).
//
// Index types are template parameters, so the host passes the arrays it has
// without a conversion on every call: the edge list (row, col) as int32 or
// int64 (mlx-graphs uses int32), the edge ids of the CSR (`edge`) as uint32
// (from argsort, as coo_to_csr returns them) or int64. CSR offsets and
// neighbours are int64, as in spmm_csr.

#include <metal_stdlib>
using namespace metal;

inline float leaky(float z, float slope) {
  return z > 0.0f ? z : slope * z;
}

inline float leaky_grad(float z, float slope) {
  return z > 0.0f ? 1.0f : slope;
}

// ---------------------------------------------------------------------------
// Forward: one SIMD group per (edge, head), lanes stride over the channels,
// then simd_sum. The host dispatches a (32, H, E) grid with (32, th, tz)
// threadgroups: x is the lane, so the 32 lanes of a SIMD group share
// (head, edge) and exit together, and neither needs an integer division.
//
// CH > 0 fixes the channel count at compile time, so the channel loop is
// unrolled (the host picks it for common channel counts); CH == 0 reads
// the count from C.
// ---------------------------------------------------------------------------
template <typename I, uint CH>
[[kernel]] void gatv2_scores(
    const device I* row [[buffer(0)]],
    const device I* col [[buffer(1)]],
    const device float* x_dst [[buffer(2)]],
    const device float* x_src [[buffer(3)]],
    const device float* att [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant uint& H [[buffer(6)]],
    constant uint& C_runtime [[buffer(7)]],
    constant uint& E [[buffer(8)]],
    constant float& slope [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint h = gid.y;
  const uint e = gid.z;
  if (h >= H || e >= E) {
    return;
  }
  const uint C = CH > 0 ? CH : C_runtime;
  const ulong d = (static_cast<ulong>(row[e]) * H + h) * C;
  const ulong s = (static_cast<ulong>(col[e]) * H + h) * C;
  const ulong a = static_cast<ulong>(h) * C;
  float acc = 0.0f;
  for (uint c = lane; c < C; c += 32) {
    const float z = x_src[s + c] + x_dst[d + c];
    acc += att[a + c] * leaky(z, slope);
  }
  acc = simd_sum(acc);
  if (lane == 0) {
    out[static_cast<ulong>(e) * H + h] = acc;
  }
}

// ---------------------------------------------------------------------------
// Backward, short rows: one thread per (feature f, node r), f = h * C + c on
// the fast axis. The same kernel serves both directions:
//   dst pass: self = x_dst, other = x_src, CSR by destination, WITH_ATT
//   src pass: self = x_src, other = x_dst, transposed CSR
// nbr[k] is the other endpoint of CSR edge k and edge[k] its original id
// (the index into g). Rows with more than long_threshold edges are skipped
// and written by the long-row kernels below.
// ---------------------------------------------------------------------------
template <bool WITH_ATT, typename E>
[[kernel]] void gatv2_grad_rows(
    const device long* rowptr [[buffer(0)]],
    const device long* nbr [[buffer(1)]],
    const device E* edge [[buffer(2)]],
    const device float* x_self [[buffer(3)]],
    const device float* x_other [[buffer(4)]],
    const device float* att [[buffer(5)]],
    const device float* g [[buffer(6)]],
    device float* grad [[buffer(7)]],
    device float* att_part [[buffer(8)]],
    constant uint& H [[buffer(9)]],
    constant uint& C [[buffer(10)]],
    constant uint& N [[buffer(11)]],
    constant uint& long_threshold [[buffer(12)]],
    constant float& slope [[buffer(13)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint D = H * C;
  const uint f = gid.x;
  const uint r = gid.y;
  if (f >= D || r >= N) {
    return;
  }
  const long start = rowptr[r];
  const long end = rowptr[r + 1];
  if (end - start > static_cast<long>(long_threshold)) {
    return; // long row: written by gatv2_grad_long_reduce
  }
  const uint h = f / C;
  const float xs = x_self[static_cast<ulong>(r) * D + f];
  float acc = 0.0f;
  float acc_att = 0.0f;
  for (long k = start; k < end; ++k) {
    const float z = xs + x_other[static_cast<ulong>(nbr[k]) * D + f];
    const float gk = g[static_cast<ulong>(edge[k]) * H + h];
    acc += gk * leaky_grad(z, slope);
    if (WITH_ATT) {
      acc_att += gk * leaky(z, slope);
    }
  }
  grad[static_cast<ulong>(r) * D + f] = att[f] * acc;
  if (WITH_ATT) {
    att_part[static_cast<ulong>(r) * D + f] = acc_att;
  }
}

// ---------------------------------------------------------------------------
// Backward, long rows, phase 1: one threadgroup per (feature tile, chunk).
// Threads are (FX features) x (EY edge lanes), FX * EY = 256; each thread
// sums every EY-th edge of the chunk for one feature, then the EY partial
// sums are combined in threadgroup memory -> partials[chunk, f].
// owner[k] is the index (into rows) of the long row that chunk k belongs to.
// ---------------------------------------------------------------------------
template <bool WITH_ATT, typename E>
[[kernel]] void gatv2_grad_long_partial(
    const device long* chunk_start [[buffer(0)]],
    const device long* chunk_end [[buffer(1)]],
    const device long* owner [[buffer(2)]],
    const device long* rows [[buffer(3)]],
    const device long* nbr [[buffer(4)]],
    const device E* edge [[buffer(5)]],
    const device float* x_self [[buffer(6)]],
    const device float* x_other [[buffer(7)]],
    const device float* g [[buffer(8)]],
    device float* partials [[buffer(9)]],
    device float* att_partials [[buffer(10)]],
    constant uint& H [[buffer(11)]],
    constant uint& C [[buffer(12)]],
    constant float& slope [[buffer(13)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tsize [[threads_per_threadgroup]]) {
  threadgroup float scratch[256];
  threadgroup float scratch_att[256];
  const uint D = H * C;
  const uint f = tg.x * tsize.x + lid.x;
  const uint k = tg.y;
  float acc = 0.0f;
  float acc_att = 0.0f;
  if (f < D) {
    const ulong r = static_cast<ulong>(rows[owner[k]]);
    const uint h = f / C;
    const float xs = x_self[r * D + f];
    const long e1 = chunk_end[k];
    for (long q = chunk_start[k] + lid.y; q < e1; q += tsize.y) {
      const float z = xs + x_other[static_cast<ulong>(nbr[q]) * D + f];
      const float gq = g[static_cast<ulong>(edge[q]) * H + h];
      acc += gq * leaky_grad(z, slope);
      if (WITH_ATT) {
        acc_att += gq * leaky(z, slope);
      }
    }
  }
  // no early return above: every thread must reach the barrier
  const uint slot = lid.y * tsize.x + lid.x;
  scratch[slot] = acc;
  if (WITH_ATT) {
    scratch_att[slot] = acc_att;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid.y == 0 && f < D) {
    float total = 0.0f;
    float total_att = 0.0f;
    for (uint j = 0; j < tsize.y; ++j) {
      total += scratch[j * tsize.x + lid.x];
      if (WITH_ATT) {
        total_att += scratch_att[j * tsize.x + lid.x];
      }
    }
    partials[static_cast<ulong>(k) * D + f] = total;
    if (WITH_ATT) {
      att_partials[static_cast<ulong>(k) * D + f] = total_att;
    }
  }
}

// ---------------------------------------------------------------------------
// Backward, long rows, phase 2: one thread per (feature, long row). Adds the
// row's chunk sums in a fixed order (deterministic) and writes the row.
// ---------------------------------------------------------------------------
template <bool WITH_ATT>
[[kernel]] void gatv2_grad_long_reduce(
    const device long* rows [[buffer(0)]],
    const device long* chunk_ptr [[buffer(1)]],
    const device float* partials [[buffer(2)]],
    const device float* att_partials [[buffer(3)]],
    const device float* att [[buffer(4)]],
    device float* grad [[buffer(5)]],
    device float* att_part [[buffer(6)]],
    constant uint& D [[buffer(7)]],
    constant uint& L [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint f = gid.x;
  const uint l = gid.y;
  if (f >= D || l >= L) {
    return;
  }
  float total = 0.0f;
  float total_att = 0.0f;
  for (long k = chunk_ptr[l]; k < chunk_ptr[l + 1]; ++k) {
    total += partials[static_cast<ulong>(k) * D + f];
    if (WITH_ATT) {
      total_att += att_partials[static_cast<ulong>(k) * D + f];
    }
  }
  const ulong r = static_cast<ulong>(rows[l]);
  grad[r * D + f] = att[f] * total;
  if (WITH_ATT) {
    att_part[r * D + f] = total_att;
  }
}

#define instantiate_kernel(name, func, ...) \
  template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) \
      func<__VA_ARGS__>;

instantiate_kernel("gatv2_scores_i32_c0", gatv2_scores, int, 0)
instantiate_kernel("gatv2_scores_i32_c8", gatv2_scores, int, 8)
instantiate_kernel("gatv2_scores_i32_c16", gatv2_scores, int, 16)
instantiate_kernel("gatv2_scores_i32_c32", gatv2_scores, int, 32)
instantiate_kernel("gatv2_scores_i32_c64", gatv2_scores, int, 64)
instantiate_kernel("gatv2_scores_i32_c128", gatv2_scores, int, 128)
instantiate_kernel("gatv2_scores_i64_c0", gatv2_scores, long, 0)
instantiate_kernel("gatv2_scores_i64_c8", gatv2_scores, long, 8)
instantiate_kernel("gatv2_scores_i64_c16", gatv2_scores, long, 16)
instantiate_kernel("gatv2_scores_i64_c32", gatv2_scores, long, 32)
instantiate_kernel("gatv2_scores_i64_c64", gatv2_scores, long, 64)
instantiate_kernel("gatv2_scores_i64_c128", gatv2_scores, long, 128)
instantiate_kernel("gatv2_grad_rows_att_u32", gatv2_grad_rows, true, uint)
instantiate_kernel("gatv2_grad_rows_att_i64", gatv2_grad_rows, true, long)
instantiate_kernel("gatv2_grad_rows_u32", gatv2_grad_rows, false, uint)
instantiate_kernel("gatv2_grad_rows_i64", gatv2_grad_rows, false, long)
instantiate_kernel("gatv2_grad_long_partial_att_u32", gatv2_grad_long_partial, true, uint)
instantiate_kernel("gatv2_grad_long_partial_att_i64", gatv2_grad_long_partial, true, long)
instantiate_kernel("gatv2_grad_long_partial_u32", gatv2_grad_long_partial, false, uint)
instantiate_kernel("gatv2_grad_long_partial_i64", gatv2_grad_long_partial, false, long)
instantiate_kernel("gatv2_grad_long_reduce_att", gatv2_grad_long_reduce, true)
instantiate_kernel("gatv2_grad_long_reduce", gatv2_grad_long_reduce, false)
