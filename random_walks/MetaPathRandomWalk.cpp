#include <dlfcn.h>
#include <filesystem>
#include <sstream>
#include <stdexcept>

#include "mlx/backend/cpu/encoder.h"
#include "mlx/ops.h"
#include "mlx/utils.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

#include "random_walks/MetaPathRandomWalk.h"

namespace mlx_metapath_walk {

namespace {

#ifdef _METAL_
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
#endif

void check_int64_1d(const mx::array& a, const char* name) {
  if (a.dtype() != mx::int64 || a.ndim() != 1) {
    std::ostringstream msg;
    msg << "[metapath_random_walk] `" << name
        << "` must be a 1-D int64 array.";
    throw std::invalid_argument(msg.str());
  }
}

} // namespace

// ---------------------------------------------------------------------------
// CPU
// ---------------------------------------------------------------------------

void MetaPathRandomWalk::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& n_out = outputs[0];
  auto& e_out = outputs[1];
  n_out.set_data(mx::allocator::malloc(n_out.nbytes()));
  e_out.set_data(mx::allocator::malloc(e_out.nbytes()));

  auto& enc = mx::cpu::get_command_encoder(stream());
  for (auto& in : inputs) {
    enc.set_input_array(in);
  }
  enc.set_output_array(n_out);
  enc.set_output_array(e_out);

  const int64_t num_starts = inputs[5].size();
  const int L = walk_length_;

  enc.dispatch([rowptr = inputs[0],
                col = inputs[1],
                rowptr_off = inputs[2],
                col_off = inputs[3],
                step_rel = inputs[4],
                start = inputs[5],
                rand = inputs[6],
                n_ptr = n_out.data<int64_t>(),
                e_ptr = e_out.data<int64_t>(),
                num_starts,
                L]() {
    const int64_t* rp = rowptr.data<int64_t>();
    const int64_t* cl = col.data<int64_t>();
    const int64_t* rp_off = rowptr_off.data<int64_t>();
    const int64_t* cl_off = col_off.data<int64_t>();
    const int64_t* rel = step_rel.data<int64_t>();
    const int64_t* st = start.data<int64_t>();
    const float* rnd = rand.data<float>();

    for (int64_t n = 0; n < num_starts; ++n) {
      int64_t cur = st[n];
      n_ptr[n * (L + 1)] = cur;
      for (int l = 0; l < L; ++l) {
        int64_t e_cur = -1;
        if (cur >= 0) {
          const int64_t r = rel[l];
          const int64_t* rp_r = rp + rp_off[r];
          const int64_t row_start = rp_r[cur];
          const int64_t deg = rp_r[cur + 1] - row_start;
          if (deg == 0) {
            cur = -1; // dead end: wrong-typed continuation is not allowed
          } else {
            int64_t idx = static_cast<int64_t>(rnd[n * L + l] * deg);
            idx = idx < deg ? idx : deg - 1; // guard against rand == 1.0
            e_cur = row_start + idx;
            cur = cl[cl_off[r] + e_cur];
          }
        }
        n_ptr[n * (L + 1) + l + 1] = cur;
        e_ptr[n * L + l] = e_cur;
      }
    }
  });
}

// ---------------------------------------------------------------------------
// GPU (Metal): one thread per walk
// ---------------------------------------------------------------------------

#ifdef _METAL_
void MetaPathRandomWalk::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& n_out = outputs[0];
  auto& e_out = outputs[1];
  n_out.set_data(mx::allocator::malloc(n_out.nbytes()));
  e_out.set_data(mx::allocator::malloc(e_out.nbytes()));

  const size_t num_starts = inputs[5].size();
  if (num_starts == 0) {
    return;
  }

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto lib = d.get_library("mlx_cluster", binary_dir());
  auto kernel = d.get_kernel("metapath_random_walk", lib);

  auto& enc = mx::metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(kernel);
  for (int i = 0; i < 7; ++i) {
    enc.set_input_array(inputs[i], i);
  }
  enc.set_output_array(n_out, 7);
  enc.set_output_array(e_out, 8);
  enc.set_bytes(walk_length_, 9);

  size_t tg = std::min<size_t>(kernel->maxTotalThreadsPerThreadgroup(), 256);
  enc.dispatch_threads(MTL::Size(num_starts, 1, 1), MTL::Size(tg, 1, 1));
}
#else
void MetaPathRandomWalk::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[metapath_random_walk] This build has no Metal backend.");
}
#endif

// ---------------------------------------------------------------------------
// Transformations (walks are not differentiable)
// ---------------------------------------------------------------------------

std::vector<mx::array> MetaPathRandomWalk::jvp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("[metapath_random_walk] jvp is not supported.");
}

std::vector<mx::array> MetaPathRandomWalk::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&) {
  throw std::runtime_error("[metapath_random_walk] vjp is not supported.");
}

std::pair<std::vector<mx::array>, std::vector<int>> MetaPathRandomWalk::vmap(
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("[metapath_random_walk] vmap is not supported.");
}

bool MetaPathRandomWalk::is_equivalent(const mx::Primitive& other) const {
  auto* o = dynamic_cast<const MetaPathRandomWalk*>(&other);
  return o && o->walk_length_ == walk_length_;
}

// ---------------------------------------------------------------------------
// Op
// ---------------------------------------------------------------------------

std::vector<mx::array> metapath_random_walk(
    const mx::array& rowptr,
    const mx::array& col,
    const mx::array& rowptr_off,
    const mx::array& col_off,
    const mx::array& step_rel,
    const mx::array& start,
    const mx::array& rand,
    int walk_length,
    mx::StreamOrDevice s) {
  check_int64_1d(rowptr, "rowptr");
  check_int64_1d(col, "col");
  check_int64_1d(rowptr_off, "rowptr_off");
  check_int64_1d(col_off, "col_off");
  check_int64_1d(step_rel, "step_rel");
  check_int64_1d(start, "start");
  if (walk_length < 1) {
    throw std::invalid_argument(
        "[metapath_random_walk] walk_length must be >= 1.");
  }
  if (step_rel.size() != static_cast<size_t>(walk_length)) {
    throw std::invalid_argument(
        "[metapath_random_walk] step_rel must have walk_length entries.");
  }
  if (rowptr_off.size() != col_off.size()) {
    throw std::invalid_argument(
        "[metapath_random_walk] rowptr_off and col_off must match in size.");
  }
  const int num_starts = static_cast<int>(start.size());
  if (rand.dtype() != mx::float32 || rand.ndim() != 2 ||
      rand.shape(0) != num_starts || rand.shape(1) != walk_length) {
    throw std::invalid_argument(
        "[metapath_random_walk] rand must be float32 of shape "
        "[num_starts, walk_length].");
  }

  // The kernels index raw buffers, so inputs must be row-contiguous (views
  // such as the output of mx.repeat or slicing may be strided).
  auto stream = mx::to_stream(s);
  auto c = [&](const mx::array& a) { return mx::contiguous(a, false, stream); };

  auto primitive = std::make_shared<MetaPathRandomWalk>(walk_length, stream);
  return mx::array::make_arrays(
      {{num_starts, walk_length + 1}, {num_starts, walk_length}},
      {mx::int64, mx::int64},
      primitive,
      {c(rowptr), c(col), c(rowptr_off), c(col_off), c(step_rel), c(start),
       c(rand)});
}

} // namespace mlx_metapath_walk
