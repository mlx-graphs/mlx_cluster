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

#include "random_walks/RandomWalk.h"

namespace mlx_random_walk {

namespace {

#ifdef _METAL_
std::string current_binary_dir() {
  static std::string binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&current_binary_dir), &info)) {
      throw std::runtime_error("Unable to get current binary dir.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return binary_dir;
}
#endif

void check_int64_1d(const mx::array& a, const char* name) {
  if (a.dtype() != mx::int64 || a.ndim() != 1) {
    std::ostringstream msg;
    msg << "[random_walk] `" << name << "` must be a 1-D int64 array (got "
        << a.ndim() << "-D " << a.dtype() << ").";
    throw std::invalid_argument(msg.str());
  }
}

} // namespace

// ---------------------------------------------------------------------------
// CPU
// ---------------------------------------------------------------------------

void RandomWalk::eval_cpu(
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

  const int64_t numel = inputs[2].size();
  const int L = walk_length_;

  enc.dispatch([rowptr = inputs[0],
                col = inputs[1],
                start = inputs[2],
                rand = inputs[3],
                n_ptr = n_out.data<int64_t>(),
                e_ptr = e_out.data<int64_t>(),
                numel,
                L]() {
    const int64_t* rp = rowptr.data<int64_t>();
    const int64_t* cl = col.data<int64_t>();
    const int64_t* st = start.data<int64_t>();
    const float* rnd = rand.data<float>();

    for (int64_t n = 0; n < numel; ++n) {
      int64_t cur = st[n];
      n_ptr[n * (L + 1)] = cur;
      for (int l = 0; l < L; ++l) {
        const int64_t row_start = rp[cur];
        const int64_t deg = rp[cur + 1] - row_start;
        int64_t e_cur = -1;
        if (deg > 0) {
          int64_t idx = static_cast<int64_t>(rnd[n * L + l] * deg);
          idx = idx < deg ? idx : deg - 1; // guard against rand == 1.0
          e_cur = row_start + idx;
          cur = cl[e_cur];
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
void RandomWalk::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& n_out = outputs[0];
  auto& e_out = outputs[1];
  n_out.set_data(mx::allocator::malloc(n_out.nbytes()));
  e_out.set_data(mx::allocator::malloc(e_out.nbytes()));

  const size_t numel = inputs[2].size();
  if (numel == 0) {
    return;
  }

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto lib = d.get_library("mlx_cluster", current_binary_dir());
  auto kernel = d.get_kernel("random_walk", lib);

  auto& enc = mx::metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(kernel);
  enc.set_input_array(inputs[0], 0);
  enc.set_input_array(inputs[1], 1);
  enc.set_input_array(inputs[2], 2);
  enc.set_input_array(inputs[3], 3);
  enc.set_output_array(n_out, 4);
  enc.set_output_array(e_out, 5);
  enc.set_bytes(walk_length_, 6);

  size_t tg = std::min<size_t>(kernel->maxTotalThreadsPerThreadgroup(), 256);
  enc.dispatch_threads(MTL::Size(numel, 1, 1), MTL::Size(tg, 1, 1));
}
#else
void RandomWalk::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[random_walk] This build has no Metal backend.");
}
#endif

// ---------------------------------------------------------------------------
// Transformations (walks are not differentiable)
// ---------------------------------------------------------------------------

std::vector<mx::array> RandomWalk::jvp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("[random_walk] Random walks are not differentiable.");
}

std::vector<mx::array> RandomWalk::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&) {
  throw std::runtime_error("[random_walk] Random walks are not differentiable.");
}

std::pair<std::vector<mx::array>, std::vector<int>> RandomWalk::vmap(
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("[random_walk] vmap is not supported.");
}

bool RandomWalk::is_equivalent(const mx::Primitive& other) const {
  // Randomness comes in through `rand`, so equal inputs give equal walks and
  // MLX may safely merge two identical calls.
  auto* o = dynamic_cast<const RandomWalk*>(&other);
  return o && o->walk_length_ == walk_length_;
}

// ---------------------------------------------------------------------------
// Op
// ---------------------------------------------------------------------------

std::vector<mx::array> random_walk(
    const mx::array& rowptr,
    const mx::array& col,
    const mx::array& start,
    const mx::array& rand,
    int walk_length,
    mx::StreamOrDevice s) {
  check_int64_1d(rowptr, "rowptr");
  check_int64_1d(col, "col");
  check_int64_1d(start, "start");
  if (walk_length < 1) {
    throw std::invalid_argument("[random_walk] walk_length must be >= 1.");
  }
  const int numel = static_cast<int>(start.size());
  if (rand.dtype() != mx::float32 || rand.ndim() != 2 ||
      rand.shape(0) != numel || rand.shape(1) != walk_length) {
    std::ostringstream msg;
    msg << "[random_walk] `rand` must be float32 with shape [" << numel << ", "
        << walk_length << "] (got " << rand.dtype() << " " << rand.shape()
        << ").";
    throw std::invalid_argument(msg.str());
  }

  // The kernels index raw buffers, so inputs must be row-contiguous (views
  // such as the output of mx.repeat or slicing may be strided).
  auto stream = mx::to_stream(s);
  auto c = [&](const mx::array& a) { return mx::contiguous(a, false, stream); };

  auto primitive = std::make_shared<RandomWalk>(walk_length, stream);
  return mx::array::make_arrays(
      {{numel, walk_length + 1}, {numel, walk_length}},
      {mx::int64, mx::int64},
      primitive,
      {c(rowptr), c(col), c(start), c(rand)});
}

} // namespace mlx_random_walk
