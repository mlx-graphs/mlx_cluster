#include <algorithm>
#include <dlfcn.h>
#include <filesystem>
#include <sstream>
#include <stdexcept>

#include "mlx/backend/cpu/encoder.h"
#include "mlx/ops.h"
#include "mlx/random.h"
#include "mlx/utils.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

#include "random_walks/BiasedRandomWalk.h"
#include "random_walks/node2vec_walk.h"

namespace mlx_biased_random_walk {

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
    msg << "[rejection_sampling] `" << name
        << "` must be a 1-D int64 array (got " << a.ndim() << "-D "
        << a.dtype() << ").";
    throw std::invalid_argument(msg.str());
  }
}

} // namespace

BiasedRandomWalk::BiasedRandomWalk(
    mx::Stream stream,
    int walk_length,
    double p,
    double q,
    int max_attempts)
    : mx::Primitive(stream),
      walk_length_(walk_length),
      p_(p),
      q_(q),
      max_attempts_(max_attempts) {
  const double max_prob = std::max({1.0 / p, 1.0, 1.0 / q});
  prob_0_ = static_cast<float>(1.0 / p / max_prob);
  prob_1_ = static_cast<float>(1.0 / max_prob);
  prob_2_ = static_cast<float>(1.0 / q / max_prob);
}

// ---------------------------------------------------------------------------
// CPU
// ---------------------------------------------------------------------------

void BiasedRandomWalk::eval_cpu(
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

  enc.dispatch([rowptr = inputs[0],
                col = inputs[1],
                start = inputs[2],
                key = inputs[3],
                n_ptr = n_out.data<int64_t>(),
                e_ptr = e_out.data<int64_t>(),
                numel,
                L = walk_length_,
                p0 = prob_0_,
                p1 = prob_1_,
                p2 = prob_2_,
                max_attempts = max_attempts_]() {
    const int64_t* rp = rowptr.data<int64_t>();
    const int64_t* cl = col.data<int64_t>();
    const int64_t* st = start.data<int64_t>();
    const uint32_t* k = key.data<uint32_t>();

    for (int64_t n = 0; n < numel; ++n) {
      mlxc_node2vec_walk(
          rp,
          cl,
          st[n],
          n_ptr + n * (L + 1),
          e_ptr + n * L,
          static_cast<uint64_t>(n),
          L,
          k[0],
          k[1],
          p0,
          p1,
          p2,
          max_attempts);
    }
  });
}

// ---------------------------------------------------------------------------
// GPU (Metal): one thread per walk, RNG inside the kernel
// ---------------------------------------------------------------------------

#ifdef _METAL_
void BiasedRandomWalk::eval_gpu(
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
  auto kernel = d.get_kernel("rejection_sampling", lib);

  auto& enc = mx::metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(kernel);
  enc.set_input_array(inputs[0], 0);
  enc.set_input_array(inputs[1], 1);
  enc.set_input_array(inputs[2], 2);
  enc.set_input_array(inputs[3], 3);
  enc.set_output_array(n_out, 4);
  enc.set_output_array(e_out, 5);
  enc.set_bytes(walk_length_, 6);
  enc.set_bytes(prob_0_, 7);
  enc.set_bytes(prob_1_, 8);
  enc.set_bytes(prob_2_, 9);
  enc.set_bytes(max_attempts_, 10);

  size_t tg = std::min<size_t>(kernel->maxTotalThreadsPerThreadgroup(), 256);
  enc.dispatch_threads(MTL::Size(numel, 1, 1), MTL::Size(tg, 1, 1));
}
#else
void BiasedRandomWalk::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[rejection_sampling] This build has no Metal backend.");
}
#endif

// ---------------------------------------------------------------------------
// Transformations (walks are not differentiable)
// ---------------------------------------------------------------------------

std::vector<mx::array> BiasedRandomWalk::jvp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error(
      "[rejection_sampling] Random walks are not differentiable.");
}

std::vector<mx::array> BiasedRandomWalk::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&) {
  throw std::runtime_error(
      "[rejection_sampling] Random walks are not differentiable.");
}

std::pair<std::vector<mx::array>, std::vector<int>> BiasedRandomWalk::vmap(
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("[rejection_sampling] vmap is not supported.");
}

bool BiasedRandomWalk::is_equivalent(const mx::Primitive& other) const {
  // Randomness comes in through `key`, so equal inputs give equal walks.
  auto* o = dynamic_cast<const BiasedRandomWalk*>(&other);
  return o && o->walk_length_ == walk_length_ && o->p_ == p_ &&
      o->q_ == q_ && o->max_attempts_ == max_attempts_;
}

// ---------------------------------------------------------------------------
// Op
// ---------------------------------------------------------------------------

std::vector<mx::array> rejection_sampling(
    const mx::array& rowptr,
    const mx::array& col,
    const mx::array& start,
    int walk_length,
    double p,
    double q,
    std::optional<mx::array> key,
    int max_attempts,
    mx::StreamOrDevice s) {
  check_int64_1d(rowptr, "rowptr");
  check_int64_1d(col, "col");
  check_int64_1d(start, "start");
  if (walk_length < 1) {
    throw std::invalid_argument(
        "[rejection_sampling] walk_length must be >= 1.");
  }
  if (!(p > 0.0) || !(q > 0.0)) {
    throw std::invalid_argument("[rejection_sampling] p and q must be > 0.");
  }
  if (max_attempts < 1) {
    throw std::invalid_argument(
        "[rejection_sampling] max_attempts must be >= 1.");
  }

  auto stream = mx::to_stream(s);
  mx::array k = key.has_value()
      ? *key
      : mx::random::bits({2}, 4, std::nullopt, stream);
  if (k.dtype() != mx::uint32 || k.size() != 2) {
    throw std::invalid_argument(
        "[rejection_sampling] `key` must be a uint32 array with 2 elements, "
        "e.g. mx.random.key(0).");
  }
  k = mx::reshape(k, {2}, stream);

  // Kernels index raw buffers, so make every input row-contiguous.
  auto c = [&](const mx::array& a) { return mx::contiguous(a, false, stream); };

  const int numel = static_cast<int>(start.size());
  auto primitive = std::make_shared<BiasedRandomWalk>(
      stream, walk_length, p, q, max_attempts);
  return mx::array::make_arrays(
      {{numel, walk_length + 1}, {numel, walk_length}},
      {mx::int64, mx::int64},
      primitive,
      {c(rowptr), c(col), c(start), c(k)});
}

} // namespace mlx_biased_random_walk
