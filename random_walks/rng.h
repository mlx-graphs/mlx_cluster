#pragma once

// Counter-based random numbers shared by the CPU code and the Metal kernels.
//
// Metal has no RNG, and kernels that need an unknown number of random draws
// (e.g. rejection sampling) cannot be fed a precomputed `rand` array. Instead
// every draw is a pure function of (key, stream_id, counter):
//
//     bits = threefry2x32(key, {stream_id, counter})
//
// so each GPU thread can draw as many numbers as it needs without state, and
// the CPU and GPU produce bit-identical results for the same key.
//
// threefry2x32-20 is the generator MLX itself uses for mx.random. It only
// needs 32-bit integer adds, rotates and xors.
//
// This header compiles both as C++ and as Metal Shading Language.

#ifdef __METAL_VERSION__
#define MLXC_FN inline
typedef uint mlxc_u32;
typedef ulong mlxc_u64;
typedef long mlxc_i64;
#else
#include <cstdint>
#define MLXC_FN inline
typedef uint32_t mlxc_u32;
typedef uint64_t mlxc_u64;
typedef int64_t mlxc_i64;
#endif

struct mlxc_u32x2 {
  mlxc_u32 x0;
  mlxc_u32 x1;
};

MLXC_FN mlxc_u32 mlxc_rotl32(mlxc_u32 x, mlxc_u32 r) {
  return (x << r) | (x >> (32u - r));
}

#define MLXC_TF_ROUND(R) \
  x0 += x1;              \
  x1 = mlxc_rotl32(x1, R); \
  x1 ^= x0;

// Threefry-2x32 with 20 rounds (Salmon et al., "Parallel random numbers: as
// easy as 1, 2, 3", SC'11). Matches the Random123 reference implementation.
MLXC_FN mlxc_u32x2
mlxc_threefry2x32(mlxc_u32 k0, mlxc_u32 k1, mlxc_u32 c0, mlxc_u32 c1) {
  const mlxc_u32 k2 = k0 ^ k1 ^ 0x1BD11BDAu;
  mlxc_u32 x0 = c0 + k0;
  mlxc_u32 x1 = c1 + k1;

  MLXC_TF_ROUND(13) MLXC_TF_ROUND(15) MLXC_TF_ROUND(26) MLXC_TF_ROUND(6)
  x0 += k1; x1 += k2 + 1u;
  MLXC_TF_ROUND(17) MLXC_TF_ROUND(29) MLXC_TF_ROUND(16) MLXC_TF_ROUND(24)
  x0 += k2; x1 += k0 + 2u;
  MLXC_TF_ROUND(13) MLXC_TF_ROUND(15) MLXC_TF_ROUND(26) MLXC_TF_ROUND(6)
  x0 += k0; x1 += k1 + 3u;
  MLXC_TF_ROUND(17) MLXC_TF_ROUND(29) MLXC_TF_ROUND(16) MLXC_TF_ROUND(24)
  x0 += k1; x1 += k2 + 4u;
  MLXC_TF_ROUND(13) MLXC_TF_ROUND(15) MLXC_TF_ROUND(26) MLXC_TF_ROUND(6)
  x0 += k2; x1 += k0 + 5u;

  mlxc_u32x2 out;
  out.x0 = x0;
  out.x1 = x1;
  return out;
}

#undef MLXC_TF_ROUND

// A per-walk random stream. `id` is the walk index; its high 32 bits are
// folded into the key so more than 2^32 walks still get distinct streams.
struct mlxc_rng {
  mlxc_u32 k0;
  mlxc_u32 k1;
  mlxc_u32 id;
  mlxc_u32 counter;
};

MLXC_FN mlxc_rng mlxc_rng_init(mlxc_u32 k0, mlxc_u32 k1, mlxc_u64 id) {
  mlxc_rng r;
  r.k0 = k0;
  r.k1 = k1 ^ static_cast<mlxc_u32>(id >> 32);
  r.id = static_cast<mlxc_u32>(id);
  r.counter = 0u;
  return r;
}

#ifdef __METAL_VERSION__
#define MLXC_RNG_REF thread mlxc_rng&
#else
#define MLXC_RNG_REF mlxc_rng&
#endif

MLXC_FN mlxc_u32x2 mlxc_next(MLXC_RNG_REF r) {
  mlxc_u32x2 b = mlxc_threefry2x32(r.k0, r.k1, r.id, r.counter);
  r.counter += 1u;
  return b;
}

// Uniform float in [0, 1) with 24 bits of precision (exact in float32).
MLXC_FN float mlxc_uniform(MLXC_RNG_REF r) {
  mlxc_u32x2 b = mlxc_next(r);
  return static_cast<float>(b.x0 >> 8) * (1.0f / 16777216.0f);
}

// Uniform integer in [0, n), n > 0. Uses 64 random bits, so the modulo bias
// is at most n / 2^64 (negligible for any graph degree).
MLXC_FN mlxc_i64 mlxc_randint(MLXC_RNG_REF r, mlxc_i64 n) {
  mlxc_u32x2 b = mlxc_next(r);
  mlxc_u64 v = (static_cast<mlxc_u64>(b.x0) << 32) | static_cast<mlxc_u64>(b.x1);
  return static_cast<mlxc_i64>(v % static_cast<mlxc_u64>(n));
}
