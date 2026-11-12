#pragma once

// One node2vec (second-order biased) random walk via rejection sampling.
//
// Shared by BiasedRandomWalk::eval_cpu and the `rejection_sampling` Metal
// kernel, so both backends run literally the same code and, given the same
// key, produce identical walks.
//
// Transition rule (Grover & Leskovec, 2016): having moved t -> v, the next
// node x is a neighbor of v chosen with unnormalised weight
//     1/p  if x == t            (return)
//     1    if x is a neighbor of t
//     1/q  otherwise           (move outward)
// Rejection sampling: draw x uniformly from v's neighbors and accept it with
// probability weight(x) / max(1/p, 1, 1/q). `prob_0/1/2` are those
// normalised acceptance probabilities.
//
// A node without neighbors keeps the walk in place (edge id -1). After
// `max_attempts` rejections the last candidate is accepted, which bounds the
// work per step (an unbounded loop can hang the GPU); with sensible p and q
// the cap is essentially never reached.

#include "random_walks/rng.h"

#ifdef __METAL_VERSION__
#define MLXC_IN const device
#define MLXC_OUT device
#else
#define MLXC_IN const
#define MLXC_OUT
#endif

MLXC_FN bool mlxc_is_neighbor(
    MLXC_IN mlxc_i64* rowptr,
    MLXC_IN mlxc_i64* col,
    mlxc_i64 v,
    mlxc_i64 w) {
  for (mlxc_i64 i = rowptr[v]; i < rowptr[v + 1]; ++i) {
    if (col[i] == w) {
      return true;
    }
  }
  return false;
}

// Writes walk_length + 1 nodes to `n_out` and walk_length edges to `e_out`.
MLXC_FN void mlxc_node2vec_walk(
    MLXC_IN mlxc_i64* rowptr,
    MLXC_IN mlxc_i64* col,
    mlxc_i64 start,
    MLXC_OUT mlxc_i64* n_out,
    MLXC_OUT mlxc_i64* e_out,
    mlxc_u64 walk_id,
    int walk_length,
    mlxc_u32 k0,
    mlxc_u32 k1,
    float prob_0,
    float prob_1,
    float prob_2,
    int max_attempts) {
  mlxc_rng rng = mlxc_rng_init(k0, k1, walk_id);
  const float lo = prob_1 < prob_2 ? prob_1 : prob_2;
  const float hi = prob_1 < prob_2 ? prob_2 : prob_1;

  // Step 0: uniform first-order step t -> v.
  mlxc_i64 t = start;
  mlxc_i64 v = t;
  mlxc_i64 e_cur = -1;
  n_out[0] = t;
  {
    const mlxc_i64 row_start = rowptr[t];
    const mlxc_i64 deg = rowptr[t + 1] - row_start;
    if (deg > 0) {
      e_cur = row_start + mlxc_randint(rng, deg);
      v = col[e_cur];
    }
  }
  n_out[1] = v;
  e_out[0] = e_cur;

  // Steps 1..walk_length-1: second-order steps v -> x given previous t.
  for (int l = 1; l < walk_length; ++l) {
    const mlxc_i64 row_start = rowptr[v];
    const mlxc_i64 deg = rowptr[v + 1] - row_start;
    mlxc_i64 x = v;
    e_cur = -1;
    if (deg == 1) {
      e_cur = row_start;
      x = col[e_cur];
    } else if (deg > 1) {
      for (int a = 0; a < max_attempts; ++a) {
        e_cur = row_start + mlxc_randint(rng, deg);
        x = col[e_cur];
        const float r = mlxc_uniform(rng);
        bool accept;
        if (x == t) {
          accept = r < prob_0;
        } else if (r < lo) {
          accept = true; // accepted whichever of prob_1 / prob_2 applies
        } else if (r >= hi) {
          accept = false; // rejected whichever applies
        } else {
          // Only now pay for the O(deg(x)) neighbor check.
          accept = r < (mlxc_is_neighbor(rowptr, col, x, t) ? prob_1 : prob_2);
        }
        if (accept) {
          break;
        }
      }
    }
    n_out[l + 1] = x;
    e_out[l] = e_cur;
    t = v;
    v = x;
  }
}

#undef MLXC_IN
#undef MLXC_OUT
