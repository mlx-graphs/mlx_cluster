#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"
#include "random_walks/node2vec_walk.h"
using namespace metal;

kernel void random_walk(
    const device int64_t* rowptr [[buffer(0)]],
    const device int64_t* col [[buffer(1)]],
    const device int64_t* start [[buffer(2)]],
    const device float* rand [[buffer(3)]],
    device int64_t* n_out [[buffer(4)]],
    device int64_t* e_out [[buffer(5)]],
    constant int& walk_length [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    int64_t n_cur = start[tid];
    n_out[tid * (walk_length + 1)] = n_cur;
    
    for (int l = 0; l < walk_length; l++) {
        int64_t row_start = rowptr[n_cur];
        int64_t row_end = rowptr[n_cur + 1];
        int64_t e_cur;
        
        if (row_end - row_start == 0) {
            e_cur = -1;
        } else {
            int64_t deg = row_end - row_start;
            float r = rand[tid * walk_length + l];
            int64_t idx = static_cast<int64_t>(r * deg);
            idx = idx < deg ? idx : deg - 1; // guard against rand == 1.0
            e_cur = row_start + idx;
            n_cur = col[e_cur];
        }
        
        n_out[tid * (walk_length + 1) + (l + 1)] = n_cur;
        e_out[tid * walk_length + l] = e_cur;
    }
}
// Meta-path guided random walk (metapath2vec). One thread per walk.
// Relations are concatenated CSRs; step_rel[l] picks the relation for step l.
// A walk that hits a node without edges of the required relation ends and
// writes -1 into every remaining node/edge slot.
kernel void metapath_random_walk(
    const device int64_t* rowptr [[buffer(0)]],
    const device int64_t* col [[buffer(1)]],
    const device int64_t* rowptr_off [[buffer(2)]],
    const device int64_t* col_off [[buffer(3)]],
    const device int64_t* step_rel [[buffer(4)]],
    const device int64_t* start [[buffer(5)]],
    const device float* rand [[buffer(6)]],
    device int64_t* n_out [[buffer(7)]],
    device int64_t* e_out [[buffer(8)]],
    constant int& walk_length [[buffer(9)]],
    uint tid [[thread_position_in_grid]]
) {
    const int L = walk_length;
    int64_t cur = start[tid];
    n_out[tid * (L + 1)] = cur;

    for (int l = 0; l < L; l++) {
        int64_t e_cur = -1;
        if (cur >= 0) {
            const int64_t r = step_rel[l];
            const device int64_t* rp = rowptr + rowptr_off[r];
            const int64_t row_start = rp[cur];
            const int64_t deg = rp[cur + 1] - row_start;
            if (deg == 0) {
                cur = -1;
            } else {
                int64_t idx = static_cast<int64_t>(rand[tid * L + l] * deg);
                idx = idx < deg ? idx : deg - 1;
                e_cur = row_start + idx;
                cur = col[col_off[r] + e_cur];
            }
        }
        n_out[tid * (L + 1) + (l + 1)] = cur;
        e_out[tid * L + l] = e_cur;
    }
}

// node2vec biased random walk (rejection sampling). One thread per walk.
// Random numbers come from a counter-based generator (random_walks/rng.h)
// keyed by `key`, so a step can draw as many candidates as it needs. The
// walk logic itself lives in random_walks/node2vec_walk.h and is shared with
// the CPU implementation, so both produce identical walks for the same key.
kernel void rejection_sampling(
    const device int64_t* rowptr [[buffer(0)]],
    const device int64_t* col [[buffer(1)]],
    const device int64_t* start [[buffer(2)]],
    const device uint* key [[buffer(3)]],
    device int64_t* n_out [[buffer(4)]],
    device int64_t* e_out [[buffer(5)]],
    constant int& walk_length [[buffer(6)]],
    constant float& prob_0 [[buffer(7)]],
    constant float& prob_1 [[buffer(8)]],
    constant float& prob_2 [[buffer(9)]],
    constant int& max_attempts [[buffer(10)]],
    uint tid [[thread_position_in_grid]]
) {
    const int L = walk_length;
    mlxc_node2vec_walk(
        rowptr,
        col,
        start[tid],
        n_out + static_cast<ulong>(tid) * (L + 1),
        e_out + static_cast<ulong>(tid) * L,
        static_cast<ulong>(tid),
        L,
        key[0],
        key[1],
        prob_0,
        prob_1,
        prob_2,
        max_attempts);
}
