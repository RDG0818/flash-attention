#include <cuda_runtime.h>
#include <float.h>

#define WARP_SIZE 32
#define BR_WARP   4    // warps (rows) per thread block; blockDim.x = BR_WARP * WARP_SIZE
#define BC_WARP   32   // K/V cols per tile (shared memory tile width)

// Requires d % WARP_SIZE == 0 and d <= 128.
// Each warp (32 lanes) owns one query row.
// Qi stays in per-lane registers — not in shared memory.
// Dot products parallelized: each lane handles d/32 elements, then __shfl_down_sync reduces.

__global__ void flash_attn_warp_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N, int d, float scale
) {
    const int lane    = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int row     = blockIdx.x * BR_WARP + warp_id;
    const int elems   = d / WARP_SIZE;           // elements per lane (d % 32 == 0 required)
    const unsigned mask = 0xFFFFFFFFu;

    // Kj and Vj in shared memory; Qi distributed across warp lanes in registers
    extern __shared__ float smem[];
    float* Kj = smem;
    float* Vj = smem + BC_WARP * d;

    // Load this lane's slice of Q[row] into registers — stays resident the whole kernel
    float Qi_reg[4];  // max elems = 128 / 32 = 4
    if (row < N)
        for (int k = 0; k < elems; ++k)
            Qi_reg[k] = Q[row * d + lane * elems + k];

    float mi = -FLT_MAX;
    float li = 0.0f;
    float Oi[4];
    for (int k = 0; k < elems; ++k) Oi[k] = 0.0f;

    const int block_threads = BR_WARP * WARP_SIZE;
    const int Tc = (N + BC_WARP - 1) / BC_WARP;

    for (int j_tile = 0; j_tile < Tc; ++j_tile) {
        // All threads cooperatively load Kj and Vj tile
        for (int idx = threadIdx.x; idx < BC_WARP * d; idx += block_threads) {
            int r = idx / d, c = idx % d;
            int gcol = j_tile * BC_WARP + r;
            Kj[idx] = (gcol < N) ? K[gcol * d + c] : 0.0f;
            Vj[idx] = (gcol < N) ? V[gcol * d + c] : 0.0f;
        }
        __syncthreads();

        if (row < N) {
            const int jj_max = min(BC_WARP, N - j_tile * BC_WARP);
            for (int jj = 0; jj < jj_max; ++jj) {
                // Each lane computes its partial dot product contribution
                float partial = 0.0f;
                for (int k = 0; k < elems; ++k)
                    partial += Qi_reg[k] * Kj[jj * d + lane * elems + k];

                // Warp reduce: sum partials across all 32 lanes
                #pragma unroll
                for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                    partial += __shfl_down_sync(mask, partial, off);
                // Lane 0 holds full dot product; broadcast to all lanes
                float Sij = __shfl_sync(mask, partial, 0) * scale;

                // Online softmax update — identical on all lanes (same Sij)
                float mi_new = fmaxf(mi, Sij);
                float alpha  = expf(mi - mi_new);
                float beta   = expf(Sij - mi_new);
                float li_new = alpha * li + beta;
                float c1 = (alpha * li) / li_new;
                float c2 = beta / li_new;

                // Each lane updates its slice of Oi
                for (int k = 0; k < elems; ++k)
                    Oi[k] = c1 * Oi[k] + c2 * Vj[jj * d + lane * elems + k];

                mi = mi_new;
                li = li_new;
            }
        }
        __syncthreads();
    }

    // Each lane writes its output slice
    if (row < N)
        for (int k = 0; k < elems; ++k)
            O[row * d + lane * elems + k] = Oi[k];
}

void flash_attn_warp_launch(
    const float* Q, const float* K, const float* V, float* O,
    int N, int d, float scale, cudaStream_t stream
) {
    const int Tr = (N + BR_WARP - 1) / BR_WARP;
    const dim3 block(BR_WARP * WARP_SIZE);
    const size_t smem = 2 * (size_t)BC_WARP * d * sizeof(float);
    flash_attn_warp_kernel<<<Tr, block, smem, stream>>>(Q, K, V, O, N, d, scale);
}
