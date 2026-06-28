#include <cuda_runtime.h>
#include <float.h>

#define D_MAX 128
#define BR    32   // query rows per block (= threads per block)
#define BC    32   // key/value cols per tile

// Shared memory per block: (BR + 2*BC) * d * 4 bytes
// For d=64: (32 + 64) * 64 * 4 = 24 KB — fits in 48 KB default SRAM

__global__ void flash_attn_smem_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N, int d, float scale
) {
    const int tid = threadIdx.x;          // row within query tile [0, BR)
    const int row = blockIdx.x * BR + tid;

    extern __shared__ float smem[];
    float* Qi = smem;                     // BR * d — loaded once, reused each tile
    float* Kj = smem + BR * d;           // BC * d
    float* Vj = smem + (BR + BC) * d;   // BC * d

    // Load this thread's Q row into shared memory
    if (row < N)
        for (int k = 0; k < d; ++k) Qi[tid * d + k] = Q[row * d + k];
    else
        for (int k = 0; k < d; ++k) Qi[tid * d + k] = 0.0f;
    __syncthreads();

    float mi = -FLT_MAX;
    float li = 0.0f;
    float Oi[D_MAX];
    for (int k = 0; k < d; ++k) Oi[k] = 0.0f;

    const int Tc = (N + BC - 1) / BC;
    for (int j_tile = 0; j_tile < Tc; ++j_tile) {
        int col = j_tile * BC + tid;

        // All BR threads cooperatively load one row of Kj and Vj each
        if (col < N)
            for (int k = 0; k < d; ++k) {
                Kj[tid * d + k] = K[col * d + k];
                Vj[tid * d + k] = V[col * d + k];
            }
        else
            for (int k = 0; k < d; ++k) {
                Kj[tid * d + k] = 0.0f;
                Vj[tid * d + k] = 0.0f;
            }
        __syncthreads();

        if (row < N) {
            const int jj_max = min(BC, N - j_tile * BC);
            for (int jj = 0; jj < jj_max; ++jj) {
                // Qi is in shared memory — no repeated global reads
                float Sij = 0.0f;
                for (int k = 0; k < d; ++k)
                    Sij += Qi[tid * d + k] * Kj[jj * d + k];
                Sij *= scale;

                float mi_new = fmaxf(mi, Sij);
                float alpha  = expf(mi - mi_new);
                float beta   = expf(Sij - mi_new);
                float li_new = alpha * li + beta;
                float c1 = (alpha * li) / li_new;
                float c2 = beta / li_new;
                for (int k = 0; k < d; ++k)
                    Oi[k] = c1 * Oi[k] + c2 * Vj[jj * d + k];

                mi = mi_new;
                li = li_new;
            }
        }
        __syncthreads();
    }

    if (row < N)
        for (int k = 0; k < d; ++k)
            O[row * d + k] = Oi[k];
}

void flash_attn_smem_launch(
    const float* Q, const float* K, const float* V, float* O,
    int N, int d, float scale, cudaStream_t stream
) {
    const int Tr = (N + BR - 1) / BR;
    const size_t smem = (size_t)(BR + 2 * BC) * d * sizeof(float);
    flash_attn_smem_kernel<<<Tr, BR, smem, stream>>>(Q, K, V, O, N, d, scale);
}
