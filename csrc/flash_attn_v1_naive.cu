#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

// d must be <= D_MAX (register array bound)
#define D_MAX 128

__global__ void flash_attn_naive_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N, int d, float scale
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    float mi = -FLT_MAX;
    float li = 0.0f;
    float Oi[D_MAX];
    for (int k = 0; k < d; ++k) Oi[k] = 0.0f;

    // FA1 online softmax recurrence — one element of the N×N matrix at a time,
    // reading Q[row] and all K[j], V[j] from global memory
    for (int j = 0; j < N; ++j) {
        float Sij = 0.0f;
        for (int k = 0; k < d; ++k)
            Sij += Q[row * d + k] * K[j * d + k];
        Sij *= scale;

        float mi_new = fmaxf(mi, Sij);
        float alpha  = expf(mi - mi_new);   // rescale old accumulator
        float beta   = expf(Sij - mi_new);  // weight for this element
        float li_new = alpha * li + beta;

        float c1 = (alpha * li) / li_new;
        float c2 = beta / li_new;
        for (int k = 0; k < d; ++k)
            Oi[k] = c1 * Oi[k] + c2 * V[j * d + k];

        mi = mi_new;
        li = li_new;
    }

    for (int k = 0; k < d; ++k)
        O[row * d + k] = Oi[k];
}

void flash_attn_naive_launch(
    const float* Q, const float* K, const float* V, float* O,
    int N, int d, float scale, cudaStream_t stream
) {
    const int block = 128;
    const int grid  = (N + block - 1) / block;
    flash_attn_naive_kernel<<<grid, block, 0, stream>>>(Q, K, V, O, N, d, scale);
}
