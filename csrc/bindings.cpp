#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

void flash_attn_naive_launch(const float*, const float*, const float*, float*, int, int, float, cudaStream_t);
void flash_attn_smem_launch (const float*, const float*, const float*, float*, int, int, float, cudaStream_t);
void flash_attn_warp_launch (const float*, const float*, const float*, float*, int, int, float, cudaStream_t);

static torch::Tensor run_kernel(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    void (*launcher)(const float*, const float*, const float*, float*, int, int, float, cudaStream_t)
) {
    TORCH_CHECK(Q.is_cuda(),                  "Q must be a CUDA tensor");
    TORCH_CHECK(Q.dim() == 4,                 "Q must be (B, H, N, d)");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Only float32 supported");
    Q = Q.contiguous(); K = K.contiguous(); V = V.contiguous();

    auto O = torch::zeros_like(Q);
    const int B = Q.size(0), H = Q.size(1), N = Q.size(2), d = Q.size(3);
    const float scale = 1.0f / std::sqrt((float)d);
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    const float* q = Q.data_ptr<float>();
    const float* k = K.data_ptr<float>();
    const float* v = V.data_ptr<float>();
    float*       o = O.data_ptr<float>();
    const int stride = N * d;

    for (int b = 0; b < B; ++b)
        for (int h = 0; h < H; ++h) {
            int off = (b * H + h) * stride;
            launcher(q+off, k+off, v+off, o+off, N, d, scale, stream);
        }
    return O;
}

torch::Tensor forward_naive(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    return run_kernel(Q, K, V, flash_attn_naive_launch);
}
torch::Tensor forward_smem(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    return run_kernel(Q, K, V, flash_attn_smem_launch);
}
torch::Tensor forward_warp(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    return run_kernel(Q, K, V, flash_attn_warp_launch);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_naive", &forward_naive, "Flash Attention v1 naive (CUDA)");
    m.def("forward_smem",  &forward_smem,  "Flash Attention v1 shared memory (CUDA)");
    m.def("forward_warp",  &forward_warp,  "Flash Attention v1 warp optimized (CUDA)");
}
