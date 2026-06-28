# Flash Attention CUDA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Flash Attention v1 forward pass in three progressive CUDA kernels + Triton, expose via PyTorch extension, verify correctness, and benchmark against `torch.sdpa`.

**Architecture:** PyTorch `CUDAExtension` wraps three `.cu` kernels (naive → shared-mem tiled → warp-optimized) under `flash_attn.forward_{naive,smem,warp}`. A separate Triton kernel provides a Python-native comparison point. `verify.py` gates `benchmark.py`; all variants are tested against `reference_attention` from the existing `flash_attention.py`.

**Tech Stack:** CUDA C++ (sm_86), PyTorch C++ extension (pybind11), Triton, matplotlib, numpy.

## Global Constraints

- FP32 only
- Head dim `d` ≤ 128; warp kernel additionally requires `d % 32 == 0`
- All kernels handle one `(N, d)` head at a time; B×H loop is in the Python/C++ caller
- Build target: `-arch=sm_86 -O3 --use_fast_math`
- All files live under `flash_attention/` inside the existing `ml` repo
- Existing files (`flash_attention.py`, `gpt.py`, etc.) are untouched
- `verify.py` must exit 0 before `benchmark.py` may run

---

## File Map

| File | Role |
|------|------|
| `flash_attention/setup.py` | `CUDAExtension` build config |
| `flash_attention/csrc/bindings.cpp` | pybind11 entry: `forward_naive`, `forward_smem`, `forward_warp` |
| `flash_attention/csrc/flash_attn_v1_naive.cu` | Stage 1: one thread per row, global memory |
| `flash_attention/csrc/flash_attn_v1_smem.cu` | Stage 2: cooperative tile load into shared memory |
| `flash_attention/csrc/flash_attn_v1_warp.cu` | Stage 3: warp-parallelized dot product + `__shfl_down_sync` |
| `flash_attention/triton/flash_attn_triton.py` | Triton FA1 kernel (no build step) |
| `flash_attention/verify.py` | Correctness tests for all 5 variants |
| `flash_attention/benchmark.py` | Timing sweep + 3 PNGs |
| `flash_attention/figures/` | Committed benchmark plots |
| `flash_attention/README.md` | Portfolio README |

---

## Task 1: Scaffold + Build System

**Files:**
- Create: `flash_attention/setup.py`
- Create: `flash_attention/csrc/bindings.cpp`
- Create: `flash_attention/csrc/flash_attn_v1_naive.cu` (stub)
- Create: `flash_attention/csrc/flash_attn_v1_smem.cu` (stub)
- Create: `flash_attention/csrc/flash_attn_v1_warp.cu` (stub)
- Create: `flash_attention/triton/__init__.py` (empty)

**Interfaces:**
- Produces: `flash_attn` Python module with `forward_naive(Q,K,V)`, `forward_smem(Q,K,V)`, `forward_warp(Q,K,V)` — each accepts `(B,H,N,d)` float32 CUDA tensors and returns zeros until kernels are implemented

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p flash_attention/csrc flash_attention/triton flash_attention/figures
touch flash_attention/triton/__init__.py
```

- [ ] **Step 2: Write `flash_attention/setup.py`**

```python
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='flash_attn',
    ext_modules=[
        CUDAExtension(
            name='flash_attn',
            sources=[
                'csrc/flash_attn_v1_naive.cu',
                'csrc/flash_attn_v1_smem.cu',
                'csrc/flash_attn_v1_warp.cu',
                'csrc/bindings.cpp',
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': ['-arch=sm_86', '-O3', '--use_fast_math', '-std=c++17'],
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
```

- [ ] **Step 3: Write stub `flash_attention/csrc/flash_attn_v1_naive.cu`**

```cuda
#include <cuda_runtime.h>

// Stub — returns zeros. Real implementation in Task 3.
void flash_attn_naive_launch(
    const float*, const float*, const float*, float*,
    int, int, float, cudaStream_t
) {}
```

- [ ] **Step 4: Write stub `flash_attention/csrc/flash_attn_v1_smem.cu`**

```cuda
#include <cuda_runtime.h>

// Stub — returns zeros. Real implementation in Task 4.
void flash_attn_smem_launch(
    const float*, const float*, const float*, float*,
    int, int, float, cudaStream_t
) {}
```

- [ ] **Step 5: Write stub `flash_attention/csrc/flash_attn_v1_warp.cu`**

```cuda
#include <cuda_runtime.h>

// Stub — returns zeros. Real implementation in Task 5.
void flash_attn_warp_launch(
    const float*, const float*, const float*, float*,
    int, int, float, cudaStream_t
) {}
```

- [ ] **Step 6: Write `flash_attention/csrc/bindings.cpp`**

```cpp
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
    TORCH_CHECK(Q.is_cuda(),                    "Q must be a CUDA tensor");
    TORCH_CHECK(Q.dim() == 4,                   "Q must be (B, H, N, d)");
    TORCH_CHECK(Q.dtype() == torch::kFloat32,   "Only float32 supported");
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
```

- [ ] **Step 7: Build the extension**

Run from `flash_attention/`:
```bash
cd flash_attention && pip install -e . 2>&1 | tail -5
```
Expected: `Successfully installed flash-attn`

- [ ] **Step 8: Verify import**

```bash
python -c "import flash_attn; print(dir(flash_attn))"
```
Expected output contains: `['forward_naive', 'forward_smem', 'forward_warp', ...]`

- [ ] **Step 9: Commit**

```bash
git init flash_attention  # only if moving to own repo later; else skip
git -C .. add flash_attention/csrc flash_attention/triton flash_attention/setup.py
git -C .. commit -m "feat(flash-attn): scaffold build system with stub CUDA kernels"
```

---

## Task 2: Correctness Harness (`verify.py`)

**Files:**
- Create: `flash_attention/verify.py`

**Interfaces:**
- Consumes: `flash_attn.forward_naive/smem/warp(Q,K,V) → Tensor`, `flash_attn_triton(Q,K,V) → Tensor` (Triton added Task 6), `reference_attention` from `../flash_attention.py`
- Produces: exit code 0 (all pass) or 1 (any fail); used as gate in `benchmark.py`

- [ ] **Step 1: Write `flash_attention/verify.py`**

```python
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from flash_attention import reference_attention, flash_attention as py_flash_attn, causal_mask

DEVICE = 'cuda'
TOL = 1e-4

CONFIGS = [
    (1, 1,  64, 32, None),
    (2, 4, 128, 64, None),
    (1, 2,  63, 32, None),      # non-power-of-two
    (1, 1,  64, 32, 'causal'),  # causal mask
]


def check(name, fn, Q, K, V, ref, mask=None):
    try:
        if mask is not None:
            out = fn(Q, K, V, mask=mask)
        else:
            out = fn(Q, K, V)
        err = (out - ref).abs().max().item()
        ok = err < TOL
    except Exception as e:
        print(f"  [ERROR] {name:30s}  {e}")
        return False
    status = 'PASS' if ok else 'FAIL'
    print(f"  [{status}] {name:30s}  max_err={err:.2e}")
    return ok


def run_cuda_check(name, fn, Q, K, V, ref):
    try:
        out = fn(Q, K, V)
        err = (out - ref).abs().max().item()
        ok = err < TOL
    except Exception as e:
        print(f"  [ERROR] {name:30s}  {e}")
        return False
    status = 'PASS' if ok else 'FAIL'
    print(f"  [{status}] {name:30s}  max_err={err:.2e}")
    return ok


def main():
    import flash_attn

    # Triton may not be installed — skip gracefully
    try:
        from triton.flash_attn_triton import flash_attn_triton
        has_triton = True
    except ImportError:
        has_triton = False
        print("  [SKIP] Triton not installed — skipping Triton variant")

    all_pass = True
    for B, H, N, d, mask_type in CONFIGS:
        print(f"\nConfig: B={B} H={H} N={N} d={d} mask={mask_type or 'none'}")
        torch.manual_seed(0)
        Q = torch.randn(B, H, N, d, device=DEVICE)
        K = torch.randn(B, H, N, d, device=DEVICE)
        V = torch.randn(B, H, N, d, device=DEVICE)

        mask = None
        if mask_type == 'causal':
            mask = causal_mask(N, device=DEVICE).unsqueeze(0).unsqueeze(0)

        ref = reference_attention(Q, K, V, mask=mask)

        # Python FA1 (no mask arg for CUDA variants — they skip mask for now)
        all_pass &= check("Python FA1", py_flash_attn, Q, K, V, ref, mask=mask)

        # CUDA variants do not support masking in this phase (FYI note)
        if mask is None:
            all_pass &= run_cuda_check("CUDA naive", flash_attn.forward_naive, Q, K, V, ref)
            all_pass &= run_cuda_check("CUDA smem",  flash_attn.forward_smem,  Q, K, V, ref)
            all_pass &= run_cuda_check("CUDA warp",  flash_attn.forward_warp,  Q, K, V, ref)
            if has_triton:
                all_pass &= run_cuda_check("Triton", flash_attn_triton, Q, K, V, ref)

    print()
    if all_pass:
        print("All tests passed.")
    else:
        print("FAILURES detected.")
        sys.exit(1)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run verify.py — expect Python FA1 PASS, CUDA variants return zeros (FAIL)**

```bash
cd flash_attention && python verify.py
```
Expected: Python FA1 `PASS`, all CUDA variants `FAIL` (max_err ≈ ref magnitude — stubs output zeros).

- [ ] **Step 3: Commit**

```bash
git -C .. add flash_attention/verify.py
git -C .. commit -m "test(flash-attn): add correctness harness — Python FA1 passes, CUDA stubs fail"
```

---

## Task 3: Naive CUDA Kernel

**Files:**
- Modify: `flash_attention/csrc/flash_attn_v1_naive.cu`

**Interfaces:**
- Consumes: `(Q, K, V: float* device, N, d: int, scale: float, stream: cudaStream_t)`
- Produces: `O[row * d + k]` for all rows; `void flash_attn_naive_launch(...)` callable from `bindings.cpp`

Algorithm: one CUDA thread per output row. Thread iterates over all N key/value vectors in global memory, running the FA1 online softmax recurrence. `d ≤ 128` (register array bound).

- [ ] **Step 1: Replace stub with full implementation**

```cuda
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

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

    for (int j = 0; j < N; ++j) {
        float Sij = 0.0f;
        for (int k = 0; k < d; ++k)
            Sij += Q[row * d + k] * K[j * d + k];
        Sij *= scale;

        float mi_new = fmaxf(mi, Sij);
        float alpha  = expf(mi - mi_new);
        float beta   = expf(Sij - mi_new);
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
```

- [ ] **Step 2: Rebuild**

```bash
cd flash_attention && pip install -e . 2>&1 | tail -3
```

- [ ] **Step 3: Run verify.py — CUDA naive should now PASS**

```bash
python verify.py
```
Expected: `[PASS] CUDA naive` for all non-masked configs. CUDA smem/warp still FAIL.

- [ ] **Step 4: Commit**

```bash
git -C .. add flash_attention/csrc/flash_attn_v1_naive.cu
git -C .. commit -m "feat(flash-attn): implement naive CUDA kernel — one thread per row, global memory"
```

---

## Task 4: Shared Memory CUDA Kernel

**Files:**
- Modify: `flash_attention/csrc/flash_attn_v1_smem.cu`

**Interfaces:**
- Consumes: same launcher signature as naive
- Produces: `void flash_attn_smem_launch(...)` callable from `bindings.cpp`

Algorithm: one thread block (Br=32 threads) per query tile. Threads cooperatively load `Qi`, `Kj`, `Vj` tiles into `__shared__` before inner loop. Each thread owns one query row and its own `mi`, `li`, `Oi` register accumulators. Tile sizes: `Br = Bc = 32`. Shared memory per block: `(Br + 2*Bc) * d * 4` bytes ≈ 24 KB for d=64, fits in 48 KB.

- [ ] **Step 1: Replace stub with full implementation**

```cuda
#include <cuda_runtime.h>
#include <float.h>

#define D_MAX 128
#define BR    32
#define BC    32

__global__ void flash_attn_smem_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N, int d, float scale
) {
    const int tid  = threadIdx.x;
    const int row  = blockIdx.x * BR + tid;

    extern __shared__ float smem[];
    float* Qi = smem;              // BR * d
    float* Kj = smem + BR * d;    // BC * d
    float* Vj = smem + (BR + BC) * d; // BC * d

    // Load this thread's row of Q into shared memory once
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

        // Cooperatively load Kj and Vj (one row per thread)
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
    const size_t smem = (BR + 2 * BC) * d * sizeof(float);
    flash_attn_smem_kernel<<<Tr, BR, smem, stream>>>(Q, K, V, O, N, d, scale);
}
```

- [ ] **Step 2: Rebuild**

```bash
cd flash_attention && pip install -e . 2>&1 | tail -3
```

- [ ] **Step 3: Run verify.py — CUDA smem should now PASS**

```bash
python verify.py
```
Expected: `[PASS] CUDA naive` and `[PASS] CUDA smem` for all non-masked configs.

- [ ] **Step 4: Commit**

```bash
git -C .. add flash_attention/csrc/flash_attn_v1_smem.cu
git -C .. commit -m "feat(flash-attn): implement shared-memory tiled CUDA kernel — cooperative Q/K/V tile load"
```

---

## Task 5: Warp-Optimized CUDA Kernel

**Files:**
- Modify: `flash_attention/csrc/flash_attn_v1_warp.cu`

**Interfaces:**
- Consumes: same launcher signature as naive/smem
- Produces: `void flash_attn_warp_launch(...)` callable from `bindings.cpp`

Algorithm: one warp (32 lanes) per query row. Each lane owns `d/32` elements of `Qi` and `Oi` in registers (Qi loaded once, stays in registers — not in shared memory). Threads cooperatively load `Kj`/`Vj` tiles into shared memory. Dot products parallelized: each lane computes a partial sum over its `d/32` elements, then `__shfl_down_sync` reduces to full dot product. `d % 32 == 0` required.

- [ ] **Step 1: Replace stub with full implementation**

```cuda
#include <cuda_runtime.h>
#include <float.h>

#define WARP_SIZE 32
#define BR_WARP   4    // warps (rows) per thread block
#define BC_WARP   32   // K/V cols per tile

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
    const int elems   = d / WARP_SIZE;  // elements per lane (d must be % 32 == 0)
    const unsigned mask = 0xFFFFFFFFu;

    // Kj and Vj in shared memory; Qi lives in per-lane registers
    extern __shared__ float smem[];
    float* Kj = smem;
    float* Vj = smem + BC_WARP * d;

    // Load Qi slice into registers (once, reused each tile)
    float Qi_reg[4];  // elems <= 128/32 = 4
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
        // All threads in block cooperatively load Kj, Vj
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
                // Partial dot product: lane computes its slice of Q[row] · K[jj]
                float partial = 0.0f;
                for (int k = 0; k < elems; ++k)
                    partial += Qi_reg[k] * Kj[jj * d + lane * elems + k];

                // Warp reduce sum → all lanes get full dot product
                #pragma unroll
                for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                    partial += __shfl_down_sync(mask, partial, off);
                float Sij = __shfl_sync(mask, partial, 0) * scale;

                // Online softmax update (same values on all lanes)
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

    // Write Oi: each lane writes its slice
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
    const size_t smem = 2 * BC_WARP * d * sizeof(float);
    flash_attn_warp_kernel<<<Tr, block, smem, stream>>>(Q, K, V, O, N, d, scale);
}
```

- [ ] **Step 2: Rebuild**

```bash
cd flash_attention && pip install -e . 2>&1 | tail -3
```

- [ ] **Step 3: Run verify.py — all CUDA variants should PASS**

```bash
python verify.py
```
Expected: `[PASS]` for Python FA1, CUDA naive, CUDA smem, CUDA warp on all non-masked configs.

- [ ] **Step 4: Commit**

```bash
git -C .. add flash_attention/csrc/flash_attn_v1_warp.cu
git -C .. commit -m "feat(flash-attn): implement warp-optimized kernel — Qi in registers, __shfl_down_sync dot product"
```

---

## Task 6: Triton Kernel

**Files:**
- Create: `flash_attention/triton/flash_attn_triton.py`

**Interfaces:**
- Produces: `flash_attn_triton(Q: Tensor, K: Tensor, V: Tensor) -> Tensor` accepting `(B,H,N,d)` float32 CUDA tensors
- Consumed by: `verify.py`, `benchmark.py`

Algorithm: one Triton program per query tile (BLOCK_M=32 rows). Each program loads Q tile once, then loops over K/V tiles (BLOCK_N=32). Uses `tl.dot` for `Q @ K^T` and `P @ V`. Online softmax uses unnormalized accumulator `O_acc`; divides by `l_i` once after the loop. `HEAD_DIM` is a `tl.constexpr` set at dispatch time.

- [ ] **Step 1: Write `flash_attention/triton/flash_attn_triton.py`**

```python
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    N, scale,
    stride_qn, stride_qd,
    stride_kn, stride_kd,
    stride_vn, stride_vd,
    stride_on, stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    i_tile = tl.program_id(0)

    offs_m = i_tile * BLOCK_M + tl.arange(0, BLOCK_M)  # (BLOCK_M,)
    offs_d = tl.arange(0, HEAD_DIM)                     # (HEAD_DIM,)
    offs_n = tl.arange(0, BLOCK_N)                      # (BLOCK_N,)

    # Load Q tile: (BLOCK_M, HEAD_DIM)
    Q_ptrs = Q_ptr + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    Q = tl.load(Q_ptrs, mask=offs_m[:, None] < N, other=0.0)

    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    O_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    Tc = tl.cdiv(N, BLOCK_N)
    for j in range(Tc):
        offs_n_j = j * BLOCK_N + offs_n

        # Load K tile as (HEAD_DIM, BLOCK_N) for tl.dot(Q, K)
        K_ptrs = K_ptr + offs_n_j[None, :] * stride_kn + offs_d[:, None] * stride_kd
        K = tl.load(K_ptrs, mask=offs_n_j[None, :] < N, other=0.0)

        # Load V tile as (BLOCK_N, HEAD_DIM)
        V_ptrs = V_ptr + offs_n_j[:, None] * stride_vn + offs_d[None, :] * stride_vd
        V = tl.load(V_ptrs, mask=offs_n_j[:, None] < N, other=0.0)

        # S = Q @ K * scale: (BLOCK_M, BLOCK_N)
        S = tl.dot(Q, K) * scale

        # Mask padding columns to -inf so they don't affect softmax
        S = tl.where(offs_n_j[None, :] < N, S, float('-inf'))

        # Online softmax (unnormalized accumulator)
        m_j   = tl.max(S, axis=1)                    # (BLOCK_M,)
        m_new = tl.maximum(m_i, m_j)
        alpha = tl.exp(m_i - m_new)                  # (BLOCK_M,)
        P     = tl.exp(S - m_new[:, None])            # (BLOCK_M, BLOCK_N)
        l_j   = tl.sum(P, axis=1)                    # (BLOCK_M,)
        l_new = alpha * l_i + l_j

        O_acc = alpha[:, None] * O_acc + tl.dot(P, V)
        m_i = m_new
        l_i = l_new

    # Normalize
    O_out = O_acc / l_i[:, None]

    O_ptrs = O_ptr + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(O_ptrs, O_out, mask=offs_m[:, None] < N)


def flash_attn_triton(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Q, K, V: (B, H, N, d) float32 CUDA tensors.
    Returns O: (B, H, N, d).
    d must be a power of two (Triton tl.dot constraint).
    """
    B, H, N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    O = torch.zeros_like(Q)

    BLOCK_M = 32
    BLOCK_N = 32

    for b in range(B):
        for h in range(H):
            q = Q[b, h].contiguous()
            k = K[b, h].contiguous()
            v = V[b, h].contiguous()
            o = O[b, h]

            grid = (triton.cdiv(N, BLOCK_M),)
            _flash_attn_fwd_kernel[grid](
                q, k, v, o,
                N, scale,
                q.stride(0), q.stride(1),
                k.stride(0), k.stride(1),
                v.stride(0), v.stride(1),
                o.stride(0), o.stride(1),
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                HEAD_DIM=d,
            )
    return O
```

- [ ] **Step 2: Run verify.py — Triton should now PASS**

```bash
cd flash_attention && python verify.py
```
Expected: `[PASS]` for all variants including Triton. All tests passed.

- [ ] **Step 3: Commit**

```bash
git -C .. add flash_attention/triton/flash_attn_triton.py flash_attention/triton/__init__.py
git -C .. commit -m "feat(flash-attn): implement Triton FA1 kernel — tl.dot + online softmax, unnormalized accumulator"
```

---

## Task 7: Benchmark Script + Figures

**Files:**
- Create: `flash_attention/benchmark.py`

**Interfaces:**
- Consumes: `flash_attn.forward_naive/smem/warp`, `flash_attn_triton`, `torch.nn.functional.scaled_dot_product_attention`
- Produces: `flash_attention/figures/latency_vs_seqlen.png`, `speedup_vs_naive.png`, `memory_vs_seqlen.png`

- [ ] **Step 1: Write `flash_attention/benchmark.py`**

```python
import sys
import os
import subprocess

# Gate: correctness must pass before benchmarking
result = subprocess.run(
    [sys.executable, os.path.join(os.path.dirname(__file__), 'verify.py')],
    capture_output=True
)
if result.returncode != 0:
    print("verify.py failed — fix correctness before running benchmarks.")
    print(result.stdout.decode())
    sys.exit(1)

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import flash_attn
from triton.flash_attn_triton import flash_attn_triton

DEVICE   = 'cuda'
B, H, D  = 1, 8, 64
SEQ_LENS = [128, 256, 512, 1024, 2048, 4096]
WARMUP   = 10
REPS     = 50

VARIANTS = {
    'naive CUDA': flash_attn.forward_naive,
    'smem CUDA':  flash_attn.forward_smem,
    'warp CUDA':  flash_attn.forward_warp,
    'Triton':     flash_attn_triton,
    'torch.sdpa': lambda Q, K, V: F.scaled_dot_product_attention(Q, K, V),
}


def time_fn(fn, Q, K, V):
    for _ in range(WARMUP):
        fn(Q, K, V)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(REPS):
        start.record()
        fn(Q, K, V)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return float(np.median(times))


def measure_memory(fn, Q, K, V):
    torch.cuda.reset_peak_memory_stats(DEVICE)
    fn(Q, K, V)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(DEVICE) / 1024 ** 2  # MB


latency = {name: [] for name in VARIANTS}
memory  = {name: [] for name in VARIANTS}

for N in SEQ_LENS:
    print(f"N={N} ...", flush=True)
    Q = torch.randn(B, H, N, D, device=DEVICE)
    K = torch.randn(B, H, N, D, device=DEVICE)
    V = torch.randn(B, H, N, D, device=DEVICE)
    for name, fn in VARIANTS.items():
        t = time_fn(fn, Q, K, V)
        m = measure_memory(fn, Q, K, V)
        latency[name].append(t)
        memory[name].append(m)
        print(f"  {name:15s}  {t:8.3f} ms   {m:6.1f} MB")

FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# --- Plot 1: Latency vs seq len ---
plt.figure(figsize=(10, 6))
for name, times in latency.items():
    plt.plot(SEQ_LENS, times, marker='o', label=name)
plt.xscale('log', base=2)
plt.xlabel('Sequence Length')
plt.ylabel('Latency (ms, median of 50)')
plt.title('Flash Attention FA1 — Latency vs Sequence Length (B=1, H=8, d=64)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'latency_vs_seqlen.png'), dpi=150)
plt.close()

# --- Plot 2: Speedup vs naive CUDA ---
naive_times = latency['naive CUDA']
speedups = {
    name: [naive_times[i] / t for i, t in enumerate(times)]
    for name, times in latency.items()
    if name != 'naive CUDA'
}
x = np.arange(len(SEQ_LENS))
width = 0.18
plt.figure(figsize=(13, 6))
for idx, (name, spd) in enumerate(speedups.items()):
    plt.bar(x + idx * width, spd, width, label=name)
plt.xlabel('Sequence Length')
plt.ylabel('Speedup over Naive CUDA (higher = better)')
plt.title('Flash Attention FA1 — Speedup vs Naive CUDA')
plt.xticks(x + width * len(speedups) / 2, [str(n) for n in SEQ_LENS])
plt.axhline(1.0, color='gray', linestyle='--', alpha=0.6, label='baseline')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'speedup_vs_naive.png'), dpi=150)
plt.close()

# --- Plot 3: Peak memory vs seq len ---
plt.figure(figsize=(10, 6))
for name, mems in memory.items():
    plt.plot(SEQ_LENS, mems, marker='o', label=name)
plt.xlabel('Sequence Length')
plt.ylabel('Peak Memory (MB)')
plt.title('Flash Attention FA1 — Peak Memory vs Sequence Length')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'memory_vs_seqlen.png'), dpi=150)
plt.close()

print(f"\nFigures saved to {FIG_DIR}/")
```

- [ ] **Step 2: Transfer code to remote machine and run**

On the remote (RTX 3060 Ti):
```bash
cd flash_attention
pip install -e .
pip install triton matplotlib numpy
python benchmark.py
```
Expected: timing table printed, then `Figures saved to .../figures/`

- [ ] **Step 3: Copy figures back and commit**

```bash
scp remote:path/to/flash_attention/figures/*.png flash_attention/figures/
git -C .. add flash_attention/benchmark.py flash_attention/figures/
git -C .. commit -m "feat(flash-attn): add benchmark script and commit baseline figures"
```

---

## Task 8: README

**Files:**
- Create: `flash_attention/README.md`

- [ ] **Step 1: Write `flash_attention/README.md`**

```markdown
# Flash Attention — CUDA from Scratch

A ground-up implementation of [Flash Attention v1](https://arxiv.org/abs/2205.14135) in three
progressive CUDA kernels, a Triton companion, and a PyTorch extension — benchmarked against
`torch.nn.functional.scaled_dot_product_attention` on an RTX 3060 Ti.

---

## The Algorithm

Standard attention materializes an N×N score matrix — O(N²) memory. Flash Attention avoids this
by tiling Q, K, V into blocks that fit in SRAM and fusing the softmax into the tile loop using an
**online softmax** recurrence (Milakov & Gimelshein 2018):

```
for each query tile Qi:
    mi, li, Oi = -∞, 0, 0
    for each key/value tile Kj, Vj:
        Sij = Qi @ Kj^T · scale          # (Br, Bc) scores — never stored in full
        mi_new = max(mi, rowmax(Sij))
        α = exp(mi − mi_new)              # rescale old accumulator
        β = exp(Sij − mi_new)             # unnorm probs for this tile
        li = α·li + rowsum(β)
        Oi = (α·li·Oi + β @ Vj) / li    # running weighted avg
    O[Qi rows] = Oi
```

Memory drops from **O(N²)** to **O(N)**. IO complexity drops from O(N²d) HBM reads to O(N²d / M)
where M is SRAM size.

---

## Kernel Progression

| Kernel | Key technique | Notes |
|--------|--------------|-------|
| `flash_attn_v1_naive.cu` | One thread per row, global memory | Direct CUDA translation of Python FA1 |
| `flash_attn_v1_smem.cu`  | Cooperative tile load into `__shared__` | Eliminates redundant HBM reads — core FA1 IO win |
| `flash_attn_v1_warp.cu`  | One warp per row, `__shfl_down_sync` dot products | Qi in registers, parallelized reduction |
| `triton/flash_attn_triton.py` | `tl.dot` + unnormalized accumulator | Python-native, no build step |

---

## Benchmarks (RTX 3060 Ti, B=1 H=8 d=64)

### Latency vs Sequence Length
![Latency](figures/latency_vs_seqlen.png)

### Speedup over Naive CUDA
![Speedup](figures/speedup_vs_naive.png)

### Peak Memory
![Memory](figures/memory_vs_seqlen.png)

---

## Build & Run

**Requirements:** PyTorch ≥ 2.0 with CUDA, Triton, matplotlib, numpy.

```bash
# Build CUDA extension (requires CUDA toolkit, sm_86 GPU)
cd flash_attention
pip install -e .

# Verify correctness of all variants
python verify.py

# Run benchmarks (saves figures/ PNGs)
python benchmark.py
```

---

## Repo Layout

```
flash_attention/
├── csrc/           CUDA kernels + pybind11 bindings
├── triton/         Triton FA1 kernel
├── setup.py        CUDAExtension build
├── verify.py       Correctness tests (gates benchmark)
├── benchmark.py    Timing sweep → figures/
└── figures/        Committed benchmark plots
```

The existing `flash_attention.py` at the repo root is the Python algorithmic reference
that inspired these CUDA implementations.

---

## What's Next

- **Flash Attention v2** — reorder work across warps to increase occupancy; eliminate
  inter-warp communication for the query loop; closer to what `torch.sdpa` uses internally.
- **Backward pass** — recompute softmax from saved `(m, l)` statistics to avoid storing the N×N matrix.
- **FP16 / BF16** — `half2` vectorized loads, Tensor Core `wmma` or `mma.sync`.
```

- [ ] **Step 2: Commit**

```bash
git -C .. add flash_attention/README.md
git -C .. commit -m "docs(flash-attn): write portfolio README with algorithm, kernel table, benchmark plots"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** scaffold ✓, naive kernel ✓, smem kernel ✓, warp kernel ✓, Triton ✓, verify.py ✓, benchmark.py ✓, 3 PNGs ✓, README sections ✓
- [x] **Placeholders:** none — all steps contain actual code
- [x] **Type consistency:** `flash_attn_naive_launch` / `flash_attn_smem_launch` / `flash_attn_warp_launch` signatures match across all tasks; `forward_naive/smem/warp` Python names consistent throughout
- [x] **d % 32 constraint** for warp kernel noted in Global Constraints and kernel comment
- [x] **verify.py gates benchmark.py** via subprocess check ✓
- [x] **Triton graceful skip** in verify.py if not installed ✓
