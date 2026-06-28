# Flash Attention CUDA Implementation — Design Spec

**Date:** 2026-06-28
**Scope:** FA1 forward pass — CUDA (3 staged kernels) + Triton + PyTorch extension + benchmarks + README

---

## Goal

Implement Flash Attention v1 forward pass from scratch in CUDA, expose it as a PyTorch extension, add a Triton companion kernel, and benchmark all variants against `torch.nn.functional.scaled_dot_product_attention`. Package as a portfolio-grade GitHub repo.

FA2 is explicitly out of scope for this phase but is the planned next chapter (teased in README).

---

## Repo Structure

All work lives under `flash_attention/` inside the existing `ml` repo. Existing files (`gpt.py`, `bigram.py`, `flash_attention.py`, etc.) are untouched.

```
flash_attention/
├── csrc/
│   ├── flash_attn_v1_naive.cu      # stage 1: global memory, mirrors Python FA1
│   ├── flash_attn_v1_smem.cu       # stage 2: Q/K/V tiles in shared memory
│   ├── flash_attn_v1_warp.cu       # stage 3: warp shuffle + vectorized loads
│   └── bindings.cpp                # torch extension entry point (registers all 3)
├── triton/
│   └── flash_attn_triton.py        # Triton FA1 kernel (no build step)
├── setup.py                        # torch.utils.cpp_extension build
├── verify.py                       # correctness check for all variants
├── benchmark.py                    # timing sweep, saves PNGs to figures/
├── figures/                        # committed benchmark plots
│   ├── latency_vs_seqlen.png
│   ├── speedup_vs_naive.png
│   └── memory_vs_seqlen.png
└── README.md
```

---

## CUDA Kernel Progression

Each kernel exposes the same interface: `(Q, K, V, scale, B, H, N, d) → O` via flat device pointers. `bindings.cpp` wraps all three and registers them under `flash_attn.forward_naive`, `flash_attn.forward_smem`, `flash_attn.forward_warp`.

### Stage 1 — Naive (`flash_attn_v1_naive.cu`)

One CUDA thread per output row `i`. Thread loops over all K/V column tiles in global memory, executing the FA1 online softmax recurrence. Direct translation of the Python `flash_attention.py` loops into CUDA. No shared memory.

**Purpose:** establish correctness in CUDA, provide the baseline latency that tiling will beat.

### Stage 2 — Shared Memory Tiled (`flash_attn_v1_smem.cu`)

One thread block per query tile (Br rows). Threads cooperatively load `Qi`, `Kj`, `Vj` tiles into `__shared__` buffers before computing `S = Q @ K^T`. Eliminates redundant global memory reads across the inner loop.

**Purpose:** demonstrate the IO-aware speedup. This is the core contribution of the FA1 paper — analogous to the tiling in `matmul_flat.cu`.

### Stage 3 — Warp Optimized (`flash_attn_v1_warp.cu`)

Keeps shared memory tiling. Adds:
- `__shfl_xor_sync` warp reductions for online softmax max/sum (avoids shared memory round-trips for reductions)
- `float4` vectorized loads for Q/K/V

**Purpose:** push toward peak hardware utilization, demonstrate awareness of warp-level CUDA programming.

### Build

```
setup.py  →  torch.utils.cpp_extension.CUDAExtension
             sources: [csrc/flash_attn_v1_naive.cu, csrc/flash_attn_v1_smem.cu,
                       csrc/flash_attn_v1_warp.cu, csrc/bindings.cpp]
             extra_compile_args: -arch=sm_86 -O3 --use_fast_math
```

---

## Triton Kernel

`triton/flash_attn_triton.py` — single `@triton.jit` kernel. Each program instance handles one query tile (matching FA1 outer loop structure). Uses `tl.load` with block pointers, `tl.dot` for `Q @ K^T`, explicit online softmax with `tl.max` + `tl.exp`. No build step — importable as plain Python.

Role: readable algorithmic companion to the dense CUDA kernels, and independent benchmark point between warp CUDA and `torch.sdpa`.

---

## Correctness (`verify.py`)

Runs all 5 variants against `reference_attention` from `flash_attention.py`:
- Python FA1 (existing)
- Naive CUDA
- Shared memory CUDA
- Warp CUDA
- Triton

Test cases: full attention, causal mask, non-power-of-two sequence length. Prints pass/fail table with max absolute error per variant. `benchmark.py` refuses to run if `verify.py` exits non-zero.

---

## Benchmarking (`benchmark.py`)

**Sweep:** `N ∈ {128, 256, 512, 1024, 2048, 4096}`, fixed `B=1, H=8, d=64`.

**Variants timed:** naive CUDA, smem CUDA, warp CUDA, Triton, `torch.sdpa`.

**Timing:** `torch.cuda.Event` with 10 warmup iterations + 50 timed iterations, median reported.

**Memory:** `torch.cuda.memory_allocated()` delta per variant (shows O(N) vs O(N²) gap).

**Output — 3 PNGs saved to `figures/`:**

1. `latency_vs_seqlen.png` — all variants on one plot, log-x axis
2. `speedup_vs_naive.png` — bar chart, speedup relative to naive CUDA at each N
3. `memory_vs_seqlen.png` — allocated bytes vs N, highlights FA's O(N) memory vs naive O(N²)

---

## README

Sections:
1. **What this is** — 2-sentence pitch
2. **The algorithm** — inline tiling diagram + online softmax recurrence (LaTeX)
3. **Kernel progression** — table: kernel | key technique | latency at N=2048 | speedup
4. **Benchmark plots** — embeds `figures/*.png`
5. **Build & run** — exact shell commands
6. **What's next** — FA2 (backward pass teased as future work)

---

## Target Hardware

RTX 3060 Ti — Ampere, compute capability 8.6. Tile sizes and warp counts tuned to GA104: 48KB shared memory default (configurable to 100KB), 4096 threads/SM.

---

## Out of Scope (This Phase)

- Backward pass
- Flash Attention v2
- Multi-head attention module wrapper (the kernels are standalone)
- FP16 / BF16 (FP32 only for clarity)
- Batch > 1 in kernel (loop over batch in Python caller)
