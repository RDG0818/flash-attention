import sys
import os
import subprocess

# Correctness gate — benchmark refuses to run if verify.py fails
result = subprocess.run(
    [sys.executable, os.path.join(os.path.dirname(__file__), 'verify.py')],
    capture_output=True, text=True,
)
if result.returncode != 0:
    print("verify.py failed — fix correctness before benchmarking.")
    print(result.stdout)
    sys.exit(1)
print("verify.py passed. Starting benchmarks...\n")

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import flash_attn
from fa_triton.flash_attn_triton import flash_attn_triton

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
    return float(np.median(times))  # ms


def measure_memory(fn, Q, K, V):
    torch.cuda.reset_peak_memory_stats(DEVICE)
    fn(Q, K, V)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(DEVICE) / 1024 ** 2  # MB


latency = {name: [] for name in VARIANTS}
memory  = {name: [] for name in VARIANTS}

for N in SEQ_LENS:
    print(f"N={N}", flush=True)
    Q = torch.randn(B, H, N, D, device=DEVICE)
    K = torch.randn(B, H, N, D, device=DEVICE)
    V = torch.randn(B, H, N, D, device=DEVICE)
    for name, fn in VARIANTS.items():
        t = time_fn(fn, Q, K, V)
        m = measure_memory(fn, Q, K, V)
        latency[name].append(t)
        memory[name].append(m)
        print(f"  {name:15s}  {t:8.3f} ms   {m:7.2f} MB")

FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

COLORS = {
    'naive CUDA': '#e74c3c',
    'smem CUDA':  '#e67e22',
    'warp CUDA':  '#f1c40f',
    'Triton':     '#2ecc71',
    'torch.sdpa': '#3498db',
}

# --- Plot 1: Latency vs sequence length ---
fig, ax = plt.subplots(figsize=(10, 6))
for name, times in latency.items():
    ax.plot(SEQ_LENS, times, marker='o', label=name, color=COLORS[name], linewidth=2)
ax.set_xscale('log', base=2)
ax.set_xlabel('Sequence Length', fontsize=13)
ax.set_ylabel('Latency (ms, median of 50 runs)', fontsize=13)
ax.set_title('Flash Attention FA1 — Latency vs Sequence Length\n(RTX 3060 Ti, B=1 H=8 d=64)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'latency_vs_seqlen.png'), dpi=150)
plt.close(fig)

# --- Plot 2: Speedup over naive CUDA ---
naive_times = latency['naive CUDA']
speedups = {
    name: [naive_times[i] / t for i, t in enumerate(times)]
    for name, times in latency.items()
    if name != 'naive CUDA'
}
x = np.arange(len(SEQ_LENS))
width = 0.18
fig, ax = plt.subplots(figsize=(13, 6))
for idx, (name, spd) in enumerate(speedups.items()):
    ax.bar(x + idx * width, spd, width, label=name, color=COLORS[name], alpha=0.85)
ax.axhline(1.0, color='gray', linestyle='--', alpha=0.6, linewidth=1.5)
ax.set_xlabel('Sequence Length', fontsize=13)
ax.set_ylabel('Speedup over Naive CUDA  (higher = better)', fontsize=13)
ax.set_title('Flash Attention FA1 — Speedup over Naive CUDA\n(RTX 3060 Ti, B=1 H=8 d=64)', fontsize=14)
ax.set_xticks(x + width * len(speedups) / 2)
ax.set_xticklabels([str(n) for n in SEQ_LENS])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'speedup_vs_naive.png'), dpi=150)
plt.close(fig)

# --- Plot 3: Peak memory vs sequence length ---
fig, ax = plt.subplots(figsize=(10, 6))
for name, mems in memory.items():
    ax.plot(SEQ_LENS, mems, marker='o', label=name, color=COLORS[name], linewidth=2)
ax.set_xlabel('Sequence Length', fontsize=13)
ax.set_ylabel('Peak Memory (MB)', fontsize=13)
ax.set_title('Flash Attention FA1 — Peak Memory vs Sequence Length\n(RTX 3060 Ti, B=1 H=8 d=64)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'memory_vs_seqlen.png'), dpi=150)
plt.close(fig)

print(f"\nFigures saved to {FIG_DIR}/")
