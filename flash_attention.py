"""
Flash Attention 1 — forward pass.

Paper: Dao et al. 2022 "FlashAttention: Fast and Memory-Efficient Exact
Attention with IO-Awareness" https://arxiv.org/abs/2205.14135

Key idea: tile Q/K/V into blocks that fit in SRAM. Use online softmax
(running max + sum) to accumulate the output without ever materializing
the full N×N attention matrix. Memory drops from O(N²) to O(N).

This Python implementation is algorithmically faithful — the actual
memory/speed gains require fused CUDA kernels (see Triton/CUDA versions).
"""

import math
import torch
import torch.nn.functional as F


def flash_attention(Q, K, V, mask=None, scale=None, block_size=64):
    """
    Flash Attention 1 forward pass.

    Args:
        Q, K, V : (batch, heads, seq_len, head_dim)
        mask    : bool tensor broadcastable to (batch, heads, seq_len, seq_len)
                  True = keep, False = mask out (e.g. causal mask)
        scale   : softmax scale; defaults to 1/sqrt(head_dim)
        block_size: tile size Br = Bc (SRAM budget proxy)

    Returns:
        O : (batch, heads, seq_len, head_dim)
    """
    B, H, N, d = Q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    Q = Q * scale

    Br = min(block_size, N)
    Bc = min(block_size, N)
    Tr = math.ceil(N / Br)
    Tc = math.ceil(N / Bc)

    O = torch.zeros_like(Q)                                        # output accumulator
    l = torch.zeros(B, H, N, device=Q.device, dtype=Q.dtype)      # running softmax denominator
    m = torch.full((B, H, N), float('-inf'), device=Q.device, dtype=Q.dtype)  # running row max

    for i in range(Tr):
        i0, i1 = i * Br, min((i + 1) * Br, N)

        Qi = Q[:, :, i0:i1, :]    # (B, H, Br, d)
        Oi = O[:, :, i0:i1, :]    # (B, H, Br, d)
        li = l[:, :, i0:i1]       # (B, H, Br)
        mi = m[:, :, i0:i1]       # (B, H, Br)

        for j in range(Tc):
            j0, j1 = j * Bc, min((j + 1) * Bc, N)

            Kj = K[:, :, j0:j1, :]    # (B, H, Bc, d)
            Vj = V[:, :, j0:j1, :]    # (B, H, Bc, d)

            Sij = torch.matmul(Qi, Kj.transpose(-2, -1))   # (B, H, Br, Bc)

            if mask is not None:
                Sij = Sij.masked_fill(~mask[:, :, i0:i1, j0:j1], float('-inf'))

            # --- online softmax update ---
            mij = Sij.amax(dim=-1)                          # (B, H, Br)  block row-max
            Pij = torch.exp(Sij - mij.unsqueeze(-1))        # (B, H, Br, Bc) unnorm probs
            lij = Pij.sum(dim=-1)                           # (B, H, Br)  block row-sum

            mi_new = torch.maximum(mi, mij)                 # (B, H, Br)
            alpha  = torch.exp(mi  - mi_new)                # rescale old accumulator
            beta   = torch.exp(mij - mi_new)                # rescale new block

            li_new = alpha * li + beta * lij

            # Oi = (alpha * li * Oi + beta * Pij @ Vj) / li_new
            Oi = (
                (alpha * li).unsqueeze(-1) * Oi
                + beta.unsqueeze(-1) * torch.matmul(Pij, Vj)
            ) / li_new.unsqueeze(-1)

            mi, li = mi_new, li_new

        O[:, :, i0:i1, :] = Oi
        l[:, :, i0:i1]    = li
        m[:, :, i0:i1]    = mi

    return O


# ---------------------------------------------------------------------------
# Reference implementation for correctness verification
# ---------------------------------------------------------------------------

def causal_mask(seq_len, device='cpu'):
    """Lower-triangular mask. True = attend to."""
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def reference_attention(Q, K, V, mask=None, scale=None):
    if scale is None:
        scale = 1.0 / math.sqrt(Q.shape[-1])
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    if mask is not None:
        S = S.masked_fill(~mask, float('-inf'))
    return torch.matmul(F.softmax(S, dim=-1), V)


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    configs = [
        (1, 1,  16, 8,  None),        # tiny, no mask
        (2, 4, 128, 32, 'causal'),     # typical, causal
        (1, 2,  63, 16, 'causal'),     # non-power-of-two seq len
        (2, 4, 256, 64, None),         # larger, full attention
    ]

    for B, H, N, d, mask_type in configs:
        Q = torch.randn(B, H, N, d, device=device)
        K = torch.randn(B, H, N, d, device=device)
        V = torch.randn(B, H, N, d, device=device)

        mask = None
        if mask_type == 'causal':
            mask = causal_mask(N, device).unsqueeze(0).unsqueeze(0)

        out   = flash_attention(Q, K, V, mask=mask)
        ref   = reference_attention(Q, K, V, mask=mask)
        err   = (out - ref).abs().max().item()

        status = 'PASS' if err < 1e-4 else 'FAIL'
        print(f"[{status}] B={B} H={H} N={N} d={d} mask={mask_type or 'none':6s}  max_err={err:.2e}")
