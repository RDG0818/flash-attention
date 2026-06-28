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
    """One program instance = one query tile (BLOCK_M rows)."""
    i_tile = tl.program_id(0)

    offs_m = i_tile * BLOCK_M + tl.arange(0, BLOCK_M)  # (BLOCK_M,) row indices
    offs_d = tl.arange(0, HEAD_DIM)

    # Load Q tile: (BLOCK_M, HEAD_DIM)
    Q_ptrs = Q_ptr + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    Q = tl.load(Q_ptrs, mask=offs_m[:, None] < N, other=0.0)

    # Running accumulators
    m_i   = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i   = tl.zeros([BLOCK_M], dtype=tl.float32)
    O_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)  # unnormalized

    Tc = tl.cdiv(N, BLOCK_N)
    for j in range(Tc):
        offs_n_j = j * BLOCK_N + tl.arange(0, BLOCK_N)

        # K tile as (HEAD_DIM, BLOCK_N) so tl.dot(Q, K) = (BLOCK_M, BLOCK_N)
        K_ptrs = K_ptr + offs_n_j[None, :] * stride_kn + offs_d[:, None] * stride_kd
        K = tl.load(K_ptrs, mask=offs_n_j[None, :] < N, other=0.0)

        # V tile as (BLOCK_N, HEAD_DIM)
        V_ptrs = V_ptr + offs_n_j[:, None] * stride_vn + offs_d[None, :] * stride_vd
        V = tl.load(V_ptrs, mask=offs_n_j[:, None] < N, other=0.0)

        # Attention scores
        S = tl.dot(Q, K) * scale                                  # (BLOCK_M, BLOCK_N)
        S = tl.where(offs_n_j[None, :] < N, S, float('-inf'))    # mask padding

        # Online softmax — unnormalized accumulator avoids division in inner loop
        m_j   = tl.max(S, axis=1)                                 # (BLOCK_M,)
        m_new = tl.maximum(m_i, m_j)
        alpha = tl.exp(m_i - m_new)                               # rescale old accum
        P     = tl.exp(S - m_new[:, None])                        # (BLOCK_M, BLOCK_N)
        l_j   = tl.sum(P, axis=1)                                 # (BLOCK_M,)
        l_new = alpha * l_i + l_j

        O_acc = alpha[:, None] * O_acc + tl.dot(P, V)
        m_i = m_new
        l_i = l_new

    # Normalize once after loop
    O_out = O_acc / l_i[:, None]

    O_ptrs = O_ptr + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(O_ptrs, O_out, mask=offs_m[:, None] < N)


def flash_attn_triton(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Flash Attention v1 forward via Triton.

    Args:
        Q, K, V: (B, H, N, d) float32 CUDA tensors. d must be a power of two.
    Returns:
        O: (B, H, N, d)
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
