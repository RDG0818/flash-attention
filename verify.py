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
    (1, 2,  63, 32, None),       # non-power-of-two seq len
    (1, 1,  64, 32, 'causal'),   # causal mask (CUDA/Triton skip — not yet supported)
]


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

    try:
        from fa_triton.flash_attn_triton import flash_attn_triton
        has_triton = True
    except ImportError:
        has_triton = False
        print("  [SKIP] Triton not installed — skipping Triton variant\n")

    all_pass = True
    for B, H, N, d, mask_type in CONFIGS:
        print(f"Config: B={B} H={H} N={N} d={d} mask={mask_type or 'none'}")
        torch.manual_seed(0)
        Q = torch.randn(B, H, N, d, device=DEVICE)
        K = torch.randn(B, H, N, d, device=DEVICE)
        V = torch.randn(B, H, N, d, device=DEVICE)

        mask = None
        if mask_type == 'causal':
            mask = causal_mask(N, device=DEVICE).unsqueeze(0).unsqueeze(0)

        ref = reference_attention(Q, K, V, mask=mask)

        # Python FA1
        try:
            out = py_flash_attn(Q, K, V, mask=mask)
            err = (out - ref).abs().max().item()
            ok = err < TOL
            print(f"  [{'PASS' if ok else 'FAIL'}] {'Python FA1':30s}  max_err={err:.2e}")
            all_pass &= ok
        except Exception as e:
            print(f"  [ERROR] {'Python FA1':30s}  {e}")
            all_pass = False

        # CUDA + Triton (full attention only — mask not yet implemented in kernels)
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
