"""
MMA (mma.sync) correctness test vs FP32 naive reference.
Inputs are FP32 randn converted to half inside the module -> FP16 tolerances.
"""
import torch
from torch.utils.cpp_extension import load

mod = load(
    name="flash_attn_mma",
    sources=["cuda/flash_attn_mma.cu"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_89,code=sm_89"],
    verbose=False,
)


def naive_attention(Q, K, V):
    D = Q.shape[-1]
    scale = D ** -0.5
    S = Q @ K.transpose(-2, -1) * scale
    P = torch.softmax(S, dim=-1)
    O = P @ V
    L = torch.logsumexp(S, dim=-1)
    return O, L


def test_config(B, H, N, D, device="cuda", dtype=torch.float32, amp=1.0):
    torch.manual_seed(42)
    Q = (torch.randn(B, H, N, D, device=device, dtype=torch.float32) * amp).to(dtype)
    K = (torch.randn(B, H, N, D, device=device, dtype=torch.float32) * amp).to(dtype)
    V = (torch.randn(B, H, N, D, device=device, dtype=torch.float32) * amp).to(dtype)

    # Reference from HALF-CAST inputs in fp32 math: isolates kernel error from
    # the unavoidable fp16 input-cast error, so tolerances can be tight
    # (atol 1e-2 would let a ~2% systematic kernel bug pass at large N).
    # (.half() is a no-op for fp16 direct inputs.)
    Qh, Kh, Vh = Q.half().float(), K.half().float(), V.half().float()
    O_ref, L_ref = naive_attention(Qh, Kh, Vh)
    O_mma, L_mma = mod.forward(Q, K, V)
    O_mma = O_mma.float()

    O_diff = (O_mma - O_ref).abs().max().item()
    L_diff = (L_mma - L_ref).abs().max().item()

    # amp>1 blows up |L| ~ amp^2; keep the absolute check on O (bounded by
    # softmax) and rely on rtol for L in stress configs.
    O_pass = torch.allclose(O_mma, O_ref, atol=2e-3 * max(amp, 1.0), rtol=2e-3)
    L_pass = torch.allclose(L_mma, L_ref, atol=2e-3, rtol=1e-3)
    ok = O_pass and L_pass

    tag = f" dtype={str(dtype).split('.')[-1]}" if dtype != torch.float32 else ""
    tag += f" amp={amp:g}" if amp != 1.0 else ""
    print(f"[{'PASS' if ok else 'FAIL'}] B={B}, H={H}, N={N:>5}, D={D}{tag}  |  "
          f"O_diff={O_diff:.3e}  L_diff={L_diff:.3e}")
    if not ok:
        if not O_pass: print(f"       O FAILED")
        if not L_pass: print(f"       L FAILED")
    return ok


def main():
    print("=" * 80)
    print("MMA (mma.sync) Correctness Test")
    print("=" * 80)
    configs = [
        (1, 1,     1, 64),   # sub-warp-tile N
        (1, 1,     2, 64),
        (1, 1,     7, 64),
        (1, 1,    15, 64),
        (1, 1,    31, 64),
        (1, 1,    32, 64),
        (1, 1,    33, 64),   # sub-tile tail
        (1, 1,    64, 64),
        (1, 1,   128, 64),
        (1, 1,    63, 64),   # non-aligned
        (1, 1,   127, 64),
        (2, 4,   256, 64),
        (2, 8,   512, 64),
        (1, 1,  1024, 64),
        (1, 1,  2048, 64),
        (1, 1,  4095, 64),   # large non-aligned tail
        (1, 1,  4096, 64),
    ]
    passed = sum(test_config(*c) for c in configs)
    total = len(configs)

    # FP16 direct inputs (no host-side cast) + large-logit softmax stress
    extras = [
        dict(B=1, H=1, N=64, D=64, dtype=torch.float16),
        dict(B=1, H=1, N=1024, D=64, dtype=torch.float16),
        dict(B=1, H=1, N=128, D=64, amp=8.0),
        dict(B=1, H=1, N=128, D=64, amp=16.0),
        dict(B=1, H=1, N=2048, D=64, amp=16.0),
    ]
    for e in extras:
        passed += test_config(**e)
    total += len(extras)
    print("=" * 80)
    print(f"Result: {passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
