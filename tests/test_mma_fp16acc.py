"""FP16-accumulate QK accuracy across input scales.

Reports O/L error against the FP32 reference from half-cast inputs and
db_full (FP32 accumulation). Only amp=1 configurations affect the exit
status. The amp>=4 configurations report errors without pass/fail checks.
"""
import torch
from torch.utils.cpp_extension import load

FLAGS = ["-O3", "--use_fast_math", "-gencode=arch=compute_89,code=sm_89"]
mod = load(name="flash_attn_mma_fp16acc", sources=["cuda/flash_attn_mma_fp16acc.cu"],
           extra_cuda_cflags=FLAGS, verbose=False)
mod_full = load(name="flash_attn_mma_db_full", sources=["cuda/flash_attn_mma_db_full.cu"],
                extra_cuda_cflags=FLAGS, verbose=False)
print(f"so: {mod.__file__}")
print(f"so: {mod_full.__file__}")


def naive_attention(Q, K, V):
    D = Q.shape[-1]
    scale = D ** -0.5
    S = Q @ K.transpose(-2, -1) * scale
    P = torch.softmax(S, dim=-1)
    return P @ V, torch.logsumexp(S, dim=-1)


def run_config(B, H, N, D, amp=1.0, gate=True, device="cuda"):
    torch.manual_seed(42)
    Q = (torch.randn(B, H, N, D, device=device) * amp).half()
    K = (torch.randn(B, H, N, D, device=device) * amp).half()
    V = (torch.randn(B, H, N, D, device=device) * amp).half()

    Qh, Kh, Vh = Q.float(), K.float(), V.float()
    O_ref, L_ref = naive_attention(Qh, Kh, Vh)
    O_a, L_a = mod.forward(Q, K, V)
    O_f, L_f = mod_full.forward(Q, K, V)

    O_diff = (O_a.float() - O_ref).abs().max().item()
    L_diff = (L_a - L_ref).abs().max().item()
    O_vs_f32acc = (O_a.float() - O_f.float()).abs().max().item()
    L_vs_f32acc = (L_a - L_f).abs().max().item()

    if gate:
        ok = (torch.allclose(O_a.float(), O_ref, atol=5e-3, rtol=5e-3)
              and torch.allclose(L_a, L_ref, atol=2e-2, rtol=5e-3))
        status = "PASS" if ok else "FAIL"
    else:
        ok = True
        status = "INFO"

    print(f"[{status}] N={N:>5} amp={amp:>4g}  |  vs fp32-ref: O={O_diff:.3e} L={L_diff:.3e}"
          f"  |  vs db_full(f32acc): O={O_vs_f32acc:.3e} L={L_vs_f32acc:.3e}")
    return ok


def main():
    print("=" * 100)
    print("fp16-accumulate QK — accuracy characterization")
    print("=" * 100)
    gated = [
        dict(B=1, H=1, N=64, D=64),
        dict(B=1, H=1, N=127, D=64),
        dict(B=2, H=8, N=512, D=64),
        dict(B=1, H=1, N=1024, D=64),
        dict(B=1, H=1, N=4095, D=64),
        dict(B=1, H=1, N=4096, D=64),
    ]
    info = [
        dict(B=1, H=1, N=1024, D=64, amp=4.0, gate=False),
        dict(B=1, H=1, N=1024, D=64, amp=8.0, gate=False),
        dict(B=1, H=1, N=2048, D=64, amp=16.0, gate=False),
    ]
    passed = sum(run_config(**c) for c in gated)
    for c in info:
        run_config(**c)
    print("=" * 100)
    print(f"Gated (amp=1): {passed}/{len(gated)} passed; amp>=4 rows are characterization data")
    return 0 if passed == len(gated) else 1


if __name__ == "__main__":
    raise SystemExit(main())
