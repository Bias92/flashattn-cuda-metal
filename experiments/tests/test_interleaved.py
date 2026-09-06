"""MMA interleave (softmax/PV interleave): correctness vs half-cast FP32 reference.

Also reports bitwise equality with the current kernel. Exit status depends on
reference tolerances and agreement between forward and forward_only outputs.
"""
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[2]

FLAGS = ["-O3", "--use_fast_math", "-gencode=arch=compute_89,code=sm_89"]
mod = load(name="attention_interleaved_cuda",
           sources=[str(ROOT / "experiments/cuda/interleaved.cu")],
           extra_cuda_cflags=FLAGS, verbose=False)
mod_full = load(name="attention_forward_cuda",
                sources=[str(ROOT / "cuda/attention_forward.cu")],
                extra_cuda_cflags=FLAGS, verbose=False)
print(f"so: {mod.__file__}")
print(f"so: {mod_full.__file__}")

BR, BC = 64, 32


def naive_attention(Q, K, V):
    D = Q.shape[-1]
    scale = D ** -0.5
    S = Q @ K.transpose(-2, -1) * scale
    P = torch.softmax(S, dim=-1)
    return P @ V, torch.logsumexp(S, dim=-1)


def test_config(B, H, N, D, device="cuda", dtype=torch.float32, amp=1.0):
    torch.manual_seed(42)
    Q = (torch.randn(B, H, N, D, device=device, dtype=torch.float32) * amp).to(dtype)
    K = (torch.randn(B, H, N, D, device=device, dtype=torch.float32) * amp).to(dtype)
    V = (torch.randn(B, H, N, D, device=device, dtype=torch.float32) * amp).to(dtype)

    Qh, Kh, Vh = Q.half().float(), K.half().float(), V.half().float()
    O_ref, L_ref = naive_attention(Qh, Kh, Vh)
    O_i, L_i = mod.forward(Q, K, V)
    O_if = O_i.float()
    O_only = mod.forward_only(Q, K, V)
    oo_same = torch.equal(O_only, O_i)

    O_f, L_f = mod_full.forward(Q, K, V)
    bit_same = torch.equal(O_i, O_f) and torch.equal(L_i, L_f)

    O_diff = (O_if - O_ref).abs().max().item()
    L_diff = (L_i - L_ref).abs().max().item()
    ok = (torch.allclose(O_if, O_ref, atol=2e-3 * max(amp, 1.0), rtol=2e-3)
          and torch.allclose(L_i, L_ref, atol=2e-3, rtol=1e-3)
          and oo_same)
    path = "full" if (N % BR == 0 and N % BC == 0) else "guarded"
    tag = f" dtype={str(dtype).split('.')[-1]}" if dtype != torch.float32 else ""
    tag += f" amp={amp:g}" if amp != 1.0 else ""
    print(f"[{'PASS' if ok else 'FAIL'}] B={B}, H={H}, N={N:>5}, D={D} [{path:>7}]{tag}  |  "
          f"O_diff={O_diff:.3e}  L_diff={L_diff:.3e}  o_only={oo_same}  bit==full={bit_same}")
    return ok, bit_same


def main():
    print("=" * 96)
    print("MMA interleave (softmax/PV interleave) Correctness Test")
    print("=" * 96)
    configs = [
        (1, 1, 64, 64), (1, 1, 128, 64), (2, 4, 256, 64), (2, 8, 512, 64),
        (1, 1, 1024, 64), (1, 1, 2048, 64), (1, 1, 4096, 64),
        (1, 1, 1, 64), (1, 1, 2, 64), (1, 1, 7, 64), (1, 1, 31, 64),
        (1, 1, 33, 64), (1, 1, 63, 64), (1, 1, 127, 64), (1, 1, 4095, 64),
    ]
    extras = [
        dict(B=1, H=1, N=1024, D=64, dtype=torch.float16),
        dict(B=1, H=1, N=127, D=64, dtype=torch.float16),
        dict(B=1, H=1, N=2048, D=64, amp=16.0),
        dict(B=1, H=1, N=4095, D=64, amp=16.0),
    ]

    passed, bit_all = 0, True
    for c in configs:
        ok, bit = test_config(*c)
        passed += ok
        bit_all &= bit
    for e in extras:
        ok, bit = test_config(**e)
        passed += ok
        bit_all &= bit
    total = len(configs) + len(extras)

    print("=" * 96)
    print(f"Result: {passed}/{total} passed | bitwise == custom: {bit_all}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
