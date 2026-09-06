"""
Paired 10-rep benchmark: current kernel vs interleaving.

Per rep, the two implementations run back-to-back with alternating order,
and the improvement is computed for that pair. The summary uses a 1%
median improvement threshold for the KEEP/REJECT label.
"""
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[2]
FLAGS = ["-O3", "--use_fast_math", "-gencode=arch=compute_89,code=sm_89"]
mod_full = load(name="attention_forward_cuda", sources=[str(ROOT / "cuda/attention_forward.cu")],
                extra_cuda_cflags=FLAGS, verbose=False)
mod_intl = load(name="attention_interleaved_cuda",
                sources=[str(ROOT / "experiments/cuda/interleaved.cu")],
                extra_cuda_cflags=FLAGS, verbose=False)
for _m in (mod_full, mod_intl):
    print(f"so: {_m.__file__}")

REPS = 10


def time_once(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def med(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def main():
    B, H, D = 1, 8, 64
    torch.manual_seed(42)
    print("=" * 100)
    print(f"current kernel vs interleaving paired ({REPS} reps, forward()+L both sides)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 100)

    Qb = torch.randn(B, H, 4096, D, device="cuda", dtype=torch.float16)
    for _ in range(100):
        mod_full.forward_only(Qb, Qb, Qb)
    torch.cuda.synchronize()

    for N in [2048, 4096]:
        Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        warmup = 30
        iters = 100 if N <= 2048 else 50

        t_full, t_intl, paired = [], [], []
        for r in range(REPS):
            if r % 2 == 0:
                f = time_once(lambda: mod_full.forward(Q, K, V), warmup, iters)
                i = time_once(lambda: mod_intl.forward(Q, K, V), warmup, iters)
            else:
                i = time_once(lambda: mod_intl.forward(Q, K, V), warmup, iters)
                f = time_once(lambda: mod_full.forward(Q, K, V), warmup, iters)
            t_full.append(f)
            t_intl.append(i)
            paired.append((f - i) / f * 100.0)

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            t_sdpa = med([time_once(
                lambda: F.scaled_dot_product_attention(Q, K, V), warmup, iters)
                for _ in range(5)])

        imp = med(paired)
        verdict = "KEEP (>= 1%)" if imp >= 1.0 else "REJECT (< 1%)"
        print(f"N={N}: current {med(t_full):.4f}ms  interleaving {med(t_intl):.4f}ms  "
              f"PyTorch Flash {t_sdpa:.4f}ms | paired median improvement {imp:+.2f}% -> {verdict}")
        print(f"       paired per-rep %: {[f'{p:+.2f}' for p in paired]}")

    print("=" * 100)


if __name__ == "__main__":
    main()
