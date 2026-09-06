"""
Paired benchmark: Custom CUDA forward()+L vs PyTorch Flash, 10 reps.

After clock warmup, both implementations run back-to-back with alternating
order. The result is the median of 10 per-rep latency gaps. Custom CUDA returns O and L.
"""
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[1]
FLAGS = ["-O3", "--use_fast_math", "-gencode=arch=compute_89,code=sm_89"]
mod = load(name="attention_forward_cuda", sources=[str(ROOT / "cuda/attention_forward.cu")],
           extra_cuda_cflags=FLAGS, verbose=False)
print(f"so: {mod.__file__}")

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
    print(f"Custom CUDA (+L) vs PyTorch Flash — {REPS} paired reps, "
          f"B={B} H={H} D={D} FP16 non-causal")
    print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print("=" * 100)

    Qb = torch.randn(B, H, 4096, D, device="cuda", dtype=torch.float16)
    for _ in range(100):
        mod.forward_only(Qb, Qb, Qb)
    torch.cuda.synchronize()

    for N in [1024, 2048, 4096]:
        Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        warmup = 30
        iters = 200 if N <= 1024 else (100 if N <= 2048 else 50)

        def run_ours():
            return time_once(lambda: mod.forward(Q, K, V), warmup, iters)

        def run_sdpa():
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return time_once(
                    lambda: F.scaled_dot_product_attention(Q, K, V), warmup, iters)

        t_ours, t_sdpa, gaps = [], [], []
        for r in range(REPS):
            if r % 2 == 0:
                o = run_ours(); s = run_sdpa()
            else:
                s = run_sdpa(); o = run_ours()
            t_ours.append(o)
            t_sdpa.append(s)
            gaps.append((o / s - 1.0) * 100.0)

        print(f"N={N:>5}: Custom CUDA {med(t_ours):.4f}ms  PyTorch Flash {med(t_sdpa):.4f}ms  "
              f"| paired median gap {med(gaps):+.2f}%  "
              f"(per-rep: {', '.join(f'{g:+.1f}' for g in gaps)})")

    print("=" * 100)


if __name__ == "__main__":
    main()
