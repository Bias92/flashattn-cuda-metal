"""
MMA forward benchmark vs torch SDPA (forced Flash backend).
FP16, non-causal, B=1 H=8 D=64. CUDA-event timing, clock burn-in,
median of repeated runs to resist DVFS drift.
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils.cpp_extension import load

mod = load(
    name="flash_attn_mma",
    sources=["cuda/flash_attn_mma.cu"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_89,code=sm_89"],
    verbose=False,
)
mod_db = load(
    name="flash_attn_mma_db",
    sources=["cuda/flash_attn_mma_db.cu"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_89,code=sm_89"],
    verbose=False,
)


def bench_events(fn, warmup=30, iters=100, reps=3):
    """Median-of-reps, CUDA-event timed average per iteration (ms)."""
    times = []
    for _ in range(reps):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / iters)
    times.sort()
    return times[len(times) // 2]


def main():
    B, H, D = 1, 8, 64
    torch.manual_seed(42)
    print("=" * 84)
    print(f"MMA vs SDPA-Flash forward (B={B}, H={H}, D={D}, FP16, non-causal)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 84)

    # Clock burn-in: heavy work first so early configs aren't measured at idle clocks
    Qb = torch.randn(B, H, 4096, D, device="cuda", dtype=torch.float16)
    for _ in range(50):
        mod.forward_only(Qb, Qb, Qb)
    torch.cuda.synchronize()

    print(f"{'N':>6} | {'mma (ms)':>10} | {'mma-db (ms)':>11} | {'sdpa-flash (ms)':>15} | {'mma/sdpa':>8} | {'db/sdpa':>8}")
    print("-" * 84)
    for N in [128, 256, 512, 1024, 2048, 4096]:
        Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        iters = 200 if N <= 512 else (100 if N <= 2048 else 50)
        t_mma = bench_events(lambda: mod.forward_only(Q, K, V), iters=iters)
        t_db = bench_events(lambda: mod_db.forward_only(Q, K, V), iters=iters)

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            t_sdpa = bench_events(
                lambda: F.scaled_dot_product_attention(Q, K, V), iters=iters)

        print(f"{N:>6} | {t_mma:>10.4f} | {t_db:>11.4f} | {t_sdpa:>15.4f} | "
              f"{t_mma / t_sdpa:>8.2f}x | {t_db / t_sdpa:>7.2f}x")

    print("=" * 84)


if __name__ == "__main__":
    main()
