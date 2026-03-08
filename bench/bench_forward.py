"""
Benchmark: FlashAttention vs naive attention.
Measures wall-clock time and peak GPU memory.
"""

import torch
import time
import sys
sys.path.insert(0, ".")

from ref.naive_attn import naive_attention

try:
    import flash_attn_cuda
except ImportError:
    print("Build first: python setup.py install")
    sys.exit(1)


def bench_fn(fn, *args, warmup=5, iters=20):
    """Benchmark a function. Returns avg time in ms."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters * 1000  # ms
    return elapsed


def bench_memory(fn, *args):
    """Measure peak GPU memory of a function call."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    fn(*args)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    return peak


def run_benchmark(B, H, N, D):
    torch.manual_seed(42)
    Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)

    # Time
    t_naive = bench_fn(lambda: naive_attention(Q, K, V))
    t_flash = bench_fn(lambda: flash_attn_cuda.forward(Q, K, V))
    speedup = t_naive / t_flash

    # Memory
    torch.cuda.empty_cache()
    mem_naive = bench_memory(lambda: naive_attention(Q, K, V))
    torch.cuda.empty_cache()
    mem_flash = bench_memory(lambda: flash_attn_cuda.forward(Q, K, V))
    mem_ratio = mem_naive / mem_flash if mem_flash > 0 else float("inf")

    print(f"N={N:>5}  |  naive: {t_naive:>8.2f}ms  {mem_naive:>8.1f}MB  |  "
          f"flash: {t_flash:>8.2f}ms  {mem_flash:>8.1f}MB  |  "
          f"speedup: {speedup:.2f}x  mem_save: {mem_ratio:.2f}x")


if __name__ == "__main__":
    B, H, D = 1, 8, 64
    print("=" * 80)
    print(f"FlashAttention Benchmark (B={B}, H={H}, D={D}, fp32)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    seq_lengths = [128, 256, 512, 1024, 2048, 4096]
    for N in seq_lengths:
        run_benchmark(B, H, N, D)
