"""
Benchmark: FP32 FlashAttention vs PyTorch SDPA's math backend.
Measures wall-clock time and peak GPU memory.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.cpp_extension import load
import time

ROOT = Path(__file__).resolve().parents[2]
attention_fp32_cuda = load(
    name="attention_fp32_cuda",
    sources=[str(ROOT / "experiments/cuda/fp32.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_89,code=sm_89"],
    verbose=False,
)


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
    with sdpa_kernel(SDPBackend.MATH):
        t_sdpa = bench_fn(lambda: F.scaled_dot_product_attention(
            Q, K, V, dropout_p=0.0, is_causal=False
        ))
    t_flash = bench_fn(lambda: attention_fp32_cuda.forward(Q, K, V))
    speedup = t_sdpa / t_flash

    # Memory
    torch.cuda.empty_cache()
    with sdpa_kernel(SDPBackend.MATH):
        mem_sdpa = bench_memory(lambda: F.scaled_dot_product_attention(
            Q, K, V, dropout_p=0.0, is_causal=False
        ))
    torch.cuda.empty_cache()
    mem_flash = bench_memory(lambda: attention_fp32_cuda.forward(Q, K, V))
    mem_ratio = mem_sdpa / mem_flash if mem_flash > 0 else float("inf")

    print(f"N={N:>5}  |  SDPA-MATH: {t_sdpa:>8.2f}ms  {mem_sdpa:>8.1f}MB  |  "
          f"FP32 CUDA: {t_flash:>8.2f}ms  {mem_flash:>8.1f}MB  |  "
          f"speedup: {speedup:.2f}x  mem_save: {mem_ratio:.2f}x")


if __name__ == "__main__":
    B, H, D = 1, 8, 64
    print("=" * 80)
    print(f"FP32 FlashAttention vs PyTorch SDPA-MATH (B={B}, H={H}, D={D})")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    seq_lengths = [128, 256, 512, 1024, 2048, 4096]
    for N in seq_lengths:
        run_benchmark(B, H, N, D)
