"""
FlashAttention Backward Benchmark
===================================
Compares naive attention backward vs flash attention backward.
Outputs CSV with timing and memory measurements.
"""
import torch
import flash_attn_cuda
import time
import os
import csv
from datetime import datetime

def naive_attention_backward(Q, K, V, dO):
    """Naive O(N^2) backward — baseline for comparison."""
    D = Q.shape[-1]
    scale = D ** -0.5
    S = Q @ K.transpose(-2, -1) * scale
    P = torch.softmax(S, dim=-1)
    O = P @ V
    dV = P.transpose(-2, -1) @ dO
    dP = dO @ V.transpose(-2, -1)
    Di = (dO * O).sum(dim=-1, keepdim=True)
    dS = P * (dP - Di)
    dQ = dS @ K * scale
    dK = dS.transpose(-2, -1) @ Q * scale
    return dQ, dK, dV

def benchmark_backward(B, H, N, D, warmup=5, repeats=20, device="cuda"):
    torch.manual_seed(42)
    Q = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    K = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    V = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    dO = torch.randn(B, H, N, D, device=device, dtype=torch.float32)

    # Flash forward (need O and L for backward)
    O_flash, L_flash = flash_attn_cuda.forward(Q, K, V)

    # --- Benchmark Naive Backward ---
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        naive_attention_backward(Q, K, V, dO)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(repeats):
        naive_attention_backward(Q, K, V, dO)
        torch.cuda.synchronize()
    naive_ms = (time.perf_counter() - start) / repeats * 1000
    naive_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # --- Benchmark Flash Backward ---
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        flash_attn_cuda.backward(Q, K, V, O_flash, dO, L_flash)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(repeats):
        flash_attn_cuda.backward(Q, K, V, O_flash, dO, L_flash)
        torch.cuda.synchronize()
    flash_ms = (time.perf_counter() - start) / repeats * 1000
    flash_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

    speedup = naive_ms / flash_ms if flash_ms > 0 else float("inf")
    mem_save = naive_mem / flash_mem if flash_mem > 0 else float("inf")

    return {
        "seq_len": N,
        "naive_ms": naive_ms,
        "flash_ms": flash_ms,
        "speedup": speedup,
        "naive_mem_mb": naive_mem,
        "flash_mem_mb": flash_mem,
        "mem_save": mem_save,
    }

def main():
    device = "cuda"
    B, H, D = 1, 8, 64
    seq_lens = [128, 256, 512, 1024, 2048, 4096]

    gpu_name = torch.cuda.get_device_name(0)

    print("=" * 90)
    print(f"FlashAttention Backward Benchmark")
    print(f"GPU: {gpu_name} | Precision: FP32 | Config: B={B}, H={H}, D={D}")
    print("=" * 90)
    print(f"{'Seq Len':>8} | {'Naive (ms)':>10} | {'Flash (ms)':>10} | {'Speedup':>8} | "
          f"{'Naive Mem':>10} | {'Flash Mem':>10} | {'Mem Save':>8}")
    print("-" * 90)

    results = []
    for N in seq_lens:
        try:
            r = benchmark_backward(B, H, N, D, device=device)
            results.append(r)
            print(f"{r['seq_len']:>8} | {r['naive_ms']:>10.2f} | {r['flash_ms']:>10.2f} | "
                  f"{r['speedup']:>7.2f}x | {r['naive_mem_mb']:>8.1f} MB | "
                  f"{r['flash_mem_mb']:>8.1f} MB | {r['mem_save']:>6.2f}x")
        except RuntimeError as e:
            print(f"{N:>8} | SKIPPED ({e})")

    print("=" * 90)

    # Save CSV
    os.makedirs("bench/results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"bench/results/backward_{timestamp}.csv"

    with open(csv_path, "w", newline="") as f:
        f.write(f"# gpu: {gpu_name}\n")
        f.write(f"# dtype: fp32\n")
        f.write(f"# batch: {B}, heads: {H}, head_dim: {D}\n")
        f.write(f"# date: {datetime.now().isoformat()}\n")
        f.write(f"# torch: {torch.__version__}, cuda: {torch.version.cuda}\n")
        writer = csv.DictWriter(f, fieldnames=[
            "batch", "heads", "seq_len", "head_dim",
            "naive_ms", "flash_ms", "speedup",
            "naive_mem_mb", "flash_mem_mb", "mem_save"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "batch": B, "heads": H, "seq_len": r["seq_len"], "head_dim": D,
                "naive_ms": f"{r['naive_ms']:.4f}",
                "flash_ms": f"{r['flash_ms']:.4f}",
                "speedup": f"{r['speedup']:.2f}",
                "naive_mem_mb": f"{r['naive_mem_mb']:.1f}",
                "flash_mem_mb": f"{r['flash_mem_mb']:.1f}",
                "mem_save": f"{r['mem_save']:.2f}",
            })

    print(f"CSV saved to: {csv_path}")

if __name__ == "__main__":
    main()
