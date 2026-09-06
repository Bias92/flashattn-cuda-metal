"""
Variant comparison: mma-db(+L) vs db_addr(+L / O-only) vs db_full(+L / O-only)
vs SDPA-Flash. Order rotated per rep.

The O-only db_addr and db_full paths skip L computation. SDPA-Flash computes
softmax_lse internally. bench_mma_headline.py compares db_full forward()+L
against SDPA-Flash with 10 paired repetitions.
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils.cpp_extension import load

FLAGS = ["-O3", "--use_fast_math", "-gencode=arch=compute_89,code=sm_89"]
mod_db = load(name="flash_attn_mma_db", sources=["cuda/flash_attn_mma_db.cu"],
              extra_cuda_cflags=FLAGS, verbose=False)
mod_addr = load(name="flash_attn_mma_db_addr", sources=["cuda/flash_attn_mma_db_addr.cu"],
                extra_cuda_cflags=FLAGS, verbose=False)
mod_full = load(name="flash_attn_mma_db_full", sources=["cuda/flash_attn_mma_db_full.cu"],
                extra_cuda_cflags=FLAGS, verbose=False)

# Log loaded binaries to identify stale builds.
for _m in (mod_db, mod_addr, mod_full):
    print(f"so: {_m.__file__}")

REPS = 5


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
    return sorted(xs)[len(xs) // 2]


def main():
    B, H, D = 1, 8, 64
    torch.manual_seed(42)
    print("=" * 110)
    print(f"MMA variants (B={B}, H={H}, D={D}, FP16) — median of {REPS} rotated reps")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 110)

    Qb = torch.randn(B, H, 4096, D, device="cuda", dtype=torch.float16)
    for _ in range(100):
        mod_db.forward_only(Qb, Qb, Qb)
    torch.cuda.synchronize()

    print(f"{'N':>6} | {'db+L (ms)':>10} | {'addr+L':>9} | {'addr O-o':>9} | "
          f"{'full+L':>9} | {'full O-o':>9} | {'sdpa':>8} | {'fullL/sdpa':>9} | {'fullL/addrL':>9}")
    print("-" * 110)
    for N in [1024, 2048, 4096]:
        Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        warmup = 30
        iters = 100 if N <= 2048 else 50

        def run_dbL():
            return time_once(lambda: mod_db.forward(Q, K, V), warmup, iters)

        def run_db():
            return time_once(lambda: mod_db.forward_only(Q, K, V), warmup, iters)

        def run_addr():
            return time_once(lambda: mod_addr.forward_only(Q, K, V), warmup, iters)

        def run_addrL():
            return time_once(lambda: mod_addr.forward(Q, K, V), warmup, iters)

        def run_full():
            return time_once(lambda: mod_full.forward_only(Q, K, V), warmup, iters)

        def run_fullL():
            return time_once(lambda: mod_full.forward(Q, K, V), warmup, iters)

        def run_sdpa():
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return time_once(
                    lambda: F.scaled_dot_product_attention(Q, K, V), warmup, iters)

        runners = {"dbL": run_dbL, "addrL": run_addrL, "addr": run_addr,
                   "fullL": run_fullL, "full": run_full, "sdpa": run_sdpa}
        base = list(runners)
        nk = len(base)
        times = {k: [] for k in base}
        for r in range(REPS):
            order = base[r % nk:] + base[:r % nk]
            for k in order:
                times[k].append(runners[k]())

        m = {k: med(v) for k, v in times.items()}
        print(f"{N:>6} | {m['dbL']:>10.4f} | {m['addrL']:>9.4f} | {m['addr']:>9.4f} | "
              f"{m['fullL']:>9.4f} | {m['full']:>9.4f} | {m['sdpa']:>8.4f} | "
              f"{m['fullL'] / m['sdpa']:>9.2f}x | {m['fullL'] / m['addrL']:>9.2f}x")

    print("=" * 110)


if __name__ == "__main__":
    main()
