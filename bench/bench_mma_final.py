"""
Final MMA benchmark with variance control.

Protocol:
  - clock burn-in before any measurement
  - per config: 5 reps of (mma, mma-db, sdpa-flash), measurement ORDER
    ROTATED each rep so DVFS drift hits every implementation in every
    position instead of biasing whichever always ran last
  - CUDA-event timing, per-rep average over `iters` launches
  - report median and (min..max) spread across reps
  - note: mma forward_only returns O only but still computes L internally,
    matching SDPA-Flash which also computes softmax_lse unconditionally
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils.cpp_extension import load

FLAGS = ["-O3", "--use_fast_math", "-gencode=arch=compute_89,code=sm_89"]
mod = load(name="flash_attn_mma", sources=["cuda/flash_attn_mma.cu"],
           extra_cuda_cflags=FLAGS, verbose=False)
mod_db = load(name="flash_attn_mma_db", sources=["cuda/flash_attn_mma_db.cu"],
              extra_cuda_cflags=FLAGS, verbose=False)

# stale-.so guard: always show exactly which binaries this run measured
for _m in (mod, mod_db):
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
    xs = sorted(xs)
    return xs[len(xs) // 2]


def main():
    B, H, D = 1, 8, 64
    torch.manual_seed(42)
    print("=" * 100)
    print(f"MMA final benchmark (B={B}, H={H}, D={D}, FP16, non-causal) — "
          f"median of {REPS} interleaved reps, spread in parens")
    print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print("=" * 100)

    Qb = torch.randn(B, H, 4096, D, device="cuda", dtype=torch.float16)
    for _ in range(100):
        mod_db.forward_only(Qb, Qb, Qb)
    torch.cuda.synchronize()

    hdr = f"{'N':>6} | {'mma (ms)':>22} | {'mma-db (ms)':>22} | {'sdpa-flash (ms)':>22} | {'db/sdpa':>7}"
    print(hdr)
    print("-" * 100)
    for N in [128, 256, 512, 1024, 2048, 4096]:
        Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        warmup = 30
        iters = 200 if N <= 512 else (100 if N <= 2048 else 50)

        def run_mma():
            return time_once(lambda: mod.forward_only(Q, K, V), warmup, iters)

        def run_db():
            return time_once(lambda: mod_db.forward_only(Q, K, V), warmup, iters)

        def run_sdpa():
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return time_once(
                    lambda: F.scaled_dot_product_attention(Q, K, V), warmup, iters)

        runners = {"mma": run_mma, "db": run_db, "sdpa": run_sdpa}
        base = ["mma", "db", "sdpa"]
        times = {k: [] for k in base}
        for r in range(REPS):
            order = base[r % 3:] + base[:r % 3]   # rotate order each rep
            for k in order:
                times[k].append(runners[k]())

        def fmt(ts):
            return f"{med(ts):8.4f} ({min(ts):.4f}..{max(ts):.4f})"

        print(f"{N:>6} | {fmt(times['mma']):>22} | {fmt(times['db']):>22} | "
              f"{fmt(times['sdpa']):>22} | "
              f"{med(times['db']) / med(times['sdpa']):>6.2f}x")

    print("=" * 100)


if __name__ == "__main__":
    main()
