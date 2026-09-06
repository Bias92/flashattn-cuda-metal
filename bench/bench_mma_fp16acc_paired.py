"""
Paired 10-rep comparison of FP32 and FP16 QK accumulation.
Both custom and fp16acc use forward()+L. The measured delta includes
the precision change in QK accumulation.
"""
import torch
from torch.utils.cpp_extension import load

FLAGS = ["-O3", "--use_fast_math", "-gencode=arch=compute_89,code=sm_89"]
mod_full = load(name="attention_forward_cuda", sources=["cuda/attention_forward.cu"],
                extra_cuda_cflags=FLAGS, verbose=False)
mod_a = load(name="flash_attn_mma_fp16acc", sources=["cuda/flash_attn_mma_fp16acc.cu"],
             extra_cuda_cflags=FLAGS, verbose=False)
for _m in (mod_full, mod_a):
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
    print(f"custom (fp32-acc) vs fp16acc ablation — {REPS} paired reps, forward()+L")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 100)

    Qb = torch.randn(B, H, 4096, D, device="cuda", dtype=torch.float16)
    for _ in range(100):
        mod_full.forward_only(Qb, Qb, Qb)
    torch.cuda.synchronize()

    for N in [1024, 2048, 4096]:
        Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        warmup = 30
        iters = 200 if N <= 1024 else (100 if N <= 2048 else 50)

        t_f, t_a, paired = [], [], []
        for r in range(REPS):
            if r % 2 == 0:
                f = time_once(lambda: mod_full.forward(Q, K, V), warmup, iters)
                a = time_once(lambda: mod_a.forward(Q, K, V), warmup, iters)
            else:
                a = time_once(lambda: mod_a.forward(Q, K, V), warmup, iters)
                f = time_once(lambda: mod_full.forward(Q, K, V), warmup, iters)
            t_f.append(f)
            t_a.append(a)
            paired.append((f - a) / f * 100.0)

        print(f"N={N:>5}: f32acc {med(t_f):.4f}ms  f16acc {med(t_a):.4f}ms  "
              f"| paired median speedup {med(paired):+.2f}%")
        print(f"        per-rep %: {', '.join(f'{p:+.1f}' for p in paired)}")

    print("=" * 100)


if __name__ == "__main__":
    main()
