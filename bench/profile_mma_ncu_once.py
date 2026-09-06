"""Profile exactly one MMA-family forward launch with Nsight Compute.

Usage:
  python3 bench/profile_mma_ncu_once.py mma 1024
  python3 bench/profile_mma_ncu_once.py db_full 1024

Run ncu with --profile-from-start off so warmup and tensor initialization do
not pollute the report.
"""
import sys

import torch
from torch.utils.cpp_extension import load


FLAGS = ["-O3", "--use_fast_math", "-gencode=arch=compute_89,code=sm_89"]


def load_variant(name):
    if name == "mma":
        return load(
            name="flash_attn_mma",
            sources=["cuda/flash_attn_mma.cu"],
            extra_cuda_cflags=FLAGS,
            verbose=False,
        )
    if name == "db_full":
        return load(
            name="flash_attn_mma_db_full",
            sources=["cuda/flash_attn_mma_db_full.cu"],
            extra_cuda_cflags=FLAGS,
            verbose=False,
        )
    raise SystemExit(f"unknown variant: {name}")


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "db_full"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    mod = load_variant(variant)
    print(f"variant={variant}")
    print(f"so={mod.__file__}")

    torch.manual_seed(42)
    q = torch.randn(1, 8, n, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 8, n, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 8, n, 64, device="cuda", dtype=torch.float16)

    for _ in range(10):
        mod.forward(q, k, v)
    torch.cuda.synchronize()

    torch.cuda.profiler.start()
    mod.forward(q, k, v)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print("profiled")


if __name__ == "__main__":
    main()
