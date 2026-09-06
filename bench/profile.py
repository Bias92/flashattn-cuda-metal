"""Profile one current-kernel forward+L launch with Nsight Compute.

Usage:
  python3 bench/profile.py 1024

Use ncu --profile-from-start off to exclude warmup and tensor initialization.
"""
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[1]
FLAGS = ["-O3", "--use_fast_math", "-gencode=arch=compute_89,code=sm_89"]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    mod = load(
        name="attention_forward_cuda",
        sources=[str(ROOT / "cuda/attention_forward.cu")],
        extra_cuda_cflags=FLAGS,
        verbose=False,
    )
    print("kernel=current")
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
