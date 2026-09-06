"""
Peak GPU memory vs PyTorch's OWN materialized-attention implementations.

Why this script exists: the old demo compared against a hand-written
matmul/softmax expression. A self-authored baseline is not falsifiable, so
every baseline here ships with PyTorch and is reproducible from its public API:

  flash   sdpa_kernel(SDPBackend.FLASH_ATTENTION) + F.scaled_dot_product_attention
  mha16   nn.MultiheadAttention(..., dtype=fp16)(x, x, x, need_weights=True)
          -> must materialize the N x N attention weights in the INPUT dtype
             (fp16). Same-dtype materialized baseline. Includes in/out
             projections (O(N*E), negligible next to N*N).
  math    sdpa_kernel(SDPBackend.MATH) + F.scaled_dot_product_attention
          -> PyTorch's composite reference. On torch 2.10 it UPCASTS fp16
             inputs to fp32 (observed: ampere_sgemm, softmax<float,...>), so
             its column is NOT a same-dtype comparison and is reported with
             that label only.

The dtype of each baseline is OBSERVED, not assumed: the CUDA kernel symbols
each one launches carry their scalar types, and they are printed first.

Peak = max_memory_allocated during the call minus memory already allocated
(the inputs). Memory statistics come from the caching allocator and do not
depend on GPU contention, so no idle gate is needed. This script measures
memory only; latency claims live in compare_pytorch.py.
"""
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.profiler import profile, ProfilerActivity
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[1]
FLAGS = ["-O3", "--use_fast_math", "-gencode=arch=compute_89,code=sm_89"]
mod = load(name="attention_forward_cuda", sources=[str(ROOT / "cuda/attention_forward.cu")],
           extra_cuda_cflags=FLAGS, verbose=False)
print(f"so: {mod.__file__}")


def observed_dtypes(name, fn):
    """Print the CUDA kernels fn launches and the scalar types visible in them."""
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    seen, tags = set(), set()
    print(f"[{name}] CUDA kernels observed:")
    for e in prof.events():
        if e.device_type.name != "CUDA":
            continue
        n = e.name
        key = n[:110]
        if key in seen:
            continue
        seen.add(key)
        t = []
        if re.search(r"(^|[<, (])float([>, )]|$)|sgemm", n):
            t.append("float")
        if re.search(r"Half|half|hgemm|fp16|f16_", n):
            t.append("half")
        tags.update(t)
        print(f"   [{'/'.join(t) or '?':10s}] {key}")
    print(f"   => scalar types: {sorted(tags) or ['unknown']}")
    return tags


def peak_mib(fn):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    try:
        out = fn()
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None
    p = (torch.cuda.max_memory_allocated() - base) / 2**20
    del out
    torch.cuda.empty_cache()
    return p


def fmt(x, w=9):
    return f"{'OOM':>{w}}" if x is None else f"{x:{w}.1f}"


def ratio(num, den, w=8):
    if num is None or den is None or den == 0:
        return f"{'OOM':>{w}}"
    return f"{num / den:{w-1}.1f}x"


def main():
    B, H, D = 1, 8, 64
    E = H * D
    torch.manual_seed(42)
    print("=" * 112)
    print(f"Peak memory: custom CUDA vs PyTorch baselines -- B={B} H={H} D={D}, fp16 inputs, non-causal")
    print(f"GPU: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory // 2**20} MiB)  torch {torch.__version__}")
    print("=" * 112)

    with torch.no_grad():
        mha = nn.MultiheadAttention(E, H, batch_first=True, bias=False,
                                    device="cuda", dtype=torch.float16).eval()

        Qp = torch.randn(B, H, 1024, D, device="cuda", dtype=torch.float16)
        xp = torch.randn(B, 1024, E, device="cuda", dtype=torch.float16)
        t_math = observed_dtypes("math ", lambda: sdpa_math(Qp, Qp, Qp))
        t_mha = observed_dtypes("mha16", lambda: mha(xp, xp, xp, need_weights=True))
        del Qp, xp
        torch.cuda.empty_cache()
        print("=" * 112)
        print(f"{'N':>6} | {'ours':>9} {'flash':>9} {'mha16':>9} {'math':>9} | "
              f"{'mha16/ours':>10} {'math/ours':>10} | {'NxN fp16':>9}")
        print(f"{'':>6} | {'MiB':>9} {'MiB':>9} {'MiB':>9} {'MiB':>9} | "
              f"{'same dtype':>10} {'fp32!':>10} | {'MiB':>9}")
        print("-" * 112)

        for N in [1024, 2048, 4096, 8192]:
            Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
            K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
            V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
            x = torch.randn(B, N, E, device="cuda", dtype=torch.float16)

            m_ours = peak_mib(lambda: mod.forward(Q, K, V))
            m_flash = peak_mib(lambda: sdpa_flash(Q, K, V))
            m_mha = peak_mib(lambda: mha(x, x, x, need_weights=True))
            m_math = peak_mib(lambda: sdpa_math(Q, K, V))
            nxn = N * N * H * 2 / 2**20

            print(f"{N:>6} | {fmt(m_ours)} {fmt(m_flash)} {fmt(m_mha)} {fmt(m_math)} | "
                  f"{ratio(m_mha, m_ours, 10)} {ratio(m_math, m_ours, 10)} | {nxn:9.1f}")
            del Q, K, V, x
            torch.cuda.empty_cache()

    print("=" * 112)
    print("ours   : attention_forward.forward (O + L). flash: SDPA FLASH_ATTENTION backend.")
    print("mha16  : nn.MultiheadAttention need_weights=True, fp16 -> materializes N x N in fp16 "
          f"(observed {sorted(t_mha)}). Same-dtype baseline; includes in/out projections.")
    print("math   : SDPA MATH backend "
          f"(observed {sorted(t_math)}). NOT same-dtype on this torch; reported for reference only.")
    print("NxN fp16: one N x N x H fp16 score tensor = lower bound for ANY fp16 implementation "
          "that materializes scores.")


def sdpa_math(Q, K, V):
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(Q, K, V)


def sdpa_flash(Q, K, V):
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(Q, K, V)


if __name__ == "__main__":
    main()
