"""
Peak GPU memory vs the official eager attention and PyTorch's SDPA backends.

Why this script exists: the old demo compared against a hand-written
matmul/softmax expression. A self-authored baseline is not falsifiable, so
every baseline here is an unmodified, importable implementation:

  eager   transformers.models.llama.modeling_llama.eager_attention_forward
          -> the function HuggingFace runs for attn_implementation="eager".
             Imported as-is (not copied). matmul in the input dtype, softmax
             in fp32 then cast back, matmul in the input dtype. This is what
             "eager attention" means in the ecosystem.
  flash   sdpa_kernel(SDPBackend.FLASH_ATTENTION) + F.scaled_dot_product_attention
  math    sdpa_kernel(SDPBackend.MATH) + F.scaled_dot_product_attention
          -> PyTorch's composite reference. Accepts fp16 inputs but keeps
             ALL intermediates (QK^T, softmax, PV) in fp32 on torch 2.10
             (observed). eager keeps matmuls in fp16 and only the softmax in
             fp32, so its footprint is closer to a real fp16 deployment; math
             is shown for reference with that difference labeled.

The compute dtype of eager and math is OBSERVED from the CUDA kernel symbols
they launch and printed before any ratio.

Peak = max_memory_allocated during the call minus memory already allocated
(the inputs). Allocator statistics do not depend on GPU contention, so no
idle gate is needed. Memory only; latency claims live in compare_pytorch.py.
"""
import re
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.profiler import profile, ProfilerActivity
from torch.utils.cpp_extension import load

# bench/profile.py shadows the stdlib "profile" module when this script is run as
# "python3 bench/compare_memory.py" (script dir becomes sys.path[0]); transformers
# imports it indirectly and fails. Drop the script dir before importing.
_here = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _here, str(Path(_here)))]

import transformers  # noqa: E402
from transformers.models.llama.modeling_llama import eager_attention_forward

ROOT = Path(__file__).resolve().parents[1]
FLAGS = ["-O3", "--use_fast_math", "-gencode=arch=compute_89,code=sm_89"]
mod = load(name="attention_forward_cuda", sources=[str(ROOT / "cuda/attention_forward.cu")],
           extra_cuda_cflags=FLAGS, verbose=False)
print(f"so: {mod.__file__}")
print(f"transformers {transformers.__version__}  eager_attention_forward from "
      f"{eager_attention_forward.__module__}")

# eager_attention_forward only reads these two attributes from `module`.
EAGER_MODULE = types.SimpleNamespace(num_key_value_groups=1, training=False)


def eager(Q, K, V):
    out, _ = eager_attention_forward(EAGER_MODULE, Q, K, V, attention_mask=None,
                                     scaling=Q.shape[-1] ** -0.5, dropout=0.0)
    return out


def sdpa_math(Q, K, V):
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(Q, K, V)


def sdpa_flash(Q, K, V):
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(Q, K, V)


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


def ratio(num, den, w=10):
    if num is None or den is None or den == 0:
        return f"{'OOM':>{w}}"
    return f"{num / den:{w-1}.1f}x"


def main():
    B, H, D = 1, 8, 64
    torch.manual_seed(42)
    print("=" * 108)
    print(f"Peak memory: custom CUDA vs eager / SDPA baselines -- B={B} H={H} D={D}, fp16 inputs, non-causal")
    print(f"GPU: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory // 2**20} MiB)  torch {torch.__version__}")
    print("=" * 108)

    with torch.no_grad():
        Qp = torch.randn(B, H, 1024, D, device="cuda", dtype=torch.float16)
        t_eager = observed_dtypes("eager", lambda: eager(Qp, Qp, Qp))
        t_math = observed_dtypes("math ", lambda: sdpa_math(Qp, Qp, Qp))
        del Qp
        torch.cuda.empty_cache()
        print("=" * 108)
        print(f"{'N':>6} | {'ours':>9} {'flash':>9} {'eager':>9} {'math':>9} | "
              f"{'eager/ours':>10} {'math/ours':>10} | {'NxN fp16':>9}")
        print(f"{'':>6} | {'MiB':>9} {'MiB':>9} {'MiB':>9} {'MiB':>9} | "
              f"{'':>10} {'(fp32 int.)':>10} | {'MiB':>9}")
        print("-" * 108)

        for N in [1024, 2048, 4096, 8192]:
            Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
            K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
            V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

            m_ours = peak_mib(lambda: mod.forward(Q, K, V))
            m_flash = peak_mib(lambda: sdpa_flash(Q, K, V))
            m_eager = peak_mib(lambda: eager(Q, K, V))
            m_math = peak_mib(lambda: sdpa_math(Q, K, V))
            nxn = N * N * H * 2 / 2**20

            print(f"{N:>6} | {fmt(m_ours)} {fmt(m_flash)} {fmt(m_eager)} {fmt(m_math)} | "
                  f"{ratio(m_eager, m_ours)} {ratio(m_math, m_ours)} | {nxn:9.1f}")
            del Q, K, V
            torch.cuda.empty_cache()

    print("=" * 108)
    print("ours  : attention_forward.forward (O + L).   flash: SDPA FLASH_ATTENTION backend.")
    print(f"eager : transformers eager_attention_forward, unmodified (observed {sorted(t_eager)}: "
          "fp16 matmuls, softmax computed in fp32 then cast back -- that is how eager runs in practice).")
    print(f"math  : SDPA MATH backend (observed {sorted(t_math)}); accepts fp16 but computes all "
          "intermediates (QK^T, softmax, PV) in fp32 on this torch. Shown for reference; the")
    print("        difference from eager is degree (all-fp32 vs fp32-softmax-only), not kind.")
    print("NxN fp16: one N x N x H fp16 score tensor = lower bound for any implementation that "
          "materializes scores in fp16.")


if __name__ == "__main__":
    main()
