"""Single mma-db forward launch for ncu profiling. N=4096 B=1 H=8 D=64 FP16."""
import torch
from torch.utils.cpp_extension import load

mod = load(
    name="flash_attn_mma_db",
    sources=["cuda/flash_attn_mma_db.cu"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_89,code=sm_89"],
    verbose=False,
)

Q = torch.randn(1, 8, 4096, 64, device="cuda", dtype=torch.float16)
K = torch.randn(1, 8, 4096, 64, device="cuda", dtype=torch.float16)
V = torch.randn(1, 8, 4096, 64, device="cuda", dtype=torch.float16)

for _ in range(5):  # warmup launches (ncu skips these via --launch-skip)
    mod.forward_only(Q, K, V)
torch.cuda.synchronize()
mod.forward_only(Q, K, V)
torch.cuda.synchronize()
print("done")
