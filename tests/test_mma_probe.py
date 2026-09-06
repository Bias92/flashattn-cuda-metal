"""
Layout validation for mma.sync.m16n8k16 + ldmatrix on real hardware.
Checks QK, PV, register reuse, and FP16-accumulate QK layouts.
"""
import torch
from torch.utils.cpp_extension import load

torch.manual_seed(0)

mod = load(
    name="mma_probe",
    sources=["cuda/mma_probe.cu"],
    extra_cuda_cflags=["-O2", "-gencode=arch=compute_89,code=sm_89"],
    verbose=False,
)


def report(name, got, ref):
    diff = (got - ref).abs().max().item()
    ok = diff < 1e-3
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<40s} max_diff={diff:.3e}")
    return ok


def main():
    dev = "cuda"
    A = torch.randn(16, 16, device=dev).half()
    K = torch.randn(8, 16, device=dev).half()
    P = torch.randn(16, 16, device=dev).half()
    V = torch.randn(16, 8, device=dev).half()
    Kf = torch.randn(16, 16, device=dev).half()

    ok = True

    # Probe 1: S = A @ K^T
    ref_qk = A.float() @ K.float().T
    ok &= report("QK  variant=0 (no-trans, expected)", mod.probe_qk(A, K, 0), ref_qk)
    d1 = (mod.probe_qk(A, K, 1) - ref_qk).abs().max().item()
    print(f"       control: variant=1 (trans) max_diff={d1:.3e} (expected LARGE)")

    # Probe 2: O = P @ V
    ref_pv = P.float() @ V.float()
    ok &= report("PV  variant=0 (trans, expected)", mod.probe_pv(P, V, 0), ref_pv)
    d2 = (mod.probe_pv(P, V, 1) - ref_pv).abs().max().item()
    print(f"       control: variant=1 (no-trans) max_diff={d2:.3e} (expected LARGE)")

    # Probe 3: C->A register reuse chain
    S = A.float() @ Kf.float().T
    ref_chain = S.half().float() @ V.float()
    ok &= report("CHAIN S=A@Kf^T -> half -> @V", mod.probe_chain(A, Kf, V), ref_chain)

    # Probe 4: f16-accumulate QK layout (looser tol: accumulation itself is fp16)
    got = mod.probe_qk_f16acc(A, K)
    diff = (got - ref_qk).abs().max().item()
    ok4 = diff < 5e-2   # layout errors would be O(1); fp16-acc rounding is ~1e-2
    print(f"[{'PASS' if ok4 else 'FAIL'}] {'QK f16-accumulate layout':<40s} max_diff={diff:.3e}")
    ok &= ok4

    print("=" * 60)
    print("ALL LAYOUT PROBES PASSED" if ok else "LAYOUT PROBE FAILURE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
