"""
Test: FlashAttention forward vs naive attention.
Verifies numerical correctness across multiple configs.
"""

import torch
import sys
sys.path.insert(0, ".")

from ref.naive_attn import naive_attention

try:
    import flash_attn_cuda
except ImportError:
    print("Build first: python setup.py install")
    print("  or: pip install -e .")
    sys.exit(1)


def test_forward(B, H, N, D, atol=1e-3, rtol=1e-3):
    """Compare flash forward output against naive."""
    torch.manual_seed(42)
    Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)

    # Reference
    O_ref, L_ref = naive_attention(Q, K, V)

    # FlashAttention
    O_flash = flash_attn_cuda.forward(Q, K, V)

    # Compare
    max_diff = (O_flash - O_ref).abs().max().item()
    allclose = torch.allclose(O_flash, O_ref, atol=atol, rtol=rtol)

    status = "PASS" if allclose else "FAIL"
    print(f"[{status}] B={B}, H={H}, N={N:>5}, D={D}  |  max_diff={max_diff:.6e}")

    if not allclose:
        # Show where the biggest differences are
        diff = (O_flash - O_ref).abs()
        idx = diff.argmax()
        print(f"  ref={O_ref.flatten()[idx].item():.6f}, "
              f"flash={O_flash.flatten()[idx].item():.6f}")

    return allclose


if __name__ == "__main__":
    print("=" * 60)
    print("FlashAttention Forward Correctness Test")
    print("=" * 60)

    configs = [
        # (B, H, N, D)
        (1, 1, 32, 64),      # minimal: single block
        (1, 1, 64, 64),      # exactly 2 Q blocks (BR=32)
        (1, 1, 128, 64),     # multiple blocks
        (1, 1, 63, 64),      # non-aligned N
        (1, 1, 127, 64),     # non-aligned N
        (2, 4, 256, 64),     # multi-batch multi-head
        (2, 8, 512, 64),     # larger
        (1, 1, 1024, 64),    # 1K sequence
        (1, 1, 2048, 64),    # 2K sequence
    ]

    passed = 0
    for B, H, N, D in configs:
        if test_forward(B, H, N, D):
            passed += 1

    print("=" * 60)
    print(f"Result: {passed}/{len(configs)} passed")
    if passed == len(configs):
        print("All tests passed!")
    else:
        print("Some tests FAILED.")
        sys.exit(1)
