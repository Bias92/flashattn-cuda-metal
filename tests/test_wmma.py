"""
FlashAttention WMMA Correctness Test
======================================
Tests WMMA Tensor Core forward against naive reference.
"""
import torch
import flash_attn_wmma

def naive_attention(Q, K, V):
    D = Q.shape[-1]
    scale = D ** -0.5
    S = Q @ K.transpose(-2, -1) * scale
    P = torch.softmax(S, dim=-1)
    O = P @ V
    L = torch.logsumexp(S, dim=-1)
    return O, L

def test_config(B, H, N, D, device="cuda"):
    torch.manual_seed(42)
    Q = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    K = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    V = torch.randn(B, H, N, D, device=device, dtype=torch.float32)

    # Reference
    O_ref, L_ref = naive_attention(Q, K, V)

    # WMMA forward
    O_wmma, L_wmma = flash_attn_wmma.forward(Q, K, V)

    # FP16 tolerance
    fwd_atol = 1e-2

    O_diff = (O_wmma - O_ref).abs().max().item()
    O_pass = torch.allclose(O_wmma, O_ref, atol=fwd_atol, rtol=fwd_atol)

    status = "PASS" if O_pass else "FAIL"

    print(f"[{status}] B={B}, H={H}, N={N:>5}, D={D}  |  "
          f"O={O_diff:.2e}")

    if not O_pass:
        print(f"       Forward FAILED (max_diff={O_diff:.2e})")

    return O_pass

def main():
    print("=" * 90)
    print("FlashAttention WMMA Tensor Core — Correctness Test")
    print("=" * 90)

    configs = [
        (1, 1,   32, 64),
        (1, 1,   64, 64),
        (1, 1,  128, 64),
        (1, 1,   63, 64),
        (1, 1,  127, 64),
        (2, 4,  256, 64),
        (2, 8,  512, 64),
        (1, 1, 1024, 64),
        (1, 1, 2048, 64),
    ]

    passed = 0
    for B, H, N, D in configs:
        if test_config(B, H, N, D):
            passed += 1

    total = len(configs)
    print("=" * 90)
    print(f"Result: {passed}/{total} passed")
    if passed == total:
        print("All tests passed!")
    else:
        print(f"WARNING: {total - passed} test(s) FAILED")
    print("=" * 90)

if __name__ == "__main__":
    main()
