"""
FlashAttention Backward Correctness Test
=========================================
Compares flash_attn_cuda.backward() against naive PyTorch backward.
Tests dQ, dK, dV across multiple configurations.
"""
import torch
import flash_attn_cuda

def naive_attention_forward(Q, K, V):
    """Naive O(N^2) attention forward — returns O, L"""
    D = Q.shape[-1]
    scale = D ** -0.5
    S = Q @ K.transpose(-2, -1) * scale
    P = torch.softmax(S, dim=-1)
    O = P @ V
    L = torch.logsumexp(S, dim=-1)
    return O, L

def naive_attention_backward(Q, K, V, dO):
    """Naive O(N^2) attention backward — returns dQ, dK, dV"""
    D = Q.shape[-1]
    scale = D ** -0.5
    S = Q @ K.transpose(-2, -1) * scale   # [B, H, N, N]
    P = torch.softmax(S, dim=-1)           # [B, H, N, N]
    O = P @ V                               # [B, H, N, D]

    # dV = P^T @ dO
    dV = P.transpose(-2, -1) @ dO

    # dP = dO @ V^T
    dP = dO @ V.transpose(-2, -1)

    # D_i = rowsum(dO * O)
    Di = (dO * O).sum(dim=-1, keepdim=True)  # [B, H, N, 1]

    # dS = P * (dP - D_i)
    dS = P * (dP - Di)

    # dQ = dS @ K * scale
    dQ = dS @ K * scale

    # dK = dS^T @ Q * scale
    dK = dS.transpose(-2, -1) @ Q * scale

    return dQ, dK, dV

def test_backward(B, H, N, D, device="cuda"):
    """Test backward for a single config."""
    torch.manual_seed(42)
    Q = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    K = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    V = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    dO = torch.randn(B, H, N, D, device=device, dtype=torch.float32)

    # Flash forward to get O and L
    O_flash, L_flash = flash_attn_cuda.forward(Q, K, V)

    # Flash backward
    dQ_flash, dK_flash, dV_flash = flash_attn_cuda.backward(Q, K, V, O_flash, dO, L_flash)

    # Naive backward (reference)
    dQ_ref, dK_ref, dV_ref = naive_attention_backward(Q, K, V, dO)

    # Compare
    dQ_diff = (dQ_flash - dQ_ref).abs().max().item()
    dK_diff = (dK_flash - dK_ref).abs().max().item()
    dV_diff = (dV_flash - dV_ref).abs().max().item()

    atol = 1e-2   # backward has more numerical error due to recomputation
    rtol = 1e-2

    dQ_pass = torch.allclose(dQ_flash, dQ_ref, atol=atol, rtol=rtol)
    dK_pass = torch.allclose(dK_flash, dK_ref, atol=atol, rtol=rtol)
    dV_pass = torch.allclose(dV_flash, dV_ref, atol=atol, rtol=rtol)

    all_pass = dQ_pass and dK_pass and dV_pass
    status = "PASS" if all_pass else "FAIL"

    print(f"[{status}] B={B}, H={H}, N={N:>5}, D={D}  |  "
          f"dQ_diff={dQ_diff:.6e}  dK_diff={dK_diff:.6e}  dV_diff={dV_diff:.6e}")

    if not all_pass:
        if not dQ_pass: print(f"       dQ FAILED (max_diff={dQ_diff:.6e})")
        if not dK_pass: print(f"       dK FAILED (max_diff={dK_diff:.6e})")
        if not dV_pass: print(f"       dV FAILED (max_diff={dV_diff:.6e})")

    return all_pass

def main():
    print("=" * 80)
    print("FlashAttention Backward Correctness Test")
    print("=" * 80)

    configs = [
        # (B, H, N, D)
        (1, 1,   32, 64),   # single block
        (1, 1,   64, 64),   # exactly 2 Q blocks
        (1, 1,  128, 64),   # multi block
        (1, 1,   63, 64),   # non-aligned
        (1, 1,  127, 64),   # non-aligned
        (2, 4,  256, 64),   # multi batch, multi head
        (2, 8,  512, 64),   # larger multi batch/head
        (1, 1, 1024, 64),   # 1K sequence
        (1, 1, 2048, 64),   # 2K sequence
    ]

    passed = 0
    total = len(configs)

    for B, H, N, D in configs:
        if test_backward(B, H, N, D):
            passed += 1

    print("=" * 80)
    print(f"Result: {passed}/{total} passed")
    if passed == total:
        print("All tests passed!")
    else:
        print(f"WARNING: {total - passed} test(s) FAILED")
    print("=" * 80)

if __name__ == "__main__":
    main()
