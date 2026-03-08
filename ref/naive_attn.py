"""
Naive (standard) attention implementation for correctness verification.
O(N^2) memory - materializes full attention matrix.
"""

import torch
import torch.nn.functional as F


def naive_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Standard scaled dot-product attention.
    
    Args:
        Q: [B, H, N, D] query
        K: [B, H, N, D] key
        V: [B, H, N, D] value
    
    Returns:
        O: [B, H, N, D] output
        L: [B, H, N] logsumexp (for backward pass verification)
    """
    B, H, N, D = Q.shape
    scale = D ** -0.5

    # S = Q @ K^T * scale  ->  [B, H, N, N]
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale

    # P = softmax(S)  ->  [B, H, N, N]
    P = F.softmax(S, dim=-1)

    # O = P @ V  ->  [B, H, N, D]
    O = torch.matmul(P, V)

    # logsumexp for backward verification
    L = torch.logsumexp(S, dim=-1)  # [B, H, N]

    return O, L


if __name__ == "__main__":
    torch.manual_seed(42)
    B, H, N, D = 2, 4, 128, 64
    Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)

    O, L = naive_attention(Q, K, V)
    print(f"Q: {Q.shape}, O: {O.shape}, L: {L.shape}")
    print(f"O max: {O.max().item():.4f}, O min: {O.min().item():.4f}")
