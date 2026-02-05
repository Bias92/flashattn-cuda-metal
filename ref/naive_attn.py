import math
import torch


def naive_attention(q, k, v, causal: bool = False):
    """
    Reference attention (baseline).
    Shapes (recommended): q,k,v = (B, H, S, D)  fp16/bf16/float32
    Returns: o = (B, H, S, D)
    """
    assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4
    B, H, S, D = q.shape
    assert k.shape == (B, H, S, D) and v.shape == (B, H, S, D)

    # Do softmax in fp32 for numerical stability
    qf = q.float()
    kf = k.float()
    vf = v.float()

    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(qf, kf.transpose(-2, -1)) * scale  # (B,H,S,S)

    if causal:
        # mask upper triangle (j > i)
        mask = torch.triu(torch.ones((S, S), device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

    probs = torch.softmax(scores, dim=-1)  # (B,H,S,S)
    out = torch.matmul(probs, vf)          # (B,H,S,D)

    # return in original dtype (like real kernels)
    return out.to(dtype=q.dtype)


@torch.no_grad()
def max_abs_err(a, b):
    return (a.float() - b.float()).abs().max().item()


if __name__ == "__main__":
    # quick sanity run
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, H, S, D = 2, 4, 128, 64
    q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
    k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

    o_nc = naive_attention(q, k, v, causal=False)
    o_c  = naive_attention(q, k, v, causal=True)
    print("ok", o_nc.shape, o_c.shape)
