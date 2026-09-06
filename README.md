# flashattn-cuda

FlashAttention-2 style forward in CUDA and inline PTX, no CUTLASS.
FP16 in, FP32 accumulate, D=64, non-causal. RTX 4060 Ti (sm_89).

## Latency vs PyTorch SDPA Flash

B=1, H=8, D=64, fp16. 10 paired runs, median. Positive gap = custom slower.

| N | Custom CUDA (O+L) | PyTorch Flash | gap |
|---:|---:|---:|---:|
| 1024 | 0.0618 ms | 0.0572 ms | +6.75% |
| 2048 | 0.2194 ms | 0.2183 ms | +1.24% |
| 4096 | 0.8544 ms | 0.8413 ms | +1.37% |

## Peak memory

MiB above the Q/K/V inputs. HF eager = Transformers `eager_attention_forward`, unmodified.

| N | Custom CUDA (O+L) | PyTorch Flash | HF eager |
|---:|---:|---:|---:|
| 1024 | 1.0 | 1.0 | 64.0 |
| 2048 | 2.1 | 2.1 | 256.0 |
| 4096 | 4.1 | 4.1 | 1024.0 |
| 8192 | 8.2 | 8.3 | 4096.0 |

![](docs/profiling/memory.png)

## Kernel

`cuda/attention_forward.cu`. Four warps per block, each owns 16 query rows.
QK and PV on `mma.sync`, softmax and O kept in registers, K/V double-buffered with `cp.async`.
Not implemented: causal, dropout, varlen, GQA, backward.

## Files

`bench/compare_pytorch.py`, `bench/compare_memory.py`, `tests/test_attention_forward.py`.
Run record: `bench/results/`. Earlier kernels: `experiments/`. Profiles: `docs/profiling/`.
