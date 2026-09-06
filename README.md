# flashattn-cuda

CUDA attention forward for RTX 4060 Ti. FP16 inputs and output, FP32 accumulation,
non-causal self-attention, head dimension 64.

## Run

Use an environment with CUDA-enabled PyTorch, the CUDA toolkit and Ninja installed.
Run these commands from the repository root:

```bash
python3 -m pip install --no-build-isolation -e .
python3 tests/test_attention_forward.py
python3 bench/compare_pytorch.py
```

The package builds only [attention_forward.cu](cuda/attention_forward.cu).
The test and benchmark also support JIT compilation without installing the package.

To profile one forward launch after warmup:

```bash
ncu --profile-from-start off python3 bench/profile_once.py 4096
```

## Measurements

RTX 4060 Ti, WSL2 Ubuntu, CUDA 12.8, PyTorch 2.10.0+cu128.
B=1, H=8, D=64, FP16 inputs, non-causal forward.
[Run record, September 6, 2026](bench/results/rerun_2026-09-06_rtx4060ti.md).

### Latency

[compare_pytorch.py](bench/compare_pytorch.py) compares the current CUDA kernel with
`torch.nn.functional.scaled_dot_product_attention`, explicitly selecting
`SDPBackend.FLASH_ATTENTION`. Both calls use the same FP16 Q/K/V tensors.
The custom call returns the output and softmax logsumexp (`O`, `L`).

Ten paired runs, alternating execution order.

| N | Custom CUDA (O+L) | PyTorch Flash | Paired latency gap |
|---:|---:|---:|---:|
| 1024 | 0.0618 ms | 0.0572 ms | +6.75% |
| 2048 | 0.2194 ms | 0.2183 ms | +1.24% |
| 4096 | 0.8544 ms | 0.8413 ms | +1.37% |

Latency columns are medians for each implementation. The gap is the median of the
per-pair differences `(custom / pytorch - 1) * 100`; positive means custom is slower.

### Memory

[compare_memory.py](bench/compare_memory.py) imports Hugging Face Llama's
`eager_attention_forward` from Transformers 4.57.6. It uses FP16 matmuls and an
FP32 softmax, then casts probabilities back to FP16.

Peak allocated memory above the existing Q/K/V tensors, in MiB:

| N | Custom CUDA (O+L) | PyTorch Flash | HF eager |
|---:|---:|---:|---:|
| 1024 | 1.0 | 1.0 | 64.0 |
| 2048 | 2.1 | 2.1 | 256.0 |
| 4096 | 4.1 | 4.1 | 1024.0 |
| 8192 | 8.2 | 8.3 | 4096.0 |

```bash
python3 -m pip install transformers==4.57.6
python3 bench/compare_memory.py
```

The script also measures SDPA MATH, which uses FP32 intermediates in this environment.

## Implementation

Q rows are split across four warps. QK and PV use `mma.sync`, with online softmax
and output accumulators held in registers. Two shared-memory buffers prefetch K/V
with `cp.async`. Shapes divisible by the tile sizes use a path without boundary
checks; other shapes use the guarded path.

The [correctness test](tests/test_attention_forward.py) checks output and logsumexp
against a PyTorch expression using FP16-rounded inputs and FP32 arithmetic. It
includes non-aligned sequence lengths and larger input magnitudes.

Supported: dense Q/K/V with identical shapes, D=64, non-causal forward.
Causal masking, dropout, variable-length batches, GQA/MQA and backward are not implemented.

## Earlier work

- [Experiments](experiments/): previous kernels, their tests and benchmarks
- [Profiling records](docs/profiling/): Nsight reports and screenshots
- [Old demo recordings](docs/demo/): WMMA versus a separate matmul/softmax expression

Experimental kernels are not included in the default build. Their scripts compile
only the sources they use.
