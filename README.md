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
ncu --profile-from-start off python3 bench/profile.py 4096
```

## PyTorch comparison

[compare_pytorch.py](bench/compare_pytorch.py) compares the current CUDA kernel with
`torch.nn.functional.scaled_dot_product_attention`, explicitly selecting
`SDPBackend.FLASH_ATTENTION`. Both calls use the same FP16 Q/K/V tensors.
The custom call returns the output and softmax logsumexp (`O`, `L`).

Recorded on RTX 4060 Ti, WSL2 Ubuntu, CUDA 12.8, PyTorch 2.10.0+cu128.
B=1, H=8, D=64, non-causal forward. Ten paired runs, alternating execution order.

| N | Custom CUDA (O+L) | PyTorch Flash | Paired latency gap |
|---:|---:|---:|---:|
| 1024 | 0.0620 ms | 0.0584 ms | +5.3% |
| 2048 | 0.2221 ms | 0.2182 ms | +1.3% |
| 4096 | 0.8726 ms | 0.8480 ms | +1.6% |

Latency columns are medians for each implementation. The gap is the median of the
per-pair differences `(custom / pytorch - 1) * 100`; positive means custom is slower.

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
