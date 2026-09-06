# Experiments

Earlier implementations and tuning experiments. None are built by the root
`setup.py`. Each test or benchmark builds the CUDA sources it uses through
`torch.utils.cpp_extension.load`.

The current kernel is [attention_forward.cu](../cuda/attention_forward.cu).

## Implementations

| Source | Difference | Test |
|---|---|---|
| [fp32.cu](cuda/fp32.cu) | Scalar FP32 tiled attention | [test_fp32.py](tests/test_fp32.py) |
| [wmma.cu](cuda/wmma.cu) | WMMA matmuls, shared-memory score and probability tiles | [test_wmma.py](tests/test_wmma.py) |
| [mma.cu](cuda/mma.cu) | Direct `mma.sync`, register softmax | [test_mma.py](tests/test_mma.py) |
| [double_buffer.cu](cuda/double_buffer.cu) | Adds K/V `cp.async` double buffering and an O-only path | [test_double_buffer.py](tests/test_double_buffer.py) |
| [precomputed_addresses.cu](cuda/precomputed_addresses.cu) | Hoists copy and matrix-load address calculations | [test_precomputed_addresses.py](tests/test_precomputed_addresses.py) |
| [tile64.cu](cuda/tile64.cu) | 64-row K/V tile | [test_tile64.py](tests/test_tile64.py) |
| [interleaved.cu](cuda/interleaved.cu) | Interleaves softmax and PV work | [test_interleaved.py](tests/test_interleaved.py) |
| [fp16_accumulation.cu](cuda/fp16_accumulation.cu) | FP16 QK accumulation | [test_fp16_accumulation.py](tests/test_fp16_accumulation.py) |
| [layout_probe.cu](cuda/layout_probe.cu) | Checks MMA/ldmatrix fragment layouts | [test_layout_probe.py](tests/test_layout_probe.py) |

## Benchmarks

Run from the repository root with the same CUDA/PyTorch environment as the main kernel.

| Script | Measures |
|---|---|
| [compare_fp32_pytorch.py](bench/compare_fp32_pytorch.py) | FP32 implementation vs PyTorch SDPA MATH |
| [compare_mma_buffering.py](bench/compare_mma_buffering.py) | MMA and double-buffered MMA vs PyTorch Flash |
| [compare_variants.py](bench/compare_variants.py) | Double buffer, precomputed addresses and current kernel, O+L/O-only |
| [compare_interleaving.py](bench/compare_interleaving.py) | Current kernel vs interleaved experiment, plus PyTorch Flash |
| [compare_fp16_accumulation.py](bench/compare_fp16_accumulation.py) | Current kernel vs FP16-accumulation experiment |

```bash
python3 experiments/tests/test_layout_probe.py
python3 experiments/bench/compare_mma_buffering.py
```

`bench/profile_double_buffer.py` profiles the older double-buffered O-only path.
The shell scripts in `bench/` inspect compiled extensions with `cuobjdump`; they
use the CUDA 12.8 paths from the original profiling environment.

## Recorded optimization history

The current kernel adds full-tile specialization to the precomputed-addresses
implementation. Old `fa3`/`db_full` labels in figures refer to these local revisions.

At `N=4096`, forward latency decreased from about 3.2 ms to about 0.873 ms.

<p align="center">
  <img src="../docs/profiling/fa3_optimization_chain.png" width="720" alt="Optimization chain">
</p>

| Attempt | Result | Status |
|---|---|---|
| FP32 tiled baseline | FP32 reference implementation | kept as baseline |
| WMMA path | initial Tensor Core implementation | superseded |
| mma | direct `mma.sync`, register softmax, no shared S/P round trip | kept in chain |
| mma-db | added two-stage K/V `cp.async` | kept in chain |
| db_addr | removed repeated shared/global address calculations and reduced SASS integer instructions | kept in chain |
| Custom CUDA | removed full-tile predicates for N divisible by the tile size | benchmark kernel |
| fp16-acc QK | about 21-24% faster with FP16 QK accumulation, but loses accuracy on larger logits | ablation only |
| BC=64 tile | doubles K/V shared storage from 18 KiB to 36 KiB | rejected |
| softmax/PV source interleave | correct and nearly bit-identical, but slower in paired runs | rejected |
| cross-iteration precompute | correct, but the extra live state hurt scheduling more than it helped | rejected |
| launch bounds, max register count, PAD changes, static N only | no stable win in paired runs | rejected |

Earlier FP32 and WMMA experiments:

| # | Attempt | Result | Observation |
|---:|---|---|---|
| 1 | tile size 32 -> 16 plus `launch_bounds` | 11.18 ms -> 15.44 ms | K/V loop count doubled while register pressure stayed high |
| 2 | FP16 shared memory | 11.18 ms -> 16.28 ms | occupancy improved, but `half2float` conversion became the cost |
| 3 | WMMA 16x16 | 11.18 ms -> 11.83 ms | Tensor Cores were not enough with small tiles and scalar softmax between MMA phases |
| 4 | WMMA plus `half2` load | 11.63 ms at N=4096, 0.09 ms at N=128 | removed spill and helped small N, but still not enough for large N |
| 5 | multi-warp, 4 warps per block | rolled back | about 35 KB shared memory per block cut residency too hard |
| 6 | `sQ/sK/sV` +8 padding | 10.41 ms at N=4096 | shared load bank conflict dropped from 4.6-way to 2.7-way |
| 7 | `sO` +1 padding | slower | conflict barely moved, so residual conflict was not from the output accumulator |
| 8 | `sP` +8 padding | 10.41 ms -> 10.55 ms | store conflict improved, but runtime regressed from shared-memory pressure |
