# Profiling records

These records were collected before the directory cleanup. Names such as `fa3` and
`db_full` below identify this repository's older kernels, not external FlashAttention
releases. The original reports, filenames and screenshots are unchanged.

Older FP32/WMMA reports are in [legacy/](legacy/).

## Recorded comparison

Nsight Compute profiles of the earlier `mma` kernel and Custom CUDA at `N=4096`, using one launch
per kernel. The [main latency benchmark](../../bench/compare_pytorch.py) uses CUDA
events and paired runs.

| Metric | mma before | Custom CUDA after |
|---|---:|---:|
| Duration | 1.38 ms | 0.987 ms |
| Compute throughput | 35.18% | 43.32% |
| Memory throughput | 35.18% | 41.84% |
| Achieved occupancy | 43.22% | 38.01% |

The full Nsight Compute reports, text exports, and original uncropped UI
captures are kept under `ncu/` and `ncu_sections/`.

The following FP32 and Custom CUDA profiles use different shapes and dtypes.
Their durations are not directly comparable.

| Metric | old FP32 kernel screenshot | Custom CUDA report |
|---|---:|---:|
| Shape | N=1024, FP32 | N=4096, FP16 input / FP32 accumulate |
| Duration | 1.14 ms | 0.989 ms |
| Compute throughput | 25.30% | 43.33% |
| Memory throughput | 25.30% | 41.78% |
| L1/TEX throughput | 25.70% | 34.76% |
| L2 throughput | 7.81% | 42.31% |
| DRAM throughput | 4.46% | 4.56% |
| Theoretical occupancy | 10.42% | 41.67% |
| Achieved occupancy | 7.90% | 38.06% |
| Active warps per SM | 3.79 | 18.27 |
| Local spill signal | visible in Memory Workload | 0 local memory spilling requests |

Nsight Compute screenshots:

| Section | Before | After |
|---|---|---|
| GPU Speed of Light | <img src="gpu_speed_of_light.png" width="420" alt="before GPU Speed of Light"> | <img src="ncu_compare/after_gpu_sol.png" width="420" alt="after GPU Speed of Light"> |
| Roofline | <img src="roofline_fp32.png" width="420" alt="before roofline"> | <img src="ncu_compare/after_roofline.png" width="420" alt="after roofline"> |
| Memory Workload | <img src="memory_workload.png" width="420" alt="before memory workload"> | <img src="ncu_compare/after_memory_workload.png" width="420" alt="after memory workload"> |
| Occupancy | <img src="occupancy.png" width="420" alt="before occupancy"> | <img src="ncu_compare/after_occupancy.png" width="420" alt="after occupancy"> |
