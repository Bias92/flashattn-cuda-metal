# Paper Skeleton — ISPASS 2027 target

Working title: *Anatomy of a Scratch FlashAttention Kernel: A Profiling-Driven
Optimization Chain on a Consumer Ada GPU*

원칙: 모든 수치는 `docs/mma_handoff.md` §0의 확정 headline과 §4 ablation 표에서만
가져온다. "faster than SDPA" 계열 표현 금지. 문구는 "within 1–3% of SDPA-Flash
(N=2048/4096, paired), while remaining slower overall".

## 1. Introduction
- 논지: 프로덕션 attention 커널은 블랙박스다. scratch 구현을 *controlled
  experimental vehicle*로 사용해 각 최적화 단계의 비용/이득을 커널 레벨에서 계측.
- 기여 3개:
  (a) mma.sync 기반 FA forward의 단계별 최적화 체인 (3.7x, 최종 SDPA-Flash 1–3% 이내)
  (b) 각 단계의 ncu/SASS 정량 분석 — 무엇이 병목을 옮겼는지
  (c) negative ablation 3건 포함 — 왜 실패했는지가 재현 가능한 형태로
- 명시적 non-goal: SDPA를 이기는 것.

## 2. Background
- FlashAttention forward (online softmax, tiling) — Algorithm 1 요약.
- Ada(sm_89) 실행 모델: mma.sync.m16n8k16, ldmatrix, cp.async, fragment 레이아웃.
- C→A fragment 재사용 트릭 (QK accumulator 레이아웃 == PV A operand 레이아웃).

## 3. Methodology
- **Probe-first**: 커널 작성 전 `mma_probe`로 operand/accumulator 레이아웃을
  하드웨어에서 실측 (QK non-trans, PV trans, C-layout, C→A reuse 각각 단위검증).
- Correctness: half-cast reference (입력 캐스팅 오차와 커널 오차 분리),
  atol 2e-3, N=1~4096 (비정렬/스트레스 포함), O-only 경로 비트일치 확인.
- Benchmark: 10-run paired median, 순서 교대, clock burn-in, CUDA events,
  .so 경로 로깅 (stale binary 방지). forward()+L만 headline (SDPA도 lse 계산).
- 기각 기준 사전 등록 (kill conditions) — §6의 negative ablation에 적용.

## 4. Optimization Chain (positive)
각 단계: 무엇을 바꿨나 → ncu/SASS에서 무엇이 변했나 → 시간이 얼마나 변했나.
| 단계 | N=4096 | 핵심 계측 |
|---|---|---|
| fa2 (WMMA API, scalar PV) | ~3.2ms | roofline: scalar PV의 산술적 하한이 SDPA 전체 시간과 동급 |
| mma (mma.sync, reg softmax) | 1.04ms | sS/sP 왕복 소멸, sync 10+→2/tile |
| +cp.async double buffer | 0.92ms | 로드-계산 오버랩; DRAM 4%, L2 hit 98.5% |
| +address strength reduction | 0.89ms | SASS 정수 연산 ~58%↓, HMMA 간격 4-10→2-4 |
| +FULL_TILES specialization | 0.873ms | ISETP 20→6, SEL 5→1 (+L 커널) |

## 5. Profiling Analysis
- ncu: occupancy 5 blocks/SM (REG/smem 동시 캡), tensor pipe 43% active,
  math-pipe-throttle이 지배 스톨 — "메모리가 아니라 발행 리듬이 병목"이라는 서사.
- SASS 분석: 주소 산술이 exp2f보다 큰 오버헤드였음 (IMAD/LEA vs MUFU 카운트).
- 히스토그램 수치는 방향성 근거로만 (whole-.so 한계 명시).

## 6. Negative Results (동급 비중으로)
| 시도 | 결과 | 해석 |
|---|---|---|
| BC=64 타일 | +5% (느려짐) | KV루프 절반 < occupancy 반토막 (2 blocks/SM) |
| softmax/PV 소스 레벨 interleave | paired -0.6~-1.0% | ptxas가 이미 스케줄링 중; 강제 순서가 자유도만 축소. 비트일치로 순수 스케줄링 효과임을 증명 |
| (fa2 시절) 다수: multi-warp, sP 패딩 등 | 기존 README 표 | — |

## 7. Results
- headline 표 (mma_handoff §0 그대로) + 시퀀스 길이 스케일링.
- 대 naive 메모리 절약 (39x)은 별도 축 — speedup과 혼용 금지.

## 8. Related Work
- FlashAttention 1/2/3, CUTLASS/CuTe, Triton attention, ThunderKittens 등.
- 차별점: 최적화 "결과"가 아니라 "결정 체인 + 실패 포함 계측"이 기여.

## 9. Future Work
- Jetson AGX Orin (sm_87) 동일 커널/방법론 교차 프로파일링 (졸프2 계획).
- causal masking, D=128, backward의 mma 구조 이식.
- cross-iteration 파이프라이닝 (barrier/stage lifetime 재설계 필요 — 난이도 상).

## Artifact
- repo 공개 (MIT), probe→test→bench 재현 스크립트 전부 포함.
- 재현 순서: mma_probe → test_mma_db_full → bench_mma_headline.
