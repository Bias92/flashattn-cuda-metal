# mma.sync 커널 핸드오프 — 2026-07-07

## 0. 결과 요약 (전부 검증된 수치)

RTX 4060 Ti, B=1 H=8 D=64 FP16 non-causal, torch 2.10.0+cu128, CUDA 12.8.
5회 반복(측정 순서 rep마다 회전) 중앙값, CUDA event 타이밍, 클럭 burn-in 후 측정
(`bench/bench_mma_final.py`).

**최종 커널 = mma-db-full (freeze).** 확정 headline — `bench/bench_mma_headline.py`,
**10-run paired median**, 양쪽 다 forward()+L (SDPA도 softmax_lse 항상 계산):

| N | db_full+L (ms) | SDPA-Flash (ms) | paired median gap |
|---|---|---|---|
| 1024 | 0.0620 | 0.0584 | +5.3% |
| 2048 | 0.2221 | 0.2182 | **+1.3%** |
| 4096 | 0.8726 | 0.8480 | **+1.6%** |

공식 문구 (이대로만 인용):
> On RTX 4060 Ti, for B=1, H=8, D=64, FP16 input / FP32 accumulate, non-causal
> forward, `db_full(+L)` reaches **within 1–3% of PyTorch SDPA-Flash at N=2048/4096**
> in paired runs, **while remaining slower overall**. Compared with the fa2
> scalar-PV baseline, latency improves from roughly 3.2 ms to roughly 0.87 ms.

> (한) SDPA를 이긴 건 아니지만, scratch CUDA FA forward가 FP16 input/FP32 accumulate
> 조건에서 N=2048/4096 기준 PyTorch SDPA-Flash의 1–3% 근처까지 접근했고, fa2
> scalar-PV baseline 대비 약 3.6배 개선됐다.

고정 문구 (ablation 경계, 이대로만):
> Final mainline: mma-db-full, FP16 input / FP32 accumulate.
> Ablation only: fp16-acc QK.
> fp16-acc QK improves N=4096 latency by 21–24%, but changes the numerical
> contract and fails robustness as logit amplitude increases. It is not used
> for the SDPA comparison headline.

- 개선 사슬 (N=4096): fa2 ~3.2ms → mma 1.04 → db 0.92 (**fa2→db ~3.4x**) → addr 0.89
  → **full 0.873 (fa2→full ~3.6–3.7x)** — 배수 인용 시 기준 커널 명시할 것
- 달성 처리량 N=4096: 34.4 GFLOP / 0.873ms ≈ **39 TFLOPS**

### 리소스 (fresh build 2026-07-07 21:02 KST에서 cuobjdump 재검증)

| 커널 (인스턴스) | REG |
|---|---|
| db_addr guarded O-only / +L | 80 / 80 |
| db_full guarded O-only / +L | 80 / 80 |
| db_full FULL_TILES O-only | 84 |
| db_full FULL_TILES **+L (headline 경로)** | **95** — REG≤85(6 blocks/SM) 미충족 |

LOCAL 0, SHARED 18432B (전 인스턴스). **주의: REG는 리빌드마다 흔들림** (96→80 사례
2회) — 인용 전 반드시 fresh build에서 cuobjdump 재확인. 벤치가 .so 경로 출력하는
이유가 이것.

### SASS 핵심 diff (+L 커널 기준, db_addr → db_full FULL_TILES)

ISETP 20→6, SEL 5→1, IMAD 54→48, HMMA 32 불변. (히스토그램류 수치는 방향성 근거
전용 — 논문에 정밀 수치로 인용 금지)
- N≤512 구간은 순서 회전 벤치에서 spread가 중앙값 이상으로 벌어짐 (런치/디스패치 지배)
  → **이 구간 수치는 인용 금지.** "SDPA보다 빠르다" 주장 금지 유지.
- correctness: **22/22** (N=1,2,7,15,31,32,33,63,64,127,128,256,512,1024,2048,4095,4096
  + FP16 직통 2개 + amp=8/16 대형 로짓 스트레스 3개)
  - 레퍼런스 = half-cast 입력의 fp32 계산 (입력 캐스팅 오차와 커널 오차 분리)
  - **커널 자체 오차: O_diff ≤ 5.3e-4, L_diff ≤ 1.9e-6** (amp=1 기준; amp=16 스트레스는 O 1.6e-2)
- `forward_only`는 **진짜 O-only임** (`WRITE_L=false` 템플릿: epilogue logf/L store/L 할당
  전부 skip). 단 SDPA는 softmax_lse를 무조건 계산하므로 O-only 비교는 우리가 일을 덜 하는
  것 → **headline은 반드시 `forward()`(+L) 숫자 사용.** (실측 L 코스트 ~0.4%, 노이즈 수준)

## 1. 파일

| 파일 | 내용 |
|---|---|
| `cuda/mma_probe.cu` + `tests/test_mma_probe.py` | mma.sync/ldmatrix 레이아웃 실측 검증 (3/3 PASS). 커널 수정 전 반드시 재실행 |
| `cuda/flash_attn_mma.cu` | v1: mma.sync + 레지스터 상주 softmax. REG 80, spill 0, smem 9KB |
| `cuda/flash_attn_mma_db.cu` | v2(최종): v1 + cp.async 2-stage K/V 더블버퍼. REG 95, spill 0, smem 18KB |
| `tests/test_mma.py`, `tests/test_mma_db.py` | correctness 11 configs |
| `cuda/flash_attn_mma_db_addr.cu` + `tests/test_mma_db_addr.py` | db + address strength-reduction (§4-⑥). REG 96, N=4096 ~0.89ms |
| `cuda/flash_attn_mma_db_full.cu` + `tests/test_mma_db_full.py` | **최종 (freeze)**: db_addr + FULL_TILES 특수화 (§4-⑦). N%64==0이면 predicate-free 경로, 아니면 db_addr와 동일한 guarded 경로. full+L REG 95, N=4096 0.873ms |
| `bench/bench_mma_headline.py` | **headline 확정용**: db_full+L vs SDPA, 10-run paired median. README/논문 수치는 여기서만 |
| `cuda/flash_attn_mma_bc64.cu` + `tests/test_mma_bc64.py` | BC=64 negative ablation (5% 느림, §4-③). setup.py 미등록, JIT 로드 |
| `cuda/flash_attn_mma_db_full_intl.cu` + `tests/test_mma_db_full_intl.py` + `bench/bench_mma_intl_paired.py` | softmax/PV interleave negative ablation (§4-⑧, paired -0.6~-1%). setup.py 미등록 |
| `cuda/flash_attn_mma_fp16acc.cu` + `tests/test_mma_fp16acc.py` + `bench/bench_mma_fp16acc_paired.py` | **fp16-acc QK ablation** (§4-⑨, +22% but 정확도 붕괴 특성 포함). headline 금지. setup.py 미등록 |
| `bench/bench_mma_forward.py` | 3-way 벤치 (개발용) |
| `bench/bench_mma_final.py` | 최종 벤치 (분산 통제 프로토콜, 순서 회전) |
| `bench/bench_mma_variants.py` | db+L / db O-only / bc64 / SDPA 4-way 비교 |
| `bench/profile_mma_once.py` | ncu 단일 런치 타겟 |
| `bench/sass_histo.sh` | SASS opcode 히스토그램 + HMMA 간격 분석 |

빌드: pip 없이 `torch.utils.cpp_extension.load()` JIT (테스트/벤치 스크립트가 알아서 빌드).
`setup.py`에도 `flash_attn_mma`, `flash_attn_mma_db` 모듈 추가됨.

## 2. 왜 이 설계인가 (5W1H)

**fa2가 3.7x 느렸던 근본 원인** = scalar PV. PV 17.2 GFLOP을 CUDA 코어(FP32 ~22 TFLOPS 피크)로 돌리면
100% 가동 가정으로도 0.78ms — SDPA 전체 시간과 같음. 산수로 막혀 있었음.
WMMA로 PV를 못 살린 이유 = fragment 레이아웃 불투명 → sP shared 왕복 강제.

**해결** = `mma.sync.m16n8k16` 직접 사용:
- QK accumulator(C fragment)의 레지스터 레이아웃 == PV A operand 레이아웃 (FA2 논문의 핵심 트릭)
- → softmax를 레지스터에서 수행 (row max/sum은 쿼드 4-lane `__shfl_xor` 1,2)
- → P를 레지스터에서 half로 패킹해 바로 PV mma 투입. **sS/sP shared 왕복 소멸**
- KV블록당 sync 10회+ → 2회

**구조**: BR=64 (4 warp × m16 rows), BC=32, D=64 고정.
QK 16 mma + PV 16 mma per KV block per warp. exp2f 기반 log2-domain online softmax
(scale×log2e 폴딩, L = m·ln2 + ln(l)로 자연로그 복원).
shared는 K/V 타일만: Q는 스테이징 버퍼에 한 번 올려 ldmatrix로 레지스터에 뽑고 버퍼 재사용.

**레이아웃은 전부 실측 검증** (`mma_probe`, sm_89에서 3/3):
- QK B operand: K-storage([n][k] row-major)에 ldmatrix **non-trans**
- PV B operand: V-storage([k][n] row-major)에 ldmatrix **x2.trans**
- C→A 재사용: a0=h2(c0,c1)/a1=h2(c2,c3) (tile n0-7), a2/a3 (tile n8-15)
- C store: c0,c1→(row=l/4, col=2(l%4)+{0,1}), c2,c3→row+8
- PAD=8 (stride 72 halves = 144B): ldmatrix 뱅크 컨플릭트 프리 + cp.async 16B 정렬 (144=9×16)

**v2 (cp.async)**: 다음 K/V 타일을 `cp.async.cg` 16B로 미리 발행, 현재 타일 계산과 오버랩.
tail 행은 src-size 0 → 하드웨어 zero-fill. Q 스테이징은 stage-1 버퍼 자리에서 수행하고
그동안 stage-0에 kv0 로드가 이미 날아감. 효과: N=4096 1.044 → 0.928ms (-11%).

## 3. ncu 진단 (mma-db, N=4096)

- DRAM 4.3%, **L2 히트 98.5%** (K+V 2MB가 L2 32MB 상주) → 메모리 병목 아님
- spill 0, divergence 없음 (31.99 active threads/warp)
- occupancy 37.7% (이론 41.7%, REG 5블록/smem 5블록 제한)
- 스톨 1위: **math pipe throttle 49%** — tensor pipe 43% active, ALU 20%, FMA 12%
- 해석: fp32-acc HMMA는 이슈 슬롯 기준 실효 ~86% 수준. 남은 갭은 exp2f(XU) 버스트 + softmax 스칼라.

## 4. 추가 최적화 시도 결과 (2026-07-07 저녁, 전부 실측)

| # | 시도 | 결과 | 판정 |
|---|---|---|---|
| ① | forward_only 진짜 O-only (WRITE_L 템플릿, L 할당도 skip) | N=4096: 0.9254 vs db+L 0.9289ms (~0.4%) | 노이즈 수준. L 쓰기는 원래 공짜였음 (BH·N번 logf+store vs O(N²) 메인루프). **벤치 headline은 forward(+L) 사용** — SDPA는 lse 무조건 계산하므로 O-only 비교는 우리가 일 덜 하는 것 |
| ③ | BC=64 커널 (`flash_attn_mma_bc64.cu`) | REG 140 spill 0, smem 36.9KB → 2블록/SM (occ 16.7%). N=4096 **0.968ms, db 대비 5% 느림** | **기각.** KV루프/sync 절반 < occupancy 손실. BC=32 유지. negative ablation으로 기록. correctness는 22/22 통과 |
| ④ | SASS 리듬 분석 (`bench/sass_histo.sh`) | HMMA 64개 대비 **정수 연산 ~815개** (IMAD 298, LEA 181, SHF 136, IADD3 131, LOP3 69). float 스칼라 ~250, MUFU 41. HMMA 간격: mma 체인 내부 4-10, **반복당 ~318명령 HMMA-free 구간 1개** (softmax+pack+cp.async 발행+주소 재계산) | **범인은 exp2f가 아니라 정수 주소 연산** (HMMA당 int ~13개). ldmatrix 주소가 `(i&1)*STAGE`에서 매 반복 풀 재계산됨 |

| ⑥ | **주소 strength-reduction** (`flash_attn_mma_db_addr.cu`): issue_kv 스레드당 4-copy 고정형(오프셋 프리컴퓨트), 스테이지 토글 = base 포인터 swap (XOR 금지 — STAGE_BYTES 0x2400이 offset bit와 겹침), ldmatrix 주소 = cur_base + 상수 | **REG 96** (95→96, ±0; 초기 "80" 기록은 스테일 빌드 오독 — cuobjdump 재검증으로 정정) spill 0, SASS 정수 연산 ~58%↓ (IMAD 298→101, LEA 181→76), HMMA 간격 4-10→2-4. N=4096 **0.878~0.893ms, db 대비 -4~5%, SDPA 대비 1.05~1.06x** | **채택. 새 최종 커널.** 이득은 occupancy가 아니라 순수 명령 수 감소에서 나옴 (5블록 캡 동일). correctness 21/21, diff는 db와 완전 동일 |

| ⑦ | **FULL_TILES 특수화** (`flash_attn_mma_db_full.cu`): `N%BR==0 && N%BC==0`일 때 호스트가 predicate-free 인스턴스로 디스패치 — cp.async 가드/tail 마스크/epilogue bounds 전부 컴파일 아웃. guarded 경로는 db_addr와 동일 | 함수 단위 SASS(+L): **ISETP 20→6, SEL 5→1, IMAD 54→48** (HMMA 32 불변). full+L REG 95, spill 0. addr 대비 -3~6% | **채택. 최종 커널 (freeze).** 19/19 correctness (full/guarded 양 경로), diff 동일. 확정 수치는 §0 headline 참조 |

| ⑧ | **softmax/PV interleave** (`flash_attn_mma_db_full_intl.cu`): slice별 exp+pack→PV mma 발행 순서 재배열, rs 셔플/l/m 갱신은 마지막 HMMA와 오버랩되게 지연. 수학 순서 불변 → **db_full과 비트 일치 확인** (19/19) | REG 94/LOCAL 0 (통과)였으나 **10-rep paired median: N=2048 -1.02%, N=4096 -0.61% — 오히려 느림** | **기각** (사전 kill condition: <1% 개선). 해석: ptxas가 이미 스칼라를 HMMA 사이에 스케줄링 중 — 소스 레벨 재배열은 방해만 됨. negative ablation |

| ⑨ | **fp16-acc QK ablation** (`flash_attn_mma_fp16acc.cu`): QK mma만 `m16n8k16.f16.f16.f16.f16` (컨슈머 Ada TC 이슈레이트 2배), S는 mma 직후 fp32 언팩, softmax/PV는 불변. f16 C-fragment 레이아웃은 `mma_probe.probe_qk_f16acc`로 사전 실측 | **+21~24% (10-rep paired, N=1024/2048/4096), N=4096 0.873→0.688ms (~50 TFLOPS)**. 정확도: amp=1에서 O오차 4e-4~3e-3 (사용가능), **amp=4부터 붕괴 (0.27), amp=16에서 18.1**. REG 92-94, spill 0 | **논문 ablation 전용 — mainline/headline 금지** (SDPA는 fp32-acc, 계약 다름). 서사: "fp32-acc의 비용 = 런타임 ~22%, 포기하면 로짓 O(10)부터 softmax 붕괴 — 프로덕션 커널이 fp32-acc를 유지하는 이유" |

클린업 (2026-07-07 밤): ① `WRITE_L=false`에서 L 포인터 산술 제거 (`if constexpr` 안으로) — db_addr/db_full 적용, 비트일치 재확인 ② setup.py에 db_addr/db_full 등록 ③ 모든 벤치가 측정한 .so 경로 출력 (스테일 바이너리 오독 방지)

### 남은 레버 (미실행, 우선순위 순)

주의: ⑧에서 같은-반복 내 재배열은 컴파일러가 이미 하고 있음이 확인됨. 남은 것 중
의미 있는 건 **반복 경계를 넘는** 파이프라이닝뿐 (아래 1) — 난이도 급상승.

1. **cross-iteration 파이프라이닝**: 다음 타일의 QK mma를 현재 타일 softmax/PV와 오버랩.
   S 레지스터 2세트 (+16 regs, full+L 95→111 → **4블록/SM으로 하락**, 65536/(128×111)=4.6).
   레지스터 ~10개 다이어트 선행 없이는 수학적으로 마이너스. 난이도 상.
2. smem swizzle로 PAD 제거 (18.4→16KB) + REG ≤85 → 6블록/SM. full+L(95)은 10개 초과.
   1번과 같은 레지스터 벽. 난이도 중.
3. ~~fp16 누산 QK~~ → **⑨로 완료** (ablation 전용 확정).
4. causal masking, D=128, backward를 mma 구조로.

수확체감 명확함: 남은 갭이 동일 런 ~2%라, 여기서부턴 공학적 이득보다 논문 서사 가치로
판단하는 게 맞음.

## 4.5 적대적 리뷰 결과 (5개 공격 영역, 별도 검증 에이전트로 교차확인)

동기화/cp.async 파이프라인/레이아웃/PTX 시맨틱/경계 처리: **클린 판정** (barrier 시퀀스,
버퍼 재사용 핸드오프, C→A 재사용, volatile 배치까지 전부 워크스루 검증됨).

컨펌돼서 수정 반영한 것:
1. **디바이스 가드 누락** (major) → `at::cuda::CUDAGuard` + K/V 디바이스 체크 추가.
   멀티GPU에서 cuda:1 텐서 + cuda:0 컨텍스트면 오답/크래시 가능했음.
2. **테스트 oracle 물렁함** (major) → atol 1e-2는 fp16 노이즈 플로어의 25~50배라
   2% systematic 버그(rescale 버그 시그니처)가 전체 통과 가능했음. half-cast 레퍼런스 +
   atol 2e-3 + N=4095 config로 수정. (검증 에이전트가 2% 버그 시뮬레이션으로 실증함)
3. minor: `-1e30f` 센티널 → `-INFINITY` (causal 확장 시 조용히 썩는 경로 제거),
   N=0/BH>65535 가드, K/V shape 검증 추가.

## 5. 절대 규칙 (변동 없음)

- `cuda/flash_attn_kernel.cu` 수정 금지. `flash_attn_wmma.cu` 보존.
- "SDPA보다 빠르다" 금지. 39.16x는 메모리 절약이지 speedup 아님.
- 시간 speedup은 naive 대비만.
- 커널 수정 시 `tests/test_mma_probe.py` → `tests/test_mma*.py` → 벤치 순서로 재검증.

## 6. 이 작업의 위치

- 작업 트리: `C:\Users\PC\flashattn-cuda-dev` (= `/mnt/c/Users/PC/flashattn-cuda-dev`), GitHub main(16a13f7)에서 클론.
- **커밋/푸시 안 했음.** 커밋 여부/메시지/분할은 유저 결정.
- 주의: 핸드오프에 있던 fa2 파일들(`flash_attn_fa2.cu` 등)은 이 머신 어디에도 없음
  (WSL rootfs + /mnt/c 검색 완료). fa2 수치는 핸드오프 문서 기준.
