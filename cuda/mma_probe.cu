// ============================================================
// mma_probe.cu — layout validation for mma.sync.m16n8k16 + ldmatrix
//
// Tests operand and accumulator register layouts on the GPU:
//   1. A operand:  ldmatrix.x4 on row-major Q-style storage
//   2. B operand (QK):  ldmatrix.x2 (no trans) on K-style storage [n][k]
//   3. B operand (PV):  ldmatrix.x2.trans on V-style storage [k][n]
//   4. C accumulator -> A operand reuse (the FlashAttention register trick)
//   5. C store layout: c0,c1 -> (row=l/4, col=2(l%4),+1); c2,c3 -> row+8
//
// Each probe compares the mma output with a reference matrix.
// ============================================================
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define SMEM_STRIDE 24   // deliberately padded (16+8) to prove stride independence

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t& r0, uint32_t& r1, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1) : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t& r0, uint32_t& r1, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1) : "r"(addr));
}

__device__ __forceinline__ void mma_m16n8k16(
    float& c0, float& c1, float& c2, float& c3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// fp16-accumulate variant: D/C are 2 b32 regs = 4 halves.
// Assumed layout: c0 = half2{(row l/4, col 2(l%4)), (row l/4, col+1)},
//                 c1 = half2 for row l/4+8, same columns.
__device__ __forceinline__ void mma_m16n8k16_f16acc(
    uint32_t& c0, uint32_t& c1,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(c0), "+r"(c1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Load A [16][16] half row-major into a-fragments via ldmatrix.x4.
// Expected mat order: {rows0-7/k0-7, rows8-15/k0-7, rows0-7/k8-15, rows8-15/k8-15}
__device__ __forceinline__ void load_a_frag(
    uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3,
    const half (*sA)[SMEM_STRIDE])
{
    int lane = threadIdx.x;
    int r = lane % 16;                 // lanes 0-15 -> rows 0-15, 16-31 -> rows 0-15 again
    int kofs = (lane < 16) ? 0 : 8;    // lanes 16-31 supply the k8-15 halves
    uint32_t addr = smem_u32(&sA[r][kofs]);
    ldmatrix_x4(a0, a1, a2, a3, addr);
}

// ------------------------------------------------------------
// Probe 1: S = A @ K^T   (A [16][16] row-major, K [8][16] row-major n-by-k)
// variant 0: B via ldmatrix.x2 no-trans on K rows (expected correct)
// variant 1: B via ldmatrix.x2.trans (control)
// ------------------------------------------------------------
__global__ void probe_qk_kernel(const half* A, const half* K, float* out, int variant)
{
    __shared__ half sA[16][SMEM_STRIDE];
    __shared__ half sK[8][SMEM_STRIDE];
    int lane = threadIdx.x;

    for (int i = lane; i < 16 * 16; i += 32) sA[i / 16][i % 16] = A[i];
    for (int i = lane; i < 8 * 16; i += 32)  sK[i / 16][i % 16] = K[i];
    __syncthreads();

    uint32_t a0, a1, a2, a3;
    load_a_frag(a0, a1, a2, a3, sA);

    // B addrs: mat0 = K rows 0-7 cols 0-7 (lanes 0-7), mat1 = K rows 0-7 cols 8-15 (lanes 8-15)
    int r = lane % 8;
    int kofs = (lane < 8) ? 0 : 8;
    uint32_t baddr = smem_u32(&sK[r][kofs]);

    uint32_t b0, b1;
    if (variant == 0) ldmatrix_x2(b0, b1, baddr);
    else              ldmatrix_x2_trans(b0, b1, baddr);

    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    mma_m16n8k16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);

    // C layout: c0,c1 -> (l/4, 2(l%4)+{0,1}); c2,c3 -> (l/4+8, same cols)
    int row = lane / 4, col = 2 * (lane % 4);
    out[row * 8 + col] = c0;
    out[row * 8 + col + 1] = c1;
    out[(row + 8) * 8 + col] = c2;
    out[(row + 8) * 8 + col + 1] = c3;
}

// ------------------------------------------------------------
// Probe 1b: S = A @ K^T with FP16 ACCUMULATION (f16.f16.f16.f16)
// Validates the f16 C-fragment layout (c0 = row l/4 half2, c1 = row l/4+8).
// ------------------------------------------------------------
__global__ void probe_qk_f16acc_kernel(const half* A, const half* K, float* out)
{
    __shared__ half sA[16][SMEM_STRIDE];
    __shared__ half sK[8][SMEM_STRIDE];
    int lane = threadIdx.x;

    for (int i = lane; i < 16 * 16; i += 32) sA[i / 16][i % 16] = A[i];
    for (int i = lane; i < 8 * 16; i += 32)  sK[i / 16][i % 16] = K[i];
    __syncthreads();

    uint32_t a0, a1, a2, a3;
    load_a_frag(a0, a1, a2, a3, sA);

    int r = lane % 8;
    int kofs = (lane < 8) ? 0 : 8;
    uint32_t baddr = smem_u32(&sK[r][kofs]);
    uint32_t b0, b1;
    ldmatrix_x2(b0, b1, baddr);

    uint32_t c0 = 0, c1 = 0;
    mma_m16n8k16_f16acc(c0, c1, a0, a1, a2, a3, b0, b1);

    half2 h01 = *reinterpret_cast<half2*>(&c0);
    half2 h23 = *reinterpret_cast<half2*>(&c1);
    float2 f01 = __half22float2(h01);
    float2 f23 = __half22float2(h23);

    int row = lane / 4, col = 2 * (lane % 4);
    out[row * 8 + col] = f01.x;
    out[row * 8 + col + 1] = f01.y;
    out[(row + 8) * 8 + col] = f23.x;
    out[(row + 8) * 8 + col + 1] = f23.y;
}

// ------------------------------------------------------------
// Probe 2: O = P @ V   (P [16][16] row-major m-by-k, V [16][8] row-major k-by-n)
// variant 0: B via ldmatrix.x2.trans on V rows (expected correct)
// variant 1: no-trans (control)
// ------------------------------------------------------------
__global__ void probe_pv_kernel(const half* P, const half* V, float* out, int variant)
{
    __shared__ half sP[16][SMEM_STRIDE];
    __shared__ half sV[16][SMEM_STRIDE];
    int lane = threadIdx.x;

    for (int i = lane; i < 16 * 16; i += 32) sP[i / 16][i % 16] = P[i];
    for (int i = lane; i < 16 * 8; i += 32)  sV[i / 8][i % 8] = V[i];
    __syncthreads();

    uint32_t a0, a1, a2, a3;
    load_a_frag(a0, a1, a2, a3, sP);

    // B addrs: mat0 = V rows 0-7 (k0-7, lanes 0-7), mat1 = V rows 8-15 (k8-15, lanes 8-15)
    int r = lane % 16;
    uint32_t baddr = smem_u32(&sV[r][0]);

    uint32_t b0, b1;
    if (variant == 0) ldmatrix_x2_trans(b0, b1, baddr);
    else              ldmatrix_x2(b0, b1, baddr);

    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    mma_m16n8k16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);

    int row = lane / 4, col = 2 * (lane % 4);
    out[row * 8 + col] = c0;
    out[row * 8 + col + 1] = c1;
    out[(row + 8) * 8 + col] = c2;
    out[(row + 8) * 8 + col + 1] = c3;
}

// ------------------------------------------------------------
// Probe 3: the C->A register reuse chain (FlashAttention trick)
//   S[16][16] = A @ Kfull^T   (two n8 C-tiles, Kfull [16][16] n-by-k)
//   Sh = half(S) converted IN REGISTERS with the assumed C->A mapping:
//     a0 = h2(c0,c1) of n-tile0, a1 = h2(c2,c3) of n-tile0,
//     a2 = h2(c0,c1) of n-tile1, a3 = h2(c2,c3) of n-tile1
//   O[16][8] = Sh @ V   (V [16][8] k-by-n, trans ldmatrix)
// Reference: ((A@Kfull.T).half()).float() @ V
// ------------------------------------------------------------
__global__ void probe_chain_kernel(const half* A, const half* Kfull, const half* V, float* out)
{
    __shared__ half sA[16][SMEM_STRIDE];
    __shared__ half sK[16][SMEM_STRIDE];
    __shared__ half sV[16][SMEM_STRIDE];
    int lane = threadIdx.x;

    for (int i = lane; i < 16 * 16; i += 32) {
        sA[i / 16][i % 16] = A[i];
        sK[i / 16][i % 16] = Kfull[i];
    }
    for (int i = lane; i < 16 * 8; i += 32) sV[i / 8][i % 8] = V[i];
    __syncthreads();

    uint32_t a0, a1, a2, a3;
    load_a_frag(a0, a1, a2, a3, sA);

    // ---- S = A @ Kfull^T, two n8 tiles ----
    float s[2][4];
    for (int t = 0; t < 2; t++) {
        int r = lane % 8;
        int kofs = (lane < 8) ? 0 : 8;
        uint32_t baddr = smem_u32(&sK[t * 8 + r][kofs]);   // n-tile t = K rows t*8 .. t*8+7
        uint32_t b0, b1;
        ldmatrix_x2(b0, b1, baddr);
        s[t][0] = s[t][1] = s[t][2] = s[t][3] = 0.f;
        mma_m16n8k16(s[t][0], s[t][1], s[t][2], s[t][3], a0, a1, a2, a3, b0, b1);
    }

    // ---- convert C tiles -> A fragments in registers ----
    uint32_t p0, p1, p2, p3;
    half2 h;
    h = __floats2half2_rn(s[0][0], s[0][1]); p0 = *reinterpret_cast<uint32_t*>(&h);
    h = __floats2half2_rn(s[0][2], s[0][3]); p1 = *reinterpret_cast<uint32_t*>(&h);
    h = __floats2half2_rn(s[1][0], s[1][1]); p2 = *reinterpret_cast<uint32_t*>(&h);
    h = __floats2half2_rn(s[1][2], s[1][3]); p3 = *reinterpret_cast<uint32_t*>(&h);

    // ---- O = Sh @ V ----
    int r = lane % 16;
    uint32_t baddr = smem_u32(&sV[r][0]);
    uint32_t b0, b1;
    ldmatrix_x2_trans(b0, b1, baddr);

    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    mma_m16n8k16(c0, c1, c2, c3, p0, p1, p2, p3, b0, b1);

    int row = lane / 4, col = 2 * (lane % 4);
    out[row * 8 + col] = c0;
    out[row * 8 + col + 1] = c1;
    out[(row + 8) * 8 + col] = c2;
    out[(row + 8) * 8 + col + 1] = c3;
}

// ============================================================
// Host wrappers
// ============================================================
static void check_inputs(const torch::Tensor& t, int rows, int cols, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.dtype() == torch::kHalf && t.is_contiguous(),
                name, " must be contiguous CUDA half");
    TORCH_CHECK(t.size(0) == rows && t.size(1) == cols, name, " shape mismatch");
}

torch::Tensor probe_qk(torch::Tensor A, torch::Tensor K, int64_t variant) {
    check_inputs(A, 16, 16, "A");
    check_inputs(K, 8, 16, "K");
    auto out = torch::zeros({16, 8}, A.options().dtype(torch::kFloat));
    probe_qk_kernel<<<1, 32>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
        out.data_ptr<float>(), (int)variant);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor probe_qk_f16acc(torch::Tensor A, torch::Tensor K) {
    check_inputs(A, 16, 16, "A");
    check_inputs(K, 8, 16, "K");
    auto out = torch::zeros({16, 8}, A.options().dtype(torch::kFloat));
    probe_qk_f16acc_kernel<<<1, 32>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
        out.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor probe_pv(torch::Tensor P, torch::Tensor V, int64_t variant) {
    check_inputs(P, 16, 16, "P");
    check_inputs(V, 16, 8, "V");
    auto out = torch::zeros({16, 8}, P.options().dtype(torch::kFloat));
    probe_pv_kernel<<<1, 32>>>(
        reinterpret_cast<const half*>(P.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
        out.data_ptr<float>(), (int)variant);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor probe_chain(torch::Tensor A, torch::Tensor Kfull, torch::Tensor V) {
    check_inputs(A, 16, 16, "A");
    check_inputs(Kfull, 16, 16, "Kfull");
    check_inputs(V, 16, 8, "V");
    auto out = torch::zeros({16, 8}, A.options().dtype(torch::kFloat));
    probe_chain_kernel<<<1, 32>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(Kfull.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
        out.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("probe_qk", &probe_qk, "S = A @ K^T via mma.m16n8k16");
    m.def("probe_qk_f16acc", &probe_qk_f16acc, "S = A @ K^T via mma.m16n8k16 f16 accumulate");
    m.def("probe_pv", &probe_pv, "O = P @ V via mma.m16n8k16");
    m.def("probe_chain", &probe_chain, "C->A reuse chain probe");
}
