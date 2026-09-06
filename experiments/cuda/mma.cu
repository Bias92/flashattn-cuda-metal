// ============================================================
// experiments/cuda/mma.cu -- attention forward using mma.sync
//
// Layout probes: experiments/tests/test_layout_probe.py.
//
// Implementation:
//   - mma.sync.m16n8k16 for BOTH QK^T and PV (no WMMA API)
//   - S, P, O live in registers; softmax entirely in registers
//     (quad-lane shuffle reductions). No sS/sP shared round-trip.
//   - P feeds PV through C->A fragment reuse.
//   - Block = 4 warps, BR=64 Q rows (16 per warp), BC=32 KV rows.
//   - Shared memory: one 9 KiB buffer, used once to stage Q for
//     ldmatrix, then reused as the K/V tile buffer every iteration.
//   - exp2f-based online softmax (scale folded into log2 domain).
//   - 2 __syncthreads() per KV block.
//
// Scope: forward only, non-causal, D=64, half in/out, fp32 L.
// ============================================================
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

constexpr int HD = 64;
constexpr int BR = 64;      // Q rows per block (4 warps x m16)
constexpr int BC = 32;      // KV rows per tile
constexpr int PAD = 8;      // shared row padding (halves)
constexpr int NWARPS = BR / 16;

#define LN2f 0.69314718056f
#define LOG2Ef 1.44269504089f

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

__device__ __forceinline__ uint32_t pack_half2(float x, float y) {
    half2 h = __floats2half2_rn(x, y);
    return *reinterpret_cast<uint32_t*>(&h);
}

// ============================================================
// Forward kernel
// ============================================================
template <int D>
__global__ void __launch_bounds__(BR / 16 * 32)
mma_fwd_kernel(
    const half* __restrict__ Q,   // [BH, N, D]
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    float* __restrict__ L,        // [BH, N]  (natural-log logsumexp)
    int N)
{
    constexpr int LDS = D + PAD;          // shared row stride in halves
    constexpr int KSLICES = D / 16;       // 4 k-slices for QK
    constexpr int NTILES_S = BC / 8;      // 4 n8 tiles of S per warp
    constexpr int KSLICES_PV = BC / 16;   // 2 k-slices for PV
    constexpr int NTILES_O = D / 8;       // 8 n8 tiles of O per warp

    const int tid  = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int bh = blockIdx.y;
    const int q_block = blockIdx.x * BR;

    const half* Q_bh = Q + (size_t)bh * N * D;
    const half* K_bh = K + (size_t)bh * N * D;
    const half* V_bh = V + (size_t)bh * N * D;
    half* O_bh = O + (size_t)bh * N * D;
    float* L_bh = L + (size_t)bh * N;

    // One buffer, two lives: Q staging (BR x LDS), then K/V tiles (2 x BC x LDS).
    // BR*LDS == 2*BC*LDS by construction (BR == 2*BC).
    __shared__ __align__(16) half smem[BR * LDS];
    half (*sQ)[LDS] = reinterpret_cast<half(*)[LDS]>(smem);
    half (*sK)[LDS] = reinterpret_cast<half(*)[LDS]>(smem);
    half (*sV)[LDS] = reinterpret_cast<half(*)[LDS]>(smem + BC * LDS);

    // ---- Stage Q tile, pull into A-fragments, then release the buffer ----
    {
        const int total_h2 = BR * D / 2;
        for (int i = tid; i < total_h2; i += blockDim.x) {
            int flat = i * 2;
            int r = flat / D, c = flat % D;
            int gr = q_block + r;
            half2 val = (gr < N)
                ? *reinterpret_cast<const half2*>(&Q_bh[(size_t)gr * D + c])
                : __float2half2_rn(0.0f);
            *reinterpret_cast<half2*>(&sQ[r][c]) = val;
        }
    }
    __syncthreads();

    uint32_t qf[KSLICES][4];
    {
        int r = warp * 16 + (lane % 16);
        int kbase = (lane < 16) ? 0 : 8;
        #pragma unroll
        for (int ks = 0; ks < KSLICES; ks++) {
            uint32_t addr = smem_u32(&sQ[r][ks * 16 + kbase]);
            ldmatrix_x4(qf[ks][0], qf[ks][1], qf[ks][2], qf[ks][3], addr);
        }
    }
    __syncthreads();   // Q fragments live in registers; smem now belongs to K/V

    // ---- Online softmax state (log2 domain) + O accumulators ----
    float o_acc[NTILES_O][4];
    #pragma unroll
    for (int t = 0; t < NTILES_O; t++)
        o_acc[t][0] = o_acc[t][1] = o_acc[t][2] = o_acc[t][3] = 0.0f;

    float m_lo = -INFINITY, m_hi = -INFINITY;   // running max (scaled, log2 units)
    float l_lo = 0.0f,  l_hi = 0.0f;      // running sum

    const float scale_log2 = rsqrtf((float)D) * LOG2Ef;

    for (int kv = 0; kv < N; kv += BC) {
        // ---- Load K/V tile (zero-padded tail rows) ----
        {
            const int total_h2 = BC * D / 2;
            for (int i = tid; i < total_h2; i += blockDim.x) {
                int flat = i * 2;
                int r = flat / D, c = flat % D;
                int gr = kv + r;
                half2 kval, vval;
                if (gr < N) {
                    kval = *reinterpret_cast<const half2*>(&K_bh[(size_t)gr * D + c]);
                    vval = *reinterpret_cast<const half2*>(&V_bh[(size_t)gr * D + c]);
                } else {
                    kval = __float2half2_rn(0.0f);
                    vval = __float2half2_rn(0.0f);
                }
                *reinterpret_cast<half2*>(&sK[r][c]) = kval;
                *reinterpret_cast<half2*>(&sV[r][c]) = vval;
            }
        }
        __syncthreads();

        // ---- S = Q K^T (scaled into log2 domain) ----
        float s[NTILES_S][4];
        {
            int br = lane % 8;
            int bk = (lane < 8) ? 0 : 8;
            #pragma unroll
            for (int t = 0; t < NTILES_S; t++) {
                s[t][0] = s[t][1] = s[t][2] = s[t][3] = 0.0f;
                #pragma unroll
                for (int ks = 0; ks < KSLICES; ks++) {
                    uint32_t b0, b1;
                    ldmatrix_x2(b0, b1, smem_u32(&sK[t * 8 + br][ks * 16 + bk]));
                    mma_m16n8k16(s[t][0], s[t][1], s[t][2], s[t][3],
                                 qf[ks][0], qf[ks][1], qf[ks][2], qf[ks][3], b0, b1);
                }
                #pragma unroll
                for (int i = 0; i < 4; i++) s[t][i] *= scale_log2;
            }
        }

        // ---- Mask out-of-range columns in the tail tile ----
        if (kv + BC > N) {
            #pragma unroll
            for (int t = 0; t < NTILES_S; t++) {
                int col0 = kv + t * 8 + 2 * (lane % 4);
                if (col0 >= N)     { s[t][0] = -INFINITY; s[t][2] = -INFINITY; }
                if (col0 + 1 >= N) { s[t][1] = -INFINITY; s[t][3] = -INFINITY; }
            }
        }

        // ---- Row max (local -> quad shuffle) ----
        float bm_lo = -INFINITY, bm_hi = -INFINITY;
        #pragma unroll
        for (int t = 0; t < NTILES_S; t++) {
            bm_lo = fmaxf(bm_lo, fmaxf(s[t][0], s[t][1]));
            bm_hi = fmaxf(bm_hi, fmaxf(s[t][2], s[t][3]));
        }
        bm_lo = fmaxf(bm_lo, __shfl_xor_sync(0xffffffff, bm_lo, 1));
        bm_lo = fmaxf(bm_lo, __shfl_xor_sync(0xffffffff, bm_lo, 2));
        bm_hi = fmaxf(bm_hi, __shfl_xor_sync(0xffffffff, bm_hi, 1));
        bm_hi = fmaxf(bm_hi, __shfl_xor_sync(0xffffffff, bm_hi, 2));

        float mn_lo = fmaxf(m_lo, bm_lo);
        float mn_hi = fmaxf(m_hi, bm_hi);
        float alpha_lo = exp2f(m_lo - mn_lo);
        float alpha_hi = exp2f(m_hi - mn_hi);

        // ---- P = exp2(S - m), row sums ----
        float rs_lo = 0.0f, rs_hi = 0.0f;
        #pragma unroll
        for (int t = 0; t < NTILES_S; t++) {
            s[t][0] = exp2f(s[t][0] - mn_lo);
            s[t][1] = exp2f(s[t][1] - mn_lo);
            s[t][2] = exp2f(s[t][2] - mn_hi);
            s[t][3] = exp2f(s[t][3] - mn_hi);
            rs_lo += s[t][0] + s[t][1];
            rs_hi += s[t][2] + s[t][3];
        }
        rs_lo += __shfl_xor_sync(0xffffffff, rs_lo, 1);
        rs_lo += __shfl_xor_sync(0xffffffff, rs_lo, 2);
        rs_hi += __shfl_xor_sync(0xffffffff, rs_hi, 1);
        rs_hi += __shfl_xor_sync(0xffffffff, rs_hi, 2);

        l_lo = l_lo * alpha_lo + rs_lo;
        l_hi = l_hi * alpha_hi + rs_hi;
        m_lo = mn_lo;
        m_hi = mn_hi;

        // ---- Pack P into A-fragments (C->A register reuse) ----
        uint32_t pf[KSLICES_PV][4];
        #pragma unroll
        for (int ks = 0; ks < KSLICES_PV; ks++) {
            pf[ks][0] = pack_half2(s[2 * ks][0],     s[2 * ks][1]);
            pf[ks][1] = pack_half2(s[2 * ks][2],     s[2 * ks][3]);
            pf[ks][2] = pack_half2(s[2 * ks + 1][0], s[2 * ks + 1][1]);
            pf[ks][3] = pack_half2(s[2 * ks + 1][2], s[2 * ks + 1][3]);
        }

        // ---- Rescale O, then O += P V ----
        #pragma unroll
        for (int t = 0; t < NTILES_O; t++) {
            o_acc[t][0] *= alpha_lo;
            o_acc[t][1] *= alpha_lo;
            o_acc[t][2] *= alpha_hi;
            o_acc[t][3] *= alpha_hi;
        }
        {
            int vr = lane % 16;
            #pragma unroll
            for (int t = 0; t < NTILES_O; t++) {
                #pragma unroll
                for (int ks = 0; ks < KSLICES_PV; ks++) {
                    uint32_t b0, b1;
                    ldmatrix_x2_trans(b0, b1, smem_u32(&sV[ks * 16 + vr][t * 8]));
                    mma_m16n8k16(o_acc[t][0], o_acc[t][1], o_acc[t][2], o_acc[t][3],
                                 pf[ks][0], pf[ks][1], pf[ks][2], pf[ks][3], b0, b1);
                }
            }
        }
        __syncthreads();   // protect sK/sV before next tile load
    }

    // ---- Epilogue: normalize, write O (half2) and L ----
    const float inv_lo = 1.0f / l_lo;
    const float inv_hi = 1.0f / l_hi;
    const int r_lo = q_block + warp * 16 + lane / 4;
    const int r_hi = r_lo + 8;
    const int cbase = 2 * (lane % 4);

    #pragma unroll
    for (int t = 0; t < NTILES_O; t++) {
        int col = t * 8 + cbase;
        if (r_lo < N) {
            half2 v = __floats2half2_rn(o_acc[t][0] * inv_lo, o_acc[t][1] * inv_lo);
            *reinterpret_cast<half2*>(&O_bh[(size_t)r_lo * D + col]) = v;
        }
        if (r_hi < N) {
            half2 v = __floats2half2_rn(o_acc[t][2] * inv_hi, o_acc[t][3] * inv_hi);
            *reinterpret_cast<half2*>(&O_bh[(size_t)r_hi * D + col]) = v;
        }
    }
    if (lane % 4 == 0) {
        if (r_lo < N) L_bh[r_lo] = m_lo * LN2f + logf(l_lo);
        if (r_hi < N) L_bh[r_hi] = m_hi * LN2f + logf(l_hi);
    }
}

// ============================================================
// Host launchers
// ============================================================
static std::pair<torch::Tensor, torch::Tensor> mma_forward_impl(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V)
{
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Q/K/V must be CUDA tensors");
    TORCH_CHECK(Q.dim() == 4, "Q must be 4D [B, H, N, D]");

    int B = Q.size(0), H = Q.size(1), N = Q.size(2), D = Q.size(3);
    TORCH_CHECK(D == HD, "Head dimension must be ", HD);
    TORCH_CHECK(N > 0, "N must be > 0");
    TORCH_CHECK(K.sizes() == Q.sizes() && V.sizes() == Q.sizes(),
                "K and V must have the same shape as Q (self-attention only)");
    TORCH_CHECK(K.device() == Q.device() && V.device() == Q.device(),
                "Q/K/V must be on the same device");
    int64_t BH = (int64_t)B * H;
    TORCH_CHECK(BH <= 65535, "B*H must be <= 65535 (gridDim.y limit)");

    const at::cuda::CUDAGuard guard(Q.device());
    auto Q_h = Q.to(torch::kHalf).reshape({BH, N, D}).contiguous();
    auto K_h = K.to(torch::kHalf).reshape({BH, N, D}).contiguous();
    auto V_h = V.to(torch::kHalf).reshape({BH, N, D}).contiguous();

    auto O_h = torch::empty({BH, N, D}, Q_h.options());
    auto L = torch::empty({BH, N}, Q.options().dtype(torch::kFloat));

    dim3 grid((N + BR - 1) / BR, BH);
    dim3 block(NWARPS * 32);
    auto stream = at::cuda::getCurrentCUDAStream();

    mma_fwd_kernel<HD><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(Q_h.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(K_h.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(V_h.data_ptr<at::Half>()),
        reinterpret_cast<half*>(O_h.data_ptr<at::Half>()),
        L.data_ptr<float>(),
        N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {O_h.reshape({B, H, N, D}), L.reshape({B, H, N})};
}

std::vector<torch::Tensor> mma_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    auto [O, L] = mma_forward_impl(Q, K, V);
    return {O, L};
}

torch::Tensor mma_forward_only(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    auto [O, L] = mma_forward_impl(Q, K, V);
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mma_forward, "MMA forward (mma.sync, register-resident softmax): returns O half, L float");
    // forward_only returns O; L is still allocated, computed, and written.
    m.def("forward_only", &mma_forward_only, "MMA forward returning O only (L still computed internally)");
}
