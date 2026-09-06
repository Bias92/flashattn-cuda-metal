// ============================================================
// experiments/cuda/interleaved.cu -- custom + softmax/PV interleave
//
// Identical math/layout/cp.async structure to attention_forward.cu.
// ONLY the issue order inside the KV iteration changes:
//   before: [exp+pack ALL P slices] -> [rescale O] -> [ALL PV mmas]
//   after:  [rescale O] ->
//           per PV k-slice ks: [exp+pack slice ks] -> [PV mmas slice ks]
//           -> [row-sum shuffles + l/m update]  (overlaps last HMMAs)
// so slice-1's exp2f/pack scalar work executes while slice-0's HMMAs are
// in flight, and the reduction shuffles overlap slice-1's HMMAs.
// Scalar op ORDER per element is unchanged -> bit-identical results.
// ============================================================
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

constexpr int HD = 64;
constexpr int BR = 64;
constexpr int BC = 32;
constexpr int PAD = 8;
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

__device__ __forceinline__ void cp_async_16(uint32_t saddr, const void* gptr, int src_size) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(saddr), "l"(gptr), "r"(src_size));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

template <int NGROUPS>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(NGROUPS));
}

// ============================================================
// Forward kernel
// ============================================================
template <int D, bool WRITE_L, bool FULL_TILES>
__global__ void __launch_bounds__(NWARPS * 32)
mma_db_full_intl_fwd_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    float* __restrict__ L,    // may be nullptr when WRITE_L == false
    int N)
{
    constexpr int LDS = D + PAD;
    constexpr int KSLICES = D / 16;
    constexpr int NTILES_S = BC / 8;
    constexpr int KSLICES_PV = BC / 16;
    constexpr int NTILES_O = D / 8;
    constexpr int STAGE = 2 * BC * LDS;
    constexpr uint32_t STAGE_BYTES = STAGE * sizeof(half);
    constexpr uint32_t ROW_BYTES = LDS * sizeof(half);

    const int tid  = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int bh = blockIdx.y;
    const int q_block = blockIdx.x * BR;

    const half* Q_bh = Q + (size_t)bh * N * D;
    const half* K_bh = K + (size_t)bh * N * D;
    const half* V_bh = V + (size_t)bh * N * D;
    half* O_bh = O + (size_t)bh * N * D;
    // L_bh is derived inside the WRITE_L epilogue only (no pointer
    // arithmetic on a null L when WRITE_L == false).

    __shared__ __align__(16) half smem[2 * STAGE];

    const uint32_t smem_base = smem_u32(smem);
    uint32_t cur_base  = smem_base;
    uint32_t next_base = smem_base + STAGE_BYTES;

    const int r0 = tid >> 3;
    const int r1 = r0 + 16;
    const int cc = (tid & 7) << 3;
    const uint32_t k0_off = (uint32_t)(r0 * LDS + cc) * sizeof(half);
    const uint32_t k1_off = (uint32_t)(r1 * LDS + cc) * sizeof(half);
    const uint32_t v0_off = (uint32_t)((BC + r0) * LDS + cc) * sizeof(half);
    const uint32_t v1_off = (uint32_t)((BC + r1) * LDS + cc) * sizeof(half);

    auto issue_kv_fast = [&](uint32_t sbase, int kv) {
        if constexpr (FULL_TILES) {
            // N % BC == 0 guarantees every issued row is in range.
            cp_async_16(sbase + k0_off, K_bh + (size_t)(kv + r0) * D + cc, 16);
            cp_async_16(sbase + k1_off, K_bh + (size_t)(kv + r1) * D + cc, 16);
            cp_async_16(sbase + v0_off, V_bh + (size_t)(kv + r0) * D + cc, 16);
            cp_async_16(sbase + v1_off, V_bh + (size_t)(kv + r1) * D + cc, 16);
        } else {
            int g0 = kv + r0, g1 = kv + r1;
            const half* k0 = K_bh + (size_t)(g0 < N ? g0 : 0) * D + cc;
            const half* k1 = K_bh + (size_t)(g1 < N ? g1 : 0) * D + cc;
            const half* v0 = V_bh + (size_t)(g0 < N ? g0 : 0) * D + cc;
            const half* v1 = V_bh + (size_t)(g1 < N ? g1 : 0) * D + cc;
            cp_async_16(sbase + k0_off, k0, (g0 < N) ? 16 : 0);
            cp_async_16(sbase + k1_off, k1, (g1 < N) ? 16 : 0);
            cp_async_16(sbase + v0_off, v0, (g0 < N) ? 16 : 0);
            cp_async_16(sbase + v1_off, v1, (g1 < N) ? 16 : 0);
        }
        cp_async_commit();
    };

    // ---- Kick off K/V stage 0 while staging Q into the other stage ----
    issue_kv_fast(cur_base, 0);
    {
        half (*sQ)[LDS] = reinterpret_cast<half(*)[LDS]>(smem + STAGE);
        const int total_h2 = BR * D / 2;
        for (int i = tid; i < total_h2; i += blockDim.x) {
            int flat = i * 2;
            int r = flat / D, c = flat % D;
            if constexpr (FULL_TILES) {
                // N % BR == 0 guarantees every Q row in this block is valid.
                *reinterpret_cast<half2*>(&sQ[r][c]) =
                    *reinterpret_cast<const half2*>(&Q_bh[(size_t)(q_block + r) * D + c]);
            } else {
                int gr = q_block + r;
                half2 val = (gr < N)
                    ? *reinterpret_cast<const half2*>(&Q_bh[(size_t)gr * D + c])
                    : __float2half2_rn(0.0f);
                *reinterpret_cast<half2*>(&sQ[r][c]) = val;
            }
        }
        __syncthreads();
    }

    uint32_t qf[KSLICES][4];
    {
        half (*sQ)[LDS] = reinterpret_cast<half(*)[LDS]>(smem + STAGE);
        int r = warp * 16 + (lane % 16);
        int kbase = (lane < 16) ? 0 : 8;
        #pragma unroll
        for (int ks = 0; ks < KSLICES; ks++) {
            uint32_t addr = smem_u32(&sQ[r][ks * 16 + kbase]);
            ldmatrix_x4(qf[ks][0], qf[ks][1], qf[ks][2], qf[ks][3], addr);
        }
    }
    __syncthreads();

    const uint32_t qk_lane_base =
        (uint32_t)((lane & 7) * LDS + ((lane < 8) ? 0 : 8)) * sizeof(half);
    const uint32_t pv_lane_base =
        (uint32_t)((BC + (lane & 15)) * LDS) * sizeof(half);

    float o_acc[NTILES_O][4];
    #pragma unroll
    for (int t = 0; t < NTILES_O; t++)
        o_acc[t][0] = o_acc[t][1] = o_acc[t][2] = o_acc[t][3] = 0.0f;

    float m_lo = -INFINITY, m_hi = -INFINITY;
    float l_lo = 0.0f,  l_hi = 0.0f;
    const float scale_log2 = rsqrtf((float)D) * LOG2Ef;

    const int nblocks = (N + BC - 1) / BC;

    for (int i = 0; i < nblocks; i++) {
        const int kv = i * BC;
        const bool has_next = (i + 1) < nblocks;

        if (has_next) {
            issue_kv_fast(next_base, kv + BC);
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        const uint32_t qk_base = cur_base + qk_lane_base;
        const uint32_t pv_base = cur_base + pv_lane_base;

        // ---- S = Q K^T ----
        float s[NTILES_S][4];
        #pragma unroll
        for (int t = 0; t < NTILES_S; t++) {
            s[t][0] = s[t][1] = s[t][2] = s[t][3] = 0.0f;
            #pragma unroll
            for (int ks = 0; ks < KSLICES; ks++) {
                uint32_t b0, b1;
                ldmatrix_x2(b0, b1,
                    qk_base + (uint32_t)(t * 8) * ROW_BYTES
                            + (uint32_t)(ks * 16) * sizeof(half));
                mma_m16n8k16(s[t][0], s[t][1], s[t][2], s[t][3],
                             qf[ks][0], qf[ks][1], qf[ks][2], qf[ks][3], b0, b1);
            }
            #pragma unroll
            for (int j = 0; j < 4; j++) s[t][j] *= scale_log2;
        }

        if constexpr (!FULL_TILES) {
            if (kv + BC > N) {
                #pragma unroll
                for (int t = 0; t < NTILES_S; t++) {
                    int col0 = kv + t * 8 + 2 * (lane % 4);
                    if (col0 >= N)     { s[t][0] = -INFINITY; s[t][2] = -INFINITY; }
                    if (col0 + 1 >= N) { s[t][1] = -INFINITY; s[t][3] = -INFINITY; }
                }
            }
        }

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

        // ---- Interleaved softmax/PV ----
        // Rescale O first: PV mmas accumulate into it.
        #pragma unroll
        for (int t = 0; t < NTILES_O; t++) {
            o_acc[t][0] *= alpha_lo;
            o_acc[t][1] *= alpha_lo;
            o_acc[t][2] *= alpha_hi;
            o_acc[t][3] *= alpha_hi;
        }

        // Per PV k-slice: exp+pack that slice, then issue its 8 HMMAs.
        // Slice ks+1's exp2f/pack runs while slice ks's HMMAs are in flight.
        float rs_lo = 0.0f, rs_hi = 0.0f;
        #pragma unroll
        for (int ks = 0; ks < KSLICES_PV; ks++) {
            uint32_t pf[4];
            #pragma unroll
            for (int tt = 2 * ks; tt <= 2 * ks + 1; tt++) {
                s[tt][0] = exp2f(s[tt][0] - mn_lo);
                s[tt][1] = exp2f(s[tt][1] - mn_lo);
                s[tt][2] = exp2f(s[tt][2] - mn_hi);
                s[tt][3] = exp2f(s[tt][3] - mn_hi);
                rs_lo += s[tt][0] + s[tt][1];
                rs_hi += s[tt][2] + s[tt][3];
            }
            pf[0] = pack_half2(s[2 * ks][0],     s[2 * ks][1]);
            pf[1] = pack_half2(s[2 * ks][2],     s[2 * ks][3]);
            pf[2] = pack_half2(s[2 * ks + 1][0], s[2 * ks + 1][1]);
            pf[3] = pack_half2(s[2 * ks + 1][2], s[2 * ks + 1][3]);

            #pragma unroll
            for (int t = 0; t < NTILES_O; t++) {
                uint32_t b0, b1;
                ldmatrix_x2_trans(b0, b1,
                    pv_base + (uint32_t)(ks * 16) * ROW_BYTES
                            + (uint32_t)(t * 8) * sizeof(half));
                mma_m16n8k16(o_acc[t][0], o_acc[t][1], o_acc[t][2], o_acc[t][3],
                             pf[0], pf[1], pf[2], pf[3], b0, b1);
            }
        }

        // Deferred reductions/state update: overlaps the last HMMA slice.
        rs_lo += __shfl_xor_sync(0xffffffff, rs_lo, 1);
        rs_lo += __shfl_xor_sync(0xffffffff, rs_lo, 2);
        rs_hi += __shfl_xor_sync(0xffffffff, rs_hi, 1);
        rs_hi += __shfl_xor_sync(0xffffffff, rs_hi, 2);
        l_lo = l_lo * alpha_lo + rs_lo;
        l_hi = l_hi * alpha_hi + rs_hi;
        m_lo = mn_lo;
        m_hi = mn_hi;
        __syncthreads();

        uint32_t tmp = cur_base; cur_base = next_base; next_base = tmp;
    }

    const float inv_lo = 1.0f / l_lo;
    const float inv_hi = 1.0f / l_hi;
    const int r_lo = q_block + warp * 16 + lane / 4;
    const int r_hi = r_lo + 8;
    const int cbase = 2 * (lane % 4);

    #pragma unroll
    for (int t = 0; t < NTILES_O; t++) {
        int col = t * 8 + cbase;
        if (FULL_TILES || r_lo < N) {
            half2 v = __floats2half2_rn(o_acc[t][0] * inv_lo, o_acc[t][1] * inv_lo);
            *reinterpret_cast<half2*>(&O_bh[(size_t)r_lo * D + col]) = v;
        }
        if (FULL_TILES || r_hi < N) {
            half2 v = __floats2half2_rn(o_acc[t][2] * inv_hi, o_acc[t][3] * inv_hi);
            *reinterpret_cast<half2*>(&O_bh[(size_t)r_hi * D + col]) = v;
        }
    }
    if constexpr (WRITE_L) {
        if (lane % 4 == 0) {
            float* L_bh = L + (size_t)bh * N;
            if (FULL_TILES || r_lo < N) L_bh[r_lo] = m_lo * LN2f + logf(l_lo);
            if (FULL_TILES || r_hi < N) L_bh[r_hi] = m_hi * LN2f + logf(l_hi);
        }
    }
}

// ============================================================
// Host launchers
// ============================================================
static std::pair<torch::Tensor, torch::Tensor> mma_db_full_intl_forward_impl(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, bool want_L)
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
    torch::Tensor L;
    float* L_ptr = nullptr;
    if (want_L) {
        L = torch::empty({BH, N}, Q.options().dtype(torch::kFloat));
        L_ptr = L.data_ptr<float>();
    }

    dim3 grid((N + BR - 1) / BR, BH);
    dim3 block(NWARPS * 32);
    auto stream = at::cuda::getCurrentCUDAStream();

    auto launch = [&](auto kernel) {
        kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const half*>(Q_h.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(K_h.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(V_h.data_ptr<at::Half>()),
            reinterpret_cast<half*>(O_h.data_ptr<at::Half>()),
            L_ptr,
            N);
    };
    const bool full_tiles = (N % BR == 0) && (N % BC == 0);
    if (want_L) {
        if (full_tiles) launch(mma_db_full_intl_fwd_kernel<HD, true, true>);
        else            launch(mma_db_full_intl_fwd_kernel<HD, true, false>);
    } else {
        if (full_tiles) launch(mma_db_full_intl_fwd_kernel<HD, false, true>);
        else            launch(mma_db_full_intl_fwd_kernel<HD, false, false>);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {O_h.reshape({B, H, N, D}),
            want_L ? L.reshape({B, H, N}) : torch::Tensor()};
}

std::vector<torch::Tensor> mma_db_full_intl_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    auto [O, L] = mma_db_full_intl_forward_impl(Q, K, V, /*want_L=*/true);
    return {O, L};
}

torch::Tensor mma_db_full_intl_forward_only(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    auto [O, L] = mma_db_full_intl_forward_impl(Q, K, V, /*want_L=*/false);
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mma_db_full_intl_forward, "Custom CUDA + softmax/PV interleave: returns O half, L float");
    m.def("forward_only", &mma_db_full_intl_forward_only, "Custom CUDA + softmax/PV interleave, true O-only");
}
