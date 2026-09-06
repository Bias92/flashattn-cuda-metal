// ============================================================
// experiments/cuda/double_buffer.cu -- mma + cp.async 2-stage K/V double buffering
//
// Prefetches the next K/V tile with cp.async.cg into an alternating shared buffer.
// Q staging reuses buffer 1 while the first K/V group is in flight into buffer 0.
//
// Shared memory: 2 stages x (K+V) x 32 x 72 halves = 18 KiB.
// PAD=8 keeps every 16B cp.async destination aligned (72*2 = 144 = 9*16).
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
template <int D, bool WRITE_L>
__global__ void __launch_bounds__(NWARPS * 32)
mma_db_fwd_kernel(
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
    constexpr int STAGE = 2 * BC * LDS;          // halves per stage (K tile + V tile)
    constexpr int CHUNKS = 2 * BC * D / 8;       // 16B chunks per stage (K+V)

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

    __shared__ __align__(16) half smem[2 * STAGE];

    // Issue cp.async for the K/V tile at kv into stage buffer s.
    // Layout inside a stage: K rows [0,BC) then V rows [0,BC), stride LDS.
    auto issue_kv = [&](int s, int kv) {
        half* base = smem + s * STAGE;
        #pragma unroll
        for (int c = tid; c < CHUNKS; c += NWARPS * 32) {
            int mat = c / (CHUNKS / 2);          // 0 = K, 1 = V
            int rem = c % (CHUNKS / 2);
            int r = rem / (D / 8);
            int cc = (rem % (D / 8)) * 8;        // column offset in halves
            int gr = kv + r;
            const half* src = (mat == 0 ? K_bh : V_bh) + (size_t)(gr < N ? gr : 0) * D + cc;
            uint32_t dst = smem_u32(base + (mat * BC + r) * LDS + cc);
            cp_async_16(dst, src, (gr < N) ? 16 : 0);
        }
        cp_async_commit();
    };

    // ---- Kick off K/V stage 0 while staging Q into stage-1 space ----
    issue_kv(0, 0);
    {
        half (*sQ)[LDS] = reinterpret_cast<half(*)[LDS]>(smem + STAGE);
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
    __syncthreads();   // Q fragments in registers; stage-1 buffer now free for K/V

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
            issue_kv((i + 1) & 1, kv + BC);
            cp_async_wait<1>();          // stage i complete; stage i+1 stays in flight
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();                 // stage i visible to all warps

        half (*sK)[LDS] = reinterpret_cast<half(*)[LDS]>(smem + (i & 1) * STAGE);
        half (*sV)[LDS] = reinterpret_cast<half(*)[LDS]>(smem + (i & 1) * STAGE + BC * LDS);

        // ---- S = Q K^T ----
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
                for (int j = 0; j < 4; j++) s[t][j] *= scale_log2;
            }
        }

        if (kv + BC > N) {
            #pragma unroll
            for (int t = 0; t < NTILES_S; t++) {
                int col0 = kv + t * 8 + 2 * (lane % 4);
                if (col0 >= N)     { s[t][0] = -INFINITY; s[t][2] = -INFINITY; }
                if (col0 + 1 >= N) { s[t][1] = -INFINITY; s[t][3] = -INFINITY; }
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

        uint32_t pf[KSLICES_PV][4];
        #pragma unroll
        for (int ks = 0; ks < KSLICES_PV; ks++) {
            pf[ks][0] = pack_half2(s[2 * ks][0],     s[2 * ks][1]);
            pf[ks][1] = pack_half2(s[2 * ks][2],     s[2 * ks][3]);
            pf[ks][2] = pack_half2(s[2 * ks + 1][0], s[2 * ks + 1][1]);
            pf[ks][3] = pack_half2(s[2 * ks + 1][2], s[2 * ks + 1][3]);
        }

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
        __syncthreads();   // all warps done with stage i before its buffer is re-issued
    }

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
    if (WRITE_L && lane % 4 == 0) {
        if (r_lo < N) L_bh[r_lo] = m_lo * LN2f + logf(l_lo);
        if (r_hi < N) L_bh[r_hi] = m_hi * LN2f + logf(l_hi);
    }
}

// ============================================================
// Host launchers
// ============================================================
static std::pair<torch::Tensor, torch::Tensor> mma_db_forward_impl(
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
    if (want_L) launch(mma_db_fwd_kernel<HD, true>);
    else        launch(mma_db_fwd_kernel<HD, false>);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {O_h.reshape({B, H, N, D}),
            want_L ? L.reshape({B, H, N}) : torch::Tensor()};
}

std::vector<torch::Tensor> mma_db_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    auto [O, L] = mma_db_forward_impl(Q, K, V, /*want_L=*/true);
    return {O, L};
}

torch::Tensor mma_db_forward_only(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    auto [O, L] = mma_db_forward_impl(Q, K, V, /*want_L=*/false);
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mma_db_forward, "MMA forward + cp.async double buffering: returns O half, L float");
    // WRITE_L=false skips logf, L stores and the host-side L allocation.
    m.def("forward_only", &mma_db_forward_only, "MMA forward (cp.async), true O-only (no L compute/alloc)");
}
