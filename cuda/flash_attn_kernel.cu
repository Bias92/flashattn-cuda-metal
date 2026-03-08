/*
 * FlashAttention Forward Kernel (CUDA)
 * 
 * Algorithm: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact 
 *            Attention with IO-Awareness" (NeurIPS 2022)
 * 
 * Key ideas:
 *   1. Tiling: Split Q into row blocks (B_r), K/V into column blocks (B_c)
 *   2. Online softmax: Maintain running (max, sum) per row to avoid
 *      materializing the full N×N attention matrix
 *   3. IO-awareness: Minimize HBM reads by keeping tiles in SRAM (shared mem)
 * 
 * Memory: O(N) instead of O(N²) - never materialize S or P matrices
 * 
 * Thread model:
 *   - grid:  (ceil(N / B_r), B * H)
 *   - block: (B_r,) — one thread per Q row in the block
 *   - Each thread maintains its own running max, sum, and output accumulator
 *   - Threads collaborate on loading K/V tiles into shared memory
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))


/*
 * Forward kernel template.
 * BR, BC: tile sizes for Q rows and K/V columns
 * D: head dimension (compile-time constant for register allocation)
 */
template <int BR, int BC, int D>
__global__ void flash_attn_fwd_kernel(
    const float* __restrict__ Q,   // [B*H, N, D]
    const float* __restrict__ K,   // [B*H, N, D]
    const float* __restrict__ V,   // [B*H, N, D]
    float*       __restrict__ O,   // [B*H, N, D]
    float*       __restrict__ L,   // [B*H, N]  logsumexp (for backward)
    const int N,
    const float scale              // 1 / sqrt(D)
) {
    const int bh = blockIdx.y;           // batch-head index
    const int block_row = blockIdx.x;    // which Q row-block
    const int tid = threadIdx.x;         // [0, BR) — one thread per Q row

    const int row = block_row * BR + tid;

    // Pointers for this batch-head
    const float* q_bh = Q + bh * N * D;
    const float* k_bh = K + bh * N * D;
    const float* v_bh = V + bh * N * D;
    float*       o_bh = O + bh * N * D;
    float*       l_bh = L + bh * N;

    /* ── Shared memory for K and V tiles ────────────────────────── */
    __shared__ float sK[BC][D];
    __shared__ float sV[BC][D];

    /* ── Per-thread state: registers ────────────────────────────── */
    // Load this thread's Q row into registers (stays fixed across all K/V blocks)
    float q_reg[D];
    if (row < N) {
        for (int d = 0; d < D; d++) {
            q_reg[d] = q_bh[row * D + d];
        }
    }

    // Online softmax running state
    float m_i = -FLT_MAX;   // running row-wise max
    float l_i = 0.0f;       // running row-wise sum of exp(s - m)

    // Output accumulator (un-normalized)
    float acc[D];
    for (int d = 0; d < D; d++) {
        acc[d] = 0.0f;
    }

    /* ── Main loop: iterate over K/V column blocks ──────────────── */
    const int num_kv_blocks = CEIL_DIV(N, BC);

    for (int block_col = 0; block_col < num_kv_blocks; block_col++) {
        const int col_start = block_col * BC;

        /* ── Step 1: Collaboratively load K tile into shared memory ── */
        // BR threads loading BC * D elements
        // Each thread loads multiple elements in a strided pattern
        for (int idx = tid; idx < BC * D; idx += BR) {
            int kc = idx / D;   // row in K tile [0, BC)
            int kd = idx % D;   // col in K tile [0, D)
            int global_row = col_start + kc;
            if (global_row < N) {
                sK[kc][kd] = k_bh[global_row * D + kd];
            } else {
                sK[kc][kd] = 0.0f;
            }
        }

        /* ── Step 2: Collaboratively load V tile into shared memory ── */
        for (int idx = tid; idx < BC * D; idx += BR) {
            int vc = idx / D;
            int vd = idx % D;
            int global_row = col_start + vc;
            if (global_row < N) {
                sV[vc][vd] = v_bh[global_row * D + vd];
            } else {
                sV[vc][vd] = 0.0f;
            }
        }
        __syncthreads();

        if (row < N) {
            /* ── Step 3: Compute S_ij = q_i · K_j^T * scale ─────── */
            // s[c] = dot(q_reg, sK[c]) * scale   for c in [0, BC)
            float s[BC];
            float block_max = -FLT_MAX;

            for (int c = 0; c < BC; c++) {
                int global_col = col_start + c;
                if (global_col < N) {
                    float dot = 0.0f;
                    for (int d = 0; d < D; d++) {
                        dot += q_reg[d] * sK[c][d];
                    }
                    s[c] = dot * scale;
                    block_max = fmaxf(block_max, s[c]);
                } else {
                    s[c] = -FLT_MAX;  // mask out-of-bounds
                }
            }

            /* ── Step 4: Online softmax update ──────────────────── *
             *
             * Given previous state (m_i, l_i, acc[]),
             * and new block scores s[0..BC-1]:
             *
             *   m_new = max(m_i, max(s))
             *   alpha = exp(m_i - m_new)          // correction for old accumulators
             *   l_i   = l_i * alpha + sum_c exp(s[c] - m_new)
             *   acc[] = acc[] * alpha + sum_c exp(s[c] - m_new) * V[c]
             *   m_i   = m_new
             */
            float m_new = fmaxf(m_i, block_max);
            float alpha = expf(m_i - m_new);

            // Rescale previous accumulator
            l_i *= alpha;
            for (int d = 0; d < D; d++) {
                acc[d] *= alpha;
            }

            // Add contribution from this block
            for (int c = 0; c < BC; c++) {
                int global_col = col_start + c;
                if (global_col < N) {
                    float p = expf(s[c] - m_new);
                    l_i += p;
                    for (int d = 0; d < D; d++) {
                        acc[d] += p * sV[c][d];
                    }
                }
            }

            m_i = m_new;
        }
        __syncthreads();
    }

    /* ── Write output: O = acc / l_i ────────────────────────────── */
    if (row < N) {
        float inv_l = 1.0f / l_i;
        for (int d = 0; d < D; d++) {
            o_bh[row * D + d] = acc[d] * inv_l;
        }
        // Store logsumexp = m_i + log(l_i) for backward pass
        l_bh[row] = m_i + logf(l_i);
    }
}


/* ── Host-side launcher ─────────────────────────────────────────── */

torch::Tensor flash_attn_forward(
    torch::Tensor Q,   // [B, H, N, D]
    torch::Tensor K,
    torch::Tensor V
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(Q.dim() == 4, "Q must be [B, H, N, D]");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Only fp32 supported (for now)");

    const int B = Q.size(0);
    const int H = Q.size(1);
    const int N = Q.size(2);
    const int D = Q.size(3);

    TORCH_CHECK(D == 64, "Only D=64 supported (for now)");

    // Ensure contiguous layout
    Q = Q.contiguous();
    K = K.contiguous();
    V = V.contiguous();

    // Allocate output
    auto O = torch::zeros_like(Q);                          // [B, H, N, D]
    auto L = torch::zeros({B, H, N}, Q.options());          // [B, H, N]

    // Reshape to [B*H, N, D] for kernel
    auto Q_flat = Q.reshape({B * H, N, D});
    auto K_flat = K.reshape({B * H, N, D});
    auto V_flat = V.reshape({B * H, N, D});
    auto O_flat = O.reshape({B * H, N, D});
    auto L_flat = L.reshape({B * H, N});

    const float scale = 1.0f / sqrtf((float)D);

    // Tile sizes
    constexpr int BR = 32;
    constexpr int BC = 32;
    constexpr int D_CONST = 64;

    dim3 grid(CEIL_DIV(N, BR), B * H);
    dim3 block(BR);

    // Shared memory: sK[BC][D] + sV[BC][D]
    // = 2 * 32 * 64 * 4 = 16 KB
    flash_attn_fwd_kernel<BR, BC, D_CONST><<<grid, block>>>(
        Q_flat.data_ptr<float>(),
        K_flat.data_ptr<float>(),
        V_flat.data_ptr<float>(),
        O_flat.data_ptr<float>(),
        L_flat.data_ptr<float>(),
        N,
        scale
    );

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return O;  // L is stored internally for backward
}


/* ── Pybind11 module ────────────────────────────────────────────── */

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attn_forward, "FlashAttention forward (CUDA)");
}
