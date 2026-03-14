#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

// ============================================================
// Constants
// ============================================================
constexpr int BR = 32;   // Q row block size
constexpr int BC = 32;   // K/V column block size
constexpr int HD = 64;   // head dimension (compile-time)

// ============================================================
// Forward Kernel (Algorithm 1)
// ============================================================
template <int B_r, int B_c, int D>
__global__ void flash_attn_fwd_kernel(
    const float* __restrict__ Q,   // [BH, N, D]
    const float* __restrict__ K,   // [BH, N, D]
    const float* __restrict__ V,   // [BH, N, D]
    float* __restrict__ O,         // [BH, N, D]
    float* __restrict__ L,         // [BH, N]
    int N)
{
    int tid = threadIdx.x;                        // row within Q block
    int block_row = blockIdx.x;                   // which Q block
    int bh = blockIdx.y;                          // batch * head index
    int row = block_row * B_r + tid;              // global Q row

    // Pointers for this batch-head
    const float* Q_bh = Q + bh * N * D;
    const float* K_bh = K + bh * N * D;
    const float* V_bh = V + bh * N * D;
    float* O_bh = O + bh * N * D;
    float* L_bh = L + bh * N;

    // NOTE: Do NOT early-return here. All threads must participate in
    // shared memory loads and __syncthreads(), even if row >= N.
    bool valid = (row < N);

    // Load Q row into registers (stays fixed across all K/V blocks)
    float q_reg[D];
    if (valid) {
        for (int d = 0; d < D; d++)
            q_reg[d] = Q_bh[row * D + d];
    }

    float scale = rsqrtf((float)D);

    // Running online softmax state
    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float acc[D];
    for (int d = 0; d < D; d++)
        acc[d] = 0.0f;

    // Shared memory for K and V tiles
    __shared__ float sK[B_c][D];
    __shared__ float sV[B_c][D];

    int num_kv_blocks = (N + B_c - 1) / B_c;

    for (int j = 0; j < num_kv_blocks; j++) {
        // ALL threads participate in collaborative load
        int kv_start = j * B_c;
        for (int c = tid; c < B_c; c += B_r) {
            int global_c = kv_start + c;
            for (int d = 0; d < D; d++) {
                sK[c][d] = (global_c < N) ? K_bh[global_c * D + d] : 0.0f;
                sV[c][d] = (global_c < N) ? V_bh[global_c * D + d] : 0.0f;
            }
        }
        __syncthreads();

        if (valid) {
            // Compute S[c] = dot(q_reg, sK[c]) * scale
            float s[B_c];
            for (int c = 0; c < B_c; c++) {
                float dot = 0.0f;
                for (int d = 0; d < D; d++)
                    dot += q_reg[d] * sK[c][d];
                s[c] = dot * scale;
            }

            // Find block max
            float block_max = -FLT_MAX;
            for (int c = 0; c < B_c; c++) {
                int global_c = kv_start + c;
                if (global_c < N && s[c] > block_max)
                    block_max = s[c];
            }

            // Online softmax update
            float m_new = fmaxf(m_i, block_max);
            float alpha = expf(m_i - m_new);

            l_i = l_i * alpha;
            for (int d = 0; d < D; d++)
                acc[d] = acc[d] * alpha;

            for (int c = 0; c < B_c; c++) {
                int global_c = kv_start + c;
                float p = (global_c < N) ? expf(s[c] - m_new) : 0.0f;
                l_i += p;
                for (int d = 0; d < D; d++)
                    acc[d] += p * sV[c][d];
            }

            m_i = m_new;
        }
        __syncthreads();
    }

    // Normalize and write output (valid threads only)
    if (valid) {
        float inv_l = 1.0f / l_i;
        for (int d = 0; d < D; d++)
            O_bh[row * D + d] = acc[d] * inv_l;
        L_bh[row] = m_i + logf(l_i);
    }
}

// ============================================================
// Forward Host Launcher
// ============================================================
std::vector<torch::Tensor> flash_attn_forward(
    torch::Tensor Q,   // [B, H, N, D]
    torch::Tensor K,
    torch::Tensor V)
{
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(Q.dim() == 4, "Q must be 4D [B, H, N, D]");

    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);
    int D = Q.size(3);
    TORCH_CHECK(D == HD, "Head dimension must be " + std::to_string(HD));

    int BH = B * H;

    // Reshape to [BH, N, D]
    auto Q_flat = Q.reshape({BH, N, D}).contiguous();
    auto K_flat = K.reshape({BH, N, D}).contiguous();
    auto V_flat = V.reshape({BH, N, D}).contiguous();

    auto O_flat = torch::zeros_like(Q_flat);
    auto L = torch::zeros({BH, N}, Q.options());

    int num_q_blocks = (N + BR - 1) / BR;
    dim3 grid(num_q_blocks, BH);
    dim3 block(BR);

    flash_attn_fwd_kernel<BR, BC, HD><<<grid, block>>>(
        Q_flat.data_ptr<float>(),
        K_flat.data_ptr<float>(),
        V_flat.data_ptr<float>(),
        O_flat.data_ptr<float>(),
        L.data_ptr<float>(),
        N);

    auto O_out = O_flat.reshape({B, H, N, D});
    auto L_out = L.reshape({B, H, N});

    return {O_out, L_out};
}

// ============================================================
// Backward Kernel 1: Precompute D_i = rowsum(dO * O)
// ============================================================
__global__ void flash_attn_precompute_D_kernel(
    const float* __restrict__ O,    // [BH, N, D]
    const float* __restrict__ dO,   // [BH, N, D]
    float* __restrict__ Di,         // [BH, N]
    int N, int D_dim)
{
    int bh = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    const float* O_row = O + bh * N * D_dim + row * D_dim;
    const float* dO_row = dO + bh * N * D_dim + row * D_dim;

    float sum = 0.0f;
    for (int d = 0; d < D_dim; d++)
        sum += O_row[d] * dO_row[d];

    Di[bh * N + row] = sum;
}

// ============================================================
// Backward Kernel 2: Compute dQ
//   Grid: (ceil(N/BR), BH), Block: (BR,)
//   Same structure as forward: one thread per Q row,
//   iterate over all K/V blocks.
// ============================================================
template <int B_r, int B_c, int D>
__global__ void flash_attn_bwd_dq_kernel(
    const float* __restrict__ Q,    // [BH, N, D]
    const float* __restrict__ K,    // [BH, N, D]
    const float* __restrict__ V,    // [BH, N, D]
    const float* __restrict__ dO,   // [BH, N, D]
    const float* __restrict__ L,    // [BH, N]
    const float* __restrict__ Di,   // [BH, N]
    float* __restrict__ dQ,         // [BH, N, D]
    int N)
{
    int tid = threadIdx.x;
    int block_row = blockIdx.x;
    int bh = blockIdx.y;
    int row = block_row * B_r + tid;

    const float* Q_bh  = Q  + bh * N * D;
    const float* K_bh  = K  + bh * N * D;
    const float* V_bh  = V  + bh * N * D;
    const float* dO_bh = dO + bh * N * D;
    const float* L_bh  = L  + bh * N;
    const float* Di_bh = Di + bh * N;
    float* dQ_bh = dQ + bh * N * D;

    bool valid = (row < N);

    float scale = rsqrtf((float)D);

    // Load Q row and dO row into registers
    float q_reg[D], do_reg[D];
    float l_i = 0.0f, d_i = 0.0f;
    if (valid) {
        for (int d = 0; d < D; d++) {
            q_reg[d]  = Q_bh[row * D + d];
            do_reg[d] = dO_bh[row * D + d];
        }
        l_i = L_bh[row];
        d_i = Di_bh[row];
    }

    // Accumulator for dQ
    float dq_acc[D];
    for (int d = 0; d < D; d++)
        dq_acc[d] = 0.0f;

    __shared__ float sK[B_c][D];
    __shared__ float sV[B_c][D];

    int num_kv_blocks = (N + B_c - 1) / B_c;

    for (int j = 0; j < num_kv_blocks; j++) {
        int kv_start = j * B_c;

        // ALL threads participate in collaborative load
        for (int c = tid; c < B_c; c += B_r) {
            int global_c = kv_start + c;
            for (int d = 0; d < D; d++) {
                sK[c][d] = (global_c < N) ? K_bh[global_c * D + d] : 0.0f;
                sV[c][d] = (global_c < N) ? V_bh[global_c * D + d] : 0.0f;
            }
        }
        __syncthreads();

        if (valid) {
            for (int c = 0; c < B_c; c++) {
                int global_c = kv_start + c;
                if (global_c >= N) break;

                float s_c = 0.0f;
                for (int d = 0; d < D; d++)
                    s_c += q_reg[d] * sK[c][d];
                s_c *= scale;

                float p_c = expf(s_c - l_i);

                float dp_c = 0.0f;
                for (int d = 0; d < D; d++)
                    dp_c += do_reg[d] * sV[c][d];

                float ds_c = p_c * (dp_c - d_i);

                for (int d = 0; d < D; d++)
                    dq_acc[d] += ds_c * sK[c][d];
            }
        }
        __syncthreads();
    }

    // Write dQ = scale * dq_acc
    if (valid) {
        for (int d = 0; d < D; d++)
            dQ_bh[row * D + d] = scale * dq_acc[d];
    }
}

// ============================================================
// Backward Kernel 3: Compute dK, dV
//   Grid: (ceil(N/BC), BH), Block: (BC,)
//   One thread per K/V row, iterate over all Q blocks.
// ============================================================
template <int B_r, int B_c, int D>
__global__ void flash_attn_bwd_dkdv_kernel(
    const float* __restrict__ Q,    // [BH, N, D]
    const float* __restrict__ K,    // [BH, N, D]
    const float* __restrict__ V,    // [BH, N, D]
    const float* __restrict__ dO,   // [BH, N, D]
    const float* __restrict__ L,    // [BH, N]
    const float* __restrict__ Di,   // [BH, N]
    float* __restrict__ dK,         // [BH, N, D]
    float* __restrict__ dV,         // [BH, N, D]
    int N)
{
    int tid = threadIdx.x;
    int block_col = blockIdx.x;
    int bh = blockIdx.y;
    int col = block_col * B_c + tid;   // global K/V row index

    const float* Q_bh  = Q  + bh * N * D;
    const float* K_bh  = K  + bh * N * D;
    const float* V_bh  = V  + bh * N * D;
    const float* dO_bh = dO + bh * N * D;
    const float* L_bh  = L  + bh * N;
    const float* Di_bh = Di + bh * N;
    float* dK_bh = dK + bh * N * D;
    float* dV_bh = dV + bh * N * D;

    bool valid = (col < N);

    float scale = rsqrtf((float)D);

    // Load K row and V row into registers
    float k_reg[D], v_reg[D];
    if (valid) {
        for (int d = 0; d < D; d++) {
            k_reg[d] = K_bh[col * D + d];
            v_reg[d] = V_bh[col * D + d];
        }
    }

    // Accumulators
    float dk_acc[D], dv_acc[D];
    for (int d = 0; d < D; d++) {
        dk_acc[d] = 0.0f;
        dv_acc[d] = 0.0f;
    }

    // Shared memory for Q and dO tiles, plus L and Di
    __shared__ float sQ[B_r][D];
    __shared__ float sdO[B_r][D];
    __shared__ float sL[B_r];
    __shared__ float sD[B_r];

    int num_q_blocks = (N + B_r - 1) / B_r;

    for (int i = 0; i < num_q_blocks; i++) {
        int q_start = i * B_r;

        // ALL threads participate in collaborative load
        for (int r = tid; r < B_r; r += B_c) {
            int global_r = q_start + r;
            if (global_r < N) {
                for (int d = 0; d < D; d++) {
                    sQ[r][d]  = Q_bh[global_r * D + d];
                    sdO[r][d] = dO_bh[global_r * D + d];
                }
                sL[r] = L_bh[global_r];
                sD[r] = Di_bh[global_r];
            } else {
                for (int d = 0; d < D; d++) {
                    sQ[r][d]  = 0.0f;
                    sdO[r][d] = 0.0f;
                }
                sL[r] = 0.0f;
                sD[r] = 0.0f;
            }
        }
        __syncthreads();

        if (valid) {
            for (int r = 0; r < B_r; r++) {
                int global_r = q_start + r;
                if (global_r >= N) break;

                float s_val = 0.0f;
                for (int d = 0; d < D; d++)
                    s_val += sQ[r][d] * k_reg[d];
                s_val *= scale;

                float p_val = expf(s_val - sL[r]);

                for (int d = 0; d < D; d++)
                    dv_acc[d] += p_val * sdO[r][d];

                float dp_val = 0.0f;
                for (int d = 0; d < D; d++)
                    dp_val += sdO[r][d] * v_reg[d];

                float ds_val = p_val * (dp_val - sD[r]);

                for (int d = 0; d < D; d++)
                    dk_acc[d] += ds_val * sQ[r][d];
            }
        }
        __syncthreads();
    }

    // Write results (valid threads only)
    if (valid) {
        for (int d = 0; d < D; d++) {
            dK_bh[col * D + d] = scale * dk_acc[d];
            dV_bh[col * D + d] = dv_acc[d];
        }
    }
}

// ============================================================
// Backward Host Launcher
// ============================================================
std::vector<torch::Tensor> flash_attn_backward(
    torch::Tensor Q,    // [B, H, N, D]
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,    // forward output
    torch::Tensor dO,   // upstream gradient
    torch::Tensor L)    // logsumexp from forward
{
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(dO.is_cuda(), "dO must be a CUDA tensor");

    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);
    int D = Q.size(3);
    TORCH_CHECK(D == HD, "Head dimension must be " + std::to_string(HD));

    int BH = B * H;

    // Reshape to [BH, N, D]
    auto Q_flat  = Q.reshape({BH, N, D}).contiguous();
    auto K_flat  = K.reshape({BH, N, D}).contiguous();
    auto V_flat  = V.reshape({BH, N, D}).contiguous();
    auto O_flat  = O.reshape({BH, N, D}).contiguous();
    auto dO_flat = dO.reshape({BH, N, D}).contiguous();
    auto L_flat  = L.reshape({BH, N}).contiguous();

    // Allocate outputs
    auto dQ_flat = torch::zeros_like(Q_flat);
    auto dK_flat = torch::zeros_like(K_flat);
    auto dV_flat = torch::zeros_like(V_flat);

    // Allocate Di (precomputed D_i = rowsum(dO * O))
    auto Di = torch::zeros({BH, N}, Q.options());

    // --- Kernel 1: Precompute D_i ---
    {
        int threads = 256;
        int blocks_per_bh = (N + threads - 1) / threads;
        dim3 grid(blocks_per_bh, BH);
        flash_attn_precompute_D_kernel<<<grid, threads>>>(
            O_flat.data_ptr<float>(),
            dO_flat.data_ptr<float>(),
            Di.data_ptr<float>(),
            N, D);
    }

    // --- Kernel 2: Compute dQ ---
    {
        int num_q_blocks = (N + BR - 1) / BR;
        dim3 grid(num_q_blocks, BH);
        dim3 block(BR);
        flash_attn_bwd_dq_kernel<BR, BC, HD><<<grid, block>>>(
            Q_flat.data_ptr<float>(),
            K_flat.data_ptr<float>(),
            V_flat.data_ptr<float>(),
            dO_flat.data_ptr<float>(),
            L_flat.data_ptr<float>(),
            Di.data_ptr<float>(),
            dQ_flat.data_ptr<float>(),
            N);
    }

    // --- Kernel 3: Compute dK, dV ---
    {
        int num_kv_blocks = (N + BC - 1) / BC;
        dim3 grid(num_kv_blocks, BH);
        dim3 block(BC);
        flash_attn_bwd_dkdv_kernel<BR, BC, HD><<<grid, block>>>(
            Q_flat.data_ptr<float>(),
            K_flat.data_ptr<float>(),
            V_flat.data_ptr<float>(),
            dO_flat.data_ptr<float>(),
            L_flat.data_ptr<float>(),
            Di.data_ptr<float>(),
            dK_flat.data_ptr<float>(),
            dV_flat.data_ptr<float>(),
            N);
    }

    auto dQ_out = dQ_flat.reshape({B, H, N, D});
    auto dK_out = dK_flat.reshape({B, H, N, D});
    auto dV_out = dV_flat.reshape({B, H, N, D});

    return {dQ_out, dK_out, dV_out};
}

// ============================================================
// PyBind11 Bindings
// ============================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attn_forward, "FlashAttention forward (CUDA)");
    m.def("backward", &flash_attn_backward, "FlashAttention backward (CUDA)");
}
