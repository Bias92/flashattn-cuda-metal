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
// PyBind11 Bindings
// ============================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attn_forward, "FlashAttention forward (CUDA)");
}
