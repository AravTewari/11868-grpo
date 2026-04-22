/*
 * Fused CUDA kernels for GRPO training.
 *
 * Kernel 1: fused_log_prob_gather  — single-pass logsumexp + gather over vocab
 * Kernel 2: group_advantage_norm   — per-group mean/std/normalize
 * Kernel 3: fused_grpo_objective   — clipped surrogate + KL in one pass
 *
 * Build:  nvcc -shared -o grpo_kernels.so grpo_kernels.cu -Xcompiler -fPIC
 */

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

#define BLOCK_DIM 256


/* ── Kernel 1: fused gather + logsumexp ──────────────────────────── */

__global__ void fused_log_prob_gather_kernel(
    const float* __restrict__ scores,      // [B, T, V]
    const long*  __restrict__ token_ids,   // [B, T]
    float*       __restrict__ out,         // [B, T]
    int B, int T, int V)
{
    /*
     * Each thread block handles one (b, t) position.
     * Threads cooperate to reduce over the vocab dimension V:
     *   1. Find max for numerical stability
     *   2. Compute sum(exp(x - max))
     *   3. Grab the target token logit
     *   4. out = target_logit - max - log(sum)
     */
    int bt = blockIdx.x;
    if (bt >= B * T) return;

    int b = bt / T;
    int t = bt % T;
    int tid = threadIdx.x;

    const float* row = scores + (b * T + t) * V;
    long target = token_ids[b * T + t];

    __shared__ float s_max[BLOCK_DIM];
    __shared__ float s_sum[BLOCK_DIM];

    // Phase 1: find max over vocab
    float local_max = -FLT_MAX;
    for (int v = tid; v < V; v += blockDim.x)
        local_max = fmaxf(local_max, row[v]);
    s_max[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        __syncthreads();
    }
    float row_max = s_max[0];

    // Phase 2: sum(exp(x - max))
    float local_sum = 0.0f;
    for (int v = tid; v < V; v += blockDim.x)
        local_sum += expf(row[v] - row_max);
    s_sum[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_sum[tid] += s_sum[tid + s];
        __syncthreads();
    }

    // Phase 3: write result
    if (tid == 0)
        out[b * T + t] = row[target] - row_max - logf(s_sum[0]);
}


/* ── Kernel 2: group advantage normalization ─────────────────────── */

__global__ void group_advantage_norm_kernel(
    const float* __restrict__ rewards,   // [N, G]
    float*       __restrict__ out,       // [N, G]
    int N, int G, float eps)
{
    /*
     * One block per group (row). Threads cooperate over G elements.
     * Two passes: mean, then variance + normalize.
     */
    int n = blockIdx.x;
    if (n >= N) return;
    int tid = threadIdx.x;

    const float* row = rewards + n * G;
    float* out_row = out + n * G;

    __shared__ float s_buf[BLOCK_DIM];

    // Mean
    float local_sum = 0.0f;
    for (int g = tid; g < G; g += blockDim.x)
        local_sum += row[g];
    s_buf[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_buf[tid] += s_buf[tid + s];
        __syncthreads();
    }
    float mean = s_buf[0] / G;

    // Variance
    float local_var = 0.0f;
    for (int g = tid; g < G; g += blockDim.x) {
        float d = row[g] - mean;
        local_var += d * d;
    }
    s_buf[tid] = local_var;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_buf[tid] += s_buf[tid + s];
        __syncthreads();
    }
    float std_val = sqrtf(s_buf[0] / G);

    // Normalize
    for (int g = tid; g < G; g += blockDim.x) {
        if (std_val > 0.0f)
            out_row[g] = (row[g] - mean) / (std_val + eps);
        else
            out_row[g] = 0.0f;
    }
}


/* ── Kernel 3: fused GRPO objective ──────────────────────────────── */

__global__ void fused_grpo_objective_kernel(
    const float* __restrict__ new_lp,
    const float* __restrict__ old_lp,
    const float* __restrict__ ref_lp,
    const float* __restrict__ advantages,
    float*       __restrict__ out,       // [3]: policy_loss, kl_div, total_loss
    int L, float clip_eps, float kl_coeff)
{
    /*
     * Grid-stride loop over L tokens.
     * Each thread accumulates partial sums for policy_loss and kl,
     * then block-reduce to get means.
     */
    int tid = threadIdx.x;

    __shared__ float s_policy[BLOCK_DIM];
    __shared__ float s_kl[BLOCK_DIM];

    float local_policy = 0.0f;
    float local_kl = 0.0f;

    for (int i = blockIdx.x * blockDim.x + tid; i < L; i += gridDim.x * blockDim.x) {
        float ratio = expf(new_lp[i] - old_lp[i]);
        float clipped = fminf(fmaxf(ratio, 1.0f - clip_eps), 1.0f + clip_eps);
        float adv = advantages[i];
        local_policy += fminf(ratio * adv, clipped * adv);

        float log_r = new_lp[i] - ref_lp[i];
        local_kl += (expf(log_r) - 1.0f) - log_r;
    }

    s_policy[tid] = local_policy;
    s_kl[tid] = local_kl;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_policy[tid] += s_policy[tid + s];
            s_kl[tid] += s_kl[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float pl = -s_policy[0] / (float)L;
        float kl = s_kl[0] / (float)L;
        atomicAdd(&out[0], pl);
        atomicAdd(&out[1], kl);
        atomicAdd(&out[2], pl + kl_coeff * kl);
    }
}


/* ── Host wrappers (extern "C") ──────────────────────────────────── */

extern "C" {

void fused_log_prob_gather(
    const void* scores, const void* token_ids, void* out,
    int B, int T, int V)
{
    int n_blocks = B * T;
    fused_log_prob_gather_kernel<<<n_blocks, BLOCK_DIM>>>(
        (const float*)scores, (const long*)token_ids, (float*)out, B, T, V);
}

void group_advantage_norm(
    const void* rewards, void* out,
    int N, int G, float eps)
{
    group_advantage_norm_kernel<<<N, BLOCK_DIM>>>(
        (const float*)rewards, (float*)out, N, G, eps);
}

void fused_grpo_objective(
    const void* new_lp, const void* old_lp, const void* ref_lp,
    const void* advantages, void* out,
    int L, float clip_eps, float kl_coeff)
{
    // Zero output before atomic adds
    cudaMemset(out, 0, 3 * sizeof(float));
    int n_blocks = (L + BLOCK_DIM - 1) / BLOCK_DIM;
    fused_grpo_objective_kernel<<<n_blocks, BLOCK_DIM>>>(
        (const float*)new_lp, (const float*)old_lp, (const float*)ref_lp,
        (const float*)advantages, (float*)out, L, clip_eps, kl_coeff);
}

}  // extern "C"
