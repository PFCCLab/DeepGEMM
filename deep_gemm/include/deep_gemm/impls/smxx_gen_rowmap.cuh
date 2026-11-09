#pragma once

#include <cutlass/numeric_types.h>

#include <deep_gemm/common/fp8_quant.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

template <uint32_t kNumThreads, uint32_t kNumTopk, uint32_t kNumExperts>
CUTLASS_GLOBAL void __launch_bounds__(kNumThreads)
smxx_gen_rowmap_block_count(const int* __restrict__ token_indices,
                            int* __restrict__ zipped_rowmap,
                            int* __restrict__ block_count,
                            uint32_t num_tokens) {
    // Split num_tokens evenly into kNumSMs parts
    const auto num_tokens_per_block = math::ceil_div(num_tokens, gridDim.x);
    const auto row_begin = num_tokens_per_block * blockIdx.x;
    const auto row_end = std::min(row_begin + num_tokens_per_block, num_tokens);

    __shared__ int smem_expert_count[kNumExperts];
    for (uint32_t i = threadIdx.x; i < kNumExperts; i += kNumThreads)
        smem_expert_count[i] = 0;

    cudaGridDependencySynchronize();

    for (uint32_t i = row_begin * kNumExperts + threadIdx.x; i < row_end * kNumExperts;
         i += kNumThreads) {
        zipped_rowmap[i] = -1;
    }
    __syncthreads();

    for (uint32_t i = row_begin * kNumTopk + threadIdx.x; i < row_end * kNumTopk;
         i += kNumThreads) {
        const auto expert_idx = __ldg(token_indices + i);
        const auto row_idx = i / kNumTopk;
        if (expert_idx >= 0) {
            atomicAdd(smem_expert_count + expert_idx, 1);
            zipped_rowmap[row_idx * kNumExperts + expert_idx] = 1;
        }
    }
    __syncthreads();

    for (uint32_t i = threadIdx.x; i < kNumExperts; i += kNumThreads)
        block_count[blockIdx.x * kNumExperts + i] = smem_expert_count[i];
}

template <uint32_t kNumThreads, uint32_t kNumExperts>
CUTLASS_GLOBAL void __launch_bounds__(kNumThreads)
smxx_gen_rowmap_scan(int* __restrict__ block_count,
                     uint32_t num_blocks) {
    // One block handles one expert
    const auto expert_idx = blockIdx.x;
    const auto warp_idx = threadIdx.x / 32;
    const auto lane_idx = ptx::get_lane_idx();
    constexpr auto kNumWarps = kNumThreads / 32;

    uint32_t global_prefix = 0;
    __shared__ uint32_t smem_count_w[32], smem_count_r[32];

    cudaGridDependencySynchronize();

    for (uint32_t n = 0; n < num_blocks; n += kNumThreads) {
        // NOTES: all threads should participate
        const auto i = n + threadIdx.x;
        const auto count = static_cast<uint32_t>(
            i < num_blocks ? block_count[i * kNumExperts + expert_idx] : 0);

        const auto prefix_in_warp_incl = math::warp_inclusive_sum(count, lane_idx);
        const auto prefix_in_warp = prefix_in_warp_incl - count;

        if (lane_idx == 31)
            smem_count_w[warp_idx] = prefix_in_warp_incl;
        __syncthreads();
        if (warp_idx == 0)
            smem_count_r[lane_idx] = math::warp_inclusive_sum(smem_count_w[lane_idx], lane_idx);
        __syncthreads();

        const auto warp_prefix = warp_idx > 0 ? smem_count_r[warp_idx - 1] : 0u;
        const auto prefix = global_prefix + warp_prefix + prefix_in_warp;
        if (i < num_blocks)
            block_count[i * kNumExperts + expert_idx] = static_cast<int>(prefix);

        global_prefix += smem_count_r[kNumWarps - 1];
    }
}

template <uint32_t kNumThreads, uint32_t kNumTopk>
CUTLASS_GLOBAL void __launch_bounds__(kNumThreads)
smxx_gen_rowmap_store(const int* __restrict__ tokens_per_expert,
                      const int* __restrict__ block_count,
                      int* __restrict__ zipped_rowmap,
                      uint32_t num_tokens) {
    const auto num_tokens_per_block = math::ceil_div(num_tokens, gridDim.x);
    const auto row_begin = num_tokens_per_block * blockIdx.x;
    const auto row_end = std::min(row_begin + num_tokens_per_block, num_tokens);

    // One thread handles one expert
    constexpr auto kNumExperts = kNumThreads;
    const auto expert_idx = threadIdx.x;

    cudaGridDependencySynchronize();

    uint32_t prefix = __ldg(block_count + blockIdx.x * kNumExperts + expert_idx);
    for (uint32_t i = 0; i < expert_idx; ++ i) {
        const auto n = static_cast<uint32_t>(__ldg(tokens_per_expert + i));
        prefix += math::align(n, quant::kGranK);
    }

    for (uint32_t i = row_begin; i < row_end; ++ i) {
        const auto row_idx = zipped_rowmap[i * kNumExperts + expert_idx];
        if (row_idx >= 0) {
            zipped_rowmap[i * kNumExperts + expert_idx] = prefix;
            ++ prefix;
        }
    }
}

}  // namespace deep_gemm
