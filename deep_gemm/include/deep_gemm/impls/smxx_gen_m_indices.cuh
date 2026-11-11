#pragma once

#include <cutlass/numeric_types.h>

#include <deep_gemm/common/fp8_quant.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

template <uint32_t kNumThreads, uint32_t kNumExperts, uint32_t kAlignment>
CUTLASS_GLOBAL void __launch_bounds__(kNumThreads)
smxx_gen_m_indices(const int* __restrict__ tokens_per_expert,
                   int* __restrict__ m_indices,
                   uint32_t num_tokens) {
    constexpr uint32_t kVecSize = kAlignment % 128 == 0 ? 4 : 1;
    const auto token_idx = (blockIdx.x * kNumThreads + threadIdx.x) * kVecSize;
    const auto warp_idx = threadIdx.x / 32;
    const auto lane_idx = ptx::get_lane_idx();

    cudaGridDependencySynchronize();

    if (token_idx >= num_tokens)
        return;

    int expert_idx = -1;
    uint32_t prefix = 0;

    if constexpr (kVecSize == 4 && kNumExperts >= 16) {
        // Fast path: warp scan
        for (uint32_t i = 0; i < kNumExperts; i += 32) {
            const auto j = i + lane_idx;
            const auto n = j < kNumExperts ? __ldg(tokens_per_expert + j) : 0;
            const auto m = math::align(static_cast<uint32_t>(n), kAlignment);
            const auto k = math::warp_inclusive_sum(m, lane_idx);
            const auto hit = token_idx >= prefix + k - m && token_idx < prefix + k;
            const auto bits = __ballot_sync(0xffffffff, hit);
            if (bits) {
                if (__popc(bits) != 1 && hit) {
                    printf("gen_m_indices multiple experts: expert_idx=%u token_idx=%u "
                           "bits=0x%x\n", j, token_idx, bits);
                    DG_TRAP_ONLY_DEVICE_ASSERT(0);
                }
                expert_idx = i + __popc(bits - 1);
                break;
            }
            prefix += __shfl_sync(0xffffffff, k, 31);
        }
    } else {
        // Slow path: accumulate one by one
        for (uint32_t i = 0; i < kNumExperts; ++ i) {
            prefix += math::align(
                static_cast<uint32_t>(__ldg(tokens_per_expert + i)), kAlignment);
            if (prefix > token_idx) {
                expert_idx = static_cast<int>(i);
                break;
            }
        }
    }

    if (expert_idx < 0) {
        printf("gen_m_indices no expert: block=%u token_idx=%u prefix=%u num_tokens=%u\n",
            blockIdx.x, token_idx, prefix, num_tokens);
        DG_TRAP_ONLY_DEVICE_ASSERT(0);
    }

    if constexpr (kVecSize == 4)
        *reinterpret_cast<int4*>(m_indices + token_idx) =
            int4{expert_idx, expert_idx, expert_idx, expert_idx};
    else
        m_indices[token_idx] = expert_idx;
}

}  // namespace deep_gemm
