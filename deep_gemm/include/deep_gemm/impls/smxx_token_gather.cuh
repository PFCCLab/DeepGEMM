#pragma once

#include <deep_gemm/common/utils.cuh>

namespace deep_gemm {

// Row gather with -1 meaning a zeroed row
template <uint32_t kNumSMs, uint32_t kNumThreads, uint32_t kNumVecsPerRow, uint32_t kVecSize>
CUTLASS_GLOBAL void __launch_bounds__(kNumThreads, 1)
smxx_token_gather_impl(const void* x, const int* index, void* out, uint32_t num_rows) {
    cudaGridDependencySynchronize();

    DG_STATIC_ASSERT(kVecSize == 1 || kVecSize == 2 || kVecSize == 4 || kVecSize == 8 ||
                     kVecSize == 16, "Only supports char, short, int, int2 and int4");
    using vec_t = std::conditional_t<kVecSize == 1, char,
                  std::conditional_t<kVecSize == 2, short,
                  std::conditional_t<kVecSize == 4, int,
                  std::conditional_t<kVecSize == 8, int2, int4>>>>;

    constexpr uint32_t kGridStride = kNumSMs * kNumThreads;
    const auto num_vecs = num_rows * kNumVecsPerRow;

    // Persistent: a fixed number of CTAs strides over the flattened output
    for (uint32_t i = blockIdx.x * kNumThreads + threadIdx.x; i < num_vecs; i += kGridStride) {
        const auto src_row = index[i / kNumVecsPerRow];
        reinterpret_cast<vec_t*>(out)[i] =
            src_row >= 0 ? __ldg(reinterpret_cast<const vec_t*>(x) +
                                 static_cast<uint64_t>(src_row) * kNumVecsPerRow +
                                 i % kNumVecsPerRow)
                         : vec_t{};
    }
}

}  // namespace deep_gemm
