#pragma once

#include <deep_gemm/common/fp8_quant.cuh>
#include <deep_gemm/common/utils.cuh>

namespace deep_gemm {

// Row gather with -1 meaning a zeroed row
template <uint32_t kNumSMs, uint32_t kNumThreads, uint32_t kNumVecsPerRow, uint32_t kVecSize,
          bool kDequant>
CUTLASS_GLOBAL void __launch_bounds__(kNumThreads, 1)
smxx_token_gather_impl(const void* __restrict__ x,
                       const int* __restrict__ index,
                       const uint8_t* __restrict__ scales,
                       void* __restrict__ out,
                       uint32_t num_rows,
                       uint32_t scale_stride) {
    cudaGridDependencySynchronize();

    DG_STATIC_ASSERT(kVecSize == 1 || kVecSize == 2 || kVecSize == 4 || kVecSize == 8 ||
                     kVecSize == 16, "Only supports char, short, int, int2 and int4");
    using vec_t = std::conditional_t<kVecSize == 1, char,
                  std::conditional_t<kVecSize == 2, short,
                  std::conditional_t<kVecSize == 4, int,
                  std::conditional_t<kVecSize == 8, int2, int4>>>>;

    constexpr uint32_t kGridStride = kNumSMs * kNumThreads;
    const auto num_vecs = static_cast<uint64_t>(num_rows) * kNumVecsPerRow;

    // Persistent: a fixed number of CTAs strides over the flattened output
    for (uint64_t i = blockIdx.x * kNumThreads + threadIdx.x; i < num_vecs; i += kGridStride) {
        const auto src_row = index[i / kNumVecsPerRow];
        const auto x_vec =
            src_row >= 0 ? __ldg(reinterpret_cast<const vec_t*>(x) +
                                 static_cast<uint64_t>(src_row) * kNumVecsPerRow +
                                 i % kNumVecsPerRow)
                         : vec_t{};

        if constexpr (kDequant) {
            DG_STATIC_ASSERT(kVecSize == 16, "Dequant requires 16B align");

            vec_t u_vec, v_vec;
            const auto* x_fp8 = reinterpret_cast<const uint8_t*>(&x_vec);
            auto* u_bf16 = reinterpret_cast<cutlass::bfloat16_t*>(&u_vec);
            auto* v_bf16 = reinterpret_cast<cutlass::bfloat16_t*>(&v_vec);
            const auto pack_idx = (i % kNumVecsPerRow) * kVecSize / quant::kGranK;
            const auto pack_col = pack_idx / quant::kNumGranKPerPack;
            const auto pack_row = pack_idx % quant::kNumGranKPerPack;

            const auto exponent =
                src_row >= 0 ? __ldg(scales + (pack_col * scale_stride + src_row) *
                                     quant::kNumGranKPerPack + pack_row)
                             : uint8_t(0);
            const auto scale = __uint_as_float(static_cast<uint32_t>(exponent) << 23);

            #pragma unroll
            for (uint32_t k = 0; k < kVecSize; ++ k) {
                const auto x_bf16 = static_cast<cutlass::bfloat16_t>(
                    quant::fp8_to_float(x_fp8[k]) * scale);
                if (k < kVecSize / 2)
                    u_bf16[k] = x_bf16;
                else
                    v_bf16[k - kVecSize / 2] = x_bf16;
            }

            reinterpret_cast<vec_t*>(out)[i * 2] = u_vec;
            reinterpret_cast<vec_t*>(out)[i * 2 + 1] = v_vec;
        } else {
            reinterpret_cast<vec_t*>(out)[i] = x_vec;
        }
    }
}

}  // namespace deep_gemm
