#pragma once

#include <cuda_fp8.h>
#include <cutlass/numeric_types.h>

#include <deep_gemm/common/fp8_quant.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

// Unzip the zipped rows into the standard (sorted) layout, fused with the 1x128 FP8 quantization
//
// NOTES: `rowmap[i, j]` is the destination row of the zipped row `i` under the local expert `j`,
//        `-1` meaning that the token does not go to that expert; `num_local_experts` is a
//        template parameter, hence the whole row fits a shared memory array
// NOTES: one CTA owns `[1 zipped row, kBlockH channels]`, which is one vector per thread, so a
//        destination row is written by exactly one store per buffer and the quantization block's
//        `amax` is a plain intra-warp butterfly
template <uint32_t kNumThreads, uint32_t kBlockH, uint32_t kNumExperts>
CUTLASS_GLOBAL void __launch_bounds__(kNumThreads)
smxx_token_scatter_impl(const cutlass::bfloat16_t* __restrict__ x,
                        const int* __restrict__ rowmap,
                        const int* __restrict__ tokens_per_expert,
                        cutlass::bfloat16_t* __restrict__ y,
                        uint8_t* __restrict__ y_fp8,
                        uint8_t* __restrict__ y_scales,
                        uint32_t num_zipped_tokens, uint32_t hidden,
                        uint32_t scale_stride, uint32_t padding_alignment) {
    constexpr uint32_t kNumElemsPerVec = quant::kNumElemsPerVec;
    DG_STATIC_ASSERT(kBlockH == kNumThreads * kNumElemsPerVec, "Invalid channel block");
    DG_STATIC_ASSERT(kBlockH % (quant::kGranK * quant::kNumGranKPerPack) == 0, "Invalid block");
    DG_STATIC_ASSERT(kNumThreads % 32 == 0, "Invalid thread number");

    using vec_t = int4;
    DG_STATIC_ASSERT(kNumElemsPerVec * sizeof(cutlass::bfloat16_t) == sizeof(vec_t), "Bad vector");
    constexpr uint32_t kNumPacksPerTile = kBlockH / (quant::kGranK * quant::kNumGranKPerPack);

    // Offline, so the producer is whatever filled the zipped buffer in
    cudaGridDependencySynchronize();

    const auto col = blockIdx.y * kBlockH + threadIdx.x * kNumElemsPerVec;
    const auto pack_idx = blockIdx.y * kNumPacksPerTile + threadIdx.x;
    const auto lane_idx = ptx::get_lane_idx();

    // Use tailing blocks to zero padding tokens
    if (blockIdx.x >= num_zipped_tokens) {
        const auto expert_idx = blockIdx.x - num_zipped_tokens;
        uint32_t m_start = 0;
        for (uint32_t i = lane_idx; i < expert_idx; i += 32)
            m_start += math::align(static_cast<uint32_t>(__ldg(tokens_per_expert + i)),
                                   padding_alignment);
        #pragma unroll
        for (uint32_t offset = 16; offset > 0; offset /= 2)
            m_start += __shfl_xor_sync(0xffffffff, m_start, offset);

        const auto num_tokens = static_cast<uint32_t>(__ldg(tokens_per_expert + expert_idx));
        const auto row_begin = m_start + num_tokens;
        const auto row_end = m_start + math::align(num_tokens, padding_alignment);

        for (uint32_t row = row_begin; row < row_end; ++ row) {
            *reinterpret_cast<vec_t*>(y + static_cast<uint64_t>(row) * hidden + col) = vec_t{};
            *reinterpret_cast<uint2*>(y_fp8 + static_cast<uint64_t>(row) * hidden + col) = uint2{};
            if (threadIdx.x < kNumPacksPerTile)
                *quant::get_scale_pack(y_scales, scale_stride, row, pack_idx) =
                    quant::kZeroBlockPack;
        }
        return;
    }

    // Compact this row's destinations
    __shared__ int smem_dst[kNumExperts];
    uint32_t num_dst = 0;

    for (uint32_t offset = 0; offset < kNumExperts; offset += 32) {
        const auto i = offset + lane_idx;
        int dst = -1;
        if (i < kNumExperts)
            dst = __ldg(rowmap + static_cast<uint64_t>(blockIdx.x) * kNumExperts + i);
        const auto bits = __ballot_sync(0xffffffff, dst >= 0);
        if (dst >= 0) {
            const auto prefix_sum = __popc(bits & ((1u << lane_idx) - 1u));
            smem_dst[num_dst + prefix_sum] = dst;
        }
        num_dst += __popc(bits);
    }
    // NOTES: all warps write the same content to smem, so no block sync is needed
    __syncwarp();

    // Load once and quantize once
    const auto vec = *reinterpret_cast<const vec_t*>(
        x + static_cast<uint64_t>(blockIdx.x) * hidden + col);
    const auto* elems = reinterpret_cast<const cutlass::bfloat16_t*>(&vec);

    float act[kNumElemsPerVec], amax = 0.0f;
    #pragma unroll
    for (uint32_t i = 0; i < kNumElemsPerVec; ++ i) {
        act[i] = static_cast<float>(elems[i]);
        amax = fmaxf(amax, fabsf(act[i]));
    }
    const auto block = quant::compute_pow2_scale(
        quant::group_amax<quant::kNumLanesPerBlock>(amax));

    __nv_fp8x2_e4m3 quantized[kNumElemsPerVec / 2];
    #pragma unroll
    for (uint32_t i = 0; i < kNumElemsPerVec / 2; ++ i)
        quantized[i] = quant::scale_to_fp8x2(act[i * 2], act[i * 2 + 1], block.scale);

    // Store to scatter destinations
    const auto is_block_leader = lane_idx % quant::kNumLanesPerBlock == 0;

    #pragma unroll 1
    for (uint32_t i = 0; i < num_dst; ++ i) {
        const auto row = static_cast<uint32_t>(smem_dst[i]);
        *reinterpret_cast<vec_t*>(y + static_cast<uint64_t>(row) * hidden + col) = vec;
        *reinterpret_cast<uint2*>(y_fp8 + static_cast<uint64_t>(row) * hidden + col) =
            *reinterpret_cast<const uint2*>(quantized);
        if (is_block_leader)
            quant::store_scale(y_scales, scale_stride, row, col / quant::kGranK, block.exponent);
    }
}

}  // namespace deep_gemm
