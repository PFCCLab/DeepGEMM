#pragma once

#include <cutlass/numeric_types.h>

#include <deep_gemm/common/chunk_task.cuh>
#include <deep_gemm/common/fp8_quant.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

template <uint32_t kNumThreads, uint32_t kNumVecsPerRow, uint32_t kNumElemsPerAccess>
CUTLASS_GLOBAL void __launch_bounds__(kNumThreads)
smxx_weighted_swiglu_grad_impl(const cutlass::bfloat16_t* __restrict__ o1,
                               const float* __restrict__ probs,
                               const cutlass::bfloat16_t* __restrict__ do2,
                               cutlass::bfloat16_t* __restrict__ o2_bwd,
                               cutlass::bfloat16_t* __restrict__ do1,
                               float* __restrict__ dprobs,
                               uint8_t* __restrict__ do1_fp8,
                               uint8_t* __restrict__ do1_scales,
                               uint32_t scale_stride) {
    using vec_t = int4;
    constexpr bool kPrecise = false;
    DG_STATIC_ASSERT(kNumElemsPerAccess * sizeof(cutlass::bfloat16_t) == sizeof(vec_t), "Invalid vector size");
    DG_STATIC_ASSERT(kNumElemsPerAccess % 2 == 0, "A gate/up pair must fit into a vector");
    DG_STATIC_ASSERT(kNumThreads % 32 == 0, "Invalid thread number");
    constexpr uint32_t kNumPairsPerVec = kNumElemsPerAccess / 2;
    constexpr uint32_t kShapeN = kNumVecsPerRow * kNumElemsPerAccess;
    constexpr uint32_t kNumElemsPerPack = quant::kGranK * quant::kNumGranKPerPack;
    DG_STATIC_ASSERT(quant::kGranK % kNumElemsPerAccess == 0, "Invalid granularity");
    DG_STATIC_ASSERT(kShapeN % kNumElemsPerPack == 0, "Invalid shape N");
    constexpr uint32_t kNumLanesPerBlock = quant::kGranK / kNumElemsPerAccess;
    constexpr uint32_t kNumPacksPerRow = kShapeN / kNumElemsPerPack;
    constexpr uint32_t kNumScalesPerRow = kShapeN / quant::kGranK;

    // Wait for the producer (the down-grad GEMM) when PDL is enabled
    cudaGridDependencySynchronize();

    const auto lane_idx = ptx::get_lane_idx();
    const auto warp_idx = threadIdx.x / 32;
    const auto row = blockIdx.x;

    const auto prob = probs[row];
    constexpr auto kNumVec4 = kNumElemsPerAccess / 4;
    float dprobs_sum_vec4[kNumVec4] = {};

    // The exponents are staged and flushed as whole int32 packs to improve cache granularity
    __shared__ __align__(4) uint8_t smem_exp_do1[kNumScalesPerRow * 2];

    for (uint32_t col = threadIdx.x; col < kNumVecsPerRow; col += kNumThreads) {
        const auto off = static_cast<uint64_t>(row) * kNumVecsPerRow + col;
        const auto do2_vec = __ldg(reinterpret_cast<const vec_t*>(do2) + off);
        vec_t o2_vec, d_lo_vec, d_hi_vec;
        float d_act[4], w_act[4];
        float dg_act[kNumElemsPerAccess], du_act[kNumElemsPerAccess];

        const auto off2 = static_cast<uint64_t>(row) * kNumVecsPerRow * 2 + col;
        const auto lo_vec = __ldg(reinterpret_cast<const vec_t*>(o1) + off2);
        const auto hi_vec = __ldg(reinterpret_cast<const vec_t*>(o1) + off2 + kNumVecsPerRow);

        const auto* grad = reinterpret_cast<const cutlass::bfloat16_t*>(&do2_vec);
        const auto* lo = reinterpret_cast<const cutlass::bfloat16_t*>(&lo_vec);
        const auto* hi = reinterpret_cast<const cutlass::bfloat16_t*>(&hi_vec);
        auto* o2_out = reinterpret_cast<cutlass::bfloat16_t*>(&o2_vec);
        auto* d_lo = reinterpret_cast<cutlass::bfloat16_t*>(&d_lo_vec);
        auto* d_hi = reinterpret_cast<cutlass::bfloat16_t*>(&d_hi_vec);

        #pragma unroll
        for (uint32_t i = 0; i < kNumElemsPerAccess; ++ i) {
            const auto k = (i % kNumPairsPerVec) * 2;
            const auto d = static_cast<float>(grad[i]);
            const auto g = static_cast<float>(lo[i]);
            const auto u = static_cast<float>(hi[i]);

            const auto s = kPrecise ? 1.0f / (1.0f + expf(-g))
                                    : __frcp_rn(1.0f + __expf(-g));
            const auto silu = g * s;
            const auto w = silu * u;

            const auto dw = d * prob;
            const auto du = dw * silu;
            const auto dg = kPrecise ? dw * u * s * (1.0f + g * (1.0f - s))
                                     : dw * (u * (s * ((g + 1.0f) - silu)));

            // NOTES: sum dprobs in stride of 4 as paddle reference
            d_act[i % 4] = d;
            w_act[i % 4] = w;
            if (i % 4 == 3) {
                const auto dprobs_sum = d_act[0] * w_act[0] + d_act[1] * w_act[1] +
                                        d_act[2] * w_act[2] + d_act[3] * w_act[3];
                dprobs_sum_vec4[i / 4] += dprobs_sum;
            }

            o2_out[i] = static_cast<cutlass::bfloat16_t>(w * prob);
            d_lo[i] = static_cast<cutlass::bfloat16_t>(dg);
            d_hi[i] = static_cast<cutlass::bfloat16_t>(du);
        }

        // Store do1 and o2
        *(reinterpret_cast<vec_t*>(do1) + off2) = d_lo_vec;
        *(reinterpret_cast<vec_t*>(do1) + off2 + kNumVecsPerRow) = d_hi_vec;
        *(reinterpret_cast<vec_t*>(o2_bwd) + off) = o2_vec;

        #pragma unroll
        for (uint32_t i = 0; i < kNumElemsPerAccess; ++ i) {
            // NOTES: rounds through BF16 here because the paddle reference path downcasts
            //        do1/o2 before quantization
            dg_act[i] = static_cast<float>(d_lo[i]);
            du_act[i] = static_cast<float>(d_hi[i]);
        }

        // Store do1_fp8
        const auto group = col / kNumLanesPerBlock;
        const auto e_dg = quant::store_quantized(do1_fp8, dg_act, row, col,
                                                 true, kNumVecsPerRow * 2);
        const auto e_du = quant::store_quantized(do1_fp8, du_act, row, col + kNumVecsPerRow,
                                                 true, kNumVecsPerRow * 2);
        if (lane_idx % kNumLanesPerBlock == 0) {
            smem_exp_do1[group] = e_dg;
            smem_exp_do1[group + kNumScalesPerRow] = e_du;
        }
    }

    constexpr auto kReduceDim = kNumThreads * kNumVec4;
    __shared__ float smem_dprobs_sum[kReduceDim];
    #pragma unroll
    for (uint32_t i = 0; i < kNumVec4; ++ i)
        smem_dprobs_sum[threadIdx.x * kNumVec4 + i] = dprobs_sum_vec4[i];
    __syncthreads();

    // Store do1_scales
    quant::store_scale_packs(
        do1_scales, scale_stride, row, smem_exp_do1, kNumPacksPerRow * 2, threadIdx.x);

    // Reduce dprobs in warp 0. Strictly fold the back half to the front half
    if (warp_idx == 0) {
        float reg_dprobs_sum[kReduceDim / 32];
        #pragma unroll
        for (uint32_t i = 0; i < kReduceDim / 32; ++ i)
            reg_dprobs_sum[i] = smem_dprobs_sum[i * 32 + lane_idx];

        #pragma unroll
        for (uint32_t stride = kReduceDim / 64; stride > 0; stride /= 2) {
            #pragma unroll
            for (uint32_t i = 0; i < stride; ++ i)
                reg_dprobs_sum[i] += reg_dprobs_sum[i + stride];
        }

        auto dprobs_sum = reg_dprobs_sum[0];
        #pragma unroll
        for (uint32_t stride = 16; stride > 0; stride /= 2)
            dprobs_sum += __shfl_down_sync(0xffffffff, dprobs_sum, stride);

        if (lane_idx == 0)
            dprobs[row] = dprobs_sum;
    }
}

}  // namespace deep_gemm
