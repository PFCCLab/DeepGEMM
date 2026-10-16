#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"

namespace deep_gemm {

class SMXXTokenScatterRuntime final: public LaunchRuntime<SMXXTokenScatterRuntime> {
public:
    struct Args {
        LaunchArgs launch_args;

        int num_threads, block_h, num_experts;
        int num_pad_rows_per_block, num_pad_blocks_per_expert;
        bool quant;
        void* x;
        void* rowmap;
        void* tokens_per_expert;
        void* y;
        void* y_fp8;
        void* y_scales;
        uint32_t num_zipped_tokens, hidden, scale_stride, padding_alignment;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/smxx_token_scatter.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&smxx_token_scatter_impl<
        {}, {}, {}, {}
    >);
}};
)",
        args.num_threads, args.block_h, args.num_experts, args.quant);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.x, args.rowmap, args.tokens_per_expert,
            args.y, args.y_fp8, args.y_scales,
            args.num_zipped_tokens, args.hidden,
            args.scale_stride, args.padding_alignment));
    }
};

// Scatter the zipped rows into the unzipped layout, one CTA per `[1 zipped row, kBlockH channel]`
// plus a tail of CTAs that zero the experts' padded rows
static void smxx_token_scatter(const torch::Tensor& x,
                              const torch::Tensor& rowmap,
                              const torch::Tensor& tokens_per_expert,
                              const torch::Tensor& y,
                              const std::optional<torch::Tensor>& y_fp8,
                              const std::optional<torch::Tensor>& y_scales,
                              const int& num_zipped_tokens,
                              const int& num_experts,
                              const int& hidden,
                              const int& padding_alignment) {
    constexpr int kNumElemsPerVec = 8;

    // One thread owns one 16B vector, so the tile is as wide as the biggest packed scale
    // multiple that divides the hidden dim
    int block_h = 0;
    for (const int& candidate: {2048, 1024, 512}) {
        if (hidden % candidate == 0) {
            block_h = candidate;
            break;
        }
    }
    DG_HOST_ASSERT(block_h > 0 and "The hidden dim must be a multiple of 512");

    const bool quant = y_fp8.has_value();
    const auto num_blocks = num_zipped_tokens + num_experts;

    const SMXXTokenScatterRuntime::Args args = {
        .launch_args = LaunchArgs({num_blocks, hidden / block_h}, block_h / kNumElemsPerVec),
        .num_threads = block_h / kNumElemsPerVec,
        .block_h = block_h,
        .num_experts = num_experts,
        .quant = quant,
        .x = x.data_ptr(),
        .rowmap = rowmap.data_ptr(),
        .tokens_per_expert = tokens_per_expert.data_ptr(),
        .y = y.data_ptr(),
        .y_fp8 = quant ? y_fp8->data_ptr() : nullptr,
        .y_scales = quant ? y_scales->data_ptr() : nullptr,
        .num_zipped_tokens = static_cast<uint32_t>(num_zipped_tokens),
        .hidden = static_cast<uint32_t>(hidden),
        .scale_stride = quant ? static_cast<uint32_t>(y_scales->stride(-2)) : 0,
        .padding_alignment = static_cast<uint32_t>(padding_alignment)
    };
    const auto code = SMXXTokenScatterRuntime::generate(args);
    const auto runtime = compiler->build("smxx_token_scatter", code);
    SMXXTokenScatterRuntime::launch(runtime, args);
}

} // namespace deep_gemm
