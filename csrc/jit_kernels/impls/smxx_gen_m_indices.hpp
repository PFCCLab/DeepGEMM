#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/runtime.hpp"

namespace deep_gemm {

class SMXXGenMIndicesRuntime final: public LaunchRuntime<SMXXGenMIndicesRuntime> {
public:
    struct Args {
        LaunchArgs launch_args;
        int num_threads;
        int num_experts;
        int alignment;
        void* tokens_per_expert;
        void* m_indices;
        uint32_t num_tokens;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/smxx_gen_m_indices.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&smxx_gen_m_indices<
        {}, {}, {}
    >);
}};
)",
        args.num_threads, args.num_experts, args.alignment);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.tokens_per_expert, args.m_indices, args.num_tokens));
    }
};

static void smxx_gen_m_indices(const torch::Tensor& tokens_per_expert,
                               const torch::Tensor& m_indices,
                               const int& num_unzipped_tokens,
                               const int& padding_alignment) {
    constexpr int kNumThreads = 1024;
    const auto vec_size = padding_alignment % 128 == 0 ? 4 : 1;

    const auto num_experts = tokens_per_expert.size(0);
    const auto num_blocks = ceil_div(num_unzipped_tokens, kNumThreads * vec_size);

    const SMXXGenMIndicesRuntime::Args args = {
        .launch_args = LaunchArgs(num_blocks, kNumThreads),
        .num_threads = kNumThreads,
        .num_experts = num_experts,
        .alignment = padding_alignment,
        .tokens_per_expert = tokens_per_expert.data_ptr(),
        .m_indices = m_indices.data_ptr(),
        .num_tokens = num_unzipped_tokens
    };
    const auto code = SMXXGenMIndicesRuntime::generate(args);
    const auto runtime = compiler->build("smxx_gen_m_indices", code);
    SMXXGenMIndicesRuntime::launch(runtime, args);
}

} // namespace deep_gemm
