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

class SMXXGenRowmapBlockCountRuntime final: public LaunchRuntime<SMXXGenRowmapBlockCountRuntime> {
public:
    struct Args {
        LaunchArgs launch_args;
        int num_threads;
        int num_topk;
        int num_experts;
        void* token_indices;
        void* zipped_rowmap;
        void* block_count;
        uint32_t num_tokens;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/smxx_gen_rowmap.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&smxx_gen_rowmap_block_count<
        {}, {}, {}
    >);
}};
)",
        args.num_threads, args.num_topk, args.num_experts);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.token_indices, args.zipped_rowmap, args.block_count, args.num_tokens));
    }
};

class SMXXGenRowmapScanRuntime final: public LaunchRuntime<SMXXGenRowmapScanRuntime> {
public:
    struct Args {
        LaunchArgs launch_args;
        int num_threads;
        int num_experts;
        void* block_count;
        uint32_t num_blocks;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/smxx_gen_rowmap.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&smxx_gen_rowmap_scan<
        {}, {}
    >);
}};
)",
        args.num_threads, args.num_experts);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.block_count, args.num_blocks));
    }
};

class SMXXGenRowmapStoreRuntime final: public LaunchRuntime<SMXXGenRowmapStoreRuntime> {
public:
    struct Args {
        LaunchArgs launch_args;
        int num_threads;
        int num_topk;
        void* tokens_per_expert;
        void* block_count;
        void* zipped_rowmap;
        uint32_t num_tokens;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/smxx_gen_rowmap.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&smxx_gen_rowmap_store<
        {}, {}
    >);
}};
)",
        args.num_threads, args.num_topk);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.tokens_per_expert, args.block_count, args.zipped_rowmap, args.num_tokens));
    }
};

static void smxx_gen_rowmap(const torch::Tensor& token_indices,
                            const torch::Tensor& tokens_per_expert,
                            const torch::Tensor& zipped_rowmap) {
    constexpr int kNumThreads = 512;
    constexpr int kNumScanThreads = 1024;
    constexpr int kNumScanRows = 64;

    const auto num_tokens = token_indices.size(0);
    const auto num_topk = token_indices.size(1);
    const auto num_experts = tokens_per_expert.size(0);
    const auto num_blocks = ceil_div(static_cast<int>(num_tokens), kNumScanRows);

    const auto block_count = torch::empty(
        {num_blocks, num_experts}, token_indices.options());

    {
        const SMXXGenRowmapBlockCountRuntime::Args args = {
            .launch_args = LaunchArgs(num_blocks, kNumThreads),
            .num_threads = kNumThreads,
            .num_topk = num_topk,
            .num_experts = num_experts,
            .token_indices = token_indices.data_ptr(),
            .zipped_rowmap = zipped_rowmap.data_ptr(),
            .block_count = block_count.data_ptr(),
            .num_tokens = num_tokens
        };
        const auto code = SMXXGenRowmapBlockCountRuntime::generate(args);
        const auto runtime = compiler->build("smxx_gen_rowmap_block_count", code);
        SMXXGenRowmapBlockCountRuntime::launch(runtime, args);
    }

    {
        const SMXXGenRowmapScanRuntime::Args args = {
            .launch_args = LaunchArgs(num_experts, kNumScanThreads),
            .num_threads = kNumScanThreads,
            .num_experts = num_experts,
            .block_count = block_count.data_ptr(),
            .num_blocks = num_blocks
        };
        const auto code = SMXXGenRowmapScanRuntime::generate(args);
        const auto runtime = compiler->build("smxx_gen_rowmap_scan", code);
        SMXXGenRowmapScanRuntime::launch(runtime, args);
    }

    {
        const SMXXGenRowmapStoreRuntime::Args args = {
            .launch_args = LaunchArgs(num_blocks, num_experts),
            .num_threads = num_experts,
            .num_topk = num_topk,
            .tokens_per_expert = tokens_per_expert.data_ptr(),
            .block_count = block_count.data_ptr(),
            .zipped_rowmap = zipped_rowmap.data_ptr(),
            .num_tokens = num_tokens
        };
        const auto code = SMXXGenRowmapStoreRuntime::generate(args);
        const auto runtime = compiler->build("smxx_gen_rowmap_store", code);
        SMXXGenRowmapStoreRuntime::launch(runtime, args);
    }
}

} // namespace deep_gemm
