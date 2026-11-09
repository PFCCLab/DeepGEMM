#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"

namespace deep_gemm {

class SMXXTokenGatherRuntime final: public LaunchRuntime<SMXXTokenGatherRuntime> {
public:
    struct Args {
        LaunchArgs launch_args;

        int num_sms, num_threads, num_vecs_per_row, vec_size;
        bool dequant;
        void* x;
        void* index;
        void* scales;
        void* out;
        uint32_t num_rows;
        uint32_t scale_stride;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/smxx_token_gather.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&smxx_token_gather_impl<
        {}, {}, {}, {}, {}
    >);
}};
)",
        args.num_sms, args.num_threads, args.num_vecs_per_row, args.vec_size, args.dequant);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.x, args.index, args.scales, args.out, args.num_rows, args.scale_stride));
    }
};

static int get_vec_size(uint64_t size, int vec_size) {
    while (size % vec_size != 0)
        vec_size /= 2;
    return vec_size;
}

// Row gather with `-1` meaning a zeroed row, in the persistent form
static void smxx_token_gather(const torch::Tensor& x,
                              const torch::Tensor& index,
                              const torch::Tensor& out,
                              const std::optional<torch::Tensor>& scales,
                              const int& row_bytes) {
    constexpr int kNumThreads = 1024;
    constexpr int kNumBytesPerAccess = 16;

    int vec_size = get_vec_size(row_bytes, kNumBytesPerAccess);
    for (const auto& t: {x, out})
        vec_size = get_vec_size(reinterpret_cast<uint64_t>(t.data_ptr()), vec_size);

    const auto num_sms = device_runtime->get_num_sms();
    const SMXXTokenGatherRuntime::Args args = {
        .launch_args = LaunchArgs(num_sms, kNumThreads),
        .num_sms = num_sms,
        .num_threads = kNumThreads,
        .num_vecs_per_row = row_bytes / vec_size,
        .vec_size = vec_size,
        .dequant = scales.has_value(),
        .x = x.data_ptr(),
        .index = index.data_ptr(),
        .scales = scales.has_value() ? scales->data_ptr() : nullptr,
        .out = out.data_ptr(),
        .num_rows = static_cast<uint32_t>(index.numel()),
        .scale_stride = scales.has_value() ? scales->stride(-1) : 0
    };
    const auto code = SMXXTokenGatherRuntime::generate(args);
    const auto runtime = compiler->build("smxx_token_gather", code);
    SMXXTokenGatherRuntime::launch(runtime, args);
}

} // namespace deep_gemm
