#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../heuristics/runtime.hpp"

namespace deep_gemm {

class SMXXWeightedSwigluGradRuntime final: public LaunchRuntime<SMXXWeightedSwigluGradRuntime> {
public:
    struct Args {
        LaunchArgs launch_args;

        int num_threads;
        int num_vecs_per_row, num_elems_per_access;
        void* o1;
        void* probs;
        void* do2;
        void* o2_bwd;
        void* do1;
        void* dprobs;
        void* do1_fp8;
        void* do1_scales;
        uint32_t scale_stride;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/smxx_weighted_swiglu_grad.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&smxx_weighted_swiglu_grad_impl<
        {}, {}, {}
    >);
}};
)",
        args.num_threads, args.num_vecs_per_row, args.num_elems_per_access);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.o1, args.probs, args.do2, args.o2_bwd, args.do1, args.dprobs,
            args.do1_fp8, args.do1_scales, args.scale_stride));
    }
};

static void smxx_weighted_swiglu_grad(const torch::Tensor& o1,
                                      const torch::Tensor& probs,
                                      const torch::Tensor& do2,
                                      const torch::Tensor& o2_bwd,
                                      const torch::Tensor& do1,
                                      const torch::Tensor& dprobs,
                                      const torch::Tensor& do1_fp8,
                                      const torch::Tensor& do1_scales) {
    constexpr int kNumElemsPerAccess = 8;

    // Align to paddle reduce stride: paddle uses 256 threads and vec4, we use 128 threads and vec8
    const auto num_threads = 128;
    const auto num_blocks = static_cast<int>(o1.size(0));
    const auto shape_n = static_cast<int>(do2.size(-1));
    DG_HOST_ASSERT(shape_n % kNumElemsPerAccess == 0);
    const auto num_vecs_per_row = shape_n / kNumElemsPerAccess;

    const SMXXWeightedSwigluGradRuntime::Args args = {
        .launch_args = LaunchArgs(num_blocks, num_threads),
        .num_threads = num_threads,
        .num_vecs_per_row = num_vecs_per_row,
        .num_elems_per_access = kNumElemsPerAccess,
        .o1 = o1.data_ptr(),
        .probs = probs.data_ptr(),
        .do2 = do2.data_ptr(),
        .o2_bwd = o2_bwd.data_ptr(),
        .do1 = do1.data_ptr(),
        .dprobs = dprobs.data_ptr(),
        .do1_fp8 = do1_fp8.data_ptr(),
        .do1_scales = do1_scales.data_ptr(),
        .scale_stride = static_cast<uint32_t>(do1_scales.stride(-2))
    };
    const auto code = SMXXWeightedSwigluGradRuntime::generate(args);
    const auto runtime = compiler->build("smxx_weighted_swiglu_grad", code);
    SMXXWeightedSwigluGradRuntime::launch(runtime, args);
}

} // namespace deep_gemm
