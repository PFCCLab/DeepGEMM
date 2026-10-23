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
        int activation;
        bool quant;
        void* o1;
        void* probs;
        void* do2;
        void* o2_bwd;
        void* do1;
        void* dprobs;
        void* do1_fp8;
        void* do1_scales;
        uint32_t scale_stride;
        float arg0, arg1;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/smxx_weighted_swiglu_grad.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&smxx_weighted_swiglu_grad_impl<
        {}, {}, {}, {}, {}
    >);
}};
)",
        args.num_threads, args.num_vecs_per_row, args.num_elems_per_access, args.activation,
        args.quant);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.o1, args.probs, args.do2, args.o2_bwd, args.do1, args.dprobs,
            args.do1_fp8, args.do1_scales, args.scale_stride, args.arg0, args.arg1));
    }
};

static void smxx_weighted_swiglu_grad(const torch::Tensor& o1,
                                      const torch::Tensor& probs,
                                      const torch::Tensor& do2,
                                      const torch::Tensor& o2_bwd,
                                      const torch::Tensor& do1,
                                      const torch::Tensor& dprobs,
                                      const std::optional<torch::Tensor>& do1_fp8,
                                      const std::optional<torch::Tensor>& do1_scales,
                                      const int& activation,
                                      const float& arg0,
                                      const float& arg1) {
    constexpr int kNumElemsPerAccess = 8;

    const auto quant = do1_fp8.has_value();
    // Align to paddle reduce stride: paddle uses 256 threads and vec4, we use 128 threads and vec8
    const auto num_threads = (quant && activation == 0) ? 128 : 256;
    const auto num_blocks = static_cast<int>(o1.size(0));
    const auto shape_n = static_cast<int>(do2.size(-1));
    DG_HOST_ASSERT(shape_n % kNumElemsPerAccess == 0);
    const auto num_vecs_per_row = shape_n / kNumElemsPerAccess;

    const SMXXWeightedSwigluGradRuntime::Args args = {
        .launch_args = LaunchArgs(num_blocks, num_threads),
        .num_threads = num_threads,
        .num_vecs_per_row = num_vecs_per_row,
        .num_elems_per_access = kNumElemsPerAccess,
        .activation = activation,
        .quant = quant,
        .o1 = o1.data_ptr(),
        .probs = probs.data_ptr(),
        .do2 = do2.data_ptr(),
        .o2_bwd = o2_bwd.data_ptr(),
        .do1 = do1.data_ptr(),
        .dprobs = dprobs.data_ptr(),
        .do1_fp8 = quant ? do1_fp8->data_ptr() : nullptr,
        .do1_scales = quant ? do1_scales->data_ptr() : nullptr,
        .scale_stride = quant ? static_cast<uint32_t>(do1_scales->stride(-2)) : 0,
        .arg0 = arg0,
        .arg1 = arg1
    };
    const auto code = SMXXWeightedSwigluGradRuntime::generate(args);
    const auto runtime = compiler->build("smxx_weighted_swiglu_grad", code);
    SMXXWeightedSwigluGradRuntime::launch(runtime, args);
}

} // namespace deep_gemm
