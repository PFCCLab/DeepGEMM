#include <pybind11/pybind11.h>
#include <torch/python.h>

#include "apis/attention.hpp"
#include "apis/einsum.hpp"
#include "apis/hyperconnection.hpp"
#include "apis/gemm.hpp"
#include "apis/layout.hpp"
#include "apis/mega.hpp"
#include "apis/runtime.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _C
#endif

// Allow Paddle build system to override module name via PADDLE_EXTENSION_NAME
#ifdef PADDLE_EXTENSION_NAME
#undef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME PADDLE_EXTENSION_NAME
#endif

// ReSharper disable once CppParameterMayBeConstPtrOrRef
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DeepGEMM C++ library";

    // Register at::ScalarType enum so it can be used as default argument in pybind11 bindings
    py::enum_<at::ScalarType>(m, "ScalarType")
        .value("Byte", at::ScalarType::Byte)
        .value("Char", at::ScalarType::Char)
        .value("Short", at::ScalarType::Short)
        .value("Int", at::ScalarType::Int)
        .value("Long", at::ScalarType::Long)
        .value("Half", at::ScalarType::Half)
        .value("Float", at::ScalarType::Float)
        .value("Double", at::ScalarType::Double)
        .value("BFloat16", at::ScalarType::BFloat16)
        .value("Float8_e4m3fn", at::ScalarType::Float8_e4m3fn)
        .value("Float8_e5m2", at::ScalarType::Float8_e5m2)
        .export_values();

    // TODO: make SM80 incompatible issues raise errors
    deep_gemm::attention::register_apis(m);
    deep_gemm::einsum::register_apis(m);
    deep_gemm::hyperconnection::register_apis(m);
    deep_gemm::gemm::register_apis(m);
    deep_gemm::layout::register_apis(m);
    deep_gemm::mega::register_apis(m);
    deep_gemm::runtime::register_apis(m);
}
