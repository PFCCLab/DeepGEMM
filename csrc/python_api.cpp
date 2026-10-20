#include <pybind11/pybind11.h>
#include <torch/python.h>

// Paddle only registers `phi::DataType` (exposed as `paddle.dtype`) with pybind11,
// so bindings using `at::ScalarType` (a standalone enum in the compat layer) need
// a caster that converts through `phi::DataType`.
// NOTES: PaddlePaddle/Paddle#79570 registers this caster upstream, but release/3.4 does not include it,
// so `setup.py` defines `DG_PADDLE_HAS_SCALAR_TYPE_CASTER` only when the installed Paddle already provides it
#ifndef DG_PADDLE_HAS_SCALAR_TYPE_CASTER
namespace pybind11::detail {
template <>
struct type_caster<c10::ScalarType> {
    PYBIND11_TYPE_CASTER(c10::ScalarType, const_name("paddle.dtype"));

    bool load(handle src, bool) {
        try {
            value = compat::_PD_PhiDataTypeToAtenScalarType(src.cast<phi::DataType>());
            return true;
        } catch (...) {
            return false;
        }
    }

    static handle cast(const c10::ScalarType& src, return_value_policy, handle) {
        return handle(paddle::pybind::ToPyObject(compat::_PD_AtenScalarTypeToPhiDataType(src)));
    }
};
} // namespace pybind11::detail
#endif

#include <deep_jit/backend/cuda/backend.hpp>
#include <deep_jit/python_api.hpp>

#include "apis/config.hpp"
#include "apis/attention.hpp"
#include "apis/einsum.hpp"
#include "apis/epilogue_class.hpp"
#include "apis/hyperconnection.hpp"
#include "apis/gemm.hpp"
#include "apis/layout.hpp"
#include "apis/locality_domain.hpp"
#include "apis/mega_moe.hpp"
#include "apis/mega_mhc.hpp"
#include "apis/mega_gate.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _C
#endif

// ReSharper disable once CppParameterMayBeConstPtrOrRef
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DeepGEMM C++ library";

    // Register JIT objects
    deep_jit::register_python_api(m, deep_gemm::jit);

    // Register config APIs
    deep_gemm::config::register_apis(m);

    // Register locality domain APIs
    deep_gemm::locality_domain::register_apis(m);

    // Register epilogue classes
    deep_gemm::epilogue_class::register_apis(m);

    // Register kernels
    // TODO: make SM80 incompatible issues raise errors
    deep_gemm::attention::register_apis(m);
    deep_gemm::einsum::register_apis(m);
    deep_gemm::hyperconnection::register_apis(m);
    deep_gemm::gemm::register_apis(m);
    deep_gemm::layout::register_apis(m);
    deep_gemm::mega::register_apis(m);
    deep_gemm::mega_mhc::register_apis(m);
    deep_gemm::mega_gate::register_apis(m);
}
