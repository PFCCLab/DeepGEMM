#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <deep_gemm/common/exception.cuh>

DG_STATIC_ASSERT(CUDA_VERSION >= 12090, "DeepGEMM requires CUDA Driver API 12.9 or newer");
DG_STATIC_ASSERT(CUDART_VERSION >= 12090, "DeepGEMM requires CUDA Runtime API 12.9 or newer");
