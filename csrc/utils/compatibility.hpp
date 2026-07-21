#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

// `torch::kFloat8_e4m3fn` is supported since PyTorch 2.1
#define DG_FP8_COMPATIBLE true

// `cuTensorMapEncodeTiled` is supported since CUDA Driver API 12.1
#define DG_TENSORMAP_COMPATIBLE true

// `cublasGetErrorString` is supported since CUDA Runtime API 11.4.2
#define DG_CUBLAS_GET_ERROR_STRING_COMPATIBLE true

// `CUBLASLT_MATMUL_DESC_FAST_ACCUM` and `CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET` are supported since CUDA Runtime API 11.8
#define DG_CUBLASLT_ADVANCED_FEATURES_COMPATIBLE true
