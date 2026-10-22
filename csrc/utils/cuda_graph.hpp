#pragma once

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include "exception.hpp"

namespace deep_gemm {

static cudaStreamCaptureStatus get_current_stream_capture_status() {
    cudaStreamCaptureStatus status;
    DG_CUDA_RUNTIME_CHECK(cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream().stream(), &status));
    return status;
}

}  // namespace deep_gemm
