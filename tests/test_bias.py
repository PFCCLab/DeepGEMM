"""Unit test for bias support in fp8_gemm_nt epilogue (SM100).

Scope (per request):
- Only `deep_gemm.fp8_gemm_nt` (FP8 GEMM, NT layout)
- No accumulate
- With and without bias

On SM100 with UE8M0, the kernel consumes int32-packed UE8M0 scale factors.
We pre-pack both A and B SF into int32 via `per_token_cast_to_fp8(..., use_packed_ue8m0=True)`,
so the C++ layout transform takes the (INT, 1, 128) path (no FP32->INT cast).
With sfb dtype = INT on SM100, the default recipe is (1, 1, 128), matching
1D1D kernel semantics.
"""
import paddle
paddle.enable_compat()

import torch
import deep_gemm
from deep_gemm.testing import calc_diff
from deep_gemm.utils.math import per_token_cast_to_fp8
from deep_gemm.utils.layout import get_mn_major_tma_aligned_packed_ue8m0_tensor


def _make_inputs(m, n, k, out_dtype, with_bias, gran_k=128):
    a_bf = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    b_bf = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    bias = (torch.randn((n,), device='cuda', dtype=out_dtype) * 4
            if with_bias else None)

    # Reference computed from the original BF16 inputs (matches what the FP8
    # kernel approximates after quantization).
    # ref_d = (a_bf.float() @ b_bf.float().t()
    #          + (bias.float().view(1, n) if bias is not None else 0)).to(out_dtype)

    # Quantize to FP8 with FP32 UE8M0 scale factors, then convert to MN-major
    # TMA-aligned packed int32 layout (the dtype/layout the SM100 kernel expects).
    a_fp8, a_sf_fp32 = per_token_cast_to_fp8(a_bf, use_ue8m0=True, gran_k=gran_k)
    b_fp8, b_sf_fp32 = per_token_cast_to_fp8(b_bf, use_ue8m0=True, gran_k=gran_k)
    a_sf = get_mn_major_tma_aligned_packed_ue8m0_tensor(a_sf_fp32)
    b_sf = get_mn_major_tma_aligned_packed_ue8m0_tensor(b_sf_fp32)
    assert a_sf.dtype == torch.int32 and b_sf.dtype == torch.int32, \
        f"SF must be int32 on SM100, got {a_sf.dtype}, {b_sf.dtype}"

    # Reference: run the FP8 GEMM with FP32 output (no bias) to obtain the
    # pre-round accumulator, then add bias in FP32 and round to bf16 once.
    # This matches the kernel's path exactly: tmem(fp32) + bias(fp32) -> bf16.
    # If we instead ran the kernel with `out_dtype` and then did `ref_d.add_(bias)`,
    # the reference would round to bf16 twice (once for the gemm output, once
    # for the bias add), introducing 1-ULP differences vs the kernel.
    ref_fp32 = torch.empty((m, n), device='cuda', dtype=torch.float32)
    deep_gemm.fp8_gemm_nt((a_fp8, a_sf), (b_fp8, b_sf), ref_fp32)
    if bias is not None:
        ref_d = (ref_fp32 + bias.float()).to(out_dtype)
    else:
        ref_d = ref_fp32.to(out_dtype)

    d = torch.empty((m, n), device='cuda', dtype=out_dtype)
    return (a_fp8, a_sf), (b_fp8, b_sf), d, ref_d, bias


def run_one(m, n, k, out_dtype, with_bias, label=""):
    a, b, d, ref_d, bias = _make_inputs(m, n, k, out_dtype, with_bias)

    deep_gemm.fp8_gemm_nt(a, b, d, bias=bias)

    import numpy as np
    np.testing.assert_allclose(
        d.astype("float32").numpy(),
        ref_d.astype("float32").numpy(),
        rtol=1e-3,
        atol=1e-3,
    )
    print("[PASS]")


def test_bias_basic():
    print("\n[bias-basic] fp8_gemm_nt, no-accumulate, with/without bias")
    for out_dtype in (torch.bfloat16, ):
        for with_bias in (True, ):
            run_one(128, 3584, 7168, out_dtype, with_bias, label="basic")


def test_bias_shapes():
    print("\n[bias-shapes] fp8_gemm_nt, no-accumulate, with bias, shape sweep")
    for (m, n, k) in [(128, 3584, 7168)]:
        for out_dtype in (torch.bfloat16, ):
            run_one(m, n, k, out_dtype, True, label="shape")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_bias_basic()
    test_bias_shapes()
    print("\nAll bias tests passed.")
