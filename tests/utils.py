import torch

from deep_gemm.utils import align, get_mk_alignment_for_contiguous_layout


def _is_all_zero(x: torch.Tensor) -> bool:
    # Paddle has no `equal` kernel for float8; compare the raw bit pattern
    # instead (all-zero bits == all-zero fp8 values).
    if x.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        x = x.contiguous().view(torch.uint8)
    return bool((x == torch.zeros_like(x)).all())


def assert_psum_zero_padding(a: torch.Tensor | tuple, d: torch.Tensor, grouped_layout: torch.Tensor, dtype_label: str) -> None:
    a_data = a[0] if isinstance(a, tuple) else a
    for group_idx, current_m in enumerate(grouped_layout.cpu().tolist()):
        aligned_m = align(current_m, get_mk_alignment_for_contiguous_layout())
        if current_m < aligned_m:
            a_padding = a_data[current_m: aligned_m]
            d_padding = d[current_m: aligned_m]
            assert _is_all_zero(a_padding), f'{group_idx=}, nonzero {dtype_label} input padding'
            assert _is_all_zero(d_padding), f'{group_idx=}, nonzero {dtype_label} output padding'
