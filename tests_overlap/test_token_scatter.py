"""token_scatter (unzip + fp8 量化融合) 的单卡正确性测试.

token_scatter 以 zipped 的 recv 张量和 moe_permute 给出的 rowmap 为输入, 一次读入就同时产出
unzipped 的 bf16 结果和它的 1x128 fp8 量化结果, 替代 "token_gather + paddle quant" 两趟.
"""

import paddle

paddle.set_printoptions(linewidth=200)
paddle.seed(0)

from utils import deep_gemm, make_deepep_layout

EP = 16
ALIGNMENT = 128

# (num_local_experts, hidden, seqlen, topk)
CONFIGS = [
    (16, 4096, 8192, 10),
    (16, 7168, 8192, 8),
    (64, 4096, 8192, 8),   # num_local_experts > 32
    (8, 1536, 8192, 4),    # hidden 不是 1024 的倍数
]


def check(name, out, ref):
    if out.dtype == paddle.float8_e4m3fn:
        out, ref = out.view("int8"), ref.view("int8")
    diff = int((out != ref).sum())
    print(f"    {name}: diff {diff}" + ("" if diff == 0 else " " + "-" * 40))
    assert diff == 0, name


def reference_scatter(x, rowmap, num_unzipped_tokens):
    """按 rowmap 的定义独立实现一遍 scatter, 用于交叉验证 moe_permute 的输出."""
    y = paddle.zeros([num_unzipped_tokens, x.shape[1]], dtype=x.dtype)
    dst, src = valid_pairs(rowmap)
    y[dst] = x[src]
    return y


def valid_pairs(rowmap):
    """rowmap 里所有有效项的 (unzipped 下标, zipped 下标)."""
    valid = rowmap >= 0
    dst = rowmap[valid].cast("int64")
    src = paddle.arange(rowmap.shape[0], dtype="int32")[:, None].expand(rowmap.shape)[valid]
    return dst, src


def run(num_local_experts, hidden, seqlen, topk):
    topk_indices, tokens_per_expert, m_start, _ = make_deepep_layout(
        EP, seqlen, num_local_experts, topk, ALIGNMENT)
    num_zipped_tokens, num_unzipped_tokens = len(topk_indices), m_start[-1]
    print(f"E={num_local_experts}, H={hidden}, zipped={num_zipped_tokens}, "
          f"unzipped={num_unzipped_tokens}, "
          f"padding={num_unzipped_tokens - sum(tokens_per_expert)}")

    x = paddle.randn([num_zipped_tokens, hidden], dtype="bfloat16")
    probs = paddle.randn(topk_indices.shape)

    # rowmap 由 paddle 的 moe_permute 给出, 它的 unzip 结果同时作为 bf16 参考值
    y_ref, rowmap, _, _ = paddle.nn.functional.moe_permute(
        x, None, topk_indices.cast("int32"), probs,
        padding_alignment=ALIGNMENT,
        num_experts=num_local_experts,
        tokens_per_expert=tokens_per_expert,
    )
    assert rowmap.shape == [num_zipped_tokens, num_local_experts], rowmap.shape
    assert rowmap.dtype == paddle.int32, rowmap.dtype
    assert y_ref.shape == [num_unzipped_tokens, hidden], y_ref.shape
    check("y_ref (moe_permute vs rowmap)", y_ref,
          reference_scatter(x, rowmap, num_unzipped_tokens))

    # padding 行必须是 0, 其 1x128 量化 scale 即 1.0 的指数位 (0x7f7f7f7f)
    y_fp8_ref, y_scale_t_ref = paddle.incubate.nn.functional.fp8_quant_blockwise(
        y_ref, quant_method="1x128", output_scale_transpose=True, using_ue8m0_scale=True)
    assert y_scale_t_ref.shape == [hidden // 512, num_unzipped_tokens], y_scale_t_ref.shape

    tokens_per_expert_gpu = paddle.to_tensor(tokens_per_expert, dtype="int32")
    y, y_fp8, y_scale_t = deep_gemm.token_scatter(
        x, rowmap, tokens_per_expert_gpu, num_unzipped_tokens, ALIGNMENT, quant=True)

    assert y.shape == y_ref.shape and y.dtype == y_ref.dtype
    assert y_fp8.shape == y_fp8_ref.shape and y_fp8.dtype == y_fp8_ref.dtype
    assert y_scale_t.shape == y_scale_t_ref.shape and y_scale_t.dtype == y_scale_t_ref.dtype
    check("y", y, y_ref)
    check("y_fp8", y_fp8, y_fp8_ref)
    check("y_scale_t", y_scale_t, y_scale_t_ref)

    # test bf16 only
    y_, y_fp8_, y_scale_t_ = deep_gemm.token_scatter(
        x, rowmap, tokens_per_expert_gpu, num_unzipped_tokens, ALIGNMENT)
    check("y_bf16", y_, y_ref)
    assert y_fp8_ is None
    assert y_scale_t_ is None

    # 与现行的 "token_gather + paddle quant" 两趟实现对比耗时, 仅作参考
    index = paddle.full([num_unzipped_tokens], -1, dtype="int32")
    dst, src = valid_pairs(rowmap)
    index[dst] = src

    t = deep_gemm.testing.bench_kineto(
        lambda: deep_gemm.token_scatter(
            x, rowmap, tokens_per_expert_gpu, num_unzipped_tokens, ALIGNMENT, quant=True),
        "token_scatter", suppress_kineto_output=True)
    t_gather = deep_gemm.testing.bench_kineto(
        lambda: deep_gemm.token_gather(x, index), "token_gather",
        suppress_kineto_output=True)
    t_quant = deep_gemm.testing.bench_kineto(
        lambda: paddle.incubate.nn.functional.fp8_quant_blockwise(
            y_ref, quant_method="1x128", output_scale_transpose=True, using_ue8m0_scale=True),
        "quant", suppress_kineto_output=True, with_multiple_kernels=True)
    print(f"    token_scatter: {t * 1e6:.0f} us, "
          f"token_gather + paddle quant: {(t_gather + t_quant) * 1e6:.0f} us")


def main():
    for config in CONFIGS:
        run(*config)


if __name__ == "__main__":
    main()
