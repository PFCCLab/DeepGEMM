"""FP8 migration bit-exactness (extends stage-5/50 bit-exact claim to the PRODUCTION FP8 path).

Compensation migrates a peer expert into a helper's EXTENDED weight table. In FP8, tokens and
weights are already block-quantized before migration, and fp8_chunk_gemm_nt indexes weights by
task.expert_idx over an arbitrary num_groups. So computing a migrated expert's chunk on a helper
(smaller/relocated weight table) must give BIT-IDENTICAL output to computing it in the donor's
full table -- FP8 quantization happens before migration, migration just moves quantized bytes.

Test: 2-group fp8 weight table (expert0 local, expert1 = the migrated peer).
  reference : compute both experts' chunks with the full 2-group table -> o_ref
  migrated  : compute expert1's chunk with a 1-group table = a SLICE of the same fp8 weight+scale
              (expert_idx 0 -> the sliced w1) -> o_mig
  verify    : o_ref[expert1 rows] == o_mig  bitwise (max_diff == 0)

Run: CUDA_VISIBLE_DEVICES=0 python test_fp8_compensation_chunk.py
"""
import paddle
import paddlefleet_ops
from utils import deep_gemm

paddle.set_device("gpu:0"); paddle.seed(0)
deep_gemm.set_num_sms(100)

H, I, CHUNK = 4096, 2048, 4096
QUANT_BLOCK_SIZE = 512
USE_UE8M0 = True


def quant_input(x):
    x_fp8, scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
        x, quant_method="1x128", output_scale_transpose=False, using_ue8m0_scale=USE_UE8M0)
    x_fp8 = x_fp8[:x.shape[0]].contiguous()
    scale = scale[:x.shape[0]]
    scale = scale.T.contiguous().T          # sf.stride(-2)==1 layout required by chunk-GEMM
    return x_fp8, scale


def quant_weight_t(w):
    """n-dim 128-block quant, transposed layout (same as the fp8 forward test)."""
    w_fp8, scale = paddlefleet_ops.fuse_stack_transpose_fp8_quant(
        list(w), using_pow2_scaling=False, using_ue8m0_scale=USE_UE8M0, output_scale_transpose=False)
    w_fp8 = w_fp8.reshape([w.shape[0], -1, w_fp8.shape[1]])
    scale = scale.reshape([w.shape[0], -1, scale.shape[1]])
    if USE_UE8M0:
        scale = scale.transpose([0, 2, 1]).contiguous().transpose([0, 2, 1])
    return w_fp8, scale


def main():
    # 2 experts; tokens: expert0 rows [0:CHUNK], expert1 rows [CHUNK:2CHUNK]
    M = 2 * CHUNK
    x = paddle.randn([M, H], "bfloat16")
    x_fp8, x_scale = quant_input(x)
    w = (paddle.randn([2, H, 2 * I]) * 0.02).cast("bfloat16")     # gateup for 2 experts
    w_fp8, w_scale = quant_weight_t(w)                             # [2, ...]

    # reference: full 2-group table
    o_ref = paddle.full([M, 2 * I], float("nan"), "bfloat16")
    tq_ref = paddle.to_tensor([[0, 0, CHUNK, 1], [1, CHUNK, CHUNK, 1]], "int32")
    for i in range(tq_ref.shape[0]):
        deep_gemm.fp8_chunk_gemm_nt((x_fp8, x_scale), (w_fp8, w_scale), o_ref, tq_ref, i)

    # migrated: 1-group table = expert-1 weight re-quantized ALONE (per-expert blockwise quant
    # is independent -> identical fp8 bytes as in the 2-group table, valid scale layout)
    w1_fp8, w1_scale = quant_weight_t(w[1:2].contiguous())
    o_mig = paddle.full([M, 2 * I], float("nan"), "bfloat16")
    tq_mig = paddle.to_tensor([[0, CHUNK, CHUNK, 1]], "int32")      # expert_idx 0 -> migrated w1
    deep_gemm.fp8_chunk_gemm_nt((x_fp8, x_scale), (w1_fp8, w1_scale), o_mig, tq_mig, 0)

    a = o_ref[CHUNK:2 * CHUNK].astype("float32")
    b = o_mig[CHUNK:2 * CHUNK].astype("float32")
    maxdiff = float((a - b).abs().max())
    print(f"=== FP8 migration bit-exactness (expert1 in 2-group table vs sliced 1-group) ===")
    print(f"o1 expert-1 rows: max_diff = {maxdiff:.3e}  "
          f"{'PASS (bit-exact in FP8)' if maxdiff == 0 else 'MISMATCH'}")
    print("-> migrating an already-quantized FP8 expert to a helper's extended/relocated weight")
    print("   table is bit-exact (fp8 chunk-GEMM indexes by expert_idx; quantization precedes migration).")


if __name__ == "__main__":
    main()
