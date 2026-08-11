import time
import paddle
paddle.enable_compat(scope={"deep_gemm"})
import deep_gemm
from deep_gemm.utils import per_custom_dims_cast_to_fp8
import numpy as np
from deep_gemm.testing import (
    bench_kineto,
    calc_diff, count_bytes,
    ignore_env, get_arch_major,
    test_filter
)
from deep_gemm.utils import ceil_div, per_custom_dims_cast_to_fp8


# 1. 随机种子
np.random.seed(0)


# 2. Q, K, V维度
# 2.1 Q有2048个，K, V有4096个模拟长上下文
# 2.2 Q为128 * 64 = 8192维
# 2.3 K为128维
# 2.3 V为128维
seq_lens = [2048, 4096]        # Q长度
seq_lens_kv = [4096, 8192]     # KV长度
num_heads = 64
head_dim = 128


# 3. 工具函数，测试算子计算一次的平均运行时间
def bench(func, warmup=10, repeat=50):

    # 1. warmup
    for _ in range(warmup):
        out = func()
        paddle.device.synchronize()

    # 2. benchmark
    start = time.perf_counter()

    for _ in range(repeat):
        out = func()
        paddle.device.synchronize()

    end = time.perf_counter()

    return (end - start) / repeat


# 4. 测试fp8_mqa_logits性能
print("Testing FP8 MQA Logits:")
for seq_len in seq_lens:
    for seq_len_kv in seq_lens_kv:
        # 3.1 np格式张量
        q_np = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
        kv_np = np.random.randn(seq_len_kv, head_dim).astype(np.float32)
        weights_np = np.random.randn(seq_len, num_heads).astype(np.float32)
        ks_np = np.zeros(seq_len, dtype=np.int32)
        ke_np = np.arange(seq_len, dtype=np.int32) + (seq_len_kv - seq_len)

        # 3. Q, K, V张量
        # 3.1 q.shape = [2048, 64, 128]，转fp8
        # 3.2 k.shape = [4096, 128]，转fp8
        # 3.3 weights.shape = [2048, 64]
        # 3.4 ks.shape = [2048]，元素值为0
        # 3.4 ks.shape = [2048]，元素值为2048, 2049, ... 4095
        q = paddle.to_tensor(q_np, dtype='bfloat16')
        kv = paddle.to_tensor(kv_np, dtype='bfloat16')
        q_fp8 = paddle.cast(q, 'float8_e4m3fn')
        kv_fp8 = per_custom_dims_cast_to_fp8(kv, (0,), False)
        weights = paddle.to_tensor(weights_np, dtype='float32')
        ks = paddle.to_tensor(ks_np, dtype='int32')
        ke = paddle.to_tensor(ke_np, dtype='int32')

        # 4. 算子性能测试
        def run():
            return deep_gemm.fp8_mqa_logits(q_fp8, kv_fp8, weights, ks, ke)

        t = bench(run)

        # 5. 打印结果
        flops = 2 * seq_len * seq_len_kv * num_heads * head_dim
        tflops = flops / t / 1e12
        bytes_q = seq_len * num_heads * head_dim
        bytes_kv = seq_len_kv * head_dim
        bytes_logits = seq_len * seq_len_kv * 4
        total_bytes = bytes_q + bytes_kv + bytes_logits
        bandwidth = total_bytes / t / 1e9
        print(
            f" > S={seq_len}, SKV={seq_len_kv:6}, H={num_heads:3}, D={head_dim}: "
            f"{tflops:4.0f} TFLOPS, {t*1e6:4.0f} us, {bandwidth:4.0f} GB/s"
        )


# 5. 测试fp8_paged_mqa_logits性能
def kv_cache_cast_to_fp8(x: paddle.Tensor) -> paddle.Tensor:

    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1

    # 1 计算amax
    x_amax = paddle.amax(
        paddle.abs(x).astype("float32"),
        axis=3,
        keepdim=True
    )

    x_amax = paddle.clip(x_amax, min=1e-4)

    # 2 scale factor
    sf = x_amax / 448.0

    # 3 scale并转fp8
    x_scaled = paddle.cast(x * (1.0 / sf), "float8_e4m3fn")

    # 4 创建打包buffer
    x_fp8 = paddle.empty(
        (num_blocks, block_size * (head_dim + 4)),
        dtype="uint8"
    )

    # 5 写入fp8数据
    x_scaled_uint8 = paddle.view(
        x_scaled.reshape([num_blocks, block_size * head_dim]),
        dtype="uint8"
    )

    x_fp8[:, : block_size * head_dim] = x_scaled_uint8

    # 6 写入scale
    sf_uint8 = paddle.view(
        sf.reshape([num_blocks, block_size]),
        dtype="uint8"
    )

    x_fp8[:, block_size * head_dim :] = sf_uint8

    # 7 reshape回paged layout
    return x_fp8.reshape([num_blocks, block_size, num_heads, head_dim + 4])


print("Testing FP8 Paged MQA Logits:")
max_model_len = 111 * 1000
for is_context_lens_2d in (False, True):
    for batch_size, next_n in [(64, 1), (64, 2), (128, 1)]:
        for heads, index_dim in [(64, 128)]:
            for avg_kv in (8192, 32768):
                num_blocks, blocksize = max_model_len * 3, 64

                q_np = np.random.randn(batch_size, next_n, heads, index_dim).astype(np.float32)
                q = paddle.to_tensor(q_np, dtype="bfloat16")
                q_fp8 = paddle.cast(q, "float8_e4m3fn")

                kv_cache = paddle.randn([num_blocks, blocksize, 1, index_dim],dtype="bfloat16")
                kv_cache_fp8 = kv_cache_cast_to_fp8(kv_cache)

                weights = paddle.randn([batch_size * next_n, heads],dtype="float32")

                # context lens
                context_lens_np = np.random.randint(int(0.7 * avg_kv),int(1.3 * avg_kv),size=(batch_size,)).astype(np.int32)
                context_lens = paddle.to_tensor(context_lens_np)
                context_lens_list = context_lens_np.tolist()
                max_block_len = (max(context_lens_list) + blocksize - 1) // blocksize * blocksize
                block_tables = paddle.zeros((batch_size, max_block_len), dtype="int32")

                # block mapping
                counter = 0
                block_idx_pool = paddle.randperm(num_blocks, dtype="int32")
                for i in range(batch_size):
                    num_blocks_i = ceil_div(context_lens_list[i], blocksize)
                    block_tables[i, :num_blocks_i] = block_idx_pool[counter:counter + num_blocks_i]
                    counter += num_blocks_i

                if is_context_lens_2d:
                    context_lens_2d = paddle.cast((context_lens.astype("float32").unsqueeze(1) + 1)* paddle.rand([batch_size, next_n]),"int32")
                    context_lens_2d[:, next_n - 1] = context_lens
                    schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(context_lens_2d,blocksize,deep_gemm.get_num_sms())

                    def run():
                        return deep_gemm.fp8_paged_mqa_logits(
                            q_fp8,
                            kv_cache_fp8,
                            weights,
                            context_lens_2d,
                            block_tables,
                            schedule_metadata,
                            max_model_len
                        )

                else:
                    schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(context_lens,blocksize,deep_gemm.get_num_sms())

                    def run():
                        return deep_gemm.fp8_paged_mqa_logits(
                            q_fp8,
                            kv_cache_fp8,
                            weights,
                            context_lens,
                            block_tables,
                            schedule_metadata,
                            max_model_len
                        )

                # benchmark
                t = bench(run)

                sum_lens = sum(context_lens_list)
                flops = 2 * sum_lens * next_n * heads * index_dim
                tflops = flops / t / 1e12
                input_bytes = (
                    q_fp8.numel()
                    + weights.numel()
                    + context_lens.numel()
                    + sum_lens * (index_dim + 4)
                    + (sum_lens / blocksize) * 4
                )
                output_bytes = sum_lens * next_n * 4
                bandwidth = (input_bytes + output_bytes) / t / 1e9
                print(
                    f" > BSZ={batch_size:3}, NextN={next_n:1}, "
                    f"H={heads:2}, D={index_dim:3}, L={avg_kv:6}: "
                    f"{tflops:4.0f} TFLOPS, {t*1e6:4.0f} us, "
                    f"{bandwidth:4.0f} GB/s"
                )

print()
