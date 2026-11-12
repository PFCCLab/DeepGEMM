import os
import numpy as np

import paddle

os.environ["HOME"] = os.path.dirname(__file__)  # deep_gemm cache

try:
    from teramoe import deep_gemm
except ImportError:
    import deep_gemm

print("deep_gemm:", deep_gemm.__file__)


def make_deepep_layout(ep_size, seqlen, local_e, topk, alignment=128, router_bias=0.1):
    """
    模拟在一个 EP 组内, 每个 rank 有 E 个专家的情况下, rank 0 收到的 dispatch+unzip 后的
    (recv_token_indices, tokens_per_expert).
    """
    # 模拟全局 token 打分
    scores = paddle.randn([ep_size * seqlen, ep_size * local_e])
    if router_bias:  # add some system bias to experts
        scores += paddle.randn([ep_size * local_e]) * router_bias
    _, topk_indices = scores.topk(topk)

    # 只保留命中 rank0 的 token, 故 topk_indices 里每一行至少有一个非 -1 值
    topk_hit = topk_indices < local_e
    token_hit = topk_hit.any(axis=1).nonzero().squeeze(1)
    topk_indices[~topk_hit] = -1
    topk_indices = topk_indices[token_hit]

    # DeepEP 并未对每行内容进行排序, 甚至有效值和 -1 是交错排列的, 这里对每行进行乱序
    row_perm = paddle.randn(topk_indices.shape).argsort(axis=1)
    topk_indices = topk_indices.index_sample(row_perm)

    tokens_per_expert = paddle.sum(
        paddle.arange(local_e)[:, None] == topk_indices.flatten(), axis=1).tolist()

    m_start, m_indices = [0], []

    for expert_idx, n in enumerate(tokens_per_expert):
        n_aligned = (n + alignment - 1) // alignment * alignment
        m_start.append(m_start[-1] + n_aligned)
        m_indices.append(paddle.full([n_aligned], expert_idx, dtype="int32"))

    m_indices = paddle.concat(m_indices)

    return topk_indices, tokens_per_expert, m_start, m_indices


def make_atomic_layout(topk_indices, tokens_per_expert, m_start):
    """
    根据 DeepEP 序构造一份随机顺序的 atomic 序.
    DeepEP 序前反向是相同的, 但是 atomic 序不同, 可以调用两次本函数模拟不同的 atomic 序.
    """
    atomic_to_zip = paddle.full([m_start[-1]], -1, dtype="int32")
    zip_to_atomic = paddle.full(topk_indices.shape, -1, dtype="int32")

    for expert_idx, (n, offset) in enumerate(zip(tokens_per_expert, m_start)):
        # 选出属于 expert_idx 的 token 并打乱顺序
        slot_hit = topk_indices == expert_idx
        token_idxs = slot_hit.any(axis=1).nonzero().squeeze(1).cast("int32")
        assert len(token_idxs) == n
        perm = paddle.randperm(n)

        atomic_to_zip[offset : offset + n] = token_idxs[perm]

        zip_to_atomic[slot_hit] = paddle.empty([n], dtype="int32").scatter_(
            perm, paddle.arange(n, dtype="int32") + offset
        )

    return atomic_to_zip, zip_to_atomic


def make_task_queue(counts, m_start, chunk, ready, seed=0):
    """The chunk arrival queue: one `[expert_idx, m_start, m_size, ready]` row per task.

    A real DeepEP dispatch appends a row when a chunk lands, so the expert order is
    random while chunks of one expert stay ordered.
    """
    per_expert_tasks = []
    for expert_idx, (count, start) in enumerate(zip(counts, m_start)):
        per_expert_tasks.append([
            [expert_idx, start + offset, min(chunk, count - offset), ready]
            for offset in range(0, count, chunk)
        ])

    # Interleave experts randomly, keeping each expert's chunks in order
    rng = np.random.default_rng(seed)
    cursors, remaining, queue = [0] * len(counts), [len(tasks) for tasks in per_expert_tasks], []
    while sum(remaining) > 0:
        candidates = [i for i, n in enumerate(remaining) if n > 0]
        i = candidates[rng.integers(len(candidates))]
        queue.append(per_expert_tasks[i][cursors[i]])
        cursors[i] += 1
        remaining[i] -= 1

    return paddle.to_tensor(queue, dtype="int32")


def get_atomic_perm(tokens_per_expert, m_start, atomic_to_zip):
    """将 atomic 序的 o1/o2/o3 等转换为参考序的映射表, padding 保留原位."""
    perm = paddle.arange(m_start[-1])
    for n, offset in zip(tokens_per_expert, m_start):
        perm[offset : offset + n] = atomic_to_zip[offset : offset + n].argsort() + offset
    return perm


def parse_activation(s):
    name, *args = s.split(",")
    if name == "swiglu":
        if len(args) == 0:
            return {"activation": 0}
        if len(args) == 1:
            return {"activation": 1, "arg0": float(args[0])}
        raise ValueError("swiglu accepts 0 or 1 argument")
    if name == "situ":
        if len(args) == 1:
            return {"activation": 2, "arg0": float(args[0])}
        if len(args) == 2:
            return {"activation": 3, "arg0": float(args[0]), "arg1": float(args[1])}
        raise ValueError("situ accepts 1 or 2 arguments")
    raise ValueError("unknown activation type")


def check(name, x, y, tol=None):
    diff = (x.float() - y.float()).abs()
    avg, max = float(diff.mean()), float(diff.max())
    if tol is None:
        ok = max == 0
    else:
        ok = (diff <= tol + tol * y.abs().float()).all().item()
    avg = "0" if avg == 0 else f"{avg:e}"
    max = "0" if max == 0 else f"{max:e}"
    banner = "" if ok else (" " + "-" * 40)
    print(f"{name}: avg: {avg} max: {max}{banner}")
    return ok


def check_fp8(name, x, y):
    fp8_diff = int((x[0].view("int8") != y[0].view("int8")).sum())
    scale_diff = int((x[1] != y[1]).sum())
    ok = fp8_diff == 0 and scale_diff == 0
    banner = "" if ok else (" " + "-" * 40)
    print(f"{name}: fp8: {fp8_diff} scale: {scale_diff}{banner}")
    return ok


def make_mini_task_queue(tokens_per_expert):
    tokens_per_expert = np.array(tokens_per_expert, np.int32)
    tasks_per_expert = (tokens_per_expert + 127) // 128
    task_queue = np.empty((tasks_per_expert.sum(), 4), np.int32)

    # expert_id can be omitted
    task_queue[:, 1] = np.arange(0, len(task_queue) * 128, 128, np.int32)
    task_queue[:, 2] = 128
    task_queue[:, 3] = 1

    last_task_per_expert = tasks_per_expert.cumsum() - 1
    last_size_per_expert = tokens_per_expert % 128
    task_queue[last_task_per_expert, 2] = np.where(last_size_per_expert, last_size_per_expert, 128)
    return task_queue


def quant_input(x):
    """对于 hidden_states, 在 hidden 维上使用 128 分块量化."""
    x_fp8, scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
        x,
        quant_method="1x128",
        output_scale_transpose=False,
        using_ue8m0_scale=True,
    )
    # quant 算子因为历史原因会将长度向 4 对齐, 这里截断不影响连续性
    scale = scale[:x.shape[0]]
    assert x_fp8.shape == x.shape
    assert scale.shape == [x.shape[0], x.shape[1] // 512]
    assert x_fp8.is_contiguous()
    assert scale.is_contiguous()
    return x_fp8, scale


def quant_weight(w, transpose=False):
    """对于 weight, 在 n 维上使用 128 分块量化."""
    import paddlefleet_ops
    if transpose:
        w_fp8, scale = paddlefleet_ops.fuse_stack_transpose_fp8_quant(
            list(w),
            using_pow2_scaling=False,
            using_ue8m0_scale=True,
            output_scale_transpose=False,
        )
        assert w_fp8.shape == [w.shape[0] * w.shape[2], w.shape[1]]
        assert scale.shape == [w.shape[0] * w.shape[2], w.shape[1] // 512]
        assert w_fp8.is_contiguous()
        assert scale.is_contiguous()
    else:
        w_fp8, scale = paddlefleet_ops.fuse_stack_fp8_quant(
            list(w),
            using_pow2_scaling=False,
            using_ue8m0_scale=True,
            output_scale_transpose=False,
        )
        assert w_fp8.shape == [w.shape[0] * w.shape[1], w.shape[2]]
        assert scale.shape == [w.shape[0] * w.shape[1], w.shape[2] // 512]
        assert w_fp8.is_contiguous()
        assert scale.is_contiguous()
    # quant 算子输出把专家维铺平了，需要重新展开
    w_fp8 = w_fp8.reshape([w.shape[0], -1, w_fp8.shape[1]])
    scale = scale.reshape([w.shape[0], -1, scale.shape[1]])
    # ue8m0 要求 scale 最后两维 transpose
    scale = scale.mT.contiguous().mT
    return w_fp8, scale
