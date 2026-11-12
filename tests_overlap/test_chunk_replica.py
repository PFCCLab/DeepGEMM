"""FP8 chunk GEMM with UltraEP replica experts, whose weights live in a separate buffer.

The task queue numbers the main experts `0 .. E-1` and the replicas `E .. E+R-1`, and every
launch has to pick the right weight buffer by itself, as the arrival order is unknown.
"""
import time

import paddle

paddle.empty([32, 1024, 1024, 1024], "uint8")
paddle.set_printoptions(linewidth=200)
paddle.seed(0)

from utils import deep_gemm, make_task_queue, check, quant_input, quant_weight

E = 16
R = 4
H = 4096
I = 2048

CHUNK = 4096
NUM_SMS = 148
ALIGNMENT = 128


def make_layout(num_experts, max_tokens):
    tokens_per_expert = paddle.randint(0, max_tokens, [num_experts]).tolist()
    tokens_per_expert[1] = 0  # 空专家不产生 task
    tokens_per_expert[E + 1] = 0
    m_start, m_indices = [0], []
    for expert_idx, n in enumerate(tokens_per_expert):
        n_aligned = (n + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
        m_start.append(m_start[-1] + n_aligned)
        m_indices.append(paddle.full([n_aligned], expert_idx, dtype="int32"))
    return tokens_per_expert, m_start, paddle.concat(m_indices)


def valid_rows(tokens_per_expert, m_start, experts):
    rows = [paddle.arange(m_start[e], m_start[e] + tokens_per_expert[e]) for e in experts]
    return paddle.concat(rows)


def run_case(name, n, k, tokens_per_expert, m_start, m_indices):
    M_total = m_start[-1]
    M = m_start[E]  # 主专家的行在前, 复制专家的行在后

    x = quant_input(paddle.randn([M_total, k], dtype="bfloat16"))
    x = (x[0], x[1].T.contiguous().T)

    # 模拟前向场景, 使用 transpose 的 weight
    w_master = quant_weight(paddle.randn([E, k, n], dtype="bfloat16") * 0.02, transpose=True)
    w_replica = quant_weight(paddle.randn([R, k, n], dtype="bfloat16") * 0.02, transpose=True)

    ref = paddle.empty([M_total, n], dtype="bfloat16")
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        (x[0][:M], x[1][:M].T.contiguous().T), w_master, ref[:M], m_indices[:M])
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        (x[0][M:], x[1][M:].T.contiguous().T), w_replica, ref[M:], m_indices[M:] - E)

    ################################ VALIDATE #################################

    # task queue 里混着主专家和复制专家, 顺序随机
    task_queue = make_task_queue(tokens_per_expert, m_start, CHUNK, ready=1, seed=0)
    num_replica_tasks = int((task_queue[:, 0] >= E).sum())
    print(f"[{name}] n: {n} k: {k} M: {M} M_replica: {M_total - M} "
          f"tasks: {len(task_queue)} replica tasks: {num_replica_tasks}")
    assert 0 < num_replica_tasks < len(task_queue)

    # 传 b_replica 时对比全量输出
    out = paddle.full([M_total, n], float("nan"), dtype="bfloat16")
    for task_idx in range(len(task_queue)):
        deep_gemm.fp8_chunk_gemm_nt(x, w_master, out, task_queue, task_idx, w_replica)
    ok = check(f"[{name}] master+replica", out, ref)

    # 不传 b_replica 时行为不变
    master_queue = task_queue[task_queue[:, 0] < E]
    out = paddle.full([M_total, n], float("nan"), dtype="bfloat16")
    for task_idx in range(len(master_queue)):
        deep_gemm.fp8_chunk_gemm_nt(x, w_master, out, master_queue, task_idx)
    rows = valid_rows(tokens_per_expert, m_start, range(E))
    ok = check(f"[{name}] master only", out[rows], ref[rows]) and ok

    ################################# PROFILE #################################

    # 多了两个 TMA 描述符后 CPU 发射时间是否变化
    repeat = 20
    replica_cpu, replica_gpu, master_cpu, master_gpu = [], [], [], []
    e0, e1 = [paddle.cuda.Event(enable_timing=True) for _ in range(2)]

    for i in range(repeat * 2):
        for _ in range(10):
            paddle.randn([128, 1024, 1024])
        paddle.base.core.nvprof_nvtx_push(name)
        e0.record()
        t0 = time.perf_counter()
        if i % 2 == 0:
            for task_idx in range(len(task_queue)):
                deep_gemm.fp8_chunk_gemm_nt(x, w_master, out, task_queue, task_idx, w_replica)
        else:
            for task_idx in range(len(master_queue)):
                deep_gemm.fp8_chunk_gemm_nt(x, w_master, out, master_queue, task_idx)
        t1 = time.perf_counter()
        e1.record()
        paddle.base.core.nvprof_nvtx_pop()
        paddle.device.synchronize()
        (replica_cpu if i % 2 == 0 else master_cpu).append((t1 - t0) * 1e6)
        (replica_gpu if i % 2 == 0 else master_gpu).append(e0.elapsed_time(e1))

    middel_mean = lambda ts: sum(sorted(ts)[5:-5]) / (len(ts) - 10)
    replica_cpu = middel_mean(replica_cpu) / len(task_queue)
    master_cpu = middel_mean(master_cpu) / len(master_queue)
    replica_gpu = middel_mean(replica_gpu)
    master_gpu = middel_mean(master_gpu)
    print(f"[{name}] cpu_us_per_task: replica: {replica_cpu:.3f} master: {master_cpu:.3f}")
    print(f"[{name}] gpu_ms: replica: {replica_gpu:.3f} master: {master_gpu:.3f}")

    return ok


def main():
    deep_gemm.set_num_sms(NUM_SMS)
    tokens_per_expert, m_start, m_indices = make_layout(E + R, 3 * CHUNK)
    print("tokens_per_expert:", tokens_per_expert)

    paddle.base.core.nvprof_start()

    ok = run_case("gateup", 2 * I, H, tokens_per_expert, m_start, m_indices)
    ok = run_case("down", H, I, tokens_per_expert, m_start, m_indices) and ok

    paddle.base.core.nvprof_stop()

    assert ok
    print("PASSED")



if __name__ == "__main__":
    main()
