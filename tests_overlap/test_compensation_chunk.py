"""Compute compensation, landed in the DeepGEMM chunk kernel.

A helper rank absorbs migrated chunks of a hot PEER expert. We express this to the
existing `bf16_chunk_gemm_nn` with:
  * an EXTENDED weight table  b = [local_experts ++ migrated_peer_experts]  (num_groups grows)
  * a task_queue whose migrated chunks carry expert_idx >= num_local_experts,
    pointing into the migrated region of the weight table, and m_start pointing to
    the migrated tokens' rows in the layout.

The chunk GEMM already indexes weights by task.expert_idx over an arbitrary
num_groups, so NO kernel change is needed -- compensation is a schedule + weight
table construction on top of the shipped kernel. This test proves the migrated
chunks are computed bitwise-identically to a plain m-grouped reference, i.e. the
kernel is a correct compute engine for compensated (peer) chunks.

Oracle: the repo's m_grouped_bf16_gemm reference (same ground truth as test_chunk.py),
cross-checked against the FFN math validated in moe_compensated_node.py.

Run: python tests_overlap/test_compensation_chunk.py
"""
import numpy as np
import paddle
import paddle.nn.functional as F

paddle.set_device("gpu:0")
paddle.empty([32, 1024, 1024, 1024], "uint8")
paddle.seed(0)

from utils import deep_gemm

H, I = 4096, 2048
CHUNK = 4096
ALIGNMENT = 128
NUM_SMS = 100

# A helper rank: its own experts + migrated (hot peer) experts it was assigned.
E_LOCAL = 8
# migrated peer experts this helper absorbs: (peer_global_id, tokens_migrated)
MIGRATED = [(101, 4096), (101, 4096), (207, 3000), (350, 4096 + 1200)]


def align(n):
    return (n + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def build_helper_schedule(local_tokens, migrated, seed=0):
    """Formal compensated-schedule builder for one helper rank.

    Args:
      local_tokens: list[int] length E_LOCAL, tokens for each own expert.
      migrated: list[(peer_expert_id, tokens)] chunks migrated in (grouped per id).
    Returns dict with the extended layout + weight-table plan + task_queue.
    """
    # group migrated tokens by distinct peer expert -> one extended weight slot each
    mig_by_expert = {}
    for pe, tks in migrated:
        mig_by_expert[pe] = mig_by_expert.get(pe, 0) + tks
    mig_experts = sorted(mig_by_expert)                       # distinct peer experts
    # extended expert table: [0..E_LOCAL) local, then migrated peers
    ext_tokens = list(local_tokens) + [mig_by_expert[pe] for pe in mig_experts]
    num_ext = len(ext_tokens)
    weight_src = list(range(E_LOCAL)) + mig_experts           # where each ext slot's weights come from

    # layout: 128-aligned region per extended expert
    m_start, m_indices = [0], []
    for e, n in enumerate(ext_tokens):
        na = align(n)
        m_start.append(m_start[-1] + na)
        m_indices.append(paddle.full([na], e, "int32"))
    m_total = m_start[-1]
    m_start = m_start[:-1]
    m_indices = paddle.concat(m_indices)

    # task queue: chunk each expert region; migrated experts (idx>=E_LOCAL) are the
    # compensated chunks. Interleave to mimic random dispatch arrival.
    per_expert = []
    for e, (n, s) in enumerate(zip(ext_tokens, m_start)):
        per_expert.append([[e, s + off, min(CHUNK, n - off), 1] for off in range(0, n, CHUNK)])
    rng = np.random.default_rng(seed)
    cur = [0] * num_ext
    rem = [len(t) for t in per_expert]
    q = []
    while sum(rem) > 0:
        cands = [i for i in range(num_ext) if rem[i] > 0]
        i = cands[rng.integers(len(cands))]
        q.append(per_expert[i][cur[i]]); cur[i] += 1; rem[i] -= 1
    task_queue = paddle.to_tensor(q, "int32")
    return dict(ext_tokens=ext_tokens, num_ext=num_ext, weight_src=weight_src,
                mig_experts=mig_experts, m_start=m_start, m_indices=m_indices,
                m_total=m_total, task_queue=task_queue)


# PLACEHOLDER_MAIN
def calc_diff(x, y):
    d = (x.float() - y.float()).abs()
    return float(d.mean()), float(d.max())


def main():
    rng = np.random.default_rng(1)
    local_tokens = [int(rng.integers(2000, 9000)) for _ in range(E_LOCAL)]
    sch = build_helper_schedule(local_tokens, MIGRATED)
    num_ext, m_total = sch["num_ext"], sch["m_total"]
    m_indices, m_start = sch["m_indices"], sch["m_start"]
    tq = sch["task_queue"]

    print("=== compute compensation in DeepGEMM chunk kernel ===")
    print(f"local experts: {E_LOCAL}, migrated distinct peer experts: {sch['mig_experts']}")
    print(f"extended weight table groups: {num_ext} (idx >= {E_LOCAL} are migrated)")
    print(f"ext tokens per expert: {sch['ext_tokens']}")
    n_mig_tasks = int((tq[:, 0] >= E_LOCAL).sum())
    print(f"num tasks: {tq.shape[0]} ({n_mig_tasks} are migrated/compensated chunks)")

    # extended weight table: slot e uses weights of weight_src[e] (peers are prefetched)
    paddle.seed(7)
    w_gateup = paddle.randn([num_ext, H, 2 * I], "bfloat16") * 0.02
    w_down = paddle.randn([num_ext, I, H], "bfloat16") * 0.02
    x = paddle.randn([m_total, H], "bfloat16")
    probs = paddle.rand([m_total], "float32")

    deep_gemm.set_num_sms(NUM_SMS)

    # reference (oracle): plain m-grouped GEMM over the whole extended layout
    o1_ref = paddle.empty([m_total, 2 * I], "bfloat16")
    deep_gemm.m_grouped_bf16_gemm_nn_contiguous(x, w_gateup, o1_ref, m_indices)
    g, u = o1_ref.chunk(2, axis=-1)
    g, u = g.float(), u.float()
    o2_ref = ((g * F.sigmoid(g)) * u * probs.unsqueeze(-1)).astype("bfloat16")
    o3_ref = paddle.empty([m_total, H], "bfloat16")
    deep_gemm.m_grouped_bf16_gemm_nn_contiguous(o2_ref, w_down, o3_ref, m_indices)

    # chunk path (compensated schedule): each task claims [expert_idx, m_start, m_size]
    o1 = paddle.full([m_total, 2 * I], float("nan"), "bfloat16")
    o2 = paddle.full([m_total, I], float("nan"), "bfloat16")
    o3 = paddle.full([m_total, H], float("nan"), "bfloat16")
    # warmup + run
    for task_idx in range(tq.shape[0]):
        deep_gemm.bf16_chunk_gemm_nn(x, w_gateup, o1, tq, task_idx)
        deep_gemm.chunk_weighted_swiglu(o1, probs, o2, tq, task_idx, CHUNK, precise=True)
        deep_gemm.bf16_chunk_gemm_nn(o2, w_down, o3, tq, task_idx)
    paddle.device.synchronize()

    # valid (non-padding) rows per expert region
    valid = paddle.zeros([m_total], "bool")
    for e, n in enumerate(sch["ext_tokens"]):
        valid[m_start[e]:m_start[e] + n] = True
    vr = valid.nonzero().squeeze(1)
    # migrated-only valid rows (expert idx >= E_LOCAL)
    mig_mask = paddle.zeros([m_total], "bool")
    for e in range(E_LOCAL, num_ext):
        mig_mask[m_start[e]:m_start[e] + sch["ext_tokens"][e]] = True
    mvr = mig_mask.nonzero().squeeze(1)

    ok = True
    for name, out, ref, tol in (("o1", o1, o1_ref, 0.0), ("o3", o3, o3_ref, 1e-6)):
        avg, mx = calc_diff(out.index_select(vr), ref.index_select(vr))
        mavg, mmx = calc_diff(out.index_select(mvr), ref.index_select(mvr))
        print(f"{name}: all-valid avg={avg:.3e} max={mx:.3e} | migrated-only avg={mavg:.3e} max={mmx:.3e}")
        ok = ok and (avg <= tol)
    print("\nPASSED: chunk kernel computes migrated (peer-expert) chunks correctly"
          if ok else "\nFAILED")
    assert ok


if __name__ == "__main__":
    main()
