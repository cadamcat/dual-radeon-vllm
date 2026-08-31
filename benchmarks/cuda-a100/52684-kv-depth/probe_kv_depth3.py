"""Pass 3: the same sweep timed at the kernel, not through the wrapper.

Passes 1 and 2 both time `unified_attention`, and every small-shape row came out
near 0.2 ms whichever way the timer was arranged. Batching showed that constant
is a real per-call cost rather than an instrument floor, but not *what* it is,
and both arms pay it, so the ratio is pulled toward 1 and the numbers are a
conservative bound rather than a matched scale. vllm#52684's author reports
0.058 ms for shapes where our floor alone is 0.2, so his figures are evidently
not paying it either.

This measures both quantities in one run, per cell:

  wall_ms    the wrapper, timed as pass 2 did -- 50 calls per event pair
  dev_ms     device time of the CUDA kernels only, from torch.profiler

`wall_ms - dev_ms` is then the host-side cost, measured rather than inferred,
and `dev_speedup` is the ratio the two boards can actually be compared on.

Everything else is passes 1 and 2's, unchanged: the arms, forcing past the
gate, head patterns, dtype, head_size, and `speedup = production / bm64`.

    python3 probe_kv_depth3.py /content/kv_depth3.jsonl
"""
import json
import sys

import torch
import triton
from torch.profiler import ProfilerActivity, profile

from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.ops import triton_unified_attention as tua
from vllm.v1.attention.ops.triton_unified_attention import unified_attention

OUT = sys.argv[1] if len(sys.argv) > 1 else "/content/kv_depth3.jsonl"

HEAD_PAIRS = [(32, 8), (28, 4)]
Q_LENS = [16, 32, 64, 96, 128, 256, 1024]
KV_LENS = [4096, 16384]
HEAD_SIZE = 128
DTYPE, DNAME = torch.bfloat16, "bf16"
BLOCK_SIZE = 16
NUM_BLOCKS = 8192
WARMUP = 20
REPS = 50            # wall-clock: calls inside one event pair
BATCHES = 8
PROF_REPS = 20       # profiled calls; device time is summed over these

_pow2 = triton.next_power_of_2
_ORIG_SELECT = tua._select_query_block


def want_for(arm, nq):
    if arm == "production":
        bm = 16 if nq <= 16 else _pow2(nq)
        return (bm, bm // nq, False)
    return (64, 64 // _pow2(nq), True)


def install_arm(arm, nq):
    want = want_for(arm, nq)
    tua._is_gfx1100 = lambda: True
    tua._select_query_block = lambda msq, n, _w=want: _w
    got = tua._select_query_block(0, nq)
    assert got == want, f"{arm}: seam gave {got}, wanted {want}"
    return got


def build(q_len, kv_len, num_qh, num_kvh):
    set_random_seed(0)
    query = torch.randn(q_len, num_qh, HEAD_SIZE, dtype=DTYPE)
    key_cache = torch.randn(NUM_BLOCKS, BLOCK_SIZE, num_kvh, HEAD_SIZE, dtype=DTYPE)
    max_blocks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    assert max_blocks <= NUM_BLOCKS
    return dict(
        q=query, k=key_cache, v=torch.randn_like(key_cache),
        out=torch.empty_like(query),
        cu_seqlens_q=torch.tensor([0, q_len], dtype=torch.int32),
        seqused_k=torch.tensor([kv_len], dtype=torch.int32),
        max_seqlen_q=q_len, max_seqlen_k=kv_len,
        softmax_scale=HEAD_SIZE ** -0.5, causal=True, window_size=(-1, -1),
        block_table=torch.randint(0, NUM_BLOCKS, (1, max_blocks), dtype=torch.int32),
        softcap=0, q_descale=None, k_descale=None, v_descale=None,
    )


def wall_ms(kwargs):
    for _ in range(WARMUP):
        unified_attention(**kwargs)
    torch.cuda.synchronize()
    per = []
    for _ in range(BATCHES):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(REPS):
            unified_attention(**kwargs)
        e.record()
        torch.cuda.synchronize()
        per.append(s.elapsed_time(e) / REPS)
    per.sort()
    return per[len(per) // 2]


def device_ms(kwargs):
    """Summed CUDA-kernel time per call. Only our calls run inside the window,
    so every device event in it belongs to this cell."""
    for _ in range(WARMUP):
        unified_attention(**kwargs)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(PROF_REPS):
            unified_attention(**kwargs)
        torch.cuda.synchronize()
    total_us, names = 0.0, {}
    for ev in prof.key_averages():
        t = getattr(ev, "self_device_time_total", None)
        if t is None:
            t = getattr(ev, "self_cuda_time_total", 0.0)
        if t and t > 0:
            total_us += t
            names[ev.key] = names.get(ev.key, 0.0) + t
    top = sorted(names.items(), key=lambda kv: -kv[1])[:2]
    return total_us / 1000.0 / PROF_REPS, [(k[:44], round(v / 1000.0 / PROF_REPS, 5))
                                           for k, v in top]


def main():
    torch.set_default_device("cuda")
    print(f"device={torch.cuda.get_device_name(0)} torch={torch.__version__} "
          f"triton={triton.__version__} reps={REPS} prof_reps={PROF_REPS}", flush=True)
    rows = []
    for kv_len in KV_LENS:
        for num_qh, num_kvh in HEAD_PAIRS:
            nq = num_qh // num_kvh
            for q_len in Q_LENS:
                r = {"dtype": DNAME, "head_size": HEAD_SIZE,
                     "heads": f"{num_qh}:{num_kvh}", "num_queries_per_kv": nq,
                     "q_len": q_len, "kv_len": kv_len, "kv_mode": str(kv_len)}
                for arm in ("production", "bm64"):
                    sel = install_arm(arm, nq)
                    kwargs = build(q_len, kv_len, num_qh, num_kvh)
                    r[f"{arm}_wall_ms"] = wall_ms(kwargs)
                    d, top = device_ms(kwargs)
                    r[f"{arm}_dev_ms"] = d
                    r[f"{arm}_kernels"] = top
                    r[f"{arm}_select"] = list(sel)
                    r[f"{arm}_host_ms"] = r[f"{arm}_wall_ms"] - d
                tua._select_query_block = _ORIG_SELECT
                r["wall_speedup"] = r["production_wall_ms"] / r["bm64_wall_ms"]
                r["dev_speedup"] = r["production_dev_ms"] / r["bm64_dev_ms"]
                rows.append(r)
                print(f"kv={kv_len:>5d} h={r['heads']:>5s} q={q_len:>5d}  "
                      f"dev {r['production_dev_ms']:.5f}/{r['bm64_dev_ms']:.5f}"
                      f"={r['dev_speedup']:.3f}   "
                      f"wall {r['production_wall_ms']:.5f}/{r['bm64_wall_ms']:.5f}"
                      f"={r['wall_speedup']:.3f}   "
                      f"host {r['production_host_ms']:.4f}", flush=True)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"=== KV_DEPTH3 DONE {len(rows)} rows -> {OUT} ===", flush=True)


if __name__ == "__main__":
    main()
