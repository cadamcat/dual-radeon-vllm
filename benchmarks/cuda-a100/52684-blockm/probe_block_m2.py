"""Pass 2 for vllm#52684 on A100: quantify the numerical delta, locate the crossover.

Pass 1 (block_m.jsonl, 140 rows) left two things unresolved:

  (a) 23 of 140 rows were not bitwise-equal between BLOCK_M=16 and 64, all of
      them in one slice (bf16, head_size=64, no sliding window), with the
      largest absolute difference 0.00390625. Absolute size alone does not say
      whether that is one bf16 ULP or eight, so measure the ULP distance and
      how many elements move.
  (b) Every q_len=512 row regressed (0.848-0.991) and q_len=1024 was a wash,
      while 2048+ won by 1.10-2.19x. The PR gates at max_seqlen_q >= 512, so
      the crossover's location is the actionable number.

Also re-runs ``base`` last as ``base2`` so ordering drift inside a row is
measured rather than assumed away: pass 1 ran arms in a fixed order and its
q_len=256 control sat at a median 1.016, which is drift, not signal.

argv: [out.jsonl]
"""

import json
import sys

import torch
import triton

from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.ops import triton_unified_attention as tua
from vllm.v1.attention.ops.triton_unified_attention import unified_attention

OUT = sys.argv[1] if len(sys.argv) > 1 else "/content/block_m2.jsonl"

BLOCK_SIZE = 16
NUM_BLOCKS = 4096
WARMUP, ITERS = 10, 40
MANT_BITS = {torch.bfloat16: 7, torch.float16: 10}

# (a) numerics: the slice that moved, plus two controls that did not.
NUM_SLICES = [
    (torch.bfloat16, "bf16", 64, None),
    (torch.bfloat16, "bf16", 128, None),
    (torch.float16, "fp16", 128, None),
]
NUM_LENS = [1024, 4096, 16384]
# (b) crossover: fine grid through the region where pass 1 flipped sign.
XOVER_SLICES = [
    (torch.bfloat16, "bf16", 128, None),
    (torch.bfloat16, "bf16", 64, None),
    (torch.bfloat16, "bf16", 256, None),
]
XOVER_LENS = [512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 3072]
HEAD_PAIRS = [(32, 8), (32, 4), (16, 1)]

_pow2 = triton.next_power_of_2
_ORIG_SELECT = tua._select_query_block


def install_arm(arm):
    tua._select_query_block = _ORIG_SELECT
    tua._is_gfx1100 = lambda: arm == "pr"


def build(q_len, num_qh, num_kvh, head_size, dtype, window):
    set_random_seed(0)
    query = torch.randn(q_len, num_qh, head_size, dtype=dtype)
    key_cache = torch.randn(NUM_BLOCKS, BLOCK_SIZE, num_kvh, head_size, dtype=dtype)
    value_cache = torch.randn_like(key_cache)
    max_blocks = (q_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    return dict(
        q=query, k=key_cache, v=value_cache, out=torch.empty_like(query),
        cu_seqlens_q=torch.tensor([0, q_len], dtype=torch.int32),
        seqused_k=torch.tensor([q_len], dtype=torch.int32),
        max_seqlen_q=q_len, max_seqlen_k=q_len,
        softmax_scale=head_size ** -0.5, causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        block_table=torch.randint(0, NUM_BLOCKS, (1, max_blocks), dtype=torch.int32),
        softcap=0, q_descale=None, k_descale=None, v_descale=None,
    )


def time_call(kwargs):
    for _ in range(WARMUP):
        unified_attention(**kwargs)
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
          for _ in range(ITERS)]
    for s, e in ev:
        s.record()
        unified_attention(**kwargs)
        e.record()
    torch.cuda.synchronize()
    ms = sorted(s.elapsed_time(e) for s, e in ev)
    return ms[len(ms) // 2]


def ulp_stats(a, b, dtype):
    """Distance between two same-dtype tensors in units of the local ULP."""
    af, bf = a.float(), b.float()
    diff = (af - bf).abs()
    ne = diff != 0
    n_diff = int(ne.sum())
    if n_diff == 0:
        return {"n_elems": a.numel(), "n_diff": 0, "frac_diff": 0.0,
                "max_abs_diff": 0.0, "max_ulp": 0.0}
    mag = torch.maximum(af.abs(), bf.abs())
    # ULP at x for a float with MANT_BITS explicit mantissa bits.
    exp = torch.floor(torch.log2(mag.clamp(min=torch.finfo(torch.float32).tiny)))
    ulp = torch.pow(2.0, exp - MANT_BITS[dtype])
    d = (diff[ne] / ulp[ne])
    return {"n_elems": a.numel(), "n_diff": n_diff,
            "frac_diff": n_diff / a.numel(),
            "max_abs_diff": float(diff.max()),
            "max_ulp": float(d.max()), "mean_ulp": float(d.mean()),
            "ulp_hist": {str(k): int((d.round() == k).sum()) for k in (1, 2, 3)},
            "n_gt_1ulp": int((d > 1.0000001).sum())}


def main():
    torch.set_default_device("cuda")
    print(f"device={torch.cuda.get_device_name(0)} "
          f"torch={torch.__version__} triton={triton.__version__}", flush=True)
    fh = open(OUT, "w")

    print("\n=== (a) numerics ===", flush=True)
    for dtype, dname, hs, win in NUM_SLICES:
        for num_qh, num_kvh in HEAD_PAIRS:
            nq = num_qh // num_kvh
            for q_len in NUM_LENS:
                outs = {}
                for arm in ("base", "pr"):
                    install_arm(arm)
                    kw = build(q_len, num_qh, num_kvh, hs, dtype, win)
                    unified_attention(**kw)
                    torch.cuda.synchronize()
                    outs[arm] = kw["out"].clone()
                st = ulp_stats(outs["base"], outs["pr"], dtype)
                row = {"kind": "numerics", "dtype": dname, "head_size": hs,
                       "num_queries_per_kv": nq, "q_len": q_len,
                       "sliding_window": win, **st}
                fh.write(json.dumps(row) + "\n"); fh.flush()
                print(f"{dname} hs={hs:<3d} nq={nq:<3d} q={q_len:<6d} "
                      f"diff {st['n_diff']}/{st['n_elems']} "
                      f"({100*st['frac_diff']:.4f}%) maxabs={st['max_abs_diff']:.3g} "
                      f"max_ulp={st['max_ulp']:.3f} >1ulp={st.get('n_gt_1ulp',0)}",
                      flush=True)

    print("\n=== (b) crossover ===", flush=True)
    for dtype, dname, hs, win in XOVER_SLICES:
        for num_qh, num_kvh in HEAD_PAIRS:
            nq = num_qh // num_kvh
            for q_len in XOVER_LENS:
                t = {}
                for arm in ("base", "pr", "base2"):
                    install_arm("base" if arm.startswith("base") else "pr")
                    kw = build(q_len, num_qh, num_kvh, hs, dtype, win)
                    t[arm] = time_call(kw)
                row = {"kind": "crossover", "dtype": dname, "head_size": hs,
                       "num_queries_per_kv": nq, "q_len": q_len,
                       "sliding_window": win,
                       "base_ms": t["base"], "pr_ms": t["pr"], "base2_ms": t["base2"],
                       "speedup": t["base"] / t["pr"],
                       "speedup_vs_base2": t["base2"] / t["pr"],
                       "drift": t["base2"] / t["base"]}
                fh.write(json.dumps(row) + "\n"); fh.flush()
                print(f"{dname} hs={hs:<3d} nq={nq:<3d} q={q_len:<6d} "
                      f"base={t['base']:.4f} pr={t['pr']:.4f} base2={t['base2']:.4f} "
                      f"x{row['speedup']:.3f} (vs base2 x{row['speedup_vs_base2']:.3f}) "
                      f"drift={row['drift']:.3f}", flush=True)

    fh.close()
    print("PROBE2_DONE", flush=True)


if __name__ == "__main__":
    main()
