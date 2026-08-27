"""Does vllm#52684's BLOCK_M=64 long-prefill block help, hurt, or do nothing on CUDA?

vllm#52585 measured BLOCK_M=64 (vs the 16 default) worth 1.26-2.18x on the
gfx1100 unified-attention kernel for long prefill. PR #52684 ships that as a
gfx1100-gated launch choice, and the issue's open question -- asked by the
reporter, unanswered by anyone -- is whether the gate should be there at all.
Nobody has posted a CUDA number.

This probe answers it at the kernel, not through a server. It drives
``unified_attention`` directly with the input construction lifted from
upstream's own ``tests/kernels/attention/test_triton_unified_attention.py``,
and flips PR #52684's own seam (``_is_gfx1100``) to select the launch config.

Three arms, because the PR's tuned path changes two things at once:

  base   (16, BLOCK_M//nq, tuned=False)   Triton's default num_warps
  pr     (64, 64//pow2(nq), tuned=True)   num_warps=4      <- the PR verbatim
  bm64   (64, 64//pow2(nq), tuned=False)  Triton's default num_warps

``base`` vs ``pr`` is the decision the maintainers face. ``bm64`` separates
the wider query block from the warp-count pin, so a regression can be
attributed rather than guessed at.

max_seqlen_q > 1 forces the 2D path (``use_3d`` is False), so every row here
is the 2D prefill kernel the PR targets; the 3D decode path is untouched by
construction and ``q_len=256`` is the below-the-gate control.

Requires the PR applied to the installed tree first (kernel hunk only):

    patch -p1 -d "$(python -c 'import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))')" \
        -i 52684-kernel.diff

vLLM 0.28.0's copy of this kernel file is byte-identical to main, so the PR
applies to the release unmodified (pristine md5 49fab3b6..., patched
f1d7a7e3...; both asserted by setup_vm.py).

argv: [out.jsonl]
"""

import json
import sys

import torch
import triton

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.ops import triton_unified_attention as tua
from vllm.v1.attention.ops.triton_unified_attention import unified_attention

OUT = sys.argv[1] if len(sys.argv) > 1 else "/content/block_m.jsonl"

# (num_query_heads, num_kv_heads); the gate reads num_queries_per_kv.
# 4 = llama-3-8B, 7 = Qwen2-7B (28/4, the non-pow2 case the PR's unit test
# calls out), 8 = llama-3-70B and many others, 16 = the gate's upper edge.
HEAD_PAIRS = [(32, 8), (28, 4), (32, 4), (16, 1)]
# 256 is below the PR's >=512 gate: the control that must show no change.
PREFILL_LENS = [256, 512, 1024, 2048, 4096, 8192, 16384]
BLOCK_SIZE = 16
NUM_BLOCKS = 4096
WARMUP, ITERS = 10, 40

# (dtype, dtype-name, head_size, sliding_window)
SLICES = (
    [(torch.bfloat16, "bf16", 128, None), (torch.bfloat16, "bf16", 128, 4096)]
    + [(torch.bfloat16, "bf16", hs, None) for hs in (64, 256)]
    + [(torch.float16, "fp16", 128, None)]
)

_pow2 = triton.next_power_of_2


def select_for(arm, q_len, nq):
    """The (BLOCK_M, BLOCK_Q, tuned) each arm should produce."""
    if arm == "base":
        bm = 16 if nq <= 16 else _pow2(nq)
        return (bm, bm // nq, False)
    gated = q_len >= 512 and nq <= 16
    if not gated:  # below the gate all arms collapse onto base
        bm = 16 if nq <= 16 else _pow2(nq)
        return (bm, bm // nq, False)
    return (64, 64 // _pow2(nq), arm == "pr")


def install_arm(arm, q_len, nq):
    """Point the PR's own seam at the launch config this arm wants."""
    want = select_for(arm, q_len, nq)
    tua._is_gfx1100 = lambda: arm != "base"
    if arm == "bm64":
        # tuned=False keeps launch_num_warps at None (Triton's default) while
        # still handing the kernel the wider query block.
        tua._select_query_block = lambda msq, n, _w=want: (_w[0], _w[1], False)
    else:
        tua._select_query_block = _ORIG_SELECT
    got = tua._select_query_block(q_len, nq)
    assert got == want, f"{arm}: seam gave {got}, wanted {want}"
    return got


def build(q_len, num_qh, num_kvh, head_size, dtype, sliding_window):
    """Inputs for a single-sequence prefill, upstream's construction."""
    set_random_seed(0)
    kv_len = q_len
    query = torch.randn(q_len, num_qh, head_size, dtype=dtype)
    key_cache = torch.randn(NUM_BLOCKS, BLOCK_SIZE, num_kvh, head_size, dtype=dtype)
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor([0, q_len], dtype=torch.int32)
    kv_lens = torch.tensor([kv_len], dtype=torch.int32)
    max_blocks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_tables = torch.randint(0, NUM_BLOCKS, (1, max_blocks), dtype=torch.int32)
    out = torch.empty_like(query)
    window = (sliding_window - 1, 0) if sliding_window else (-1, -1)
    return dict(
        q=query, k=key_cache, v=value_cache, out=out,
        cu_seqlens_q=cu_query_lens, seqused_k=kv_lens,
        max_seqlen_q=q_len, max_seqlen_k=kv_len,
        softmax_scale=head_size ** -0.5, causal=True, window_size=window,
        block_table=block_tables, softcap=0,
        q_descale=None, k_descale=None, v_descale=None,
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
    return ms[len(ms) // 2], ms[0], ms[-1]


_ORIG_SELECT = tua._select_query_block


def main():
    torch.set_default_device("cuda")
    print(f"device={torch.cuda.get_device_name(0)} "
          f"cap={torch.cuda.get_device_capability(0)} "
          f"torch={torch.__version__} triton={triton.__version__}", flush=True)
    # The kernel's other tuned path (head_size 256 + capability family 100)
    # must be off here or it would confound the 256 rows.
    print("is_device_capability_family(100)=",
          current_platform.is_device_capability_family(100), flush=True)

    fh = open(OUT, "w")
    for dtype, dname, head_size, window in SLICES:
        for num_qh, num_kvh in HEAD_PAIRS:
            nq = num_qh // num_kvh
            for q_len in PREFILL_LENS:
                row = {"dtype": dname, "head_size": head_size,
                       "num_query_heads": num_qh, "num_kv_heads": num_kvh,
                       "num_queries_per_kv": nq, "q_len": q_len,
                       "sliding_window": window, "arms": {}}
                outs = {}
                for arm in ("base", "pr", "bm64"):
                    sel = install_arm(arm, q_len, nq)
                    kwargs = build(q_len, num_qh, num_kvh, head_size, dtype, window)
                    try:
                        med, lo, hi = time_call(kwargs)
                    except Exception as exc:
                        row["arms"][arm] = {"error": f"{type(exc).__name__}: {exc}"}
                        continue
                    row["arms"][arm] = {"block_m": sel[0], "block_q": sel[1],
                                        "tuned": sel[2], "median_ms": med,
                                        "min_ms": lo, "max_ms": hi}
                    outs[arm] = kwargs["out"].clone()
                base = row["arms"]["base"]
                if "error" not in base:
                    for arm in ("pr", "bm64"):
                        a = row["arms"][arm]
                        if "error" in a:
                            continue
                        a["speedup_vs_base"] = base["median_ms"] / a["median_ms"]
                        a["bitwise_equal"] = bool(torch.equal(outs["base"], outs[arm]))
                        a["max_abs_diff"] = float(
                            (outs["base"].float() - outs[arm].float()).abs().max())
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                pr = row["arms"].get("pr", {})
                bm = row["arms"].get("bm64", {})
                if "speedup_vs_base" in pr:
                    print(f"{dname} hs={head_size:<3d} nq={nq:<3d} q={q_len:<6d} "
                          f"win={str(window):<5s} "
                          f"BM {base['block_m']}->{pr['block_m']} "
                          f"base={base['median_ms']:.3f} pr={pr['median_ms']:.3f} "
                          f"bm64={bm.get('median_ms', float('nan')):.3f} ms | "
                          f"pr x{pr['speedup_vs_base']:.3f} "
                          f"bm64 x{bm.get('speedup_vs_base', float('nan')):.3f} | "
                          f"{'bit-equal' if pr['bitwise_equal'] else 'DIFF %.3g' % pr['max_abs_diff']}",
                          flush=True)
                else:
                    print(f"{dname} hs={head_size} nq={nq} q={q_len} ROW {row}", flush=True)
    fh.close()
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
