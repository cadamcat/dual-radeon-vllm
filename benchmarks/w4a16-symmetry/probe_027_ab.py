"""Hybrid vs the patched RDNA3 kernel, same asymmetric checkpoint, on gfx1100.

This is the comparison the 0.23.1 work could not make, because
RDNAHybridW4A16LinearKernel (vllm#40977, 2026-07-14) postdates that container.
On 0.27.0 both kernels are present, so the two arms differ only in which one
serves the layers:

  hybrid   stock image. uint4 fails RDNA3's type gate, Hybrid takes it.
  rdna3    the three-line patch. RDNA3 accepts uint4 and, being first in the
           ROCm priority list, takes it before Hybrid sees it.

    python3 probe_027_ab.py <hybrid|rdna3> <out.jsonl>

Every assumption is asserted rather than assumed, because a silent fallback
here would look exactly like a result. If the patch anchors have moved in
0.27.0 this aborts instead of measuring an unpatched arm twice.

Same prompt and timing method as the 0.23.1 probes, but the absolute numbers
are NOT comparable to those: different ROCm (7.14 -> 10.0), different vLLM
(0.23 -> 0.27). Only hybrid-vs-rdna3 within this run is meaningful.
"""

import ast
import json
import os
import pathlib
import sys
import time

ARM = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "/work/w4a16-027.jsonl"
MODEL = "/data/incoming/Qwen3.8-27B-AWQ-INT4"
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "16"))
assert ARM in ("hybrid", "rdna3"), ARM

import torch

import kernel_record

QUESTION = (
    "Answer in two or three sentences. What is the capital city of France, "
    "and what river runs through it?"
)


def apply_rdna3_patch():
    """The three lines under test. Aborts loudly if 0.27.0 has moved them."""
    import vllm.model_executor.kernels.linear.mixed_precision.rdna3_w4a16 as RK

    p = pathlib.Path(RK.__file__)
    src = p.read_text()

    old1 = "    SUPPORTED_QUANT_TYPES = [scalar_types.uint4b8]\n"
    assert src.count(old1) == 1, f"quant-type line: {src.count(old1)} matches"
    src = src.replace(
        old1, "    SUPPORTED_QUANT_TYPES = [scalar_types.uint4b8, scalar_types.uint4]\n")

    anchor = "        # Act-order: convert g_idx to the inverse permutation array exllama\n"
    assert src.count(anchor) == 1, f"act-order anchor: {src.count(anchor)} matches"
    src = src.replace(anchor,
        "        if c.zero_points:\n"
        "\n"
        "            def transform_w_zp(x):\n"
        "                assert isinstance(x, BasevLLMParameter)\n"
        "                permute_param_layout_(x, input_dim=0, output_dim=1,\n"
        "                                      packed_dim=1)\n"
        "                x.data = x.data.contiguous()\n"
        "                return x\n"
        "\n"
        "            self._transform_param(layer, self.w_zp_name, transform_w_zp)\n"
        "\n" + anchor)

    old3 = "        output = ops.gptq_gemm_rdna3(x_2d, w_q, w_zp, w_s, w_g_idx, False)\n"
    assert src.count(old3) == 1, f"apply line: {src.count(old3)} matches"
    src = src.replace(old3,
        "        output = ops.gptq_gemm_rdna3(x_2d, w_q, w_zp, w_s, w_g_idx,\n"
        "                                     not c.weight_type.has_bias())\n")

    p.write_text(src)
    ast.parse(src)
    txt = p.read_text()
    assert "scalar_types.uint4]" in txt and "transform_w_zp" in txt \
        and "has_bias())" in txt, "patch did not take"
    print("PATCH applied: all three anchors matched on 0.27.0", flush=True)


def main():
    import vllm
    print(f"ARM={ARM} vllm={vllm.__version__} model={MODEL}", flush=True)

    rec = kernel_record.install(f"027-{ARM}")
    if ARM == "rdna3":
        apply_rdna3_patch()

    from vllm import LLM, SamplingParams

    # max_num_seqs is pinned identically on both arms, but the value has to be
    # chosen per stack, not carried over: this model leaves 214 Mamba cache
    # blocks on 0.23.1 and only 23 on 0.27.1, so the 128 that worked there
    # aborts graph capture here. 16 clears the 0.27.1 floor with margin and is
    # irrelevant to a batch-of-1 decode measurement.
    llm = LLM(model=MODEL, tensor_parallel_size=2, max_model_len=1536,
              gpu_memory_utilization=0.92, max_num_seqs=MAX_NUM_SEQS)
    tok = llm.get_tokenizer()

    sp = SamplingParams(temperature=0, max_tokens=64, logprobs=1)
    out = llm.generate([QUESTION], sp)[0].outputs[0]
    lps = [next(iter(d.values())).logprob for d in (out.logprobs or []) if d]
    mean_lp = sum(lps) / len(lps) if lps else None
    answer = out.text

    ids = tok("hello " * 5000).input_ids[:1024]
    prompt = tok.decode(ids)

    def timed(n):
        s = SamplingParams(temperature=0, max_tokens=n, ignore_eos=True)
        t0 = time.time()
        o = llm.generate([prompt], s)
        return time.time() - t0, o[0].outputs[0].text

    timed(8)
    t8, _ = timed(8)
    t64, _ = timed(64)
    tps = (64 - 8) / (t64 - t8)

    chosen = kernel_record.read(rec)
    expect = "RDNA3W4A16LinearKernel" if ARM == "rdna3" else "RDNAHybridW4A16LinearKernel"
    got_expected = any(f"'{expect}'" in seg and "True)" in seg
                       for seg in chosen.split("|"))

    row = {"arm": ARM, "model": MODEL, "vllm": vllm.__version__,
           "decode_tok_s": tps, "t8": t8, "t64": t64,
           "mean_logprob": mean_lp, "n_logprobs": len(lps),
           "answer": answer, "kernels": chosen,
           "expected_kernel": expect, "expected_kernel_selected": got_expected,
           "max_num_seqs": MAX_NUM_SEQS,
           "note": "ROCm 10.0 / vLLM 0.27.1.dev5; not comparable to the "
                   "0.23.1 numbers (different ROCm, vLLM and max_num_seqs)"}
    with open(OUT, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"RESULT decode_tok_s={tps:.2f} mean_logprob="
          f"{'NA' if mean_lp is None else format(mean_lp, '.4f')} "
          f"expected_kernel_selected={got_expected}", flush=True)
    print(f"ANSWER {answer!r}", flush=True)
    print(f"KERNELS {chosen}", flush=True)
    assert got_expected, f"{expect} did not serve the layers; arm is invalid"
    print("CELL_DONE", flush=True)


if __name__ == "__main__":
    main()
