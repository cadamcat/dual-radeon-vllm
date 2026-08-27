"""The three-line fix, tested: asymmetric int4 on the native RDNA3 kernel.

`probe_w4a16_forced.py` showed that admitting `uint4` flips kernel selection
and then dies in the HIP entry check, because the checkpoint's zero-point
tensor is the transpose of what the kernel reads. Reading the kernel source
turns up a second thing the Python side never uses:

    const int zero_offset = use_v2_format ? 0 : 1;      q_gemm_rdna3.cu:668

`use_v2_format` is already a parameter of `gptq_gemm_rdna3`, and it selects
between the two zero-point conventions. `apply_weights` hard-codes `False`,
i.e. GPTQv1, where the kernel adds 1 and the symmetric path therefore stores
`bias - 1`. compressed-tensors stores the *true* zero point, which is exactly
what `use_v2_format=True` expects. So no zero-point arithmetic is needed, and
in particular no "subtract one" — which matters, because this checkpoint does
use a zero point of 0 (22 occurrences in 22.3M sampled entries) and 0 has no
representation under the GPTQv1 convention.

Three changes, all Python, all in `rdna3_w4a16.py`:

  1. admit `uint4` to `SUPPORTED_QUANT_TYPES`
  2. permute `w_zp` to the group-major layout, the way `w_s` already is
  3. pass `use_v2_format=c.zero_points` instead of a hard-coded `False`

    python3 probe_w4a16_fix.py <fixed|layout_only> <out.jsonl>

`layout_only` applies 1 and 2 but leaves `use_v2_format=False`. It is the
control that shows the convention actually matters: if it scores as well as
`fixed`, then step 3 is not doing anything and the reasoning above is wrong.

Scored against `probe_w4a16_forced.py`'s stock arm on the same checkpoint and
prompt: 11.41 tok/s, mean logprob -0.1859, correct answer.
"""

import ast
import json
import pathlib
import sys
import time

ARM = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "/work/w4a16-fix.jsonl"
MODEL = "/data/incoming/Qwen3.8-27B-AWQ-INT4"
assert ARM in ("fixed", "layout_only"), ARM

import torch
import vllm.model_executor.kernels.linear as LK
import vllm.model_executor.kernels.linear.mixed_precision.rdna3_w4a16 as RK
from vllm import LLM, SamplingParams

QUESTION = (
    "Answer in two or three sentences. What is the capital city of France, "
    "and what river runs through it?"
)


def patch_kernel_record():
    kpath = pathlib.Path(LK.__file__)
    ksrc = kpath.read_text()
    anchor = "        can_implement, failure_reason = kernel.can_implement(config)\n"
    assert ksrc.count(anchor) == 1
    rec = f"/work/kernels-fix-{ARM}.txt"
    probe = (
        "        try:\n"
        "            import os as _os\n"
        "            _ok, _why = kernel.can_implement(config)\n"
        "            _k = (kernel.__name__, str(config.weight_type),\n"
        "                  config.group_size, bool(config.zero_points), bool(_ok))\n"
        "            _s = getattr(choose_mp_linear_kernel, '_seen', None)\n"
        "            if _s is None:\n"
        "                _s = set(); choose_mp_linear_kernel._seen = _s\n"
        "            if _k not in _s:\n"
        "                _s.add(_k)\n"
        "                with open('" + rec + "', 'a') as _fh:\n"
        "                    _fh.write('pid=%d %s %s\\n' % (_os.getpid(), _k,\n"
        "                              ('' if _ok else (_why or '')[:70])))\n"
        "        except Exception:\n"
        "            pass\n"
    )
    kpath.write_text(ksrc.replace(anchor, probe + anchor))
    ast.parse(kpath.read_text())
    return rec


def apply_fix():
    p = pathlib.Path(RK.__file__)
    src = p.read_text()

    # (1) admit uint4
    old1 = "    SUPPORTED_QUANT_TYPES = [scalar_types.uint4b8]\n"
    new1 = "    SUPPORTED_QUANT_TYPES = [scalar_types.uint4b8, scalar_types.uint4]\n"
    assert src.count(old1) == 1
    src = src.replace(old1, new1)

    # (2) give w_zp the group-major layout w_s already gets. An asymmetric
    # checkpoint ships (N/8, groups); the kernel reads [groups, N/8]. The
    # symmetric branch above never needs this because it builds the tensor
    # itself, already group-major.
    anchor2 = "        # Act-order: convert g_idx to the inverse permutation array exllama\n"
    assert src.count(anchor2) == 1
    ins = (
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
        "\n"
    )
    src = src.replace(anchor2, ins + anchor2)

    # (3) pick the zero-point convention from the checkpoint instead of
    # hard-coding GPTQv1. Skipped on the layout_only arm, which is the control.
    old3 = "        output = ops.gptq_gemm_rdna3(x_2d, w_q, w_zp, w_s, w_g_idx, False)\n"
    assert src.count(old3) == 1
    if ARM == "fixed":
        new3 = ("        # compressed-tensors stores true zero points (v2); the\n"
                "        # synthesized symmetric ones use the GPTQv1 +1 convention.\n"
                "        output = ops.gptq_gemm_rdna3(x_2d, w_q, w_zp, w_s, w_g_idx,\n"
                "                                     c.zero_points)\n")
        src = src.replace(old3, new3)

    p.write_text(src)
    ast.parse(p.read_text())
    txt = p.read_text()
    assert "scalar_types.uint4]" in txt
    assert "transform_w_zp" in txt
    if ARM == "fixed":
        assert "w_g_idx,\n                                     c.zero_points)" in txt
    print(f"PATCH applied arm={ARM} "
          f"(v2_format={'c.zero_points' if ARM == 'fixed' else 'False (control)'})",
          flush=True)


def main():
    rec = patch_kernel_record()
    apply_fix()
    print(f"ARM={ARM} model={MODEL}", flush=True)

    llm = LLM(model=MODEL, tensor_parallel_size=2, max_model_len=1536,
              gpu_memory_utilization=0.92, max_num_seqs=128)
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

    chosen = ""
    if pathlib.Path(rec).exists():
        chosen = " | ".join(sorted(set(pathlib.Path(rec).read_text().splitlines())))

    row = {"arm": ARM, "model": MODEL, "decode_tok_s": tps, "t8": t8, "t64": t64,
           "mean_logprob": mean_lp, "n_logprobs": len(lps),
           "answer": answer, "kernels": chosen}
    with open(OUT, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"RESULT decode_tok_s={tps:.2f} mean_logprob="
          f"{'NA' if mean_lp is None else format(mean_lp, '.4f')}", flush=True)
    print(f"ANSWER {answer!r}", flush=True)
    print(f"KERNELS {chosen}", flush=True)
    print("FIX_CELL_DONE", flush=True)


if __name__ == "__main__":
    main()
