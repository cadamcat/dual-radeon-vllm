"""Port of vllm#45450's spec-admission mechanism onto installed vLLM 0.28.0.

Mechanism 1 only: admit uniform spec-verify steps (query_len = 1 + num_spec)
into the Triton 3D flash-decoding path, sizing the softmax_segm buffers per
token. The window-relative segmentation (WINDOW_SEG_3D) is NOT ported:
sliding layers keep today's window-blind-but-consistent full-seq
segmentation, which is what they use for q=1 3D already.

Adds one probe-only instrumentation print (once per process) proving the
3D path actually ran with max_seqlen_q > 1.
"""
import pathlib


def patch(path, replacements):
    p = pathlib.Path(path)
    src = p.read_text()
    if "decode_query_len" in src:
        print("already injected:", path)
        return
    for old, new, count in replacements:
        assert src.count(old) == count, (
            f"anchor x{src.count(old)} (want {count}) in {path}:\n{old[:90]}")
        src = src.replace(old, new)
    compile(src, path, "exec")
    p.write_text(src)
    print("patched:", path)


import vllm.v1.attention.ops.triton_unified_attention as TUA
patch(TUA.__file__, [
    # signature: thread the new knob
    ("    seq_threshold_3D=None,\n",
     "    seq_threshold_3D=None,\n    decode_query_len: int = 1,\n", 1),
    # the 3D-admission guard
    ("        or max_seqlen_q > 1\n",
     "        or max_seqlen_q > decode_query_len\n", 1),
    # probe-only proof that 3D ran with q>1 (prints once per process)
    ("    # The kernel signature is the same for 2D and 3D",
     "    if (use_3d and max_seqlen_q > 1\n"
     "            and not getattr(unified_attention, '_p45450_logged', False)):\n"
     "        print('PROBE_3D_SPEC_ACTIVE max_seqlen_q=%d' % max_seqlen_q,\n"
     "              flush=True)\n"
     "        unified_attention._p45450_logged = True\n"
     "    # The kernel signature is the same for 2D and 3D", 1),
])

import vllm.v1.attention.backends.triton_attn as TA
patch(TA.__file__, [
    # metadata field (defaulted, placed at the head of the defaulted tail)
    ("    # Optional aot scheduling\n    scheduler_metadata:",
     "    decode_query_len: int = 1\n\n"
     "    # Optional aot scheduling\n    scheduler_metadata:", 1),
    # builder: derive decode_query_len from the speculative config
    ("        self.num_par_softmax_segments = NUM_PAR_SOFTMAX_SEGMENTS\n",
     "        spec_cfg = vllm_config.speculative_config\n"
     "        self.decode_query_len = 1 + (\n"
     "            spec_cfg.num_speculative_tokens\n"
     "            if spec_cfg is not None and spec_cfg.num_speculative_tokens\n"
     "            else 0\n"
     "        )\n"
     "        self.num_par_softmax_segments = NUM_PAR_SOFTMAX_SEGMENTS\n", 1),
    # segm buffers: first dim per token, not per sequence
    ("                self.seq_threshold_3D,\n",
     "                self.seq_threshold_3D * self.decode_query_len,\n", 1),
    ("            (self.seq_threshold_3D, self.num_heads_q, self.num_par_softmax_segments),\n",
     "            (self.seq_threshold_3D * self.decode_query_len, self.num_heads_q, self.num_par_softmax_segments),\n", 2),
    # build(): carry it on the metadata
    ("            seq_threshold_3D=self.seq_threshold_3D,\n",
     "            seq_threshold_3D=self.seq_threshold_3D,\n"
     "            decode_query_len=self.decode_query_len,\n", 1),
    # impl: extract and pass through
    ("        seq_threshold_3D = attn_metadata.seq_threshold_3D\n",
     "        seq_threshold_3D = attn_metadata.seq_threshold_3D\n"
     "        decode_query_len = getattr(attn_metadata, 'decode_query_len', 1)\n", 1),
    ("            seq_threshold_3D=seq_threshold_3D,\n",
     "            seq_threshold_3D=seq_threshold_3D,\n"
     "            decode_query_len=decode_query_len,\n", 1),
])

print("INJECT_OK")
