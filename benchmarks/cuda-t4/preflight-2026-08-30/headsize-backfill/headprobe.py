"""What head_size does the T4 run actually use, and would vllm#39018's gate fire?

The 2026-08-30 pre-flight recorded neither. Its README says gemma-4's head_dim
is 256, which is the config's local value; what decides the shared-memory demand
is what the kernel receives. #39018 gates on `head_size_padded >= 512`, so 256
against 512 is the difference between that PR covering this case and missing it.

Two independent readings:
  1. what vLLM's own ModelConfig resolves head_size to, before any kernel
  2. what `unified_attention` is actually called with, recorded from inside the
     worker right before the launch that then fails

Run before vllm is imported: it edits the installed kernel file.
"""
import glob, hashlib, json, os, pathlib, subprocess, sys, time

MODEL = "/content/models/gemma-4-12B-it-qat-w4a16-ct"
D = "/content/work"
OUT = f"{D}/headsize.jsonl"
REC = f"{D}/kernel-args.txt"


def emit(o):
    o["ts"] = round(time.time(), 1)
    with open(OUT, "a") as f:
        f.write(json.dumps(o) + "\n")
    print("EMIT", json.dumps(o), flush=True)


def main():
    assert "vllm" not in sys.modules, "vllm imported before the edit"
    SP = glob.glob("/usr/local/lib/python3*/dist-packages") + \
         glob.glob("/usr/lib/python3*/dist-packages")
    SP = [p for p in SP if os.path.exists(os.path.join(p, "vllm"))]
    assert len(SP) == 1, SP
    tua = pathlib.Path(SP[0], "vllm/v1/attention/ops/triton_unified_attention.py")
    md5 = hashlib.md5(tua.read_bytes()).hexdigest()
    print("triton_unified_attention.py md5", md5, tua.read_text().count("\n"), "lines", flush=True)

    # ---- probe 2: record what the kernel is called with, before it launches
    body = tua.read_text()
    ANCHOR = "    kernel_unified_attention[grid](\n"
    assert body.count(ANCHOR) == 1, body.count(ANCHOR)
    rec = (
        "    try:\n"
        "        import triton as _tr\n"
        "        _hsp = _tr.next_power_of_2(head_size)\n"
        "        _k = (head_size, _hsp, num_queries_per_kv, BLOCK_M, BLOCK_Q,\n"
        "              TILE_SIZE_PREFILL, TILE_SIZE_DECODE, tile_size, bool(use_3d),\n"
        "              int(max_seqlen_q), num_query_heads, num_kv_heads,\n"
        "              str(q.dtype), q.element_size())\n"
        "        import vllm.v1.attention.ops.triton_unified_attention as _m\n"
        "        _s = getattr(_m, '_seen_args', None)\n"
        "        if _s is None:\n"
        "            _s = set(); _m._seen_args = _s\n"
        "        if _k not in _s:\n"
        "            _s.add(_k)\n"
        "            with open('" + REC + "', 'a') as _fh:\n"
        "                _fh.write('pid=%d head_size=%d head_size_padded=%d nq_per_kv=%d "
        "BLOCK_M=%d BLOCK_Q=%d TILE_PREFILL=%d TILE_DECODE=%d tile=%d use_3d=%s "
        "max_seqlen_q=%d q_heads=%d kv_heads=%d dtype=%s esize=%d\\n' % ((os.getpid(),) + _k))\n"
        "    except Exception as _e:\n"
        "        with open('" + REC + ".err', 'a') as _fh:\n"
        "            _fh.write('%r\\n' % (_e,))\n"
    )
    if "_seen_args" not in body:
        tua.write_text(body.replace(ANCHOR, rec + ANCHOR, 1))
        print("recorder installed", flush=True)
    for pyc in glob.glob(SP[0] + "/vllm/**/__pycache__/*.pyc", recursive=True):
        os.remove(pyc)

    # ---- probe 1: what ModelConfig says, no GPU work at all
    import vllm  # noqa: E402
    from vllm.config import ModelConfig  # noqa: E402
    print("vllm", vllm.__version__, flush=True)
    mc = ModelConfig(model=MODEL, dtype="float16", max_model_len=33000,
                     limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0})
    arch = mc.model_arch_config
    row = {"kind": "model_arch_config", "vllm": vllm.__version__, "tua_md5": md5,
           "head_size": arch.head_size,
           "total_num_attention_heads": arch.total_num_attention_heads,
           "total_num_kv_heads": arch.total_num_kv_heads,
           "total_num_hidden_layers": arch.total_num_hidden_layers,
           "model_type": arch.model_type, "text_model_type": arch.text_model_type}
    # per-layer, if this build resolves them that way
    try:
        row["per_layer_head_sizes"] = sorted({arch[i].head_size
                                              for i in range(arch.total_num_hidden_layers)})
    except Exception as e:
        row["per_layer_head_sizes"] = f"n/a: {e!r}"
    emit(row)

    # what the raw config says, for contrast
    raw = json.load(open(MODEL + "/config.json"))
    tc = raw.get("text_config", raw)
    emit({"kind": "raw_config",
          "head_dim": tc.get("head_dim"), "global_head_dim": tc.get("global_head_dim"),
          "num_attention_heads": tc.get("num_attention_heads"),
          "num_key_value_heads": tc.get("num_key_value_heads"),
          "layer_types_counts": {k: (tc.get("layer_types") or []).count(k)
                                 for k in set(tc.get("layer_types") or [])},
          "model_type": raw.get("model_type")})
    print("PROBE1_DONE", flush=True)


if __name__ == "__main__":
    main()
