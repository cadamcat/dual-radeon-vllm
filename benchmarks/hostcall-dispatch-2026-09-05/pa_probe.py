#!/usr/bin/env python3
"""Is vLLM's own paged-attention kernel refused where PCIe AtomicOps are absent?

C1 counted 348 kernels in vllm/_rocm_C that declare hidden_hostcall_buffer,
all one family: paged_attention_ll4mi_QKV_mfma4_kernel. Qwen3-8B (head 128,
gqa 4, block 16) satisfies use_rocm_custom_paged_attention on gfx1100 and this
box's serve logs select ROCM_ATTN for it with no Triton fallback -- but it has
only ever been served here with AtomicOps present. This probe dispatches that
kernel directly, one arm per process, on whatever platform it is run on.

    pa_probe.py --arm {as_shipped,ck_forced,triton_forced} --row LABEL --out X.jsonl
                [--caps N --dmesg N --host H --image I]

Runs INSIDE the vLLM 0.23 container. Modelled on vllm-50603/probe_53856.py:
the CK path is entered through vLLM's own chunked_prefill_paged_decode with
use_rocm_custom_paged_attention forced.

What the row proves, and how (revised after TASK-0007's review):
  * which kernel ran: the C++ op torch.ops._rocm_C.paged_attention is wrapped
    (ck_op_calls), not only the Python helper ops.paged_attention_rocm
    (ck_helper_calls); vLLM's "falling back to Triton" warning is captured
    from its logger during the arm (fallback_warning_seen);
  * refusal, read three ways because a custom op's launch failure need not
    raise (hipgate3.cpp saw launch and sync both succeed on bare metal, only
    hipGetLastError reporting): the exception text, hipGetLastError after a
    synchronize, and the output buffer, pre-filled with NaN;
  * the platform's own answer, hipDeviceAttributeHostNativeAtomicSupported on
    EVERY device, from hipattr.cpp compiled in the container (the compiler
    resolves the enum; a separate process, so nothing sticks to this one);
  * provenance: vllm/torch/hip versions, the md5 of the _rocm_C extension,
    the container's hostname, and the guest host and image id passed in.

Outcomes (one per row; `refused` only on the runtime's own refusal strings):
  dispatched_ok      output written, finite, max_rel_err <= REL_OK
  dispatched_wrong   output written, finite, max_rel_err >  REL_OK
  dispatched_nan     output written, not finite
  refused            exception or hipGetLastError names the hostcall refusal
                     (hipErrorIllegalState / "present state" / "Pcie atomics
                     not enabled" / "hostcall not supported")
  not_launched       no exception, hipGetLastError clean, output untouched, and
                     neither the helper nor the C++ op was entered
  launched_no_output the op was entered, no exception, hipGetLastError clean,
                     and the output is untouched -- hipgate3's bare-metal shape
  dispatch_error     some other exception or HIP error from the dispatch
  gate_declined      as_shipped only: the shipped gate did not admit the shape
  harness_error      build / reference / readback failed; nothing measured
"""
import argparse, ctypes, hashlib, json, logging, os, re, socket, subprocess, sys, time

HEAD_SIZE, BLOCK_SIZE = 128, 16
REL_OK = 2e-2                       # probe_53856 measured ~3e-3 for this shape
REFUSAL_STRINGS = ("the operation cannot be performed in the present state",
                   "hipErrorIllegalState", "Pcie atomics not enabled",
                   "hostcall not supported")
HIP_ERROR_ILLEGAL_STATE = 401


def hipattr(path="./hipattr"):
    """The platform's own answer, from a separate process: hipattr.cpp resolves
    hipDeviceAttributeHostNativeAtomicSupported with the compiler and prints
    both the attribute query and the device-properties field, per device.
    Returns (devices, raw). devices is None if the helper did not finish."""
    try:
        r = subprocess.run([path], capture_output=True, text=True, timeout=60)
    except Exception as e:                              # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"[:160]
    devs = []
    for line in r.stdout.splitlines():
        m = re.match(r"device=(\d+) attr_rc=(-?\d+) attr=(-?\d+) props_rc=(-?\d+) props=(-?\d+) name=(.*?) arch=(\S+)$", line)
        if m:
            devs.append({"device": int(m[1]), "attr_rc": int(m[2]), "attr": int(m[3]), "props_rc": int(m[4]),
                         "props": int(m[5]), "name": m[6], "arch": m[7]})
    done = "HIPATTR_DONE" in r.stdout
    return (devs if done else None), (r.stdout[-400:] + r.stderr[-200:])


def hip_from_maps():
    """ctypes handle on the libamdhip64 torch has ALREADY mapped (dlopen of a
    loaded path returns the same object), so no second HIP runtime is loaded.
    The container ships three copies, and a bare-name CDLL picked the wrong one
    on the first run (row 1, 2026-09-05 07:34)."""
    for line in open("/proc/self/maps"):
        if "libamdhip64" in line:
            path = line.split()[-1]
            lib = ctypes.CDLL(path)
            lib.hipGetLastError.argtypes = []
            lib.hipGetLastError.restype = ctypes.c_int
            lib.hipGetErrorString.argtypes = [ctypes.c_int]
            lib.hipGetErrorString.restype = ctypes.c_char_p
            return lib, path
    return None, None


def last_error(lib):
    if lib is None:
        return None, None
    code = lib.hipGetLastError()
    return int(code), lib.hipGetErrorString(code).decode(errors="replace")


def build(torch, ctx_len, num_heads, num_kv, bs=2):
    try:
        from vllm.utils.torch_utils import set_random_seed
    except ImportError:
        try:
            from vllm.utils import set_random_seed
        except ImportError:
            set_random_seed = torch.manual_seed
    set_random_seed(0)
    dt = torch.bfloat16
    blocks = (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    cache_size = blocks * bs + 8
    query = torch.randn(bs, num_heads, HEAD_SIZE, dtype=dt)
    kv = torch.randn(bs, ctx_len, 2, num_kv, HEAD_SIZE, dtype=dt)
    key, value = kv.unbind(dim=2)
    shape = (cache_size, BLOCK_SIZE, num_kv, HEAD_SIZE)
    k_cache = torch.zeros(shape, dtype=dt)
    v_cache = torch.zeros(shape, dtype=dt)
    perm = torch.randperm(cache_size)[: blocks * bs].to(torch.int32)
    block_table = perm.view(bs, blocks)
    fk, fv = k_cache.view(-1, num_kv, HEAD_SIZE), v_cache.view(-1, num_kv, HEAD_SIZE)
    for i in range(bs):
        for b in range(blocks):
            lo, hi = b * BLOCK_SIZE, min((b + 1) * BLOCK_SIZE, ctx_len)
            slot = int(block_table[i, b]) * BLOCK_SIZE
            fk[slot: slot + (hi - lo)].copy_(key[i, lo:hi])
            fv[slot: slot + (hi - lo)].copy_(value[i, lo:hi])
    k_cache = (k_cache.view(-1, BLOCK_SIZE, num_kv, HEAD_SIZE // 8, 8)
               .permute(0, 2, 3, 1, 4).contiguous())
    v_cache = (v_cache.view(-1, BLOCK_SIZE, num_kv, HEAD_SIZE).permute(0, 2, 3, 1).contiguous())
    s = torch.tensor(1.0, dtype=torch.float32)
    return dict(query=query, key=key, value=value, k_cache=k_cache, v_cache=v_cache,
                block_table=block_table,
                seq_lens=torch.full((bs,), ctx_len, dtype=torch.int32),
                query_start_loc=torch.arange(bs + 1, dtype=torch.int32),
                k_scale=s, v_scale=s, ctx_len=ctx_len, bs=bs)


def reference(torch, d):
    q, k, v = d["query"].float(), d["key"].float(), d["value"].float()
    bs, H, D = q.shape
    rep = H // k.shape[2]
    out = torch.empty(bs, H, D, dtype=torch.float32)
    for i in range(bs):
        for h in range(H):
            kh = h // rep
            p = torch.softmax((k[i, :, kh, :] @ q[i, h]) * D ** -0.5, dim=0)
            out[i, h] = p @ v[i, :, kh, :]
    return out


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(); self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("as_shipped", "ck_forced", "triton_forced"))
    ap.add_argument("--row", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--caps", type=int, default=None, help="root ports with completer support, read by the wrapper")
    ap.add_argument("--dmesg", type=int, default=None, help="'PCIE atomic ops is not supported' lines, read by the wrapper")
    ap.add_argument("--host", default=None, help="guest hostname, from the wrapper")
    ap.add_argument("--image", default=None, help="container image and id, from the wrapper")
    ap.add_argument("--vm", default=None, help="VM identity label, from the wrapper (e.g. 101@pve)")
    ap.add_argument("--product-uuid", default=None, help="/sys/class/dmi/id/product_uuid, read by the wrapper")
    ap.add_argument("--profile", action="store_true", help="record the device kernel names the dispatch ran, via torch.profiler")
    a = ap.parse_args()
    gqa = a.heads // a.kv_heads
    row = {"kind": "pa_cell", "row": a.row, "arm": a.arm, "ts": time.time(),
           "head_size": HEAD_SIZE, "block_size": BLOCK_SIZE, "gqa_ratio": gqa, "ctx_len": a.ctx,
           "root_ports_with_completer_support": a.caps, "dmesg_no_atomics_lines": a.dmesg,
           "guest_host": a.host, "image": a.image, "container_host": socket.gethostname(),
           "vm": a.vm, "product_uuid": a.product_uuid,
           "outcome": None}

    def finish(outcome, **extra):
        row["outcome"] = outcome
        row.update(extra)
        with open(a.out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print("row=%s arm=%s outcome=%s gate_as_shipped=%s ck_op_calls=%s fallback_warning=%s "
              "host_native_atomic=%s hip_last_error=%s exc=%s rel_err=%s" % (
                  a.row, a.arm, outcome, row.get("gate_as_shipped"), row.get("ck_op_calls"),
                  row.get("fallback_warning_seen"), row.get("host_native_atomic_supported"),
                  row.get("hip_last_error_text"), (row.get("exception") or "")[:120], row.get("max_rel_err")),
              flush=True)
        print("PA_PROBE_DONE", flush=True)

    devs, raw = hipattr(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hipattr"))
    row["hipattr_devices"] = devs
    row["hipattr_raw"] = raw[-300:]
    row["hipattr_ok"] = bool(devs) and len(devs) == 2 and all(d["attr_rc"] == 0 and d["props_rc"] == 0 for d in devs)
    row["host_native_atomic_supported"] = [d["attr"] for d in devs] if row["hipattr_ok"] else None
    row["host_native_atomic_props"] = [d["props"] for d in devs] if row["hipattr_ok"] else None
    if row["hipattr_ok"] and a.caps is not None:
        want = 1 if a.caps > 0 else 0
        row["attr_matches_platform"] = all(d["attr"] == want and d["props"] == want for d in devs)
    else:
        row["attr_matches_platform"] = None
    lib = None

    try:
        import torch
        import vllm
        import vllm._custom_ops as ops
        import vllm.platforms.rocm as rp
        from vllm.v1.attention.ops.chunked_prefill_paged_decode import chunked_prefill_paged_decode
        torch.set_default_device("cuda")
        row["vllm"] = vllm.__version__
        row["torch"] = torch.__version__
        row["hip"] = getattr(torch.version, "hip", None)
        row["device"] = torch.cuda.get_device_name(0)
        row["device_count"] = torch.cuda.device_count()
        row["ran_on_device"] = 0
        lib, row["hip_library"] = hip_from_maps()
        try:
            import vllm._rocm_C as _rc
            row["rocm_C_path"] = _rc.__file__
            row["rocm_C_md5"] = md5_of(_rc.__file__)
        except Exception as e:                          # noqa: BLE001
            row["rocm_C_path"], row["rocm_C_md5"] = None, f"unavailable: {type(e).__name__}: {e}"[:120]
    except Exception as e:                              # noqa: BLE001
        return finish("harness_error", exception=f"import: {type(e).__name__}: {e}"[:300])

    # --- attribution: wrap the C++ op itself, and the Python helper, and vLLM's logger
    calls = {"op": 0, "helper": 0}
    orig_gate, orig_helper = rp.use_rocm_custom_paged_attention, ops.paged_attention_rocm

    def counting_helper(*args, **kw):
        calls["helper"] += 1
        return orig_helper(*args, **kw)
    ops.paged_attention_rocm = counting_helper
    op_wrapped = False

    class CountingOp:
        """Stands in for the OpOverloadPacket: calls count, and attributes such as
        .default resolve to the original's, wrapped so those count too."""
        def __init__(self, orig):
            self._orig = orig

        def __call__(self, *args, **kw):
            calls["op"] += 1
            return self._orig(*args, **kw)

        def __getattr__(self, name):
            attr = getattr(self._orig, name)
            if callable(attr):
                def wrapped(*args, **kw):
                    calls["op"] += 1
                    return attr(*args, **kw)
                return wrapped
            return attr

    try:
        rocm_ops = torch.ops._rocm_C
        orig_op = rocm_ops.paged_attention
        rocm_ops.paged_attention = CountingOp(orig_op)
        op_wrapped = True
    except Exception as e:                              # noqa: BLE001
        row["ck_op_wrap_error"] = f"{type(e).__name__}: {e}"[:160]
    row["ck_op_wrapped"] = op_wrapped
    cap = _Capture()
    logging.getLogger("vllm").addHandler(cap)

    try:
        row["gate_as_shipped"] = bool(orig_gate(torch.bfloat16, HEAD_SIZE, BLOCK_SIZE, gqa, a.ctx, 0, "auto", None, None))
    except Exception as e:                              # noqa: BLE001
        return finish("harness_error", exception=f"gate: {type(e).__name__}: {e}"[:300])
    force = {"as_shipped": None, "ck_forced": True, "triton_forced": False}[a.arm]
    if force is not None:
        rp.use_rocm_custom_paged_attention = lambda *args, **kw: force
    if a.arm == "as_shipped" and not row["gate_as_shipped"]:
        return finish("gate_declined")

    try:
        d = build(torch, a.ctx, a.heads, a.kv_heads)
        ref = reference(torch, d).cpu()
        out = torch.full_like(d["query"], float("nan"))     # stays NaN if nothing ran
        torch.cuda.synchronize()                             # setup fully retired before the clear
    except Exception as e:                              # noqa: BLE001
        return finish("harness_error", exception=f"build/reference: {type(e).__name__}: {e}"[:300])

    pre_code, pre_text = last_error(lib)                # clear, as hipgate3 does before launch
    row["hip_error_before_dispatch"] = pre_code
    if pre_code not in (None, 0):
        return finish("harness_error", exception=f"HIP error pending before dispatch: {pre_code}:{pre_text}")
    # --profile: which device kernels the dispatch actually ran. This wheel's HIP
    # runtime prints no ShaderName line at any AMD_LOG_LEVEL (measured 2026-09-05),
    # so kernel names come from torch.profiler, the way prof-31b.py got them here.
    prof = None
    if a.profile:
        try:
            from torch.profiler import profile, ProfilerActivity
            prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            prof.__enter__()
        except Exception as e:                          # noqa: BLE001
            row["profile_error"] = f"{type(e).__name__}: {e}"[:160]
            prof = None
    exc = None
    try:
        chunked_prefill_paged_decode(
            query=d["query"], key=None, value=None, output=out, kv_cache_dtype="auto",
            key_cache=d["k_cache"], value_cache=d["v_cache"],
            block_table=d["block_table"], query_start_loc=d["query_start_loc"],
            seq_lens=d["seq_lens"], max_seq_len=d["ctx_len"], max_query_len=1,
            k_scale=d["k_scale"], v_scale=d["v_scale"])
        torch.cuda.synchronize()
    except Exception as e:                              # noqa: BLE001
        exc = f"{type(e).__name__}: {e}"[:300]
    if prof is not None:
        try:
            prof.__exit__(None, None, None)
            kernels = []
            for ev in prof.key_averages():
                dt = None
                for attr in ("self_device_time_total", "self_cuda_time_total", "device_time_total", "cuda_time_total"):
                    if hasattr(ev, attr):
                        dt = getattr(ev, attr); break
                if dt and dt > 0 and not ev.key.startswith(("aten::", "cuda", "hip", "Memcpy", "Memset")):
                    kernels.append({"name": ev.key[:200], "count": ev.count, "device_us": round(float(dt), 1)})
            kernels.sort(key=lambda k: -k["device_us"])
            row["profiled_kernels"] = kernels[:12]
            row["profiled_kernel_families"] = sorted({k["name"].split("<")[0].split("(")[0] for k in kernels})
        except Exception as e:                          # noqa: BLE001
            row["profile_error"] = f"{type(e).__name__}: {e}"[:160]
    code, text = last_error(lib)
    row.update(ck_op_calls=calls["op"], ck_helper_calls=calls["helper"], exception=exc,
               hip_last_error=code, hip_last_error_text=text,
               fallback_warning_seen=any("falling back to Triton" in l for l in cap.lines),
               vllm_log_lines=[l[:160] for l in cap.lines[:8]])
    try:
        o = out.float().cpu()
        written = not bool(torch.isnan(o).all())
        finite = bool(torch.isfinite(o).all())
        rel = (float((o - ref).abs().max()) / float(ref.abs().max())) if finite else None
    except Exception as e:                              # noqa: BLE001
        return finish("harness_error", exception=(exc or "") + f" | readback: {type(e).__name__}: {e}"[:200])
    row.update(output_written=written, all_finite=finite, max_rel_err=rel)
    msg = " ".join(x for x in (exc or "", text or "") if x)
    if any(s in msg for s in REFUSAL_STRINGS) or code == HIP_ERROR_ILLEGAL_STATE:
        return finish("refused")
    if exc or (code not in (None, 0)):
        return finish("dispatch_error")
    if not written:
        return finish("launched_no_output" if calls["op"] > 0 or calls["helper"] > 0 else "not_launched")
    if not finite:
        return finish("dispatched_nan")
    return finish("dispatched_ok" if rel <= REL_OK else "dispatched_wrong")


if __name__ == "__main__":
    main()
