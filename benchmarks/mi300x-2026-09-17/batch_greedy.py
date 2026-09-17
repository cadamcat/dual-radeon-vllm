#!/usr/bin/env python3
"""The batch dimension and greedy determinism, in each model's own serve.

`runner.py` measures one request at a time, sixteen rungs, two rounds.
This is the second pass over the same six configurations: batch 1/2/4/8 at
three depths with five repeats, and — for the two models the RDNA3 W4A16
reading names — eight greedy repeats. It reuses that runner's serve, telemetry
and row writer by loading it as a module, so the only new code is the
concurrency and the greedy comparison.

Why a second serve per model rather than one session for both passes: the
ladder is a proven file and a timing campaign; adding a second driver to it
would put an untested branch inside the run that produces the comparable rows.
A reload costs about 2.5 minutes (measured: 0.24 h for six models on a B300).

Batch 1 is measured here too, in the same session as 2/4/8, so batch scaling
never crosses a session boundary. It is not the ladder's batch-1 number and is
not a substitute for it: different generation length, different session.

    BENCH_MACHINE=MI300X BENCH_WORK=/work BENCH_MODELS=/models \
    python3 batch_greedy_rocm.py [cfg,cfg,...]
"""
import hashlib, importlib.util, json, os, statistics as st, sys, threading, time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def load_runner(path=None):
    path = path or os.environ.get("BENCH_RUNNER") or os.path.join(HERE, "runner.py")
    spec = importlib.util.spec_from_file_location("runner", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = load_runner()

DEPTHS = [int(x) for x in os.environ.get("BENCH_DEPTHS", "500,8000,32000").split(",")]
BATCHES = [int(x) for x in os.environ.get("BENCH_BATCHES", "1,2,4,8").split(",")]
REPEATS = int(os.environ.get("BENCH_REPEATS", 5))
GEN = int(os.environ.get("BENCH_BATCH_GEN", 256))
MNS = int(os.environ.get("BENCH_MNS", 16))          # the Radeon serves' value
GREEDY_CFGS = os.environ.get("BENCH_GREEDY_CFGS", "B8,MG30").split(",")
GREEDY_REPEATS = int(os.environ.get("BENCH_GREEDY_REPEATS", 8))
GREEDY_DEPTH = int(os.environ.get("BENCH_GREEDY_DEPTH", 500))
GREEDY_GEN = int(os.environ.get("BENCH_GREEDY_GEN", 128))


def post(model, prompt, max_tokens, timeout, temperature, want_text=False):
    """runner.post with temperature exposed and the text kept.

    The ladder samples at 0.8 and throws the text away; a greedy repeat needs
    0 and the exact completion. Streaming and the usage block are the same, so
    a row from here is built from the same three quantities as a ladder row.
    """
    body = json.dumps({
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": temperature, "stream": True,
        "stream_options": {"include_usage": True}, "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(R.URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft, n, usage, chunks = None, 0, {}, []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                piece = (ch.get("delta") or {}).get("content") or ""
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    n += 1
                    if want_text:
                        chunks.append(piece)
    return ttft, n, time.time() - t0, usage, "".join(chunks)


def parallel(model, prompt, max_tokens, batch, timeout, temperature=0.8):
    """`batch` requests at once. Each thread returns its own row; the cell's
    wall clock is the span that covers all of them, not the sum."""
    out, errs = [None] * batch, [None] * batch
    def one(i):
        try:
            out[i] = post(model, prompt, max_tokens, timeout, temperature)
        except Exception as ex:                      # a failed request is a row, not a crash
            errs[i] = repr(ex)[:300]
    ts = [threading.Thread(target=one, args=(i,)) for i in range(batch)]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out, errs, time.time() - t0


def done_cells():
    done = set()
    if not os.path.exists(R.RES):
        return done
    for line in open(R.RES):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("kind") == "batch":
            done.add(("batch", r["cfg"], r["target"], r["batch"], r["round"]))
        if r.get("kind") == "greedy":
            done.add(("greedy", r["cfg"], r["round"]))
    return done


def served_config(cid):
    """The mml and max_num_seqs the ladder actually got this model started with.

    Two of the six carry a position cap and one has a Mamba pool smaller than
    the default; the ladder discovered those by retrying. Reusing what it found
    means this pass does not rediscover them at $1.99/h.
    """
    mml = mns = None
    if os.path.exists(R.RES):
        for line in open(R.RES):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("kind") == "model_meta" and r.get("cfg") == cid:
                mml, mns = r.get("mml") or mml, r.get("mns") or mns
    return mml, mns


def quant_kernel(cid):
    """The quantisation kernel vLLM chose, read out of the serve log as the
    2026-09-08 greedy A/B did. None means the log does not name one."""
    path = f"{R.D}/serve-{cid}.log"
    if not os.path.exists(path):
        return None
    names = set()
    for line in open(path, errors="replace"):
        for tok in line.split():
            if tok.endswith("LinearKernel") or tok.endswith("LinearKernel,"):
                names.add(tok.strip(",").strip("'\""))
    return sorted(names) or None


def run_cfg(cfg, done):
    cid = cfg["id"]
    mml, mns = served_config(cid)
    mml = mml or cfg.get("mml") or R.MML
    mns = max(MNS, max(BATCHES)) if not mns else min(mns, max(MNS, max(BATCHES)))
    st_, info = R.start_server(cfg, mml, mns)
    if st_ != "ready":
        R.log(f"{cid}: serve not ready ({st_}), skipping")
        R.emit({"kind": "config_failed", "cfg": cid, "why": f"batch pass: {st_}",
                "tail": str(info)[-800:]})
        return
    R.emit(R.meta_from(cid, info) | {"mml": mml, "mns": mns, "pass": "batch"})
    model = f"{R.MODELS}/{cfg['model']}"
    try:                                            # the discarded warm-up, as every runner here does
        post(model, "Say OK briefly.", 8, 180, 0.8)
        R.log(f"{cid}: warmup ok")
    except Exception as ex:
        R.emit({"kind": "note", "cfg": cid, "note": f"batch warmup failed: {ex!r}"[:200]})

    lad = {e["target"]: e for e in R.ladder_for(model, DEPTHS)}
    for target in DEPTHS:
        e = lad.get(target)
        if e is None:
            R.emit({"kind": "note", "cfg": cid, "note": f"no ladder entry for {target}"})
            continue
        if e["prompt_tokens"] + GEN + 100 > mml:
            R.emit({"kind": "note", "cfg": cid,
                    "note": f"depth {target} exceeds mml {mml}, skipped in the batch pass"})
            continue
        for batch in BATCHES:
            for rnd in range(1, REPEATS + 1):
                if ("batch", cid, target, batch, rnd) in done:
                    continue
                smp = R.Sampler()
                with smp:
                    rows, errs, wall = parallel(model, e["text"], GEN, batch, 900)
                # the row keeps the wall clock the rate was computed from, so
                # the cell's throughput recomputes from the row itself
                wall = round(wall, 4)
                tele = dict(smp.result, wall_s=wall)
                okrows = [r for r in rows if r]
                for i, r in enumerate(rows):
                    if r is None:
                        R.emit({"kind": "batch_request", "cfg": cid, "machine": R.MACHINE,
                                "target": target, "batch": batch, "round": rnd, "idx": i,
                                "err": errs[i]})
                        continue
                    ttft, n, w, usage, _ = r
                    R.emit({"kind": "batch_request", "cfg": cid, "machine": R.MACHINE,
                            "target": target, "batch": batch, "round": rnd, "idx": i,
                            "prompt_tokens": usage.get("prompt_tokens", e["prompt_tokens"]),
                            "gen_tokens": n, "ttft": round(ttft or 0, 4), "wall_s": round(w, 3),
                            "decode_tps": round((n - 1) / (w - ttft), 4)
                            if ttft and w > ttft and n > 1 else 0.0})
                gen_total = sum(r[1] for r in okrows)
                R.emit({"kind": "batch", "cfg": cid, "machine": R.MACHINE, "target": target,
                        "batch": batch, "round": rnd, "requests": batch,
                        "requests_ok": len(okrows), "gen_tokens": gen_total,
                        "ttft_median": round(st.median([r[0] or 0 for r in okrows]), 4) if okrows else 0,
                        "ttft_max": round(max([r[0] or 0 for r in okrows]), 4) if okrows else 0,
                        # the cell's throughput is the batch's tokens over the
                        # batch's wall clock, not the sum of per-request rates
                        "decode_tps_cell": round(gen_total / wall, 4) if wall else 0.0} | tele)
                R.log(f"{cid}: {target} b{batch} r{rnd} {gen_total}/{wall:.1f}s "
                      f"= {gen_total / wall if wall else 0:.1f} tok/s")

    if cid in GREEDY_CFGS:
        e = R.ladder_for(model, [GREEDY_DEPTH])[0]
        seen = []
        for rnd in range(1, GREEDY_REPEATS + 1):
            if ("greedy", cid, rnd) in done:
                continue
            try:
                ttft, n, w, usage, text = post(model, e["text"], GREEDY_GEN, 900, 0.0, want_text=True)
            except Exception as ex:
                R.emit({"kind": "greedy", "cfg": cid, "round": rnd, "err": repr(ex)[:300]})
                continue
            h = hashlib.sha1(text.encode("utf-8")).hexdigest()
            seen.append(h)
            R.emit({"kind": "greedy", "cfg": cid, "machine": R.MACHINE, "round": rnd,
                    "target": GREEDY_DEPTH, "gen_tokens": n, "sha1": h,
                    "chars": len(text), "wall_s": round(w, 3),
                    "quant_kernel": quant_kernel(cid)})
            R.log(f"{cid}: greedy r{rnd} {n} tokens {h[:12]}")
        if seen:
            R.emit({"kind": "greedy_summary", "cfg": cid, "machine": R.MACHINE,
                    "rounds": len(seen), "distinct": len(set(seen)),
                    "deterministic": len(set(seen)) == 1,
                    "quant_kernel": quant_kernel(cid)})
    R.emit({"kind": "config_complete", "cfg": cid, "pass": "batch"})


def main(which=None):
    R.preflight_stack()
    cfgs = [c for c in R.CFGS if not which or c["id"] in which]
    R.log(f"batch/greedy pass over {[c['id'] for c in cfgs]}: "
          f"depths {DEPTHS} batches {BATCHES} repeats {REPEATS}")
    done = done_cells()
    for cfg in cfgs:
        run_cfg(cfg, done)
    R.log("batch/greedy pass done")


if __name__ == "__main__":
    want = (sys.argv[1].split(",") if len(sys.argv) > 1
            else (os.environ["BENCH_CFGS"].split(",") if os.environ.get("BENCH_CFGS") else None))
    main(want)
