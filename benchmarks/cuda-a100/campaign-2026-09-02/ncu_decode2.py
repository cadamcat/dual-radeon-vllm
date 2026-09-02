"""Achieved DRAM bandwidth of a decode step. Second attempt.

The first used `--target-processes all` and profiled nothing: vLLM's V1 engine
runs in a child process, ncu did not follow it, and the workload finished in
1.24 s -- far too fast to have been replayed. VLLM_ENABLE_V1_MULTIPROCESSING=0
keeps the engine in this process, which is what ncu can profile.

Output goes to a log file rather than stdout: the CSV carries full demangled
kernel names, and vLLM's own logging is interleaved on stdout.

    python3 ncu_decode2.py <model-dir>
"""
import csv, io, os, subprocess, sys
MODEL = sys.argv[1]; GEN = int(sys.argv[2]) if len(sys.argv) > 2 else 8
LOG = "/content/work/ncu-" + os.path.basename(MODEL) + f"-gen{GEN}.csv"
env = dict(os.environ, VLLM_ENABLE_V1_MULTIPROCESSING="0", VLLM_LOGGING_LEVEL="WARNING")
cmd = ["ncu", "--csv", "--log-file", LOG, "--nvtx", "--nvtx-include", "decode/",
       "--metrics", "dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum",
       sys.executable, "/content/work/decode_step.py", MODEL, str(GEN)]
r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, env=env)
print("rc", r.returncode, "| stdout tail:", " ".join((r.stdout or "").splitlines()[-2:])[:200], flush=True)
if r.returncode != 0:
    print("STDERR:", (r.stderr or "")[-1200:], flush=True)
if not os.path.exists(LOG):
    print("NCU_NO_LOG"); sys.exit(1)
txt = open(LOG).read(); i = txt.find('"ID"')
if i < 0:
    print("NCU_NO_CSV"); print(txt[-1500:]); sys.exit(1)
rows = list(csv.DictReader(io.StringIO(txt[i:])))
rd = wr = t = 0.0; names = {}
SC = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9,
      "nsecond": 1e-9, "usecond": 1e-6, "msecond": 1e-3, "second": 1,
      "ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1}   # ncu writes the short forms
for row in rows:
    m, u = row.get("Metric Name"), row.get("Metric Unit", "")
    try: v = float((row.get("Metric Value") or "0").replace(",", ""))
    except ValueError: continue
    s = SC.get(u)
    if s is None: continue
    if m == "dram__bytes_read.sum":
        rd += v * s; k = (row.get("Kernel Name") or "?")[:44]; names[k] = names.get(k, 0) + v * s
    elif m == "dram__bytes_write.sum": wr += v * s
    elif m == "gpu__time_duration.sum": t += v * s
print(f"NCU2 model={os.path.basename(MODEL)} gen={GEN} rows={len(rows)} distinct_kernels={len(names)} "
      f"read_GB={rd/1e9:.3f} write_GB={wr/1e9:.3f} kernel_s={t:.5f} "
      f"achieved_GBs={(rd+wr)/t/1e9 if t else 0:.1f} read_GB_per_token={rd/1e9/GEN:.4f}", flush=True)
for k, v in sorted(names.items(), key=lambda kv: -kv[1])[:5]:
    print(f"   {v/1e9:8.3f} GB  {k}", flush=True)
