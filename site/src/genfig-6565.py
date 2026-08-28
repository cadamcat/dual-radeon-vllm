"""Figures for reporting-a-non-reproduction.html.

The arm tallies are parsed back out of the committed logs, the same way
verify_doc_figures.py parses them, so a stale results.json cannot agree with
prose the logs disagree with. The machine contrast is extracted from the
directory's README table.
"""
import hashlib, json, pathlib, re
R = pathlib.Path(__file__).resolve().parents[2]
D = R / "benchmarks" / "rccl-6565"
read = lambda *p: (D.joinpath(*p)).read_text(errors="replace")
ARM_RE = (r"=== arm=(\S+) RESULT pass=(\d+) fail=(\d+)(?: error=(\d+))? of (\d+)")


def tally(text):
    return [{"arm": m.group(1), "passed": int(m.group(2)), "failed": int(m.group(3)),
             "error": int(m.group(4) or 0), "n": int(m.group(5))}
            for m in re.finditer(ARM_RE, text)]


# ---- fig1: the eight arms, twice -----------------------------------------
ENV = {"default": "none",
       "p2pdisable": "NCCL_P2P_DISABLE=1",
       "prod": "NCCL_P2P_DISABLE=1 HSA_ENABLE_SDMA=0",
       "ch1": "+ NCCL_MIN/MAX_NCHANNELS=1",
       "ch4": "+ NCCL_MIN_NCHANNELS=4",
       "ch8": "+ NCCL_MIN_NCHANNELS=8",
       "ch16": "+ NCCL_MIN_NCHANNELS=16",
       "shmoff": "NCCL_SHM_DISABLE=1"}
first = tally(read("logs", "stage1.log")) + tally(read("logs", "stage2a.log"))
third = tally(read("logs", "stage3-allranks.log"))
by3 = {a["arm"]: a for a in third}
fig1 = {"arms": [dict(a, env=ENV[a["arm"]], cross=by3[a["arm"]]) for a in first],
        "sweeps": [
            {"id": "rank0", "date": "2026-08-27", "script": "rccl_allgather_truth.py",
             "verdict": "rank 0 only", "total": sum(a["n"] for a in first),
             "passed": sum(a["passed"] for a in first),
             "failed": sum(a["failed"] for a in first),
             "error": sum(a["error"] for a in first)},
            {"id": "allranks", "date": "2026-08-28",
             "script": "rccl_allgather_allranks.py", "verdict": "every rank",
             "total": sum(a["n"] for a in third),
             "passed": sum(a["passed"] for a in third),
             "failed": sum(a["failed"] for a in third),
             "error": sum(a["error"] for a in third)}],
        "same_arms_same_counts": sorted((a["arm"], a["n"]) for a in first)
                                 == sorted((a["arm"], a["n"]) for a in third),
        "cases_per_init": 12,
        "reporter_md5": hashlib.md5((D / "rccl_allgather_truth.py").read_bytes()).hexdigest(),
        "verbatim": (hashlib.md5((D / "rccl_allgather_truth.py").read_bytes()).hexdigest()
                     == "bffbc297cad9f1956c8bb2b7e8a4bb0f")}
s1 = read("logs", "stage1.log")
fig1["default_channels"] = 2 if ("Channel 00/02" in s1 and "Channel 01/02" in s1) else 0
fig1["rccl_version"] = re.search(r"RCCL version : (\S+)", s1).group(1)

# ---- fig2: the two machines, extracted from the directory's own table -----
md = read("README.md")


def table_after(heading):
    body = md.split(heading, 1)[1]
    rows = []
    for line in body.split("\n"):
        if line.startswith("|") and not re.match(r"^\|[\s|:-]+\|$", line):
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
        elif rows and not line.startswith("|"):
            break
    return rows


strip = lambda s: re.sub(r"\s+", " ", re.sub(r"[*`]", "", s)).strip()
raw = table_after("## The machine, against theirs")
fig2 = {"header": [strip(c) for c in raw[0]],
        "rows": [{"axis": strip(r[0]), "theirs": strip(r[1]), "ours": strip(r[2]),
                  # the rows the directory marks as not differing are the ones
                  # that stop the negative being explained away
                  "same": "same" in r[2].lower()}
                 for r in raw[1:]]}
fig2["same_rows"] = sum(1 for r in fig2["rows"] if r["same"])
env = read("logs", "environment.txt")
fig2["atomic_complaints"] = int(re.search(r"PCIE-atomic complaints: (\d+)", env).group(1))
fig2["reqen"] = env.count("AtomicOpsCtl: ReqEn+")

# ---- fig3: the blind spot, demonstrated rather than argued ----------------
bs = read("logs", "blindspot-check.log")
fig3 = {"ok": "BLINDSPOT_CHECK_OK" in bs,
        "injections": bs.count("injected into "),
        "rows": [
            {"who": "reporter", "script": "rccl_allgather_truth.py",
             "verdict": "ALL CORRECT", "exit": 0,
             "saw_it": False},
            {"who": "variant", "script": "rccl_allgather_allranks.py",
             "verdict": "1 FAILING CASES", "exit": 1, "saw_it": True},
            {"who": "runner", "script": "scripts/run6565_allranks.sh",
             "verdict": "pass=0 fail=1 error=0", "exit": 1, "saw_it": True}],
        "reporter_said_all_correct": bool(re.search(r"==> ALL CORRECT\s*\n\s*exit=0", bs)),
        "variant_counted_it": "RESULT pass=0 fail=1 error=0 of 1" in bs,
        "runner_exit": 1 if "runner exit=1" in bs else 0,
        "md5_after": hashlib.md5((D / "rccl_allgather_truth.py").read_bytes()).hexdigest()}
# the counting defect the first arm runner had, kept because it is the same
# class of mistake the stage exists to fix
fig3["grep_defect"] = {"reported_one_sided": 6, "of": 20,
                       "cause": "grep -c counts lines, and both ranks often print on one"}

out = {"_what": "Every figure in reporting-a-non-reproduction.html. Arm tallies "
                "parsed from benchmarks/rccl-6565/logs/, the machine contrast "
                "extracted from that directory's README. Derived by "
                "site/src/genfig-6565.py.",
       "fig1": fig1, "fig2": fig2, "fig3": fig3}
json.dump(out, open(pathlib.Path(__file__).parent / "figures-6565.json", "w"),
          ensure_ascii=False, indent=1)
print("fig1 arms:", len(fig1["arms"]), "sweeps:",
      [(s["id"], f'{s["passed"]}/{s["total"]}') for s in fig1["sweeps"]],
      "same arms/counts:", fig1["same_arms_same_counts"])
print("fig1 rccl", fig1["rccl_version"], "channels", fig1["default_channels"],
      "verbatim", fig1["verbatim"])
print("fig2 rows:", len(fig2["rows"]), "marked same:", fig2["same_rows"],
      "atomic complaints", fig2["atomic_complaints"], "ReqEn+", fig2["reqen"])
print("fig3:", {k: v for k, v in fig3.items() if k not in ("rows", "grep_defect")})
print("bytes:", len(json.dumps(out)))
