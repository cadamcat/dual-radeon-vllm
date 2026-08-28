"""Figures for the RCCL article, extracted from docs/root-cause.md so the two
cannot drift. The elimination table is parsed rather than retyped."""
import json, pathlib, re
R = pathlib.Path(__file__).resolve().parents[2]
rc = (R / "docs/root-cause.md").read_text()

def table_after(heading, doc):
    body = doc.split(heading, 1)[1]
    rows = []
    for line in body.split("\n"):
        if line.startswith("|") and not re.match(r"^\|[\s|:-]+\|$", line):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            rows.append(cells)
        elif rows and not line.startswith("|"):
            break
    return rows[1:]  # drop the header row

chain = table_after("## 1. The causal chain", rc)
shipped = table_after("## 2. Why downgrading appears to work", rc)
ruled = table_after("## 3. What was ruled out", rc)

def strip_md(s):
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)        # links -> text
    s = re.sub(r"[*`]", "", s)
    return re.sub(r"\s+", " ", s).strip()

out = {
 "_what": "Figures for rccl-atomics-hostcall.html, extracted from docs/root-cause.md.",
 "chain": [{"n": int(r[0]), "link": strip_md(r[1]).split(". In a guest")[0],
            "evidence": strip_md(r[2])} for r in chain],
 "shipped": [{"rccl": strip_md(r[0]), "hostcall": strip_md(r[1]),
              "behaviour": strip_md(r[2])} for r in shipped],
 "ruled_out": [{"hypothesis": strip_md(r[0]), "verdict": strip_md(r[1]),
                "how": strip_md(r[2])} for r in ruled],
}
out["counts"] = {
  "chain_links": len(out["chain"]),
  "hypotheses_total": len(out["ruled_out"]),
  "hypotheses_eliminated": sum(1 for h in out["ruled_out"] if h["verdict"].startswith("❌")),
  "hypotheses_confirmed": sum(1 for h in out["ruled_out"] if h["verdict"].startswith("✅")),
}
json.dump(out, open(pathlib.Path(__file__).parent / "figures-rccl.json", "w"), ensure_ascii=False, indent=1)
print("chain links:", out["counts"]["chain_links"])
print("shipped rows:", len(out["shipped"]), [s["rccl"] + " -> " + s["hostcall"] for s in out["shipped"]])
print("hypotheses:", out["counts"])
for h in out["ruled_out"]:
    print(f"   {h['verdict'][:1]} {h['hypothesis'][:52]}")
