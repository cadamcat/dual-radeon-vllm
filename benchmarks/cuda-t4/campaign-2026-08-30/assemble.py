"""Assemble the T4 campaign file out of the two sessions that survived.

Three sessions measured this ladder. `t4b` reached 19 of 22 rungs and was lost
with its VM; `t4c` reached 21 and was lost before round 2 of 32 000; `t4d` was
given `t4c`'s lower rungs as a checkpoint seed and measured **both** rounds of
32 000 inside one engine.

`results.jsonl` is `t4c`'s rungs **below** 32 000 plus `t4d`'s 32 000 pair.

Two rows are deliberately left out, and the reason is a number:

  * `t4d`'s file also contains the twenty seeded `decode` rows, which are
    `t4c`'s own rows carrying `t4c`'s timestamps. Kept, they would count one
    measurement twice.
  * **`t4c`'s 32 000 round 1 is left out of this file.** It is 50.8 minutes
    before `t4d`'s round 1, and `build_ledger.SESSION_GAP_S` is **3 600 s**, so
    `latest_session` would NOT see a session boundary between them: all three
    values would aggregate into one cell with `runs=3`, averaging across two
    VMs and reporting a spread that describes neither. The published pair is
    the pair measured in one engine. `t4c`'s round 1 survives in
    `results-t4c-21rungs.jsonl` beside it, and the agreement between the two
    VMs at that rung -- 8.8631 against 8.99 tok/s, 1.43 % -- is stated in the
    README as what it is: a cross-VM check, not a third round.

Neither sibling file is a `build_prefill.SOURCES` entry. They are the record,
not the measurement.
"""
import json
import sys

C = "colab-harvest/t4c/results.jsonl"
D = "colab-harvest/t4d/results.jsonl"
OUT = sys.argv[1] if len(sys.argv) > 1 else "campaign-0830e/results.jsonl"
SPLICE = 32000

seed_keys = set()
for line in open(C):
    r = json.loads(line)
    if r.get("kind") == "decode":
        seed_keys.add((r["cfg"], r["target"], r["round"], r["ts"]))

out, dropped_c, kept_d, dropped_seed = [], 0, 0, 0
for line in open(C):
    r = json.loads(line)
    if r.get("target") == SPLICE:
        dropped_c += 1
        continue
    out.append(line.rstrip("\n"))
for line in open(D):
    r = json.loads(line)
    if r.get("kind") == "decode" and (r["cfg"], r["target"], r["round"], r["ts"]) in seed_keys:
        dropped_seed += 1
        continue
    out.append(line.rstrip("\n"))
    kept_d += 1

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")

print(f"t4c rows carried  : {sum(1 for _ in open(C)) - dropped_c}")
print(f"t4c 32000 dropped : {dropped_c}  (expected 2: one prefill, one decode)")
print(f"t4d rows kept     : {kept_d}")
print(f"t4d seeds dropped : {dropped_seed}  (expected 20)")
print(f"wrote {OUT}: {len(out)} rows")

print("\nevery (kind, target, round) in the result, with its source session:")
seen = {}
for line in out:
    r = json.loads(line)
    if r.get("kind") in ("prefill", "decode"):
        seen.setdefault(r["target"], set()).add(r["round"])
for t in sorted(seen):
    print(f"  target {t:6d}  rounds {sorted(seen[t])}")
