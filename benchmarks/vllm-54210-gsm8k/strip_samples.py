#!/usr/bin/env python3
"""Distil lm_eval's per-document samples into the rows this campaign commits.

    python3 strip_samples.py <samples_*.jsonl> <out.jsonl>

Two fields are dropped and nothing else is touched:

  arguments  the rendered 5-shot prompt, 8.25 MB of the 12 MB per arm, the same
             five exemplars in front of all 1 319 questions.
  doc        gsm8k's own question and answer, which is the pinned dataset --
             openai/gsm8k at revision 740312add88f, md5s in the README.

Both are dropped for size, and neither is dropped blind: lm_eval already writes
`prompt_hash`, `doc_hash` and `target_hash` into every record, and those stay.
So the claim the paired comparison rests on -- that the two arms were asked the
same questions with the same prompts -- is still checkable from these rows
alone, and `verify_doc_figures.py` checks it.
"""
import json, sys

DROP = ("arguments", "doc")
src, dst = sys.argv[1], sys.argv[2]
n = 0
with open(dst, "w") as out:
    for line in open(src):
        d = json.loads(line)
        for k in DROP:
            d.pop(k, None)
        out.write(json.dumps(d, ensure_ascii=False, sort_keys=True) + "\n")
        n += 1
print(f"{n} rows -> {dst}")
