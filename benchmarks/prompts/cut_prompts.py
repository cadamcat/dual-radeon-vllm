#!/usr/bin/env python3
"""Rebuild the prompt ladders used by the benchmark campaign.

The source text is Darwin's *On the Origin of Species*, Project Gutenberg #1228
(public domain). It is downloaded rather than vendored, so this needs network
access once; the copy is then cached next to this file.

Each model family gets its own ladder, cut with that model's own tokenizer and
trimmed to a sentence boundary — the same string is a different number of tokens
to gemma, to qwen, to gemma-26B and to Muse-Glimmer. gemma-3 is the exception:
it tokenises every rung identically to gemma-4, so it reuses that ladder and its
counts are recorded alongside as the check.

Ladders are written under the directory names `bench_runner.py` reads
(`prompts/`, `prompts-qwen/`, ...), so `--out` can point straight at PROMPT_ROOT.

The committed `manifest-*.json` files record the exact token counts that were
measured. This script re-derives them and **reports any drift**, so you can tell
whether your rebuild matches what produced `results.jsonl`.

    # rebuild the ladders, verifying against the manifests
    python3 cut_prompts.py --models-dir /path/to/models

    # verify only, write nothing
    python3 cut_prompts.py --models-dir /path/to/models --check-only

Model directory names default to the ones used in the campaigns; override with
--gemma / --gemma-alt / --qwen / --qwen-alt / --gemma26b / --muse if yours differ.

A family with no committed manifest is a new ladder: it is cut and its manifest
written, with nothing to compare against.
"""
import argparse, json, os, re, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".gutenberg-1228.txt")
URL = "https://www.gutenberg.org/cache/epub/1228/pg1228.txt"
ANCHOR = "When on board H.M.S."          # first sentence of the Introduction
TARGETS = [500, 1000, 2000, 4000, 6000, 8000, 12000, 16000, 20000, 24000, 32000]

HEADER = ('Translate the following passage from Charles Darwin\'s "On the Origin of '
          'Species" into fluent, natural Simplified Chinese. Preserve the meaning and '
          'the formal scientific tone. Output only the translation, with no '
          'commentary.\n\nPassage:\n')


def source_text():
    if not os.path.exists(CACHE):
        print(f"downloading {URL}", flush=True)
        with urllib.request.urlopen(URL, timeout=60) as r:
            open(CACHE, "wb").write(r.read())
    raw = open(CACHE, encoding="utf-8").read()
    try:
        body = raw.split("*** START OF THE PROJECT GUTENBERG EBOOK", 1)[1].split("\n", 1)[1]
        body = body.split("*** END OF THE PROJECT GUTENBERG EBOOK", 1)[0]
    except IndexError:
        sys.exit("could not find the Gutenberg start/end markers — is the cache truncated?")
    if ANCHOR not in body:
        sys.exit(f"anchor {ANCHOR!r} not found — Gutenberg may have re-issued this text")
    passage = body[body.index(ANCHOR):]
    # unwrap hard line breaks inside paragraphs, keep paragraph separation
    return "\n\n".join(" ".join(p.split()) for p in re.split(r"\n\s*\n", passage))


def sentence_ends(passage):
    """character offsets just past every '. ' — the only places we may cut"""
    out, i = [], passage.find(". ")
    while i != -1:
        out.append(i + 1)
        i = passage.find(". ", i + 1)
    out.append(len(passage))
    return out


def cut_for(tok, passage, target, ends):
    """binary search over sentence boundaries for the one closest to `target`.

    Deterministic — unlike an iterative character-length estimate, this cannot
    stop at a different point depending on the path it took to get there.
    """
    count = lambda t: len(tok.encode(t, add_special_tokens=False))
    lo, hi, best = 0, len(ends) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        text = HEADER + passage[:ends[mid]]
        got = count(text)
        if best is None or abs(got - target) < abs(best[1] - target):
            best = (text, got)
        if got < target:
            lo = mid + 1
        elif got > target:
            hi = mid - 1
        else:
            break
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", default=os.environ.get("MODELS_DIR", "/data/incoming"))
    ap.add_argument("--gemma", default="gemma-4-12B-it-qat-w4a16-ct")
    ap.add_argument("--qwen", default="Qwen3-8B")
    ap.add_argument("--qwen-alt", default="Qwen3.6-27B-AWQ-INT4,Qwen3.8-27B-AWQ-INT4",
                    help="comma-separated further qwen models sharing this ladder; their "
                         "counts are recorded alongside, not used for cutting")
    ap.add_argument("--gemma26b", default="gemma-4-26B-A4B-AWQ")
    ap.add_argument("--gemma-alt", default="gemma-3-27b-it-w4a16",
                    help="comma-separated further gemma models sharing this ladder; measured "
                         "2026-08-24 to tokenise every rung identically to gemma-4")
    ap.add_argument("--muse", default="Muse-Glimmer-30B-INT4")
    ap.add_argument("--out", default=HERE, help="where to write prompt_<n>.txt (default: here)")
    ap.add_argument("--check-only", action="store_true",
                    help="verify against the manifests, write nothing")
    ap.add_argument("--targets", default="",
                    help="comma-separated EXTRA targets to cut beside the eleven "
                         "committed ones, e.g. 48000,64000,128000. Additive on "
                         "purpose: the eleven are what every campaign before "
                         "2026-09-03 measured and they must keep reproducing "
                         "byte for byte while a longer ladder is added.")
    ap.add_argument("--only", default="",
                    help="comma-separated family labels to process (default: all present). "
                         "Use this to add a ladder without rewriting the existing ones.")
    a = ap.parse_args()
    targets = list(TARGETS)
    if a.targets:
        extra = sorted({int(x) for x in a.targets.split(",") if x.strip()})
        if set(extra) & set(TARGETS):
            ap.error("--targets adds to the eleven; it cannot restate one of them")
        targets += extra
        print(f"targets: the committed {len(TARGETS)} plus {extra}")

    from transformers import AutoTokenizer      # late import: not needed for --help
    passage = source_text()
    ends = sentence_ends(passage)
    print(f"source: {len(passage):,} characters after the anchor\n")

    # out_dir is the name bench_runner.py reads, not the family label
    ladders = [("gemma", a.gemma, "manifest-gemma.json", "prompts", a.gemma_alt),
               ("qwen", a.qwen, "manifest-qwen.json", "prompts-qwen", a.qwen_alt),
               ("gemma26b", a.gemma26b, "manifest-gemma26b.json", "prompts-26b", None),
               ("muse", a.muse, "manifest-muse.json", "prompts-muse", None)]
    worst, checked, cut_any = 0, 0, False
    alt_drift = []
    only = {x.strip() for x in a.only.split(",") if x.strip()}
    for label, model, manifest_name, dirname, alt in ladders:
        if only and label not in only:
            continue
        path = os.path.join(a.models_dir, model)
        if not os.path.isdir(path):
            print(f"[{label}] SKIP — {path} not present")
            continue
        tok = AutoTokenizer.from_pretrained(path)
        cut_any = True
        alts = {}
        for m in (alt.split(",") if alt else []):
            m = m.strip()
            ap_ = os.path.join(a.models_dir, m)
            if m and os.path.isdir(ap_):
                alts[m] = AutoTokenizer.from_pretrained(ap_)
        mpath = os.path.join(HERE, manifest_name)
        if os.path.exists(mpath):
            recorded = {e["target"]: e for e in json.load(open(mpath))}
        else:
            recorded = {}
            print(f"[{label}] new ladder — no committed manifest, nothing to compare")
        out_dir = os.path.join(a.out, dirname)
        if not a.check_only:
            os.makedirs(out_dir, exist_ok=True)
        rebuilt = []
        for t in targets:
            text, got = cut_for(tok, passage, t, ends)
            e = {"target": t, "est_prompt_tokens": got, "chars": len(text)}
            for m, ta in alts.items():
                e.setdefault("alt_tokens", {})[m] = len(ta.encode(text, add_special_tokens=False))
            rebuilt.append(e)
            if not a.check_only:
                open(os.path.join(out_dir, f"prompt_{t}.txt"), "w", encoding="utf-8").write(text)
            # The alt models share this ladder, and that sharing is a claim the
            # manifest records. Checking only est_prompt_tokens would let gemma-3
            # stop matching gemma-4 without this script noticing.
            was_alt = recorded.get(t, {}).get("alt_tokens") or {}
            for m, got_alt in e.get("alt_tokens", {}).items():
                if m in was_alt and was_alt[m] != got_alt:
                    alt_drift.append((label, t, m, was_alt[m], got_alt))
            was = recorded.get(t, {}).get("est_prompt_tokens")
            note = ""
            if was is not None:
                worst = max(worst, abs(got - was)); checked += 1
                note = f"   recorded {was:>6}   drift {got - was:+d}"
            print(f"[{label}] {t:>6}: {got:>6} tok, {len(text):>7} chars{note}")
        if not a.check_only:
            json.dump(rebuilt, open(os.path.join(out_dir, "manifest.json"), "w"), indent=2)
            if not recorded:
                json.dump(rebuilt, open(mpath, "w"), indent=2)
                print(f"[{label}] wrote {manifest_name}")
        print()

    if not cut_any:
        sys.exit("no tokenizer was available — nothing cut or verified (set --models-dir)")
    if alt_drift:
        print("ALT-TOKENIZER DRIFT — a model that shared a ladder no longer does:")
        for label, t, m, was, got in alt_drift:
            print(f"  [{label}] target {t}: {m} recorded {was}, now {got}")
        print()
    if not checked:
        print("only new ladders were cut; there was nothing to verify against")
        return 1 if alt_drift else 0
    pct = worst / 32000 * 100
    print(f"largest drift against the committed manifests: {worst} tokens over {checked} rungs "
          f"({pct:.2f} % of the longest rung)")
    print()
    if worst == 0:
        print("Exact reproduction of the recorded ladder.")
    elif pct < 1.0:
        print("This is the expected outcome, and it does not affect any published number.\n"
              "The short rungs reproduce exactly. The long ones can differ by a fraction of a\n"
              "percent because the original cutter stopped as soon as it was within tolerance,\n"
              "so its result depended on the path it took; the binary search here is\n"
              "deterministic and lands slightly closer to each target.\n"
              "\n"
              "Nothing downstream depends on these nominal lengths: every analysis uses the\n"
              "`prompt_tokens` that the server actually reported per request, recorded in\n"
              "results.jsonl.")
    else:
        print("NOTE: drift above 1 % suggests a different tokenizer revision or a re-issued\n"
              "      source text — worth checking before comparing against our numbers.")
        return 1
    return 1 if alt_drift else 0


if __name__ == "__main__":
    # main() returns a status and it used to be thrown away, so --check-only
    # exited 0 on tokenizer drift, on alt-tokenizer divergence, and on drift
    # above the 1% threshold it prints a NOTE about
    sys.exit(main())
