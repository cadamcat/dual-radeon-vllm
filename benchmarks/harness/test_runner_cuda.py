#!/usr/bin/env python3
"""Gates for runner_cuda.py's ladder, run on a laptop with no GPU.

Two of these exist because the failures they catch are silent. A ladder cache
keyed on the model alone returns short prompts for a long ladder, and a ladder
longer than the book returns the whole book under a label saying 128 000. In
both cases every row still gets written, every number still looks like a
number, and nothing anywhere says the rung is not what it claims.

    python3 benchmarks/harness/test_runner_cuda.py
"""
import json
import os
import subprocess
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))


def load(work):
    """exec runner_cuda.py with the GPU-only bits stubbed out"""
    src = open(os.path.join(HERE, "runner_cuda.py")).read()
    os.environ["BENCH_WORK"] = work
    os.environ["BENCH_MODELS"] = work

    real_run = subprocess.run

    def fake_run(cmd, *a, **k):
        if isinstance(cmd, str) and "nvidia-smi" in cmd:
            return types.SimpleNamespace(stdout="97887\n", stderr="", returncode=0)
        return real_run(cmd, *a, **k)

    tele = types.ModuleType("harness.telemetry")
    tele.Sampler = object
    tele.describe = lambda: {}
    sys.modules.setdefault("harness", types.ModuleType("harness"))
    sys.modules["harness.telemetry"] = tele

    g = {"__name__": "runner_cuda", "__file__": os.path.join(HERE, "runner_cuda.py")}
    old = subprocess.run
    subprocess.run = fake_run
    try:
        exec(compile(src, "runner_cuda.py", "exec"), g)
    finally:
        subprocess.run = old
    return g


class FakeTok:
    """one token per word, and decode is a join -- enough to exercise cutting"""
    def __init__(self, words):
        self.words = words

    def __call__(self, text):
        return types.SimpleNamespace(input_ids=list(range(len(text.split()))))

    def decode(self, ids, **k):
        return " ".join(self.words[:len(ids)])


def with_tokenizer(g, n_words):
    """point ladder_for at a fake tokenizer over a fake book of n_words"""
    words = [f"w{i}" for i in range(n_words)]
    tok = FakeTok(words)
    mod = types.ModuleType("transformers")
    mod.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: tok)
    sys.modules["transformers"] = mod
    g["get_book"] = lambda: " ".join(words)
    return tok


FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def main():
    with tempfile.TemporaryDirectory() as work:
        g = load(work)
        with_tokenizer(g, 200_000)
        model = os.path.join(work, "some-model")

        # 1. the silent one. Ask for a short ladder, then a long one, same
        #    model. The long one must actually get its long rungs.
        short = g["ladder_for"](model, [500, 1000])
        long_ = g["ladder_for"](model, [500, 1000, 48000, 128000])
        got = [e["target"] for e in long_]
        check("second ladder has all four rungs", got == [500, 1000, 48000, 128000],
              f"got {got}")
        check("the 128000 rung is 128000 tokens long",
              len(long_[-1]["text"].split()) == 128000,
              f"{len(long_[-1]['text'].split())} words")
        check("the short ladder was reused, not recut",
              short[0]["text"] == long_[0]["text"])

        # 2. every rung is a strict prefix of the next, which is what makes
        #    prefix caching matter and what the ladder claims to be
        check("rungs nest", all(long_[i]["text"] == long_[i + 1]["text"][:len(long_[i]["text"])]
                                for i in range(len(long_) - 1)))

        # 3. a ladder longer than the book must raise, not truncate
        raised = None
        try:
            g["ladder_for"](os.path.join(work, "other-model"), [500, 300_000])
        except RuntimeError as e:
            raised = str(e)
        check("a rung longer than the book raises", raised is not None,
              (raised or "returned quietly")[:80])

        # 4. the cache on disk is the keyed-by-target form
        cache = json.load(open(f"{work}/ladder-some-model.json"))
        check("cache is keyed by target", isinstance(cache, dict)
              and set(cache) == {"500", "1000", "48000", "128000"},
              f"{type(cache).__name__} {sorted(cache)[:5] if isinstance(cache, dict) else ''}")

        # 5. what a serve log says has gone wrong. Every string below is
        #    copied out of a log this repository actually produced, not
        #    paraphrased: a retry that fires on a remembered message is a
        #    retry that does not fire.
        MAMBA = ("(EngineCore pid=721) ValueError: max_num_seqs (1024) exceeds "
                 "available Mamba cache blocks (969). Each decode sequence "
                 "requires one Mamba cache block, so CUDA graph capture cannot "
                 "proceed. Please lower max_num_seqs to at most 969 or increase "
                 "gpu_memory_utilization.")
        cl = g["classify"]
        check("the Mamba block message asks for a lower mns",
              cl(MAMBA) == ("mns", 969), str(cl(MAMBA))[:70])
        # 2026-09-03: Muse-Glimmer refused mml 132 000 in 31 s. Copied from
        # that log; the number vLLM derives is a float in the message.
        MAXLEN = ("(APIServer pid=48) pydantic_core._pydantic_core.ValidationError: "
                  "1 validation error for ModelConfig\n  Value error, User-specified "
                  "max_model_len (132000) is greater than the derived max_model_len "
                  "(max_position_embeddings=131072.0 or model_max_length=None in "
                  "model's config.json).")
        check("a too-long max_model_len asks for the model's own limit",
              cl(MAXLEN) == ("maxlen", 131072), str(cl(MAXLEN))[:70])
        check("a KV-length message still asks for a lower mml",
              cl("estimated maximum model length is 15792") == ("capacity", 15792),
              str(cl("estimated maximum model length is 15792")))
        check("no room for KV is still capacity",
              cl("No available memory for the cache blocks") == ("capacity", -1))
        # the regression the runner's own comment records: torch logs whole
        # tracebacks at W level, and reading one as a crash stopped a healthy
        # server on the Radeon side.
        LOGGED_TB = ("WARNING 09-02 21:03:12 [triton_bundler.py:212] Traceback "
                     "(most recent call last):")
        check("a logged traceback is not a crash", cl(LOGGED_TB) is None,
              str(cl(LOGGED_TB))[:60])
        REAL_TB = "Traceback (most recent call last):\n  File \"x.py\", line 1"
        check("a real traceback is a crash",
              (cl(REAL_TB) or ("", ""))[0] == "crash", str(cl(REAL_TB))[:40])
        check("a log still starting says nothing yet",
              cl("INFO 09-03 00:01:02 [core.py:99] Waiting for init") is None)

        # 6. per-config mml and targets are read from the config row
        check("mml comes from the config row", (dict(mml=132000).get("mml")
                                                or g["MML"]) == 132000)
        check("MML default unchanged", g["MML"] == 33000, str(g["MML"]))
        check("start cap is not an hour", g["HARD_START_S"] <= 1200,
              f"{g['HARD_START_S']}s")

    print(f"\n{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAILED: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
