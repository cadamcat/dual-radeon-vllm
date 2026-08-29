"""Assemble every page under docs/ from one head template plus per-page bodies.

The head is shared, so a change to the theme or the floating bar reaches all of
them. Where an article exists in two languages the figures block and the script
are inserted from one source, which is what lets verify_doc_figures assert the
two are byte-identical rather than hope so.
"""
import json, pathlib, re, sys

CHECK = "--check" in sys.argv   # build into memory and compare, writing nothing
D = pathlib.Path(__file__).parent / "src"
OUT = pathlib.Path(__file__).resolve().parent.parent / "docs"
head = (D / "article-head.html").read_text()


def lang_nav(current, en, zh):
    """Both links always present; the current one is not a link to itself."""
    rows = []
    for code, label, href in (("en", "EN", en), ("zh", "中", zh)):
        cur = ' aria-current="page"' if code == current else ""
        rows.append(f'  <a class="lang" href="{href}" hreflang="{code}"{cur}>{label}</a>')
    return "\n".join(rows)


def page(body, *, lang, title, desc, out, extra_css=None, nav=None, labels, figures=None,
         script_from=None):
    h = head
    if extra_css:
        h = h.replace("  .rv { opacity:0;", (D / extra_css).read_text() + "\n  .rv { opacity:0;")
    if nav is None:                       # the index has no language pair
        h = h.replace('__LANG_NAV__\n  <span class="sep" aria-hidden="true"></span>\n', "")
    else:
        h = h.replace("__LANG_NAV__", nav)
    h = (h.replace("__LANG__", lang).replace("__TITLE__", title).replace("__DESC__", desc)
          .replace("__NAVLABEL__", labels[0]).replace("__T_AUTO__", labels[1])
          .replace("__T_LIGHT__", labels[2]).replace("__T_DARK__", labels[3]))
    b = (D / body).read_text()
    if script_from:                       # reuse the English script verbatim
        m = re.search(r"<script>\n\(function \(\).*?\n</script>\n", (D / script_from).read_text(), re.S)
        assert m, f"no script block in {script_from}"
        assert "__SCRIPT__" in b, f"{body} has no __SCRIPT__ slot"
        b = b.replace("__SCRIPT__", m.group(0).rstrip("\n"))
    if figures:
        b = b.replace("__FIGURES_JSON__", (D / figures).read_text())
    p = OUT / out
    text = h + b
    if CHECK:
        if not p.exists():
            MISMATCH.append(f"{out}: not published")
        elif p.read_text() != text:
            MISMATCH.append(f"{out}: published copy differs from its source")
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


MISMATCH = []
EN_LABELS = ("Language and colour theme", "Match system", "Light", "Dark")
ZH_LABELS = ("语言与配色主题", "跟随系统",
             "浅色", "深色")
IDX_LABELS = ("Colour theme", "Match system", "Light", "Dark")

H_EN, H_ZH = "hybrid-ssm-collapse.html", "hybrid-ssm-collapse.zh.html"
R_EN, R_ZH = "rccl-atomics-hostcall.html", "rccl-atomics-hostcall.zh.html"
W_EN, W_ZH = "w4a16-two-problems.html", "w4a16-two-problems.zh.html"
M_EN, M_ZH = "moe-written-off-by-eager.html", "moe-written-off-by-eager.zh.html"
L_EN, L_ZH = "weight-loading-19x.html", "weight-loading-19x.zh.html"
S_EN, S_ZH = "speculative-decoding-net-loss.html", "speculative-decoding-net-loss.zh.html"
A_EN, A_ZH = "a100-vs-two-radeons.html", "a100-vs-two-radeons.zh.html"
Q_EN, Q_ZH = "gqa-gate-costs-nothing.html", "gqa-gate-costs-nothing.zh.html"
N_EN, N_ZH = "reporting-a-non-reproduction.html", "reporting-a-non-reproduction.zh.html"
X_EN, X_ZH = "measuring-decode.html", "measuring-decode.zh.html"

built = []
built.append(page("article-body.html", lang="en", figures="figures.json",
                  title="A hybrid-SSM model that decodes slower the longer you talk to it",
                  desc="Qwen3.6-27B falls from 12.1 to 4.2 tok/s between 500 and 32000 tokens of "
                       "context on 2x RX 7900 XT. Attribution, mechanism, control, and the upstream "
                       "fix, measured.",
                  out="articles/" + H_EN, nav=lang_nav("en", H_EN, H_ZH), labels=EN_LABELS))
built.append(page("article-body-zh.html", lang="zh-CN", figures="figures.json",
                  script_from="article-body.html",
                  title="一个上下文越长解码越慢的"
                        "混合 SSM 模型",
                  desc="Qwen3.6-27B 在 2× RX 7900 XT 上，上下文从 "
                       "500 到 32000 token，解码由 12.1 掉到 4.2 tok/s。"
                       "归因、机理、对照，以及上游"
                       "修复的实测。",
                  out="articles/" + H_ZH, nav=lang_nav("zh", H_EN, H_ZH), labels=ZH_LABELS))
built.append(page("rccl-body.html", lang="en", figures="figures-rccl.json",
                  extra_css="rccl-extra.css",
                  title="The RCCL crash was never about RCCL",
                  desc="Two Radeons, tensor parallelism, and hipErrorIllegalState. The cause is four "
                       "layers below RCCL: no PCIe atomics, no hostcall buffer, refused dispatch. "
                       "Thirteen hypotheses, and a one-line VM fix.",
                  out="articles/" + R_EN, nav=lang_nav("en", R_EN, R_ZH), labels=EN_LABELS))
if (D / "rccl-body-zh.html").exists():
    built.append(page("rccl-body-zh.html", lang="zh-CN", figures="figures-rccl.json",
                      extra_css="rccl-extra.css", script_from="rccl-body.html",
                      title="RCCL 崩溃从来不是 RCCL 的问题",
                      desc="两张 Radeon、张量并行，以及 "
                           "hipErrorIllegalState。根因在 RCCL 下面四层："
                           "没有 PCIe atomics，就没有 hostcall buffer，"
                           "dispatch 被拒。",
                      out="articles/" + R_ZH, nav=lang_nav("zh", R_EN, R_ZH), labels=ZH_LABELS))

built.append(page("w4a16-body.html", lang="en", figures="figures-w4a16.json",
                  extra_css="w4a16-extra.css",
                  title="Twelve tokens a second was two problems",
                  desc="One 27B model, two independent costs: a flat 60 ms per decode step from the "
                       "W4A16 linear kernel, and a paged-attention term that grows with context. "
                       "Separated by an A/B, each fixed upstream by someone else.",
                  out="articles/" + W_EN, nav=lang_nav("en", W_EN, W_ZH), labels=EN_LABELS))
if (D / "w4a16-body-zh.html").exists():
    built.append(page("w4a16-body-zh.html", lang="zh-CN", figures="figures-w4a16.json",
                      extra_css="w4a16-extra.css", script_from="w4a16-body.html",
                      title="12 tok/s 是两个问题",
                      desc="一个 27B 模型上叠着两笔独立开销：W4A16 线性 "
                           "kernel 每步固定 60 ms，以及随上下文增长的 paged "
                           "attention。用一次 A/B 把它们分开，两个修复都是"
                           "别人做的。",
                      out="articles/" + W_ZH, nav=lang_nav("zh", W_EN, W_ZH), labels=ZH_LABELS))

built.append(page("moe-body.html", lang="en", figures="figures-moe.json",
                  extra_css="moe-extra.css",
                  title="The fastest model here was written off at 15 tok/s",
                  desc="A 128-expert MoE recorded at 15 tok/s under --enforce-eager decodes at 107.8 "
                       "compiled. The flag also fabricated an asymmetric power draw and a "
                       "context-independent rate, both read as architecture.",
                  out="articles/" + M_EN, nav=lang_nav("en", M_EN, M_ZH), labels=EN_LABELS))
if (D / "moe-body-zh.html").exists():
    built.append(page("moe-body-zh.html", lang="zh-CN", figures="figures-moe.json",
                      extra_css="moe-extra.css", script_from="moe-body.html",
                      title="全机最快的模型曾被 15 tok/s 判死刑",
                      desc="一个 128 专家的 MoE 在 --enforce-eager 下记作 15 "
                           "tok/s，编译之后是 107.8。这个开关还伪造了功耗左右"
                           "不对称和吞吐与上下文无关两个现象，都被当成了架构结论。",
                      out="articles/" + M_ZH, nav=lang_nav("zh", M_EN, M_ZH), labels=ZH_LABELS))

built.append(page("loader-body.html", lang="en", figures="figures-loader.json",
                  extra_css="loader-extra.css",
                  title="Loading weights was slower than the disk, twice over",
                  desc="A permission read off the VMA makes every host-to-device copy break "
                       "copy-on-write, and a split kernel backport turned each occurrence into a "
                       "one-second timeout. Two effects, one reproducer, three kernel states.",
                  out="articles/" + L_EN, nav=lang_nav("en", L_EN, L_ZH), labels=EN_LABELS))
if (D / "loader-body-zh.html").exists():
    built.append(page("loader-body-zh.html", lang="zh-CN", figures="figures-loader.json",
                      extra_css="loader-extra.css", script_from="loader-body.html",
                      title="加载权重比磁盘还慢，而且慢了两次",
                      desc="从 VMA 取权限让每次 host→device 拷贝都破坏 "
                           "copy-on-write；而一个被拆开的内核 backport 把每一次"
                           "触发都变成一秒的超时。两个效应，一个复现器，三种内核状态。",
                      out="articles/" + L_ZH, nav=lang_nav("zh", L_EN, L_ZH), labels=ZH_LABELS))

built.append(page("spec-body.html", lang="en", figures="figures-spec.json",
                  extra_css="spec-extra.css",
                  title="One boolean costs 71% on a Radeon and 61% on an A100",
                  desc="Speculative decoding is +36.9% at 1K of context and -70.8% at 32K, because "
                       "max_seqlen_q > 1 drops the Triton attention kernel from 128 workgroups to 8. "
                       "Measured on two vendors.",
                  out="articles/" + S_EN, nav=lang_nav("en", S_EN, S_ZH), labels=EN_LABELS))
if (D / "spec-body-zh.html").exists():
    built.append(page("spec-body-zh.html", lang="zh-CN", figures="figures-spec.json",
                      extra_css="spec-extra.css", script_from="spec-body.html",
                      title="一个布尔值，在 Radeon 上是 71%，在 A100 上是 61%",
                      desc="投机解码在 1K 上下文是 +36.9%，在 32K 上是 "
                           "-70.8%，因为 max_seqlen_q > 1 把 Triton 注意力 "
                           "kernel 从 128 个 workgroup 降到 8 个。两个厂商都测了。",
                      out="articles/" + S_ZH, nav=lang_nav("zh", S_EN, S_ZH), labels=ZH_LABELS))

built.append(page("a100-body.html", lang="en", figures="figures-a100.json",
                  extra_css="a100-extra.css",
                  title="Two consumer Radeons against one A100",
                  desc="On batch-1 decode of the same 31B model the A100 is 1.48x ahead at 1K, 1.14x "
                       "at 16K and 1.87x at 32K. The gap is U-shaped, and both ends are about tensor "
                       "parallelism rather than about the silicon.",
                  out="articles/" + A_EN, nav=lang_nav("en", A_EN, A_ZH), labels=EN_LABELS))
if (D / "a100-body-zh.html").exists():
    built.append(page("a100-body-zh.html", lang="zh-CN", figures="figures-a100.json",
                      extra_css="a100-extra.css", script_from="a100-body.html",
                      title="两张消费级 Radeon 对一张 A100",
                      desc="同一个 31B 模型的 batch-1 解码，A100 在 1K 上领先 "
                           "1.48×，16K 上 1.14×，32K 上 1.87×。差距是 U 形的，"
                           "而两端都关于张量并行，不关于硅片本身。",
                      out="articles/" + A_ZH, nav=lang_nav("zh", A_EN, A_ZH), labels=ZH_LABELS))

built.append(page("gqa-body.html", lang="en", figures="figures-gqa.json",
                  extra_css="gqa-extra.css",
                  title="A gate that costs 2 to 7 times and buys nothing",
                  desc="vLLM's ROCm custom paged attention is gated off below gqa_ratio 3 on gfx11. "
                       "In the excluded range it is 1.70x to 7.28x faster than the fallback, in every "
                       "one of sixty measured cells.",
                  out="articles/" + Q_EN, nav=lang_nav("en", Q_EN, Q_ZH), labels=EN_LABELS))
if (D / "gqa-body-zh.html").exists():
    built.append(page("gqa-body-zh.html", lang="zh-CN", figures="figures-gqa.json",
                      extra_css="gqa-extra.css", script_from="gqa-body.html",
                      title="一道 2–7 倍的门，什么也没换来",
                      desc="vLLM 的 ROCm 定制 paged attention 在 gfx11 上被 "
                           "gqa_ratio 3 这道门挡在外面。而在被排除的区间里，"
                           "它比兜底路径快 1.70–7.28 倍，六十个格子无一例外。",
                      out="articles/" + Q_ZH, nav=lang_nav("zh", Q_EN, Q_ZH), labels=ZH_LABELS))

built.append(page("n6565-body.html", lang="en", figures="figures-6565.json",
                  extra_css="n6565-extra.css",
                  title="How to report a bug you cannot reproduce",
                  desc="135 clean communicator initialisations say almost nothing on their own. What "
                       "makes a negative result usable: a sweep that could have exposed the defect, a "
                       "stated contrast, and finding what your instrument is blind to.",
                  out="articles/" + N_EN, nav=lang_nav("en", N_EN, N_ZH), labels=EN_LABELS))
if (D / "n6565-body-zh.html").exists():
    built.append(page("n6565-body-zh.html", lang="zh-CN", figures="figures-6565.json",
                      extra_css="n6565-extra.css", script_from="n6565-body.html",
                      title="怎么报告一个你复现不出来的 bug",
                      desc="135 次干净的通信器初始化本身几乎什么也说明不了。"
                           "让一个否定结论变得有用的三件事：一次本可以暴露它的"
                           "扫描、把差异摆明，以及找出你的仪器看不见什么。",
                      out="articles/" + N_ZH, nav=lang_nav("zh", N_EN, N_ZH), labels=ZH_LABELS))

built.append(page("measure-body.html", lang="en", figures="figures-measure.json",
                  extra_css="measure-extra.css",
                  title="How to measure decode on a machine like this",
                  desc="Two harnesses agree to 0.44%, but only once the machine is warm: the first of "
                       "four identical runs read 31% low. Why every point carries its run count and "
                       "range, and why the range rather than a standard deviation.",
                  out="articles/" + X_EN, nav=lang_nav("en", X_EN, X_ZH), labels=EN_LABELS))
if (D / "measure-body-zh.html").exists():
    built.append(page("measure-body-zh.html", lang="zh-CN", figures="figures-measure.json",
                      extra_css="measure-extra.css", script_from="measure-body.html",
                      title="在这样一台机器上怎么测 decode",
                      desc="两套 harness 吻合到 0.44%，但前提是机器已经热了："
                           "四次相同运行里的第一次低了 31%。为什么每个点都带 run "
                           "数和极差，以及为什么报极差而不是标准差。",
                      out="articles/" + X_ZH, nav=lang_nav("zh", X_EN, X_ZH), labels=ZH_LABELS))

articles = {"articles": [
    {"href": "articles/" + X_EN, "title": "How to measure decode on a machine like this",
     "blurb": "The two harnesses agree to 0.44%, and finding that out took four identical runs the "
              "first of which read 31% low. What every point carries, where the chart-grade line sits, "
              "and one cell where the range and the standard deviation disagree about the direction.",
     "measured": "measured 2026-08-26",
     "langs": ["EN", "\u4e2d"] if (D / "measure-body-zh.html").exists() else ["EN"],
     "tags": ["methodology", "calibration", "nondeterminism"]},
    {"href": "articles/" + N_EN, "title": "How to report a bug you cannot reproduce",
     "blurb": "A clean run is the least useful sentence on a bug tracker. Three things make it worth "
              "something, and the third found that the reporter's own script counts failures on rank 0 "
              "only \u2014 demonstrated by injecting a one-sided fault rather than argued.",
     "measured": "measured 2026-08-28",
     "langs": ["EN", "\u4e2d"] if (D / "n6565-body-zh.html").exists() else ["EN"],
     "tags": ["RCCL", "negative result", "ROCm#6565"]},
    {"href": "articles/" + Q_EN, "title": "A gate that costs 2 to 7 times and buys nothing",
     "blurb": "The bound that keeps vLLM's ROCm custom paged attention off gfx11 below gqa_ratio 3 is "
              "a performance heuristic, and it is inverted here: sixty cells, two vLLM versions, and "
              "the excluded band overlaps the admitted one on both.",
     "measured": "measured 2026-08-28",
     "langs": ["EN", "\u4e2d"] if (D / "gqa-body-zh.html").exists() else ["EN"],
     "tags": ["paged attention", "gqa_ratio", "vllm#54210"]},
    {"href": "articles/" + A_EN, "title": "Two consumer Radeons against one A100",
     "blurb": "Batch-1 decode of the same 31B model, each side on its healthy path: 1.48x apart at 1K, "
              "1.14x at 16K, 1.87x at 32K. The two ends have different causes and both are about "
              "splitting the work across two cards.",
     "measured": "measured 2026-08-26",
     "langs": ["EN", "\u4e2d"] if (D / "a100-body-zh.html").exists() else ["EN"],
     "tags": ["A100", "tensor parallelism", "bandwidth"]},
    {"href": "articles/" + S_EN, "title": "One boolean costs 71% on a Radeon and 61% on an A100",
     "blurb": "vLLM's own documented MTP assistant makes gemma-4-31B 36.9% faster at 1K of context and "
              "70.8% slower at 32K. One clause in an or chain reads speculation's second query row as "
              "\"not decode\" and gives up 120 of 128 workgroups.",
     "measured": "measured 2026-08-26",
     "langs": ["EN", "\u4e2d"] if (D / "spec-body-zh.html").exists() else ["EN"],
     "tags": ["speculative decoding", "Triton attention", "vllm#45450"]},
    {"href": "articles/" + L_EN, "title": "Loading weights was slower than the disk, twice over",
     "blurb": "A host-to-device copy only reads its source, but KFD asks for write access because the "
              "mapping is writable, and that breaks copy-on-write on every resident page. On one "
              "distro kernel each occurrence also cost a full second.",
     "measured": "measured 2026-08-23",
     "langs": ["EN", "\u4e2d"] if (D / "loader-body-zh.html").exists() else ["EN"],
     "tags": ["HMM", "copy-on-write", "ROCm#6523"]},
    {"href": "articles/" + M_EN, "title": "The fastest model here was written off at 15 tok/s",
     "blurb": "torch.compile was given twenty minutes and needed twenty-six, so the run was forced "
              "into eager mode and a 128-expert MoE was recorded at 15 tok/s. Compiled it is 107.8, "
              "and the flag had invented two qualitative findings on the way.",
     "measured": "measured 2026-07-25",
     "langs": ["EN", "\u4e2d"] if (D / "moe-body-zh.html").exists() else ["EN"],
     "tags": ["MoE", "torch.compile", "vllm#53892"]},
    {"href": "articles/" + W_EN, "title": "Twelve tokens a second was two problems",
     "blurb": "The same model family packaged two ways differs by 3.24x at 1K of context and 1.27x "
              "at 32K. Read as milliseconds rather than as a ratio, that is one flat cost under one "
              "growing cost, and each has its own upstream fix.",
     "measured": "measured 2026-08-27",
     "langs": ["EN", "\u4e2d"] if (D / "w4a16-body-zh.html").exists() else ["EN"],
     "tags": ["W4A16", "kernel selection", "vllm#40977"]},
    {"href": "articles/" + R_EN, "title": "The RCCL crash was never about RCCL",
     "blurb": "Two Radeons, tensor parallelism, and hipErrorIllegalState. The cause is four layers "
              "below RCCL, thirty lines of HIP reproduce it without RCCL at all, and for a virtual "
              "machine the fix is one line of configuration.",
     "measured": "reported 2026-07-26",
     "langs": ["EN", "中"] if (D / "rccl-body-zh.html").exists() else ["EN"],
     "tags": ["PCIe atomics", "hostcall", "ROCm#6520"]},
    {"href": "articles/" + H_EN,
     "title": "A hybrid-SSM model that decodes slower the longer you talk to it",
     "blurb": "Qwen3.6-27B falls from 12.1 to 4.2 tok/s between 500 and 32000 tokens. One kernel "
              "accounts for all of it, the custom kernel is unreachable three conditions over, and "
              "llama.cpp on the same machine rules out the driver.",
     "measured": "measured 2026-07-25", "langs": ["EN", "中"],
     "tags": ["hybrid SSM", "paged attention", "vllm#45916"]},
]}
(D / "articles.json").write_text(json.dumps(articles, ensure_ascii=False, indent=1))
built.append(page("index-body.html", lang="en", extra_css="index-extra.css",
                  title="dual-radeon-vllm · write-ups",
                  desc="Long-form write-ups from a repository of measurements on 2x RX 7900 XT under "
                       "ROCm and vLLM. Every figure is checked against the committed data it is drawn "
                       "from.",
                  out="index.html", labels=IDX_LABELS))
idx = OUT / "index.html"
if not CHECK:
    idx.write_text(idx.read_text().replace("__ARTICLES_JSON__",
                                           json.dumps(articles, ensure_ascii=False, indent=1)))

if CHECK:
    # the index carries a placeholder until after it is written, so compare the
    # rest of it and let the articles carry the byte-for-byte guarantee
    MISMATCH[:] = [m for m in MISMATCH if not m.startswith("index.html")]
    for m in MISMATCH:
        print("  MISMATCH", m)
    print(f"  {len(built)} pages checked, {len(MISMATCH)} differ from their source")
    sys.exit(1 if MISMATCH else 0)

for p in built:
    print(f"  {str(p.relative_to(OUT)):40s} {p.stat().st_size:,} bytes")

# no page may LOAD anything from outside the repository. A hyperlink is not an
# asset -- the articles cite trackers other than GitHub, and those are held to
# an allowlist instead, so a stray link still cannot creep in unnoticed.
LINK_HOSTS = {"github.com", "bugs.launchpad.net"}
for p in built:
    t = p.read_text()
    assets = (re.findall(r'\ssrc="(https?://[^"]+)"', t)
              + re.findall(r'<link[^>]+href="(https?://[^"]+)"', t))
    assert not assets, f"{p.name} loads external assets: {assets}"
    hosts = {u.split("/")[2] for u in
             re.findall(r'<a [^>]*href="(https?://[^"]+)"', t)}
    assert hosts <= LINK_HOSTS, f"{p.name} links to {sorted(hosts - LINK_HOSTS)}"
# the language pairs must agree on the parts that are not prose
for en, zh in ((H_EN, H_ZH), (R_EN, R_ZH), (W_EN, W_ZH), (M_EN, M_ZH),
                 (L_EN, L_ZH), (S_EN, S_ZH), (A_EN, A_ZH), (Q_EN, Q_ZH),
                 (N_EN, N_ZH), (X_EN, X_ZH)):
    a, b = OUT / "articles" / en, OUT / "articles" / zh
    if not b.exists():
        continue
    ta, tb = a.read_text(), b.read_text()
    grab = lambda t, i: re.search(r'<script type="application/json" id="%s">(.*?)</script>' % i, t, re.S)
    assert grab(ta, "figures").group(1) == grab(tb, "figures").group(1), f"{en}/{zh} figures diverged"
    sa = re.search(r"<script>\n\(function \(\).*?\n</script>", ta, re.S).group(0)
    sb = re.search(r"<script>\n\(function \(\).*?\n</script>", tb, re.S).group(0)
    assert sa == sb, f"{en}/{zh} scripts diverged"
    print(f"  {en} / {zh}: figures block and script identical")
print(f"  no page loads an external asset; links stay within {sorted(LINK_HOSTS)}")
