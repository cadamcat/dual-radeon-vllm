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

# The masthead chips are data. Every machine string -- a card name, a version,
# an issue id, a date -- is written once in chips.json and rendered into both
# language versions from here, so the two cannot disagree about the hardware.
# chipwords-*.json holds only the connectives and the word order.
CHIPS = json.loads((D / "chips.json").read_text())
CHIPWORDS = {l: json.loads((D / f"chipwords-{l}.json").read_text()) for l in ("en", "zh")}


def chips_html(slug, lang):
    out = []
    for n, c in enumerate(CHIPS[slug]):
        t = CHIPWORDS[lang][c["w"]]
        for i, v in enumerate(c["v"]):
            assert "{%d}" % i in t, f'{slug}: {c["w"]} has no slot {i}'
            t = t.replace("{%d}" % i, v)
        assert "{" not in t, f'{slug}: {c["w"]} left a slot unfilled in {lang}'
        # the type is what the dot means, so it is also spelled out: colour on
        # its own is not a label
        k = "k" + c["kind"][0].upper() + c["kind"][1:]
        out.append(f'    <span class="chip k-{c["kind"]}" title="{CHIPWORDS[lang][k]}"'
                   f' style="--i:{n}">{t}</span>')
    return "\n".join(out)


def lang_nav(current, en, zh):
    """Both links always present; the current one is not a link to itself."""
    rows = []
    for code, label, href in (("en", "EN", en), ("zh", "中", zh)):
        cur = ' aria-current="page"' if code == current else ""
        rows.append(f'  <a class="lang" href="{href}" hreflang="{code}"{cur}>{label}</a>')
    return "\n".join(rows)


def page(body, *, lang, title, desc, out, extra_css=None, nav=None, labels, figures=None,
         script_from=None, subs=None):
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
    # the slug is the published filename, so the chips cannot be attached to the
    # wrong article without also publishing it under the wrong name
    slug = out.split("/")[-1].replace(".zh.html", "").replace(".html", "")
    short = "zh" if lang.startswith("zh") else "en"
    if "__CHIPS__" in b:
        b = b.replace("__CHIPS__", chips_html(slug, short))
    for k, v in (subs or {}).items():
        assert k in b, f"{body} has no {k} slot"
        b = b.replace(k, v)
    left = re.findall(r"__[A-Z][A-Z_]*__", b)
    assert not left, f"{body} left {left} unfilled"
    p = OUT / out
    TITLES[out] = title
    # the subtitle is the sentence the page itself prints, read back rather
    # than retyped, so the index cannot introduce a third version of it
    m = re.search(r'<p class="sub">(.*?)</p>', b)
    SUBS[out] = m.group(1) if m else None
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
TITLES = {}
SUBS = {}
EN_LABELS = ("Language and colour theme", "Match system", "Light", "Dark")
ZH_LABELS = ("语言与配色主题", "跟随系统",
             "浅色", "深色")
IDX_LABELS = ("Language and colour theme", "Match system", "Light", "Dark")
IDX_LABELS_ZH = ZH_LABELS

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
Z_EN, Z_ZH = "rdna3-second-class.html", "rdna3-second-class.zh.html"

built = []
built.append(page("article-body.html", lang="en", figures="figures.json",
                  extra_css="article-extra.css",
                  title='One kernel accounts for the whole 12.1 to 4.2 tok/s fall',
                  desc="Qwen3.6-27B falls from 12.1 to 4.2 tok/s between 500 and 32000 tokens of "
                       "context on 2x RX 7900 XT. Attribution, mechanism, control, and the upstream "
                       "fix, measured.",
                  out="articles/" + H_EN, nav=lang_nav("en", H_EN, H_ZH), labels=EN_LABELS))
built.append(page("article-body-zh.html", lang="zh-CN", figures="figures.json",
                  extra_css="article-extra.css", script_from="article-body.html",
                  title='一个 kernel 就解释了 12.1 到 4.2 tok/s 的全部下滑',
                  desc="Qwen3.6-27B 在 2× RX 7900 XT 上，上下文从 "
                       "500 到 32000 token，解码由 12.1 掉到 4.2 tok/s。"
                       "归因、机理、对照，以及上游"
                       "修复的实测。",
                  out="articles/" + H_ZH, nav=lang_nav("zh", H_EN, H_ZH), labels=ZH_LABELS))
built.append(page("rccl-body.html", lang="en", figures="figures-rccl.json",
                  extra_css="rccl-extra.css",
                  title='No PCIe atomics, no hostcall buffer, and every collective fails at dispatch',
                  desc="Two Radeons, tensor parallelism, and hipErrorIllegalState. The cause is four "
                       "layers below RCCL: no PCIe atomics, no hostcall buffer, refused dispatch. "
                       "Thirteen hypotheses, and a one-line VM fix.",
                  out="articles/" + R_EN, nav=lang_nav("en", R_EN, R_ZH), labels=EN_LABELS))
if (D / "rccl-body-zh.html").exists():
    built.append(page("rccl-body-zh.html", lang="zh-CN", figures="figures-rccl.json",
                      extra_css="rccl-extra.css", script_from="rccl-body.html",
                      title='没有 PCIe atomics 就没有 hostcall buffer，每一次集合通信都在 dispatch 处失败',
                      desc="两张 Radeon、张量并行，以及 "
                           "hipErrorIllegalState。根因在 RCCL 下面四层："
                           "没有 PCIe atomics，就没有 hostcall buffer，"
                           "dispatch 被拒。",
                      out="articles/" + R_ZH, nav=lang_nav("zh", R_EN, R_ZH), labels=ZH_LABELS))

built.append(page("w4a16-body.html", lang="en", figures="figures-w4a16.json",
                  extra_css="w4a16-extra.css",
                  title='A flat 60 ms per step, under a cost that grows with context',
                  desc="One 27B model, two independent costs: a flat 60 ms per decode step from the "
                       "W4A16 linear kernel, and a paged-attention term that grows with context. "
                       "Separated by an A/B, each fixed upstream by someone else.",
                  out="articles/" + W_EN, nav=lang_nav("en", W_EN, W_ZH), labels=EN_LABELS))
if (D / "w4a16-body-zh.html").exists():
    built.append(page("w4a16-body-zh.html", lang="zh-CN", figures="figures-w4a16.json",
                      extra_css="w4a16-extra.css", script_from="w4a16-body.html",
                      title='每步固定 60 ms，压在一笔随上下文增长的开销下面',
                      desc="一个 27B 模型上叠着两笔独立开销：W4A16 线性 "
                           "kernel 每步固定 60 ms，以及随上下文增长的 paged "
                           "attention。用一次 A/B 把它们分开，两个修复都是"
                           "别人做的。",
                      out="articles/" + W_ZH, nav=lang_nav("zh", W_EN, W_ZH), labels=ZH_LABELS))

built.append(page("moe-body.html", lang="en", figures="figures-moe.json",
                  extra_css="moe-extra.css",
                  title='Eager mode recorded 107.8 tok/s as 15, and invented two findings on the way',
                  desc="A 128-expert MoE recorded at 15 tok/s under --enforce-eager decodes at 107.8 "
                       "compiled. The flag also fabricated an asymmetric power draw and a "
                       "context-independent rate, both read as architecture.",
                  out="articles/" + M_EN, nav=lang_nav("en", M_EN, M_ZH), labels=EN_LABELS))
if (D / "moe-body-zh.html").exists():
    built.append(page("moe-body-zh.html", lang="zh-CN", figures="figures-moe.json",
                      extra_css="moe-extra.css", script_from="moe-body.html",
                      title='eager 模式把 107.8 tok/s 记成 15，顺手还造出两个结论',
                      desc="一个 128 专家的 MoE 在 --enforce-eager 下记作 15 "
                           "tok/s，编译之后是 107.8。这个开关还伪造了功耗左右"
                           "不对称和吞吐与上下文无关两个现象，都被当成了架构结论。",
                      out="articles/" + M_ZH, nav=lang_nav("zh", M_EN, M_ZH), labels=ZH_LABELS))

built.append(page("loader-body.html", lang="en", figures="figures-loader.json",
                  extra_css="loader-extra.css",
                  title='A read-only copy asks for write access, and one kernel charged a second for it',
                  desc="A permission read off the VMA makes every host-to-device copy break "
                       "copy-on-write, and a split kernel backport turned each occurrence into a "
                       "one-second timeout. Two effects, one reproducer, three kernel states.",
                  out="articles/" + L_EN, nav=lang_nav("en", L_EN, L_ZH), labels=EN_LABELS))
if (D / "loader-body-zh.html").exists():
    built.append(page("loader-body-zh.html", lang="zh-CN", figures="figures-loader.json",
                      extra_css="loader-extra.css", script_from="loader-body.html",
                      title='只读的拷贝去申请写权限，而某个内核为此每次多收一秒',
                      desc="从 VMA 取权限让每次 host→device 拷贝都破坏 "
                           "copy-on-write；而一个被拆开的内核 backport 把每一次"
                           "触发都变成一秒的超时。两个效应，一个复现器，三种内核状态。",
                      out="articles/" + L_ZH, nav=lang_nav("zh", L_EN, L_ZH), labels=ZH_LABELS))

built.append(page("spec-body.html", lang="en", figures="figures-spec.json",
                  extra_css="spec-extra.css",
                  title="Speculation's second query row costs 120 of 128 workgroups",
                  desc="Speculative decoding is +36.9% at 1K of context and -70.8% at 32K, because "
                       "max_seqlen_q > 1 drops the Triton attention kernel from 128 workgroups to 8. "
                       "Measured on two vendors.",
                  out="articles/" + S_EN, nav=lang_nav("en", S_EN, S_ZH), labels=EN_LABELS))
if (D / "spec-body-zh.html").exists():
    built.append(page("spec-body-zh.html", lang="zh-CN", figures="figures-spec.json",
                      extra_css="spec-extra.css", script_from="spec-body.html",
                      title='投机的第二行 query，要付掉 128 个 workgroup 里的 120 个',
                      desc="投机解码在 1K 上下文是 +36.9%，在 32K 上是 "
                           "-70.8%，因为 max_seqlen_q > 1 把 Triton 注意力 "
                           "kernel 从 128 个 workgroup 降到 8 个。两个厂商都测了。",
                      out="articles/" + S_ZH, nav=lang_nav("zh", S_EN, S_ZH), labels=ZH_LABELS))

built.append(page("a100-body.html", lang="en", figures="figures-a100.json",
                  extra_css="a100-extra.css",
                  title='Two consumer RX 7900 XTs against one A100',
                  desc="On batch-1 decode of the same 31B model the A100 is 1.48x ahead at 1K, 1.14x "
                       "at 16K and 1.87x at 32K. The gap is U-shaped, and both ends are about tensor "
                       "parallelism rather than about the silicon.",
                  out="articles/" + A_EN, nav=lang_nav("en", A_EN, A_ZH), labels=EN_LABELS))
if (D / "a100-body-zh.html").exists():
    built.append(page("a100-body-zh.html", lang="zh-CN", figures="figures-a100.json",
                      extra_css="a100-extra.css", script_from="a100-body.html",
                      title='两张消费级 Radeon 7900 XT 对一张 A100',
                      desc="同一个 31B 模型的 batch-1 解码，A100 在 1K 上领先 "
                           "1.48×，16K 上 1.14×，32K 上 1.87×。差距是 U 形的，"
                           "而两端都关于张量并行，不关于硅片本身。",
                      out="articles/" + A_ZH, nav=lang_nav("zh", A_EN, A_ZH), labels=ZH_LABELS))

built.append(page("gqa-body.html", lang="en", figures="figures-gqa.json",
                  extra_css="gqa-extra.css",
                  title='The excluded kernel wins all sixty cells, and the two bands overlap',
                  desc="vLLM's ROCm custom paged attention is gated off below gqa_ratio 3 on gfx11. "
                       "In the excluded range it is 1.70x to 7.28x faster than the fallback, in every "
                       "one of sixty measured cells.",
                  out="articles/" + Q_EN, nav=lang_nav("en", Q_EN, Q_ZH), labels=EN_LABELS))
if (D / "gqa-body-zh.html").exists():
    built.append(page("gqa-body-zh.html", lang="zh-CN", figures="figures-gqa.json",
                      extra_css="gqa-extra.css", script_from="gqa-body.html",
                      title='被排除的 kernel 在六十个格子里全赢，而两个区间还互相重叠',
                      desc="vLLM 的 ROCm 定制 paged attention 在 gfx11 上被 "
                           "gqa_ratio 3 这道门挡在外面。而在被排除的区间里，"
                           "它比兜底路径快 1.70–7.28 倍，六十个格子无一例外。",
                      out="articles/" + Q_ZH, nav=lang_nav("zh", Q_EN, Q_ZH), labels=ZH_LABELS))

built.append(page("n6565-body.html", lang="en", figures="figures-6565.json",
                  extra_css="n6565-extra.css",
                  title="The reporter's own script counts rank 0, shown by injecting a one-sided fault",
                  desc="135 clean communicator initialisations say almost nothing on their own. What "
                       "makes a negative result usable: a sweep that could have exposed the defect, a "
                       "stated contrast, and finding what your instrument is blind to.",
                  out="articles/" + N_EN, nav=lang_nav("en", N_EN, N_ZH), labels=EN_LABELS))
if (D / "n6565-body-zh.html").exists():
    built.append(page("n6565-body-zh.html", lang="zh-CN", figures="figures-6565.json",
                      extra_css="n6565-extra.css", script_from="n6565-body.html",
                      title='上报者自己的脚本只数 rank 0，用一个单边故障演示出来',
                      desc="135 次干净的通信器初始化本身几乎什么也说明不了。"
                           "让一个否定结论变得有用的三件事：一次本可以暴露它的"
                           "扫描、把差异摆明，以及找出你的仪器看不见什么。",
                      out="articles/" + N_ZH, nav=lang_nav("zh", N_EN, N_ZH), labels=ZH_LABELS))

built.append(page("measure-body.html", lang="en", figures="figures-measure.json",
                  extra_css="measure-extra.css",
                  title='Two harnesses agree to 0.44 %, and the first of four runs read 31 % low',
                  desc="Two harnesses agree to 0.44%, but only once the machine is warm: the first of "
                       "four identical runs read 31% low. Why every point carries its run count and "
                       "range, and why the range rather than a standard deviation.",
                  out="articles/" + X_EN, nav=lang_nav("en", X_EN, X_ZH), labels=EN_LABELS))
if (D / "measure-body-zh.html").exists():
    built.append(page("measure-body-zh.html", lang="zh-CN", figures="figures-measure.json",
                      extra_css="measure-extra.css", script_from="measure-body.html",
                      title='两套 harness 吻合到 0.44 %，而四次运行里的第一次低了 31 %',
                      desc="两套 harness 吻合到 0.44%，但前提是机器已经热了："
                           "四次相同运行里的第一次低了 31%。为什么每个点都带 run "
                           "数和极差，以及为什么报极差而不是标准差。",
                      out="articles/" + X_ZH, nav=lang_nav("zh", X_EN, X_ZH), labels=ZH_LABELS))

built.append(page("rdna3-body.html", lang="en", figures="figures-rdna3.json",
                  extra_css="rdna3-extra.css",
                  title="Three of eight findings are RDNA3's, and measuring another vendor removed two",
                  desc="Eight findings sorted by what the evidence supports. Three are "
                       "architecture-specific, one is AMD-wide, two were proved vendor-neutral by "
                       "measuring another vendor, and one is not about the GPU at all.",
                  out="articles/" + Z_EN, nav=lang_nav("en", Z_EN, Z_ZH), labels=EN_LABELS))
if (D / "rdna3-body-zh.html").exists():
    built.append(page("rdna3-body-zh.html", lang="zh-CN", figures="figures-rdna3.json",
                      extra_css="rdna3-extra.css", script_from="rdna3-body.html",
                      title='八个结论里只有三个是 RDNA3 的，而测另一个厂商拿掉了两个',
                      desc="八个结论按证据支持的范围分类。三个是架构特有的，"
                           "一个是 AMD 全线的，两个通过在另一个厂商上实测被证明"
                           "与厂商无关，还有一个根本不是 GPU 的事。",
                      out="articles/" + Z_ZH, nav=lang_nav("zh", Z_EN, Z_ZH), labels=ZH_LABELS))

# ---- the index, as data ----------------------------------------------------
# Nothing here is retyped. The titles are the ones the pages were built with,
# the chips come from chips.json, the dates and what kind of claim they are come
# from each article's chip of kind "date", and the one-line "what this
# establishes" comes from the synthesis article's own figure data -- so the
# synthesis and the index cannot describe a finding two different ways.
RD = json.loads((D / "figures-rdna3.json").read_text())["fig1"]["findings"]
ESTABLISHES = {f["slug"]: {"en": f["mechanism"], "zh": f["mechanism_zh"]} for f in RD}
# the three the synthesis does not classify: it is one of them itself, and the
# other two are about method rather than about a mechanism in the stack
ESTABLISHES.update({
    "reporting-a-non-reproduction": {
        "en": "a clean run says something only once you have shown what the "
              "instrument is blind to",
        "zh": "一次干净的运行，只有在你"
              "说清了仪器看不见什么之后"
              "才说明问题"},
    "measuring-decode": {
        "en": "a decode number needs a run count and a range beside it, because this "
              "machine's first run is not its steady state",
        "zh": "一个解码数字旁边得写上运"
              "行次数和极差，因为这台机"
              "器的第一次运行不是稳态"},
    "rdna3-second-class": {
        "en": "the useful unit is the single finding, not the architecture: each of "
              "the eight has its own scope and its own evidence",
        "zh": "有用的单位是单个结论，不"
              "是整个架构：八个里每一个"
              "都有自己的适用范围和自己"
              "的证据"},
})

ART = [
 {"slug": "rdna3-second-class", "en": Z_EN, "zh": Z_ZH, "zhbody": "rdna3-body-zh.html",
  "tags": ["synthesis", "RDNA3", "ROCm"],
  "blurb": {
   "en": "This repository's own summary says RDNA3 is a second-class citizen in ROCm's kernel "
         "ecosystem. Sorting eight findings by what the evidence supports, three of them are "
         "that, and the measurements that mattered most took findings off the list.",
   "zh": "仓库自己的总结说 RDNA3 是 ROCm kernel "
         "生态里的二等公民。把八个结"
         "论按证据支持的范围排一排，"
         "其中三个是；而最关键的几次"
         "测量都是把结论从表上拿掉的。"}},
 {"slug": "measuring-decode", "en": X_EN, "zh": X_ZH, "zhbody": "measure-body-zh.html",
  "tags": ["methodology", "calibration", "nondeterminism"],
  "blurb": {
   "en": "The two harnesses agree to 0.44%, and finding that out took four identical runs the "
         "first of which read 31% low. What every point carries, where the chart-grade line sits, "
         "and one cell where the range and the standard deviation disagree about the direction.",
   "zh": "两套 harness 吻合到 0.44%，而弄清这一"
         "点用了四次相同的运行，其中"
         "第一次低了 31%。每个点带什么、"
         "制图级的线划在哪里，以及有"
         "一格里极差和标准差对方向的"
         "判断相反。"}},
 {"slug": "reporting-a-non-reproduction", "en": N_EN, "zh": N_ZH, "zhbody": "n6565-body-zh.html",
  "tags": ["RCCL", "negative result", "ROCm#6565"],
  "blurb": {
   "en": "A clean run is the least useful sentence on a bug tracker. Three things make it worth "
         "something, and the third found that the reporter's own script counts failures on rank 0 "
         "only — demonstrated by injecting a one-sided fault rather than argued.",
   "zh": "一次干净的运行是 bug tracker 上最没"
         "用的一句话。有三件事能让它"
         "值点钱，而第三件找出了上报"
         "者自己的脚本只数 rank 0 的失败"
         "—— 是注入一个单边故障演示"
         "出来的，不是论证出来的。"}},
 {"slug": "gqa-gate-costs-nothing", "en": Q_EN, "zh": Q_ZH, "zhbody": "gqa-body-zh.html",
  "tags": ["paged attention", "gqa_ratio", "vllm#54210"],
  "blurb": {
   "en": "The bound that keeps vLLM's ROCm custom paged attention off gfx11 below gqa_ratio 3 is "
         "a performance heuristic, and it is inverted here: sixty cells, two vLLM versions, and "
         "the excluded band overlaps the admitted one on both.",
   "zh": "把 vLLM 的 ROCm 定制 paged attention 在 gfx11 上挡"
         "在 gqa_ratio 3 之外的那道界是一个性"
         "能启发式，而它在这里是反的："
         "六十个格子、两个 vLLM 版本，被"
         "排除的区间在两边都与被放行"
         "的区间重叠。"}},
 {"slug": "a100-vs-two-radeons", "en": A_EN, "zh": A_ZH, "zhbody": "a100-body-zh.html",
  "tags": ["A100", "tensor parallelism", "bandwidth"],
  "blurb": {
   "en": "Batch-1 decode of the same 31B model, each side on its healthy path: 1.48x apart at 1K, "
         "1.14x at 16K, 1.87x at 32K. The two ends have different causes and both are about "
         "splitting the work across two cards.",
   "zh": "同一个 31B 模型的 batch-1 解码，两边"
         "都跑在各自的健康路径上：1K "
         "相差 1.48×，16K 是 1.14×，32K 是 1.87×。"
         "两端的成因不同，而都关于把"
         "工作拆到两张卡上。"}},
 {"slug": "speculative-decoding-net-loss", "en": S_EN, "zh": S_ZH, "zhbody": "spec-body-zh.html",
  "tags": ["speculative decoding", "Triton attention", "vllm#45450"],
  "blurb": {
   "en": "vLLM's own documented MTP assistant makes gemma-4-31B 36.9% faster at 1K of context and "
         "70.8% slower at 32K. One clause in an or chain reads speculation's second query row as "
         "\"not decode\" and gives up 120 of 128 workgroups.",
   "zh": "vLLM 自己文档里的 MTP 助手让 gemma-4-31B "
         "在 1K 上快 36.9%，在 32K 上慢 70.8%。一串"
         " or 里的一个子句把投机的第二"
         "行 query 读成了“不是 decode”，于是"
         "交出了 128 个 workgroup 里的 120 个。"}},
 {"slug": "weight-loading-19x", "en": L_EN, "zh": L_ZH, "zhbody": "loader-body-zh.html",
  "tags": ["HMM", "copy-on-write", "ROCm#6523"],
  "blurb": {
   "en": "A host-to-device copy only reads its source, but KFD asks for write access because the "
         "mapping is writable, and that breaks copy-on-write on every resident page. On one "
         "distro kernel each occurrence also cost a full second.",
   "zh": "host→device 的拷贝只读源数据，但 KFD "
         "因为映射可写就去申请写权限，"
         "于是每一个驻留页的 copy-on-write 都被"
         "破坏。在某个发行版内核上，"
         "每触发一次还要多花整整一秒。"}},
 {"slug": "moe-written-off-by-eager", "en": M_EN, "zh": M_ZH, "zhbody": "moe-body-zh.html",
  "tags": ["MoE", "torch.compile", "vllm#53892"],
  "blurb": {
   "en": "torch.compile was given twenty minutes and needed twenty-six, so the run was forced "
         "into eager mode and a 128-expert MoE was recorded at 15 tok/s. Compiled it is 107.8, "
         "and the flag had invented two qualitative findings on the way.",
   "zh": "torch.compile 只给了二十分钟，而它需"
         "要二十六，于是这次运行被压"
         "进 eager 模式，一个 128 专家的 MoE 被"
         "记作 15 tok/s。编译之后是 107.8，而"
         "这个开关路上还凭空造出了两"
         "个定性结论。"}},
 {"slug": "w4a16-two-problems", "en": W_EN, "zh": W_ZH, "zhbody": "w4a16-body-zh.html",
  "tags": ["W4A16", "kernel selection", "vllm#40977"],
  "blurb": {
   "en": "The same model family packaged two ways differs by 3.24x at 1K of context and 1.27x "
         "at 32K. Read as milliseconds rather than as a ratio, that is one flat cost under one "
         "growing cost, and each has its own upstream fix.",
   "zh": "同一个模型家族的两种打包，"
         "在1K 上差 3.24×，在32K 上差 1.27×。"
         "把它当毫秒而不是倍数来读，"
         "就是一笔固定开销压在一笔增"
         "长开销下面，而两者各自有自"
         "己的上游修复。"}},
 {"slug": "rccl-atomics-hostcall", "en": R_EN, "zh": R_ZH, "zhbody": "rccl-body-zh.html",
  "tags": ["PCIe atomics", "hostcall", "ROCm#6520"],
  "blurb": {
   "en": "Two Radeons, tensor parallelism, and hipErrorIllegalState. The cause is four layers "
         "below RCCL, thirty lines of HIP reproduce it without RCCL at all, and for a virtual "
         "machine the fix is one line of configuration.",
   "zh": "两张 Radeon、张量并行，以及 "
         "hipErrorIllegalState。根因在 RCCL 下面四层，"
         "三十行 HIP 不用 RCCL 就能复现，而"
         "对一台虚拟机来说修复是一行"
         "配置。"}},
 {"slug": "hybrid-ssm-collapse", "en": H_EN, "zh": H_ZH, "zhbody": "article-body-zh.html",
  "tags": ["hybrid SSM", "paged attention", "vllm#45916"],
  "blurb": {
   "en": "Qwen3.6-27B falls from 12.1 to 4.2 tok/s between 500 and 32000 tokens. One kernel "
         "accounts for all of it, the custom kernel is unreachable three conditions over, and "
         "llama.cpp on the same machine rules out the driver.",
   "zh": "Qwen3.6-27B 在 500 到 32000 token 之间从 12.1 掉到 "
         "4.2 tok/s。一个 kernel 就解释了全部，"
         "定制 kernel 差三个条件都进不去，"
         "而同一台机器上的 llama.cpp 排除了"
         "驱动。"}},
]

DKIND = {"measured", "reported", "reviewed"}
records = []
for a in ART:
    dchip = [c for c in CHIPS[a["slug"]] if c["kind"] == "date"]
    assert len(dchip) == 1, f'{a["slug"]}: expected one date chip, got {len(dchip)}'
    assert dchip[0]["tl"] in DKIND, f'{a["slug"]}: unknown timeline kind {dchip[0]["tl"]}'
    zh = (D / a["zhbody"]).exists()
    records.append({
        "slug": a["slug"],
        "href": {"en": "articles/" + a["en"], "zh": "articles/" + a["zh"]},
        "title": {"en": TITLES["articles/" + a["en"]], "zh": TITLES["articles/" + a["zh"]]},
        "sub": {"en": SUBS["articles/" + a["en"]], "zh": SUBS["articles/" + a["zh"]]},
        "blurb": a["blurb"],
        "establishes": ESTABLISHES[a["slug"]],
        "dates": dchip[0]["v"], "date": max(dchip[0]["v"]), "kind": dchip[0]["tl"],
        "chips": CHIPS[a["slug"]],
        "langs": ["EN", "中"] if zh else ["EN"],
        "tags": a["tags"]})
# Newest first, by the last date the article's own date chip carries -- which is
# the only date a card prints. Ties keep the order ART lists them in, which is
# the order they were written, because sort is stable.
records.sort(key=lambda r: r["date"], reverse=True)
articles = {"articles": records}
AJSON = json.dumps(articles, ensure_ascii=False, indent=1)
if CHECK:
    if (D / "articles.json").read_text() != AJSON:
        MISMATCH.append("articles.json: committed copy differs from its source")
else:
    (D / "articles.json").write_text(AJSON)

I_EN, I_ZH = "index.html", "index.zh.html"
IDX_SUBS = {"__ARTICLES_JSON__": AJSON}
built.append(page("index-body.html", lang="en", extra_css="index-extra.css",
                  figures="figures-index.json",
                  title="dual-radeon-vllm · write-ups",
                  desc="Long-form write-ups from a repository of measurements on 2x RX 7900 XT under "
                       "ROCm and vLLM. Every figure is checked against the committed data it is drawn "
                       "from.",
                  out=I_EN, nav=lang_nav("en", I_EN, I_ZH), labels=IDX_LABELS,
                  subs=dict(IDX_SUBS, __CHIPWORDS_JSON__=json.dumps(
                      CHIPWORDS["en"], ensure_ascii=False, indent=1))))
built.append(page("index-body-zh.html", lang="zh-CN", extra_css="index-extra.css",
                  figures="figures-index.json", script_from="index-body.html",
                  title="dual-radeon-vllm · 实测长文",
                  desc="一个在 2× RX 7900 XT 上做 ROCm 与 vLLM "
                       "实测的仓库，其中的长文。"
                       "每张图都对照它所出自的已"
                       "入库数据被检查过。",
                  out=I_ZH, nav=lang_nav("zh", I_EN, I_ZH), labels=IDX_LABELS_ZH,
                  subs=dict(IDX_SUBS, __CHIPWORDS_JSON__=json.dumps(
                      CHIPWORDS["zh"], ensure_ascii=False, indent=1))))


if CHECK:
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
                 (N_EN, N_ZH), (X_EN, X_ZH), (Z_EN, Z_ZH), (I_EN, I_ZH)):
    sub = OUT if en == I_EN else OUT / "articles"
    a, b = sub / en, sub / zh
    if not b.exists():
        continue
    ta, tb = a.read_text(), b.read_text()
    grab = lambda t, i: re.search(r'<script type="application/json" id="%s">(.*?)</script>' % i, t, re.S)
    data = "figures" if en != I_EN else "articles"
    assert grab(ta, data).group(1) == grab(tb, data).group(1), f"{en}/{zh} data diverged"
    sa = re.search(r"<script>\n\(function \(\).*?\n</script>", ta, re.S).group(0)
    sb = re.search(r"<script>\n\(function \(\).*?\n</script>", tb, re.S).group(0)
    assert sa == sb, f"{en}/{zh} scripts diverged"
    print(f"  {en} / {zh}: data block and script identical")
print(f"  no page loads an external asset; links stay within {sorted(LINK_HOSTS)}")
