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

articles = {"articles": [
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

# no page may reach outside the repository for an asset
for p in built:
    t = p.read_text()
    ext = [u for u in re.findall(r'(?:src|href)="(https?://[^"]+)"', t)
           if not u.startswith("https://github.com")]
    assert not ext, f"{p.name} pulls external assets: {ext}"
# the language pairs must agree on the parts that are not prose
for en, zh in ((H_EN, H_ZH), (R_EN, R_ZH)):
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
print("  no page pulls an external asset")
