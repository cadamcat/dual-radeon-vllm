# site/ — the sources the published pages are built from

`docs/` holds finished HTML that GitHub Pages serves. This directory holds what
those files are assembled from, so that a published page is never hand-edited.

```
python3 site/build.py            # write docs/index.html and docs/articles/*.html
python3 site/build.py --check    # build into memory, compare, exit non-zero on any drift
```

`--check` is what stops a published page from being edited in place: the head is
shared across every page, so a change made in one file would otherwise silently
diverge from the rest.

## What is where

| | |
|---|---|
| `src/article-head.html` | the shared head: theme, palette, components, the floating bar. Placeholders `__LANG__`, `__TITLE__`, `__DESC__`, `__LANG_NAV__`, `__NAVLABEL__`, `__T_AUTO__`, `__T_LIGHT__`, `__T_DARK__` |
| `src/<slug>-body.html` | one article's prose, its figure containers, its strings block, and its script |
| `src/<slug>-body-zh.html` | the same article in Chinese; `__SCRIPT__` is replaced with the English file's script verbatim |
| `src/<slug>-extra.css` | components only that article needs, spliced into the head |
| `src/figures*.json` | the data each article draws, embedded into the page as `<script type="application/json" id="figures">` |
| `src/genfig*.py` | derive those JSON files from `benchmarks/` and `docs/`; run them, do not hand-edit their output |
| `src/index-body.html` | the site root |

## The rules that make this worth doing

**A figure's numbers are derived, never typed.** `genfig.py` reads
`benchmarks/ledger.jsonl` and reuses `verify_doc_figures.py`'s own helpers for
anything computed, so a slope here is the number that file asserts.
`genfig-rccl.py` parses `docs/root-cause.md` tables, so an article cannot drift
from the document behind it.

**Series identity carries every axis the data varies.** Keying on
`(model, ctx)` merges the two arms of an A/B; adding patches but not `tp` merges
a model's TP=1 and TP=2 lines. Both happened while building the first article.

**Every page is self-contained.** No CDN, no external font, no external script.
`build.py` asserts this.

**Language pairs share everything except prose.** The figures block and the
script are inserted from one source, and `verify_doc_figures.py` asserts the two
published files carry them byte-identically, plus matching strings-table keys.

**Each figure states how a reader can check it.** `recomputable from repo` when
`verify_doc_figures.py` covers it, `derived from <doc>` when it is extracted,
`check with <command>` when the reader can run it, and a red marker when the raw
output is not in the repository at all.
