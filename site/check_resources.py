"""Find external HTML/CSS dependencies, not citations or metadata URLs.

This is a static dependency check for the generated pages, not a JavaScript
execution sandbox. Local paths, fragments and embedded data remain allowed.
"""
from html.parser import HTMLParser
import re
from urllib.parse import urljoin


LOAD_RELS = {"stylesheet", "icon", "preload", "modulepreload", "prefetch",
             "manifest", "preconnect", "dns-prefetch"}
SRC_TAGS = {"script", "img", "source", "audio", "video", "track", "iframe",
            "embed"}
SVG_CSS_ATTRS = {"fill", "stroke", "filter", "clip-path", "mask", "cursor",
                 "marker-start", "marker-mid", "marker-end"}
STRING = r'''(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')'''
CSS = re.compile(
    r'/\*.*?\*/|\burl\(\s*(?P<url>' + STRING + r'|[^)]*)\s*\)'
    r'|@import\s+(?P<import>' + STRING + r')|' + STRING, re.I | re.S)


def css_urls(css):
    # Skip comments and ordinary strings (e.g. content: "url(...)"), which do
    # not load resources. @font-face uses the same url() syntax as backgrounds.
    for match in CSS.finditer(css):
        value = match.group("url") or match.group("import")
        if value:
            yield value.strip().strip("\"'")


def srcset_urls(value):
    # URLs are whitespace-delimited; embedded data may contain commas. Only a
    # trailing comma or the comma after a descriptor separates candidates.
    while value:
        value = value.lstrip(" \t\n\r\f,")
        if not value:
            break
        parts = value.split(None, 1)
        url = parts[0]
        value = parts[1] if len(parts) == 2 else ""
        yield url.rstrip(",")
        if not url.endswith(","):
            _, _, value = value.partition(",")


class Resources(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.refs = []
        self.base = None
        self.in_style = False

    def add(self, purpose, value):
        if value:
            self.refs.append((purpose, value.strip()))

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "base" and self.base is None and "href" in attrs:
            self.base = attrs["href"]
        if tag in SRC_TAGS:
            self.add(f"{tag}[src]", attrs.get("src"))
        if tag == "input" and (attrs.get("type") or "").lower() == "image":
            self.add("input[src]", attrs.get("src"))
        if tag == "video":
            self.add("video[poster]", attrs.get("poster"))
        if tag == "object":
            self.add("object[data]", attrs.get("data"))
        if tag in {"image", "use", "feimage"}:
            self.add(f"{tag}[href]", attrs.get("href") or attrs.get("xlink:href"))
        if tag == "link":
            rels = set((attrs.get("rel") or "").lower().split())
            if rels & LOAD_RELS:
                self.add("link[" + " ".join(sorted(rels & LOAD_RELS)) + "]", attrs.get("href"))
            if "preload" in rels:
                for url in srcset_urls(attrs.get("imagesrcset") or ""):
                    self.add("link[imagesrcset]", url)
        if tag in {"img", "source"}:
            for url in srcset_urls(attrs.get("srcset") or ""):
                self.add(f"{tag}[srcset]", url)
        for url in css_urls(attrs.get("style") or ""):
            self.add(f"{tag}[style]", url)
        for attr in SVG_CSS_ATTRS & attrs.keys():
            for url in css_urls(attrs[attr] or ""):
                self.add(f"{tag}[{attr}]", url)
        if tag == "style":
            self.in_style = True

    def handle_endtag(self, tag):
        if tag == "style":
            self.in_style = False

    def handle_data(self, data):
        if self.in_style:
            for url in css_urls(data):
                self.add("style", url)


def external_resources(html):
    parser = Resources()
    parser.feed(html)
    parser.close()
    errors = []
    for purpose, value in parser.refs:
        resolved = urljoin(parser.base or "", value)
        if re.match(r"(?:https?:)?//", resolved, re.I):
            errors.append(f"{purpose} loads {resolved}")
    return errors
