"""Turn generated Markdown into safe HTML: raw HTML is dropped, images are off, links are limited to
http(s), mailto and same-site paths, and every link opens without handing the page to the target."""

from __future__ import annotations

from markdown_it import MarkdownIt
from markdown_it.token import Token

_md = MarkdownIt("commonmark", {"html": True, "linkify": False}).disable("image")
_SCHEMES = ("http://", "https://", "mailto:")


def _href_allowed(href: str) -> bool:
    low = href.lower()
    if low.startswith(_SCHEMES):
        return True
    # a path or fragment: protocol-relative forms such as //host and /\host leave the site
    return low.startswith(("/", "#")) and low[1:2] not in ("/", "\\")


def _tidy_links(children: list[Token]) -> None:
    for c in children:
        if c.type != "link_open":
            continue
        if not _href_allowed(str(c.attrGet("href") or "")):
            c.attrs.pop("href", None)
        c.attrSet("rel", "noopener noreferrer nofollow")


def render_markdown(text: str) -> str:
    tokens = []
    for t in _md.parse(text):
        if t.type == "html_block":
            continue
        if t.children:
            t.children = [c for c in t.children if c.type != "html_inline"]
            _tidy_links(t.children)
        tokens.append(t)
    return _md.renderer.render(tokens, _md.options, {})
