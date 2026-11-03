"""Turn generated Markdown into safe HTML: raw HTML is dropped, images are off, unsafe link schemes are refused."""

from __future__ import annotations

from markdown_it import MarkdownIt

_md = MarkdownIt("commonmark", {"html": True, "linkify": False}).disable("image")


def render_markdown(text: str) -> str:
    tokens = []
    for t in _md.parse(text):
        if t.type == "html_block":
            continue
        if t.children:
            t.children = [c for c in t.children if c.type != "html_inline"]
        tokens.append(t)
    return _md.renderer.render(tokens, _md.options, {})
