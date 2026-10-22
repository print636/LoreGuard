"""Render the fixed, public reviewer guide without dependencies or network use.

This deliberately small renderer accepts the syntax used by
docs/annotation-reader-guide.md: headings with optional {#anchor}, paragraphs,
flat ordered/unordered lists, quotes, bold/code/internal links, and native
<details><summary>...</summary> disclosures. Other HTML is escaped. It never
loads evaluation cases, annotations, model output or private answer material.
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "annotation-reader-guide.md"
ANCHOR = re.compile(r"^[a-z][a-z0-9-]*$")

STYLE = """
:root{color-scheme:light;--ink:#242338;--muted:#555268;--paper:#fff;
--ground:#f4f3f8;--accent:#58438b;--line:#d8d3e3;--soft:#f4f1fc;
font-family:'Microsoft YaHei UI','Microsoft YaHei','PingFang SC',system-ui,sans-serif;
font-size:18px;line-height:1.85;color:var(--ink);background:var(--ground)}
*{box-sizing:border-box}body{margin:0}a{color:var(--accent);text-underline-offset:4px}
a:focus-visible,summary:focus-visible{outline:3px solid var(--accent);outline-offset:4px}
.layout{max-width:1240px;display:grid;grid-template-columns:220px minmax(0,860px);
gap:40px;margin:32px auto;padding:0 24px 56px;align-items:start}
nav{position:sticky;top:24px;font-size:15px;line-height:1.7;min-width:0;padding-top:12px}
nav strong{display:block;margin:0 0 12px}nav a{display:block;min-height:44px;padding:9px 0}
nav details{margin:0;padding:0;border:0;border-radius:0}
nav summary{padding:8px 0;font-size:15px}
nav details[open]{padding-bottom:0}
nav details[open]>summary{margin-bottom:8px}
main{min-width:0;background:var(--paper);border:1px solid var(--line);
border-radius:12px;padding:36px 44px}h1{font-size:31px;line-height:1.45;margin:0 0 20px;
max-inline-size:22ch;text-wrap:balance}h2{font-size:24px;line-height:1.55;margin:44px 0 18px;
padding-top:24px;border-top:1px solid var(--line);scroll-margin-top:20px}
h3{font-size:20px;line-height:1.6;margin:26px 0 10px}p{margin:12px 0}
main>p:first-of-type{font-size:20px}ul,ol{padding-left:1.55em;margin:12px 0}
li{margin:10px 0}strong{font-weight:700}code{font-size:.92em;background:var(--soft);
border-radius:3px;padding:1px 4px;font-family:Consolas,'Microsoft YaHei',monospace}
blockquote{margin:20px 0;padding:14px 20px;border-left:3px solid var(--accent);
background:var(--soft)}blockquote p{margin:0}details{margin:16px 0;border:1px solid var(--line);
border-radius:8px;padding:0 18px;scroll-margin-top:20px}summary{font-weight:650;cursor:pointer;
padding:14px 0;min-height:48px;line-height:1.7}details[open]{padding-bottom:16px}
details[open]>summary{border-bottom:1px solid var(--line);margin-bottom:14px}
details p,details li{font-size:17px}p,li,summary,code,a{overflow-wrap:anywhere}
.skip{display:block;max-width:1240px;margin:auto;padding:14px 24px 0;font-size:15px}
footer{margin-top:32px;border-top:1px solid var(--line);padding-top:16px;font-size:15px;color:var(--muted)}
@media(max-width:1050px){.layout{display:block;max-width:850px;margin-top:12px;padding:0 16px 40px}
nav{position:static;padding:0 8px 18px}nav strong{margin-bottom:4px}nav a{display:inline-block;
padding:8px 12px 8px 0;margin-right:12px}main{padding:28px 32px}}
@media(max-width:600px){:root{font-size:17px;line-height:1.8}.layout{padding:0 8px 24px}
main{padding:24px 20px}nav{font-size:14px}nav a{min-height:44px;margin-right:8px;padding-right:6px}
h1{font-size:26px}h2{font-size:22px;margin-top:32px}h3{font-size:19px}
main>p:first-of-type{font-size:18px}details{padding-inline:14px}details p,details li{font-size:16px}}
@media print{@page{size:A4;margin:18mm 16mm}:root{font-size:12pt;background:#fff;color:#000}
.layout{display:block;max-width:none;margin:0;padding:0}.skip,nav{display:none}
main{padding:0;border:0}h1{font-size:22pt}h2{font-size:17pt}h3{font-size:14pt}
details p,details li{font-size:11pt}details{break-inside:avoid}h1,h2,h3,summary{break-after:avoid}
a{color:#000}footer{font-size:10pt}}
"""


def inline(value: str) -> str:
    """Escape source text, allowing only this guide's simple inline tokens."""

    escaped = html.escape(value)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    return re.sub(
        r"\[([^\]]+)\]\(#([a-z][a-z0-9-]*)\)",
        r'<a href="#\2">\1</a>',
        escaped,
    )


def render(source: str) -> str:
    blocks: list[str] = []
    paragraphs: list[str] = []
    navigation: list[tuple[str, str]] = []
    list_tag: str | None = None
    details_open = False
    used_ids: set[str] = set()
    h1_count = 0

    def flush() -> None:
        nonlocal list_tag
        if paragraphs:
            blocks.append("<p>" + inline(" ".join(paragraphs)) + "</p>")
            paragraphs.clear()
        if list_tag:
            blocks.append(f"</{list_tag}>")
            list_tag = None

    def register_id(value: str) -> None:
        if not ANCHOR.fullmatch(value) or value in used_ids:
            raise ValueError(f"Invalid or duplicate public-guide anchor: {value}")
        used_ids.add(value)

    for raw_line in source.splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        heading = re.fullmatch(r"(#{1,3}) (.+?)(?: \{#([a-z][a-z0-9-]*)\})?", line)
        if heading:
            flush()
            level = len(heading[1])
            title = heading[2]
            anchor = heading[3]
            if level == 1:
                h1_count += 1
            if anchor:
                register_id(anchor)
                attribute = f' id="{anchor}"'
                if level == 2:
                    navigation.append((anchor, title))
            else:
                attribute = ""
            blocks.append(f"<h{level}{attribute}>{inline(title)}</h{level}>")
            continue
        disclosure = re.fullmatch(
            r'<details(?: id="([a-z][a-z0-9-]*)")?><summary>(.+)</summary>', line
        )
        if disclosure:
            flush()
            if details_open:
                raise ValueError("Nested guide disclosures are not supported")
            anchor = disclosure[1]
            if anchor:
                register_id(anchor)
            attribute = f' id="{anchor}"' if anchor else ""
            blocks.append(f"<details{attribute}><summary>{inline(disclosure[2])}</summary>")
            details_open = True
            continue
        if line == "</details>":
            flush()
            if not details_open:
                raise ValueError("Unexpected disclosure closing tag")
            blocks.append("</details>")
            details_open = False
            continue
        item = re.fullmatch(r"(-|\d+\.) (.+)", line)
        if item:
            requested_tag = "ul" if item[1] == "-" else "ol"
            if paragraphs or list_tag != requested_tag:
                flush()
            if not list_tag:
                list_tag = requested_tag
                blocks.append(f"<{list_tag}>")
            blocks.append("<li>" + inline(item[2]) + "</li>")
            continue
        if line.startswith("> "):
            flush()
            blocks.append("<blockquote><p>" + inline(line[2:]) + "</p></blockquote>")
            continue
        if list_tag:
            flush()
        paragraphs.append(line)
    flush()
    if details_open or h1_count != 1:
        raise ValueError("Guide must have one title and closed disclosures")
    links = "".join(
        f'<a href="#{anchor}">{html.escape(title)}</a>' for anchor, title in navigation
    )
    body = "\n".join(blocks)
    return (
        '<!doctype html>\n<html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="color-scheme" content="light">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
        "style-src 'unsafe-inline'; script-src 'none'; connect-src 'none'; "
        "img-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'\">"
        '<title>LoreGuard 人工标注手册 · 好读版</title>'
        f"<style>{STYLE}</style></head><body>"
        '<a class="skip" href="#start">跳到三步说明</a>'
        '<div class="layout"><nav aria-label="按需要查阅"><details>'
        '<summary>遇到问题再查，不用先读完</summary>'
        f"{links}</details></nav><main>{body}"
        '<footer>这是操作说明，不是作答页。完全离线，无 AI 调用、上传或外部资源。</footer>'
        '</main></div></body></html>\n'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if output.suffix.lower() != ".html":
        parser.error("--output must be an .html file")
    if output.exists():
        parser.error("Output already exists; choose a new path instead of overwriting")
    content = render(SOURCE.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    print(f"Rendered the public reading guide: {output}")


if __name__ == "__main__":
    main()
