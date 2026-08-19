#!/usr/bin/env python3
"""
build_page.py — wrap the memo fragment into the standalone page served by GitHub Pages.

The memo exists in two forms and this script keeps them from drifting apart:

  landing-page.html   the content only — no <!doctype>, <html>, <head> or <body>.
                      Some hosts supply their own document skeleton and want a
                      fragment. Not tracked in git.

  docs/index.html     a complete HTML document, built from the fragment by this
                      script, with the document skeleton, viewport and character
                      set, and the Open Graph and Twitter card tags that LinkedIn
                      and Slack read to build a link preview. This is what
                      GitHub Pages serves.

Editing docs/index.html by hand works until the next time this runs, at which
point the edit is silently discarded. Edit the fragment and rebuild.

Usage:
    python scripts/build_page.py
"""
from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAGMENT = os.path.join(ROOT, "landing-page.html")
OUT = os.path.join(ROOT, "docs", "index.html")

BASE = "https://eliwitz34-dev.github.io/baseball-simulation-model/"
TITLE = ("A base-out simulation model for baseball outcomes, "
         "and its out-of-sample validation")
DESCRIPTION = (
    "A Monte Carlo model estimating the joint distribution of runs scored in a "
    "baseball game, with hierarchically shrunk parameter estimates and out-of-sample "
    "validation against a live benchmark forecast."
)


def build(fragment: str) -> str:
    style = re.search(r"<style>.*?</style>", fragment, re.S)
    fonts = re.search(r'<link rel="preconnect".*?display=swap">', fragment, re.S)
    if not style or not fonts:
        raise SystemExit("fragment is missing its <style> block or font links")
    body = fragment.split("</style>", 1)[1].strip()

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Base-Out Simulation Model — Eli Witz</title>
<meta name="description" content="{DESCRIPTION}">
<meta name="author" content="Eli Witz">
<link rel="canonical" href="{BASE}">

<!-- Link previews on LinkedIn, Slack and elsewhere are built from these. -->
<meta property="og:type" content="article">
<meta property="og:site_name" content="Eli Witz">
<meta property="og:title" content="{TITLE}">
<meta property="og:description" content="{DESCRIPTION}">
<meta property="og:url" content="{BASE}">
<meta property="og:image" content="{BASE}preview.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="Title card for the memo, with a confidence-interval plot.">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{TITLE}">
<meta name="twitter:description" content="{DESCRIPTION}">
<meta name="twitter:image" content="{BASE}preview.png">

<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><text y='13' font-size='13'>&#128202;</text></svg>">
{fonts.group(0)}
{style.group(0)}
</head>
<body>
{body}
</body>
</html>
"""


def main() -> int:
    if not os.path.exists(FRAGMENT):
        raise SystemExit(f"fragment not found: {FRAGMENT}")
    doc = build(open(FRAGMENT, encoding="utf-8").read())

    for required in ("<!doctype html>", "</head>", "<body>", "</body>", "</html>"):
        if required not in doc:
            raise SystemExit(f"generated document is missing {required}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w", encoding="utf-8").write(doc)
    print(f"wrote {os.path.relpath(OUT, ROOT)} ({len(doc):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
