"""MkDocs build hook — patches pygments html.escape(None) crash.

pymdownx.highlight passes filename=None to pygments HtmlFormatter when
code blocks have no title attribute. pygments 2.20+ calls html.escape()
on it, which crashes. This hook patches html.escape to handle None.
"""

import html

_orig_escape = html.escape


def _safe_escape(s, quote=True):
    if s is None:
        return ""
    return _orig_escape(s, quote)


html.escape = _safe_escape


def on_startup(**kwargs):
    pass
