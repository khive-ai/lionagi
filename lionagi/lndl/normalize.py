# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""LNDL text normalization — auto-fix common model-invented syntax errors.

Models trained on XML/HTML/JSON sometimes drift into related-but-wrong forms.
This module catches the most common drifts and rewrites them into valid LNDL
before the parser runs, so the model isn't penalized for surface mistakes.

Ported from krons.lndl.fuzzy.
"""

from __future__ import annotations

import re

__all__ = ("normalize_lndl_text",)

_XML_ATTR_RE = re.compile(r'\b\w+=["\'][^"\']*["\']')


def normalize_lndl_text(text: str) -> str:
    """Normalize model-invented syntax before lexing.

    Handles:
    - Curly-brace tags: ``{lact X}fn(){/lact}`` or ``{lact X}fn()</lact>``
      → ``<lact X>fn()</lact>``
    - XML attributes: ``<lact name="X" type="Y">`` → ``<lact X>``
    - Tag opening missing ``>`` before body when followed by ``</tag>``:
      ``<lact a fn()</lact>`` is NOT auto-fixed (ambiguous); the retry path
      catches this and asks the model to fix it.
    """
    if not text:
        return text

    # 1) Curly-brace tags → angle-bracket tags
    text = re.sub(r"\{(lvar|lact)(\s+[^}]*)\}", r"<\1\2>", text)
    text = re.sub(r"\{/(lvar|lact)\}", r"</\1>", text)

    # 2) XML attributes inside opening tags → strip and promote `name=` to a token
    def _clean_tag(m: re.Match) -> str:
        tag = m.group(1)
        body = m.group(2)

        attrs = dict(re.findall(r'(\w+)=["\']([^"\']*)["\']', body))
        cleaned = _XML_ATTR_RE.sub("", body).strip()

        parts = cleaned.split() if cleaned else []
        name_val = attrs.get("name", "")
        if name_val and name_val not in " ".join(parts):
            parts.append(name_val)

        tag_body = " ".join(parts)
        return f"<{tag} {tag_body}>" if tag_body else f"<{tag}>"

    text = re.sub(r"<(lvar|lact)\s+((?:[^>])*?)>", _clean_tag, text)
    return text
