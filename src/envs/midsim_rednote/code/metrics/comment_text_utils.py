# -*- coding: utf-8 -*-
"""Text helpers for comment metrics."""
from __future__ import annotations

import re
from typing import List

_AT_MENTION_RE = re.compile(r"@\S+")


def strip_at_mentions(text: str) -> str:
    """Remove @user tokens and collapse whitespace."""
    if not text:
        return ""
    cleaned = _AT_MENTION_RE.sub("", text)
    return " ".join(cleaned.split()).strip()


def tokenize(text: str) -> List[str]:
    """Chinese word segmentation via jieba when available; else char tokens."""
    text = (text or "").strip()
    if not text:
        return []
    try:
        import jieba

        return [w for w in jieba.lcut(text) if w.strip()]
    except ImportError:
        return [ch for ch in text if not ch.isspace()]
