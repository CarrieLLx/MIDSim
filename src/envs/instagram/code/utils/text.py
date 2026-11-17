# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import Any, Dict, List, Set

_AT_MENTION_RE = re.compile(r"@\S+")

_KEYWORD_STOPWORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "and", "or", "but", "not", "this", "that", "these", "those", "it", "its",
    '\u7684', '\u4e86', '\u5728', '\u662f', '\u548c', '\u4e0e', '\u6216', '\u53ca', '\u6709', '\u4e3a', '\u5bf9', '\u4e2d', '\u4e0a', '\u4e0b',
    '\u4e0d', '\u5417', '\u561b', '\u554a', '\u54e6', '\u55ef', '\u5427', '\u5440', '\u4e48', '\u5462', '\u7740', '\u8fc7', '\u8fd8', '\u5c31',
    '\u4e00\u4e2a', '\u53ef\u4ee5', '\u8fd9\u6837', '\u8fd9\u4e2a', '\u6211\u4eec', '\u4f60\u4eec', '\u4ed6\u4eec', '\u4ec0\u4e48', '\u600e\u4e48', '\u6ca1\u6709',
    '\u81ea\u5df1', '\u5982\u679c', '\u56e0\u4e3a', '\u6240\u4ee5', '\u4f46\u662f', '\u7136\u540e', '\u800c\u4e14', '\u6216\u8005',
}


def _tokenize_without_jieba(text: str) -> List[str]:
    """Latin/digit words plus overlapping CJK bigrams."""
    out: List[str] = []
    for run in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", text):
        if re.fullmatch(r"[a-z0-9_]+", run):
            out.append(run)
        elif len(run) == 1:
            out.append(run)
        else:
            out.extend(run[i : i + 2] for i in range(len(run) - 1))
    return out


def tokenize(text: str) -> List[str]:
    """Chinese word segmentation via jieba when available; else CJK bigrams."""
    text = (text or "").strip()
    if not text:
        return []
    try:
        import jieba

        return [w for w in jieba.lcut(text) if w.strip()]
    except ImportError:
        return _tokenize_without_jieba(text.lower())


def keyword_set(text: str) -> Set[str]:
    """Tokenize text into a filtered keyword set for overlap / Jaccard."""
    if not text or not str(text).strip():
        return set()
    out: Set[str] = set()
    for w in tokenize(str(text).lower()):
        w = w.strip().lower()
        if len(w) < 2 or w in _KEYWORD_STOPWORDS:
            continue
        if not re.search(r"[0-9a-z_\u4e00-\u9fff]", w):
            continue
        out.add(w)
    return out


def format_real_text(text: str) -> str:
    """Remove @user tokens and collapse whitespace."""
    if not text:
        return ""
    cleaned = _AT_MENTION_RE.sub("", text)
    return " ".join(cleaned.split()).strip()


def format_historical_summary(
    text: Any,
    *,
    max_len: int = 100,
    head: int = 50,
    tail: int = 50,
) -> str:
    """Keep full text when short; otherwise first *head* + … + last *tail* chars."""
    if text is None:
        return ""
    s = str(text).strip()
    if len(s) <= max_len:
        return s
    return s[:head] + "…" + s[-tail:]


def format_popularity_distribution(rows: Any) -> Dict[int, int]:
    """Parse profile rows into comment_count -> post_count."""
    out: Dict[int, int] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_cc = row.get("comment_count")
        raw_pc = row.get("post_count")
        if raw_cc is None or raw_pc is None:
            continue
        try:
            cc = int(float(str(raw_cc).strip()))
            pc = int(float(str(raw_pc).strip()))
        except (TypeError, ValueError):
            continue
        if pc <= 0:
            continue
        out[cc] = out.get(cc, 0) + pc
    return out
