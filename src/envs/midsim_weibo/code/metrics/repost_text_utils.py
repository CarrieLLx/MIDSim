# -*- coding: utf-8 -*-
"""Weibo repost text normalization and root-blog tracing for metrics."""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

_AT_MENTION_RE = re.compile(r"@\S+")

BARE_REPOST_FIRST_SEGMENTS = frozenset(
    {
        "转发微博",
        "转发",
        "repost",
        "轉發微博",
        "轉發",
    }
)


def _normalize_repost_segment(text: str) -> str:
    s = (text or "").strip()
    return s.casefold() if s.isascii() else s


_BARE_REPOST_NORMALIZED = frozenset(_normalize_repost_segment(x) for x in BARE_REPOST_FIRST_SEGMENTS)


def strip_at_mentions(text: str) -> str:
    if not text:
        return ""
    cleaned = _AT_MENTION_RE.sub("", text)
    return " ".join(cleaned.split()).strip()


def first_content_segment(content: str) -> str:
    """User-authored segment before the first ``//`` in a Weibo repost chain."""
    return (content or "").split("//", 1)[0].strip()


def is_bare_repost_first_segment(segment: str) -> bool:
    seg = (segment or "").strip()
    if not seg:
        return False
    return _normalize_repost_segment(seg) in _BARE_REPOST_NORMALIZED


def is_countable_repost_content(content: str) -> bool:
    first = first_content_segment(strip_at_mentions(content))
    if not first:
        return False
    return not is_bare_repost_first_segment(first)


def format_repost_text(raw: str) -> str:
    """Strip @ mentions, take ``//`` first segment, drop bare repost markers."""
    cleaned = strip_at_mentions(raw or "")
    first = first_content_segment(cleaned)
    if not first or is_bare_repost_first_segment(first):
        return ""
    return first


def get_blog_id(blog: Dict[str, Any], fallback_key: str = "") -> str:
    return str(blog.get("blog_id") or blog.get("note_id") or fallback_key or "").strip()


def original_root_ids(pool: Dict[str, Any]) -> Set[str]:
    roots: Set[str] = set()
    for bid, blog in (pool or {}).items():
        if not isinstance(blog, dict):
            continue
        if str(blog.get("reposted_blog_id") or "").strip():
            continue
        rid = get_blog_id(blog, str(bid))
        if rid:
            roots.add(rid)
    return roots


def resolve_root_blog_id(
    blog_id: str,
    blog: Dict[str, Any],
    pool: Dict[str, Any],
    cache: Optional[Dict[str, str]] = None,
) -> str:
    """Resolve chain root (multi-hop via reposted_path or reposted_blog_id)."""
    cache = cache if cache is not None else {}
    bid = str(blog_id).strip()
    if bid in cache:
        return cache[bid]

    path = blog.get("reposted_path")
    if isinstance(path, list) and path:
        r0 = path[0]
        if r0 is not None and str(r0).strip():
            cache[bid] = str(r0).strip()
            return cache[bid]

    pid = blog.get("reposted_blog_id")
    ps = str(pid).strip() if pid is not None else ""
    if not ps or ps == bid:
        cache[bid] = bid
        return cache[bid]
    if ps not in pool:
        cache[bid] = bid
        return cache[bid]
    parent = pool[ps]
    if not isinstance(parent, dict):
        cache[bid] = bid
        return cache[bid]
    root = resolve_root_blog_id(ps, parent, pool, cache)
    cache[bid] = root
    return root


def is_chain_root_blog(
    blog_id: str,
    blog: Dict[str, Any],
    pool: Dict[str, Any],
    cache: Optional[Dict[str, str]] = None,
) -> bool:
    """True if this entry is the root post of its repost chain."""
    bid = str(blog_id).strip()
    return bid == resolve_root_blog_id(bid, blog, pool, cache)


def trace_root_in_reposts_csv(
    start_id: str,
    rows_by_id: Dict[str, Dict[str, str]],
    original_roots: Optional[Set[str]] = None,
    *,
    max_hops: int = 100,
) -> Optional[str]:
    cur = (start_id or "").strip()
    if not cur:
        return None
    if original_roots is None:
        original_roots = {
            rid
            for rid, row in rows_by_id.items()
            if not (row.get("retweeted_note_id") or "").strip()
        }
    seen: Set[str] = set()
    for _ in range(max_hops):
        if cur in seen:
            return None
        seen.add(cur)
        if cur in original_roots:
            return cur
        row = rows_by_id.get(cur)
        if row is None:
            return cur
        parent = (row.get("retweeted_note_id") or "").strip()
        if not parent:
            return cur
        cur = parent
    return None


def load_reposts_csv_rows(csv_path: str) -> Dict[str, Dict[str, str]]:
    rows_by_id: Dict[str, Dict[str, str]] = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rid = (row.get("repost_id") or row.get("blog_id") or "").strip()
            if rid:
                rows_by_id[rid] = row
    return rows_by_id


def load_reference_reposts_by_root(
    csv_path: str,
    root_ids: Optional[Set[str]] = None,
) -> Dict[str, List[str]]:
    """Reference reposts.csv grouped by traced original root blog id."""
    rows_by_id = load_reposts_csv_rows(csv_path)
    original_roots = {
        rid
        for rid, row in rows_by_id.items()
        if not (row.get("retweeted_note_id") or "").strip()
    }
    by_root: Dict[str, List[str]] = defaultdict(list)
    for row in rows_by_id.values():
        parent_id = (row.get("retweeted_note_id") or "").strip()
        if not parent_id:
            continue
        if not is_countable_repost_content(row.get("content") or ""):
            continue
        root = trace_root_in_reposts_csv(parent_id, rows_by_id, original_roots)
        if not root:
            continue
        if root_ids is not None and root not in root_ids:
            continue
        text = format_repost_text(row.get("content") or "")
        if text:
            by_root[root].append(text)
    return dict(by_root)
