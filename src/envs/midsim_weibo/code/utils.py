# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import random
import re
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

_AT_MENTION_RE = re.compile(r"@\S+")


def time_to_ms(raw: Any, *, default: Optional[float] = None) -> Optional[float]:
    """Normalize timestamp to milliseconds (values < 1e12 are treated as Unix seconds)."""
    if raw is None or isinstance(raw, bool):
        return default
    try:
        x = float(raw)
    except (TypeError, ValueError):
        return default
    if x <= 0:
        return default
    if x < 1e12:
        x *= 1000.0
    return x


def time_to_sec(raw: Any, *, default: Optional[float] = None) -> Optional[float]:
    """Normalize timestamp to Unix seconds (values >= 1e12 are treated as milliseconds)."""
    if raw is None or isinstance(raw, bool):
        return default
    try:
        x = float(raw)
    except (TypeError, ValueError):
        return default
    if x <= 0:
        return default
    if x >= 1e12:
        return x / 1000.0
    return x


def time_in_window(
    obj: Any,
    lo: float,
    hi: float,
    *,
    time_keys: Tuple[str, ...] = ("time", "create_time", "timestamp"),
) -> bool:
    """Return True if a time field on *obj* falls in [lo, hi) (bounds in Unix seconds)."""
    if lo >= hi or not isinstance(obj, dict):
        return False
    raw = None
    for key in time_keys:
        if key in obj and obj[key] is not None:
            raw = obj[key]
            break
    if raw is None:
        return False
    t = time_to_sec(raw)
    if t is None:
        return False
    return lo <= t < hi


def time_to_format_utc(ms: float) -> str:
    if ms is None or ms <= 0:
        return "(Unknown)"
    dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def content_dicts_from_chunk(chunk: Union[Dict[str, Any], List[Any]]) -> List[Dict[str, Any]]:
    """Extract content dicts from a recommendation chunk or mention entry list."""
    items: List[Dict[str, Any]] = []
    if isinstance(chunk, dict):
        for v in chunk.values():
            if isinstance(v, dict):
                items.append(v)
    elif isinstance(chunk, list):
        for entry in chunk:
            if not isinstance(entry, dict):
                continue
            inner = entry.get("mention_blog")
            if not isinstance(inner, dict):
                inner = entry.get("mention_note")
            if isinstance(inner, dict):
                items.append(inner)
    return items


def is_original_blog(blog: Any, *, parent_key: str = "reposted_blog_id") -> bool:
    """True if blog has no reposted_blog_id or it is empty."""
    if not isinstance(blog, dict):
        return False
    rid = blog.get(parent_key)
    if rid is None:
        return True
    return str(rid).strip() == ""


def is_repost_of_other_blog(blog_id: str, blog: Any, *, parent_key: str = "reposted_blog_id") -> bool:
    """True if blog reposts another entry (non-empty reposted_blog_id != blog_id)."""
    if not isinstance(blog, dict):
        return False
    pid = blog.get(parent_key)
    if pid is None:
        return False
    ps = str(pid).strip()
    if not ps:
        return False
    return ps != str(blog_id).strip()


def to_float(raw: Any, *, default: float) -> float:
    """Parse profile/config scalar to float; empty or invalid values use *default*."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def format_popularity_distribution(rows: Any) -> Dict[int, int]:
    """Parse profile rows into repost_count -> post_count."""
    out: Dict[int, int] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_rc = row.get("repost_count")
        raw_pc = row.get("post_count")
        if raw_rc is None or raw_pc is None:
            continue
        try:
            rc = int(float(str(raw_rc).strip()))
            pc = int(float(str(raw_pc).strip()))
        except (TypeError, ValueError):
            continue
        if pc <= 0:
            continue
        out[rc] = out.get(rc, 0) + pc
    return out


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    try:
        import jieba

        return [w for w in jieba.lcut(text) if w.strip()]
    except ImportError:
        return [ch for ch in text if not ch.isspace()]


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


def generate_repost_id() -> str:
    """10-digit decimal string (1_000_000_000~9_999_999_999) as content_pool repost key."""
    return str(secrets.randbelow(9_000_000_000) + 1_000_000_000)


def generate_blog_timestamp(
    blog: Dict[str, Any], window_start_sec: int, window_duration_sec: int
) -> int:
    """Pick a random blog/repost timestamp from post time through the window end (Unix seconds)."""
    if window_start_sec <= 0 and window_duration_sec <= 0:
        return 0
    lo_win = int(window_start_sec)
    hi_incl = lo_win + int(window_duration_sec) - 1 if window_duration_sec > 0 else lo_win

    post_sec = None
    if isinstance(blog, dict):
        t_raw = blog.get("time", blog.get("create_time"))
        if t_raw is not None and not isinstance(t_raw, bool):
            sec = time_to_sec(t_raw)
            if sec is not None:
                post_sec = int(round(sec))

    if post_sec is None:
        return hi_incl

    lo = max(post_sec, lo_win)
    hi = hi_incl
    if lo > hi:
        lo, hi = hi, lo
    if lo < hi:
        return random.randint(lo, hi)
    jitter_max = int(os.environ.get("ONESIM_BLOG_TS_DEGENERATE_JITTER_SEC", "600"))
    if jitter_max <= 0:
        return lo
    return lo + random.randint(0, jitter_max)
