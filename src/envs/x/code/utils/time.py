# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import random
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from loguru import logger


_POST_TIME_KEYS: Tuple[str, ...] = ("time", "create_time", "created_time", "timestamp")
_COMMENT_TIME_KEYS: Tuple[str, ...] = ("timestamp", "create_time", "time")


def _iso_datetime_to_ms(text: str, *, default: Optional[float] = None) -> Optional[float]:
    s = text.strip()
    if not s:
        return default
    if s.endswith(("Z", "z")):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return default
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp() * 1000.0


def time_to_ms(raw: Any, *, default: Optional[float] = None) -> Optional[float]:
    """Normalize timestamp to milliseconds (numeric milliseconds or ISO datetime)."""
    if raw is None or isinstance(raw, bool):
        return default
    if isinstance(raw, datetime):
        dt = raw if raw.tzinfo is not None else raw.replace(tzinfo=timezone.utc)
        return dt.timestamp() * 1000.0
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return default
        try:
            x = float(text)
        except ValueError:
            return _iso_datetime_to_ms(text, default=default)
        raw = x
    try:
        x = float(raw)
    except (TypeError, ValueError):
        return default
    if x <= 0:
        return default
    return x


def _coerce_time_fields_to_ms(obj: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[int]:
    """Rewrite existing time fields on *obj* to int milliseconds. Return the first parsed value."""
    canonical: Optional[int] = None
    for key in keys:
        if key not in obj:
            continue
        raw = obj[key]
        if raw is None or raw == "":
            continue
        ms = time_to_ms(raw)
        if ms is None:
            continue
        ms_i = int(round(ms))
        if canonical is None:
            canonical = ms_i
        if obj[key] != ms_i:
            obj[key] = ms_i
    return canonical


def normalize_post_time_fields(post: Dict[str, Any]) -> bool:
    """If a post has time fields, rewrite them to milliseconds (fallback for mixed units / ISO)."""
    if not isinstance(post, dict):
        return False
    before = {k: post.get(k) for k in _POST_TIME_KEYS}
    canonical = _coerce_time_fields_to_ms(post, _POST_TIME_KEYS)
    if canonical is not None and post.get("time") != canonical:
        post["time"] = canonical
    changed = any(post.get(k) != before[k] for k in _POST_TIME_KEYS)

    comments = post.get("comments")
    if isinstance(comments, dict):
        for comment in comments.values():
            if not isinstance(comment, dict):
                continue
            c_before = {k: comment.get(k) for k in _COMMENT_TIME_KEYS}
            _coerce_time_fields_to_ms(comment, _COMMENT_TIME_KEYS)
            if any(comment.get(k) != c_before[k] for k in _COMMENT_TIME_KEYS):
                changed = True
    return changed


def normalize_content_pool_times(content_pool: Any) -> int:
    """Normalize time fields on every post in content_pool. Return how many posts changed."""
    if not isinstance(content_pool, dict):
        return 0
    n = 0
    for post in content_pool.values():
        if normalize_post_time_fields(post):
            n += 1
    return n


def time_to_format_utc(ms: float) -> str:
    if ms is None or ms <= 0:
        return "(Unknown)"
    dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def time_in_window(
    obj: Any,
    lo: float,
    hi: float,
    *,
    time_keys: Tuple[str, ...] = ("time", "create_time", "timestamp"),
) -> bool:
    """Return True if a time field on *obj* falls in [lo, hi) (bounds in milliseconds)."""
    if lo >= hi or not isinstance(obj, dict):
        return False
    for key in time_keys:
        if key not in obj:
            continue
        raw = obj[key]
        if raw is None or raw == "":
            continue
        t = time_to_ms(raw)
        if t is None:
            continue
        return lo <= t < hi
    return False


def generate_comment_timestamp(
    post: Dict[str, Any], window_start_ms: int, window_duration_ms: int
) -> int:
    """Pick a random comment timestamp from post time through the window end."""
    if window_start_ms <= 0 and window_duration_ms <= 0:
        return 0
    lo_win = int(window_start_ms)
    hi_incl = lo_win + int(window_duration_ms) - 1 if window_duration_ms > 0 else lo_win

    post_ms = None
    if isinstance(post, dict):
        t_raw = post.get("time")
        if t_raw is not None and not isinstance(t_raw, bool):
            ms = time_to_ms(t_raw)
            if ms is not None:
                post_ms = int(round(ms))

    lo = lo_win if post_ms is None else max(int(post_ms), lo_win)
    hi = hi_incl
    logger.info(f"lo: {lo}, hi: {hi}")
    if lo > hi:
        lo, hi = hi, lo
    if lo < hi:
        return random.randint(lo, hi)
    jitter_max = int(os.environ.get("ONESIM_COMMENT_TS_DEGENERATE_JITTER_MS", "600000"))
    if jitter_max <= 0:
        return lo
    return lo + random.randint(0, jitter_max)
