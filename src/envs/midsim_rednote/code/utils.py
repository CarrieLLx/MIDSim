# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import random
import re
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

_AT_MENTION_RE = re.compile(r"@\S+")

from loguru import logger


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


def to_float(raw: Any, *, default: float) -> float:
    """Parse profile/config scalar to float; empty or invalid values use *default*."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


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
    """Return True if a time field on *obj* falls in [lo, hi) (same units as bounds)."""
    if lo >= hi or not isinstance(obj, dict):
        return False
    raw = None
    for key in time_keys:
        if key in obj and obj[key] is not None:
            raw = obj[key]
            break
    if raw is None:
        return False
    try:
        t = float(raw)
    except (TypeError, ValueError):
        return False
    return lo <= t < hi


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


def generate_comment_id() -> str:
    """24-char hex comment id, e.g. 693ba19b000000001702f98e."""
    return secrets.token_hex(12)


def generate_comment_timestamp(
    note: Dict[str, Any], window_start_ms: int, window_duration_ms: int
) -> int:
    """Pick a random comment timestamp from post time through the window end."""
    if window_start_ms <= 0 and window_duration_ms <= 0:
        return 0
    lo_win = int(window_start_ms)
    hi_incl = lo_win + int(window_duration_ms) - 1 if window_duration_ms > 0 else lo_win

    post_ms = None
    if isinstance(note, dict):
        t_raw = note.get("time")
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


def resolve_parent_comment_entry(
    comments_map: Any,
    parent_raw: Any,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Map LLM parent_comment_id onto note['comments'] keys (str/int tolerant)."""
    if parent_raw is None:
        return None, None
    if not isinstance(comments_map, dict) or not comments_map:
        return None, None
    s = str(parent_raw).strip()
    if not s:
        return None, None
    ent = comments_map.get(s)
    if isinstance(ent, dict):
        return s, ent
    for k, v in comments_map.items():
        if str(k).strip() == s and isinstance(v, dict):
            return str(k), v
    return None, None