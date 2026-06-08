# -*- coding: utf-8 -*-
"""Stateless helpers for UserAgent: time, text, and ID utilities."""
from __future__ import annotations

import os
import random
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from loguru import logger


def to_sim_time_ms(raw: Any, *, default: Optional[float] = None) -> Optional[float]:
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


def format_sim_ms_utc(ms: float) -> str:
    if ms is None or ms <= 0:
        return "（未知）"
    dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def note_post_time_in_window(note: Dict[str, Any], lo: float, hi: float) -> bool:
    """Return True if the note post time falls in [lo, hi)."""
    if lo >= hi:
        return False
    raw = note.get("time", note.get("create_time"))
    try:
        return float(raw) >= lo and float(raw) < hi
    except (TypeError, ValueError):
        return False


def random_comment_timestamp(
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
            ms = to_sim_time_ms(t_raw)
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


def generate_comment_id() -> str:
    """24-char hex comment id, e.g. 693ba19b000000001702f98e."""
    return secrets.token_hex(12)
