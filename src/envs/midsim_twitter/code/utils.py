# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import random
import secrets
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from loguru import logger

TWEET_LLM_DROP_KEYS = frozenset({"quote_ids", "reply_ids", "retweet_ids", "time"})

PrepareTweetMode = Literal["discussion", "full"]


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


def generate_propagation_id() -> str:
    """Generate a new content_pool tweet key: milliseconds timestamp + 6-digit random decimal suffix."""
    ms = int(time.time() * 1000)
    suffix = secrets.randbelow(1_000_000)
    return f"{ms}{suffix:06d}"


def generate_tweet_timestamp(
    tweet: Dict[str, Any], window_start_sec: int, window_duration_sec: int
) -> int:
    """Pick a random tweet timestamp from post time through the window end (Unix seconds)."""
    if window_start_sec <= 0 and window_duration_sec <= 0:
        return 0
    lo_win = int(window_start_sec)
    if window_duration_sec > 0:
        hi_incl = lo_win + int(window_duration_sec) - 1
    else:
        hi_incl = lo_win

    post_sec = None
    if isinstance(tweet, dict):
        t_raw = tweet.get("time", tweet.get("create_time"))
        if t_raw is not None and not isinstance(t_raw, bool):
            sec = time_to_sec(t_raw)
            if sec is not None:
                post_sec = int(sec)

    if post_sec is None:
        return hi_incl

    lo = max(post_sec, lo_win)
    hi = hi_incl
    if lo > hi:
        lo, hi = hi, lo
    lo_i, hi_i = int(lo), int(hi)
    if hi_i < lo_i:
        return lo_i
    return random.randint(lo_i, hi_i)


def is_original_tweet(
    tweet: Any,
    *,
    retweet_key: str = "retweeted_tweet_id",
    reply_key: str = "replied_tweet_id",
) -> bool:
    """True if tweet has no retweeted_tweet_id or replied_tweet_id (empty or missing)."""
    if not isinstance(tweet, dict):
        return False
    rid = tweet.get(retweet_key) or tweet.get(reply_key)
    if rid is None:
        return True
    return str(rid).strip() == ""


def format_popularity_distribution(rows: Any) -> Dict[int, int]:
    """Parse profile rows into propagation_count -> post_count."""
    out: Dict[int, int] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_pc = row.get("propagation_count")
        raw_cnt = row.get("post_count")
        if raw_pc is None or raw_cnt is None:
            continue
        try:
            pc = int(float(str(raw_pc).strip()))
            cnt = int(float(str(raw_cnt).strip()))
        except (TypeError, ValueError):
            continue
        if cnt <= 0:
            continue
        out[pc] = out.get(pc, 0) + cnt
    return out


def to_float(raw: Any, *, default: float) -> float:
    """Parse profile/config scalar to float; empty or invalid values use *default*."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


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


def sample_interest_tags(tags: Any, *, limit: int = 3) -> Any:
    """Keep up to *limit* non-empty interest tags via random sampling without replacement."""
    if not isinstance(tags, (list, tuple)):
        return tags
    lst = [str(t).strip() for t in tags if str(t).strip()]
    if len(lst) <= limit:
        return lst
    return random.sample(lst, limit)


def sample_mentionable_users(
    mentionable_users: Dict[str, Any],
    *,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    For LLM prompt "Users you may @": mutual follows plus official accounts from follows,
    deduped by user_id; randomly sample down to *limit* when exceeded.
    """
    def _uid(info: Dict[str, Any]) -> str:
        return str(info.get("user_id") or info.get("id") or "").strip()

    seen: Set[str] = set()
    out: List[Dict[str, Any]] = []

    for info in mentionable_users.get("mutual") or []:
        if not isinstance(info, dict):
            continue
        u = _uid(info)
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(info)

    for info in mentionable_users.get("follows") or []:
        if not isinstance(info, dict):
            continue
        if not bool(info.get("is_official", False)):
            continue
        u = _uid(info)
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(info)

    if limit > 0 and len(out) > limit:
        out = random.sample(out, limit)
    return out


def format_content(text: Any, head: int, tail: int) -> str:
    """Keep first *head* + … + last *tail* chars when content exceeds head+tail."""
    s = text if isinstance(text, str) else ("" if text is None else str(text))
    if head <= 0 and tail <= 0:
        return s
    if len(s) <= head + tail:
        return s
    return s[:head] + "…" + s[-tail:]


def tweet_ref_key(ref: Any) -> Optional[str]:
    if ref is None:
        return None
    if isinstance(ref, str):
        s = ref.strip()
        return s if s else None
    try:
        return str(ref)
    except (TypeError, ValueError):
        return None


def resolve_retweet_id_to_root_in_pool(
    first_retweet_parent_id: Any,
    pool: Dict[str, Any],
    max_hops: int = 64,
) -> Optional[str]:
    """
    从「被转推的 tweet_id」出发，仅在 retweet 边上沿 pool 向上，
    直到某条无 retweeted_tweet_id（视为原创）或池里缺键为止。
    """
    cur = tweet_ref_key(first_retweet_parent_id)
    if not cur or not isinstance(pool, dict):
        return None
    for _ in range(max_hops):
        tw = pool.get(cur)
        if not isinstance(tw, dict):
            return cur
        nxt = tweet_ref_key(tw.get("retweeted_tweet_id") or "")
        if not nxt:
            return cur
        cur = nxt
    return cur


def enrich_tweet_quote_reply_chain(
    tweet: Dict[str, Any],
    current_tweets: Dict[str, Any],
    *,
    tweet_ref: Optional[str] = None,
    max_depth: int = 48,
    _depth: int = 0,
    _visited: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Recursively attach quoted_tweet / replied_tweet from current_tweets."""
    if _visited is None:
        _visited = set()
    key = tweet_ref or tweet_ref_key(tweet.get("id")) or tweet_ref_key(tweet.get("tweet_id"))
    if key is not None:
        if key in _visited:
            return dict(tweet)
        _visited.add(key)
    if _depth >= max_depth:
        return dict(tweet)

    out = dict(tweet)
    qid = tweet_ref_key(out.get("quoted_tweet_id"))
    rid = tweet_ref_key(out.get("replied_tweet_id"))

    if qid and qid in current_tweets:
        nested = current_tweets[qid]
        if isinstance(nested, dict):
            out["quoted_tweet"] = enrich_tweet_quote_reply_chain(
                nested,
                current_tweets,
                tweet_ref=qid,
                max_depth=max_depth,
                _depth=_depth + 1,
                _visited=_visited,
            )
    if rid and rid in current_tweets:
        nested = current_tweets[rid]
        if isinstance(nested, dict):
            out["replied_tweet"] = enrich_tweet_quote_reply_chain(
                nested,
                current_tweets,
                tweet_ref=rid,
                max_depth=max_depth,
                _depth=_depth + 1,
                _visited=_visited,
            )
    return out


def _llm_content_head_tail(
    head: Optional[int] = None,
    tail: Optional[int] = None,
) -> tuple[int, int]:
    if head is None:
        head = max(0, int(os.environ.get("ONESIM_LLM_TWEET_CONTENT_HEAD_CHARS", "50")))
    if tail is None:
        tail = max(0, int(os.environ.get("ONESIM_LLM_TWEET_CONTENT_TAIL_CHARS", "50")))
    return head, tail


def strip_tweet_for_llm_observation(
    obj: Any,
    *,
    drop_keys: frozenset[str] = TWEET_LLM_DROP_KEYS,
) -> Any:
    """Drop noisy tweet fields before LLM observation (recursive)."""
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if k in drop_keys:
                continue
            if k == "replies" and isinstance(v, dict):
                out[k] = {
                    rk: strip_tweet_for_llm_observation(rv, drop_keys=drop_keys)
                    for rk, rv in v.items()
                }
            elif isinstance(v, dict):
                out[k] = strip_tweet_for_llm_observation(v, drop_keys=drop_keys)
            elif isinstance(v, list):
                out[k] = [strip_tweet_for_llm_observation(x, drop_keys=drop_keys) for x in v]
            else:
                out[k] = v
        return out
    if isinstance(obj, list):
        return [strip_tweet_for_llm_observation(x, drop_keys=drop_keys) for x in obj]
    return obj


def shrink_tweet_content_head_tail(
    obj: Any,
    *,
    head: int,
    tail: int,
) -> Any:
    """Recursively clip every ``content`` field via format_content."""
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if k == "content":
                out[k] = format_content(v, head, tail)
            elif isinstance(v, dict):
                out[k] = shrink_tweet_content_head_tail(v, head=head, tail=tail)
            elif isinstance(v, list):
                out[k] = [
                    shrink_tweet_content_head_tail(x, head=head, tail=tail)
                    for x in v
                ]
            else:
                out[k] = v
        return out
    if isinstance(obj, list):
        return [
            shrink_tweet_content_head_tail(x, head=head, tail=tail)
            for x in obj
        ]
    return obj


def tweet_discussion_tree_content_only(
    tweet: Any,
    *,
    head: Optional[int] = None,
    tail: Optional[int] = None,
) -> Any:
    """Slim nested quote/reply/replies tree to ids, author fields, and clipped content."""
    if not isinstance(tweet, dict):
        return tweet
    head, tail = _llm_content_head_tail(head, tail)
    out: Dict[str, Any] = {}
    tid = tweet_ref_key(tweet.get("tweet_id") or tweet.get("id"))
    if tid:
        out["tweet_id"] = tid

    out["user_id"] = tweet.get("user_id", "")
    out["username"] = tweet.get("username", "")
    out["nickname"] = tweet.get("nickname", "")
    out["content"] = format_content(tweet.get("content", ""), head, tail)
    for k in ("quoted_tweet", "replied_tweet"):
        ch = tweet.get(k)
        if isinstance(ch, dict):
            out[k] = tweet_discussion_tree_content_only(ch, head=head, tail=tail)
    reps = tweet.get("replies")
    if isinstance(reps, dict):
        out["replies"] = {
            rk: tweet_discussion_tree_content_only(rv, head=head, tail=tail)
            if isinstance(rv, dict)
            else rv
            for rk, rv in reps.items()
        }
    return out


def apply_tweet_llm_json_budget(
    obj: Any,
    *,
    max_json_chars: Optional[int] = None,
    head: Optional[int] = None,
    tail: Optional[int] = None,
) -> Any:
    """Strip noisy fields; shrink content when serialized JSON exceeds the char budget."""
    stripped = strip_tweet_for_llm_observation(obj)
    if max_json_chars is None:
        max_json_chars = int(os.environ.get("ONESIM_LLM_TWEET_JSON_MAX_CHARS", "40000"))
    head, tail = _llm_content_head_tail(head, tail)
    try:
        serialized = json.dumps(stripped, ensure_ascii=False)
    except (TypeError, ValueError):
        return stripped
    if len(serialized) <= max_json_chars:
        return stripped
    shrunk = shrink_tweet_content_head_tail(stripped, head=head, tail=tail)
    try:
        again = json.dumps(shrunk, ensure_ascii=False)
        if len(again) > max_json_chars:
            logger.warning(
                f"LLM tweet JSON still ~{len(again)} chars (> {max_json_chars}) after content clip; "
                f"consider lowering ONESIM_LLM_TWEET_JSON_MAX_CHARS or shrinking observation"
            )
    except (TypeError, ValueError):
        pass
    return shrunk


def prepare_tweet_for_llm(
    obj: Any,
    *,
    mode: PrepareTweetMode = "discussion",
) -> Any:
    """
    Unified LLM tweet JSON preparation.

    - ``discussion`` (default): slim quote/reply tree, then strip + JSON budget.
    - ``full``: strip + JSON budget only (keep all non-dropped fields).
    """
    if mode == "discussion":
        payload = tweet_discussion_tree_content_only(obj)
    elif mode == "full":
        payload = obj
    else:
        raise ValueError(f"Unknown prepare_tweet_for_llm mode: {mode!r}")
    return apply_tweet_llm_json_budget(payload)


def _quote_reply_nested_node_count(tweet: Dict[str, Any]) -> int:
    """Count nested quoted_tweet / replied_tweet nodes (excluding root)."""
    if not isinstance(tweet, dict):
        return 0
    n = 0
    for k in ("quoted_tweet", "replied_tweet"):
        ch = tweet.get(k)
        if isinstance(ch, dict):
            n += 1 + _quote_reply_nested_node_count(ch)
    return n


def llm_input_item_units(item: Dict[str, Any]) -> int:
    """
    Weight units for one recommendation/mention item when batching LLM input.
    Based on reply count and quote/reply nesting depth.
    """
    if not isinstance(item, dict):
        return 1
    reps = item.get("replies")
    n_rep = len(reps) if isinstance(reps, dict) else 0
    nested = _quote_reply_nested_node_count(item)
    u = (n_rep + 1) // 2 + (nested + 1) // 2
    return max(1, u)


def pack_llm_input_chunks(
    items: Dict[str, Dict[str, Any]],
    max_units: int,
) -> List[Dict[str, Dict[str, Any]]]:
    """
    Pack id->content dicts into LLM input batches; sum of item units per batch <= max_units.
    A single item heavier than max_units becomes its own batch.
    """
    rec_items = list(items.items())
    if not rec_items:
        return []
    max_units = max(1, int(max_units))
    chunks: List[Dict[str, Dict[str, Any]]] = []
    cur: Dict[str, Dict[str, Any]] = {}
    cur_u = 0
    for item_id, payload in rec_items:
        need = llm_input_item_units(payload)
        if need > max_units:
            if cur:
                chunks.append(cur)
                cur = {}
                cur_u = 0
            chunks.append({item_id: payload})
            continue
        if cur and cur_u + need > max_units:
            chunks.append(cur)
            cur = {}
            cur_u = 0
        cur[item_id] = payload
        cur_u += need
    if cur:
        chunks.append(cur)
    return chunks


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
