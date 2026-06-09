# -*- coding: utf-8 -*-
"""Twitter quote/reply propagation text normalization and root tracing for metrics."""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

_AT_MENTION_RE = re.compile(r"@\S+")

_NESTED_KEYS = ("retweets", "qoutes", "quotes", "reply", "replies")


@dataclass(frozen=True)
class PropagationRecord:
    root_tweet_id: str
    tweet_id: str
    text: str
    embedding: Optional[List[float]] = None


def strip_at_mentions(text: str) -> str:
    if not text:
        return ""
    cleaned = _AT_MENTION_RE.sub("", text)
    return " ".join(cleaned.split()).strip()


def get_tweet_id(tweet: Dict[str, Any], fallback_key: str = "") -> str:
    return str(tweet.get("tweet_id") or tweet.get("note_id") or fallback_key or "").strip()


def tweet_parent_ids(tweet: Dict[str, Any]) -> Tuple[str, str, str]:
    rt = str(tweet.get("retweetedTweet_id") or tweet.get("retweeted_tweet_id") or "").strip()
    qt = str(tweet.get("quotedTweet_id") or tweet.get("quoted_tweet_id") or "").strip()
    rp = str(tweet.get("replyedTweet_id") or tweet.get("replied_tweet_id") or "").strip()
    return rt, qt, rp


def row_parent_ids(row: Dict[str, str]) -> Tuple[str, str, str]:
    rt = (row.get("retweetedTweet_id") or row.get("retweeted_tweet_id") or "").strip()
    qt = (row.get("quotedTweet_id") or row.get("quoted_tweet_id") or "").strip()
    rp = (row.get("replyedTweet_id") or row.get("replied_tweet_id") or "").strip()
    return rt, qt, rp


def immediate_parent_id(tweet: Dict[str, Any]) -> Optional[str]:
    rt, qt, rp = tweet_parent_ids(tweet)
    for pid in (rt, qt, rp):
        if pid:
            return pid
    return None


def immediate_parent_id_from_row(row: Dict[str, str]) -> Optional[str]:
    rt, qt, rp = row_parent_ids(row)
    for pid in (rt, qt, rp):
        if pid:
            return pid
    return None


def is_root_tweet(tweet: Dict[str, Any]) -> bool:
    rt, qt, rp = tweet_parent_ids(tweet)
    return not rt and not qt and not rp


def is_countable_propagation(tweet: Dict[str, Any]) -> bool:
    """Only quote/reply with non-empty text after @ stripping; pure retweets excluded."""
    _, qt, rp = tweet_parent_ids(tweet)
    return bool(qt or rp)


def format_propagation_text(raw: str) -> str:
    return strip_at_mentions(raw or "")


def flatten_twitter_pool(pool: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    flat: Dict[str, Dict[str, Any]] = {}

    def walk(tweet: Dict[str, Any], fallback_key: str = "") -> None:
        tid = get_tweet_id(tweet, fallback_key)
        if tid:
            flat[tid] = tweet
        for key in _NESTED_KEYS:
            nested = tweet.get(key)
            if isinstance(nested, dict):
                for nk, item in nested.items():
                    if isinstance(item, dict):
                        walk(item, str(nk))

    if not isinstance(pool, dict):
        return flat
    for key, tweet in pool.items():
        if isinstance(tweet, dict):
            walk(tweet, str(key))
    return flat


def trace_root_in_flat_pool(
    tweet_id: str,
    flat_pool: Dict[str, Dict[str, Any]],
    *,
    max_hops: int = 100,
) -> Optional[str]:
    cur = str(tweet_id or "").strip()
    if not cur:
        return None
    seen: Set[str] = set()
    for _ in range(max_hops):
        if cur in seen:
            return cur
        seen.add(cur)
        tweet = flat_pool.get(cur)
        if tweet is None:
            return cur
        if is_root_tweet(tweet):
            return cur
        pid = immediate_parent_id(tweet)
        if not pid or pid == cur:
            return cur
        cur = pid
    return cur


def trace_root_tweet_in_pool(
    tweet_id: str,
    pool: Dict[str, Any],
    *,
    max_hops: int = 100,
) -> Optional[str]:
    flat_pool = flatten_twitter_pool(pool) if pool else {}
    return trace_root_in_flat_pool(tweet_id, flat_pool, max_hops=max_hops)


def load_twitter_csv_rows(csv_path: str) -> Dict[str, Dict[str, str]]:
    rows_by_id: Dict[str, Dict[str, str]] = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            tid = (row.get("tweet_id") or row.get("note_id") or "").strip()
            if tid:
                rows_by_id[tid] = row
    return rows_by_id


def trace_root_in_csv(
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
            if not any(row_parent_ids(row))
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
        pid = immediate_parent_id_from_row(row)
        if not pid:
            return cur
        cur = pid
    return None


def original_root_ids(pool: Dict[str, Any]) -> Set[str]:
    roots: Set[str] = set()
    for tid, tweet in flatten_twitter_pool(pool).items():
        if is_root_tweet(tweet):
            roots.add(tid)
    return roots


def collect_propagation_records(
    content_pool: Dict[str, Any],
    *,
    reuse_existing: bool = True,
) -> List[PropagationRecord]:
    records: List[PropagationRecord] = []
    if not isinstance(content_pool, dict):
        return records

    flat_pool = flatten_twitter_pool(content_pool)
    seen: Set[str] = set()

    def maybe_add(tweet: Dict[str, Any], fallback_key: str = "") -> None:
        if not is_countable_propagation(tweet):
            return
        tid = get_tweet_id(tweet, fallback_key)
        if not tid or tid in seen:
            return
        text = format_propagation_text(tweet.get("content") or "")
        if not text:
            return
        root_id = trace_root_in_flat_pool(tid, flat_pool)
        if not root_id:
            return
        emb = None
        if reuse_existing:
            existing = tweet.get("embedding")
            if isinstance(existing, list) and existing:
                emb = [float(x) for x in existing]
        seen.add(tid)
        records.append(
            PropagationRecord(
                root_tweet_id=str(root_id).strip(),
                tweet_id=tid,
                text=text,
                embedding=emb,
            )
        )

    def walk(tweet: Dict[str, Any], fallback_key: str = "") -> None:
        maybe_add(tweet, fallback_key)
        for key in _NESTED_KEYS:
            nested = tweet.get(key)
            if isinstance(nested, dict):
                for nk, item in nested.items():
                    if isinstance(item, dict):
                        walk(item, str(nk))

    for key, tweet in content_pool.items():
        if isinstance(tweet, dict):
            walk(tweet, str(key))
    return records


def load_reference_propagations_by_root(
    csv_path: str,
    root_ids: Optional[Set[str]] = None,
) -> Dict[str, List[str]]:
    """Reference reposts.csv: quote/reply rows only, grouped by traced root tweet id."""
    rows_by_id = load_twitter_csv_rows(csv_path)
    original_roots = {
        rid for rid, row in rows_by_id.items() if not any(row_parent_ids(row))
    }
    by_root: Dict[str, List[str]] = defaultdict(list)
    for tid, row in rows_by_id.items():
        _, qt, rp = row_parent_ids(row)
        if not qt and not rp:
            continue
        root = trace_root_in_csv(tid, rows_by_id, original_roots)
        if not root:
            continue
        if root_ids is not None and root not in root_ids:
            continue
        text = format_propagation_text(row.get("content") or "")
        if text:
            by_root[root].append(text)
    return dict(by_root)
