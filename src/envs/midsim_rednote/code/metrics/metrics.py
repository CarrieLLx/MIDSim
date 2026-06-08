# -*- coding: utf-8 -*-
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import os
import sys
_metrics_dir = os.path.dirname(os.path.abspath(__file__))
_code_dir = os.path.dirname(_metrics_dir)
if _metrics_dir not in sys.path:
    sys.path.insert(0, _metrics_dir)
if _code_dir not in sys.path:
    sys.path.insert(0, _code_dir)
try:
    from embedding_metrics import (
        calculate_comment_max_reference_similarity,
        calculate_comment_similarity,
    )
except ImportError:
    calculate_comment_max_reference_similarity = None
    calculate_comment_similarity = None

try:
    from text_diversity import calculate_text_diversity
except ImportError:
    calculate_text_diversity = None

try:
    from ..utils import to_sim_time_ms
except ImportError:
    from utils import to_sim_time_ms

from collections import Counter, defaultdict
from loguru import logger
from onesim.monitor.utils import (
    safe_get, safe_number, safe_list, safe_sum, 
    safe_avg, safe_max, safe_min, safe_count, log_metric_error
)


def calculate_comment_generation(data: Dict[str, Any]) -> Any:
    """Calculate the number of comments generated over time"""
    try:
        if not data or not isinstance(data, dict):
            log_metric_error("comment_generation", ValueError("Invalid data input"), {"data": data})
            return 0
        
        content_pool = safe_get(data, "content_pool", {})
        if not isinstance(content_pool, dict):
            log_metric_error("comment_generation", ValueError("content_pool is not a dict"), {"content_pool_type": type(content_pool)})
            return 0
        
        total_comments = 0
        for note_id, note in content_pool.items():
            if not isinstance(note, dict):
                continue
            
            comments = note.get("comments", {})
            if isinstance(comments, dict):
                total_comments += len(comments)
            elif isinstance(comments, list):
                total_comments += len(comments)
        
        return float(total_comments)
    
    except Exception as e:
        log_metric_error("comment_generation", e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return 0


def calculate_comment_top_vs_reply_over_time(data: Dict[str, Any]) -> Any:
    """Calculate the number of comments over time, classified by whether they have a parent comment"""
    zero = {"top_level_comments": 0.0, "reply_comments": 0.0}
    try:
        if not data or not isinstance(data, dict):
            log_metric_error(
                "comment_top_vs_reply_over_time",
                ValueError("Invalid data input"),
                {"data": data},
            )
            return zero

        content_pool = safe_get(data, "content_pool", {})
        if not isinstance(content_pool, dict):
            log_metric_error(
                "comment_top_vs_reply_over_time",
                ValueError("content_pool is not a dict"),
                {"content_pool_type": type(content_pool)},
            )
            return zero

        top_level = 0
        reply_n = 0
        for _nid, note in content_pool.items():
            if not isinstance(note, dict):
                continue
            comments = note.get("comments", {})
            if isinstance(comments, dict):
                iterable = comments.values()
            elif isinstance(comments, list):
                iterable = comments
            else:
                continue
            for c in iterable:
                if not isinstance(c, dict):
                    continue
                parent = c.get("parent_comment_id")
                if parent is None or parent == "":
                    top_level += 1
                else:
                    reply_n += 1

        return {
            "top_level_comments": float(top_level),
            "reply_comments": float(reply_n),
        }
    except Exception as e:
        log_metric_error(
            "comment_top_vs_reply_over_time",
            e,
            {"data_keys": list(data.keys()) if isinstance(data, dict) else None},
        )
        return zero



def _histogram_comment_times_ms(times_ms: List[float]) -> Tuple[List[float], List[int], str]:
    """Calculate the number of comments generated over time"""
    if not times_ms:
        return [], [], ""
    lo, hi = min(times_ms), max(times_ms)
    span = hi - lo
    n = len(times_ms)
    if span <= 0:
        return [lo, lo + 1.0], [n], "single timestamp"
    n_bins = max(8, min(48, max(10, n // 3)))
    width = span / n_bins
    edges = [lo + i * width for i in range(n_bins + 1)]
    counts = [0] * n_bins
    for t in times_ms:
        idx = int((t - lo) / width)
        if idx >= n_bins:
            idx = n_bins - 1
        if idx < 0:
            idx = 0
        counts[idx] += 1
    sec = width / 1000.0
    if sec >= 86400:
        desc = f"~{sec / 86400:.2f} days per bin"
    elif sec >= 3600:
        desc = f"~{sec / 3600:.2f} hours per bin"
    elif sec >= 60:
        desc = f"~{sec / 60:.1f} minutes per bin"
    else:
        desc = f"~{sec:.1f} seconds per bin"
    return edges, counts, desc


def calculate_comment_volume_realtime(data: Dict[str, Any]) -> Any:
    """
    Calculate the number of comments generated over time
    """
    empty: Dict[str, Any] = {
        "_viz_kind": "comment_realtime",
        "timestamps_ms": [],
        "hist_bin_edges_ms": [],
        "hist_counts": [],
        "hist_bucket_description": "",
        "n_comments": 0,
    }
    try:
        if not data or not isinstance(data, dict):
            log_metric_error(
                "comment_volume_realtime",
                ValueError("无效的数据输入"),
                {"data": data},
            )
            return empty

        content_pool = safe_get(data, "content_pool", {})
        if not isinstance(content_pool, dict):
            log_metric_error(
                "comment_volume_realtime",
                ValueError("content_pool is not a dict"),
                {"content_pool_type": type(content_pool)},
            )
            return empty

        times_ms: List[float] = []
        for _nid, note in content_pool.items():
            if not isinstance(note, dict):
                continue
            comments = note.get("comments", {})
            if isinstance(comments, dict):
                iterable = comments.values()
            elif isinstance(comments, list):
                iterable = comments
            else:
                continue
            for c in iterable:
                if not isinstance(c, dict):
                    continue
                ms = to_sim_time_ms(c.get("timestamp"))
                if ms is not None:
                    times_ms.append(ms)

        times_ms.sort()
        edges, counts, desc = _histogram_comment_times_ms(times_ms)
        return {
            "_viz_kind": "comment_realtime",
            "timestamps_ms": times_ms,
            "hist_bin_edges_ms": edges,
            "hist_counts": counts,
            "hist_bucket_description": desc,
            "n_comments": len(times_ms),
        }
    except Exception as e:
        log_metric_error(
            "comment_volume_realtime",
            e,
            {"data_keys": list(data.keys()) if isinstance(data, dict) else None},
        )
        return empty


def _iter_note_comments(
    comments: Any,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Return a list of (comment_id_str, comment_dict)"""
    out: List[Tuple[str, Dict[str, Any]]] = []
    if isinstance(comments, dict):
        for cid, c in comments.items():
            if isinstance(c, dict):
                out.append((str(cid), c))
    elif isinstance(comments, list):
        for i, c in enumerate(comments):
            if not isinstance(c, dict):
                continue
            cid = c.get("comment_id")
            if cid is None or cid == "":
                cid = f"__idx_{i}"
            out.append((str(cid), c))
    return out


def _collect_content_pool_comment_stats(
    content_pool: Dict[str, Any],
) -> Tuple[List[int], Dict[str, int], Set[str]]:
    """Return per-note comment counts, per-user totals, and pool user ids."""
    per_note_ncomments: List[int] = []
    user_comment_totals: Dict[str, int] = defaultdict(int)
    pool_user_ids: Set[str] = set()

    for note_id, note in content_pool.items():
        if not isinstance(note, dict):
            continue
        aid = note.get("user_id")
        if aid is not None and str(aid).strip() != "":
            pool_user_ids.add(str(aid))
        pairs = _iter_note_comments(note.get("comments", {}))
        per_note_ncomments.append(len(pairs))
        for _cid, c in pairs:
            uid = c.get("user_id")
            if uid is None or str(uid).strip() == "":
                continue
            us = str(uid)
            pool_user_ids.add(us)
            user_comment_totals[us] += 1

    return per_note_ncomments, user_comment_totals, pool_user_ids


def _count_frequency_histogram(
    counts: List[int], denominator: int
) -> Tuple[List[int], List[float], List[int]]:
    if denominator <= 0:
        return [], [], []
    h = Counter(counts)
    max_k = max(counts) if counts else 0
    bins = list(range(0, max_k + 1))
    raw = [h[k] for k in bins]
    pct = [100.0 * c / denominator for c in raw]
    return bins, pct, raw


def calculate_user_comment_count_frequency(data: Dict[str, Any]) -> Any:
    """Distribution of total comments per user (denominator: registered users or pool users)."""
    empty: Dict[str, Any] = {
        "_viz_kind": "user_comment_count_freq_bar",
        "comment_bins": [],
        "frequency_pct": [],
        "raw_counts": [],
        "n_users_in_pool": 0,
        "user_count_basis": "content_pool_presence",
    }
    metric_id = "user_comment_count_frequency"
    try:
        if not data or not isinstance(data, dict):
            log_metric_error(metric_id, ValueError("Invalid data input"), {"data": data})
            return empty

        content_pool = safe_get(data, "content_pool", {})
        if not isinstance(content_pool, dict):
            log_metric_error(
                metric_id,
                ValueError("content_pool is not a dict"),
                {"content_pool_type": type(content_pool)},
            )
            return empty

        _per_note, user_comment_totals, pool_user_ids = _collect_content_pool_comment_stats(
            content_pool
        )

        reg = safe_get(data, "registered_user_agent_ids", None)
        if isinstance(reg, list):
            universe: Set[str] = {
                str(x).strip() for x in reg if x is not None and str(x).strip() != ""
            }
        else:
            universe = set()
        use_registered = len(universe) > 0
        if not use_registered:
            universe = pool_user_ids

        n_pool_users = len(universe)
        basis = "registered_user_agent_ids" if use_registered else "content_pool_presence"
        if n_pool_users == 0:
            return {**empty, "user_count_basis": basis}

        per_user_counts = [user_comment_totals.get(u, 0) for u in sorted(universe)]
        bins, pct, raw = _count_frequency_histogram(per_user_counts, n_pool_users)
        return {
            "_viz_kind": "user_comment_count_freq_bar",
            "comment_bins": bins,
            "frequency_pct": pct,
            "raw_counts": raw,
            "n_users_in_pool": n_pool_users,
            "user_count_basis": basis,
        }
    except Exception as e:
        log_metric_error(
            metric_id,
            e,
            {"data_keys": list(data.keys()) if isinstance(data, dict) else None},
        )
        return empty


def calculate_note_comment_count_frequency(data: Dict[str, Any]) -> Any:
    """Distribution of comment counts under each note (denominator: all notes in pool)."""
    empty: Dict[str, Any] = {
        "_viz_kind": "note_comment_count_freq_bar",
        "comment_bins": [],
        "frequency_pct": [],
        "raw_counts": [],
        "n_notes": 0,
    }
    metric_id = "note_comment_count_frequency"
    try:
        if not data or not isinstance(data, dict):
            log_metric_error(metric_id, ValueError("Invalid data input"), {"data": data})
            return empty

        content_pool = safe_get(data, "content_pool", {})
        if not isinstance(content_pool, dict):
            log_metric_error(
                metric_id,
                ValueError("content_pool is not a dict"),
                {"content_pool_type": type(content_pool)},
            )
            return empty

        per_note_ncomments, _user_totals, _pool_users = _collect_content_pool_comment_stats(
            content_pool
        )
        n_notes = len(per_note_ncomments)
        if n_notes == 0:
            return empty

        bins, pct, raw = _count_frequency_histogram(per_note_ncomments, n_notes)
        return {
            "_viz_kind": "note_comment_count_freq_bar",
            "comment_bins": bins,
            "frequency_pct": pct,
            "raw_counts": raw,
            "n_notes": n_notes,
        }
    except Exception as e:
        log_metric_error(
            metric_id,
            e,
            {"data_keys": list(data.keys()) if isinstance(data, dict) else None},
        )
        return empty


def calculate_posting_user_comment_behavior(data: Dict[str, Any]) -> Any:
    """
    Summarize the comment behavior of each user who has posted notes in the current content_pool

    - post count: the number of notes posted by the user as the author;
    - top level comments: comments with parent_comment_id empty in the user's note;
    - reply comments: comments with parent in the user's note;
    - other comments: comments posted on other people's notes;
    - total comments: the sum of the above three.
    """
    empty: Dict[str, Any] = {"users": []}
    try:
        if not data or not isinstance(data, dict):
            log_metric_error(
                "posting_user_comment_behavior",
                ValueError("Invalid data input"),
                {"data": data},
            )
            return empty

        content_pool = safe_get(data, "content_pool", {})
        if not isinstance(content_pool, dict):
            log_metric_error(
                "posting_user_comment_behavior",
                ValueError("content_pool is not a dict"),
                {"content_pool_type": type(content_pool)},
            )
            return empty

        posting_users: Set[str] = set()
        note_authors: List[Tuple[str, Dict[str, Any]]] = []
        for _nid, note in content_pool.items():
            if not isinstance(note, dict):
                continue
            aid = note.get("user_id")
            if aid is None or aid == "":
                continue
            uid = str(aid)
            posting_users.add(uid)
            note_authors.append((uid, note))

        if not posting_users:
            return empty

        notes_count: Dict[str, int] = {u: 0 for u in posting_users}
        for uid, _note in note_authors:
            notes_count[uid] = notes_count.get(uid, 0) + 1

        top_on_own: Dict[str, int] = {u: 0 for u in posting_users}
        reply_on_own: Dict[str, int] = {u: 0 for u in posting_users}
        on_others: Dict[str, int] = {u: 0 for u in posting_users}
        nickname_map: Dict[str, str] = {}

        for author_id, note in note_authors:
            nn = note.get("nickname")
            if isinstance(nn, str) and nn.strip() and author_id in posting_users:
                nickname_map.setdefault(author_id, nn.strip())

            comments = note.get("comments", {})
            if isinstance(comments, dict):
                iterable = comments.values()
            elif isinstance(comments, list):
                iterable = comments
            else:
                continue

            for c in iterable:
                if not isinstance(c, dict):
                    continue
                cid = c.get("user_id")
                if cid is None or cid == "":
                    continue
                commenter = str(cid)
                if commenter not in posting_users:
                    continue
                parent = c.get("parent_comment_id")
                is_top = parent is None or parent == ""

                if commenter == author_id:
                    if is_top:
                        top_on_own[commenter] = top_on_own.get(commenter, 0) + 1
                    else:
                        reply_on_own[commenter] = reply_on_own.get(commenter, 0) + 1
                else:
                    on_others[commenter] = on_others.get(commenter, 0) + 1

                cn = c.get("nickname")
                if isinstance(cn, str) and cn.strip():
                    nickname_map.setdefault(commenter, cn.strip())

        rows: List[Dict[str, Any]] = []
        for uid in sorted(posting_users):
            a = top_on_own.get(uid, 0)
            b = reply_on_own.get(uid, 0)
            c = on_others.get(uid, 0)
            rows.append(
                {
                    "user_id": uid,
                    "nickname": nickname_map.get(uid, ""),
                    "post count": notes_count.get(uid, 0),
                    "top level comments": a,
                    "reply comments": b,
                    "other comments": c,
                    "total comments": a + b + c,
                }
            )

        return {"users": rows}
    except Exception as e:
        log_metric_error(
            "posting_user_comment_behavior",
            e,
            {"data_keys": list(data.keys()) if isinstance(data, dict) else None},
        )
        return empty


# 指标函数字典，用于查找
METRIC_FUNCTIONS = {
    'calculate_comment_generation': calculate_comment_generation,
    'calculate_comment_top_vs_reply_over_time': calculate_comment_top_vs_reply_over_time,
    'calculate_comment_volume_realtime': calculate_comment_volume_realtime,
    'calculate_user_comment_count_frequency': calculate_user_comment_count_frequency,
    'calculate_note_comment_count_frequency': calculate_note_comment_count_frequency,
    'calculate_posting_user_comment_behavior': calculate_posting_user_comment_behavior,
}
if calculate_comment_similarity is not None:
    METRIC_FUNCTIONS['calculate_comment_similarity'] = calculate_comment_similarity
if calculate_comment_diversity is not None:
    METRIC_FUNCTIONS['calculate_comment_diversity'] = calculate_comment_diversity
if calculate_comment_max_reference_similarity is not None:
    METRIC_FUNCTIONS['calculate_comment_max_reference_similarity'] = calculate_comment_max_reference_similarity


def get_metric_function(function_name: str) -> Optional[Callable]:
    """Get the metric calculation function by function name"""
    return METRIC_FUNCTIONS.get(function_name)
