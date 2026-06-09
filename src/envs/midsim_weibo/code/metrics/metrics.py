# -*- coding: utf-8 -*-
import os
import sys
_metrics_dir = os.path.dirname(os.path.abspath(__file__))
if _metrics_dir not in sys.path:
    sys.path.insert(0, _metrics_dir)
try:
    from embedding_metrics import (
        calculate_text_max_reference_similarity,
        calculate_text_similarity,
    )
except ImportError:
    calculate_text_similarity = None
    calculate_text_max_reference_similarity = None

try:
    from text_diversity import calculate_text_diversity
except ImportError:
    calculate_text_diversity = None

from typing import Dict, Any, List, Optional, Union, Callable, Set, Tuple
from collections import Counter, defaultdict
from loguru import logger
from onesim.monitor.utils import (
    safe_get, safe_number, safe_list, safe_sum, 
    safe_avg, safe_max, safe_min, safe_count, log_metric_error
)

try:
    from .repost_text_utils import is_chain_root_blog, resolve_root_blog_id
except ImportError:
    from repost_text_utils import is_chain_root_blog, resolve_root_blog_id

try:
    from ..utils import is_repost_of_other_blog, time_to_ms
except ImportError:
    from utils import is_repost_of_other_blog, time_to_ms


def calculate_diffusion_generation(data: Dict[str, Any]) -> Any:
    """Calculate the number of comments generated over time"""
    try:
        if not data or not isinstance(data, dict):
            log_metric_error("diffusion_generation", ValueError("Invalid data input"), {"data": data})
            return 0
        
        content_pool = safe_get(data, "content_pool", {})
        if not isinstance(content_pool, dict):
            log_metric_error("diffusion_generation", ValueError("content_pool is not a dict"), {"content_pool_type": type(content_pool)})
            return 0
        
        total = 0
        for blog in content_pool.values():
            if not isinstance(blog, dict):
                continue
            rid = blog.get("reposted_blog_id")
            if rid is not None and str(rid).strip():
                total += 1

        return float(total)

    except Exception as e:
        log_metric_error("diffusion_generation", e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return 0


def _histogram_diffusion_times_ms(times_ms: List[float]) -> Tuple[List[float], List[int], str]:
    """Calculate the number of reposts generated over time"""
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


def calculate_diffusion_volume_realtime(data: Dict[str, Any]) -> Any:
    """
    Per repost entry in content_pool: use blog `time` (ms) as event time.
    Sorted timestamps, equal-width histogram → same visualization contract as Comment Volume Real Time.
    """
    empty: Dict[str, Any] = {
        "_viz_kind": "repost_realtime",
        "timestamps_ms": [],
        "hist_bin_edges_ms": [],
        "hist_counts": [],
        "hist_bucket_description": "",
        "n_reposts": 0,
    }
    try:
        if not data or not isinstance(data, dict):
            log_metric_error(
                "diffusion_volume_realtime",
                ValueError("invalid data"),
                {"data": data},
            )
            return empty

        content_pool = safe_get(data, "content_pool", {})
        if not isinstance(content_pool, dict):
            log_metric_error(
                "diffusion_volume_realtime",
                ValueError("content_pool is not a dict"),
                {"content_pool_type": type(content_pool)},
            )
            return empty

        times_ms: List[float] = []
        for bid, blog in content_pool.items():
            if not isinstance(blog, dict):
                continue
            if not is_repost_of_other_blog(str(bid), blog):
                continue
            ms = time_to_ms(blog.get("time"))
            if ms is not None:
                times_ms.append(ms)

        times_ms.sort()
        edges, counts, desc = _histogram_diffusion_times_ms(times_ms)
        return {
            "_viz_kind": "repost_realtime",
            "timestamps_ms": times_ms,
            "hist_bin_edges_ms": edges,
            "hist_counts": counts,
            "hist_bucket_description": desc,
            "n_reposts": len(times_ms),
        }
    except Exception as e:
        log_metric_error(
            "diffusion_volume_realtime",
            e,
            {"data_keys": list(data.keys()) if isinstance(data, dict) else None},
        )
        return empty


def _repost_hop_depth(
    blog_id: str,
    blog: Dict[str, Any],
    _pool: Dict[str, Any],
    memo: Dict[str, int],
) -> int:
    """Calculate the hop depth of the repost chain"""
    bid = str(blog_id).strip()
    if bid in memo:
        return memo[bid]
    if not is_repost_of_other_blog(bid, blog):
        memo[bid] = 0
        return 0

    path = blog.get("reposted_path")
    path_len = 0
    if isinstance(path, list):
        path_len = len([p for p in path if p is not None and str(p).strip()])

    if path_len < 1:
        path_len = 1
    memo[bid] = path_len
    return path_len


def calculate_diffusion_hop_depth_over_time(data: Dict[str, Any]) -> Any:
    """At each monitor sample: count repost entries in content_pool by hop depth (1,2,3,...)."""
    default = {"hop_1": 0.0, "hop_2": 0.0, "hop_3": 0.0}
    try:
        if not data or not isinstance(data, dict):
            log_metric_error(
                "diffusion_hop_depth_over_time",
                ValueError("invalid data"),
                {"data": data},
            )
            return default

        content_pool = safe_get(data, "content_pool", {})
        if not isinstance(content_pool, dict):
            log_metric_error(
                "diffusion_hop_depth_over_time",
                ValueError("content_pool is not a dict"),
                {"content_pool_type": type(content_pool)},
            )
            return default

        memo: Dict[str, int] = {}
        depth_counts: Dict[int, int] = defaultdict(int)
        for bid, blog in content_pool.items():
            if not isinstance(blog, dict):
                continue
            if not is_repost_of_other_blog(str(bid), blog):
                continue
            d = _repost_hop_depth(str(bid), blog, content_pool, memo)
            if d >= 1:
                depth_counts[d] += 1

        if not depth_counts:
            return default

        max_d = max(depth_counts.keys())
        out: Dict[str, float] = {}
        for k in range(1, max_d + 1):
            out[f"hop_{k}"] = float(depth_counts.get(k, 0))
        return out
    except Exception as e:
        log_metric_error(
            "repost_hop_depth_over_time",
            e,
            {"data_keys": list(data.keys()) if isinstance(data, dict) else None},
        )
        return default


def _collect_content_pool_repost_stats(
    content_pool: Dict[str, Any],
) -> Tuple[List[int], Dict[str, int], Set[str]]:
    """Per-root descendant counts, per-user repost totals, pool author user_ids."""
    cache: Dict[str, str] = {}
    root_ids: List[str] = []
    for bid, b in content_pool.items():
        if not isinstance(b, dict):
            continue
        if is_chain_root_blog(str(bid), b, content_pool, cache):
            root_ids.append(str(bid))

    per_root_descendants: List[int] = []
    for r in root_ids:
        n = 0
        for bid, b in content_pool.items():
            if not isinstance(b, dict):
                continue
            if str(bid) == r:
                continue
            root = resolve_root_blog_id(str(bid), b, content_pool, cache)
            if str(root) == str(r):
                n += 1
        per_root_descendants.append(n)

    pool_users: Set[str] = set()
    user_repost_totals: Dict[str, int] = defaultdict(int)
    for bid, b in content_pool.items():
        if not isinstance(b, dict):
            continue
        uid = b.get("user_id")
        if uid is None or str(uid).strip() == "":
            continue
        us = str(uid).strip()
        pool_users.add(us)
        if is_repost_of_other_blog(str(bid), b):
            user_repost_totals[us] += 1

    return per_root_descendants, user_repost_totals, pool_users


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


def calculate_user_repost_count_frequency(data: Dict[str, Any]) -> Any:
    """Distribution of repost entries per user (denominator: registered or pool authors)."""
    empty: Dict[str, Any] = {
        "_viz_kind": "user_repost_count_freq_bar",
        "repost_bins": [],
        "frequency_pct": [],
        "raw_counts": [],
        "n_users_in_pool": 0,
        "user_count_basis": "content_pool_presence",
    }
    metric_id = "user_repost_count_frequency"
    try:
        if not data or not isinstance(data, dict):
            log_metric_error(metric_id, ValueError("invalid data"), {"data": data})
            return empty

        content_pool = safe_get(data, "content_pool", {})
        if not isinstance(content_pool, dict):
            log_metric_error(
                metric_id,
                ValueError("content_pool is not a dict"),
                {"content_pool_type": type(content_pool)},
            )
            return empty

        _per_root, user_repost_totals, pool_users = _collect_content_pool_repost_stats(
            content_pool
        )

        registered = safe_get(data, "user_agent_profile_ids", None)
        if isinstance(registered, list):
            universe: Set[str] = {
                str(x).strip() for x in registered if x is not None and str(x).strip() != ""
            }
        else:
            universe = set()
        use_registered = len(universe) > 0
        if not use_registered:
            universe = pool_users

        n_pool_users = len(universe)
        basis = "user_agent_profile_ids" if use_registered else "content_pool_presence"
        if n_pool_users == 0:
            return {**empty, "user_count_basis": basis}

        per_user_counts = [user_repost_totals.get(u, 0) for u in sorted(universe)]
        bins, pct, raw = _count_frequency_histogram(per_user_counts, n_pool_users)
        return {
            "_viz_kind": "user_repost_count_freq_bar",
            "repost_bins": bins,
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


def calculate_root_repost_count_frequency(data: Dict[str, Any]) -> Any:
    """Distribution of repost nodes under each root blog (denominator: all chain roots in pool)."""
    empty: Dict[str, Any] = {
        "_viz_kind": "root_repost_count_freq_bar",
        "repost_bins": [],
        "frequency_pct": [],
        "raw_counts": [],
        "n_root_tweets": 0,
    }
    metric_id = "root_repost_count_frequency"
    try:
        if not data or not isinstance(data, dict):
            log_metric_error(metric_id, ValueError("invalid data"), {"data": data})
            return empty

        content_pool = safe_get(data, "content_pool", {})
        if not isinstance(content_pool, dict):
            log_metric_error(
                metric_id,
                ValueError("content_pool is not a dict"),
                {"content_pool_type": type(content_pool)},
            )
            return empty

        per_root_descendants, _user_totals, _pool_users = _collect_content_pool_repost_stats(
            content_pool
        )
        n_roots = len(per_root_descendants)
        if n_roots == 0:
            return empty

        bins, pct, raw = _count_frequency_histogram(per_root_descendants, n_roots)
        return {
            "_viz_kind": "root_repost_count_freq_bar",
            "repost_bins": bins,
            "frequency_pct": pct,
            "raw_counts": raw,
            "n_root_tweets": n_roots,
        }
    except Exception as e:
        log_metric_error(
            metric_id,
            e,
            {"data_keys": list(data.keys()) if isinstance(data, dict) else None},
        )
        return empty


def calculate_posting_root_author_repost_behavior(data: Dict[str, Any]) -> Any:
    """
    Per root-tweet author:
    - self_repost_hop_k: The k-th self-repost hop on the root author's root post chain.
    - repost_on_others_count: The number of reposts on **other root posts** by the user.
    """
    empty: Dict[str, Any] = {"users": []}
    try:
        if not data or not isinstance(data, dict):
            log_metric_error(
                "posting_root_author_repost_behavior",
                ValueError("invalid data"),
                {"data": data},
            )
            return empty

        content_pool = safe_get(data, "content_pool", {})
        if not isinstance(content_pool, dict):
            log_metric_error(
                "posting_root_author_repost_behavior",
                ValueError("content_pool is not a dict"),
                {"content_pool_type": type(content_pool)},
            )
            return empty

        cache: Dict[str, str] = {}
        hop_memo: Dict[str, int] = {}
        root_authors: Set[str] = set()
        root_post_count: Dict[str, int] = defaultdict(int)
        nickname_map: Dict[str, str] = {}

        for bid, b in content_pool.items():
            if not isinstance(b, dict):
                continue
            uid = b.get("user_id")
            if uid is None or str(uid).strip() == "":
                continue
            us = str(uid).strip()
            if is_chain_root_blog(str(bid), b, content_pool, cache):
                root_authors.add(us)
                root_post_count[us] += 1
            nn = b.get("nickname")
            if isinstance(nn, str) and nn.strip():
                nickname_map.setdefault(us, nn.strip())

        if not root_authors:
            return empty

        self_hop: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        for bid, blog in content_pool.items():
            if not isinstance(blog, dict):
                continue
            if not is_repost_of_other_blog(str(bid), blog):
                continue
            author = blog.get("user_id")
            if author is None or str(author).strip() == "":
                continue
            ua = str(author).strip()
            root_id = resolve_root_blog_id(str(bid), blog, content_pool, cache)
            rb = content_pool.get(root_id)
            if not isinstance(rb, dict):
                continue
            ra = rb.get("user_id")
            if ra is None or str(ra).strip() == "":
                continue
            if str(ra).strip() != ua:
                continue
            d = _repost_hop_depth(str(bid), blog, content_pool, hop_memo)
            if d >= 1:
                self_hop[ua][d] += 1

        other_root_repost: Dict[str, int] = defaultdict(int)
        for bid, blog in content_pool.items():
            if not isinstance(blog, dict):
                continue
            if not is_repost_of_other_blog(str(bid), blog):
                continue
            author = blog.get("user_id")
            if author is None or str(author).strip() == "":
                continue
            ua = str(author).strip()
            if ua not in root_authors:
                continue
            root_id = resolve_root_blog_id(str(bid), blog, content_pool, cache)
            rb = content_pool.get(root_id)
            if not isinstance(rb, dict):
                continue
            ra = rb.get("user_id")
            if ra is None or str(ra).strip() == "":
                continue
            if str(ra).strip() == ua:
                continue
            other_root_repost[ua] += 1

        max_h = 1
        for dd in self_hop.values():
            if dd:
                max_h = max(max_h, max(dd.keys()))

        rows: List[Dict[str, Any]] = []
        for uid in sorted(root_authors):
            row: Dict[str, Any] = {
                "user_id": uid,
                "nickname": nickname_map.get(uid, ""),
                "root_post_count": int(root_post_count.get(uid, 0)),
                "repost_on_others_count": int(other_root_repost.get(uid, 0)),
            }
            hops_u = self_hop.get(uid, {})
            for k in range(1, max_h + 1):
                row[f"self_repost_hop_{k}"] = int(hops_u.get(k, 0))
            rows.append(row)

        return {"users": rows}
    except Exception as e:
        log_metric_error(
            "posting_root_author_repost_behavior",
            e,
            {"data_keys": list(data.keys()) if isinstance(data, dict) else None},
        )
        return empty

METRIC_FUNCTIONS = {
    'calculate_diffusion_generation': calculate_diffusion_generation,
    'calculate_diffusion_volume_realtime': calculate_diffusion_volume_realtime,
    'calculate_diffusion_hop_depth_over_time': calculate_diffusion_hop_depth_over_time,
    'calculate_user_repost_count_frequency': calculate_user_repost_count_frequency,
    'calculate_root_repost_count_frequency': calculate_root_repost_count_frequency,
    'calculate_posting_root_author_repost_behavior': calculate_posting_root_author_repost_behavior,
}
if calculate_text_similarity is not None:
    METRIC_FUNCTIONS['calculate_text_similarity'] = calculate_text_similarity
if calculate_text_diversity is not None:
    METRIC_FUNCTIONS['calculate_text_diversity'] = calculate_text_diversity
if calculate_text_max_reference_similarity is not None:
    METRIC_FUNCTIONS['calculate_text_max_reference_similarity'] = calculate_text_max_reference_similarity


def get_metric_function(function_name: str) -> Optional[Callable]:
    """Get the metric calculation function by function name."""
    return METRIC_FUNCTIONS.get(function_name)
