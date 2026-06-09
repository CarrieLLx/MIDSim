# -*- coding: utf-8 -*-
import os
import sys
_metrics_dir = os.path.dirname(os.path.abspath(__file__))
if _metrics_dir not in sys.path:
    sys.path.insert(0, _metrics_dir)
try:
    from twitter_similarity import (
        calculate_text_similarity,
        count_direct_content_pool_tweets_for_monitor,
        count_direct_propagation_by_type,
        count_content_pool_tweets_for_monitor,
    )
except ImportError:
    calculate_text_similarity = None
    count_direct_content_pool_tweets_for_monitor = None
    count_direct_propagation_by_type = None
    count_content_pool_tweets_for_monitor = None

try:
    from embedding_metrics import calculate_text_max_reference_similarity
except ImportError:
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
    from ..utils import time_to_ms
except ImportError:
    from utils import time_to_ms


def calculate_diffusion_generation(data: Dict[str, Any]) -> Any:
    """Calculate the number of tweets generated over time"""
    try:
        if not data or not isinstance(data, dict):
            log_metric_error("tweet_pool_count", ValueError("Invalid data input"), {"data": data})
            return 0

        content_pool = safe_get(data, "content_pool", {})
        if not isinstance(content_pool, dict):
            log_metric_error(
                "tweet_pool_count",
                ValueError("content_pool is not a dict"),
                {"content_pool_type": type(content_pool)},
            )
            return 0

        raw_ts = safe_get(data, "current_timestamp", None)
        current_ts: Optional[float] = None
        if isinstance(raw_ts, (int, float)) and raw_ts > 0:
            current_ts = float(raw_ts)

        if count_content_pool_tweets_for_monitor is not None:
            return float(
                count_content_pool_tweets_for_monitor(content_pool, current_ts)
            )

        return float(
            sum(1 for v in content_pool.values() if isinstance(v, dict))
        )

    except Exception as e:
        log_metric_error(
            "tweet_pool_count",
            e,
            {"data_keys": list(data.keys()) if isinstance(data, dict) else None},
        )
        return 0

def calculate_direct_tweet_generation(data: Dict[str, Any]) -> Any:
    """Calculate the number of direct tweets generated over time"""
    try:
        if not data or not isinstance(data, dict):
            log_metric_error("tweet_pool_count", ValueError("Invalid data input"), {"data": data})
            return 0

        content_pool = safe_get(data, "content_pool", {})
        if not isinstance(content_pool, dict):
            log_metric_error(
                "tweet_pool_count",
                ValueError("content_pool is not a dict"),
                {"content_pool_type": type(content_pool)},
            )
            return 0

        raw_ts = safe_get(data, "current_timestamp", None)
        current_ts: Optional[float] = None
        if isinstance(raw_ts, (int, float)) and raw_ts > 0:
            current_ts = float(raw_ts)

        if count_direct_content_pool_tweets_for_monitor is not None:
            return float(
                count_direct_content_pool_tweets_for_monitor(content_pool, current_ts)
            )

        return 0.0

    except Exception as e:
        log_metric_error(
            "tweet_pool_count",
            e,
            {"data_keys": list(data.keys()) if isinstance(data, dict) else None},
        )
        return 0


def _tweet_ref_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _histogram_repost_times_ms(times_ms: List[float]) -> Tuple[List[float], List[int], str]:
    """Equal-width bins in real-time span."""
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


def _is_original_tweet_obj(tw: Dict[str, Any]) -> bool:
    rid = _tweet_ref_str(tw.get("retweeted_tweet_id"))
    qid = _tweet_ref_str(tw.get("quoted_tweet_id"))
    repid = _tweet_ref_str(tw.get("replied_tweet_id")) or _tweet_ref_str(tw.get("replyed_tweet_id"))
    return not rid and not qid and not repid


def _immediate_parent_tweet_id(tw: Dict[str, Any]) -> Optional[str]:
    """Immediate parent tweet id: prioritize retweet → quote → reply."""
    for key in ("retweeted_tweet_id", "quoted_tweet_id", "replied_tweet_id", "replyed_tweet_id"):
        s = _tweet_ref_str(tw.get(key))
        if s:
            return s
    return None


def _is_user_propagation_entry(tweet_id: str, tw: Dict[str, Any]) -> bool:
    """Non-original propagation tweet."""
    pid = _immediate_parent_tweet_id(tw)
    if not pid:
        return False
    return pid != str(tweet_id).strip()


def _resolve_to_env_seed_root(
    tweet_id: str,
    pool: Dict[str, Any],
    seed_ids: Set[str],
    max_hops: int = 512,
) -> Optional[str]:
    """Trace up the retweet / quote / reply parent chain until reaching an env seed id (seed_root_tweet_ids)."""
    cur = str(tweet_id).strip()
    if not cur:
        return None
    for _ in range(max_hops):
        if cur in seed_ids:
            return cur
        tw = pool.get(cur)
        if not isinstance(tw, dict):
            return None
        pid = _immediate_parent_tweet_id(tw)
        if not pid or pid == cur:
            return None
        cur = str(pid).strip()
        if not cur:
            return None
    return None


def _hop_depth_edges_to_env_seed(
    tweet_id: str,
    pool: Dict[str, Any],
    seed_ids: Set[str],
    max_hops: int = 512,
) -> Optional[int]:
    """
    Root tweet is 0-hop: the number of edges required to reach an env seed root from the current tweet along the parent chain (retweet → quote → reply).
    Return None if cannot reach any seed.
    """
    if not seed_ids:
        return None
    cur = str(tweet_id).strip()
    edges = 0
    for _ in range(max_hops):
        if cur in seed_ids:
            return edges
        tw = pool.get(cur)
        if not isinstance(tw, dict):
            return None
        pid = _immediate_parent_tweet_id(tw)
        if not pid or pid == cur:
            return None
        cur = str(pid).strip()
        if not cur:
            return None
        edges += 1
    return None


def calculate_diffusion_hop_depth_over_time(data: Dict[str, Any]) -> Any:
    """Count the number of propagation tweets in content_pool by hop depth."""
    default: Dict[str, Any] = {"hop_1": 0.0, "hop_2": 0.0, "hop_3": 0.0}
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

        raw_seeds = safe_get(data, "seed_root_tweet_ids", None)
        if raw_seeds is None:
            raw_seeds = []
        if not isinstance(raw_seeds, (list, tuple, set)):
            raw_seeds = []
        seed_ids: Set[str] = {str(x).strip() for x in raw_seeds if str(x).strip()}

        depth_counts: Dict[int, int] = defaultdict(int)
        for bid, blog in content_pool.items():
            if not isinstance(blog, dict):
                continue
            if not _is_user_propagation_entry(str(bid), blog):
                continue
            d = _hop_depth_edges_to_env_seed(str(bid), content_pool, seed_ids)
            if d is None or d < 1:
                continue
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
            "diffusion_hop_depth_over_time",
            e,
            {"data_keys": list(data.keys()) if isinstance(data, dict) else None},
        )
        return default


def calculate_diffusion_volume_realtime(data: Dict[str, Any]) -> Any:
    """Calculate the number of reposts generated over time"""
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
            if not _is_user_propagation_entry(str(bid), blog):
                continue
            ms = time_to_ms(blog.get("time"))
            if ms is not None:
                times_ms.append(ms)

        times_ms.sort()
        edges, counts, desc = _histogram_repost_times_ms(times_ms)
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


def _parse_seed_root_tweet_ids(data: Dict[str, Any]) -> Set[str]:
    raw_seeds = safe_get(data, "seed_root_tweet_ids", None)
    if raw_seeds is None:
        raw_seeds = []
    if not isinstance(raw_seeds, (list, tuple, set)):
        raw_seeds = []
    return {str(x).strip() for x in raw_seeds if str(x).strip()}


def _collect_seed_root_propagation_stats(
    content_pool: Dict[str, Any],
    seed_ids: Set[str],
) -> Tuple[List[int], Dict[str, int], Set[str]]:
    pool_users: Set[str] = set()
    user_prop_totals: Dict[str, int] = defaultdict(int)
    for bid, b in content_pool.items():
        if not isinstance(b, dict):
            continue
        uid = b.get("user_id")
        if uid is None or str(uid).strip() == "":
            continue
        us = str(uid).strip()
        pool_users.add(us)
        if _is_user_propagation_entry(str(bid), b):
            root = _resolve_to_env_seed_root(str(bid), content_pool, seed_ids)
            if root is not None:
                user_prop_totals[us] += 1

    roots_in_pool = [s for s in sorted(seed_ids) if s in content_pool]
    per_root_descendants: List[int] = []
    for r in roots_in_pool:
        n = 0
        for bid, b in content_pool.items():
            if not isinstance(b, dict):
                continue
            if str(bid) == r:
                continue
            if _resolve_to_env_seed_root(str(bid), content_pool, seed_ids) == r:
                n += 1
        per_root_descendants.append(n)

    return per_root_descendants, user_prop_totals, pool_users


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


def calculate_user_diffusion_count_frequency(data: Dict[str, Any]) -> Any:
    """Distribution of seed-attributed propagation entries per user (denominator: pool authors)."""
    empty: Dict[str, Any] = {
        "_viz_kind": "user_repost_count_freq_bar",
        "repost_bins": [],
        "frequency_pct": [],
        "raw_counts": [],
        "n_users_in_pool": 0,
        "user_count_basis": "content_pool_presence",
    }
    metric_id = "user_diffusion_count_frequency"
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

        seed_ids = _parse_seed_root_tweet_ids(data)
        _per_root, user_prop_totals, pool_users = _collect_seed_root_propagation_stats(
            content_pool, seed_ids
        )
        n_pool_users = len(pool_users)
        if n_pool_users == 0:
            return empty

        per_user_counts = [user_prop_totals.get(u, 0) for u in sorted(pool_users)]
        bins, pct, raw = _count_frequency_histogram(per_user_counts, n_pool_users)
        return {
            "_viz_kind": "user_repost_count_freq_bar",
            "repost_bins": bins,
            "frequency_pct": pct,
            "raw_counts": raw,
            "n_users_in_pool": n_pool_users,
            "user_count_basis": "content_pool_presence",
        }
    except Exception as e:
        log_metric_error(
            metric_id,
            e,
            {"data_keys": list(data.keys()) if isinstance(data, dict) else None},
        )
        return empty


def calculate_root_diffusion_count_frequency(data: Dict[str, Any]) -> Any:
    """Distribution of propagation nodes under each env seed root tweet."""
    empty: Dict[str, Any] = {
        "_viz_kind": "root_repost_count_freq_bar",
        "repost_bins": [],
        "frequency_pct": [],
        "raw_counts": [],
        "n_root_tweets": 0,
    }
    metric_id = "root_diffusion_count_frequency"
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

        seed_ids = _parse_seed_root_tweet_ids(data)
        per_root_descendants, _user_totals, _pool_users = _collect_seed_root_propagation_stats(
            content_pool, seed_ids
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


def calculate_user_received_propagation_frequency(data: Dict[str, Any]) -> Any:
    """Calculate the frequency of user received propagation counts"""
    empty: Dict[str, Any] = {
        "_viz_kind": "received_propagation_freq_bar",
        "user_received_bins": [],
        "user_received_counts": [],
        "n_users_in_pool": 0,
    }
    try:
        if not data or not isinstance(data, dict):
            log_metric_error(
                "user_received_propagation_frequency",
                ValueError("invalid data"),
                {"data": data},
            )
            return empty

        content_pool = safe_get(data, "content_pool", {})
        if not isinstance(content_pool, dict):
            log_metric_error(
                "user_received_propagation_frequency",
                ValueError("content_pool is not a dict"),
                {"content_pool_type": type(content_pool)},
            )
            return empty

        pool_users: Set[str] = set()
        for _bid, b in content_pool.items():
            if not isinstance(b, dict):
                continue
            uid = b.get("user_id")
            if uid is None or str(uid).strip() == "":
                continue
            pool_users.add(str(uid).strip())

        received: Dict[str, int] = defaultdict(int)

        for _bid, tw in content_pool.items():
            if not isinstance(tw, dict):
                continue
            parent_ids: Set[str] = set()
            r = _tweet_ref_str(tw.get("retweeted_tweet_id"))
            if r:
                parent_ids.add(r)
            q = _tweet_ref_str(tw.get("quoted_tweet_id"))
            if q:
                parent_ids.add(q)
            rep = _tweet_ref_str(tw.get("replied_tweet_id")) or _tweet_ref_str(tw.get("replyed_tweet_id"))
            if rep:
                parent_ids.add(rep)

            for pid in parent_ids:
                parent = content_pool.get(pid)
                if not isinstance(parent, dict):
                    continue
                puid = parent.get("user_id")
                if puid is None or str(puid).strip() == "":
                    continue
                received[str(puid).strip()] += 1

        if not pool_users:
            return empty

        per_user = [received.get(u, 0) for u in pool_users]
        uh = Counter(per_user)
        max_k = max(per_user) if per_user else 0
        bins = list(range(0, max_k + 1))
        counts = [uh[k] for k in bins]

        return {
            "_viz_kind": "received_propagation_freq_bar",
            "user_received_bins": bins,
            "user_received_counts": counts,
            "n_users_in_pool": len(pool_users),
        }
    except Exception as e:
        log_metric_error(
            "user_received_propagation_frequency",
            e,
            {"data_keys": list(data.keys()) if isinstance(data, dict) else None},
        )
        return empty


def calculate_received_propagation_count_frequency(data: Dict[str, Any]) -> Any:
    """    Calculate the frequency of received propagation counts"""
    empty: Dict[str, Any] = {
        "_viz_kind": "received_propagation_count_freq_bar",
        "root_repost_bins": [],
        "root_repost_counts": [],
        "n_root_tweets": 0,
        "user_repost_bins": [],
        "user_repost_counts": [],
        "n_users_in_pool": 0,
    }
    try:
        if not data or not isinstance(data, dict):
            log_metric_error(
                "received_propagation_count_frequency",
                ValueError("invalid data"),
                {"data": data},
            )
            return empty

        content_pool = safe_get(data, "content_pool", {})
        if not isinstance(content_pool, dict):
            log_metric_error(
                "received_propagation_count_frequency",
                ValueError("content_pool is not a dict"),
                {"content_pool_type": type(content_pool)},
            )
            return empty

        raw_seeds = safe_get(data, "seed_root_tweet_ids", None)
        if raw_seeds is None:
            raw_seeds = []
        if not isinstance(raw_seeds, (list, tuple, set)):
            raw_seeds = []
        seed_ids: Set[str] = {str(x).strip() for x in raw_seeds if str(x).strip()}

        pool_users: Set[str] = set()
        user_received_totals: Dict[str, int] = defaultdict(int)
        for bid, b in content_pool.items():
            if not isinstance(b, dict):
                continue
            uid = b.get("user_id")
            if uid is None or str(uid).strip() == "":
                continue
            pool_users.add(str(uid).strip())

            if not _is_user_propagation_entry(str(bid), b):
                continue
            if _resolve_to_env_seed_root(str(bid), content_pool, seed_ids) is None:
                continue
            pid = _immediate_parent_tweet_id(b)
            if not pid:
                continue
            parent = content_pool.get(pid)
            if not isinstance(parent, dict):
                continue
            puid = parent.get("user_id")
            if puid is None or str(puid).strip() == "":
                continue
            user_received_totals[str(puid).strip()] += 1

        n_pool_users = len(pool_users)
        roots_in_pool = [s for s in sorted(seed_ids) if s in content_pool]
        n_roots = len(roots_in_pool)
        per_root_descendants: List[int] = []
        for r in roots_in_pool:
            n = 0
            for bid, b in content_pool.items():
                if not isinstance(b, dict):
                    continue
                if str(bid) == r:
                    continue
                if _resolve_to_env_seed_root(str(bid), content_pool, seed_ids) == r:
                    n += 1
            per_root_descendants.append(n)

        if n_roots == 0 and n_pool_users == 0:
            return empty

        rh = Counter(per_root_descendants)
        max_k_root = max(per_root_descendants) if per_root_descendants else 0
        root_bins = list(range(0, max_k_root + 1))
        root_counts = [rh[k] for k in root_bins]

        if n_pool_users > 0:
            per_user = [user_received_totals[u] for u in pool_users]
            uh = Counter(per_user)
            max_k_user = max(per_user) if per_user else 0
            user_bins = list(range(0, max_k_user + 1))
            user_counts = [uh[k] for k in user_bins]
        else:
            user_bins = []
            user_counts = []

        return {
            "_viz_kind": "received_propagation_count_freq_bar",
            "root_repost_bins": root_bins,
            "root_repost_counts": root_counts,
            "n_root_tweets": n_roots,
            "user_repost_bins": user_bins,
            "user_repost_counts": user_counts,
            "n_users_in_pool": n_pool_users,
        }
    except Exception as e:
        log_metric_error(
            "received_propagation_count_frequency",
            e,
            {"data_keys": list(data.keys()) if isinstance(data, dict) else None},
        )
        return empty


def _count_direct_by_type(content_pool: Dict[str, Any], propagation_type: str) -> float:
    """Calculate the number of non-secondary propagation tweets of a specified type (retweet/quote/reply)"""
    if not isinstance(content_pool, dict):
        return 0.0

    if propagation_type not in {"retweet", "quote", "reply"}:
        return 0.0

    n = 0
    for tw in content_pool.values():
        if not isinstance(tw, dict):
            continue

        if propagation_type == "retweet":
            parent_id = _tweet_ref_str(tw.get("retweeted_tweet_id"))
        elif propagation_type == "quote":
            parent_id = _tweet_ref_str(tw.get("quoted_tweet_id"))
        else:
            parent_id = _tweet_ref_str(tw.get("replied_tweet_id")) or _tweet_ref_str(tw.get("replyed_tweet_id"))

        if not parent_id:
            continue

        parent = content_pool.get(parent_id)
        if not isinstance(parent, dict):
            continue
        if _is_original_tweet_obj(parent):
            n += 1

    return float(n)


def calculate_direct_retweet_generation(data: Dict[str, Any]) -> Any:
    """Calculate the number of non-secondary retweets"""
    try:
        if not data or not isinstance(data, dict):
            log_metric_error("direct_retweet_generation", ValueError("Invalid data input"), {"data": data})
            return 0
        content_pool = safe_get(data, "content_pool", {})
        if count_direct_propagation_by_type is not None:
            return float(count_direct_propagation_by_type(content_pool, "retweet"))
        return _count_direct_by_type(content_pool, "retweet")
    except Exception as e:
        log_metric_error(
            "direct_retweet_generation",
            e,
            {"data_keys": list(data.keys()) if isinstance(data, dict) else None},
        )
        return 0


def calculate_direct_quote_generation(data: Dict[str, Any]) -> Any:
    """Calculate the number of non-secondary quotes"""
    try:
        if not data or not isinstance(data, dict):
            log_metric_error("direct_quote_generation", ValueError("Invalid data input"), {"data": data})
            return 0
        content_pool = safe_get(data, "content_pool", {})
        if count_direct_propagation_by_type is not None:
            return float(count_direct_propagation_by_type(content_pool, "quote"))
        return _count_direct_by_type(content_pool, "quote")
    except Exception as e:
        log_metric_error(
            "direct_quote_generation",
            e,
            {"data_keys": list(data.keys()) if isinstance(data, dict) else None},
        )
        return 0


def calculate_direct_reply_generation(data: Dict[str, Any]) -> Any:
    """Calculate the number of non-secondary replies"""
    try:
        if not data or not isinstance(data, dict):
            log_metric_error("direct_reply_generation", ValueError("Invalid data input"), {"data": data})
            return 0
        content_pool = safe_get(data, "content_pool", {})
        if count_direct_propagation_by_type is not None:
            return float(count_direct_propagation_by_type(content_pool, "reply"))
        return _count_direct_by_type(content_pool, "reply")
    except Exception as e:
        log_metric_error(
            "direct_reply_generation",
            e,
            {"data_keys": list(data.keys()) if isinstance(data, dict) else None},
        )
        return 0


def calculate_posting_root_author_repost_behavior(data: Dict[str, Any]) -> Any:
    """Calculate the posting root author repost behavior"""
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

        raw_seeds = safe_get(data, "seed_root_tweet_ids", None)
        if raw_seeds is None:
            raw_seeds = []
        if not isinstance(raw_seeds, (list, tuple, set)):
            raw_seeds = []
        seed_ids: Set[str] = {str(x).strip() for x in raw_seeds if str(x).strip()}

        root_authors: Set[str] = set()
        root_post_count: Dict[str, int] = defaultdict(int)
        nickname_map: Dict[str, str] = {}

        for bid, b in content_pool.items():
            if str(bid) not in seed_ids:
                continue
            if not isinstance(b, dict):
                continue
            uid = b.get("user_id")
            if uid is None or str(uid).strip() == "":
                continue
            us = str(uid).strip()
            root_authors.add(us)
            root_post_count[us] += 1
            nn = b.get("nickname")
            if isinstance(nn, str) and nn.strip():
                nickname_map.setdefault(us, nn.strip())

        if not root_authors:
            return empty

        self_hop: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        other_root_repost: Dict[str, int] = defaultdict(int)

        for bid, blog in content_pool.items():
            if not isinstance(blog, dict):
                continue
            if not _is_user_propagation_entry(str(bid), blog):
                continue

            root_id = _resolve_to_env_seed_root(str(bid), content_pool, seed_ids)
            if root_id is None:
                continue
            rb = content_pool.get(root_id)
            if not isinstance(rb, dict):
                continue
            ra = rb.get("user_id")
            if ra is None or str(ra).strip() == "":
                continue
            ra_s = str(ra).strip()

            author = blog.get("user_id")
            if author is None or str(author).strip() == "":
                continue
            ua = str(author).strip()

            d = _hop_depth_edges_to_env_seed(str(bid), content_pool, seed_ids)
            if d is None or d < 1:
                continue

            if ua == ra_s:
                self_hop[ua][d] += 1

        for bid, blog in content_pool.items():
            if not isinstance(blog, dict):
                continue
            if not _is_user_propagation_entry(str(bid), blog):
                continue
            author = blog.get("user_id")
            if author is None or str(author).strip() == "":
                continue
            ua = str(author).strip()
            if ua not in root_authors:
                continue

            root_id = _resolve_to_env_seed_root(str(bid), content_pool, seed_ids)
            if root_id is None:
                continue
            rb = content_pool.get(root_id)
            if not isinstance(rb, dict):
                continue
            ra = rb.get("user_id")
            if ra is None or str(ra).strip() == "":
                continue
            ra_s = str(ra).strip()
            if ua == ra_s:
                continue
            other_root_repost[ua] += 1

        max_h = 1
        for dd in self_hop.values():
            if dd:
                max_h = max(max_h, max(dd.keys()))

        rows: List[Dict[str, Any]] = []
        for uid in sorted(root_authors):
            hops_u = self_hop.get(uid, {})
            one_hop = int(hops_u.get(1, 0))
            multi_hop = sum(int(hops_u.get(k, 0)) for k in hops_u if k >= 2)
            row: Dict[str, Any] = {
                "user_id": uid,
                "nickname": nickname_map.get(uid, ""),
                "root_post_count": int(root_post_count.get(uid, 0)),
                "repost_on_others_count": int(other_root_repost.get(uid, 0)),
                "self_propagation_one_hop": one_hop,
                "self_propagation_multi_hop": multi_hop,
            }
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


# 指标函数字典，用于查找
METRIC_FUNCTIONS = {
    "calculate_diffusion_generation": calculate_diffusion_generation,
    "calculate_direct_tweet_generation": calculate_direct_tweet_generation,
    "calculate_direct_retweet_generation": calculate_direct_retweet_generation,
    "calculate_direct_quote_generation": calculate_direct_quote_generation,
    "calculate_direct_reply_generation": calculate_direct_reply_generation,
    "calculate_user_diffusion_count_frequency": calculate_user_diffusion_count_frequency,
    "calculate_root_diffusion_count_frequency": calculate_root_diffusion_count_frequency,
    "calculate_diffusion_hop_depth_over_time": calculate_diffusion_hop_depth_over_time,
    "calculate_diffusion_volume_realtime": calculate_diffusion_volume_realtime,
    "calculate_posting_root_author_repost_behavior": calculate_posting_root_author_repost_behavior,
    "calculate_user_received_propagation_frequency": calculate_user_received_propagation_frequency,
    "calculate_received_propagation_count_frequency": calculate_received_propagation_count_frequency,
}
if calculate_text_similarity is not None:
    METRIC_FUNCTIONS["calculate_text_similarity"] = calculate_text_similarity
if calculate_text_diversity is not None:
    METRIC_FUNCTIONS["calculate_text_diversity"] = calculate_text_diversity
if calculate_text_max_reference_similarity is not None:
    METRIC_FUNCTIONS["calculate_text_max_reference_similarity"] = calculate_text_max_reference_similarity


def get_metric_function(function_name: str) -> Optional[Callable]:
    """Get the metric function by name"""
    return METRIC_FUNCTIONS.get(function_name)
