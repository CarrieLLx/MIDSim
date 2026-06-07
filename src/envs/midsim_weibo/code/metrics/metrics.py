# -*- coding: utf-8 -*-
"""
多渠道信息传播模型的监控指标计算模块
"""
import os
import sys
_metrics_dir = os.path.dirname(os.path.abspath(__file__))
if _metrics_dir not in sys.path:
    sys.path.insert(0, _metrics_dir)
try:
    from repost_similarity import calculate_repost_similarity
except ImportError:
    calculate_repost_similarity = None

from typing import Dict, Any, List, Optional, Union, Callable, Set, Tuple
from collections import Counter, defaultdict
from loguru import logger
from onesim.monitor.utils import (
    safe_get, safe_number, safe_list, safe_sum, 
    safe_avg, safe_max, safe_min, safe_count, log_metric_error
)


def calculate_repost_generation(data: Dict[str, Any]) -> Any:
    """
    计算指标: repost_generation
    描述: 统计随时间变化的转发相关规模，用于追踪信息传播的活跃度
    可视化类型: line
    更新频率: 5 秒

    统计 content_pool 顶层条目中 `reposted_blog_id` 非空（strip 后非空串）的条数；
    即视为「有父帖引用的传播类帖子」。不遍历嵌套 `reposts`。

    Args:
        data: 包含环境数据的字典，应该包含 content_pool 字段

    Returns:
        float: 上述条数
    """
    try:
        # 验证输入数据
        if not data or not isinstance(data, dict):
            log_metric_error("repost_generation", ValueError("无效的数据输入"), {"data": data})
            return 0
        
        # 获取 content_pool（字典格式，键为 blog_id）
        content_pool = safe_get(data, "content_pool", {})
        
        if not isinstance(content_pool, dict):
            log_metric_error("repost_generation", ValueError("content_pool 不是字典格式"), {"content_pool_type": type(content_pool)})
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
        log_metric_error("repost_generation", e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return 0


def _blog_time_to_ms(value: Any) -> Optional[float]:
    """Normalize blog `time` to milliseconds (same rules as comment timestamps)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        x = float(value)
        if x <= 0:
            return None
        if x < 1e11:
            return x * 1000.0
        return x
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return _blog_time_to_ms(float(s))
        except ValueError:
            return None
    return None


def _histogram_repost_times_ms(times_ms: List[float]) -> Tuple[List[float], List[int], str]:
    """Equal-width bins over real time span (same strategy as comment volume realtime)."""
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


def calculate_repost_volume_realtime(data: Dict[str, Any]) -> Any:
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
                "repost_volume_realtime",
                ValueError("invalid data"),
                {"data": data},
            )
            return empty

        content_pool = safe_get(data, "content_pool", {})
        if not isinstance(content_pool, dict):
            log_metric_error(
                "repost_volume_realtime",
                ValueError("content_pool is not a dict"),
                {"content_pool_type": type(content_pool)},
            )
            return empty

        times_ms: List[float] = []
        for bid, blog in content_pool.items():
            if not isinstance(blog, dict):
                continue
            if not _is_user_repost_entry(str(bid), blog):
                continue
            ms = _blog_time_to_ms(blog.get("time"))
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
            "repost_volume_realtime",
            e,
            {"data_keys": list(data.keys()) if isinstance(data, dict) else None},
        )
        return empty


def _root_blog_id_for_entry(
    blog_id: str,
    blog: Dict[str, Any],
    pool: Dict[str, Any],
    cache: Dict[str, str],
) -> str:
    """Resolve chain root blog_id (multi-hop via reposted_path or reposted_blog_id)."""
    if blog_id in cache:
        return cache[blog_id]
    path = blog.get("reposted_path")
    if isinstance(path, list) and len(path) > 0:
        r0 = path[0]
        if r0 is not None and str(r0).strip() != "":
            cache[blog_id] = str(r0).strip()
            return cache[blog_id]
    pid = blog.get("reposted_blog_id")
    ps = str(pid).strip() if pid is not None else ""
    bid = str(blog_id).strip()
    if not ps or ps == bid:
        cache[blog_id] = bid
        return cache[blog_id]
    if ps not in pool:
        cache[blog_id] = bid
        return cache[blog_id]
    parent = pool[ps]
    if not isinstance(parent, dict):
        cache[blog_id] = bid
        return cache[blog_id]
    r = _root_blog_id_for_entry(ps, parent, pool, cache)
    cache[blog_id] = r
    return r


def _is_original_root_blog(blog_id: str, blog: Dict[str, Any], pool: Dict[str, Any], cache: Dict[str, str]) -> bool:
    """True if this entry is the root post of its chain (counts as one root tweet)."""
    r = _root_blog_id_for_entry(blog_id, blog, pool, cache)
    return str(blog_id).strip() == str(r).strip()


def _is_user_repost_entry(blog_id: str, blog: Dict[str, Any]) -> bool:
    """True if this blog is a repost of another blog (excludes original where reposted_blog_id == self)."""
    pid = blog.get("reposted_blog_id")
    if pid is None:
        return False
    ps = str(pid).strip()
    if not ps:
        return False
    return ps != str(blog_id).strip()


def _repost_hop_depth(
    blog_id: str,
    blog: Dict[str, Any],
    _pool: Dict[str, Any],
    memo: Dict[str, int],
) -> int:
    """
    转发级数 hop_k：**仅**由 reposted_path 中非空 blog_id 的个数决定（k = 该数量）。
    1 个节点 = 一级转发，2 个 = 二级，以此类推。
    reposted_path 缺失或为空列表时，退化为 1（视为一级转发条目）。
    根帖上的 repost_ids 不参与本指标。
    """
    bid = str(blog_id).strip()
    if bid in memo:
        return memo[bid]
    if not _is_user_repost_entry(bid, blog):
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


def calculate_repost_hop_depth_over_time(data: Dict[str, Any]) -> Any:
    """
    At each monitor sample: count repost entries in content_pool by hop depth (1,2,3,...).
    级数由 _repost_hop_depth 定义：仅 reposted_path 非空节点个数（见该函数说明）。
    Same semantics as Comment Top-Level vs Reply Over Time — totals in pool, time series from samples.
    """
    default = {"hop_1": 0.0, "hop_2": 0.0, "hop_3": 0.0}
    try:
        if not data or not isinstance(data, dict):
            log_metric_error(
                "repost_hop_depth_over_time",
                ValueError("invalid data"),
                {"data": data},
            )
            return default

        content_pool = safe_get(data, "content_pool", {})
        if not isinstance(content_pool, dict):
            log_metric_error(
                "repost_hop_depth_over_time",
                ValueError("content_pool is not a dict"),
                {"content_pool_type": type(content_pool)},
            )
            return default

        memo: Dict[str, int] = {}
        depth_counts: Dict[int, int] = defaultdict(int)
        for bid, blog in content_pool.items():
            if not isinstance(blog, dict):
                continue
            if not _is_user_repost_entry(str(bid), blog):
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


def calculate_repost_count_frequency(data: Dict[str, Any]) -> Any:
    """
    Two distributions (raw counts on Y; X = repost count including 0):

    1) Per root tweet: total repost nodes under that root (all levels). Count only non-root blogs
       whose chain root equals R; root R has repost total = that count.
    2) Per user: number of repost entries authored by that user (original posts not counted).
       Denominator users: union of (optional) user_agent_profile_ids from all UserAgents and every user_id
       seen as blog author in content_pool; users with 0 reposts included.
    """
    empty: Dict[str, Any] = {
        "_viz_kind": "repost_count_freq_bar",
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
                "repost_count_frequency",
                ValueError("invalid data"),
                {"data": data},
            )
            return empty

        content_pool = safe_get(data, "content_pool", {})
        if not isinstance(content_pool, dict):
            log_metric_error(
                "repost_count_frequency",
                ValueError("content_pool is not a dict"),
                {"content_pool_type": type(content_pool)},
            )
            return empty

        cache: Dict[str, str] = {}
        root_ids: List[str] = []
        for bid, b in content_pool.items():
            if not isinstance(b, dict):
                continue
            if _is_original_root_blog(str(bid), b, content_pool, cache):
                root_ids.append(str(bid))

        n_roots = len(root_ids)
        per_root_descendants: List[int] = []
        for r in root_ids:
            n = 0
            for bid, b in content_pool.items():
                if not isinstance(b, dict):
                    continue
                if str(bid) == r:
                    continue
                root = _root_blog_id_for_entry(str(bid), b, content_pool, cache)
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
            if _is_user_repost_entry(str(bid), b):
                user_repost_totals[us] += 1

        # 与 profile/data/UserAgent.json 中全体智能体对齐：未在 content_pool 中出现过作者行的用户也计入分母（0 转发）
        registered = data.get("user_agent_profile_ids")
        if isinstance(registered, list):
            for x in registered:
                if x is None:
                    continue
                s = str(x).strip()
                if s:
                    pool_users.add(s)

        n_pool_users = len(pool_users)
        if n_roots == 0 and n_pool_users == 0:
            return empty

        rh = Counter(per_root_descendants)
        max_k_root = max(per_root_descendants) if per_root_descendants else 0
        root_bins = list(range(0, max_k_root + 1))
        root_counts = [rh[k] for k in root_bins]

        if n_pool_users > 0:
            per_user = [user_repost_totals[u] for u in pool_users]
            uh = Counter(per_user)
            max_k_user = max(per_user) if per_user else 0
            user_bins = list(range(0, max_k_user + 1))
            user_counts = [uh[k] for k in user_bins]
        else:
            user_bins = []
            user_counts = []

        return {
            "_viz_kind": "repost_count_freq_bar",
            "root_repost_bins": root_bins,
            "root_repost_counts": root_counts,
            "n_root_tweets": n_roots,
            "user_repost_bins": user_bins,
            "user_repost_counts": user_counts,
            "n_users_in_pool": n_pool_users,
        }
    except Exception as e:
        log_metric_error(
            "repost_count_frequency",
            e,
            {"data_keys": list(data.keys()) if isinstance(data, dict) else None},
        )
        return empty


def calculate_posting_root_author_repost_behavior(data: Dict[str, Any]) -> Any:
    """
    Per root-tweet author:
    - self_repost_hop_k: 根作者在自己根帖链路上的第 k 跳自转发（与链上根作者一致）。
    - repost_on_others_count: 该用户在**他人根帖**下的转发条数（链路的根帖作者不是本人）。

    Same contract as Posting User Comment Behavior: {"users": [ {...}, ... ]}.
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
            if _is_original_root_blog(str(bid), b, content_pool, cache):
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
            if not _is_user_repost_entry(str(bid), blog):
                continue
            author = blog.get("user_id")
            if author is None or str(author).strip() == "":
                continue
            ua = str(author).strip()
            root_id = _root_blog_id_for_entry(str(bid), blog, content_pool, cache)
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
            if not _is_user_repost_entry(str(bid), blog):
                continue
            author = blog.get("user_id")
            if author is None or str(author).strip() == "":
                continue
            ua = str(author).strip()
            if ua not in root_authors:
                continue
            root_id = _root_blog_id_for_entry(str(bid), blog, content_pool, cache)
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


# 指标函数字典，用于查找
METRIC_FUNCTIONS = {
    'calculate_repost_generation': calculate_repost_generation,
    'calculate_repost_volume_realtime': calculate_repost_volume_realtime,
    'calculate_repost_hop_depth_over_time': calculate_repost_hop_depth_over_time,
    'calculate_repost_count_frequency': calculate_repost_count_frequency,
    'calculate_posting_root_author_repost_behavior': calculate_posting_root_author_repost_behavior,
}
if calculate_repost_similarity is not None:
    METRIC_FUNCTIONS['calculate_repost_similarity'] = calculate_repost_similarity


def get_metric_function(function_name: str) -> Optional[Callable]:
    """
    根据函数名获取对应的指标计算函数
    
    Args:
        function_name: 函数名
        
    Returns:
        指标计算函数或None
    """
    return METRIC_FUNCTIONS.get(function_name)
