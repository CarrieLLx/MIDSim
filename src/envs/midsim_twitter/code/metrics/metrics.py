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
    from twitter_similarity import (
        calculate_tweet_text_similarity,
        count_direct_content_pool_tweets_for_monitor,
        count_direct_propagation_by_type,
        count_content_pool_tweets_for_monitor,
    )
except ImportError:
    calculate_tweet_text_similarity = None
    count_direct_content_pool_tweets_for_monitor = None
    count_direct_propagation_by_type = None
    count_content_pool_tweets_for_monitor = None

from typing import Dict, Any, List, Optional, Union, Callable, Set, Tuple
from collections import Counter, defaultdict
from loguru import logger
from onesim.monitor.utils import (
    safe_get, safe_number, safe_list, safe_sum, 
    safe_avg, safe_max, safe_min, safe_count, log_metric_error
)


def calculate_tweet_generation(data: Dict[str, Any]) -> Any:
    """
    计算指标: tweet_generation（Twitter：content_pool 中「传播类」推文条数）

    仅统计转推 / 引用 / 回复（任一类对应 id 字段有值）；**所有原创推一律不计**。

    依赖 data.content_pool。

    「非二级」传播：父帖为原创（父帖的 retweeted_tweet_id / quoted_tweet_id /
    replied_tweet_id / replyed_tweet_id 均为空）。**分类型**折线图见：
    `calculate_direct_retweet_generation` / `calculate_direct_quote_generation` /
    `calculate_direct_reply_generation`（scene_info.json 中 `direct_*_generation`，均为 line）。
    """
    try:
        if not data or not isinstance(data, dict):
            log_metric_error("tweet_pool_count", ValueError("无效的数据输入"), {"data": data})
            return 0

        content_pool = safe_get(data, "content_pool", {})
        if not isinstance(content_pool, dict):
            log_metric_error(
                "tweet_pool_count",
                ValueError("content_pool 不是字典格式"),
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

        # 无 twitter_similarity 时退化：仅统计 dict 条目数
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
    """
    计算指标: direct_tweet_generation（Twitter：content_pool 中「非二级传播」推文条数）

    仅统计转推 / 引用 / 回复中，其父推文是原创推的条目（即父推文的
    retweeted_tweet_id、quoted_tweet_id、replied_tweet_id 均为空）。

    依赖 data.content_pool。
    """
    try:
        if not data or not isinstance(data, dict):
            log_metric_error("tweet_pool_count", ValueError("无效的数据输入"), {"data": data})
            return 0

        content_pool = safe_get(data, "content_pool", {})
        if not isinstance(content_pool, dict):
            log_metric_error(
                "tweet_pool_count",
                ValueError("content_pool 不是字典格式"),
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

        # 无 twitter_similarity 时退化：返回 0（无法可靠识别父推文层级）
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


def _tweet_time_to_ms(value: Any) -> Optional[float]:
    """将 tweet 的 `time` 规范为毫秒（与微博 blog time 规则一致）。"""
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
            return _tweet_time_to_ms(float(s))
        except ValueError:
            return None
    return None


def _histogram_repost_times_ms(times_ms: List[float]) -> Tuple[List[float], List[int], str]:
    """在真实时间跨度上等宽分箱（与微博 Repost Volume Real Time 一致）。"""
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
    """向上一跳：优先 retweet → quote → reply（与溯源顺序一致）。"""
    for key in ("retweeted_tweet_id", "quoted_tweet_id", "replied_tweet_id", "replyed_tweet_id"):
        s = _tweet_ref_str(tw.get(key))
        if s:
            return s
    return None


def _is_user_propagation_entry(tweet_id: str, tw: Dict[str, Any]) -> bool:
    """非原创传播帖（任一类父引用存在且非自指）。"""
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
    """
    沿 retweet / quote / reply 父链向上追溯，直到落在 env_data 初始种子 id（seed_root_tweet_ids）上。
    无法到达任一种子则返回 None。
    """
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
    根推文为第 0 跳：从当前帖沿父链（retweet → quote → reply）向上走到某一 env 种子根所需的**边数**，
    即该传播帖的 hop_k 中的 k。无法到达任一种子则返回 None。
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


def calculate_repost_hop_depth_over_time(data: Dict[str, Any]) -> Any:
    """
    与微博 Repost Hop Depth Over Time 对齐：按「跳数」统计 content_pool 中传播类帖子的条数。

    跳数定义：从当前帖沿 retweeted_tweet_id / quoted_tweet_id / replied_tweet_id（及 replyed_tweet_id）
    逐跳向上，直到落在 seed_root_tweet_ids 中的根推；根为第 0 跳，边数 k≥1 计入 hop_k。
    仅统计可归因到 env 种子的传播帖；无法溯源到种子的条目不计入。
    """
    default: Dict[str, Any] = {"hop_1": 0.0, "hop_2": 0.0, "hop_3": 0.0}
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
            "repost_hop_depth_over_time",
            e,
            {"data_keys": list(data.keys()) if isinstance(data, dict) else None},
        )
        return default


def calculate_repost_volume_realtime(data: Dict[str, Any]) -> Any:
    """
    与微博 Repost Volume Real Time 同一契约（_viz_kind=repost_realtime）：

    每条传播类推文用其 `time`（毫秒）作为事件时刻；排序后：
    - 监控上子图 1：按时间累计传播条数（总量随真实时间变化）；
    - 子图 2：等宽时间箱内条数（单位时间内的增量 / 强度）。

    传播类：retweet / quote / reply（见 `_is_user_propagation_entry`）；需有效 `time`。
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
            if not _is_user_propagation_entry(str(bid), blog):
                continue
            ms = _tweet_time_to_ms(blog.get("time"))
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


def calculate_repost_count_frequency(data: Dict[str, Any]) -> Any:
    """
    与微博 Repost Count Frequency 同一套导出字段（_viz_kind=repost_count_freq_bar）：

    1) 对每个 env 种子根推：统计 content_pool 中可归因到该根下的**传播节点总数**（不含根自身），
       横轴为传播数、纵轴为具有该传播数的根推条数。
    2) 对每个在池中出现过的作者：统计其发布的、且可归因到任一 env 种子的传播帖条数，
       横轴为该条数、纵轴为具有该条数的用户数。

    溯源：quoted_tweet_id / retweeted_tweet_id / replied_tweet_id（及 replyed_tweet_id）逐跳向上，
    直至命中 seed_root_tweet_ids（由 env 在加载 env_data.json 后写入，对应初始 content_pool 的 key）。
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

        raw_seeds = safe_get(data, "seed_root_tweet_ids", None)
        if raw_seeds is None:
            raw_seeds = []
        if not isinstance(raw_seeds, (list, tuple, set)):
            raw_seeds = []
        seed_ids: Set[str] = {str(x).strip() for x in raw_seeds if str(x).strip()}

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
            per_user = [user_prop_totals[u] for u in pool_users]
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


def calculate_user_received_propagation_frequency(data: Dict[str, Any]) -> Any:
    """
    用户「被直接传播」次数分布（不溯源、不沿父链向上累加）：

    遍历 content_pool 中每条传播帖，仅根据其 **直接** 父 id：
    retweeted_tweet_id、quoted_tweet_id、replied_tweet_id（及 replyed_tweet_id）在池中定位父帖，
    将 +1 记到 **父帖作者** 上。同一子帖若同时填多类引用（极少见），则按字段分别各 +1。

    横轴 k = 该用户收到的直接传播次数合计；纵轴 = 恰好为 k 次的用户数（含 k=0：池中出现过但未收到任何直接引用的作者）。
    """
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
    """
    与 `calculate_repost_count_frequency` 同一套导出字段（_viz_kind=received_propagation_count_freq_bar），
    双子图契约与 Repost Count Frequency 一致：

    1) 根推：与 Repost Count Frequency 相同 —— 每个 env 种子根下可归因的**总节点数**（不含根自身），
       横轴为传播数、纵轴为具有该传播数的根推条数。
    2) 用户：对每个在池中出现过的作者，统计 **可归因到任一种子的传播帖** 中，有多少条的
       **直接父帖**（`_immediate_parent_tweet_id`，与溯源顺序一致）由该用户发布 ——
       即「在种子树内、一跳指向该用户帖」的被传播次数；横轴为该次数、纵轴为具有该次数的用户数（含 0）。

    与「User Received Propagation Frequency」区别：本指标仅统计 **seed_root_tweet_ids** 溯源可达的
    传播帖上的父边；与 Repost Count Frequency 用户图区别：彼处统计用户**发出**的传播帖条数，此处统计**收到**的一跳次数。
    """
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
    """
    统计指定传播类型（retweet/quote/reply）的「非二级传播」条数：
    当前 tweet 为该类型，且其父 tweet 为原创（父 tweet 的三类 ref id 均为空）。
    """
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
    """统计非二级 retweet 条数（父 tweet 为原创：三类 ref id 均为空）。"""
    try:
        if not data or not isinstance(data, dict):
            log_metric_error("direct_retweet_generation", ValueError("无效的数据输入"), {"data": data})
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
    """统计非二级 quote 条数（父 tweet 为原创：三类 ref id 均为空）。"""
    try:
        if not data or not isinstance(data, dict):
            log_metric_error("direct_quote_generation", ValueError("无效的数据输入"), {"data": data})
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
    """统计非二级 reply 条数（父 tweet 为原创：三类 ref id 均为空）。"""
    try:
        if not data or not isinstance(data, dict):
            log_metric_error("direct_reply_generation", ValueError("无效的数据输入"), {"data": data})
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

def calculate_original_tweet_count(data: Dict[str, Any]) -> Any:
    """
    计算指标: original_tweet_count（Twitter：content_pool 中原创推文条数）

    统计 retweeted_tweet_id、quoted_tweet_id、replied_tweet_id/replyed_tweet_id
    均为空的条目数。
    """
    try:
        if not data or not isinstance(data, dict):
            log_metric_error("original_tweet_count", ValueError("无效的数据输入"), {"data": data})
            return 0

        content_pool = safe_get(data, "content_pool", {})
        if not isinstance(content_pool, dict):
            log_metric_error(
                "original_tweet_count",
                ValueError("content_pool 不是字典格式"),
                {"content_pool_type": type(content_pool)},
            )
            return 0

        def _is_empty_ref(v: Any) -> bool:
            if v is None:
                return True
            if isinstance(v, str):
                return (v.strip() == "")
            return False

        n = 0
        for tw in content_pool.values():
            if not isinstance(tw, dict):
                continue
            rid = tw.get("retweeted_tweet_id")
            qid = tw.get("quoted_tweet_id")
            repid = tw.get("replied_tweet_id")
            repid2 = tw.get("replyed_tweet_id")
            if _is_empty_ref(rid) and _is_empty_ref(qid) and _is_empty_ref(repid) and _is_empty_ref(repid2):
                n += 1
        return float(n)
    except Exception as e:
        log_metric_error(
            "original_tweet_count",
            e,
            {"data_keys": list(data.keys()) if isinstance(data, dict) else None},
        )
        return 0


def calculate_posting_root_author_repost_behavior(data: Dict[str, Any]) -> Any:
    """
    与微博 Root Author Self-Repost Behavior 相同契约：`{"users": [ {...}, ... ]}`。

    根推文限定为 env 种子（seed_root_tweet_ids ∩ content_pool）。对每个**至少发过一条种子根推**的用户：
    - root_post_count：其作为作者的种子根推条数；
    - self_repost_hop_k：该用户在自己根推链路上的传播帖中，相对种子根为第 k 跳的条数（k≥1；一跳=直连根，多跳=k≥2）；
    - self_propagation_one_hop / self_propagation_multi_hop：同上的一跳与多跳（≥2）汇总；
    - repost_on_others_count：该用户作为根作者，在**他人根推**链路上的传播帖条数。

    溯源：retweeted_tweet_id → quoted_tweet_id → replied_tweet_id / replyed_tweet_id 向上至 seed_root_tweet_ids；
    跳数与 `calculate_repost_hop_depth_over_time` / `_hop_depth_edges_to_env_seed` 一致。
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
    "calculate_tweet_generation": calculate_tweet_generation,
    "calculate_direct_tweet_generation": calculate_direct_tweet_generation,
    "calculate_direct_retweet_generation": calculate_direct_retweet_generation,
    "calculate_direct_quote_generation": calculate_direct_quote_generation,
    "calculate_direct_reply_generation": calculate_direct_reply_generation,
    "calculate_original_tweet_count": calculate_original_tweet_count,
    "calculate_repost_count_frequency": calculate_repost_count_frequency,
    "calculate_repost_hop_depth_over_time": calculate_repost_hop_depth_over_time,
    "calculate_repost_volume_realtime": calculate_repost_volume_realtime,
    "calculate_posting_root_author_repost_behavior": calculate_posting_root_author_repost_behavior,
    "calculate_user_received_propagation_frequency": calculate_user_received_propagation_frequency,
    "calculate_received_propagation_count_frequency": calculate_received_propagation_count_frequency,
}
if calculate_tweet_text_similarity is not None:
    METRIC_FUNCTIONS["calculate_tweet_text_similarity"] = calculate_tweet_text_similarity


def get_metric_function(function_name: str) -> Optional[Callable]:
    """
    根据函数名获取对应的指标计算函数
    
    Args:
        function_name: 函数名
        
    Returns:
        指标计算函数或None
    """
    return METRIC_FUNCTIONS.get(function_name)
