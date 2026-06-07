# -*- coding: utf-8 -*-
"""
多渠道信息传播模型的监控指标计算模块
"""
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import os
import sys
_metrics_dir = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_metrics_dir, "../../../../.."))
if _metrics_dir not in sys.path:
    sys.path.insert(0, _metrics_dir)
try:
    from comment_similarity import calculate_comment_similarity
except ImportError:
    calculate_comment_similarity = None
try:
    from comment_source_mix import count_comment_source_mix
except ImportError:
    count_comment_source_mix = None
try:
    from recommendation_coverage_metric import (
        compute_step_login_validity_miss_metrics,
        compute_step_recommendation_coverage,
        load_comment_user_note_pairs,
    )
except ImportError:
    load_comment_user_note_pairs = None
    compute_step_recommendation_coverage = None
    compute_step_login_validity_miss_metrics = None

# (user_id, note_id) 列表缓存，避免每轮重复读 CSV
_RC_PAIRS_CACHE: Optional[List[Tuple[str, str]]] = None
_RC_PAIRS_PATH: Optional[str] = None


def _default_recommendation_coverage_comments_csv_path() -> str:
    """与本 env 打包的 comments.csv（profile/data），供推荐覆盖类指标默认加载。"""
    return os.path.abspath(
        os.path.join(_metrics_dir, "..", "..", "profile", "data", "comments.csv")
    )

from collections import Counter, defaultdict
from loguru import logger
from onesim.monitor.utils import (
    safe_get, safe_number, safe_list, safe_sum, 
    safe_avg, safe_max, safe_min, safe_count, log_metric_error
)


def calculate_comment_generation(data: Dict[str, Any]) -> Any:
    """
    计算指标: comment_generation
    描述: 统计随时间变化的评论生成数量，用于追踪信息传播的活跃度
    可视化类型: line
    更新频率: 5 秒
    
    统计 content_pool 中所有 note 的 comments 总数，反映当前时间步的评论生成情况。
    对于折线图，返回单个数值，监控系统会自动记录时间序列。
    
    Args:
        data: 包含环境数据的字典，应该包含 content_pool 字段
        
    Returns:
        float: 当前时间步的评论总数
    """
    try:
        # 验证输入数据
        if not data or not isinstance(data, dict):
            log_metric_error("comment_generation", ValueError("无效的数据输入"), {"data": data})
            return 0
        
        # 获取 content_pool（字典格式，键为 note_id）
        content_pool = safe_get(data, "content_pool", {})
        
        if not isinstance(content_pool, dict):
            log_metric_error("comment_generation", ValueError("content_pool 不是字典格式"), {"content_pool_type": type(content_pool)})
            return 0
        
        # 统计所有 note 的评论总数
        total_comments = 0
        
        for note_id, note in content_pool.items():
            if not isinstance(note, dict):
                continue
            
            # 获取该 note 的 comments（字典格式，键为 comment_id）
            comments = note.get("comments", {})
            
            if isinstance(comments, dict):
                # 统计该 note 的评论数
                total_comments += len(comments)
            elif isinstance(comments, list):
                # 兼容列表格式（如果存在）
                total_comments += len(comments)
        
        return float(total_comments)
    
    except Exception as e:
        log_metric_error("comment_generation", e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return 0


def calculate_comment_top_vs_reply_over_time(data: Dict[str, Any]) -> Any:
    """
    每个监控采样时刻统计 content_pool 中评论条数，按是否有父评论区分：

    - top_level_comments：parent_comment_id 为空或缺失的顶层评论；
    - reply_comments：存在非空 parent_comment_id 的回复评论。

    返回双序列数值，由监控按时间步记录并绘制成两条折线（双子图）。
    """
    zero = {"top_level_comments": 0.0, "reply_comments": 0.0}
    try:
        if not data or not isinstance(data, dict):
            log_metric_error(
                "comment_top_vs_reply_over_time",
                ValueError("无效的数据输入"),
                {"data": data},
            )
            return zero

        content_pool = safe_get(data, "content_pool", {})
        if not isinstance(content_pool, dict):
            log_metric_error(
                "comment_top_vs_reply_over_time",
                ValueError("content_pool 不是字典"),
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


def _comment_timestamp_to_ms(value: Any) -> Optional[float]:
    """将评论里的 timestamp 统一为毫秒浮点；无法解析则 None。"""
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
            return _comment_timestamp_to_ms(float(s))
        except ValueError:
            return None
    return None


def _histogram_comment_times_ms(times_ms: List[float]) -> Tuple[List[float], List[int], str]:
    """按真实时间等宽分箱，返回 [edges_ms 长度 n+1], [counts 长度 n], 桶宽说明。"""
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
    按每条评论的 timestamp（真实时间）汇总：排序后的时间列表、等宽直方图分箱。
    监控导出时用 raw_data 画「累计评论量」阶梯曲线 + 「单位时间评论数」柱状图（横轴为真实时间）。
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
                ValueError("content_pool 不是字典"),
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
                ms = _comment_timestamp_to_ms(c.get("timestamp"))
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
    """返回 (comment_id_str, comment_dict) 列表。"""
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


def calculate_comment_count_frequency(data: Dict[str, Any]) -> Any:
    """
    两类频率分布（纵轴均为占总体百分比）：

    1) 按用户：统计每人全站在 content_pool 各帖下发表的评论总条数；横轴 k=0,1,2,…；
       纵轴 = 「恰好发了 k 条的用户数」/「用户全集 N」×100%。

       用户全集 N 的约定：
       - 若 data 提供非空 ``registered_user_agent_ids``（与 UserAgent.json / 仿真中全部 UserAgent id 一致，
         通常由 SimEnv 在 ``load_initial_data`` 写入），则 N = 该列表长度，**含尚未在池中发帖/评论的用户**
         （其评论数视为 0，落在 k=0 桶）。
       - 否则退化为旧逻辑：N = content_pool 中出现过的用户并集（帖子作者 user_id ∪ 评论 user_id）。

    2) 按帖子：每条 note 下评论条数；横轴 k=0,1,2,…；
       纵轴 = 「恰好有 k 条评论的帖子数」/「content_pool 中帖子总数」×100%。
    """
    empty: Dict[str, Any] = {
        "_viz_kind": "comment_count_freq_bar",
        "user_comment_bins": [],
        "user_frequency_pct": [],
        "user_raw_counts": [],
        "n_users_in_pool": 0,
        "user_count_basis": "content_pool_presence",
        "note_comment_bins": [],
        "note_frequency_pct": [],
        "note_raw_counts": [],
        "n_notes": 0,
    }
    try:
        if not data or not isinstance(data, dict):
            log_metric_error(
                "comment_count_frequency",
                ValueError("无效的数据输入"),
                {"data": data},
            )
            return empty

        content_pool = safe_get(data, "content_pool", {})
        if not isinstance(content_pool, dict):
            log_metric_error(
                "comment_count_frequency",
                ValueError("content_pool 不是字典"),
                {"content_pool_type": type(content_pool)},
            )
            return empty

        note_ids: List[str] = []
        per_note_ncomments: List[int] = []
        user_comment_totals: Dict[str, int] = defaultdict(int)
        pool_user_ids: Set[str] = set()

        for note_id, note in content_pool.items():
            if not isinstance(note, dict):
                continue
            nid = str(note.get("note_id", note_id))
            note_ids.append(nid)
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

        n_notes = len(note_ids)
        if n_notes == 0:
            return empty

        nh = Counter(per_note_ncomments)
        max_k_note = max(per_note_ncomments) if per_note_ncomments else 0
        note_bins = list(range(0, max_k_note + 1))
        note_raw = [nh[k] for k in note_bins]
        note_pct = [100.0 * c / n_notes for c in note_raw]

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
        if n_pool_users == 0:
            return {
                "_viz_kind": "comment_count_freq_bar",
                "user_comment_bins": [],
                "user_frequency_pct": [],
                "user_raw_counts": [],
                "n_users_in_pool": 0,
                "user_count_basis": "registered_user_agent_ids"
                if use_registered
                else "content_pool_presence",
                "note_comment_bins": note_bins,
                "note_frequency_pct": note_pct,
                "note_raw_counts": note_raw,
                "n_notes": n_notes,
            }

        per_user_counts = [user_comment_totals.get(u, 0) for u in sorted(universe)]
        uh = Counter(per_user_counts)
        max_k_user = max(per_user_counts) if per_user_counts else 0
        user_bins = list(range(0, max_k_user + 1))
        user_raw = [uh[k] for k in user_bins]
        user_pct = [100.0 * c / n_pool_users for c in user_raw]

        return {
            "_viz_kind": "comment_count_freq_bar",
            "user_comment_bins": user_bins,
            "user_frequency_pct": user_pct,
            "user_raw_counts": user_raw,
            "n_users_in_pool": n_pool_users,
            "user_count_basis": "registered_user_agent_ids"
            if use_registered
            else "content_pool_presence",
            "note_comment_bins": note_bins,
            "note_frequency_pct": note_pct,
            "note_raw_counts": note_raw,
            "n_notes": n_notes,
        }
    except Exception as e:
        log_metric_error(
            "comment_count_frequency",
            e,
            {"data_keys": list(data.keys()) if isinstance(data, dict) else None},
        )
        return empty


def calculate_posting_user_comment_behavior(data: Dict[str, Any]) -> Any:
    """
    概括每个「发过帖」的用户在当前 content_pool 下的评论行为（仅统计该用户自己发表的评论）：

    - 发帖数：该用户作为作者的 note 条数；
    - 本帖一跳评论数：在本人笔记下、parent_comment_id 为空的评论（直接挂在帖下）；
    - 本帖回复评论数：在本人笔记下、有 parent 的评论（回复某条评论）；
    - 他帖评论数：在别人笔记下发表的评论；
    - 发表评论总数：以上三项之和。

    返回 {"users": [ {...}, ... ]}，供监控按步写入 snapshots.jsonl。
    """
    empty: Dict[str, Any] = {"users": []}
    try:
        if not data or not isinstance(data, dict):
            log_metric_error(
                "posting_user_comment_behavior",
                ValueError("无效的数据输入"),
                {"data": data},
            )
            return empty

        content_pool = safe_get(data, "content_pool", {})
        if not isinstance(content_pool, dict):
            log_metric_error(
                "posting_user_comment_behavior",
                ValueError("content_pool 不是字典"),
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
                    "发帖数": notes_count.get(uid, 0),
                    "本帖一跳评论数": a,
                    "本帖回复评论数": b,
                    "他帖评论数": c,
                    "发表评论总数": a + b + c,
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


def _get_recommendation_coverage_pairs(data: Dict[str, Any]) -> List[Tuple[str, str]]:
    """从 comments.csv 读取 (user_id, note_id) 列表；路径可由 env_data 的 recommendation_coverage_comments_csv 指定。"""
    global _RC_PAIRS_CACHE, _RC_PAIRS_PATH
    if load_comment_user_note_pairs is None:
        return []
    raw = safe_get(data, "recommendation_coverage_comments_csv", None)
    if isinstance(raw, str) and raw.strip():
        path = os.path.abspath(os.path.expanduser(raw.strip()))
    else:
        # 默认优先本环境 profile/data/comments.csv；兼容旧仓库路径 datasets/openreview/comments.csv
        path_default = _default_recommendation_coverage_comments_csv_path()
        path_legacy = os.path.join(_REPO_ROOT, "datasets", "openreview", "comments.csv")
        if os.path.isfile(path_default):
            path = path_default
        elif os.path.isfile(path_legacy):
            path = path_legacy
        else:
            path = path_default
    if not os.path.isfile(path):
        return []
    if _RC_PAIRS_CACHE is None or _RC_PAIRS_PATH != path:
        _RC_PAIRS_CACHE = load_comment_user_note_pairs(path)
        _RC_PAIRS_PATH = path
    return list(_RC_PAIRS_CACHE or [])


def calculate_recommendation_coverage(data: Dict[str, Any]) -> Any:
    """
    与离线 plot_recommendation_coverage v1 一致：对 comments.csv 中的 (user, note) 对，
    在 current_notes 时间窗内且 last_login 晚于帖子 time 的样本上，统计 overall_hit 与各渠道 miss 比例。
    """
    empty: Dict[str, Any] = {
        "step_dir": "live",
        "step_num": None,
        "n_effective_pairs": 0,
        "overall_hit_sum": 0,
        "overall_hit_ratio": None,
        "miss_social_sum": 0,
        "miss_social_ratio": None,
        "miss_interest_sum": 0,
        "miss_interest_ratio": None,
        "miss_random_sum": 0,
        "miss_random_ratio": None,
        "miss_hot_sum": 0,
        "miss_hot_ratio": None,
    }
    if compute_step_recommendation_coverage is None:
        log_metric_error(
            "recommendation_coverage",
            ImportError("recommendation_coverage_metric 不可用"),
            {},
        )
        return empty
    try:
        if not data or not isinstance(data, dict):
            log_metric_error(
                "recommendation_coverage",
                ValueError("无效的数据输入"),
                {"data": data},
            )
            return empty
        pairs = _get_recommendation_coverage_pairs(data)
        content_pool = safe_get(data, "content_pool", {})
        if not isinstance(content_pool, dict):
            return empty
        current_notes = safe_get(data, "current_notes", {})
        if not isinstance(current_notes, dict):
            current_notes = {}
        users_snapshot = safe_get(data, "user_recommended_note_ids_by_channel", {})
        if not isinstance(users_snapshot, dict):
            users_snapshot = {}
        row = compute_step_recommendation_coverage(
            "live",
            pairs,
            content_pool,
            current_notes,
            users_snapshot,
        )
        cs = safe_get(data, "current_step", None)
        if cs is not None:
            try:
                row["step_num"] = int(cs)
            except (TypeError, ValueError):
                pass
        return row
    except Exception as e:
        log_metric_error(
            "recommendation_coverage",
            e,
            {"data_keys": list(data.keys()) if isinstance(data, dict) else None},
        )
        return empty


def calculate_recommendation_coverage_login_validity(data: Dict[str, Any]) -> Any:
    """
    与离线 --login-validity 单步含义一致：在有效帖与 last_login 对齐本轮 current_timestamp 的用户上，
    统计各渠道 miss 比例（递推运行均值仅离线脚本有；在线由监控按步记录曲线）。
    """
    empty: Dict[str, Any] = {
        "step_dir": "live",
        "step_num": None,
        "sim_current_timestamp": 0,
        "n_effective_login_validity": 0,
        "miss_any_step": None,
        "miss_social_step": None,
        "miss_interest_step": None,
        "miss_random_step": None,
        "miss_hot_step": None,
    }
    if compute_step_login_validity_miss_metrics is None:
        log_metric_error(
            "recommendation_coverage_login_validity",
            ImportError("recommendation_coverage_metric 不可用"),
            {},
        )
        return empty
    try:
        if not data or not isinstance(data, dict):
            log_metric_error(
                "recommendation_coverage_login_validity",
                ValueError("无效的数据输入"),
                {"data": data},
            )
            return empty
        pairs = _get_recommendation_coverage_pairs(data)
        content_pool = safe_get(data, "content_pool", {})
        if not isinstance(content_pool, dict):
            return empty
        users_snapshot = safe_get(data, "user_recommended_note_ids_by_channel", {})
        if not isinstance(users_snapshot, dict):
            users_snapshot = {}
        # 在线监控：env.current_timestamp 在存盘快照后会被推进，与 profile.last_login 不一致；
        # 优先使用 SimEnv 与快照一并写入的 recommendation_snapshot_login_timestamp。
        try:
            snap_ts = safe_get(data, "recommendation_snapshot_login_timestamp", None)
            if snap_ts is not None and int(snap_ts) > 0:
                cts = int(snap_ts)
            else:
                cts = int(safe_get(data, "current_timestamp", 0) or 0)
        except (TypeError, ValueError):
            cts = int(safe_get(data, "current_timestamp", 0) or 0)
        row = compute_step_login_validity_miss_metrics(
            "live",
            pairs,
            content_pool,
            users_snapshot,
            cts,
        )
        cs = safe_get(data, "current_step", None)
        if cs is not None:
            try:
                row["step_num"] = int(cs)
            except (TypeError, ValueError):
                pass
        return row
    except Exception as e:
        log_metric_error(
            "recommendation_coverage_login_validity",
            e,
            {"data_keys": list(data.keys()) if isinstance(data, dict) else None},
        )
        return empty


def calculate_comment_source_mix(data: Dict[str, Any]) -> Any:
    """
    统计当前 content_pool 中全部评论，按评论者用户的 recommended_note_ids_by_channel
    归因到：仅算法推荐流、仅关注流、双渠道、两流均未、无用户快照。

    需要 data 中同时包含：
    - content_pool
    - user_recommended_note_ids_by_channel（与 step 快照中结构一致：user_id -> 条目）

    可视化：离线用 plot_comment_source_mix.py 画饼图；在线可返回 JSON 结构供监控记录。
    """
    empty: Dict[str, Any] = {
        "total_comments": 0,
        "counts": {
            "both": 0,
            "algo_only": 0,
            "social_only": 0,
            "neither": 0,
            "unknown_user": 0,
        },
        "ratios": {
            "both": 0.0,
            "algo_only": 0.0,
            "social_only": 0.0,
            "neither": 0.0,
            "unknown_user": 0.0,
        },
    }
    if count_comment_source_mix is None:
        log_metric_error(
            "comment_source_mix",
            ImportError("comment_source_mix 未安装"),
            {},
        )
        return empty
    try:
        if not data or not isinstance(data, dict):
            log_metric_error(
                "comment_source_mix",
                ValueError("无效的数据输入"),
                {"data": data},
            )
            return empty
        users_snapshot = safe_get(data, "user_recommended_note_ids_by_channel", {})
        if not isinstance(users_snapshot, dict):
            log_metric_error(
                "comment_source_mix",
                ValueError("user_recommended_note_ids_by_channel 不是字典"),
                {"data_keys": list(data.keys()) if isinstance(data, dict) else None},
            )
            return empty
        content_pool = safe_get(data, "content_pool", {})
        if not isinstance(content_pool, dict):
            log_metric_error(
                "comment_source_mix",
                ValueError("content_pool 不是字典"),
                {"content_pool_type": type(content_pool)},
            )
            return empty
        return count_comment_source_mix(content_pool, users_snapshot)
    except Exception as e:
        log_metric_error(
            "comment_source_mix",
            e,
            {"data_keys": list(data.keys()) if isinstance(data, dict) else None},
        )
        return empty


# 指标函数字典，用于查找
METRIC_FUNCTIONS = {
    'calculate_comment_generation': calculate_comment_generation,
    'calculate_comment_top_vs_reply_over_time': calculate_comment_top_vs_reply_over_time,
    'calculate_comment_volume_realtime': calculate_comment_volume_realtime,
    'calculate_comment_count_frequency': calculate_comment_count_frequency,
    'calculate_posting_user_comment_behavior': calculate_posting_user_comment_behavior,
    'calculate_recommendation_coverage': calculate_recommendation_coverage,
    'calculate_recommendation_coverage_login_validity': calculate_recommendation_coverage_login_validity,
    'calculate_comment_source_mix': calculate_comment_source_mix,
}
if calculate_comment_similarity is not None:
    METRIC_FUNCTIONS['calculate_comment_similarity'] = calculate_comment_similarity


def get_metric_function(function_name: str) -> Optional[Callable]:
    """
    根据函数名获取对应的指标计算函数
    
    Args:
        function_name: 函数名
        
    Returns:
        指标计算函数或None
    """
    return METRIC_FUNCTIONS.get(function_name)
