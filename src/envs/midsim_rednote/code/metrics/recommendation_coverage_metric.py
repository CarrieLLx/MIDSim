# -*- coding: utf-8 -*-
"""
基于真实评论关系（comments.csv）与每轮快照，计算推荐覆盖相关指标。

**v1** 有效样本 (user_id, note_id)：该帖在当轮 current_notes 中，且用户 last_login_timestamp 晚于帖子 time。
- overall_hit：note_id 出现在该用户 recommended_note_ids_by_channel 任一类列表中 → 1，否则 0。
- channel_miss_*：note_id 未出现在该渠道列表中 → 1，否则 0（分别对 social / interest / random / hot）。

**v2（login_validity）** 有效帖子：发帖 time <= step_metadata.current_timestamp；
有效用户：快照 last_login_timestamp 与本轮 current_timestamp 一致（与 UserAgent 更新 last_login 对齐）；
- miss_any：未出现在任意渠道 → 1；各渠道未出现在该渠道列表 → 1。
- 对每轮 step 比例做递推运行均值：R_n = R_{n-1}*(n-1)/n + x_n/n。

每轮输出各均值（除以有效样本数 N）及 N。
"""
from __future__ import annotations

import csv
import glob
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# 与 UserAgent 中渠道键一致
CHANNEL_KEYS_ALL = (
    "interest_recommendation",
    "hot_recommendation",
    "random_recommendation",
    "social_recommendation",
    "keep_following_recommendation",
)
CHANNEL_KEYS_ALGO = ("interest_recommendation", "random_recommendation", "hot_recommendation")
CHANNEL_KEY_SOCIAL = "social_recommendation"


def load_comment_user_note_pairs(
    comments_csv_path: str,
) -> List[Tuple[str, str]]:
    """从 comments.csv 读取去重后的 (user_id, note_id)。"""
    pairs: Set[Tuple[str, str]] = set()
    with open(comments_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = str(row.get("user_id", "") or "").strip()
            nid = str(row.get("note_id", "") or "").strip()
            if uid and nid:
                pairs.add((uid, nid))
    return sorted(pairs)


def _parse_user_snapshot_entry(val: Any) -> Tuple[int, Dict[str, Any]]:
    """兼容旧版：value 直接为 channel dict；新版：{last_login_timestamp, recommended_note_ids_by_channel}。"""
    if not isinstance(val, dict):
        return 0, {}
    if "recommended_note_ids_by_channel" in val:
        ll = val.get("last_login_timestamp", 0)
        try:
            ll_i = int(ll) if ll is not None else 0
        except (TypeError, ValueError):
            ll_i = 0
        raw = val.get("recommended_note_ids_by_channel", {})
        return ll_i, raw if isinstance(raw, dict) else {}
    return 0, val


def _note_time_ms(note: Dict[str, Any]) -> int:
    t = note.get("time", note.get("create_time"))
    try:
        return int(t) if t is not None else 0
    except (TypeError, ValueError):
        return 0


def _note_id_in_any_channel(note_id: str, by_ch: Dict[str, Any]) -> bool:
    sid = str(note_id).strip()
    for k in CHANNEL_KEYS_ALL:
        lst = by_ch.get(k, [])
        if not isinstance(lst, list):
            continue
        for x in lst:
            if str(x).strip() == sid:
                return True
    return False


def _note_id_in_channel(note_id: str, by_ch: Dict[str, Any], channel_key: str) -> bool:
    sid = str(note_id).strip()
    lst = by_ch.get(channel_key, [])
    if not isinstance(lst, list):
        return False
    return any(str(x).strip() == sid for x in lst)


def compute_step_recommendation_coverage(
    step_dir: str,
    pairs: Sequence[Tuple[str, str]],
    content_pool: Dict[str, Any],
    current_notes: Dict[str, Any],
    users_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    """
    对单轮目录已解析的 content_pool / current_notes / users 快照计算指标。
    """
    cn_keys: Set[str] = {str(k).strip() for k in current_notes.keys() if k}

    n_eff = 0
    s_any = 0
    s_miss_social = 0
    s_miss_interest = 0
    s_miss_random = 0
    s_miss_hot = 0

    for uid, note_id in pairs:
        note = content_pool.get(note_id)
        if not isinstance(note, dict):
            continue
        if note_id not in cn_keys:
            continue
        post_t = _note_time_ms(note)
        uentry = users_snapshot.get(uid)
        if uentry is None:
            uentry = users_snapshot.get(str(uid))
        ll, by_ch = _parse_user_snapshot_entry(uentry)
        if ll <= post_t:
            continue

        n_eff += 1
        if _note_id_in_any_channel(note_id, by_ch):
            s_any += 1
        if not _note_id_in_channel(note_id, by_ch, CHANNEL_KEY_SOCIAL):
            s_miss_social += 1
        if not _note_id_in_channel(note_id, by_ch, "interest_recommendation"):
            s_miss_interest += 1
        if not _note_id_in_channel(note_id, by_ch, "random_recommendation"):
            s_miss_random += 1
        if not _note_id_in_channel(note_id, by_ch, "hot_recommendation"):
            s_miss_hot += 1

    def ratio(num: int) -> Optional[float]:
        if n_eff <= 0:
            return None
        return float(num) / float(n_eff)

    step_num = None
    try:
        base = os.path.basename(os.path.normpath(step_dir))
        if base.startswith("step_"):
            step_num = int(base.replace("step_", ""))
    except ValueError:
        step_num = None

    return {
        "step_dir": step_dir,
        "step_num": step_num,
        "n_effective_pairs": n_eff,
        "overall_hit_sum": s_any,
        "overall_hit_ratio": ratio(s_any),
        "miss_social_sum": s_miss_social,
        "miss_social_ratio": ratio(s_miss_social),
        "miss_interest_sum": s_miss_interest,
        "miss_interest_ratio": ratio(s_miss_interest),
        "miss_random_sum": s_miss_random,
        "miss_random_ratio": ratio(s_miss_random),
        "miss_hot_sum": s_miss_hot,
        "miss_hot_ratio": ratio(s_miss_hot),
    }


def load_step_snapshots(step_dir: str) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
    """返回 (content_pool, current_notes, users_json) 或缺失时 None。"""
    cp_path = os.path.join(step_dir, "content_pool_snapshot.json")
    cn_path = os.path.join(step_dir, "current_notes_snapshot.json")
    us_path = os.path.join(step_dir, "user_recommended_note_ids_by_channel.json")
    if not os.path.isfile(cp_path) or not os.path.isfile(us_path):
        return None
    with open(cp_path, "r", encoding="utf-8") as f:
        content_pool = json.load(f)
    if not isinstance(content_pool, dict):
        content_pool = {}
    if os.path.isfile(cn_path):
        with open(cn_path, "r", encoding="utf-8") as f:
            current_notes = json.load(f)
        if not isinstance(current_notes, dict):
            current_notes = {}
    else:
        # 无快照时无法判断「当前窗」可见帖，置空（勿用全量 content_pool 代替）
        current_notes = {}
    with open(us_path, "r", encoding="utf-8") as f:
        users_snapshot = json.load(f)
    if not isinstance(users_snapshot, dict):
        users_snapshot = {}
    return content_pool, current_notes, users_snapshot


def run_recommendation_coverage_over_steps(
    datasets_root: str,
    comments_csv_path: str,
) -> List[Dict[str, Any]]:
    """
    datasets_root: 含 step_1, step_2, ... 子目录（与 SimEnv 输出 datasets 路径一致）。
    """
    pairs = load_comment_user_note_pairs(comments_csv_path)
    step_dirs = sorted(
        glob.glob(os.path.join(datasets_root, "step_*")),
        key=lambda p: _step_sort_key(p),
    )
    rows: List[Dict[str, Any]] = []
    for sd in step_dirs:
        loaded = load_step_snapshots(sd)
        if loaded is None:
            continue
        cp, cn, us = loaded
        rows.append(compute_step_recommendation_coverage(sd, pairs, cp, cn, us))
    return rows


def _step_sort_key(path: str) -> Tuple[int, str]:
    base = os.path.basename(path)
    if base.startswith("step_"):
        try:
            return (int(base.replace("step_", "")), base)
        except ValueError:
            pass
    return (10**9, base)


def rows_to_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def load_step_metadata(step_dir: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(step_dir, "step_metadata.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw if isinstance(raw, dict) else None


def running_mean_ratios(values: List[Optional[float]]) -> List[Optional[float]]:
    """
    递推均值：R_n = R_{n-1} * (n-1)/n + x_n/n，与「前 n-1 轮均值 * (n-1)/n + 本轮 * 1/n」一致。
    """
    out: List[Optional[float]] = []
    acc: Optional[float] = None
    n = 0
    for x in values:
        if x is None:
            out.append(None)
            continue
        n += 1
        fv = float(x)
        if n == 1:
            acc = fv
        else:
            acc = (acc or 0.0) * (n - 1) / n + fv / n
        out.append(acc)
    return out


def compute_step_login_validity_miss_metrics(
    step_dir: str,
    pairs: Sequence[Tuple[str, str]],
    content_pool: Dict[str, Any],
    users_snapshot: Dict[str, Any],
    current_ts: int,
) -> Dict[str, Any]:
    """
    有效帖子：发帖时间 time <= 本轮 step_metadata.current_timestamp（帖子在仿真当前时刻已存在）。
    有效用户：快照中 last_login_timestamp 与本轮 current_timestamp 一致（与 UserAgent 更新 last_login 对齐）。
    指标：若 note_id 未出现在任意 recommended_note_ids_by_channel 列表中则计 1，否则 0；
    各渠道：未出现在该渠道列表则计 1，否则 0。再对有效样本取均值得到本轮 step 比例。
    """
    T = int(current_ts)
    n_eff = 0
    s_miss_any = 0
    s_ms = 0
    s_mi = 0
    s_mr = 0
    s_mh = 0

    for uid, note_id in pairs:
        note = content_pool.get(note_id)
        if not isinstance(note, dict):
            continue
        post_t = _note_time_ms(note)
        # 帖子已发布且不晚于本轮仿真时刻（若需「严格早于」可改为 <）
        if post_t <= 0 or post_t > T:
            continue

        uentry = users_snapshot.get(uid)
        if uentry is None:
            uentry = users_snapshot.get(str(uid))
        ll, by_ch = _parse_user_snapshot_entry(uentry)
        try:
            ll_i = int(ll)
        except (TypeError, ValueError):
            ll_i = 0
        # 本轮已执行 StartEvent 中的 last_login 更新（与 metadata 时刻一致）
        if ll_i != T:
            continue

        n_eff += 1
        in_any = _note_id_in_any_channel(note_id, by_ch)
        if not in_any:
            s_miss_any += 1
        if not _note_id_in_channel(note_id, by_ch, CHANNEL_KEY_SOCIAL):
            s_ms += 1
        if not _note_id_in_channel(note_id, by_ch, "interest_recommendation"):
            s_mi += 1
        if not _note_id_in_channel(note_id, by_ch, "random_recommendation"):
            s_mr += 1
        if not _note_id_in_channel(note_id, by_ch, "hot_recommendation"):
            s_mh += 1

    def ratio(num: int) -> Optional[float]:
        if n_eff <= 0:
            return None
        return float(num) / float(n_eff)

    step_num = None
    try:
        base = os.path.basename(os.path.normpath(step_dir))
        if base.startswith("step_"):
            step_num = int(base.replace("step_", ""))
    except ValueError:
        step_num = None

    return {
        "step_dir": step_dir,
        "step_num": step_num,
        "sim_current_timestamp": T,
        "n_effective_login_validity": n_eff,
        "miss_any_step": ratio(s_miss_any),
        "miss_social_step": ratio(s_ms),
        "miss_interest_step": ratio(s_mi),
        "miss_random_step": ratio(s_mr),
        "miss_hot_step": ratio(s_mh),
    }


def run_login_validity_miss_over_steps(
    datasets_root: str,
    comments_csv_path: str,
) -> List[Dict[str, Any]]:
    """
    基于 step_metadata + 用户快照：计算每轮 step 比例，并对各比例列做递推运行均值。
    """
    pairs = load_comment_user_note_pairs(comments_csv_path)
    step_dirs = sorted(
        glob.glob(os.path.join(datasets_root, "step_*")),
        key=lambda p: _step_sort_key(p),
    )
    rows: List[Dict[str, Any]] = []
    for sd in step_dirs:
        loaded = load_step_snapshots(sd)
        if loaded is None:
            continue
        cp, _cn, us = loaded
        meta = load_step_metadata(sd)
        if not meta or "current_timestamp" not in meta:
            continue
        try:
            cts = int(meta.get("current_timestamp", 0) or 0)
        except (TypeError, ValueError):
            continue
        row = compute_step_login_validity_miss_metrics(sd, pairs, cp, us, cts)
        rows.append(row)

    keys_rm = (
        "miss_any_step",
        "miss_social_step",
        "miss_interest_step",
        "miss_random_step",
        "miss_hot_step",
    )
    series = {k: [r.get(k) for r in rows] for k in keys_rm}
    run_avgs: Dict[str, List[Optional[float]]] = {}
    for k in keys_rm:
        out_k = k.replace("_step", "_run_avg")
        run_avgs[out_k] = running_mean_ratios(series[k])

    for i, r in enumerate(rows):
        for out_k, seq in run_avgs.items():
            r[out_k] = seq[i] if i < len(seq) else None

    return rows
