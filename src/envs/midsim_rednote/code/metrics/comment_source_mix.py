# -*- coding: utf-8 -*-
"""
统计 content_pool 中全部评论：根据评论者用户在最终快照中的 recommended_note_ids_by_channel，
判断其评论的帖子 note_id 是否曾由「算法推荐流」或「关注流（social）」推送，用于饼图归因。

- 推荐流：interest / random / hot / keep_following_recommendation 任一包含该 note_id（与 UserAgent 渠道一致；keep_following 为系统推荐侧，非社交流）
- 关注流：social_recommendation 包含该 note_id
- 双渠道：同时满足上述两者
- 仅推荐流 / 仅关注流 / 双渠道 互斥；其余记为「均未通过两流推荐」（仍参与分母或单独一片）
"""
from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

from recommendation_coverage_metric import (
    CHANNEL_KEYS_ALGO,
    CHANNEL_KEY_SOCIAL,
    _parse_user_snapshot_entry,
    _note_id_in_channel,
)


def iter_all_comments(
    content_pool: Dict[str, Any],
) -> Iterator[Tuple[str, str, str]]:
    """
    遍历所有评论：yield (comment_id, note_id, commenter_user_id)。
    """
    for note_id, note in content_pool.items():
        if not isinstance(note, dict):
            continue
        nid = str(note_id).strip()
        comments = note.get("comments", {})
        if isinstance(comments, dict):
            iterable = comments.items()
        elif isinstance(comments, list):
            iterable = [(str(i), c) for i, c in enumerate(comments)]
        else:
            continue
        for cid, c in iterable:
            if not isinstance(c, dict):
                continue
            uid = c.get("user_id", c.get("userId"))
            if uid is None or uid == "":
                continue
            yield (str(cid).strip(), nid, str(uid).strip())


def note_in_algorithm_streams(note_id: str, by_ch: Dict[str, Any]) -> bool:
    for k in CHANNEL_KEYS_ALGO:
        if _note_id_in_channel(note_id, by_ch, k):
            return True
    # 与 CHANNEL_KEYS_ALGO 并列的系统推荐渠道，避免仅经「保持关注」曝光的帖子被误记为 neither
    if _note_id_in_channel(note_id, by_ch, "keep_following_recommendation"):
        return True
    return False


def note_in_social_stream(note_id: str, by_ch: Dict[str, Any]) -> bool:
    return _note_id_in_channel(note_id, by_ch, CHANNEL_KEY_SOCIAL)


def classify_comment_for_user(
    note_id: str,
    users_snapshot: Dict[str, Any],
    commenter_id: str,
) -> str:
    """
    返回: "both" | "algo_only" | "social_only" | "neither" | "unknown_user"
    """
    uentry = users_snapshot.get(commenter_id)
    if uentry is None:
        uentry = users_snapshot.get(str(commenter_id))
    if uentry is None:
        return "unknown_user"
    _, by_ch = _parse_user_snapshot_entry(uentry)
    in_a = note_in_algorithm_streams(note_id, by_ch)
    in_s = note_in_social_stream(note_id, by_ch)
    if in_a and in_s:
        return "both"
    if in_a:
        return "algo_only"
    if in_s:
        return "social_only"
    return "neither"


def count_comment_source_mix(
    content_pool: Dict[str, Any],
    users_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    counts = {
        "both": 0,
        "algo_only": 0,
        "social_only": 0,
        "neither": 0,
        "unknown_user": 0,
    }
    for _cid, nid, uid in iter_all_comments(content_pool):
        cat = classify_comment_for_user(nid, users_snapshot, uid)
        counts[cat] = counts.get(cat, 0) + 1
    total = sum(counts.values())
    out = {
        "total_comments": total,
        "counts": dict(counts),
        "ratios": {k: (counts[k] / total if total else 0.0) for k in counts},
    }
    return out


def find_last_step_dir(datasets_root: str) -> Optional[str]:
    """选择 step 编号最大的目录。"""
    pattern = os.path.join(datasets_root, "step_*")
    candidates = glob.glob(pattern)
    if not candidates:
        return None

    def key_fn(p: str) -> Tuple[int, str]:
        base = os.path.basename(p)
        if base.startswith("step_"):
            try:
                return (int(base.replace("step_", "")), base)
            except ValueError:
                pass
        return (-1, base)

    return max(candidates, key=key_fn)


def load_final_snapshots_from_datasets_root(
    datasets_root: str,
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], str]]:
    """加载最后一轮 content_pool 与 user 快照。返回 (content_pool, users, step_dir)。"""
    last_sd = find_last_step_dir(datasets_root)
    if not last_sd:
        return None
    cp_path = os.path.join(last_sd, "content_pool_snapshot.json")
    us_path = os.path.join(last_sd, "user_recommended_note_ids_by_channel.json")
    if not os.path.isfile(cp_path) or not os.path.isfile(us_path):
        return None
    with open(cp_path, "r", encoding="utf-8") as f:
        cp = json.load(f)
    if not isinstance(cp, dict):
        cp = {}
    with open(us_path, "r", encoding="utf-8") as f:
        us = json.load(f)
    if not isinstance(us, dict):
        us = {}
    return cp, us, last_sd


def save_counts_json(summary: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
