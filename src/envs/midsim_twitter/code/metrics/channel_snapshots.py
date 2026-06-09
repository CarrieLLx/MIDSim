# -*- coding: utf-8 -*-
"""Simulation step snapshots: channel records + content_pool disk export."""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Set

from loguru import logger


def _step_key(current_step: Any) -> str:
    try:
        return str(int(current_step))
    except (TypeError, ValueError):
        return str(current_step)


def _parse_last_login(raw: Any) -> int:
    try:
        return int(raw) if raw is not None else 0
    except (TypeError, ValueError):
        return 0


def _merge_tweet_ids(prev_ids: Any, new_ids: List[str]) -> List[str]:
    merged: List[str] = list(prev_ids) if isinstance(prev_ids, list) else []
    seen: Set[str] = {str(x).strip() for x in merged if str(x).strip()}
    for sid in new_ids:
        s = str(sid).strip()
        if not s or s in seen:
            continue
        merged.append(s)
        seen.add(s)
    return merged


def record_recommendations_by_source_step(
    profile: Any,
    profile_id: Any,
    source_type: str,
    current_step: int,
    recommendations: Dict[str, Any],
    event_timestamp: Any,
) -> None:
    """Append tweet ids to profile.recommended_tweet_ids_by_channel for this source + step."""
    if profile is None:
        return
    st = (source_type or "").strip()
    if not st or not recommendations:
        return

    step_key = _step_key(current_step)
    raw_root = profile.get_data("recommended_tweet_ids_by_channel", {})
    by_ch: Dict[str, Any] = dict(raw_root) if isinstance(raw_root, dict) else {}

    step_map_raw = by_ch.get(st)
    step_map: Dict[str, Any] = dict(step_map_raw) if isinstance(step_map_raw, dict) else {}

    merged = _merge_tweet_ids(step_map.get(step_key), list(recommendations.keys()))
    step_map[step_key] = merged
    by_ch[st] = step_map
    profile.update_data("recommended_tweet_ids_by_channel", by_ch)

    ts_int = 0
    if event_timestamp is not None:
        try:
            ts_int = int(event_timestamp)
        except (TypeError, ValueError):
            ts_int = 0
    if ts_int > 0:
        profile.update_data("last_login_timestamp", ts_int)

    ll_out = _parse_last_login(profile.get_data("last_login_timestamp", 0))
    logger.info(
        f"UserAgent {profile_id} recommended_by_channel: "
        f"last_login_timestamp={ll_out} "
        f"recommended_tweet_ids_by_channel={json.dumps(by_ch, ensure_ascii=False, default=str)}"
    )


def record_mentioned_tweet_ids_by_channel(
    profile: Any,
    profile_id: Any,
    current_step: int,
    mention_entries: List[Dict[str, Any]],
    event_timestamp: Any,
) -> None:
    """Append tweet ids to profile.mentioned_tweet_ids_by_channel for this mention type + step."""
    if profile is None or not mention_entries:
        return

    step_key = _step_key(current_step)
    batch: Dict[str, List[str]] = defaultdict(list)
    for entry in mention_entries:
        if not isinstance(entry, dict):
            continue
        mt = str(entry.get("mention_type") or "at").strip() or "at"
        nid = entry.get("tweet_id")
        sid = str(nid).strip() if nid is not None else ""
        if sid:
            batch[mt].append(sid)

    if not batch:
        return

    raw_root = profile.get_data("mentioned_tweet_ids_by_channel", {})
    by_ch: Dict[str, Any] = dict(raw_root) if isinstance(raw_root, dict) else {}

    for mt, ids in batch.items():
        prev = by_ch.get(mt)
        step_map: Dict[str, Any] = dict(prev) if isinstance(prev, dict) else {}
        step_map[step_key] = _merge_tweet_ids(step_map.get(step_key), ids)
        by_ch[mt] = step_map

    profile.update_data("mentioned_tweet_ids_by_channel", by_ch)

    ts_int = 0
    if event_timestamp is not None:
        try:
            ts_int = int(event_timestamp)
        except (TypeError, ValueError):
            ts_int = 0
    if ts_int > 0:
        profile.update_data("last_login_timestamp", ts_int)

    ll_out = _parse_last_login(profile.get_data("last_login_timestamp", 0))
    try:
        step_out = int(current_step)
    except (TypeError, ValueError):
        step_out = current_step

    logger.info(
        f"UserAgent {profile_id} mentioned_by_channel: "
        f"step={step_out} last_login_timestamp={ll_out} "
        f"mentioned_tweet_ids_by_channel={json.dumps(by_ch, ensure_ascii=False, default=str)}"
    )


def _snapshot_base_dir(env: Any) -> str:
    base_dir = getattr(env, "output_dir", None) or env.data.get("output_dir")
    if not isinstance(base_dir, str) or not base_dir.strip():
        base_dir = os.path.join(os.getcwd(), "runs_content_pool_snapshots")
    return base_dir


def _collect_user_profiles(env: Any) -> Dict[str, Any]:
    combined: Dict[str, Any] = {}
    agents_map = getattr(env, "agents", None) or {}
    user_map = agents_map.get("UserAgent", {}) if isinstance(agents_map, dict) else {}
    if not isinstance(user_map, dict):
        return combined
    for aid, agent in user_map.items():
        uid = str(aid).strip() if aid is not None else ""
        prof = getattr(agent, "profile", None)
        if prof is not None and not uid:
            uid = str(prof.get_data("id", "") or "").strip()
        if not uid:
            continue
        combined[uid] = prof
    return combined


def profile_recommended_by_channel_for_snapshot(prof: Any) -> Dict[str, Any]:
    """
    Twitter UserAgent writes recommended_tweet_ids_by_channel;
    fall back to recommended_note_ids_by_channel for cross-env compatibility.
    """
    if prof is None:
        return {}
    tw = prof.get_data("recommended_tweet_ids_by_channel", None)
    nt = prof.get_data("recommended_note_ids_by_channel", None)
    if isinstance(tw, dict) and tw:
        return dict(tw)
    if isinstance(nt, dict) and nt:
        return dict(nt)
    if isinstance(tw, dict):
        return dict(tw)
    if isinstance(nt, dict):
        return dict(nt)
    return {}


def profile_mentioned_by_channel_for_snapshot(prof: Any) -> Dict[str, Any]:
    """
    Twitter UserAgent writes mentioned_tweet_ids_by_channel;
    fall back to mentioned_note_ids_by_channel for cross-env compatibility.
    """
    if prof is None:
        return {}
    tw = prof.get_data("mentioned_tweet_ids_by_channel", None)
    nt = prof.get_data("mentioned_note_ids_by_channel", None)
    if isinstance(tw, dict) and tw:
        return dict(tw)
    if isinstance(nt, dict) and nt:
        return dict(nt)
    if isinstance(tw, dict):
        return dict(tw)
    if isinstance(nt, dict):
        return dict(nt)
    return {}


def save_recommended_note_ids_by_channel_snapshot(env: Any, step_num: int) -> None:
    """Write datasets/step_N/user_recommended_note_ids_by_channel.json."""
    step_dir = os.path.join(_snapshot_base_dir(env), "datasets", f"step_{step_num}")
    os.makedirs(step_dir, exist_ok=True)
    snapshot_path = os.path.join(step_dir, "user_recommended_note_ids_by_channel.json")

    combined: Dict[str, Any] = {}
    for uid, prof in _collect_user_profiles(env).items():
        if prof is None:
            combined[uid] = {"last_login_timestamp": 0, "recommended_note_ids_by_channel": {}}
            continue
        raw = profile_recommended_by_channel_for_snapshot(prof)
        combined[uid] = {
            "last_login_timestamp": _parse_last_login(prof.get_data("last_login_timestamp", 0)),
            "recommended_note_ids_by_channel": raw if isinstance(raw, dict) else {},
        }

    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2, default=str)

    env.data["user_recommended_note_ids_by_channel"] = combined
    try:
        env.data["recommendation_snapshot_login_timestamp"] = int(
            env.data.get("current_timestamp", 0) or 0
        )
    except (TypeError, ValueError):
        env.data["recommendation_snapshot_login_timestamp"] = 0
    logger.info(
        f"Step {step_num}: Saved user recommended_note_ids_by_channel snapshot to {snapshot_path} "
        f"({len(combined)} user(s))"
    )


def save_mentioned_note_ids_by_channel_snapshot(env: Any, step_num: int) -> None:
    """Write datasets/step_N/user_mentioned_note_ids_by_channel.json."""
    step_dir = os.path.join(_snapshot_base_dir(env), "datasets", f"step_{step_num}")
    os.makedirs(step_dir, exist_ok=True)
    snapshot_path = os.path.join(step_dir, "user_mentioned_note_ids_by_channel.json")

    combined: Dict[str, Any] = {}
    for uid, prof in _collect_user_profiles(env).items():
        if prof is None:
            combined[uid] = {"last_login_timestamp": 0, "mentioned_note_ids_by_channel": {}}
            continue
        raw = profile_mentioned_by_channel_for_snapshot(prof)
        combined[uid] = {
            "last_login_timestamp": _parse_last_login(prof.get_data("last_login_timestamp", 0)),
            "mentioned_note_ids_by_channel": raw if isinstance(raw, dict) else {},
        }

    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2, default=str)
    env.data["user_mentioned_note_ids_by_channel"] = combined
    logger.info(
        f"Step {step_num}: Saved user mentioned_note_ids_by_channel snapshot to {snapshot_path} "
        f"({len(combined)} user(s))"
    )


def save_content_pool_snapshot(env: Any, step_num: int, content_pool: Dict[str, Any]) -> None:
    """Write datasets/step_N/content_pool_snapshot.json for offline debugging and playback."""
    if not isinstance(content_pool, dict):
        return

    step_dir = os.path.join(_snapshot_base_dir(env), "datasets", f"step_{step_num}")
    os.makedirs(step_dir, exist_ok=True)
    snapshot_path = os.path.join(step_dir, "content_pool_snapshot.json")

    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(content_pool, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"Step {step_num}: Saved content_pool snapshot to {snapshot_path}")


def save_channel_snapshots(env: Any, step_num: int) -> None:
    """Persist recommendation + mention channel snapshots for all UserAgents."""
    save_recommended_note_ids_by_channel_snapshot(env, step_num)
    save_mentioned_note_ids_by_channel_snapshot(env, step_num)
