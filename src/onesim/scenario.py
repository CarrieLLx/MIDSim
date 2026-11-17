"""Topic bundle helpers for multi-group env/profile/env_data runs."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional

from loguru import logger

from onesim.env_layout import (
    get_algorithm_index_path,
    get_algorithm_profiles_path,
    get_platform_profile_path,
    get_platform_recommendation_path,
    get_recommendation_path,
    get_relationship_path,
    get_topic_meta_path,
    get_topics_index_path,
    get_topics_root,
    get_user_agent_profile_path,
)

COARSE_POOL_KEY = "coarse_interest_recommendation_content_pool"
FINE_POOL_KEY = "fine_interest_recommendation_content_pool"
MERGED_POOL_KEY = "interest_recommendation_content_pool"
CANDIDATE_BY_USER_KEY = "candidate_post_ids_by_user"


def load_algorithm_index(env_path: str) -> Dict[str, Any]:
    path = get_algorithm_index_path(env_path)
    if not os.path.isfile(path):
        legacy = get_algorithm_profiles_path(env_path)
        if legacy != path and os.path.isfile(legacy):
            with open(legacy, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, list):
                return {"profiles": raw}
            return raw if isinstance(raw, dict) else {}
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {"profiles": data}
    return data if isinstance(data, dict) else {}


def load_topics_index(env_path: str) -> Dict[str, Any]:
    index_path = get_topics_index_path(env_path)
    if not os.path.isfile(index_path):
        return {}
    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def list_scenarios(env_path: str) -> List[str]:
    """Return topic slugs from topics/index.json or topic subdirectories."""
    index = load_topics_index(env_path)
    slugs: List[str] = []
    for key in ("topics", "scenarios"):
        for item in index.get(key, []):
            if isinstance(item, str):
                slugs.append(item)
            elif isinstance(item, dict) and item.get("slug"):
                slugs.append(str(item["slug"]))
    if slugs:
        return list(dict.fromkeys(slugs))

    topics_root = get_topics_root(env_path)
    if not os.path.isdir(topics_root):
        return []

    for name in sorted(os.listdir(topics_root)):
        topic_dir = os.path.join(topics_root, name)
        if not os.path.isdir(topic_dir):
            continue
        if os.path.isfile(get_topic_meta_path(topic_dir)):
            slugs.append(name)
    return slugs


def get_scenario_dir(env_path: str, scenario_slug: Optional[str]) -> Optional[str]:
    if not env_path or not scenario_slug:
        return None
    return os.path.join(get_topics_root(env_path), scenario_slug)


def load_scenario_meta(scenario_dir: str) -> Dict[str, Any]:
    meta_path = get_topic_meta_path(scenario_dir)
    if not os.path.isfile(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def resolve_shared_algorithm_path(env_path: str, _index: Optional[Dict[str, Any]] = None) -> str:
    algo_index = load_algorithm_index(env_path)
    rel_path = algo_index.get("algorithm_path")
    return get_algorithm_profiles_path(env_path, rel_path)


def resolve_scenario_paths(env_path: str, scenario_slug: Optional[str]) -> Optional[Dict[str, str]]:
    scenario_dir = get_scenario_dir(env_path, scenario_slug)
    if not scenario_dir or not os.path.isdir(scenario_dir):
        return None
    index = load_topics_index(env_path)
    return {
        "scenario_dir": scenario_dir,
        "scenario_json": get_topic_meta_path(scenario_dir),
        "env_data_path": os.path.join(scenario_dir, "env_data.json"),
        "user_agent_path": get_user_agent_profile_path(env_path, scenario_slug),
        "algorithm_path": resolve_shared_algorithm_path(env_path, index),
        "relationship_path": get_relationship_path(env_path, scenario_slug),
        "recommendation_path": get_recommendation_path(env_path, scenario_slug),
        "platform_recommendation_path": get_platform_recommendation_path(env_path),
    }


def _read_json_dict(path: Optional[str]) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _as_post_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def load_recommendation_overlay(
    topic_path: Optional[str],
    platform_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load platform coarse pool + topic fine pool; merge for Algorithm.interest()."""
    platform_data = _read_json_dict(platform_path)
    topic_data = _read_json_dict(topic_path)

    coarse = _as_post_dict(
        platform_data.get(COARSE_POOL_KEY) or platform_data.get(MERGED_POOL_KEY)
    )
    fine = _as_post_dict(
        topic_data.get(FINE_POOL_KEY) or topic_data.get(MERGED_POOL_KEY)
    )
    by_user = _as_post_dict(topic_data.get(CANDIDATE_BY_USER_KEY))

    merged = dict(coarse)
    merged.update(fine)
    legacy_candidates = {key: value for data in (platform_data, topic_data) for key, value in data.items()
                         if key in ("candidate_note_ids_by_user", "candidate_tweet_ids_by_user", "candidate_blog_ids_by_user")}
    return {
        **legacy_candidates,
        COARSE_POOL_KEY: coarse,
        FINE_POOL_KEY: fine,
        MERGED_POOL_KEY: merged,
        CANDIDATE_BY_USER_KEY: by_user,
    }


def apply_recommendation_overlay_to_profiles(profiles: List[Any], overlay: Dict[str, Any]) -> None:
    """Merge topic rec data onto the recommendation Algorithm profile."""
    if not overlay or not profiles:
        return
    target = None
    for profile in profiles:
        role = str(profile.get_data("role", "") or "").strip().lower()
        if role == "recommendation":
            target = profile
            break
    if target is None:
        target = next((p for p in profiles if p.get_data("type", "") == "Interest Recommendation"), profiles[0])
    for key, value in overlay.items():
        target.update_data(key, value)
    logger.info(
        "Merged recommendation overlay: coarse={} fine={} merged={} candidates={}",
        len(overlay.get(COARSE_POOL_KEY) or {}),
        len(overlay.get(FINE_POOL_KEY) or {}),
        len(overlay.get(MERGED_POOL_KEY) or {}),
        len(overlay.get(CANDIDATE_BY_USER_KEY) or {}),
    )


def merge_scenario_portrait(env_path: str, scenario_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Merge algorithm index portrait with per-topic portrait (UserAgent)."""
    algo_index = load_algorithm_index(env_path)
    portrait: Dict[str, Any] = dict(algo_index.get("portrait") or {})
    if "Algorithm" not in portrait:
        algo_count = 0
        if isinstance(algo_index.get("recommendation_algorithm"), dict):
            algo_count += 1
        if isinstance(algo_index.get("search_algorithm"), dict):
            algo_count += 1
        if algo_count:
            portrait["Algorithm"] = algo_count
    scenario_portrait = scenario_meta.get("portrait") or {}
    if isinstance(scenario_portrait, dict):
        portrait.update(scenario_portrait)
    return portrait


def apply_scenario_to_config(config: Any, scenario_slug: str) -> None:
    """Inject topic paths and portrait counts into a OneSimConfig instance."""
    paths = resolve_scenario_paths(config.env_path, scenario_slug)
    if not paths:
        raise FileNotFoundError(
            f"Topic '{scenario_slug}' not found under {get_topics_root(config.env_path)}"
        )

    env_cfg = config.simulator_config.environment
    additional = dict(env_cfg.get("additional_config") or {})
    additional["scenario"] = scenario_slug
    additional["env_data_path"] = paths["env_data_path"]
    additional["recommendation_path"] = paths["recommendation_path"]
    additional["platform_recommendation_path"] = paths["platform_recommendation_path"]
    env_cfg["additional_config"] = additional

    meta = load_scenario_meta(paths["scenario_dir"])
    portrait = merge_scenario_portrait(config.env_path, meta)

    profile_cfg = dict(config.agent_config.profile or {})
    # Inline service profiles live in the startup config and need topic overlays
    # even when there is no platform/Algorithm.json to supply a portrait.
    for service in ("Algorithm", "Platform"):
        if service in profile_cfg:
            portrait.setdefault(service, profile_cfg[service].get("count", 1))
    for agent_type, count in portrait.items():
        entry = dict(profile_cfg.get(agent_type) or {})
        entry["count"] = count
        if agent_type == "UserAgent":
            entry["profile_path"] = paths["user_agent_path"]
        elif agent_type == "Algorithm":
            if "profiles" in entry:
                entry.pop("profile_path", None)
            else:
                entry["profile_path"] = paths["algorithm_path"]
            entry["recommendation_overlay"] = paths["recommendation_path"]
            entry["platform_recommendation_overlay"] = paths["platform_recommendation_path"]
        elif agent_type == "Platform":
            if "profiles" in entry:
                entry.pop("profile_path", None)
            else:
                entry["profile_path"] = get_platform_profile_path(config.env_path)
        profile_cfg[agent_type] = entry

    config.agent_config.profile = profile_cfg
    config.agent_config.relationship_path = paths["relationship_path"]

    logger.info(
        f"Applied topic '{scenario_slug}': UserAgent={portrait.get('UserAgent')}, "
        f"Algorithm={portrait.get('Algorithm')}, Platform={portrait.get('Platform')}"
    )


_CONFIG_DATA_FIELDS = (
    "model_config",
    "distribution_config",
    "database_config",
    "observation_config",
    "monitor_config",
    "agent_config",
    "simulator_config",
)


def _clone_onesim_config(config: Any) -> Any:
    """Copy serializable config fields; keep the shared runtime registry.

    ``OneSimConfig.registry`` holds live component instances (HTTP clients,
    locks) after the first scenario. ``copy.deepcopy`` of the whole config
    tries to pickle those and raises ``TypeError: cannot pickle '_thread.RLock'``.
    """
    cloned = copy.copy(config)
    cloned.registry = getattr(config, "registry", None)
    for name in _CONFIG_DATA_FIELDS:
        value = getattr(config, name, None)
        setattr(cloned, name, copy.deepcopy(value) if value is not None else None)
    return cloned


def clone_config_for_scenario(config: Any, scenario_slug: str) -> Any:
    """Copy config data and apply one topic bundle."""
    cloned = _clone_onesim_config(config)
    apply_scenario_to_config(cloned, scenario_slug)
    return cloned


def get_batch_scenarios(config: Any) -> List[str]:
    additional = (
        config.simulator_config.environment.get("additional_config") or {}
    )
    if additional.get("batch_from_manifest"):
        slugs = list_scenarios(config.env_path)
        if slugs:
            return slugs
    batch = additional.get("batch_scenarios")
    if batch:
        return list(batch)
    scenario = additional.get("scenario")
    if scenario:
        return [scenario]
    return []


def slugify_topic(topic: str, keyword: str = "") -> str:
    """Build a filesystem-safe slug for a topic directory."""
    keyword_part = re.sub(r"[^\w\-]+", "_", str(keyword or "").strip()).strip("_").lower()
    keyword_part = keyword_part[:40].strip("_")
    digest = hashlib.md5(topic.encode("utf-8")).hexdigest()[:6]
    if keyword_part:
        return f"{keyword_part}_{digest}"
    return f"topic_{digest}"
