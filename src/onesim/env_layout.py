"""Resolve topic/schema paths and root recommendation data; accept legacy profiles."""

from __future__ import annotations

import os
from typing import Optional


def uses_structured_layout(env_path: str) -> bool:
    if not env_path:
        return False
    return os.path.isdir(os.path.join(env_path, "topics")) or os.path.isdir(
        os.path.join(env_path, "sim")
    )


def _first_existing(*paths: str, fallback: Optional[str] = None) -> str:
    for path in paths:
        if path and os.path.isfile(path):
            return path
    if fallback:
        return fallback
    return paths[0] if paths else ""


def get_protocol_dir(env_path: str) -> str:
    if os.path.isfile(os.path.join(env_path, "actions.json")):
        return env_path
    sim_dir = os.path.join(env_path, "sim")
    if os.path.isdir(sim_dir):
        return sim_dir
    return env_path


def get_sim_dir(env_path: str) -> str:
    return get_protocol_dir(env_path)


def get_actions_path(env_path: str) -> str:
    return _first_existing(
        os.path.join(env_path, "actions.json"),
        os.path.join(env_path, "sim", "actions.json"),
        fallback=os.path.join(get_protocol_dir(env_path), "actions.json"),
    )


def get_events_path(env_path: str) -> str:
    return _first_existing(
        os.path.join(env_path, "events.json"),
        os.path.join(env_path, "sim", "events.json"),
        fallback=os.path.join(get_protocol_dir(env_path), "events.json"),
    )


def get_scene_info_path(env_path: str) -> str:
    return _first_existing(
        os.path.join(env_path, "scene_info.json"),
        os.path.join(env_path, "sim", "scene_info.json"),
        fallback=os.path.join(get_protocol_dir(env_path), "scene_info.json"),
    )


def get_system_data_model_path(env_path: str) -> str:
    return _first_existing(
        os.path.join(env_path, "system_data_model.json"),
        os.path.join(env_path, "sim", "system_data_model.json"),
        fallback=os.path.join(get_protocol_dir(env_path), "system_data_model.json"),
    )


def get_schema_path(env_path: str, agent_type: str, schema_file: Optional[str] = None) -> str:
    filename = schema_file or f"{agent_type}.json"
    canonical = os.path.join(env_path, "schema", filename)
    legacy_profile = os.path.join(env_path, "profile", "schema", filename)
    if uses_structured_layout(env_path):
        return _first_existing(
            canonical,
            legacy_profile,
            os.path.join(env_path, "sim", "schema", filename),
            fallback=canonical,
        )
    return _first_existing(
        legacy_profile,
        canonical,
        fallback=legacy_profile,
    )


def get_topics_root(env_path: str) -> str:
    if uses_structured_layout(env_path):
        return os.path.join(env_path, "topics")
    return os.path.join(env_path, "scenarios")


def get_topics_index_path(env_path: str) -> str:
    topics_root = get_topics_root(env_path)
    if uses_structured_layout(env_path):
        return os.path.join(topics_root, "index.json")
    return os.path.join(topics_root, "manifest.json")


def get_algorithm_index_path(env_path: str) -> str:
    return _first_existing(
        os.path.join(env_path, "platform", "Algorithm.json"),
        os.path.join(env_path, "profile", "Algorithm", "Algorithm.json"),
        os.path.join(env_path, "profile", "algorithm", "Algorithm.json"),
        os.path.join(env_path, "algorithm", "index.json"),
        os.path.join(env_path, "algorithm", "Algorithm.json"),
        fallback=os.path.join(env_path, "platform", "Algorithm.json"),
    )


def get_platform_profile_path(env_path: str) -> str:
    algo_path = get_algorithm_index_path(env_path)
    return os.path.join(os.path.dirname(algo_path), "Platform.json")


def get_algorithm_profiles_path(env_path: str, rel_path: Optional[str] = None) -> str:
    if rel_path:
        if os.path.isabs(rel_path):
            return rel_path
        joined = os.path.join(env_path, rel_path.replace("/", os.sep))
        if os.path.isfile(joined):
            return joined
    if uses_structured_layout(env_path):
        return get_algorithm_index_path(env_path)
    return os.path.join(get_topics_root(env_path), "Algorithm.json")


def get_user_agent_profile_dir(env_path: str, slug: str) -> str:
    topic_dir = os.path.join(get_topics_root(env_path), slug)
    if os.path.isfile(os.path.join(topic_dir, "UserAgent.json")):
        return topic_dir
    legacy = os.path.join(env_path, "profile", "UserAgent", slug)
    if os.path.isdir(legacy):
        return legacy
    return topic_dir


def get_user_agent_profile_path(env_path: str, slug: str) -> str:
    return _first_existing(
        os.path.join(get_topics_root(env_path), slug, "UserAgent.json"),
        os.path.join(env_path, "profile", "UserAgent", slug, "UserAgent.json"),
        fallback=os.path.join(get_topics_root(env_path), slug, "UserAgent.json"),
    )


def get_recommendation_path(env_path: str, slug: str) -> str:
    return os.path.join(get_topics_root(env_path), slug, "recommendation.json")


def get_platform_recommendation_path(env_path: str) -> str:
    return os.path.join(env_path, "recommendation.json")


def get_relationship_path(env_path: str, slug: str) -> str:
    return _first_existing(
        os.path.join(get_topics_root(env_path), slug, "Relationship.csv"),
        os.path.join(env_path, "profile", "UserAgent", slug, "Relationship.csv"),
        fallback=os.path.join(get_topics_root(env_path), slug, "Relationship.csv"),
    )


def get_topic_meta_path(topic_dir: str) -> str:
    env_path = _env_path_from_topic_dir(topic_dir)
    if env_path and uses_structured_layout(env_path):
        return os.path.join(topic_dir, "meta.json")
    return os.path.join(topic_dir, "scenario.json")


def _env_path_from_topic_dir(topic_dir: str) -> Optional[str]:
    path = topic_dir
    while path:
        if os.path.isdir(os.path.join(path, "topics")) or os.path.isdir(os.path.join(path, "sim")):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return None
