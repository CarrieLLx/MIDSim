"""Helpers for reading MIDSim experiment params loaded from config/params_*.json."""

import random
from typing import Any, Dict, List, Optional, Tuple, Union


def _clamp_float(value: Any, default: float, low: float = 0.0, high: float = 1.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


def _clamp_profile_limit(value: Any, default: int, cap_max: int) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = default
    return max(1, min(cap_max, parsed))


async def _midsim_root(agent) -> Dict[str, Any]:
    raw = await agent.get_env_data("midsim_params", None)
    return raw if isinstance(raw, dict) else {}


def _dig(block: Any, *keys: str) -> Any:
    cur = block
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


async def _exposure_value(agent, *keys: str, default: Any = None) -> Any:
    root = await _midsim_root(agent)
    value = _dig(root.get("exposure"), *keys)
    return default if value is None else value


async def _user_sub_block(agent, name: str) -> Dict[str, Any]:
    root = await _midsim_root(agent)
    user = root.get("user")
    if isinstance(user, dict):
        block = user.get(name)
        if isinstance(block, dict):
            return block
    gates = root.get("gates")
    if isinstance(gates, dict):
        block = gates.get(name)
        if isinstance(block, dict):
            return block
    return {}


async def recommender_sampling_params(agent, mode: str = "recommendation") -> Tuple[float, int]:
    """
    Read Bernoulli continuation alpha from midsim_params; max_limit from profile ``limit``.

    mode: "recommendation" | "search"
    """
    profile_limit = await agent.get_data("limit", 15)

    if mode == "search":
        raw = await _exposure_value(agent, "search", "alpha")
        if raw is None:
            raw = await agent.get_env_data("midsim_params.recommender.search_alpha", 0.5)
        alpha = _clamp_float(raw, default=0.5)
        max_limit = _clamp_profile_limit(profile_limit, default=15, cap_max=50)
    else:
        raw = await _exposure_value(agent, "recommendation", "alpha")
        if raw is None:
            raw = await agent.get_env_data("midsim_params.recommender.recommendation_alpha", 0.2)
        alpha = _clamp_float(raw, default=0.2)
        max_limit = _clamp_profile_limit(profile_limit, default=15, cap_max=15)

    return alpha, max_limit


async def interest_recommendation_candidate_limits(agent) -> Tuple[int, int]:
    """Max candidates from candidate_pool (interest_k) and current_feed (target_k) for LLM rerank."""
    block = await _exposure_value(agent, "recommendation", "interest_recommendation")
    if isinstance(block, dict):
        interest_raw = block.get("interest_k", 20)
        target_raw = block.get("target_k", 1)
    else:
        prefix = "midsim_params.recommender.interest"
        interest_raw = await agent.get_env_data(f"{prefix}.interest_k", 20)
        target_raw = await agent.get_env_data(f"{prefix}.target_k", 1)
    interest_k = _clamp_profile_limit(interest_raw, default=20, cap_max=10_000)
    target_k = _clamp_profile_limit(target_raw, default=1, cap_max=10_000)
    return interest_k, target_k


def _as_str_list(value: Any, default: List[str]) -> List[str]:
    if isinstance(value, list) and value:
        return [str(x) for x in value if x is not None and str(x).strip()]
    return list(default)


async def user_default_algorithm_types(agent) -> List[str]:
    raw = await _exposure_value(agent, "recommendation", "types")
    if raw is None:
        raw = await agent.get_env_data("midsim_params.user.default_algorithm_types", None)
    return _as_str_list(raw, ["Interest Recommendation"])


async def user_default_search_types(agent) -> List[str]:
    raw = await _exposure_value(agent, "search", "types")
    if raw is None:
        raw = await agent.get_env_data("midsim_params.user.default_search_types", None)
    return _as_str_list(raw, ["Relevant Search"])


async def user_attention_budget(agent) -> int:
    raw = await _exposure_value(agent, "notification", "attention_budget")
    if raw is None:
        raw = await agent.get_env_data("midsim_params.user.attention_budget", 10)
    return _clamp_profile_limit(raw, default=10, cap_max=10_000)


async def user_social_recommendation_prob(agent) -> float:
    raw = await _exposure_value(agent, "social", "probability")
    if raw is None:
        raw = await agent.get_env_data("midsim_params.user.social_recommendation_prob", 1.0)
    return _clamp_float(raw, default=1.0)


async def user_algorithmic_recommendation_prob(agent) -> float:
    raw = await _exposure_value(agent, "recommendation", "probability")
    if raw is None:
        raw = await agent.get_env_data("midsim_params.user.algorithmic_recommendation_prob", 1.0)
    return _clamp_float(raw, default=1.0)


async def user_activity_remap(agent) -> Tuple[float, float]:
    block = (await _user_sub_block(agent, "activity")).get("remap")
    if not isinstance(block, dict):
        block = await agent.get_env_data("midsim_params.user.activity_remap", None)
    if isinstance(block, dict):
        out_min = _clamp_float(block.get("out_min"), 0.4)
        out_max = _clamp_float(block.get("out_max"), 0.8)
    else:
        out_min, out_max = 0.4, 0.8
    if out_max < out_min:
        out_min, out_max = out_max, out_min
    return out_min, out_max


async def user_stale_days(agent) -> float:
    block = await _user_sub_block(agent, "freshness")
    raw = block.get("stale_days") if block else None
    if raw is None:
        raw = await agent.get_env_data("midsim_params.user.stale_days", 7.0)
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 7.0


async def user_social_feed_last_login_cap_days(agent) -> float:
    """Own-note / social-feed time lower bound cap in days; 0 disables truncation."""
    root = await _midsim_root(agent)
    user = root.get("user")
    raw = user.get("own_note_cap_days") if isinstance(user, dict) else None
    if raw is None:
        raw = await agent.get_env_data("midsim_params.user.social_feed_last_login_cap_days", 7.0)
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 7.0


async def user_social_feed_budget(agent) -> Optional[int]:
    """
    Max items per SocialRecommendationEvent batch (Twitter).
    Reads exposure.social.social_feed_budget; falls back to user.social_feed_budget.
    Returns None when <= 0 or >= 10000 (no cap).
    """
    raw = await _exposure_value(agent, "social", "social_feed_budget")
    if raw is None:
        root = await _midsim_root(agent)
        user = root.get("user")
        if isinstance(user, dict):
            raw = user.get("social_feed_budget")
            if raw is None:
                raw = user.get("social_feed_max_recommendations")
    if raw is None:
        raw = await agent.get_env_data("midsim_params.user.social_feed_budget", 0)
    try:
        n = int(float(raw))
    except (TypeError, ValueError):
        return None
    if n <= 0 or n >= 10_000:
        return None
    return n


async def user_low_activity_memory_gate_threshold(agent) -> float:
    block = await _user_sub_block(agent, "activity")
    raw = block.get("low_activity_memory_gate_threshold") if block else None
    if raw is None:
        raw = await agent.get_env_data(
            "midsim_params.user.low_activity_memory_gate_threshold", 0.7
        )
    return _clamp_float(raw, 0.7)


async def user_low_activity_time_module_threshold(agent) -> Optional[float]:
    block = await _user_sub_block(agent, "freshness")
    raw = block.get("low_activity_time_module_threshold") if block else None
    if raw is None:
        raw = await agent.get_env_data(
            "midsim_params.user.low_activity_time_module_threshold", None
        )
    if raw is None:
        return None
    return _clamp_float(raw, 0.75)


async def user_interaction_threshold_config(agent) -> dict:
    cfg = await _user_sub_block(agent, "interaction_threshold")
    if cfg:
        return cfg
    legacy = await agent.get_env_data("midsim_params.user.interaction_threshold", None)
    return legacy if isinstance(legacy, dict) else {}


async def memory_similarity_gate_params(agent) -> dict:
    block = await _user_sub_block(agent, "memory_similarity")
    if block:
        return block
    legacy = await agent.get_env_data("midsim_params.step15", None)
    return legacy if isinstance(legacy, dict) else {}


async def step15_params(agent) -> dict:
    """Alias for memory_similarity gate config (legacy name)."""
    return await memory_similarity_gate_params(agent)


def draw_discrete(
    rng: random.Random,
    block: Any,
    default_support: List[Union[int, str]],
    default_probs: List[float],
) -> Union[int, str]:
    if isinstance(block, dict):
        support = block.get("support", default_support)
        probs = block.get("probs", default_probs)
    else:
        support, probs = default_support, default_probs
    if not support:
        support, probs = default_support, default_probs
    u = rng.random()
    cumulative = 0.0
    for value, prob in zip(support, probs):
        cumulative += float(prob)
        if u <= cumulative:
            return value
    return support[-1]
