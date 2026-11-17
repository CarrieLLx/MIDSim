"""Helpers for reading MIDSim experiment params loaded from config/params_*.json."""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

_REPO_ROOT = Path(__file__).resolve().parents[3]


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


def _as_str_list(value: Any, default: List[str]) -> List[str]:
    if isinstance(value, list) and value:
        return [str(x) for x in value if x is not None and str(x).strip()]
    return list(default)


def _dig(block: Any, *keys: str) -> Any:
    cur = block
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _exposure(root: Dict[str, Any], *keys: str) -> Any:
    return _dig(root.get("exposure"), *keys)


def _user_sub(root: Dict[str, Any], name: str) -> Dict[str, Any]:
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


def _positive_float(raw: Any, default: float) -> float:
    if raw is None:
        return default
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return default


def _parse_social_feed_budget(raw: Any) -> Optional[int]:
    try:
        n = int(float(raw))
    except (TypeError, ValueError):
        return None
    if n <= 0 or n >= 10_000:
        return None
    return n


@dataclass
class UserParams:
    activity_remap: Tuple[float, float]
    attention_budget: int
    social_recommendation_prob: float
    algorithmic_recommendation_prob: float
    social_feed_last_login_cap_days: float
    social_feed_budget: Optional[int]


@dataclass
class RecommendationParams:
    alpha: float
    max_limit: int
    coarse_k: int
    current_posts_k: int
    rank_top_k: int
    types: List[str]
    posts_per_group: int = 10
    popularity_distribution: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class BM25Params:
    k1: float = 1.2
    b: float = 0.75


@dataclass
class SearchParams:
    alpha: float
    max_limit: int
    types: List[str]
    bm25: BM25Params = field(default_factory=BM25Params)


@dataclass
class GateParams:
    memory_similarity: Dict[str, Any]
    interaction_threshold: Dict[str, Any]
    low_activity_memory_threshold: float


@dataclass
class MIDSimParams:
    user: UserParams
    recommendation: RecommendationParams
    search: SearchParams
    gates: GateParams

    @classmethod
    def from_root(cls, root: Dict[str, Any], *, profile_limit: Any = 15) -> "MIDSimParams":
        return cls(
            user=_parse_user(root),
            recommendation=_parse_recommendation(root, profile_limit),
            search=_parse_search(root, profile_limit),
            gates=_parse_gates(root),
        )

    @classmethod
    async def load(cls, agent) -> "MIDSimParams":
        root = await _midsim_root(agent)
        return cls.from_root(root, profile_limit=await _profile_limit(agent))


def _parse_user(root: Dict[str, Any]) -> UserParams:
    remap = _user_sub(root, "activity").get("remap")
    if not isinstance(remap, dict):
        remap = _dig(root, "user", "activity_remap")
    if isinstance(remap, dict):
        out_min = _clamp_float(remap.get("out_min"), 0.4)
        out_max = _clamp_float(remap.get("out_max"), 0.8)
    else:
        out_min, out_max = 0.4, 0.8
    if out_max < out_min:
        out_min, out_max = out_max, out_min

    attention = _exposure(root, "notification", "attention_budget")
    if attention is None:
        attention = _dig(root, "user", "attention_budget")

    social_p = _exposure(root, "social", "probability")
    if social_p is None:
        social_p = _dig(root, "user", "social_recommendation_prob")

    algo_p = _exposure(root, "recommendation", "probability")
    if algo_p is None:
        algo_p = _dig(root, "user", "algorithmic_recommendation_prob")

    user = root.get("user") if isinstance(root.get("user"), dict) else {}
    cap_days = user.get("own_note_cap_days") if user else None
    if cap_days is None:
        cap_days = _dig(root, "user", "social_feed_last_login_cap_days")

    budget_raw = _exposure(root, "social", "social_feed_budget")
    if budget_raw is None and user:
        budget_raw = user.get("social_feed_budget")
        if budget_raw is None:
            budget_raw = user.get("social_feed_max_recommendations")
    if budget_raw is None:
        budget_raw = _dig(root, "user", "social_feed_budget")

    return UserParams(
        activity_remap=(out_min, out_max),
        attention_budget=_clamp_profile_limit(attention, default=10, cap_max=10_000),
        social_recommendation_prob=_clamp_float(social_p, default=1.0),
        algorithmic_recommendation_prob=_clamp_float(algo_p, default=1.0),
        social_feed_last_login_cap_days=_positive_float(cap_days, 7.0),
        social_feed_budget=_parse_social_feed_budget(budget_raw),
    )


def _resolve_repo_path(raw: str) -> Path:
    path = Path(str(raw).strip()).expanduser()
    if path.is_absolute():
        return path
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (_REPO_ROOT / path).resolve()


def _load_popularity_distribution(root: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load popularity buckets from inline list or a JSON file path.

    Prefer ``popularity_distribution_path``. ``popularity_distribution`` may still be
    an inline list (legacy) or a string path.
    """
    path_raw = _exposure(root, "recommendation", "popularity_distribution_path")
    distribution = _exposure(root, "recommendation", "popularity_distribution")
    if path_raw is None and isinstance(distribution, str):
        path_raw = distribution
        distribution = None

    if path_raw is not None and str(path_raw).strip():
        path = _resolve_repo_path(str(path_raw))
        if not path.is_file():
            raise FileNotFoundError(f"popularity_distribution file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"popularity_distribution file must be a JSON list: {path}")
        return [row for row in data if isinstance(row, dict)]

    if isinstance(distribution, list):
        return [row for row in distribution if isinstance(row, dict)]
    return []


def _parse_recommendation(root: Dict[str, Any], profile_limit: Any) -> RecommendationParams:
    limit = _exposure(root, "recommendation", "limit")
    if limit is None:
        limit = profile_limit
    alpha = _exposure(root, "recommendation", "alpha")
    if alpha is None:
        alpha = _dig(root, "recommender", "recommendation_alpha")
    types = _exposure(root, "recommendation", "types")
    if types is None:
        types = _dig(root, "user", "default_algorithm_types")
    interest = _exposure(root, "recommendation", "interest_recommendation")
    if isinstance(interest, dict):
        coarse_raw = interest.get("coarse_k", 20)
        current_posts_raw = interest.get("current_posts_k", 1)
        rank_raw = interest.get("rank_top_k", 20)
    else:
        coarse_raw = _dig(root, "recommender", "interest", "coarse_k")
        current_posts_raw = _dig(root, "recommender", "interest", "current_posts_k")
        rank_raw = _dig(root, "recommender", "interest", "rank_top_k")
        if coarse_raw is None:
            coarse_raw = 20
        if current_posts_raw is None:
            current_posts_raw = 1
        if rank_raw is None:
            rank_raw = 20
    return RecommendationParams(
        alpha=_clamp_float(alpha, default=0.2),
        max_limit=_clamp_profile_limit(limit, default=15, cap_max=10_000),
        coarse_k=_clamp_profile_limit(coarse_raw, default=20, cap_max=10_000),
        current_posts_k=_clamp_profile_limit(current_posts_raw, default=1, cap_max=10_000),
        rank_top_k=_clamp_profile_limit(rank_raw, default=20, cap_max=10_000),
        types=_as_str_list(types, ["Interest Recommendation"]),
        posts_per_group=_clamp_profile_limit(
            interest.get("posts_per_group") if isinstance(interest, dict) else None,
            default=10, cap_max=10_000,
        ),
        popularity_distribution=_load_popularity_distribution(root),
    )


def _parse_bm25(raw: Any) -> BM25Params:
    block = raw if isinstance(raw, dict) else {}
    values = {}
    for name, default in (("k1", 1.2), ("b", 0.75)):
        try:
            value = float(block.get(name, default))
        except (TypeError, ValueError, OverflowError):
            value = default
        if not math.isfinite(value) or value < 0 or (name == "b" and value > 1):
            value = default
        values[name] = value
    return BM25Params(**values)


def _parse_search(root: Dict[str, Any], profile_limit: Any) -> SearchParams:
    limit = _exposure(root, "search", "limit")
    if limit is None:
        limit = profile_limit
    alpha = _exposure(root, "search", "alpha")
    if alpha is None:
        alpha = _dig(root, "recommender", "search_alpha")
    types = _exposure(root, "search", "types")
    if types is None:
        types = _dig(root, "user", "default_search_types")
    return SearchParams(
        alpha=_clamp_float(alpha, default=0.5),
        max_limit=_clamp_profile_limit(limit, default=15, cap_max=50),
        types=_as_str_list(types, ["Relevant Search"]),
        bm25=_parse_bm25(_exposure(root, "search", "bm25")),
    )


def _parse_gates(root: Dict[str, Any]) -> GateParams:
    memory = _user_sub(root, "memory_similarity")
    if not memory:
        legacy = _dig(root, "step15")
        memory = legacy if isinstance(legacy, dict) else {}

    interaction = _user_sub(root, "interaction_threshold")
    if not interaction:
        legacy = _dig(root, "user", "interaction_threshold")
        interaction = legacy if isinstance(legacy, dict) else {}

    activity = _user_sub(root, "activity")
    mem_th = activity.get("low_activity_memory_gate_threshold") if activity else None
    if mem_th is None:
        mem_th = _dig(root, "user", "low_activity_memory_gate_threshold")

    return GateParams(
        memory_similarity=dict(memory),
        interaction_threshold=dict(interaction),
        low_activity_memory_threshold=_clamp_float(mem_th, 0.7),
    )


_midsim_root_cache: Optional[Dict[str, Any]] = None


def clear_midsim_root_cache() -> None:
    global _midsim_root_cache
    _midsim_root_cache = None


async def _midsim_root(agent) -> Dict[str, Any]:
    global _midsim_root_cache
    if isinstance(_midsim_root_cache, dict) and _midsim_root_cache:
        return _midsim_root_cache

    raw = None
    env = getattr(agent, "env", None)
    if env is not None and hasattr(env, "get_data"):
        try:
            raw = await env.get_data("midsim_params", None)
        except Exception:
            raw = None
    if not isinstance(raw, dict) or not raw:
        raw = await agent.get_env_data("midsim_params", None)
    if isinstance(raw, dict) and raw:
        _midsim_root_cache = raw
        return raw
    return {}


async def _profile_limit(agent) -> Any:
    try:
        return await agent.get_data("limit", 15)
    except Exception:
        return 15


async def midsim_agent_config(agent) -> dict:
    """Agent section from loaded params JSON (backbone_profile, general_agent_locale, …)."""
    root = await _midsim_root(agent)
    section = root.get("agent")
    return section if isinstance(section, dict) else {}


async def is_llama_agent_profile(agent, locale: str) -> bool:
    """True when params (or runtime) specify backbone llama + the given locale (zh/en)."""
    from onesim.agent.backbone import is_llama_backbone, normalize_backbone_profile
    from onesim.agent.locale import get_general_agent_locale, normalize_prompt_locale

    expected = normalize_prompt_locale(locale)
    cfg = await midsim_agent_config(agent)
    if cfg:
        backbone = normalize_backbone_profile(cfg.get("backbone_profile"))
        actual = normalize_prompt_locale(cfg.get("general_agent_locale"))
        if backbone == "llama" and actual == expected:
            return True
        if cfg.get("backbone_profile") or cfg.get("general_agent_locale"):
            return False
    return is_llama_backbone() and get_general_agent_locale() == expected


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
