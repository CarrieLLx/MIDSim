from typing import Any, Callable, Dict, List, Optional
import random
import asyncio

from loguru import logger
from onesim.agent import GeneralAgent
from onesim.profile import AgentProfile
from onesim.memory import MemoryStrategy
from onesim.planning import PlanningBase
from onesim.events import Event
from onesim.relationship import RelationshipManager
from onesim.utils.midsim_params import MIDSimParams

from .events import *
from .algorithms import RecommendationEngine, SearchEngine
from .embedding import default_embedding_config_path


class Algorithm(GeneralAgent):
    """Platform Algorithm"""

    def __init__(self,
                 sys_prompt: str | None = None,
                 model_config_name: str = None,
                 event_bus_queue: asyncio.Queue = None,
                 profile: AgentProfile=None,
                 memory: MemoryStrategy=None,
                 planning: PlanningBase=None,
                 relationship_manager: RelationshipManager=None) -> None:
        super().__init__(sys_prompt, model_config_name, event_bus_queue, profile, memory, planning, relationship_manager)
        self.register_event("StartEvent", "update_current_posts")
        self.register_event("GetAlgorithmRecommendationEvent", "send_recommendation_results")
        self.register_event("GetSearchResultEvent", "send_search_results")

        self.recommender = RecommendationEngine(self)
        self.search = SearchEngine(self)
        self.recommendation_algorithms: Dict[str, Callable] = {
            "random": self.recommender.random,
            "popularity": self.recommender.popularity,
            "interest": self.recommender.interest,
        }
        self.default_algorithm = "interest"
        self.search_algorithms: Dict[str, Callable] = {
            "relevant": self.search.relevant,
            "keyword": self.search.keyword,
            "random": self.search.random,
            "bm25": self.search.bm25,
        }
        self.default_search = "relevant"

    async def update_current_posts(self, event: Event) -> None:
        """Update current posts."""
        current_posts = event.current_posts
        if not isinstance(current_posts, dict) or not current_posts:
            logger.warning("No current posts found")
        self.profile.update_data("current_posts", current_posts)
        self.recommender.reset_popularity_cache()

    def _profile_role(self) -> str:
        return str(self.profile.get_data("role", "") if self.profile else "").strip().lower()

    def _resolve_configured_algorithm(
        self, event: Event, kind: str, configured_types: Optional[List[str]] = None
    ) -> str:
        table = self.recommendation_algorithms if kind == "recommendation" else self.search_algorithms
        default = self.default_algorithm if kind == "recommendation" else self.default_search

        def normalize(value: Any) -> str:
            name = str(value or "").strip()
            return name.lower().removesuffix(f" {kind}").strip()

        event_type = normalize(getattr(event, "type", ""))
        if event_type in table:
            return event_type
        # params is the sole configuration source; the first supported type wins.
        for item in configured_types or []:
            name = normalize(item)
            if name in table:
                return name
        return default

    @staticmethod
    def _keep_env_window_posts(
        selected: Any,
        current_posts: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Drop ranked items that are not in the env_data / current_posts window."""
        if not isinstance(selected, dict) or not isinstance(current_posts, dict):
            return {}
        window = {
            str(pid).strip(): post
            for pid, post in current_posts.items()
            if pid is not None and str(pid).strip() and isinstance(post, dict)
        }
        out: Dict[str, Dict[str, Any]] = {}
        dropped = 0
        for pid in selected.keys():
            sid = str(pid).strip()
            env_post = window.get(sid)
            if env_post is None:
                dropped += 1
                continue
            out[sid] = env_post
        if dropped:
            logger.info(
                f"dropped {dropped} posts not in current_posts; kept {len(out)}"
            )
        return out

    def _sample_bernoulli(self, alpha: float, max_limit: int = 3) -> int:
        """Sample recommendation limit with Bernoulli distribution."""
        try:
            alpha = float(alpha)
        except (TypeError, ValueError):
            alpha = 0.5
        alpha = max(0.0, min(1.0, alpha))
        max_limit = max(1, int(max_limit))

        limit = 1
        for _ in range(max_limit - 1):
            if random.random() < alpha:
                limit += 1
            else:
                break
        return limit

    async def send_recommendation_results(self, event: Event) -> List[Event]:
        """Send recommendation results."""
        current_timestamp = event.timestamp

        current_posts = {}
        if self.profile is not None:
            current_posts = self.profile.get_data("current_posts", {})
        if not isinstance(current_posts, dict) or not current_posts:
            current_posts = getattr(event, "current_posts", {})
        if not isinstance(current_posts, dict) or not current_posts:
            logger.warning("No current posts found from profile/event")
            return []

        if self._profile_role() == "search":
            return []

        params = await MIDSimParams.load(self)
        algorithm_name = self._resolve_configured_algorithm(event, "recommendation", params.recommendation.types)
        if algorithm_name not in self.recommendation_algorithms:
            algorithm_name = self.default_algorithm
            logger.warning(f"Unknown recommendation algorithm type, using default: {self.default_algorithm}")

        recommendation_func = self.recommendation_algorithms[algorithm_name]

        recommendation_limit = self._sample_bernoulli(
            params.recommendation.alpha, max_limit=params.recommendation.max_limit
        )
        logger.info(
            f"Algorithm recommendation group_count={recommendation_limit} "
            f"(limit={params.recommendation.max_limit}, alpha={params.recommendation.alpha})"
        )

        recommended_post_ids = list(event.recommended_post_ids)

        user_profile_raw = getattr(event, "user_profile", None)
        user_profile: Dict[str, Any] = (
            user_profile_raw if isinstance(user_profile_raw, dict) else {}
        )

        recommended_contents = await recommendation_func(
            recommended_post_ids,
            current_posts,
            recommendation_limit,
            current_timestamp=current_timestamp,
            user_profile=user_profile,
        )
        recommended_contents = self._keep_env_window_posts(recommended_contents, current_posts)

        if not recommended_contents:
            logger.debug(f"No recommended contents for user {getattr(event, 'from_agent_id', 'unknown')}")
            return []

        logger.info(
            f"Sending AlgorithmRecommendationEvent to UserAgent {event.from_agent_id}, "
            f"recommended contents: {len(recommended_contents)}"
        )

        recommendation_event = AlgorithmRecommendationEvent(
            from_agent_id=self.profile_id,
            to_agent_id=event.from_agent_id,
            timestamp=current_timestamp,
            timestamp_duration=int(getattr(event, "timestamp_duration", 0) or 0),
            current_step=getattr(event, "current_step", 1),
            max_step=getattr(event, "max_step", 8),
            recommendations=dict(recommended_contents)
        )
        return [recommendation_event]

    async def send_search_results(self, event: Event) -> List[Event]:
        """Generate search results."""
        current_timestamp = event.timestamp

        current_posts = {}
        if self.profile is not None:
            current_posts = self.profile.get_data("current_posts", {})
        if not isinstance(current_posts, dict) or not current_posts:
            current_posts = getattr(event, "current_posts", {})
        if not isinstance(current_posts, dict) or not current_posts:
            logger.warning("No current posts found from profile/event")
            return []

        if self._profile_role() == "recommendation":
            return []

        params = await MIDSimParams.load(self)
        algorithm_name = self._resolve_configured_algorithm(event, "search", params.search.types)
        if algorithm_name not in self.search_algorithms:
            algorithm_name = self.default_search
            logger.warning(f"Unknown search algorithm type, using default: {self.default_search}")

        search_func = self.search_algorithms[algorithm_name]

        search_limit = self._sample_bernoulli(
            params.search.alpha, max_limit=params.search.max_limit
        )
        logger.info(
            f"Algorithm search algorithm={algorithm_name}, max_limit={params.search.max_limit}, "
            f"sampled search_limit={search_limit}"
        )

        search_query = str(getattr(event, "search_query", "") or "")
        ms_params = params.gates.memory_similarity or {}
        embedding_config_path = str(ms_params.get("embedding_config_path") or "").strip()
        if not embedding_config_path:
            embedding_config_path = default_embedding_config_path()

        search_kwargs = {"bm25_params": params.search.bm25} if algorithm_name == "bm25" else {}
        search_results = await search_func(
            current_posts,
            search_limit,
            current_timestamp=current_timestamp,
            search_query=search_query,
            embedding_config_path=embedding_config_path,
            **search_kwargs,
        )
        search_results = self._keep_env_window_posts(search_results, current_posts)

        if not search_results:
            logger.debug(f"No search results for user {getattr(event, 'from_agent_id', 'unknown')}")
            return []

        logger.info(
            f"Sending SearchRecommendationEvent (search) to UserAgent {event.from_agent_id}, "
            f"search results: {len(search_results)}"
        )

        search_event = SearchResultEvent(
            from_agent_id=self.profile_id,
            to_agent_id=event.from_agent_id,
            timestamp=current_timestamp,
            timestamp_duration=int(getattr(event, "timestamp_duration", 0) or 0),
            current_step=getattr(event, "current_step", 1),
            max_step=getattr(event, "max_step", 8),
            recommendations=dict(search_results),
        )
        return [search_event]
