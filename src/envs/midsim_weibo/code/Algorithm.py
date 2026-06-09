from typing import Any, List, Optional, Callable, Dict, Tuple, Set
from collections import defaultdict
import json
import os
import asyncio
import random
from loguru import logger
from onesim.models import JsonBlockParser
from onesim.agent import GeneralAgent
from onesim.profile import AgentProfile
from onesim.memory import MemoryStrategy
from onesim.planning import PlanningBase
from onesim.events import Event
from onesim.relationship import RelationshipManager
from .events import *
from .embedding_client import cosine_similarity, get_embeddings, load_embedding_config
from onesim.utils.midsim_params import (
    interest_recommendation_candidate_limits,
    memory_similarity_gate_params,
    recommender_sampling_params,
)
from .user_agent_gates import MemorySimilarityGate
from .utils import format_historical_summary, format_popularity_distribution, is_original_blog, to_float

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
        self.register_event("StartEvent", "update_current_blogs")
        self.register_event("GetAlgorithmRecomendationEvent", "send_recommendation_results")
        self.register_event("GetSearchResultEvent", "send_search_results")
        
        # Recommendation algorithm mapping
        self.type_to_algorithm: Dict[str, str] = {
            "Random Recommendation": "random",
            "Popularity Recommendation": "popularity",
            "Interest Recommendation": "interest",
        }
        
        self.recommendation_algorithms: Dict[str, Callable] = {
            "random": self._random_recommendation,
            "popularity": self._popularity_recommendation,
            "interest": self._interest_recommendation,
        }
        self.default_algorithm = "popularity"
        self._popularity_distribution_cache: Optional[Dict[int, int]] = None

        # Search algorithm mapping
        self.type_to_search: Dict[str, str] = {
            "Relevant Search": "relevant",
        }
        
        self.search_algorithms: Dict[str, Callable] = {
            "relevant": self._relevant_search,
        }
        self.default_search = "relevant"

    async def update_current_blogs(self, event: Event) -> None:
        """Update current blogs."""
        current_blogs = event.current_blogs
        if not isinstance(current_blogs, dict) or not current_blogs:
            logger.warning("No current blogs found")
        self.profile.update_data("current_blogs", current_blogs)

    def _sample_bernoulli(self, alpha: float, max_limit: int = 3) -> int:
        """Sample recommendation limit with Bernoulli distribution."""
        try:
            alpha = float(alpha)
        except (TypeError, ValueError):
            alpha = 0.5
        alpha = max(0.0, min(1.0, alpha))
        max_limit = max(1, int(max_limit))

        # At least 1 recommendation; then at most max_limit-1 recommendations
        limit = 1
        for _ in range(max_limit - 1):
            if random.random() < alpha:
                limit += 1
            else:
                break
        return limit

    # ---------- Random recommendation ----------
    _RANDOM_REC_MAU_DEFAULT = 588_000_000   # Default Monthly Active Users (MAU)
    _RANDOM_REC_AVG_POSTS_PER_USER_MONTH_DEFAULT = 95.69    # Default avg posts per user per month

    def _random_rec_mau_avg_posts(self) -> Tuple[float, float]:
        """profile: random_rec_mau and random_rec_avg_posts_per_user_month."""
        mau = float(Algorithm._RANDOM_REC_MAU_DEFAULT)
        avg_posts = float(Algorithm._RANDOM_REC_AVG_POSTS_PER_USER_MONTH_DEFAULT)
        if self.profile is not None:
            mau = to_float(
                self.profile.get_data("random_rec_mau", None),
                default=mau,
            )
            avg_posts = to_float(
                self.profile.get_data("random_rec_avg_posts_per_user_month", None),
                default=avg_posts,
            )
        if mau <= 0:
            mau = float(Algorithm._RANDOM_REC_MAU_DEFAULT)
        return mau, avg_posts

    def _random_recommendation(self, recommended_blog_ids: List[str], contents: Dict[str, Dict[str, Any]], current_timestamp: float, limit: int = 10) -> Dict[str, Dict[str, Any]]:
        """Random recommendation, sample with equal probability (1/(MAU×avg)) from candidate pool."""
        if not contents:
            return {}
        blog_ids = list(contents.keys())
        if not blog_ids:
            return {}
        
        # Filter out recommended blogs
        blog_ids = [blog_id for blog_id in blog_ids if blog_id not in recommended_blog_ids]
        if not blog_ids:
            return {}
        
        # Count denominator, p = 1 / (MAU × avg)
        mau, avg_posts = self._random_rec_mau_avg_posts()
        denom = mau * avg_posts
        if denom <= 0:
            p_hit = 0.0
        else:
            p_hit = min(1.0, 1.0 / denom)

        random.shuffle(blog_ids)
        log_each = bool(
            self.profile is not None
            and self.profile.get_data("random_rec_log_per_blog_prob_check", False)
        )

        selected: List[str] = []
        for bid in blog_ids:
            if len(selected) >= limit:
                break
            ok = random.random() < p_hit
            if log_each:
                logger.debug("random_rec blog {}: Bernoulli(p=1/(MAU×avg)={:.6e}) => {}", bid, p_hit, "hit" if ok else "miss")
            if ok:
                selected.append(bid)

        w = (avg_posts / mau) if mau > 0 else 0.0
        logger.debug( "random recommendation: MAU={:.4g} avg_posts={:.6g} => p=1/(MAU×avg)={:.6e}; weight=(1/MAU)×avg={:.6e}",
            mau, avg_posts, p_hit, len(blog_ids), limit, len(selected), w)
        return {blog_id: contents[blog_id] for blog_id in selected}
    
    # ---------- Popularity recommendation ----------
    def _get_popularity_distribution_map(self) -> Dict[int, int]:
        """Get popularity distribution map for popularity recommendation."""
        if self._popularity_distribution_cache is not None:
            return self._popularity_distribution_cache

        merged: Dict[int, int] = {}
        if self.profile is not None:
            raw = self.profile.get_data("popularity_distribution", [])
            merged.update(format_popularity_distribution(raw))
        self._popularity_distribution_cache = merged
        return merged

    def _calculate_popularity(self, blog: Dict[str, Any]) -> float:
        """Calculate blog popularity: use SimEnv's popularity (repost count in sliding window) if available, otherwise use repost_count."""
        if isinstance(blog, dict) and "popularity" in blog:
            rawp = blog.get("popularity")
            try:
                return float(rawp) if rawp is not None else 0.0
            except (TypeError, ValueError):
                return 0.0
        raw = blog.get("repost_count", 0)
        if raw is None:
            return 0.0
        if isinstance(raw, bool):
            return 0.0
        if isinstance(raw, (int, float)):
            return float(raw)
        if isinstance(raw, str):
            s = raw.strip()
            if not s:
                return 0.0
            try:
                return float(s)
            except ValueError:
                return 0.0
        return 0.0

    def _popularity_recommendation(self, recommended_blog_ids: List[str], contents: Dict[str, Dict[str, Any]], current_timestamp: float, limit: int = 10) -> Dict[str, Dict[str, Any]]:
        """Popularity recommendation: mix candidate note popularity with experience popularity distribution, and take the high-popularity notes."""
        if not contents:
            return {}
        blog_ids = list(contents.keys())
        if not blog_ids:
            return {}

        # Filter out recommended blogs
        blog_ids = [blog_id for blog_id in blog_ids if blog_id not in recommended_blog_ids]
        if not blog_ids:
            return {}

        # Get repost count distribution map
        dist_map = self._get_popularity_distribution_map()
        if not dist_map:
            blogs_with_popularity: List[Tuple[str, Dict[str, Any], float]] = []
            for blog_id in blog_ids:
                blog = contents[blog_id]
                if not isinstance(blog, dict):
                    continue
                popularity = self._calculate_popularity(blog)
                blogs_with_popularity.append((blog_id, blog, popularity))
            random.shuffle(blogs_with_popularity)
            blogs_with_popularity.sort(key=lambda x: x[2], reverse=True)
            return {blog_id: blog for blog_id, blog, _ in blogs_with_popularity[:limit]}

        # Mix candidate blogs with experience popularity distribution, and sort by popularity
        buckets: Dict[int, List[Optional[str]]] = defaultdict(list)
        for rc, cnt in dist_map.items():
            try:
                n = int(cnt)
            except (TypeError, ValueError):
                continue
            for _ in range(max(0, n)):
                buckets[rc].append(None)
        for blog_id in blog_ids:
            blog = contents[blog_id]
            if not isinstance(blog, dict):
                continue
            p = int(self._calculate_popularity(blog))
            buckets[p].append(blog_id)

        merged: List[Optional[str]] = []
        for rc in sorted(buckets.keys(), reverse=True):
            row = buckets[rc]
            random.shuffle(row)
            merged.extend(row)

        # Get top blogs by popularity
        top_slice = merged[: max(0, limit)]
        chosen_ids = [x for x in top_slice if x is not None]
        return {bid: contents[bid] for bid in chosen_ids if bid in contents}

    # ---------- Interest recommendation ----------
    @staticmethod
    def _interest_interleave_pool_for_prompt(pool_ids: List[str], pool_set: Set[str]) -> List[str]:
        """Interleave candidate_pool and current_feed ids."""
        ordered = [str(x).strip() for x in pool_ids if x]
        a = [x for x in ordered if x in pool_set]
        b = [x for x in ordered if x not in pool_set]
        if not b:
            return a
        if not a:
            return b
        out: List[str] = []
        i, j = 0, 0
        while i < len(a) or j < len(b):
            if i < len(a):
                out.append(a[i])
                i += 1
            if j < len(b):
                out.append(b[j])
                j += 1
        return out

    @staticmethod
    def _compact_user_profile_for_interest_rec(
        user_profile: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Construct input profile for interest recommendation."""
        if not user_profile or not isinstance(user_profile, dict):
            return {}
        keys = ("id", "nickname", "gender", "description", "location", "interest_tags", "historical_summary")
        out: Dict[str, Any] = {}
        for k in keys:
            if k not in user_profile:
                continue
            v = user_profile[k]
            if k == "historical_summary":
                v = format_historical_summary(v)
            elif k == "description" and isinstance(v, str) and len(v) > 400:
                v = v[:400] + "…"
            out[k] = v
        return out

    async def _interest_recommendation(
        self,
        recommended_blog_ids: List[str],
        contents: Dict[str, Dict[str, Any]],
        current_timestamp: float,
        limit: int = 10,
        candidate_blog_ids: Optional[List[str]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Interest recommendation: mix target list with candidate pool, and use LLM to re-rank."""
        interest_pool: Dict[str, Dict[str, Any]] = {}
        if self.profile is not None:
            raw_ip = self.profile.get_data("interest_recommendation_content_pool", None)
            if isinstance(raw_ip, dict):
                interest_pool = {
                    str(k).strip(): v
                    for k, v in raw_ip.items()
                    if isinstance(v, dict)
                }

        def _resolve_blog(bid: str) -> Optional[Dict[str, Any]]:
            sid = str(bid).strip()
            n = interest_pool.get(sid)
            if isinstance(n, dict):
                return n
            c = contents.get(sid) if isinstance(contents, dict) else None
            return c if isinstance(c, dict) else None

        # Get candidate blog ids
        candidate_blog_ids = list(candidate_blog_ids) if candidate_blog_ids else []
        interest_k, target_k = await interest_recommendation_candidate_limits(self)

        ids_for_user: List[str] = []
        if candidate_blog_ids:
            ids_for_user = [str(x).strip() for x in candidate_blog_ids if x]
        ids_for_user = [
            bid for bid in ids_for_user
            if _resolve_blog(bid) is not None
            and bid not in recommended_blog_ids
        ]
        random.shuffle(ids_for_user)
        pool_sample: List[str] = ids_for_user[:interest_k]
        pool_set: Set[str] = set(pool_sample)

        # Construct current feed blog ids
        feed_blog_ids = list(contents.keys()) if contents else []
        feed_blog_ids = [blog_id for blog_id in feed_blog_ids if blog_id not in recommended_blog_ids]
        feed_eligible = [blog_id for blog_id in feed_blog_ids if blog_id not in pool_set]
        random.shuffle(feed_eligible)
        feed_sample: List[str] = feed_eligible[:target_k]

        # Mix candidate pool and current feed
        pool_ids: List[str] = list(pool_sample) + feed_sample
        if not pool_ids:
            return {}
        if os.environ.get("ONESIM_REC_INTERLEAVE_INPUT", "1").strip().lower() not in ("0", "false", "no", "off"):
            pool_ids = self._interest_interleave_pool_for_prompt(pool_ids, pool_set)

        def _make_candidate_payload(
            ids: List[str], desc_max: int
        ) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for blog_id in ids:
                blog = _resolve_blog(blog_id)
                if not isinstance(blog, dict):
                    continue
                popularity = self._calculate_popularity(blog)
                out.append({
                    "blog_id": blog_id,
                    "content": (blog.get("content", "") or "")[:desc_max],
                })
            return out

        # Construct prompt and avoid context overflow
        max_prompt_chars = int(
            os.environ.get("ONESIM_RECOMMENDER_INTEREST_MAX_PROMPT_CHARS", "48000")
        )
        profile_compact = self._compact_user_profile_for_interest_rec(user_profile)
        if profile_compact:
            user_profile_prefix = (
                "请求用户画像（用于兴趣个性化排序；请优先对齐 interest_tags、description、"
                "historical_summary 等所体现的主题与意图）：\n"
                f"{json.dumps(profile_compact, ensure_ascii=False)}\n\n"
            )
        else:
            user_profile_prefix = ""
            
        instruction = f"""
        你是推荐系统助手。请对下方「候选 blog_id 全集」输出一条**完整排序列表**（从高到低），并严格返回 JSON。

        【目标 — 与用户兴趣对齐】
        1) **主排序信号（权重最高）**：对照用户画像中的 **interest_tags、description、historical_summary**，估计每条微博与该用户在**主题域、身份角色、长期关切、表达习惯**上的一致程度；越像「该用户会点进、会停留、会评论/转发」的内容越靠前。
        2) **语义与标签匹配**：在标题、摘要（desc 截断）、tags_list 中，与用户显式兴趣标签或摘要中反复出现的关切**直接命中或强相关**的微博，优先于仅有弱联想的内容。
        3) **次排序信号（主信号接近时的 tie-break）**：信息是否完整可读（标题+摘要能判断主题，非空泛标题党）、标签是否与正文主题一致；同等相关下，**更具体、更有信息增量**的微博优先于空洞或重复套话的微博。
        4) **多样性轻约束**：主信号已排序后，若多条微博主题极度雷同，可适当错开相邻位次，使前段列表覆盖略广的兴趣面（仍须满足下方「覆盖全集、不丢 id」的硬约束）。

        请严格返回 JSON：
        {{
        "ranked_blog_ids": ["id1", "id2", ...],
        "per_blog_category": ["类别归纳1", "类别归纳2", ...]
        }}

        字段说明：
        - per_blog_category：与 ranked_blog_ids **等长**；第 i 项是对排序中**第 i 条**微博的**类别归纳**（2～12 字为宜），如「学术/投稿」「生活/美食」「情感/成长」等，体现主题域或意图；有用户画像时，可侧写「与用户画像中哪类兴趣更贴近」，勿写成长句理由。
        约束：
        1) 每个 id 必须来自下方候选列表中的 blog_id，且**覆盖全集**；
        2) 不要重复；
        3) per_blog_category 条数必须与 ranked_blog_ids 一致且顺序一一对应；
        """
        desc_max = 800
        candidate_payload: List[Dict[str, Any]] = []
        observation = ""
        total_len = 0
        for _ in range(512):
            candidate_payload = _make_candidate_payload(pool_ids, desc_max)
            pool_ids_json = json.dumps(pool_ids, ensure_ascii=False)
            observation = (
                f"{user_profile_prefix}"
                f"候选 blog_id 全集（须只在此集合内排序）：{pool_ids_json}\n\n"
                f"候选微博详情：\n{json.dumps(candidate_payload, ensure_ascii=False)}"
            )
            total_len = len(instruction) + len(observation)
            if total_len <= max_prompt_chars:
                break
            if len(pool_ids) > 1:
                pool_ids.pop()
                logger.warning(f"Interest recommendation prompt over budget ({total_len} > {max_prompt_chars} chars): dropped last candidate, {len(pool_ids)} left")
                continue
            if desc_max > 200:
                desc_max = max(200, desc_max // 2)
                logger.warning(f"Interest recommendation prompt over budget: shrinking desc cap to {desc_max} chars")
                continue
            if desc_max > 80:
                desc_max = 80
                logger.warning("Interest recommendation prompt over budget: shrinking desc cap to 80 chars")
                continue
            logger.error(f"Interest recommendation: prompt still {total_len} chars (limit {max_prompt_chars}); proceeding anyway")
            break

        contents_d: Dict[str, Dict[str, Any]] = (
            contents if isinstance(contents, dict) else {}
        )

        ordered_ids: List[str] = []
        pool_set_all: Set[str] = set(pool_ids)
        try:
            # Generate recommendation
            response = await self.generate_recommendation(instruction, observation)

            # Get ranked blog ids
            raw_ids = response.get("ranked_blog_ids")
            if raw_ids is None:
                raw_ids = response.get("selected_blog_ids", [])
            if isinstance(raw_ids, list):
                for blog_id in raw_ids:
                    blog_id = str(blog_id).strip()
                    if (
                        blog_id
                        and blog_id in pool_set_all
                        and _resolve_blog(blog_id) is not None
                        and blog_id not in recommended_blog_ids
                        and blog_id not in ordered_ids
                    ):
                        ordered_ids.append(blog_id)

        except Exception as e:
            logger.warning(f"LLM interest recommendation failed: {e}")
            return {}

        # Get top blog ids
        if not ordered_ids:
            return {}
        top_ids = ordered_ids[: max(0, int(limit))]

        # Filter target blogs from ranked result
        def _blog_from_contents_only(sid: str) -> Optional[Dict[str, Any]]:
            if sid in contents_d:
                v = contents_d.get(sid)
                if isinstance(v, dict):
                    return v
            for k, v in contents_d.items():
                if str(k).strip() == sid and isinstance(v, dict):
                    return v
            return None

        out: Dict[str, Dict[str, Any]] = {}
        for sid in top_ids:
            n = _blog_from_contents_only(sid)
            if n is not None:
                out[sid] = n
        return out

    async def send_recommendation_results(self, event: Event) -> List[Event]:
        """Send recommendation results."""
        # Get current timestamp
        current_timestamp = event.timestamp

        # Get content pool
        current_blogs = {}
        if self.profile is not None:
            current_blogs = self.profile.get_data("current_blogs", {})
        if not isinstance(current_blogs, dict) or not current_blogs:
            current_blogs = getattr(event, "current_blogs", {})
        if not isinstance(current_blogs, dict) or not current_blogs:
            logger.warning("No current blogs found from profile/event")
            return []

        # Check algorithm type
        expected_type = getattr(event, "type", None)
        type_value = await self.get_data("type", "")
        exp_s = str(expected_type).strip() if expected_type is not None else ""
        got_s = str(type_value).strip() if type_value is not None else ""
        if exp_s != got_s:
            if not exp_s and got_s:
                logger.debug(f"Algorithm: event.type is empty, process with type={got_s!r}")
            else:
                raise ValueError(
                    f"Algorithm: recommendation algorithm type mismatch — "
                    f"this agent get_data('type')={got_s!r}, event.type={exp_s!r}, "
                    f"from_agent_id={getattr(event, 'from_agent_id', '')!r}"
                )

        # Get recommendation algorithm function
        algorithm_name = self.type_to_algorithm.get(type_value, self.default_algorithm)
        if algorithm_name not in self.recommendation_algorithms:
            algorithm_name = self.default_algorithm
            logger.warning(f"Unknown recommendation algorithm type '{type_value}', using default: {self.default_algorithm}")
        
        recommendation_func = self.recommendation_algorithms[algorithm_name]

        # Bernoulli sampling
        alpha, max_limit = await recommender_sampling_params(self, mode="recommendation")
        recommendation_limit = self._sample_bernoulli(alpha, max_limit=max_limit)
        logger.info(f"Algorithm recommendation_limit: {recommendation_limit}")

        # Get candidate note ids by user
        by_user: Dict[str, Any] = {}
        if self.profile is not None:
            raw_map = self.profile.get_data("candidate_blog_ids_by_user", {})
            if isinstance(raw_map, dict):
                by_user = raw_map

        requester_id = str(getattr(event, "from_agent_id", "") or "").strip()
        candidate_blog_ids: List[str] = []
        if requester_id:
            raw_list = by_user.get(requester_id)
            if raw_list is None:
                for k, v in by_user.items():
                    if str(k).strip() == requester_id and isinstance(v, list):
                        raw_list = v
                        break
            if isinstance(raw_list, list):
                candidate_blog_ids = [str(x).strip() for x in raw_list if x]

        # Get recommended blog ids
        recommended_blog_ids = list(event.recommended_blog_ids)

        # Get candidate blogs for algorithm, only original 
        n_blogs = len(current_blogs)
        current_blogs_for_algorithm: Dict[str, Any] = {
            bid: b
            for bid, b in current_blogs.items()
            if is_original_blog(b)
        }
        if len(current_blogs_for_algorithm) < n_blogs:
            logger.info(f"Algorithm: current_blogs only keep original blogs: {n_blogs} -> {len(current_blogs_for_algorithm)}")

        # Get user profile for recommendation
        user_profile_for_rec: Optional[Dict[str, Any]] = None
        raw_up = getattr(event, "user_profile", None)
        if isinstance(raw_up, dict):
            user_profile_for_rec = raw_up

        if algorithm_name == "interest":
            recommended_contents = await recommendation_func(
                recommended_blog_ids,
                current_blogs_for_algorithm,
                current_timestamp,
                recommendation_limit,
                candidate_blog_ids=candidate_blog_ids,
                user_profile=user_profile_for_rec,
            )
        else:
            recommended_contents = recommendation_func(
                recommended_blog_ids,
                current_blogs_for_algorithm,
                current_timestamp,
                recommendation_limit
            )
            
        if not recommended_contents:
            logger.debug(f"No recommended contents for user {getattr(event, 'from_agent_id', 'unknown')}")
            return []

        logger.info(
            f"Sending AlgorithmRecommendationEvent to UserAgent {event.from_agent_id}, "
            f"recommended contents: {len(recommended_contents)}"
        )

        # Fill reposted_blog field
        for blog_id, blog in recommended_contents.items():
            reposted_blog_id = blog.get("reposted_blog_id", "")
            if reposted_blog_id and reposted_blog_id in current_blogs:
                blog["reposted_blog"] = current_blogs[reposted_blog_id]

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

    # ---------- Search algorithm ----------
    @staticmethod
    def _search_blog_ids(current_blogs: Dict[str, Dict[str, Any]]) -> List[str]:
        """Get blog ids that can be used as search candidates from content pool."""
        out: List[str] = []
        for bid, blog in (current_blogs or {}).items():
            if not isinstance(blog, dict):
                continue
            sid = str(bid).strip()
            if sid:
                out.append(sid)
        return out

    @staticmethod
    def _blog_text_for_search(blog: Dict[str, Any], max_len: int = 1200) -> str:
        """Concatenate content for search, and truncate to max_len."""
        content = str(blog.get("content", "") or "")
        return content if len(content) <= max_len else content[:max_len]

    def _random_search(
        self,
        current_blogs: Dict[str, Dict[str, Any]],
        current_timestamp: float,
        limit: int,
    ) -> Dict[str, Dict[str, Any]]:
        """Random search：shuffle candidate notes and take top limit notes."""
        if not current_blogs or limit <= 0:
            return {}
        blog_ids = list(current_blogs.keys())
        random.shuffle(blog_ids)
        picked = blog_ids[:limit]
        return {bid: current_blogs[bid] for bid in picked if bid in current_blogs}

    def _keyword_search(
        self,
        current_blogs: Dict[str, Dict[str, Any]],
        current_timestamp: float,
        limit: int,
        *,
        search_query: str = "",
    ) -> Dict[str, Dict[str, Any]]:
        """Keyword search：query Jaccard similarity with note text."""
        if not current_blogs or limit <= 0:
            return {}
        cands = self._search_blog_ids(current_blogs)
        if not cands:
            return {}

        query = (search_query or "").strip()[:4000]
        if not query:
            sub_blogs = {bid: current_blogs[bid] for bid in cands if bid in current_blogs}
            return self._random_search(sub_blogs, current_timestamp, limit)

        qt = MemorySimilarityGate.tokenize(query)
        scored: List[Tuple[str, float]] = []
        for bid in cands:
            blog = current_blogs.get(bid)
            if not isinstance(blog, dict):
                continue
            text = self._blog_text_for_search(blog, max_len=2000)
            nt = MemorySimilarityGate.tokenize(text)
            if not qt or not nt:
                scored.append((bid, 0.0))
                continue
            inter = len(qt & nt)
            union = len(qt | nt)
            j = (inter / union) if union else 0.0
            scored.append((bid, j))
        random.shuffle(scored)
        scored.sort(key=lambda x: x[1], reverse=True)
        return {
            bid: current_blogs[bid]
            for bid, _ in scored[:limit]
            if bid in current_blogs
        }

    def _relevant_search_sync(
        self,
        current_blogs: Dict[str, Dict[str, Any]],
        current_timestamp: float,
        limit: int,
        *,
        search_query: str,
        embedding_config_path: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Relevant search：query and note embedding, cosine similarity sort."""
        if not current_blogs or limit <= 0:
            return {}

        # Get searchable candidate ids
        cands = self._search_blog_ids(current_blogs)
        if not cands:
            return {}

        # Get search query
        query = (search_query or "").strip()[:4000]
        if not query.strip():
            sub_blogs = {bid: current_blogs[bid] for bid in cands if bid in current_blogs}
            return self._random_search(sub_blogs, current_timestamp, limit)

        # Batch embedding, query vector vs note vector cosine similarity
        try:
            base_url, model_name = load_embedding_config(embedding_config_path)
            texts = [query[:2000]]
            for bid in cands:
                blog = current_blogs.get(bid)
                texts.append(
                    self._blog_text_for_search(blog, max_len=800)
                    if isinstance(blog, dict)
                    else ""
                )
            vecs = get_embeddings(base_url, model_name, texts)
            if not vecs or len(vecs) != len(texts):
                raise ValueError("embedding return length does not match input")
            vq = vecs[0]
            scored: List[Tuple[str, float]] = []
            for i, bid in enumerate(cands):
                sim = float(cosine_similarity(vq, vecs[i + 1]))
                scored.append((bid, sim))
            random.shuffle(scored)
            scored.sort(key=lambda x: x[1], reverse=True)
            top = scored[:limit]
            return {bid: current_blogs[bid] for bid, _ in top if bid in current_blogs}
        except Exception as e:
            logger.warning(f"Relevant search embedding failed, fallback to keyword: {e}")
            return self._keyword_search(
                current_blogs,
                current_timestamp,
                limit,
                search_query=query,
            )

    async def _relevant_search(
        self,
        current_blogs: Dict[str, Dict[str, Any]],
        current_timestamp: float,
        limit: int,
        *,
        search_query: str,
        embedding_config_path: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Relevant search：query and note embedding, cosine similarity sort."""
        return await asyncio.to_thread(
            self._relevant_search_sync,
            current_blogs,
            current_timestamp,
            limit,
            search_query=search_query,
            embedding_config_path=embedding_config_path,
        )

    async def send_search_results(self, event: Event) -> List[Event]:
        """Generate search results."""
        # Get current timestamp
        current_timestamp = event.timestamp

        # Get content pool        
        current_blogs = {}
        if self.profile is not None:
            current_blogs = self.profile.get_data("current_blogs", {})
        if not isinstance(current_blogs, dict) or not current_blogs:
            current_blogs = getattr(event, "current_blogs", {})
        if not isinstance(current_blogs, dict) or not current_blogs:
            logger.warning("No current blogs found from profile/event")
            return []

        # Check search algorithm type
        expected_type = getattr(event, "type", None)
        type_value = await self.get_data("type", "")
        exp_s = str(expected_type).strip() if expected_type is not None else ""
        got_s = str(type_value).strip() if type_value is not None else ""
        if exp_s != got_s:
            if not exp_s and got_s:
                logger.debug(f"Algorithm: event.type is empty, process with type={got_s!r}")
            else:
                raise ValueError(
                    f"Algorithm: search algorithm type mismatch — "
                    f"this agent get_data('type')={got_s!r}, event.type={exp_s!r}, "
                    f"from_agent_id={getattr(event, 'from_agent_id', '')!r}"
                )

        # Map type value to search algorithm name
        algorithm_name = self.type_to_search.get(type_value, self.default_search)
        if algorithm_name not in self.search_algorithms:
            algorithm_name = self.default_search
            logger.warning(f"Unknown search algorithm type '{type_value}', using default: {self.default_search}")

        search_func = self.search_algorithms[algorithm_name]

        # Bernoulli sampling
        alpha, max_limit = await recommender_sampling_params(self, mode="search")
        search_limit = self._sample_bernoulli(alpha, max_limit=max_limit)
        top_k = max(1, min(max_limit, search_limit))
        logger.info(f"Algorithm search algorithm={algorithm_name}, max_limit={max_limit}, sampled search_limit={search_limit}")

        # Get search query
        search_query = str(getattr(event, "search_query", "") or "")
        cfg = MemorySimilarityGate.load_config(await memory_similarity_gate_params(self))

        search_results: Dict[str, Dict[str, Any]] = {}
        if algorithm_name == "relevant":
            search_results = await search_func(
                current_blogs,
                current_timestamp,
                search_limit,
                search_query=search_query,
                embedding_config_path=cfg.embedding_config_path,
            )

        if not search_results:
            logger.debug(f"No search results for user {getattr(event, 'from_agent_id', 'unknown')}")
            return []

        logger.info(
            f"Sending SearchResultEvent (search) to UserAgent {event.from_agent_id}, "
            f"search results: {len(search_results)}"
        )

        # Fill reposted_blog field
        for blog_id, blog in search_results.items():
            reposted_blog_id = blog.get("reposted_blog_id", "")
            if reposted_blog_id and reposted_blog_id in current_blogs:
                blog["reposted_blog"] = current_blogs[reposted_blog_id]

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