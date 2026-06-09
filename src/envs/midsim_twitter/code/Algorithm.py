from typing import Any, List, Optional, Callable, Dict, Set, Tuple
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
from .embedding_client import (
    cosine_similarity,
    get_embeddings,
    load_embedding_config,
)
from onesim.utils.midsim_params import recommender_sampling_params, interest_recommendation_candidate_limits, step15_params
from .user_agent_gates import MemorySimilarityGate
from .utils import (
    enrich_tweet_quote_reply_chain,
    format_historical_summary,
    format_popularity_distribution,
    is_original_tweet,
    to_float,
    tweet_ref_key,
)

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
        self.register_event("StartEvent", "update_current_tweets")
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

    async def update_current_tweets(self, event: Event) -> None:
        """Update current tweets."""
        current_tweets = event.current_tweets
        if not isinstance(current_tweets, dict) or not current_tweets:
            logger.warning("No current tweets found")
        self.profile.update_data("current_tweets", current_tweets)

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
    _RANDOM_REC_MAU_DEFAULT = 557_000_000   # Default Monthly Active Users (MAU)
    _RANDOM_REC_AVG_POSTS_PER_USER_MONTH_DEFAULT = 74.31    # Default avg posts per user per month

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

    def _random_rec_per_tweet_weight(self) -> float:
        mau, avg_posts = self._random_rec_mau_avg_posts()
        return (1.0 / mau) * avg_posts

    def _random_recommendation(self, recommended_tweet_ids: List[str], contents: Dict[str, Dict[str, Any]], current_timestamp: float, limit: int = 10) -> Dict[str, Dict[str, Any]]:
        """Random recommendation, sample with equal probability (1/(MAU×avg)) from candidate pool."""
        if not contents:
            return {}
        tweet_ids = list(contents.keys())
        if not tweet_ids:
            return {}
            
        # Filter out recommended tweets
        tweet_ids = [tweet_id for tweet_id in tweet_ids if tweet_id not in recommended_tweet_ids]
        if not tweet_ids:
            return {}
        
        # Count denominator, p = 1 / (MAU × avg)
        mau, avg_posts = self._random_rec_mau_avg_posts()
        denom = mau * avg_posts
        if denom <= 0:
            p_hit = 0.0
        else:
            p_hit = min(1.0, 1.0 / denom)

        random.shuffle(tweet_ids)
        log_each = bool(
            self.profile is not None
            and self.profile.get_data("random_rec_log_per_tweet_prob_check", False)
        )

        selected: List[str] = []
        for tid in tweet_ids:
            if len(selected) >= limit:
                break
            ok = random.random() < p_hit
            if log_each:
                logger.debug("random_rec tweet {}: Bernoulli(p=1/(MAU×avg)={:.6e}) => {}", tid, p_hit, "hit" if ok else "miss")
            if ok:
                selected.append(tid)

        w = (avg_posts / mau) if mau > 0 else 0.0
        logger.debug("random recommendation: MAU={:.4g} avg_posts={:.6g} => p=1/(MAU×avg)={:.6e}; candidate N={} limit={} =>本轮收录 {} 条；w=(1/MAU)×avg={:.6e}", mau, avg_posts, p_hit, len(tweet_ids), limit, len(selected), w)
        return {tweet_id: contents[tweet_id] for tweet_id in selected}

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

    def _calculate_popularity(self, tweet: Dict[str, Any]) -> float:
        """Calculate tweet popularity: use SimEnv's popularity (reply+quote+retweet count in sliding window) if available, otherwise use repost_count."""
        reply_count = tweet.get("reply_count", 0)
        retweet_count = tweet.get("retweet_count", 0)
        quote_count = tweet.get("quote_count", 0)
        popularity = float(reply_count) + float(retweet_count) + float(quote_count)
        return popularity

    def _popularity_recommendation(self, recommended_tweet_ids: List[str], contents: Dict[str, Dict[str, Any]], current_timestamp: float, limit: int = 10) -> Dict[str, Dict[str, Any]]:
        """Popularity recommendation: mix candidate note popularity with experience popularity distribution, and take the high-popularity notes."""
        if not contents:
            return {}
        tweet_ids = list(contents.keys())
        if not tweet_ids:
            return {}

        tweet_ids = [tweet_id for tweet_id in tweet_ids if tweet_id not in recommended_tweet_ids]
        if not tweet_ids:
            return {}

        # Get popularity distribution map
        dist_map = self._get_popularity_distribution_map()
        if not dist_map:
            tweets_with_popularity: List[Tuple[str, Dict[str, Any], float]] = []
            for tweet_id in tweet_ids:
                tweet = contents[tweet_id]
                if not isinstance(tweet, dict):
                    continue
                popularity = self._calculate_popularity(tweet)
                tweets_with_popularity.append((tweet_id, tweet, popularity))
            random.shuffle(tweets_with_popularity)
            tweets_with_popularity.sort(key=lambda x: x[2], reverse=True)
            return {tweet_id: tweet for tweet_id, tweet, _ in tweets_with_popularity[:limit]}

        # Mix candidate blogs with experience popularity distribution, and sort by popularity
        buckets: Dict[int, List[Optional[str]]] = defaultdict(list)
        for pc, cnt in dist_map.items():
            try:
                n = int(cnt)
            except (TypeError, ValueError):
                continue
            for _ in range(max(0, n)):
                buckets[pc].append(None)
        for tweet_id in tweet_ids:
            tweet = contents[tweet_id]
            if not isinstance(tweet, dict):
                continue
            p = int(self._calculate_popularity(tweet))
            buckets[p].append(tweet_id)

        merged_ids: List[Optional[str]] = []
        for pc in sorted(buckets.keys(), reverse=True):
            row = buckets[pc]
            random.shuffle(row)
            merged_ids.extend(row)

        # Get top tweets by popularity
        top_slice = merged_ids[: max(0, limit)]
        chosen_ids = [x for x in top_slice if x is not None]
        return {tid: contents[tid] for tid in chosen_ids if tid in contents}

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
        keys = ( "id", "nickname", "gender", "interest_tags", "description", "historical_summary")
        out: Dict[str, Any] = {}
        for k in keys:
            if k not in user_profile:
                continue
            v = user_profile[k]
            if k == "historical_summary":
                v = format_historical_summary(v)
            elif k == "description" and isinstance(v, str) and len(v) > 400:
                v = v[:50] + "…"
            out[k] = v
        return out

    async def _interest_recommendation(
        self,
        recommended_tweet_ids: List[str],
        contents: Dict[str, Dict[str, Any]],
        current_timestamp: float,
        limit: int = 10,
        candidate_tweet_ids: Optional[List[str]] = None,
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

        def _resolve_tweet(tid: str) -> Optional[Dict[str, Any]]:
            sid = str(tid).strip()
            n = interest_pool.get(sid)
            if isinstance(n, dict):
                return n
            c = contents.get(sid) if isinstance(contents, dict) else None
            return c if isinstance(c, dict) else None

        # Get candidate tweet ids
        candidate_tweet_ids = list(candidate_tweet_ids) if candidate_tweet_ids else []
        interest_k, target_k = await interest_recommendation_candidate_limits(self)

        ids_for_user: List[str] = []
        if candidate_tweet_ids:
            ids_for_user = [str(x).strip() for x in candidate_tweet_ids if x]
        ids_for_user = [
            tid for tid in ids_for_user
            if _resolve_tweet(tid) is not None
            and tid not in recommended_tweet_ids
        ]
        random.shuffle(ids_for_user)
        pool_sample: List[str] = ids_for_user[:interest_k]
        pool_set: Set[str] = set(pool_sample)

        # Construct current feed tweet ids
        feed_tweet_ids = list(contents.keys()) if contents else []
        feed_tweet_ids = [tweet_id for tweet_id in feed_tweet_ids if tweet_id not in recommended_tweet_ids]
        feed_eligible = [tweet_id for tweet_id in feed_tweet_ids if tweet_id not in pool_set]
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
            for tweet_id in ids:
                tweet = _resolve_tweet(tweet_id)
                if not isinstance(tweet, dict):
                    continue
                popularity = self._calculate_popularity(tweet)
                out.append({
                    "tweet_id": tweet_id,
                    "content": (tweet.get("content", "") or "")[:desc_max],
                })
            return out

        # Construct prompt and avoid context overflow
        max_prompt_chars = int(
            os.environ.get("ONESIM_RECOMMENDER_INTEREST_MAX_PROMPT_CHARS", "48000")
        )
        profile_compact = self._compact_user_profile_for_interest_rec(user_profile)
        if profile_compact:
            user_profile_prefix = (
                "Requesting user profile (for interest-based personalized ranking; prioritize alignment with "
                "themes and intent reflected in interest_tags, description, historical_summary, etc.):\n"
                f"{json.dumps(profile_compact, ensure_ascii=False)}\n\n"
            )
        else:
            user_profile_prefix = ""
            
        instruction = f"""
        You are a recommendation-system assistant. Given the full candidate tweet_id set below, output one **complete ranked list** (highest to lowest interest) and return strict JSON only.

        [Goal — align with user interests]
        1) **Primary ranking signal (highest weight)**: Using **interest_tags, description, historical_summary** from the user profile, estimate how well each tweet matches the user on **topic domain, role/identity, long-term concerns, and expression style**; rank higher content the user is more likely to open, dwell on, reply to, or retweet.
        2) **Semantic and tag matching**: Among title, truncated description (desc), and tags_list, prefer tweets that **directly match or strongly relate** to explicit interest tags or recurring concerns in the profile summary over weakly associated content.
        3) **Secondary signal (tie-break when primary scores are close)**: Whether information is complete and readable (title + desc convey the topic; not empty clickbait), and whether tags align with the body; when relevance is similar, prefer tweets that are **more specific and informative** over vague or repetitive ones.
        4) **Light diversity constraint**: After primary ranking, if adjacent tweets are nearly identical in topic, you may slightly separate them so the top of the list covers a broader interest surface (still obey the hard constraints below: full coverage, no dropped ids).

        Return strict JSON:
        {{
        "ranked_tweet_ids": ["id1", "id2", ...],
        "per_tweet_category": ["category1", "category2", ...]
        }}

        Field notes:
        - per_tweet_category: **same length** as ranked_tweet_ids; item i is a **short category label** for the i-th ranked tweet (about 2–12 words), e.g. "Academia/Submission", "Life/Food", "Emotion/Growth", reflecting topic or intent; when a user profile is present, you may hint which profile interest it aligns with—do not write long rationales.
        Constraints:
        1) Every id must come from the candidate tweet_id list below and **cover the full set**;
        2) No duplicates;
        3) per_tweet_category must match ranked_tweet_ids in count and one-to-one order;
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
                f"Full candidate tweet_id set (rank only within this set): {pool_ids_json}\n\n"
                f"Candidate tweet details:\n{json.dumps(candidate_payload, ensure_ascii=False)}"
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
            raw_ids = response.get("ranked_tweet_ids")
            if raw_ids is None:
                raw_ids = response.get("selected_tweet_ids", [])
            if isinstance(raw_ids, list):
                for tweet_id in raw_ids:
                    tweet_id = str(tweet_id).strip()
                    if (
                        tweet_id
                        and tweet_id in pool_set_all
                        and _resolve_tweet(tweet_id) is not None
                        and tweet_id not in recommended_tweet_ids
                        and tweet_id not in ordered_ids
                    ):
                        ordered_ids.append(tweet_id)

        except Exception as e:
            logger.warning(f"LLM interest recommendation failed: {e}")
            return {}

        # Get top tweet ids
        if not ordered_ids:
            return {}
        top_ids = ordered_ids[: max(0, int(limit))]

        # Filter target tweets from ranked result
        def _tweet_from_contents_only(sid: str) -> Optional[Dict[str, Any]]:
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
            n = _tweet_from_contents_only(sid)
            if n is not None:
                out[sid] = n
        return out

    async def send_recommendation_results(self, event: Event) -> List[Event]:
        """Send recommendation results."""
        # Get current timestamp
        current_timestamp = event.timestamp

        # Get current tweets
        logger.info(f"Algorithm {self.profile_id} start getting previous and current tweets")
        
        current_tweets = {}
        if self.profile is not None:
            current_tweets = self.profile.get_data("current_tweets", {})
        if not isinstance(current_tweets, dict) or not current_tweets:
            current_tweets = getattr(event, "current_tweets", {})
        if not isinstance(current_tweets, dict) or not current_tweets:
            logger.warning("No current tweets found from profile/event")
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
            raw_map = self.profile.get_data("candidate_tweet_ids_by_user", {})
            if isinstance(raw_map, dict):
                by_user = raw_map

        requester_id = str(getattr(event, "from_agent_id", "") or "").strip()
        candidate_tweet_ids: List[str] = []
        if requester_id:
            raw_list = by_user.get(requester_id)
            if raw_list is None:
                for k, v in by_user.items():
                    if str(k).strip() == requester_id and isinstance(v, list):
                        raw_list = v
                        break
            if isinstance(raw_list, list):
                candidate_tweet_ids = [str(x).strip() for x in raw_list if x]

        # Get recommended tweet ids
        recommended_tweet_ids = list(event.recommended_tweet_ids)

        # Get candidate tweets for algorithm, only original tweets
        n_tweets = len(current_tweets)
        current_tweets_for_algorithm: Dict[str, Any] = {
            tid: t
            for tid, t in current_tweets.items()
            if is_original_tweet(t)
        }
        if len(current_tweets_for_algorithm) < n_tweets:
            logger.info(f"Algorithm: current_tweets only keep original tweets: {n_tweets} -> {len(current_tweets_for_algorithm)}")

        # Get user profile for recommendation
        user_profile_for_rec: Optional[Dict[str, Any]] = None
        raw_up = getattr(event, "user_profile", None)
        if isinstance(raw_up, dict):
            user_profile_for_rec = raw_up

        if algorithm_name == "interest":
            recommended_contents = await recommendation_func(
                recommended_tweet_ids,
                current_tweets_for_algorithm,
                current_timestamp,
                recommendation_limit,
                candidate_tweet_ids=candidate_tweet_ids,
                user_profile=user_profile_for_rec,
            )
        else:
            recommended_contents = recommendation_func(
                recommended_tweet_ids,
                current_tweets,
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

        # Fill tweet field (quoted_tweet, replied_tweet, retweeted_tweet)
        for tweet_id, tweet in recommended_contents.items():
            tw_key = tweet_ref_key(tweet_id)
            recommended_contents[tweet_id] = enrich_tweet_quote_reply_chain(
                dict(tweet), current_tweets, tweet_ref=tw_key
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

    # ---------- Search algorithm ----------
    @staticmethod
    def _search_tweet_ids(current_tweets: Dict[str, Dict[str, Any]]) -> List[str]:
        """Get tweet ids that can be used as search candidates from content pool."""
        out: List[str] = []
        for tid, tweet in (current_tweets or {}).items():
            if not isinstance(tweet, dict):
                continue
            sid = str(tid).strip()
            if sid:
                out.append(sid)
        return out

    @staticmethod
    def _tweet_text_for_search(tweet: Dict[str, Any], max_len: int = 1200) -> str:
        """Concatenate content for search, and truncate to max_len."""
        content = str(tweet.get("content", "") or "")
        return content if len(content) <= max_len else content[:max_len]

    def _random_search(
        self,
        current_tweets: Dict[str, Dict[str, Any]],
        current_timestamp: float,
        limit: int,
    ) -> Dict[str, Dict[str, Any]]:
        """Random search：shuffle candidate notes and take top limit notes."""
        if not current_tweets or limit <= 0:
            return {}
        tweet_ids = list(current_tweets.keys())
        random.shuffle(tweet_ids)
        picked = tweet_ids[:limit]
        return {tid: current_tweets[tid] for tid in picked if tid in current_tweets}

    def _keyword_search(
        self,
        current_tweets: Dict[str, Dict[str, Any]],
        current_timestamp: float,
        limit: int,
        *,
        search_query: str = "",
    ) -> Dict[str, Dict[str, Any]]:
        """Keyword search：query Jaccard similarity with note text."""
        if not current_tweets or limit <= 0:
            return {}
        cands = self._search_tweet_ids(current_tweets)
        if not cands:
            return {}

        query = (search_query or "").strip()[:4000]
        if not query:
            sub_tweets = {tid: current_tweets[tid] for tid in cands if tid in current_tweets}
            return self._random_search(sub_tweets, current_timestamp, limit)

        qt = MemorySimilarityGate.tokenize(query)
        scored: List[Tuple[str, float]] = []
        for tid in cands:
            tweet = current_tweets.get(tid)
            if not isinstance(tweet, dict):
                continue
            text = self._tweet_text_for_search(tweet, max_len=2000)
            nt = MemorySimilarityGate.tokenize(text)
            if not qt or not nt:
                scored.append((tid, 0.0))
                continue
            inter = len(qt & nt)
            union = len(qt | nt)
            j = (inter / union) if union else 0.0
            scored.append((tid, j))
        random.shuffle(scored)
        scored.sort(key=lambda x: x[1], reverse=True)
        return {
            tid: current_tweets[tid]
            for tid, _ in scored[:limit]
            if tid in current_tweets
        }

    def _relevant_search_sync(
        self,
        current_tweets: Dict[str, Dict[str, Any]],
        current_timestamp: float,
        limit: int,
        *,
        search_query: str,
        embedding_config_path: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Relevant search：query and note embedding, cosine similarity sort."""
        if not current_tweets or limit <= 0:
            return {}

        # Get searchable candidate ids
        cands = self._search_tweet_ids(current_tweets)
        if not cands:
            return {}

        # Get search query
        query = (search_query or "").strip()[:4000]
        if not query.strip():
            sub_tweets = {tid: current_tweets[tid] for tid in cands if tid in current_tweets}
            return self._random_search(sub_tweets, current_timestamp, limit)

        # Batch embedding, query vector vs note vector cosine similarity
        try:
            base_url, model_name = load_embedding_config(embedding_config_path)
            texts = [query[:2000]]
            for tid in cands:
                tweet = current_tweets.get(tid)
                texts.append(
                    self._tweet_text_for_search(tweet, max_len=800)
                    if isinstance(tweet, dict)
                    else ""
                )
            vecs = get_embeddings(base_url, model_name, texts)
            if not vecs or len(vecs) != len(texts):
                raise ValueError("embedding return length does not match input")
            vq = vecs[0]
            scored: List[Tuple[str, float]] = []
            for i, tid in enumerate(cands):
                sim = float(cosine_similarity(vq, vecs[i + 1]))
                scored.append((tid, sim))
            random.shuffle(scored)
            scored.sort(key=lambda x: x[1], reverse=True)
            top = scored[:limit]
            return {tid: current_tweets[tid] for tid, _ in top if tid in current_tweets}
        except Exception as e:
            logger.warning(f"Relevant search embedding failed, fallback to keyword: {e}")
            return self._keyword_search(
                current_tweets,
                current_timestamp,
                limit,
                search_query=query,
            )

    async def _relevant_search(
        self,
        current_tweets: Dict[str, Dict[str, Any]],
        current_timestamp: float,
        limit: int,
        *,
        search_query: str,
        embedding_config_path: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Relevant search：query and note embedding, cosine similarity sort."""
        return await asyncio.to_thread(
            self._relevant_search_sync,
            current_tweets,
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
        current_tweets = {}
        if self.profile is not None:
            current_tweets = self.profile.get_data("current_tweets", {})
        if not isinstance(current_tweets, dict) or not current_tweets:
            current_tweets = getattr(event, "current_tweets", {})
        if not isinstance(current_tweets, dict) or not current_tweets:
            logger.warning("No current tweets found from profile/event")
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
        cfg = MemorySimilarityGate.load_config(await step15_params(self))

        search_results: Dict[str, Dict[str, Any]] = {}
        if algorithm_name == "relevant":
            search_results = await search_func(
                current_tweets,
                current_timestamp,
                top_k,
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

        # Fill tweet field (quoted_tweet, replied_tweet, retweeted_tweet)
        for tweet_id, tweet in search_results.items():
            tw_key = tweet_ref_key(tweet_id)
            search_results[tweet_id] = enrich_tweet_quote_reply_chain(
                dict(tweet), current_tweets, tweet_ref=tw_key
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