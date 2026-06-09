from typing import Any, List, Optional, Dict, Set, Tuple, Union
import json
import asyncio
import os
import re
import time

# SimEnv uses global lock to serialize environment updates, queueing may exceed 30s in high concurrency
_ENV_ASYNC_OP_TIMEOUT = float(os.environ.get("ONESIM_ENV_ASYNC_OP_TIMEOUT", "120"))

from grpc import Future
from loguru import logger
from onesim.models import JsonBlockParser
from onesim.agent import GeneralAgent
from onesim.profile import AgentProfile
from onesim.memory import MemoryStrategy
from onesim.planning import PlanningBase
from onesim.events import Event
from onesim.relationship import RelationshipManager
from onesim.distribution.distributed_lock import get_lock  
from onesim.utils.midsim_params import (
    user_activity_remap,
    user_algorithmic_recommendation_prob,
    user_attention_budget,
    user_default_algorithm_types,
    user_default_search_types,
    user_social_feed_last_login_cap_days,
    user_social_recommendation_prob,
)
from .events import *
import random
import math

from .user_agent_gates import UserAgentGates
from .metrics.channel_snapshots import (
    record_mentioned_blog_ids_by_channel,
    record_recommendations_by_source_step,
)
from .utils import (
    format_historical_summary,
    generate_blog_timestamp,
    generate_repost_id,
    is_repost_of_other_blog,
    time_to_ms,
    time_to_sec,
)

class UserAgent(GeneralAgent):
    def __init__(self,
                 sys_prompt: str | None = None,
                 model_config_name: str = None,
                 event_bus_queue: asyncio.Queue = None,
                 profile: AgentProfile=None,
                 memory: MemoryStrategy=None,
                 planning: PlanningBase=None,
                 relationship_manager: RelationshipManager=None) -> None:
        super().__init__(sys_prompt, model_config_name, event_bus_queue, profile, memory, planning, relationship_manager)
        self.register_event("StartEvent", "get_exposure_stream")  
        self.register_event("StartEvent", "generate_memory_from_own_blogs")
        self.register_event("SocialRecommendationEvent", "receive_recommendation")
        self.register_event("AlgorithmRecommendationEvent", "receive_recommendation")
        self.register_event("SearchResultEvent", "receive_recommendation")
        self.register_event("KeepFollowingEvent", "receive_recommendation")
        self.register_event("MentionEvent", "handle_mention")
        self.register_event("AddRepostResponseEvent", "handle_add_repost_response")
        self.register_event("MentionPoolUpdateResponseEvent", "handle_update_mention_pool_response")
        self._repost_add_futures: Dict[str, Future] = {}
        self._mention_pool_update_futures: Dict[str, Future] = {}
        self._login_lock = asyncio.Lock()  

        # Map algorithm type to recommender agent ID
        self.recommender_map: Dict[str, str] = {
            "Random Recommendation": "recomment_agent_0001",
            "Popularity Recommendation": "recomment_agent_0002",
            "Interest Recommendation": "recomment_agent_0003",
        }
        self.search_map: Dict[str, str] = {
            "Relevant Search": "search_agent_0001",
        }
        self._recommendation_earliest_post_anchor_ms: Optional[float] = None

    async def _is_official_by_agent_field(self) -> Optional[bool]:
        """Check if the account is official by agent field."""
        is_official = self.profile.get_data("is_official", False)
        return True if is_official else False

    async def _should_activate_this_round(self, current_step: int, max_step: int) -> bool:
        """Decide whether to activate based on activity_level and power law upper bound: p_t = min(activity_level, upper_t)."""
        async with self._login_lock:
            flag = self.profile.get_data("login", -1) if self.profile else -1
            if flag != -1:
                return (flag == 1)

            raw_a = self.profile.get_data("activity_level", 0.0) if self.profile else 0.0
            try:
                activity_level = float(raw_a)
            except (TypeError, ValueError):
                activity_level = 0.0
            out_min, out_max = await user_activity_remap(self)
            clamped = max(0.0, min(1.0, activity_level))
            activity_level = out_min + (out_max - out_min) * clamped
            p_t = activity_level

            activated = random.random() < p_t
            if self.profile:
                self.profile.update_data("login", 1 if activated else 0)

            return activated

    async def update_env_mention_pool(self, key: str, value: Any, parent_event_id: Optional[str] = None) -> bool:
        """Update the mention_pool in the environment (using distributed lock)"""
        # Convert mentioner_id.comment_id to the full key format: mention_pool.mentioner_id.comment_id
        full_key = f"mention_pool.{key}"
        lock_key = key.split(".")[0]
        
        # Create a unique request ID
        request_id = f"agent_env_update_mention_pool_req_{time.time()}_{id(self)}"
        future = asyncio.Future()
        self._mention_pool_update_futures[request_id] = future

        # Create a mention_pool update event
        mention_pool_update_event = MentionPoolUpdateEvent(
            from_agent_id=self.profile_id,  
            to_agent_id="ENV",              
            source_type="AGENT",            
            target_type="ENV",              
            key=full_key,                   
            value=value,                    
            request_id=request_id,          
            parent_event_id=parent_event_id 
        )

        # Get the distributed lock for this key
        lock_id = f"env_mention_pool_update_lock_{lock_key}"
        lock = await get_lock(lock_id)

        try:
            async with lock:
                from onesim.events import get_event_bus
                event_bus = get_event_bus()
                await event_bus.dispatch_event(mention_pool_update_event)

            try:
                if hasattr(self, '_sync_event'):
                    await asyncio.wait_for(self._sync_event.wait(), timeout=30.0)
                    return await future
                return await asyncio.wait_for(future, timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning(f"Environment mention_pool update timeout: {key}")
                self._mention_pool_update_futures.pop(request_id, None)
                return False
            except Exception as e:
                logger.error(f"Error updating environment mention_pool: {e}")
                self._mention_pool_update_futures.pop(request_id, None)
                return False
        except Exception as e:
            logger.error(f"Error getting environment mention_pool update lock: {e}")
            return False

    async def handle_update_mention_pool_response(self, event: MentionPoolUpdateResponseEvent) -> None:
        """Handle the incoming mention_pool update response event"""
        # Check if waiting for this response
        if event.request_id in self._mention_pool_update_futures:
            future = self._mention_pool_update_futures.pop(event.request_id)

            if not future.done():
                if event.success:
                    future.set_result(True)
                else:
                    future.set_exception(ValueError(event.error or "Unknown error"))

            # If there is a sync event, set it
            if hasattr(self, '_sync_event'):
                self._sync_event.set()
                self._sync_event.clear()

    async def get_exposure_stream(self, event: Event) -> List[Event]:
        """Get exposure (including follow stream and algorithm stream)"""
        current_step = getattr(event, "current_step", 1)
        max_step = getattr(event, "max_step", 8)
        current_ts = int(event.timestamp) if getattr(event, "timestamp", None) is not None else 0

        # Check user activation status for this round
        if not await self._should_activate_this_round(current_step, max_step):
            logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} is not activated in this round")
            return []
        logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} is activated in this round")

        if (await self._is_official_by_agent_field()) is True:
            logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} detected official account by agent field, skip exposure stream")
            return []

        events: List[Event] = []

        # Get current available blogs list
        current_blogs = getattr(event, "current_blogs", None) or {}
        if not isinstance(current_blogs, dict):
            current_blogs = {}

        # Algorithm recommendation stream
        # Get specified algorithm types
        fixed_algorithm_types = await user_default_algorithm_types(self)
        if not isinstance(fixed_algorithm_types, list) or not fixed_algorithm_types:
            raise ValueError("default_algorithm_types must be a non-empty list")
        allowed_algorithm_types = set(self.recommender_map.keys())

        # Get recommended note IDs
        recommended_blog_ids = set(self.profile.get_data("recommended_blog_ids", [])) if self.profile else set()

        # Get user profile
        profile_payload = {}
        if self.profile is not None:
            try:
                profile_payload = dict(self.profile.get_profile(include_private=True) or {})
            except Exception:
                logger.warning("Failed to serialize profile via get_profile(), fallback to empty payload.")
        
        # Iterate through all specified algorithm types and send events to recommender system
        _algorithm_request_prob = await user_algorithmic_recommendation_prob(self)
        for fixed_algorithm_type in fixed_algorithm_types:
            if fixed_algorithm_type not in allowed_algorithm_types:
                raise ValueError(f"Invalid algorithm type '{fixed_algorithm_type}'. Allowed types: {sorted(allowed_algorithm_types)}")
            mapped_id = self.recommender_map.get(fixed_algorithm_type, "")
            if not mapped_id:
                raise ValueError(f"No recommender agent mapped for algorithm type '{fixed_algorithm_type}'. Current mapping keys: {sorted(self.recommender_map.keys())}")
                continue
            if random.random() > _algorithm_request_prob:
                logger.debug(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} skip GetAlgorithmRecomendationEvent for {fixed_algorithm_type!r} (p={_algorithm_request_prob:.0%} not drawn)")
                continue
            events.append(
                GetAlgorithmRecomendationEvent(
                    from_agent_id=self.profile_id,
                    to_agent_id=mapped_id,
                    timestamp=int(getattr(event, "timestamp", 0) or 0),
                    timestamp_duration=int(getattr(event, "timestamp_duration", 0) or 0),
                    current_step=current_step,
                    max_step=max_step,
                    user_profile=profile_payload,
                    current_blogs=current_blogs,
                    recommended_blog_ids=recommended_blog_ids,
                    algorithm_type=fixed_algorithm_type,
                )
            )

        # Get mentions and send MentionEvent (if exceeds attention_budget, randomly retain and delete other items in the pool)
        mentions = getattr(event, "mentions", None) or {}
        if not isinstance(mentions, dict):
            mentions = {}
        attention_budget = await user_attention_budget(self)
        if len(mentions) > attention_budget:
            my_id = await self.get_data("id")
            all_keys = list(mentions.keys())
            sampled_keys = random.sample(all_keys, attention_budget)
            logger.info(
                f"Step {current_step}/{max_step}: UserAgent {self.profile_id} "
                f"mentions count {len(all_keys)} > {attention_budget}, sampling {attention_budget} for MentionEvent; dropping others from mention_pool"
            )
            for k in all_keys:
                if k in sampled_keys:
                    continue
                success = await self.update_env_mention_pool(
                    f"{my_id}.{k}",
                    {
                        "action": "delete",
                        "mention_message": None,
                    },
                )
                if not success:
                    logger.error(f"Failed to delete non-sampled mention {k} for user {my_id} in mention_pool")
            mentions = {k: mentions[k] for k in sampled_keys}
        if 0 < len(mentions) <= attention_budget:
            enriched_mentions: Dict[str, Any] = {}
            for mention_key, mention_message in mentions.items():
                if not isinstance(mention_message, dict):
                    enriched_mentions[mention_key] = mention_message
                    continue
                mm = dict(mention_message)
                blog = mm.get("blog")
                if isinstance(blog, dict):
                    bid = blog.get("blog_id") or blog.get("id")
                    if bid is not None:
                        bid_key = str(bid).strip()
                        pool_blog = current_blogs.get(bid_key) if isinstance(current_blogs, dict) else None
                        if isinstance(pool_blog, dict):
                            pb = dict(pool_blog)
                            reposted_blog_id = pb.get("reposted_blog_id", "")
                            if reposted_blog_id and reposted_blog_id in current_blogs:
                                pb["reposted_blog"] = current_blogs[reposted_blog_id]
                            mm["blog"] = pb
                enriched_mentions[mention_key] = mm
            mentions = enriched_mentions

            logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} send MentionEvent, length of mentions: {len(mentions)}")
            events.append(MentionEvent(
                from_agent_id=self.profile_id,
                to_agent_id=self.profile_id,
                timestamp=current_ts,
                timestamp_duration=int(getattr(event, "timestamp_duration", 0) or 0),
                current_step=current_step,
                max_step=max_step,
                mentions=mentions,
            ))

        # Handle keep_following_note_ids list: if previous round decided to keep, re-add it to the recommendation in this round
        keep_following_blogs = {}
        if self.profile and self.profile.get_data("keep_following_blog_ids", []) is not None:
            keep_ids = self.profile.get_data("keep_following_blog_ids", []) or []
            if isinstance(keep_ids, (list, tuple)) and keep_ids:
                readded = 0
                for keep_blog_id in keep_ids:
                    if keep_blog_id in keep_following_blogs:
                        continue
                    blog = current_blogs.get(keep_blog_id)
                    if not isinstance(blog, dict):
                        continue
                    reposted_blog_id = blog.get("reposted_blog_id", "")
                    if reposted_blog_id and reposted_blog_id in current_blogs:
                        blog["reposted_blog"] = current_blogs[reposted_blog_id]
                    keep_following_blogs[keep_blog_id] = blog
                    readded += 1
                if readded > 0:
                    logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} re-added keep_following blogs: {readded}")
                self.profile.update_data("keep_following_blog_ids", [])
                
        if len(keep_following_blogs) > 0:
            events.append(KeepFollowingEvent(
                from_agent_id=self.profile_id,
                to_agent_id=self.profile_id,
                timestamp=current_ts,
                timestamp_duration=int(getattr(event, "timestamp_duration", 0) or 0),
                current_step=current_step,
                max_step=max_step,
                recommendations=keep_following_blogs,
            ))

        # Social recommendation stream
        follow_ids = await self.get_data("follow_ids", [])
        follow_set = set(follow_ids) if isinstance(follow_ids, (list, tuple)) else set()

        # Get notes in the follow list that have not been viewed
        last_login = 0
        _login_cap_days = await user_social_feed_last_login_cap_days(self)
        if self.profile:
            raw = self.profile.get_data("last_login_timestamp")
            logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} last_login_timestamp: {raw}")
            if raw is not None and isinstance(raw, (int, float)) and int(raw) > 0:
                last_login = int(raw)
                if current_ts > 0 and _login_cap_days > 0:
                    lower = current_ts - int(_login_cap_days * 86400)
                    if last_login < lower:
                        last_login = lower
            else:
                last_login = 0

        step_duration = int(getattr(event, "timestamp_duration", 0) or 0)
        window_end = current_ts + step_duration

        # Notes that have been recommended are not included in the follow stream
        seen_rec_ids: Set[str] = {
            str(x).strip() for x in recommended_blog_ids if x is not None and str(x).strip()
        }

        recommendations = {}
        for blog_id, blog in current_blogs.items():
            if not isinstance(blog, dict):
                continue
            author_id = blog.get("user_id") or blog.get("author_id")
            if author_id not in follow_set:
                continue
            t = blog.get("time") or blog.get("create_time")
            if t is None:
                continue
            try:
                t = int(t)
            except (TypeError, ValueError):
                continue
            if t >= 10**12:
                t //= 1000
            if last_login <= t < window_end:
                sid = str(blog_id).strip() if blog_id is not None else ""
                if not sid or sid in seen_rec_ids:
                    continue
                chain_ids: Set[str] = set()
                reposted_path = blog.get("reposted_path", [])
                if isinstance(reposted_path, list):
                    chain_ids.update(
                        str(x).strip() for x in reposted_path if x is not None and str(x).strip()
                    )
                reposted_blog_id = blog.get("reposted_blog_id")
                if reposted_blog_id is not None and str(reposted_blog_id).strip():
                    chain_ids.add(str(reposted_blog_id).strip())
                if chain_ids.intersection(seen_rec_ids):
                    continue
                recommendations[blog_id] = blog

        # Update the last login timestamp of the agent
        if self.profile:
            _d = int(getattr(event, "timestamp_duration", 0) or 0)
            self.profile.update_data("last_login_timestamp", current_ts)
            logger.info(
                f"Step {current_step}/{max_step}: UserAgent {self.profile_id} update last_login_timestamp: {current_ts + _d}"
            )

        # Send SocialRecommendationEvent
        _social_recommendation_prob = await user_social_recommendation_prob(self)
        if recommendations and len(recommendations) > 0:
            if random.random() > _social_recommendation_prob:
                logger.debug(
                    f"Step {current_step}/{max_step}: UserAgent {self.profile_id} "
                    f"skip SocialRecommendationEvent (n={len(recommendations)}, "
                    f"p={_social_recommendation_prob:.0%} not drawn)"
                )
            else:
                logger.info(
                    f"Step {current_step}/{max_step}: UserAgent {self.profile_id} send SocialRecommendationEvent, length of recommendations: {len(recommendations)}"
                )

                # Fill reposted_blog field
                for blog_id, blog in recommendations.items():
                    reposted_blog_id = blog.get("reposted_blog_id", "")
                    if reposted_blog_id and reposted_blog_id in current_blogs:
                        blog["reposted_blog"] = current_blogs[reposted_blog_id]

                events.append(
                    SocialRecommendationEvent(
                        from_agent_id=self.profile_id,
                        to_agent_id=self.profile_id,
                        timestamp=current_ts,
                        timestamp_duration=int(getattr(event, "timestamp_duration", 0) or 0),
                        current_step=current_step,
                        max_step=max_step,
                        recommendations=recommendations,
                    )
                )
        return events

    async def add_env_reposts(self, key: str, value: Any, parent_event_id: Optional[str] = None) -> bool:
        """Add reposts to the data in the environment (using distributed lock)"""    
        # Create a unique request ID
        request_id = f"agent_env_add_reposts_req_{time.time()}_{id(self)}"

        # Create a Future to receive the response
        future = asyncio.Future()
        self._repost_add_futures[request_id] = future

        # Create a repost add event
        repost_add_event = AddRepostEvent(
            from_agent_id=self.profile_id,  
            to_agent_id="ENV",              
            source_type="AGENT",            
            target_type="ENV",              
            key=key,                        
            value=value,                    
            request_id=request_id,          
            parent_event_id=parent_event_id 
        )

        # Get the distributed lock for this key
        lock_id = f"env_repost_add_lock_content_pool"
        lock = await get_lock(lock_id)

        try:
            async with lock:
                from onesim.events import get_event_bus
                event_bus = get_event_bus()
                await event_bus.dispatch_event(repost_add_event)

            try:
                if hasattr(self, '_sync_event'):
                    await asyncio.wait_for(self._sync_event.wait(), timeout=30.0)
                    return await future
                return await asyncio.wait_for(future, timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning(f"Environment repost add timeout: {key}")
                self._repost_add_futures.pop(request_id, None)
                return False
            except Exception as e:
                logger.error(f"Error adding environment repost: {e}")
                self._repost_add_futures.pop(request_id, None)
                return False
        except Exception as e:
            logger.error(f"Error getting environment repost add lock: {e}")
            return False

    async def handle_add_repost_response(self, event: AddRepostResponseEvent) -> None:
        """Handle the incoming repost add response event"""
        # Check if waiting for this response
        if event.request_id in self._repost_add_futures:
            future = self._repost_add_futures.pop(event.request_id)

            if not future.done():
                if event.success:
                    future.set_result(True)
                else:
                    future.set_exception(ValueError(event.error or "Unknown error"))

            # If there is a sync event, set it
            if hasattr(self, '_sync_event'):
                self._sync_event.set()
                self._sync_event.clear()

    async def generate_memory_from_own_blogs(self, event: Event) -> List[Event]:
        """For user to generate memory from their own blogs"""
        current_step = getattr(event, "current_step", 1)
        max_step = getattr(event, "max_step", 8)
        user_id = await self.get_data("id")
        if not user_id:
            logger.error(f"Step {current_step}/{max_step}: User {user_id} failed to get user_id")
            return []
        uid_norm = str(user_id).strip()

        current_ts = time_to_sec(getattr(event, "timestamp", None), default=0.0) or 0.0
        duration = int(getattr(event, "timestamp_duration", 0) or 0)

        current_blogs = getattr(event, "current_blogs", None) or {}
        if not isinstance(current_blogs, dict) or not current_blogs:
            logger.error(f"Step {current_step}/{max_step}: User {user_id} failed to get current_blogs")
            return []

        # Filter own blogs
        own_entries: List[Tuple[str, Dict[str, Any]]] = []
        for blog_id, blog in current_blogs.items():
            if not isinstance(blog, dict):
                continue
            b_uid = str(blog.get("user_id") or blog.get("author_id") or "").strip()
            if b_uid != uid_norm:
                continue
            own_entries.append((blog_id, blog))

        lo = float(current_ts)
        dur_sec = float(duration)
        cap = getattr(event, "simulation_cap_timestamp", None)
        cap_f = time_to_sec(cap)
        if cap_f is not None:
            hi = min(lo + dur_sec, cap_f)
        else:
            hi = lo + dur_sec
        if hi <= lo:
            if dur_sec <= 0:
                logger.info(f"Step {current_step}/{max_step}: User {user_id} skip own-blogs memory: empty time window. lo={lo}, hi={hi}")   
            else:
                logger.warning(f"Step {current_step}/{max_step}: User {user_id} empty time window hi<=lo: lo={lo}, hi={hi}, duration={dur_sec}, simulation_cap_timestamp={cap!r}")
            return []

        if not own_entries:
            logger.info(f"Step {current_step}/{max_step}: User {user_id} no own posts in current_blogs")
            return []

        # Filter own original blogs
        own_originals = [
            (bid, b)
            for bid, b in own_entries
            if not is_repost_of_other_blog(str(bid), b)
        ]
        if self.profile:
            existing_recommended = self.profile.get_data("recommended_blog_ids", []) or []
            if not isinstance(existing_recommended, list):
                existing_recommended = list(existing_recommended) if existing_recommended else []
            own_blog_ids = [str(blog_id) for blog_id, _ in own_entries if blog_id]
            if own_blog_ids:
                merged = list(dict.fromkeys(existing_recommended + own_blog_ids))
                self.profile.update_data("recommended_blog_ids", merged)

        own_blogs_for_prompt: List[Dict[str, Any]] = []
        for blog_id, blog in own_originals:
            own_blogs_for_prompt.append({
                "blog_id": blog_id,
                "content": blog.get("content", ""),
                "time": blog.get("time", blog.get("create_time", 0)),
            })

        if own_entries and not own_originals:
            logger.info(f"Step {current_step}/{max_step}: User {user_id} skip self-reflection prompt: {len(own_entries)} own post(s) in current_blogs but all forward others (reposted_blog_id≠self)")

        # Build the instruction and observation for the memory generation
        if own_blogs_for_prompt:
            instruction = (
                "你正在复盘自己最近发布的内容。请基于这些帖子，沉淀一条第一人称记忆，"
                "重点总结：你最近持续关注的话题、表达风格、以及后续互动中可复用的表达策略。"
            )
            observation = f"你最近发布的微博：\n{json.dumps(own_blogs_for_prompt, ensure_ascii=False, indent=2)}"
            reaction = {
                "task": "self_reflection_on_own_blogs",
                "own_blog_count": len(own_blogs_for_prompt),
                "highlights": own_blogs_for_prompt,
            }
            try:
                # Generate the memory from the own blogs
                memory_text = await self.generate_memory(instruction, observation, reaction)
                if memory_text:
                    logger.info(f"Step {current_step}/{max_step}: User {user_id} generated self-memory from {len(own_blogs_for_prompt)} own blogs, memory_text: {memory_text}")
            except Exception as e:
                logger.error(f"Step {current_step}/{max_step}: User {user_id} failed to generate memory from own blogs: {e}")

        followup_prob = float(os.environ.get("WEIBO_SELF_FOLLOWUP_REPOST_PROB", "0.001"))
        if followup_prob <= 0.0:
            logger.error(f"Step {current_step}/{max_step}: User {user_id} failed to get followup_prob")
            return []

        max_batch = max(1, int(os.environ.get("WEIBO_SELF_FOLLOWUP_MAX_BATCH", "16")))
        candidates = list(own_entries)
        random.shuffle(candidates)
        subs: List[Tuple[Any, Dict[str, Any]]] = [
            (bid, b) for bid, b in candidates if random.random() < followup_prob
        ]
        if len(subs) > max_batch:
            subs = random.sample(subs, max_batch)
        if not subs:
            return []

        # Decide whether to add a repost to the own blogs
        user_nickname = await self.get_data("nickname", "") or ""
        ip_location = await self.get_data("ip_location", "") or ""
        current_ts = int(event.timestamp) if getattr(event, "timestamp", None) is not None else 0
        step_duration = int(getattr(event, "timestamp_duration", 0) or 0)
        env = getattr(self, "env", None)

        posts_payload = [
            {"blog_id": str(bid), "content": (b.get("content") or "")[:800]}
            for bid, b in subs
        ]
        n_posts = len(posts_payload)
        posts_json = json.dumps({"self_posts": posts_payload}, ensure_ascii=False, indent=2)
        follow_ids = await self.get_data("follow_ids", [])
        fan_ids = await self.get_data("fan_ids", [])
        mentionable_users = self._get_mentionable_users(follow_ids, fan_ids)
        mentionable_users_str = json.dumps(
            mentionable_users.get("follows", []), ensure_ascii=False, indent=2
        )
        observation = f"""【场景】对本人微博追加转发（罕见，批量一次决策）
        {posts_json}

        可@的用户列表（追加转发正文禁止出现「@」，仅作关系参考）：
        {mentionable_users_str}"""
        instruction = f"""这是一次罕见的「对本人历史微博追加转发」批量决策（共 {n_posts} 条）。
        请结合 Observation 中的 self_posts 与 memory：对**每一条**判断是否值得追加转发（新事实、新进展或态度变化）。
        若值得：repost=true 并填写 repost_content（≤40字，口语、第一人称）；否则 repost=false，repost_content 为空。
        repost_content 中不允许出现「@」。

        请按以下 JSON 返回（字段顺序固定），并用 ```json 代码块包裹。
        **decisions 必须恰好包含 {n_posts} 条**，且与 self_posts **顺序一致**；第 i 条的 blog_id 必须等于 self_posts[i].blog_id。

        ```json
        {{
        "persona_understanding": "1-2句。概括身份、兴趣、语言风格（简短）",
        "content_understanding": "1-2句。概括与这批本人帖的整体相关性",
        "memory_reflection": "2-3句。哪些值得追评、哪些应沉默",
        "decisions": [
        {{
        "blog_id": "<与 self_posts[0].blog_id 一致>",
        "repost": false,
        "repost_content": "",
        "decision_reason": "（1句，<20字）",
        "expression_reason": "",
        "mention_reasoning": []
        }}
        ]
        }}
        ```
        （decisions 内请展开为 {n_posts} 个对象，勿省略。）"""

        if env is not None and hasattr(env, "notify_agent_busy"):
            await env.notify_agent_busy()
        try:
            # Generate the response from the own blogs
            response = await self.generate_reaction(instruction, observation)
        finally:
            if env is not None and hasattr(env, "notify_agent_idle"):
                await env.notify_agent_idle()

        if not isinstance(response, dict):
            return []
        decisions_raw = response.get("decisions")
        if not isinstance(decisions_raw, list):
            return []

        # Add reposts to the own blogs
        by_blog_id: Dict[str, Dict[str, Any]] = {}
        for d in decisions_raw:
            if isinstance(d, dict):
                bk = str(d.get("blog_id") or "").strip()
                if bk:
                    by_blog_id[bk] = d

        for blog_id, blog in subs:
            bid = str(blog_id)
            dec = by_blog_id.get(bid)
            if not isinstance(dec, dict):
                continue
            if not dec.get("repost", False):
                continue
            repost_content = (dec.get("repost_content") or "").strip()
            repost_content = " ".join(repost_content.splitlines()).strip()[:200]
            if not repost_content:
                continue

            repost_id = generate_repost_id()

            if str(blog.get("reposted_blog_id") or "").strip():
                reposted_path = list(blog.get("reposted_path", []))
                reposted_path.append(str(blog_id))
                reposted_path = list(dict.fromkeys(reposted_path))
                reposted_blog_id = blog.get("reposted_blog_id")
                reposted_user_id = blog.get("user_id")
                blog_content = blog.get("content", "")
                emit_content = f"{repost_content}//@{reposted_user_id}: {blog_content}"
                blog_author_ids: List[str] = []
                for part in str(emit_content).split("//"):
                    match = re.search(r"@([^:：\s]+)\s*[:：]", part)
                    if match:
                        blog_author_ids.append(match.group(1))
            else:
                reposted_path = [str(blog_id)]
                reposted_blog_id = blog_id
                blog_author_ids = [str(blog.get("user_id") or user_id)]

            mention_count = 0
            success = await self.add_env_reposts(repost_id, {
                "blog_id": repost_id,
                "content": emit_content if str(blog.get("reposted_blog_id") or "").strip() else repost_content,
                "time": generate_blog_timestamp(blog, current_ts, step_duration),
                "ip_location": ip_location,
                "user_id": user_id,
                "nickname": user_nickname,
                "at_count": mention_count,
                "reposted_blog_id": reposted_blog_id,
                "reposted_path": reposted_path,
                "repost_count": 0,
                "repost_ids": [],
            })
            if not success:
                logger.error(f"Step {current_step}/{max_step}: add_env_reposts failed for self_followup_repost {repost_id}")
                continue

            for blog_author_id in blog_author_ids:
                if blog_author_id and blog_author_id != user_id:
                    ok = await self.update_env_mention_pool(f"{blog_author_id}.{repost_id}", {
                        "action": "add",
                        "mention_message": {"blog_id": repost_id, "mention_type": "repost"},
                    })
                    if not ok:
                        logger.error(f"Step {current_step}/{max_step}: mention_pool update failed for self_followup {repost_id}")
                    else:
                        logger.info(f"Step {current_step}/{max_step}: User {user_id} self_followup_repost on {blog_id} -> {repost_id}")
        return []

    def _filter_recommendations(self, recommendations: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Remove duplicate recommendations"""
        if not recommendations:
            return {}
        
        # Get the set of recommended content IDs (from profile)
        recommended_blog_ids = set(self.profile.get_data("recommended_blog_ids", [])) if self.profile else set()
        
        # Remove duplicate recommendations 
        filtered: Dict[str, Dict[str, Any]] = {}

        for blog_id, rec in recommendations.items():
            if not isinstance(rec, dict):
                logger.warning(f"Recommendation {blog_id} is not a dictionary")
                continue
            
            # If the blog ID does not exist or has already been recommended, skip
            if not blog_id or blog_id in recommended_blog_ids:
                logger.info(f"Recommendation {blog_id} is already recommended")
                continue

            # If any blog_id in the repost chain has already been recommended, skip (to avoid duplicate exposure)
            chain_ids = set()
            reposted_path = rec.get("reposted_path", [])
            if isinstance(reposted_path, list):
                chain_ids.update(str(x) for x in reposted_path if x is not None and str(x).strip())
            reposted_blog_id = rec.get("reposted_blog_id")
            if reposted_blog_id is not None and str(reposted_blog_id).strip():
                chain_ids.add(str(reposted_blog_id))

            hit_chain_ids = chain_ids.intersection(recommended_blog_ids)
            if hit_chain_ids:
                logger.info(f"Recommendation {blog_id} filtered by repost chain history, hit ids: {sorted(hit_chain_ids)}")
                continue
            filtered[blog_id] = rec
            logger.info(f"Recommendation {blog_id} is added to filtered list")
        return filtered

    def _add_recommendations(self, recommendations: Dict[str, Any]) -> None:
        """Add the blog_ids of the current recommendations to profile.recommended_blog_ids."""
        if not recommendations or not self.profile:
            return
        recommended_blog_ids = set(self.profile.get_data("recommended_blog_ids", [])) if self.profile else set()
        new_blog_ids = [str(bid) for bid in recommendations.keys() if bid]
        if not new_blog_ids:
            return

        # Add the new recommended content IDs to the recommended list
        all_recommended = list(recommended_blog_ids) + new_blog_ids
        self.profile.update_data("recommended_blog_ids", all_recommended)

    def _get_mentionable_users(
        self,
        follow_ids: List[str],
        fan_ids: List[str]
    ) -> Dict[str, Any]:
        """Get the mentionable users list (including follows, fans, and mutual)"""
        mentionable_info = {
            "follows": [],  # Follows list
            "fans": [],  # Fans list
            "mutual": []  # Mutual list
        }
        
        # Convert to a set for calculation
        follow_ids = set(follow_ids)
        fan_ids = set(fan_ids)
        
        # Calculate mutual (intersection)
        mutual_ids = follow_ids & fan_ids
        
        if not self.relationship_manager:
            logger.warning("RelationshipManager is not initialized")
            return mentionable_info
        
        # Process all lists, directly get user information from relationship_manager (synchronously, quickly)
        follows_info = []
        fans_info = []
        mutual_info = []
        
        # Process the follow list (including mutual)
        for user_id in follow_ids:
            rel = self.relationship_manager.get_relationship(user_id)
            if rel and rel.target_info and isinstance(rel.target_info, dict):
                info = dict(rel.target_info)
                if "historical_summary" in info:
                    info["historical_summary"] = format_historical_summary(info.get("historical_summary"))
                hn = info.get("historical_blogs")
                if isinstance(hn, dict) and len(hn) > 2:
                    items = list(hn.items())[:2]
                    info["historical_blogs"] = dict(items)
                follows_info.append(info)
        
        # Process the fan list (including mutual)
        for user_id in fan_ids:
            rel = self.relationship_manager.get_relationship(user_id)
            # Check if rel is None, then access target_info
            if rel and rel.target_info and isinstance(rel.target_info, dict):
                info = dict(rel.target_info)
                if "historical_summary" in info:
                    info["historical_summary"] = format_historical_summary(info.get("historical_summary"))
                hn = info.get("historical_blogs")
                if isinstance(hn, dict) and len(hn) > 2:
                    items = list(hn.items())[:2]
                    info["historical_blogs"] = dict(items)
                fans_info.append(info)
        
        # Process the mutual list (the historical notes of the mutual users are kept at most 2)
        for user_id in mutual_ids:
            rel = self.relationship_manager.get_relationship(user_id)
            # Check if rel is None, then access target_info
            if rel and rel.target_info and isinstance(rel.target_info, dict):
                info = dict(rel.target_info)
                if "historical_summary" in info:
                    info["historical_summary"] = format_historical_summary(info.get("historical_summary"))
                hn = info.get("historical_blogs")
                if isinstance(hn, dict) and len(hn) > 2:
                    items = list(hn.items())[:2]
                    info["historical_blogs"] = dict(items)
                mutual_info.append(info)

        mentionable_info["follows"] = follows_info
        mentionable_info["fans"] = fans_info
        mentionable_info["mutual"] = mutual_info
        
        return mentionable_info

    async def receive_recommendation(self, event: Event) -> List[Event]:
        """Receive recommendations and decide whether to comment."""
        current_step = getattr(event, "current_step", 1)
        max_step = getattr(event, "max_step", 8)
        if not await self._should_activate_this_round(current_step, max_step):
            logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} receive recommendation but is not activated this round")
            return []

        # Store recommendations by type
        if event.__class__.__name__ == "AlgorithmRecommendationEvent":
            source_type = "algorithm"
        elif event.__class__.__name__ == "SocialRecommendationEvent":
            source_type = "social"
        elif event.__class__.__name__ == "SearchResultEvent":
            source_type = "search"
        elif event.__class__.__name__ == "KeepFollowingEvent":
            source_type = "keep_following"
        else:
            evt_cls = event.__class__.__name__
            logger.warning(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} unknown recommendation event {evt_cls}, skip")
            return []
        
        logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} receive recommendation event, source type: {source_type}, length of recommendations: {len(event.recommendations)}")

        recommendations = event.recommendations
        record_recommendations_by_source_step(
            self.profile,
            self.profile_id,
            source_type,
            current_step,
            recommendations,
            getattr(event, "timestamp", 0),
        )

        # algorithm/social: remove duplicate recommendations from profile.recommended_note_ids
        if source_type == "algorithm" or source_type == "social":
            recommendations = self._filter_recommendations(recommendations)
            logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} after filter, length of recommendations: {len(recommendations)}")
        if len(recommendations) == 0:
            logger.info(f"Step {current_step}/{max_step}: No recommendations, skip")
            return []
        # Record the viewed notes
        self._add_recommendations(recommendations)

        # Get user information and mentionable users
        user_id = await self.get_data("id")
        user_nickname = await self.get_data("nickname", "")
        ip_location = await self.get_data("ip_location", "")
        current_timestamp = event.timestamp
        current_ts = int(event.timestamp) if getattr(event, "timestamp", None) is not None else 0
        step_duration = int(getattr(event, "timestamp_duration", 0) or 0)
        follow_ids = await self.get_data("follow_ids", [])
        fan_ids = await self.get_data("fan_ids", [])
        mentionable_users = self._get_mentionable_users(follow_ids, fan_ids)

        # Build the observation and instruction, and process the recommendations in chunks
        rec_items = list(recommendations.items())
        chunk_size = 5
        if len(rec_items) <= chunk_size:
            chunks = [recommendations]
        else:
            chunks = [dict(rec_items[i:i + chunk_size]) for i in range(0, len(rec_items), chunk_size)]

        events_to_send = []
        has_search = False

        for chunk in chunks:
            recommendations_str = json.dumps(chunk, ensure_ascii=False, indent=2)
            mentionable_users_str = json.dumps(mentionable_users.get("follows", []), ensure_ascii=False, indent=2)

            # Label recommendation source; if the recommendation contains content published by the user, label it as "自己发布"
            has_self_blog = any(
                isinstance(blog, dict) and blog.get("user_id") == user_id
                for blog in chunk.values()
            )
            if has_self_blog:
                source_name = "【⚠ 检测到该转发链包含你自己：默认repost=false】"
            else:
                source_name = "关注流（来自你关注的用户）" if source_type == "social" else "推荐流（来自算法）"

            observation = f"""【场景】你正在用手机刷信息流：大部分内容划走即可，只有偶尔才会停下来打一行字或只是点一下转发。你不是在完成实验任务，也不是写舆情分析。

            推荐来源：{source_name}

            收到的推荐内容：
            {recommendations_str}

            可@的用户列表：
            {mentionable_users_str}"""

            gates = await UserAgentGates(self).build_recommendation_coaching(
                chunk, current_timestamp
            )

            instruction = f"""根据用户的 Profile、historical_summary、memory 和推荐内容，完成转发决策与生成。

            步骤1：对每条推荐先设默认状态
            - "repost": false
            - "repost_content": ""

            **repost 的数量决策仅由步骤2/3决定，不得因步骤5、6改变；**
            {gates.memory_coaching}

            步骤2：兴趣判断
            - 默认 repost=false。通读本批次所有推荐内容：若无任何一条能同时通过步骤1.5 破例条件，则**本批全体 repost=false**，不得因「略有相关」放行。
            - 仅当**下列全部满足**且**不违反步骤2.2**，可将该条改为 repost=true：
                1. 内容与 Profile / historical_summary 所体现的身份、长期关切或口味**明显对路**，你刷到会有「想让人看到 / 想站队 / 想关注对象顶一下」的冲动——满足才可作为转发的**主因**，否则repost=false
                {gates.memory_rec}
                3. decision_reason 须**一句话**点明是哪一种；**禁止**用泛化套话（如「值得关注」「再说两句」）代替具体动机，否则repost=false
            - 对不感兴趣的微博：该微博不进入候选池，repost=false。
            - 若无任何一条满足转发条件，本批可全部为 repost=false。

            步骤2.2：转发链自检（显著提示）
            - 【⚠ 自检提醒】若该条微博的转发链（如 reposted_path、链路作者）中出现你自己的 user_id，视为“可能重复传播自己链路”。
            - 命中该情形时：默认不进入候选池，**repost=false**；极小概率二次转发，此时不允许 repost_content=""，必须写新增评论且和前面的转发内容有明显区别。

            步骤3：选取 0～{gates.k_diff_targets} 目标微博
            - {gates.k_diff_targets} 表示本轮**最多**对多少个**不同** blog_id 置 repost=true；实际条数可为 0～{gates.k_diff_targets} 中任意值，**不得**为凑满条数而放宽步骤1.5/步骤2。
            - 仅在步骤2 的候选池内、按你的排序取前若干条作为“目标 blog_id 集合”，且条数不超过 {gates.k_diff_targets}；若候选少于 {gates.k_diff_targets}，有多少算多少；其余笔记保持 repost=false。

            步骤4：为每个目标 blog_id 生成 0～{gates.k_same_target} 条转发
            - 以个人视角出发，对每个目标 blog_id：在 decisions 中写出 {gates.k_same_target} 个 repost=true 的条目（可用于追评链/补一句，或可仅转发两次）。
            - 若 {gates.k_same_target}=1：每个目标 blog_id 只生成 1 条转发。
            - 若 {gates.k_same_target}>=2：允许同帖多条，但第2条及之后必须是增量（补充新点/纠错/情绪加码），禁止复述同一句。允许不写转发内容。

            步骤5：判断是否“仅转发不加评”
            - 以个人视角出发，对每条 repost=true 的候选，做二选一
            - A) 空内容转发：repost_content=""，只想转发不想评论
            - B) 带评论转发：repost_content 非空，仅当有意愿表达个性化评论或想参与事件讨论，同时想帮忙扩散事件时才选择
            若步骤5选择选择 A (空内容转发)，则 repost_content 必须为 ""，跳过步骤6和步骤7。

            步骤6：若repost_content不为空，根据人设，为每条转发选择 repost_mode，默认为围观
            - repost_mode ∈ {{围观，玩梗吐槽，情绪抒发，分析转述}}，并在 expression_reason 体现。
            - 发生概率：围观 > 玩梗吐槽 > 情绪抒发 > 分析转述
            - 共同特征：口语化，以第一人称视角表达，不要求语法正确
                - 围观：字数极短、结构弱，多重复其他短评中的围观表达（如“哈哈”、“笑死”、“666”等）。上限 5 字，禁止使用标点
                - 玩梗吐槽：通过双关、戏仿、梗化或夸张形成轻松/讽刺表达。上限 15 字，禁止使用标点
                - 情绪抒发：情绪强度较高的主观表态，核心是惊讶、兴奋、无奈等感受输出。上限 25 字，上限 1个标点，或者以空格代替标点
                - 分析转述：信息密度高，仅在“确有新增信息”时使用。上限 50 字
            -输出风格：
                - 使用用户所在地区常用语言；默认使用中文；
                - 表达应符合 Profile 与 historical_summary 的人设口吻；
                - 减少模板化套话，如“你说得对/确实/很有道理/希望大家…”；
                - 评论内容不包含标签（#...）。
                - repost_content中不允许出现“@...”。

            步骤7：决定是否需要 @ 用户
            - 以个人视角出发，仅当对方在可@列表且与内容强相关 + 不是推文作者/父转发作者时，才在正文中额外 @user_id；否则 mention_reasoning 为 []。

            步骤8：keep_following_blog_ids
            - 仅当：高度感兴趣 + 当前内容有持续关注价值→ keep_following_blog_ids=true；否则为空列表 []

            步骤9：search
            - 仅当：高度感兴趣 + 当前信息明显不足 + memory 无相关内容 → search=true；否则 false。

            请按以下 JSON 返回（字段顺序固定）：
            {{
            "persona_understanding": "1-2句。概括身份、兴趣、语言风格（简短）",
            "content_understanding": "1-2句。概括本次内容与你的相关性、你优先关注的角度",
            "source_understanding": "关注流=关注关系；推荐流=陌生关系，自己发布（你的内容）=自己发布",
            "memory_reflection": "{gates.memory_ref}",
            "decisions": [
                {{
                "blog_id": "blog_id",
                "repost": false,
                "repost_content": "",
                "decision_reason": "repost=false的原因（1句，<20字）",
                "expression_reason": "",
                "mention_reasoning": []
                }},
                {{
                "blog_id": "blog_id",
                "repost": true,
                "repost_content": "",
                "decision_reason": "转发的理由（1句，<20字）",
                "expression_reason": "repost_mode 为 none 时，expression_reason 为空字符串",
                "mention_reasoning": []
                }},
                {{
                "blog_id": "blog_id",
                "repost": true,
                "repost_content": "个性化的转发内容",
                "decision_reason": "转发的理由（1句，<20字）",
                "expression_reason": "为何使用这种语气与措辞",
                "mention_reasoning": [
                    {{
                    "user_id": "被@用户id",
                    "persona_understanding": "对该用户的简短理解",
                    "mention_reason": "为何需要额外提醒该用户（1句，<20字）"
                    }}
                ]
                }}
            ],
            "keep_following_blog_ids": [],
            "keep_following_reason": "为何保持关注（1句，<20字）",
            "search": false,
            "search_keyword": "搜索关键词（1句，≤20字）",
            "search_reason": "是否搜索及原因（1句）",
            }}

            输出规则：
            - 先给出理解字段，再给 decisions；默认使用中文，可中英文混合
            - decisions 中每条先按默认状态填写 repost=false，再仅对满足条件的条目改为 repost=true；
            - repost=false 时，repost_content 必须为空字符串。repost=true 时，repost_content 可以为空字符串，也可以为转发内容。
            - 根据 repost_mode 重复检验选择的评论内容，是否符合模式要求。
            {gates.similarity_kw_coaching}{gates.similarity_emb_coaching}
            {gates.freshness_coaching}"""
            
            # Call LLM to generate the decision
            logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} generate reaction")
            env = getattr(self, "env", None)
            if env is not None and hasattr(env, "notify_agent_busy"):
                await env.notify_agent_busy()
            try:
                response = await self.generate_reaction(instruction, observation)
            finally:
                if env is not None and hasattr(env, "notify_agent_idle"):
                    await env.notify_agent_idle()
            logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} generate reaction response")

            # Handle search decisions
            search = response.get("search", False)
            if search and not has_search:
                has_search = True
                # Send event to the algorithm module
                search_algorithm_types = await user_default_search_types(self)
                if not isinstance(search_algorithm_types, list) or not search_algorithm_types:
                    raise ValueError("default_search_types must be a non-empty list")
                allowed_search_types = set(self.search_map.keys())

                for search_algorithm_type in search_algorithm_types:
                    if search_algorithm_type not in allowed_search_types:
                        raise ValueError(f"Invalid algorithm type '{search_algorithm_type}'. Allowed types: {sorted(allowed_search_types)}")
                    mapped_id = self.search_map.get(search_algorithm_type, "")
                    if not mapped_id:
                        raise ValueError(f"No recommender agent mapped for algorithm type '{search_algorithm_type}'. Current mapping keys: {sorted(self.search_map.keys())}")
                        continue

                    # Get user profile
                    profile_payload = {}
                    if self.profile is not None:
                        try:
                            profile_payload = dict(self.profile.get_profile(include_private=True) or {})
                        except Exception:
                            logger.warning("Failed to serialize profile via get_profile(), fallback to empty payload.")
                    logger.info(f"Sending GetSearchResultEvent to Algorithm {mapped_id} for search algorithm type {search_algorithm_type}")
                    events_to_send.append(GetSearchResultEvent(
                        from_agent_id=self.profile_id,
                        to_agent_id=mapped_id,
                        timestamp=event.timestamp,
                        timestamp_duration=int(getattr(event, "timestamp_duration", 0) or 0),
                        current_step=current_step,
                        max_step=max_step,
                        user_profile=profile_payload,
                        algorithm_type=search_algorithm_type,
                        search_query=str(response.get("search_keyword") or "").strip(),
                    ))

            # Handle keep following decisions
            if self.profile:
                keep_ids = response.get("keep_following_blog_ids", [])
                if isinstance(keep_ids, list) and keep_ids:
                    # Only allow blog_ids within the current batch, and at most 1
                    valid_keep_ids = []
                    for keep_blog_id in keep_ids:
                        if keep_blog_id in chunk:
                            valid_keep_ids.append(keep_blog_id)
                    if valid_keep_ids:
                        self.profile.update_data("keep_following_blog_ids", valid_keep_ids[:1])
                    else:
                        self.profile.update_data("keep_following_blog_ids", [])

            # Handle diffusion decisions
            decisions = response.get("decisions", [])
            if not isinstance(decisions, list):
                continue
       
            # Handle each decision: update repost count and repost content
            for decision in decisions:
                if not isinstance(decision, dict):
                    continue
                    
                blog_id = decision.get("blog_id")
                should_repost = decision.get("repost", False)
                repost_content = decision.get("repost_content", "")
                    
                if not blog_id or not should_repost:
                    continue

                # Check if the blog_id is valid
                if blog_id not in chunk:
                    logger.warning(f"Step {current_step}/{max_step}: Blog {blog_id} not found in recommendations")
                    continue
                blog = chunk[blog_id]
            
                # Parse the @users in the content, replace @id with @nickname, and return the list of user IDs
                mentioned_user_ids = []
                mention_reasoning = decision.get("mention_reasoning", [])
                if isinstance(mention_reasoning, list):
                    for mention_reason in mention_reasoning:
                        if isinstance(mention_reason, dict):
                            reason_user_id = mention_reason.get("user_id")
                            if reason_user_id:
                                mentioned_user_ids.append(reason_user_id)

                # Build the repost path and repost content, and the list of authors
                if blog.get("reposted_blog_id"):
                    reposted_path = list(blog.get("reposted_path", []))
                    reposted_path.append(blog_id)

                    # Keep the order and remove duplicates to avoid repeated accumulation in the propagation chain
                    reposted_path = list(dict.fromkeys(reposted_path))
                    reposted_blog_id = blog.get("reposted_blog_id")
                    reposted_user_id = blog.get("user_id")
                    blog_content = blog.get("content", "")
                    repost_content = f"{repost_content}//@{reposted_user_id}: {blog_content}"

                    # blog_author_ids: split by //, then extract the user_id between @ and :
                    blog_author_ids = []
                    for part in str(repost_content).split("//"):
                        match = re.search(r"@([^:：\\s]+)\\s*[:：]", part)
                        if match:
                            blog_author_ids.append(match.group(1))
                else:
                    reposted_path = [blog_id]
                    reposted_blog_id = blog_id
                    reposted_user_id = blog.get("user_id")
                    blog_author_ids = [reposted_user_id]

                # If the @ user is already in the list of authors in the repost chain, avoid duplicate reminders
                overlap_user_ids = set(uid for uid in blog_author_ids if uid)
                mentioned_user_ids = [
                    uid for uid in mentioned_user_ids if uid not in overlap_user_ids
                ]

                mention_count = len(mentioned_user_ids)

                if repost_content == "":
                    repost_content = "转发微博"
              
                # If it is a repost, add the repost
                repost_id = generate_repost_id()
                success = await self.add_env_reposts(repost_id, {
                    "blog_id": repost_id,
                    "content": repost_content,
                    "time": generate_blog_timestamp(blog, current_ts, step_duration),
                    "ip_location": ip_location,
                    "user_id": user_id,
                    "nickname": user_nickname,
                    "at_count": mention_count,
                    "reposted_blog_id": reposted_blog_id,
                    "reposted_path": reposted_path,
                    "repost_count": 0,
                    "repost_ids": []
                })
                if not success:
                    logger.error(f"Failed to add repost to content pool, blog {blog_id}")
                    continue

                # Send reminders to the note authors
                for blog_author_id in blog_author_ids:
                    if blog_author_id and blog_author_id != user_id: 
                        success = await self.update_env_mention_pool(f"{blog_author_id}.{repost_id}", {
                            "action": "add",
                            "mention_message": {
                                "blog_id": repost_id,
                                "mention_type": "repost"
                            }
                        })
                        if not success:
                            logger.error(f"Failed to update mention pool for repost {repost_id} by {user_id} on blog {blog_id}")
                            continue
                        logger.info(f"Step {current_step}/{max_step}: User {user_id} reposted blog {blog_id} by {blog_author_id}")
                   
                # Send MentionEvent to the users mentioned
                if mentioned_user_ids:
                    for mentioned_user_id in mentioned_user_ids:
                        if mentioned_user_id and mentioned_user_id != user_id:  
                            success = await self.update_env_mention_pool(f"{mentioned_user_id}.{repost_id}", {
                                "action": "add",
                                "mention_message": {
                                    "blog_id": repost_id,
                                    "mention_type": "at"
                                }
                            })
                            if not success:
                                logger.error(f"Failed to update mention pool for repost {repost_id} by {user_id} on blog {blog_id}")
                                continue
                            logger.info(f"Step {current_step}/{max_step}: User {user_id} mentioned {mentioned_user_id} in repost {repost_id} on blog {blog_id}")
                    
        # Notify the environment to update the content pool
        content_update_event = RecommendationSpreadingEvent(
            from_agent_id=self.profile_id,
            to_agent_id="ENV",
            timestamp=current_timestamp,
            timestamp_duration=int(getattr(event, "timestamp_duration", 0) or 0),
            current_step=current_step,
            max_step=max_step,
        )
        events_to_send.append(content_update_event)
        
        return events_to_send
    
    def _check_relationship(self, my_id: str,other_user_id: str, follow_ids: List[str], fan_ids: List[str]) -> str:
        """Check the relationship type with another user"""
        if not other_user_id:
            return "none"
        
        if not my_id or my_id == other_user_id:
            return "none"
        
        # Check if it is a mutual
        if other_user_id in follow_ids and other_user_id in fan_ids:
            return "mutual"
        
        # Check if it is a follow
        if other_user_id in follow_ids:
            return "follow"
        
        # Check if it is a fan
        if other_user_id in fan_ids:
            return "fan"

        return "none"

    def _has_self_in_repost_chain(
        self,
        blog: Dict[str, Any],
        my_id: str,
        content_pool: Dict[str, Any],
    ) -> bool:
        """Check if the repost chain contains yourself"""
        if not isinstance(blog, dict) or not my_id:
            return False
        my_id_str = str(my_id)

        author_id = blog.get("user_id")
        if author_id is not None and str(author_id) == my_id_str:
            return True

        path = blog.get("reposted_path", [])
        if isinstance(path, list) and any(str(x) == my_id_str for x in path):
            return True

        parent_id = blog.get("reposted_blog_id")
        visited: Set[str] = set()
        while parent_id:
            p = str(parent_id).strip()
            if not p or p in visited:
                break
            visited.add(p)
            parent_blog = content_pool.get(p)
            if not isinstance(parent_blog, dict):
                break
            p_uid = parent_blog.get("user_id")
            if p_uid is not None and str(p_uid) == my_id_str:
                return True
            parent_id = parent_blog.get("reposted_blog_id")

        return False

    async def handle_mention(self, event: MentionEvent) -> List[Event]:
        """Handle the @/comment/reply reminder event"""
        current_step = getattr(event, "current_step", 1)
        max_step = getattr(event, "max_step", 8)
        if not await self._should_activate_this_round(current_step, max_step):
            logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} handle mention but is not activated this round")
            return []

        # Get the reminder information
        mentions = getattr(event, "mentions", {}) or {}
        if not mentions:
            return []
        logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} receive mention event, length of mentions: {len(mentions)}")

        # Get the list of users that can be @
        follow_ids = await self.get_data("follow_ids", [])
        fan_ids = await self.get_data("fan_ids", [])
        my_id = await self.get_data("id")
        mentionable_users = self._get_mentionable_users(follow_ids, fan_ids)
        mentionable_users_str = json.dumps(mentionable_users.get("follows", []), ensure_ascii=False, indent=2)

        content_pool = await self.get_env_data("content_pool")
        if not isinstance(content_pool, dict):
            content_pool = {}

        # Build the reminder information
        mention_entries: List[Dict[str, Any]] = []
        for mention_key, mention_message in mentions.items():
            mention_blog = mention_message.get("blog")
            if not isinstance(mention_blog, dict):
                continue
            mentioner_id = mention_blog.get("user_id")
            mentioner_nickname = mention_blog.get("nickname", "")
            mention_type = mention_message.get("mention_type", "at")

            relationship_type = self._check_relationship(my_id, mentioner_id, follow_ids, fan_ids)
            if relationship_type == "mutual":
                relationship_hint = "（你的好友）"
            elif relationship_type == "follow":
                relationship_hint = "（你的关注）"
            else:
                relationship_hint = ""

            if mention_type == "at":
                mention_action = f"{mentioner_nickname}{relationship_hint}在转发中@了你"
                content_label = "@你的转发内容"
            elif mention_type == "repost":
                mention_action = f"{mentioner_nickname}{relationship_hint}转发了你的笔记"
                content_label = "转发内容"
            if relationship_hint:
                relationship_source = f"{mentioner_nickname} 与你的关系：{relationship_hint}"
            else:
                relationship_source = f"{mentioner_nickname} 与你的关系：陌生人"

            self_chain_hit = self._has_self_in_repost_chain(mention_blog, my_id, content_pool)
            self_chain_hint = ""
            if self_chain_hit:
                self_chain_hint = "【⚠ 检测到该转发链包含你自己：默认repost=false】"

            blog_id = mention_blog.get("blog_id") or (mention_key.split("_")[0] if "_" in str(mention_key) else str(mention_key))
            mention_entries.append({
                "mention_key": mention_key,
                "mention_blog": mention_blog,
                "blog_id": blog_id,
                "self_chain_hit": self_chain_hit,
                "self_chain_hint": self_chain_hint,
                "mentioner_id": mentioner_id,
                "mentioner_nickname": mentioner_nickname,
                "mention_type": mention_type,
                "mention_action": mention_action,
                "content_label": content_label,
                "relationship_source": relationship_source,
            })
        if not mention_entries:
            return []

        record_mentioned_blog_ids_by_channel(
            self.profile,
            self.profile_id,
            current_step,
            mention_entries,
            getattr(event, "timestamp", 0),
        )

        # Build the observation information
        observation_parts = []
        for i, entry in enumerate(mention_entries):
            observation_parts.append(f"""## 提醒 {i + 1}
            {entry["mention_action"]}

            与转发者的关系（relationship_understanding 必须据此填写，请勿臆测或颠倒）：
            {entry["relationship_source"]}

            {entry["content_label"]}：：
            {json.dumps(entry["mention_blog"], ensure_ascii=False, indent=2)}
            {entry.get("self_chain_hint", "")}
            """)

        observation = (
            "【场景】你收到了通知（@/转发）。仍可直接忽略；只有值得接话时才回复。接话时也要短，像聊天，不像写通报。\n\n"
            + "转发和@（共 {} 条提醒，请按顺序对每条分别给出决策）：\n\n".format(len(mention_entries))
            + "\n".join(observation_parts)
            + "\n可@的用户列表：\n"
            + mentionable_users_str
        )

        gates = await UserAgentGates(self).build_mention_coaching(
            mention_entries, getattr(event, "timestamp", None)
        )

        instruction = f"""下面有多条提醒，请按顺序对每条提醒分别给出一个决策（是否回复、回复内容等）。decisions 数组与提醒顺序一致，第 i 个元素对应第 i 条提醒。

        请基于 Profile、historical_summary、memory 与 Observation 中的关系信息，完成“是否转发”判断与转发生成。

        步骤1：对每条推荐先设默认状态
        - "repost": false
        - "repost_content": ""

        ** repost 的数量决策仅由步骤2/3决定，不得因步骤5改变**
        {gates.memory_coaching}

        步骤2：兴趣判断
        - 默认 repost=false。通读本批次所有推荐内容：若无任何一条能同时通过步骤1.5 破例条件，则**本批全体 repost=false**，不得因「略有相关」放行。
        - 仅当**下列全部满足**且**不违反步骤2.2**，可将该条改为 repost=true：
            1. 内容与 Profile / historical_summary 所体现的身份、长期关切或口味**明显对路**，你刷到会有「想让人看到 / 想站队 / 想关注对象顶一下」的冲动——满足才可作为转发的**主因**，否则repost=false
            {gates.memory_rec}
            3. decision_reason 须**一句话**点明是哪一种；**禁止**用泛化套话（如「值得关注」「再说两句」）代替具体动机，否则repost=false
        - 对不感兴趣的微博：该微博不进入候选池，repost=false。
        - 若无任何一条满足转发条件，本批可全部为 repost=false；。

        步骤2.2：转发链自检（显著提示）
        - 【⚠ 自检提醒】若该条微博的转发链（如 reposted_path、链路作者）中出现你自己的 user_id，视为“可能重复传播自己链路”。
        - 命中该情形时：默认不进入候选池，**repost=false**；极小概率二次转发，此时不允许 repost_content=""，必须写新增评论且和前面的转发内容有明显区别。

        步骤3：选取 0～{gates.k_diff_targets} 目标微博
        - {gates.k_diff_targets} 表示本轮**最多**对多少个**不同** blog_id 置 repost=true；实际条数可为 0～{gates.k_diff_targets} 中任意值，**不得**为凑满条数而放宽步骤1.5/步骤2。
        - 仅在步骤2 的候选池内、按你的排序取前若干条作为“目标 blog_id 集合”，且条数不超过 {gates.k_diff_targets}；若候选少于 {gates.k_diff_targets}，有多少算多少；其余笔记保持 repost=false。

        步骤4：判断是否“仅转发不加评”
        - 以个人视角出发，对每条 repost=true 的候选，做二选一
        - A) 空内容转发：repost_content=""，只想转发不想评论
        - B) 带评论转发：repost_content 非空，仅当有意愿表达个性化评论或想参与事件讨论，同时想帮忙扩散事件时才选择
        若步骤4选择选择 A (空内容转发)，则 repost_content 必须为 ""，跳过步骤6和步骤7。

        步骤5：若repost_content不为空，根据人设，为每条转发选择 repost_mode，默认选择围观
        - 发生概率：围观 > 玩梗吐槽 > 情绪抒发 > 分析转述
        - 共同特征：口语化，以第一人称视角表达，不要求语法正确
            - 围观：字数极短、结构弱，多重复其他短评中的围观表达（如“哈哈”、“笑死”、“666”等）。上限 5 字，禁止使用标点
            - 玩梗吐槽：通过双关、戏仿、梗化或夸张形成轻松/讽刺表达。上限 15 字，禁标点
            - 情绪抒发：情绪强度较高的主观表态，核心是惊讶、兴奋、无奈等感受输出。上限 25 字，上限 1个标点，或者以空格代替标点
            - 分析转述：信息密度高，仅在“确有新增信息”时使用。上限 50 字
        -输出风格：
            - 使用用户所在地区常用语言；默认使用中文；
            - 表达应符合 Profile 与 historical_summary 的人设口吻；
            - 减少模板化套话，如“你说得对/确实/很有道理/希望大家…”；
            - 评论内容不包含标签（#...）。
            - repost_content中不允许出现“@...”。

        步骤6：决定是否需要 @ 用户
        - 仅当对方在可@列表且与内容强相关 + 不是推文作者/父转发作者时，才在正文中额外 @user_id；否则 mention_reasoning 为 []。

        步骤7：keep_following_blog_ids
        - 仅当：高度感兴趣 + 当前内容有持续关注价值→ keep_following_blog_ids=true；否则为空列表 []

        步骤8：search
        - 仅当：高度感兴趣 + 当前信息明显不足 + memory 无相关内容 → search=true；否则 false。

        请按以下 JSON 返回（字段顺序固定）：
        {{
        "persona_understanding": "1-2句。概括身份、兴趣、语言风格（简短）",
        "content_understanding": "1-2句。概括本次被@/被评论内容与你的相关性、你优先关注的角度",
        "relationship_understanding": "严格依据 Observation 的关系判定，不自行扩展关系类型",
        "memory_reflection": "{gates.memory_ref}",
        "decisions": [
            {{
            "blog_id": "blog_id",
            "repost": false,
            "repost_content": "",
            "decision_reason": "repost=false的原因（1句，≤20字）",
            "expression_reason": "",
            "mention_reasoning": []
            }},
            {{
            "blog_id": "blog_id",
            "repost": true,
            "repost_content": "",
            "decision_reason": "转发的理由（1句，≤20字）",
            "expression_reason": "repost_mode 为 none 时，expression_reason 为空字符串",
            "mention_reasoning": []
            }},
            {{
            "blog_id": "blog_id",
            "repost": true,
            "repost_content": "个性化的转发内容",
            "decision_reason": "转发的理由（1句，≤20字）",
            "expression_reason": "为何使用这种语气与措辞",
            "mention_reasoning": [
                {{
                "user_id": "被@用户id",
                "persona_understanding": "对该用户的简短理解",
                "mention_reason": "为何需要额外提醒该用户（1句）"
                }}
            ]
            }}
        ],
        "keep_following_blog_ids": [],
        "keep_following_reason": "为何保持关注（1句，<20字）",
        "search": false,
        "search_keyword": "搜索关键词（1句，≤20字）",
        "search_reason": "是否搜索及原因（1句）"
        }}

        输出规则：
        - 先给出理解字段，再给 decisions；默认使用中文，可中英文混合
        - decisions 中每条先按默认状态填写 repost=false，再仅对满足条件的条目改为 repost=true；
        - repost=false 时，repost_content 必须为空字符串。repost=true 时，repost_content 可以为空字符串，也可以为转发内容。
        - 根据 repost_mode 重复检验选择的评论内容，是否符合模式要求。
        {gates.similarity_kw_coaching}{gates.similarity_emb_coaching}
        {gates.freshness_coaching}"""

        mention_blog_id_set: Set[str] = {
            str(e["blog_id"]).strip()
            for e in mention_entries
            if isinstance(e, dict) and e.get("blog_id")
        }

        # Call LLM to generate the decision
        response = await self.generate_reaction(instruction, observation)

        events_to_send = []
        # Handle search decisions
        search = response.get("search", False)
        if search:
            # Send event to the algorithm module
            search_algorithm_types = await user_default_search_types(self)
            if not isinstance(search_algorithm_types, list) or not search_algorithm_types:
                raise ValueError("default_search_types must be a non-empty list")
            allowed_search_types = set(self.search_map.keys())

            for search_algorithm_type in search_algorithm_types:
                if search_algorithm_type not in allowed_search_types:
                    raise ValueError(f"Invalid algorithm type '{search_algorithm_type}'. Allowed types: {sorted(allowed_search_types)}")
                mapped_id = self.search_map.get(search_algorithm_type, "")
                if not mapped_id:
                    raise ValueError(f"No recommender agent mapped for algorithm type '{search_algorithm_type}'. Current mapping keys: {sorted(self.search_map.keys())}")
                    continue

                # Get the user profile
                profile_payload = {}
                if self.profile is not None:
                    try:
                        profile_payload = dict(self.profile.get_profile(include_private=True) or {})
                    except Exception:
                        logger.warning("Failed to serialize profile via get_profile(), fallback to empty payload.")
                logger.info(f"Sending GetSearchResultEvent to Algorithm {mapped_id} for search algorithm type {search_algorithm_type}")
                events_to_send.append(GetSearchResultEvent(
                    from_agent_id=self.profile_id,
                    to_agent_id=mapped_id,
                    timestamp=event.timestamp,
                    timestamp_duration=int(getattr(event, "timestamp_duration", 0) or 0),
                    current_step=current_step,
                    max_step=max_step,
                    user_profile=profile_payload,
                    algorithm_type=search_algorithm_type,
                    search_query=str(response.get("search_keyword") or "").strip(),
                ))

        # Handle the keep following decision
        if self.profile:
            keep_ids = response.get("keep_following_blog_ids", [])
            if isinstance(keep_ids, list) and keep_ids:
                # Only allow blog_ids within the current batch, and at most 1
                valid_keep_ids = []
                for keep_blog_id in keep_ids:
                    if str(keep_blog_id).strip() in mention_blog_id_set:
                        valid_keep_ids.append(keep_blog_id)
                if valid_keep_ids:
                    self.profile.update_data("keep_following_blog_ids", valid_keep_ids[:1])
                else:
                    self.profile.update_data("keep_following_blog_ids", [])

        # Handle the reply decision
        decisions = response.get("decisions", [])
        if not isinstance(decisions, list):
            return events_to_send

        has_reply = False
        user_id = await self.get_data("id")
        nickname = await self.get_data("nickname", "")
        ip_location = await self.get_data("ip_location", "")
        current_timestamp = event.timestamp
        current_ts = int(event.timestamp) if getattr(event, "timestamp", None) is not None else 0
        step_duration = int(getattr(event, "timestamp_duration", 0) or 0)

        for i, decision in enumerate(decisions):
            if i >= len(mention_entries):
                break
            mention_entry = mention_entries[i]
            mention_blog = mention_entry["mention_blog"]
            mention_blog_id = mention_entry["blog_id"]
            try:
                if not isinstance(decision, dict) or not decision.get("repost", False):
                    continue

                blog_id = decision.get("blog_id")
                should_repost = decision.get("repost", False)
                repost_content = decision.get("repost_content", "")
                if not blog_id or not should_repost:
                    continue

                has_reply = True

                # Check if the blog_id is valid (should be the same as mention_blog_id)
                if blog_id != mention_blog_id:
                    logger.warning(f"Blog {blog_id} does not match mention blog {mention_blog_id}, skipping")
                    continue

                # Parse the @users in the repost content, replace @id with @nickname, and return the user ID list
                mentioned_user_ids = []
                mention_reasoning = decision.get("mention_reasoning", [])
                if isinstance(mention_reasoning, list):
                    for mention_reason in mention_reasoning:
                        if isinstance(mention_reason, dict):
                            reason_user_id = mention_reason.get("user_id")
                            if reason_user_id:
                                mentioned_user_ids.append(reason_user_id)

                # Build the repost path and repost content, and the list of authors
                if mention_blog.get("reposted_blog_id"):
                    reposted_path = list(mention_blog.get("reposted_path", []))
                    reposted_path.append(blog_id)
                    
                    # Keep the order and remove duplicates to avoid repeated accumulation in the propagation chain
                    reposted_path = list(dict.fromkeys(reposted_path))
                    reposted_blog_id = mention_blog.get("reposted_blog_id")
                    reposted_user_id = mention_blog.get("user_id")
                    mention_blog_content = mention_blog.get("content", "")
                    repost_content = f"{repost_content}//@{reposted_user_id}: {mention_blog_content}"
                    
                    # blog_author_ids: split by //, then extract the user_id between @ and :
                    blog_author_ids = []
                    for part in str(repost_content).split("//"):
                        match = re.search(r"@([^:：\\s]+)\\s*[:：]", part)
                        if match:
                            blog_author_ids.append(match.group(1))
                else:
                    reposted_path = [blog_id]
                    reposted_blog_id = blog_id
                    reposted_user_id = mention_blog.get("user_id")
                    blog_author_ids = [reposted_user_id]

                # If the @ user is already in the list of authors in the repost chain, avoid duplicate reminders
                overlap_user_ids = set(uid for uid in blog_author_ids if uid)
                mentioned_user_ids = [
                    uid for uid in mentioned_user_ids if uid not in overlap_user_ids
                ]

                mention_count = len(mentioned_user_ids)

                if repost_content == "":
                    repost_content = "转发微博"

                # Add the repost
                repost_id = generate_repost_id()
                success = await self.add_env_reposts(repost_id, {
                    "blog_id": repost_id,
                    "content": repost_content,
                    "time": generate_blog_timestamp(mention_blog, current_ts, step_duration),
                    "ip_location": ip_location,
                    "user_id": user_id,
                    "nickname": nickname,
                    "at_count": mention_count,
                    "reposted_blog_id": reposted_blog_id,
                    "reposted_path": reposted_path,
                    "repost_count": 0,
                    "repost_ids": []
                })
                if not success:
                    logger.error(f"Failed to add repost to content pool (parent blog {blog_id}, new id {repost_id})")
                    continue

                # Send reminders to the authors of the repost
                for blog_author_id in blog_author_ids:
                    if blog_author_id and blog_author_id != user_id:   
                        success = await self.update_env_mention_pool(f"{blog_author_id}.{repost_id}", {
                            "action": "add",
                            "mention_message": {
                                "blog_id": repost_id,
                                "mention_type": "repost"
                            }
                        })
                        if not success:
                            logger.error(f"Failed to update mention pool for repost {repost_id} by {user_id} on blog {blog_id}")
                            continue
                        logger.info(f"Step {current_step}/{max_step}: User {user_id} reposted on blog {blog_id} by {blog_author_id}")

                # Send MentionEvent to the users that are mentioned
                if mentioned_user_ids:
                    for mentioned_user_id in mentioned_user_ids:
                        if mentioned_user_id and mentioned_user_id != user_id:  
                            success = await self.update_env_mention_pool(f"{mentioned_user_id}.{repost_id}", {
                                "action": "add",
                                "mention_message": {
                                    "blog_id": repost_id,
                                    "mention_type": "at"
                                }
                            })
                            if not success:
                                logger.error(f"Failed to update mention pool for repost {repost_id} by {user_id} on blog {blog_id}")
                                continue
                            logger.info(f"User {user_id} mentioned {mentioned_user_id} in repost on blog {blog_id}")
            finally:
                # Delete the repost from the mention pool regardless of whether it is a reply or not
                success = await self.update_env_mention_pool(f"{user_id}.{mention_blog_id}", {
                    "action": "delete",
                    "mention_message": None
                })
                if not success:
                    logger.error(f"Failed to update mention pool for blog {mention_blog_id}")
                else:
                    logger.info(f"Step {current_step}/{max_step}: User {user_id} deleted repost {mention_blog_id} from pool (processed)")

        # If there is a reply, notify the environment to update the content pool
        if has_reply:
            content_update_event = MentionSpreadingEvent(
                from_agent_id=self.profile_id,
                to_agent_id="ENV",
                timestamp=current_timestamp,
                timestamp_duration=int(getattr(event, "timestamp_duration", 0) or 0),
                current_step=current_step,
                max_step=max_step,
            )
            events_to_send.append(content_update_event)

        return events_to_send
    