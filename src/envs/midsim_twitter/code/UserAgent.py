from typing import Any, List, Optional, Dict, Set, Tuple, Union
from collections import deque
import json
import asyncio
import os
import re
import time

# SimEnv uses global lock to serialize environment updates, queueing may exceed 30s in high concurrency
_ENV_ASYNC_OP_TIMEOUT = float(os.environ.get("ONESIM_ENV_ASYNC_OP_TIMEOUT", "300"))

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
    user_social_feed_budget,
    user_social_recommendation_prob,
)
from .events import *
import random
import math

from .user_agent_gates import TweetDepthGate, UserAgentGates
from .metrics.channel_snapshots import (
    record_mentioned_tweet_ids_by_channel,
    record_recommendations_by_source_step,
)
from .utils import (
    enrich_tweet_quote_reply_chain,
    format_historical_summary,
    generate_propagation_id,
    generate_tweet_timestamp,
    pack_llm_input_chunks,
    prepare_tweet_for_llm,
    resolve_retweet_id_to_root_in_pool,
    sample_interest_tags,
    sample_mentionable_users,
    time_in_window,
    tweet_ref_key,
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
        self.register_event("StartEvent", "generate_memory_from_own_tweets")
        self.register_event("SocialRecommendationEvent", "receive_recommendation")
        self.register_event("AlgorithmRecommendationEvent", "receive_recommendation")
        self.register_event("KeepFollowingEvent", "receive_recommendation")
        self.register_event("MentionEvent", "handle_mention")

        self.register_event("AddTweetResponseEvent", "handle_add_tweet_response")
        self.register_event("MentionPoolUpdateResponseEvent", "handle_update_mention_pool_response")
        self._tweet_add_futures: Dict[str, Future] = {}
        self._mention_pool_update_futures: Dict[str, Future] = {}
        self._login_lock = asyncio.Lock()  

        # Map algorithm type to recommender agent ID
        self.recommender_map: Dict[str, str] = {
            "Random Recommendation": "recomment_agent_0001",
            "Popularity Recommendation": "recomment_agent_0002",
            "Interest Recommendation": "recomment_agent_0003",
        }
        self.search_map: Dict[str, str] = {
            "Relevant Search": "search_agent_0001"
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
            logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} detected official account by agent field, skip recommendations and mentions")
            return []

        events: List[Event] = []

        # Get current available blogs list
        current_tweets = getattr(event, "current_tweets", None) or {}
        if not isinstance(current_tweets, dict):
            current_tweets = {}

        # Algorithm recommendation stream
        # Get specified algorithm types
        fixed_algorithm_types = await user_default_algorithm_types(self)
        if not isinstance(fixed_algorithm_types, list) or not fixed_algorithm_types:
            raise ValueError("default_algorithm_types must be a non-empty list")
        allowed_algorithm_types = set(self.recommender_map.keys())

        # Get recommended note IDs
        recommended_tweet_ids = set(self.profile.get_data("recommended_tweet_ids", [])) if self.profile else set()
        recommended_seen: Set[str] = {
            str(x).strip() for x in recommended_tweet_ids if x is not None and str(x).strip()
        }

        # Get user profile
        profile_payload = {}
        if self.profile is not None:
            try:
                profile_payload = dict(self.profile.get_profile(include_private=True) or {})
            except Exception:
                logger.warning("Failed to serialize profile via get_profile(), fallback to empty payload.")
        
        # Iterate through all specified algorithm types, send events to the recommender system
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
                    current_tweets=current_tweets,
                    recommended_tweet_ids=recommended_tweet_ids,
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
                tw = mm.get("tweet")
                if not isinstance(tw, dict):
                    enriched_mentions[mention_key] = mm
                    continue
                tw_copy = dict(tw)
                tw_key = tweet_ref_key(tw_copy.get("tweet_id") or tw_copy.get("id"))
                if not tw_key:
                    tw_key = tweet_ref_key(
                        str(mention_key).split("_")[0] if "_" in str(mention_key) else str(mention_key)
                    )
                if tw_copy.get("retweeted_tweet_id"):
                    rt_key = resolve_retweet_id_to_root_in_pool(
                        tw_copy.get("retweeted_tweet_id"), current_tweets
                    )
                    inner = current_tweets.get(rt_key) if rt_key else None
                    if isinstance(inner, dict):
                        mm["tweet"] = enrich_tweet_quote_reply_chain(inner, current_tweets, tweet_ref=rt_key)
                    else:
                        mm["tweet"] = enrich_tweet_quote_reply_chain(tw_copy, current_tweets, tweet_ref=tw_key)
                else:
                    mm["tweet"] = enrich_tweet_quote_reply_chain(tw_copy, current_tweets, tweet_ref=tw_key)
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
        keep_following_tweets = {}
        if self.profile and self.profile.get_data("keep_following_tweet_ids", []) is not None:
            keep_ids = self.profile.get_data("keep_following_tweet_ids", []) or []
            if isinstance(keep_ids, (list, tuple)) and keep_ids:
                readded = 0
                for keep_tweet_id in keep_ids:
                    if keep_tweet_id in keep_following_tweets:
                        continue
                    tweet = current_tweets.get(keep_tweet_id)
                    if not isinstance(tweet, dict):
                        continue
                    tw_key = tweet_ref_key(keep_tweet_id)
                    keep_following_tweets[keep_tweet_id] = enrich_tweet_quote_reply_chain(
                        dict(tweet), current_tweets, tweet_ref=tw_key
                    )
                    readded += 1
                if readded > 0:
                    logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} re-added keep_following tweets: {readded}")
                self.profile.update_data("keep_following_tweet_ids", [])
                
        if len(keep_following_tweets) > 0:
            events.append(KeepFollowingEvent(
                from_agent_id=self.profile_id,
                to_agent_id=self.profile_id,
                timestamp=current_ts,
                timestamp_duration=int(getattr(event, "timestamp_duration", 0) or 0),
                current_step=current_step,
                max_step=max_step,
                recommendations=keep_following_tweets,
            ))

        # Social recommendation stream
        follow_ids = await self.get_data("follow_ids", [])
        follow_set = set(follow_ids) if isinstance(follow_ids, (list, tuple)) else set()

        # Get notes in the follow list that have not been viewed
        last_login = 0
        _login_cap_days = await user_social_feed_last_login_cap_days(self)
        _login_cap_sec = int(_login_cap_days * 86400) if _login_cap_days > 0 else 0
        if self.profile:
            raw = self.profile.get_data("last_login_timestamp")
            logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} last_login_timestamp: {raw}")
            if raw is not None and isinstance(raw, (int, float)) and int(raw) > 0:
                last_login = int(raw)
                if current_ts > 0 and _login_cap_sec > 0:
                    lower = current_ts - _login_cap_sec
                    if last_login < lower:
                        last_login = lower
            else:
                last_login = 0

        step_duration = int(getattr(event, "timestamp_duration", 0) or 0)
        window_end = current_ts + step_duration

        # Notes that have been recommended are not included in the follow stream
        recommendations = {}
        for tweet_id, tweet in current_tweets.items():
            if not isinstance(tweet, dict):
                continue
            author_id = tweet.get("user_id") or tweet.get("author_id")
            if author_id not in follow_set:
                continue
            t = tweet.get("time") or tweet.get("create_time")
            if t is None:
                continue
            try:
                t = int(t)
            except (TypeError, ValueError):
                continue
            if t >= 10**12:
                t //= 1000
            if last_login <= t < window_end:
                if tweet.get("replied_tweet_id") and _login_cap_sec > 0 and t < window_end - _login_cap_sec:
                    continue
                if tweet.get("retweeted_tweet_id"):
                    rt_key = resolve_retweet_id_to_root_in_pool(
                        tweet.get("retweeted_tweet_id"), current_tweets
                    )
                    inner = current_tweets.get(rt_key) if rt_key else None
                    if isinstance(inner, dict):
                        if rt_key in recommendations:
                            continue
                        rt_sid = str(rt_key).strip() if rt_key is not None else ""
                        if rt_sid and rt_sid in recommended_seen:
                            continue
                        recommendations[rt_key] = enrich_tweet_quote_reply_chain(
                            inner, current_tweets, tweet_ref=rt_key
                        )
                    else:
                        tid = str(tweet_id).strip() if tweet_id is not None else ""
                        if tid and tid in recommended_seen:
                            continue
                        recommendations[tweet_id] = dict(tweet)
                else:
                    tw_key = tweet_ref_key(tweet_id)
                    tid = str(tweet_id).strip() if tweet_id is not None else ""
                    twk = str(tw_key).strip() if tw_key else ""
                    if (tid and tid in recommended_seen) or (twk and twk in recommended_seen):
                        continue
                    recommendations[tweet_id] = enrich_tweet_quote_reply_chain(
                        dict(tweet), current_tweets, tweet_ref=tw_key
                    )

        # Update the last login timestamp of the agent
        if self.profile:
            _d = int(getattr(event, "timestamp_duration", 0) or 0)
            self.profile.update_data("last_login_timestamp", current_ts)
            logger.info(
                f"Step {current_step}/{max_step}: UserAgent {self.profile_id} update last_login_timestamp: {current_ts + _d}"
            )

        # Send SocialRecommendationEvent
        _social_recommendation_prob = await user_social_recommendation_prob(self)
        _social_feed_budget = await user_social_feed_budget(self)
        if recommendations and len(recommendations) > 0:
            if random.random() <= _social_recommendation_prob:
                rec_payload = recommendations
                n_rec = len(recommendations)
                if _social_feed_budget is not None and n_rec > _social_feed_budget:
                    sampled_keys = random.sample(list(recommendations.keys()), _social_feed_budget)
                    rec_payload = {k: recommendations[k] for k in sampled_keys}
                    logger.info(
                        f"Step {current_step}/{max_step}: UserAgent {self.profile_id} send SocialRecommendationEvent, "
                        f"sampled {_social_feed_budget} from {n_rec} recommendations"
                    )
                else:
                    logger.info(
                        f"Step {current_step}/{max_step}: UserAgent {self.profile_id} send SocialRecommendationEvent, "
                        f"length of recommendations: {n_rec}"
                    )

                events.append(SocialRecommendationEvent(
                    from_agent_id=self.profile_id,
                    to_agent_id=self.profile_id,
                    timestamp=current_ts,
                    timestamp_duration=int(getattr(event, "timestamp_duration", 0) or 0),
                    current_step=current_step,
                    max_step=max_step,
                    recommendations=rec_payload,
                ))
            else:
                logger.debug(
                    f"Step {current_step}/{max_step}: UserAgent {self.profile_id} skip SocialRecommendationEvent "
                    f"(p={_social_recommendation_prob:.0%} not drawn), "
                    f"would have sent {len(recommendations)} recommendations"
                )
        return events

    async def add_env_tweets(self, key: str, value: Any, parent_event_id: Optional[str] = None) -> bool:
        """Add tweets to the data in the environment (using distributed lock)"""    
        # Create a unique request ID
        request_id = f"agent_env_add_tweets_req_{time.time()}_{id(self)}"
        future = asyncio.Future()
        self._tweet_add_futures[request_id] = future

        # Create a tweet add event
        tweet_add_event = AddTweetEvent(
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
        lock_id = f"env_tweet_add_lock_content_pool"
        lock = await get_lock(lock_id)

        try:
            async with lock:
                from onesim.events import get_event_bus
                event_bus = get_event_bus()
                await event_bus.dispatch_event(tweet_add_event)

            try:
                if hasattr(self, '_sync_event'):
                    await asyncio.wait_for(self._sync_event.wait(), timeout=30.0)
                    return await future
                return await asyncio.wait_for(future, timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning(f"Environment tweet add timeout: {key}")
                self._tweet_add_futures.pop(request_id, None)
                return False
            except Exception as e:
                logger.error(f"Error adding tweet to environment: {e}")
                self._tweet_add_futures.pop(request_id, None)
                return False
        except Exception as e:
            logger.error(f"Error getting environment tweet add lock: {e}")
            return False

    async def handle_add_tweet_response(self, event: AddTweetResponseEvent) -> None:
        """Handle the incoming repost add response event"""
        # Check if waiting for this response
        if event.request_id in self._tweet_add_futures:
            future = self._tweet_add_futures.pop(event.request_id)

            if not future.done():
                if event.success:
                    future.set_result(True)
                else:
                    future.set_exception(ValueError(event.error or "Unknown error"))

            # If there is a sync event, set it
            if hasattr(self, '_sync_event'):
                self._sync_event.set()
                self._sync_event.clear()

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
        
        # Convert to set for calculation
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
                    info["historical_summary"] = format_historical_summary(
                        info.get("historical_summary")
                    )
                if "interest_tags" in info:
                    info["interest_tags"] = sample_interest_tags(info.get("interest_tags"), limit=3)
                hn = info.get("historical_tweets")
                if isinstance(hn, dict) and len(hn) > 2:
                    items = list(hn.items())[:2]
                    info["historical_tweets"] = dict(items)
                follows_info.append(info)
        
        # Process the fan list (including mutual)
        for user_id in fan_ids:
            rel = self.relationship_manager.get_relationship(user_id)
            # Check if rel is None, then access target_info
            if rel and rel.target_info and isinstance(rel.target_info, dict):
                info = dict(rel.target_info)
                if "historical_summary" in info:
                    info["historical_summary"] = format_historical_summary(
                        info.get("historical_summary")
                    )
                if "interest_tags" in info:
                    info["interest_tags"] = sample_interest_tags(info.get("interest_tags"), limit=3)
                hn = info.get("historical_tweets")
                if isinstance(hn, dict) and len(hn) > 2:
                    items = list(hn.items())[:2]
                    info["historical_tweets"] = dict(items)
                fans_info.append(info)
        
        # Process the mutual list (the historical notes of the mutual users are kept at most 2)
        for user_id in mutual_ids:
            rel = self.relationship_manager.get_relationship(user_id)
            # Check if rel is None, then access target_info
            if rel and rel.target_info and isinstance(rel.target_info, dict):
                info = dict(rel.target_info)
                if "historical_summary" in info:
                    info["historical_summary"] = format_historical_summary(
                        info.get("historical_summary")
                    )
                if "interest_tags" in info:
                    info["interest_tags"] = sample_interest_tags(info.get("interest_tags"), limit=3)
                hn = info.get("historical_tweets")
                if isinstance(hn, dict) and len(hn) > 2:
                    items = list(hn.items())[:2]
                    info["historical_tweets"] = dict(items)
                mutual_info.append(info)

        mentionable_info["follows"] = follows_info
        mentionable_info["fans"] = fans_info
        mentionable_info["mutual"] = mutual_info
        
        return mentionable_info

    async def generate_memory_from_own_tweets(self, event: Event) -> List[Event]:
        """For user to generate memory from their own tweets"""
        current_step = getattr(event, "current_step", 1)
        max_step = getattr(event, "max_step", 8)
        user_id = await self.get_data("id")
        if not user_id:
            logger.error(f"Step {current_step}/{max_step}: User {user_id} failed to get user_id")
            return []

        current_ts = int(event.timestamp) if getattr(event, "timestamp", None) is not None else 0
        duration = int(getattr(event, "timestamp_duration", 0) or 0)

        current_tweets = getattr(event, "current_tweets", None) or {}
        if not isinstance(current_tweets, dict) or not current_tweets:
            logger.error(f"Step {current_step}/{max_step}: User {user_id} failed to get current_tweets")
            return []

        # Filter own tweets (no retweet, quote, reply)
        own_entries: List[Tuple[str, Dict[str, Any]]] = []
        for tweet_id, tweet in current_tweets.items():
            if not isinstance(tweet, dict):
                continue
            if (tweet.get("user_id") or tweet.get("author_id")) != user_id:
                continue
            own_entries.append((str(tweet_id), tweet))

        lo = float(current_ts)
        dur_sec = float(duration)
        cap = getattr(event, "simulation_cap_timestamp", None)
        if cap is not None:
            cap_f = float(cap)
            if cap_f >= 10**12:
                cap_f = cap_f / 1000.0
            hi = min(lo + dur_sec, cap_f)
        else:
            hi = lo + dur_sec
        if hi <= lo:
            logger.error(f"Step {current_step}/{max_step}: User {user_id} failed to get own_entries")
            return []

        own_entries = [
            (tid, tw)
            for tid, tw in own_entries
            if time_in_window(tw, lo, hi)
        ]
        if not own_entries:
            return []

        def _is_pure_original(tw: Dict[str, Any]) -> bool:
            if str(tw.get("retweeted_tweet_id") or "").strip():
                return False
            if str(tw.get("quoted_tweet_id") or "").strip():
                return False
            if str(tw.get("replied_tweet_id") or tw.get("replyed_tweet_id") or "").strip():
                return False
            return True

        own_originals = [(tid, tw) for tid, tw in own_entries if _is_pure_original(tw)]

        if self.profile:
            existing_recommended = self.profile.get_data("recommended_tweet_ids", []) or []
            if not isinstance(existing_recommended, list):
                existing_recommended = list(existing_recommended) if existing_recommended else []
            own_ids = [str(tid) for tid, _ in own_entries if tid]
            if own_ids:
                merged = list(dict.fromkeys(existing_recommended + own_ids))
                self.profile.update_data("recommended_tweet_ids", merged)

        own_tweets_for_prompt: List[Dict[str, Any]] = []
        for tweet_id, tweet in own_originals:
            own_tweets_for_prompt.append({
                "tweet_id": tweet_id,
                "content": tweet.get("content", ""),
            })

        # Build the instruction and observation for the memory generation
        if own_tweets_for_prompt:
            instruction = (
                "You are reflecting on your recent posts. From these tweets, distill one first-person memory entry "
                "summarizing: topics you keep following, your voice/style, and reusable phrasing for future interactions."
            )
            observation = f"Your recent posts:\n{json.dumps(own_tweets_for_prompt, ensure_ascii=False, indent=2)}"
            reaction = {
                "task": "self_reflection_on_own_tweets",
                "own_tweet_count": len(own_tweets_for_prompt),
                "highlights": own_tweets_for_prompt,
            }
            try:
                memory_text = await self.generate_memory(instruction, observation, reaction)
                if memory_text:
                    logger.info(f"Step {current_step}/{max_step}: User {user_id} generated self-memory from {len(own_tweets_for_prompt)} own tweets, memory_text: {memory_text}")
            except Exception as e:
                logger.error(f"Step {current_step}/{max_step}: User {user_id} failed to generate memory from own tweets: {e}")

        followup_prob = float(os.environ.get("TWITTER_SELF_FOLLOWUP_PROB", "0.001"))
        if followup_prob <= 0.0:
            return []

        max_batch = max(1, int(os.environ.get("TWITTER_SELF_FOLLOWUP_MAX_BATCH", "16")))
        pool = [
            (tid, tw)
            for tid, tw in own_entries
            if not str(tw.get("retweeted_tweet_id") or "").strip()
        ]
        random.shuffle(pool)
        subs: List[Tuple[Any, Dict[str, Any]]] = [
            (tid, tw) for tid, tw in pool if random.random() < followup_prob
        ]
        if len(subs) > max_batch:
            subs = random.sample(subs, max_batch)
        if not subs:
            return []

        # Decide whether to add a repost to the own tweets
        user_nickname = await self.get_data("nickname", "") or ""
        user_username = await self.get_data("username", "") or ""
        follow_ids = await self.get_data("follow_ids", [])
        fan_ids = await self.get_data("fan_ids", [])
        current_timestamp = event.timestamp
        mentionable_users = self._get_mentionable_users(follow_ids, fan_ids)
        mention_prompt_users = sample_mentionable_users(mentionable_users, limit=5)
        mentionable_users_str = json.dumps(mention_prompt_users, ensure_ascii=False, indent=2)

        chunk: Dict[str, Dict[str, Any]] = {str(tid): tw for tid, tw in subs}
        posts_payload = [
            {
                "tweet_id": str(tid),
                "content": (tw.get("content") or "")[:800],
            }
            for tid, tw in subs
        ]
        n_posts = len(posts_payload)
        recommendations_str = json.dumps({"self_tweets": posts_payload}, ensure_ascii=False, indent=2)
        observation = f"""[Scenario] Rare self follow-up on your own tweets (one batched decision)
        {recommendations_str}

        Users you may @:
        {mentionable_users_str}"""

        instruction = f"""This is a rare **batched** decision to add follow-up engagement on your own past tweets ({n_posts} items).
        Use Observation.self_tweets and memory: for each item, decide if it is worth engaging again (new facts, new developments, or a change of stance).
        If yes: propagation=true, propagation_type is "reply" / "quote" / "retweet" (prefer reply or quote); for retweet, propagation_content must be "".
        If no: propagation=false, propagation_type "".
        Do not use hashtags (#...) in propagation_content.

        Same schema as the main feed decision; **decisions must contain exactly {n_posts} objects**, in the same order as self_tweets; item i's tweet_id must equal self_tweets[i].tweet_id.

        ```json
        {{
        "persona_understanding": "1-2 sentences",
        "content_understanding": "1-2 sentences",
        "source_understanding": "Your historical tweets (batched self follow-up)",
        "memory_reflection": "2-3 sentences",
        "decisions": [
            {{
            "tweet_id": "<must match self_tweets[0].tweet_id>",
            "propagation": false,
            "propagation_type": "",
            "propagation_content": "",
            "decision_reason": "",
            "expression_reason": "",
            "mention_reasoning": []
            }}
        ],
        "keep_following_tweet_ids": [],
        "keep_following_reason": "",
        "search": false,
        "search_keyword": "keyword for search, <len<=20 chars>",
        "search_reason": ""
        }}
        ```
        (Expand decisions to {n_posts} objects; do not omit.)
        When propagation=true, propagation_type ∈ {{"retweet","reply","quote"}}."""

        env = getattr(self, "env", None)
        if env is not None and hasattr(env, "notify_agent_busy"):
            await env.notify_agent_busy()
        try:
            response = await self.generate_reaction(instruction, observation)
        finally:
            if env is not None and hasattr(env, "notify_agent_idle"):
                await env.notify_agent_idle()

        if not isinstance(response, dict):
            return []
        decisions_raw = response.get("decisions", [])
        if not isinstance(decisions_raw, list):
            return []

        by_tid: Dict[str, Dict[str, Any]] = {}
        for d in decisions_raw:
            if isinstance(d, dict):
                tk = str(d.get("tweet_id") or "").strip()
                if tk:
                    by_tid[tk] = d

        for tweet_id, tweet in subs:
            tid_key = str(tweet_id)
            decision = by_tid.get(tid_key)
            if not isinstance(decision, dict):
                continue
            should_propagation = decision.get("propagation", False)
            propagation_type = decision.get("propagation_type", "")
            propagation_content = decision.get("propagation_content", "")
            if not should_propagation:
                continue
            if propagation_type != "retweet" and not (propagation_content or "").strip():
                continue
            if tid_key not in chunk:
                continue

            mentioned_user_ids: List[str] = []
            mention_reasoning = decision.get("mention_reasoning", [])
            if isinstance(mention_reasoning, list):
                for mention_reason in mention_reasoning:
                    if isinstance(mention_reason, dict):
                        muid = mention_reason.get("user_id")
                        if muid:
                            mentioned_user_ids.append(muid)

            path_author_user_ids = self._collect_chain_user_ids(tid_key, chunk)
            path_author_set = {str(x).strip() for x in path_author_user_ids if x}
            filtered_mention_ids: Set[str] = set()
            for x in mentioned_user_ids:
                if x is not None:
                    sx = str(x).strip()
                    if sx and sx not in path_author_set:
                        filtered_mention_ids.add(sx)
            filtered_mention_ids.discard(user_id)
            mention_count = len(filtered_mention_ids)

            propagation_id = generate_propagation_id()

            if propagation_type == "retweet":
                success = await self.add_env_tweets(propagation_id, {
                    "tweet_id": propagation_id,
                    "content": "",
                    "time": current_timestamp if isinstance(current_timestamp, (int, float)) else 0,
                    "user_id": user_id,
                    "nickname": user_nickname,
                    "username": user_username,
                    "mention_count": mention_count,
                    "retweeted_tweet_id": tid_key,
                })
                if not success:
                    logger.error(f"Step {current_step}/{max_step}: self_followup retweet failed for {tid_key}")
                    continue
            elif propagation_type == "quote":
                success = await self.add_env_tweets(propagation_id, {
                    "tweet_id": propagation_id,
                    "content": propagation_content,
                    "time": current_timestamp if isinstance(current_timestamp, (int, float)) else 0,
                    "user_id": user_id,
                    "nickname": user_nickname,
                    "username": user_username,
                    "mention_count": mention_count,
                    "quoted_tweet_id": tid_key,
                })
                if not success:
                    logger.error(f"Step {current_step}/{max_step}: self_followup quote failed for {tid_key}")
                    continue
            elif propagation_type == "reply":
                success = await self.add_env_tweets(propagation_id, {
                    "tweet_id": propagation_id,
                    "content": propagation_content,
                    "time": current_timestamp if isinstance(current_timestamp, (int, float)) else 0,
                    "user_id": user_id,
                    "nickname": user_nickname,
                    "username": user_username,
                    "mention_count": mention_count,
                    "replied_tweet_id": tid_key,
                })
                if not success:
                    logger.error(f"Step {current_step}/{max_step}: self_followup reply failed for {tid_key}")
                    continue
            else:
                continue

            path_notify_ids = sorted(
                {
                    s
                    for s in (str(x).strip() for x in path_author_user_ids if x)
                    if s and s != user_id
                }
            )
            for target_uid in path_notify_ids:
                ok = await self.update_env_mention_pool(f"{target_uid}.{propagation_id}", {
                    "action": "add",
                    "mention_message": {
                        "tweet_id": tid_key,
                        "mention_type": propagation_type,
                    },
                })
                if not ok:
                    logger.error(f"Step {current_step}/{max_step}: mention_pool update failed for self_followup {propagation_id} -> {target_uid}")
                else:
                    logger.info(f"Step {current_step}/{max_step}: User {user_id} self_followup {propagation_type} on own tweet {tid_key} -> {propagation_id}")

            if filtered_mention_ids:
                for mentioned_user_id in filtered_mention_ids:
                    if mentioned_user_id and mentioned_user_id != user_id:
                        ok = await self.update_env_mention_pool(
                            f"{mentioned_user_id}.{propagation_id}",
                            {
                                "action": "add",
                                "mention_message": {
                                    "tweet_id": tid_key,
                                    "mention_type": "at",
                                },
                            },
                        )
                        if not ok:
                            logger.error(f"Step {current_step}/{max_step}: mention_pool @ failed for self_followup {propagation_id}")
        return []

    def _filter_recommendations(self, recommendations: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Remove duplicate recommendations"""
        if not recommendations:
            return {}
        
        # Get the set of recommended content IDs (from profile)
        recommended_tweet_ids = set(self.profile.get_data("recommended_tweet_ids", [])) if self.profile else set()
        
        # Remove duplicate recommendations 
        filtered = {}
        new_tweet_ids = []
        
        for tweet_id, rec in recommendations.items():
            if not isinstance(rec, dict):
                logger.warning(f"Recommendation {tweet_id} is not a dictionary")
                continue
            
            # If the tweet ID does not exist or has already been recommended, skip
            if not tweet_id or tweet_id in recommended_tweet_ids:
                logger.info(f"Recommendation {tweet_id} is already recommended")
                continue
            
            filtered[tweet_id] = rec
            new_tweet_ids.append(tweet_id)
            logger.info(f"Recommendation {tweet_id} is added to filtered list")
        
        return filtered
        
    def _add_recommendations(self, recommendations: Dict[str, Any]) -> None:
        """Add the tweet_ids of the current recommendations to profile.recommended_tweet_ids."""
        if not recommendations:
            return
        recommended_tweet_ids = set(self.profile.get_data("recommended_tweet_ids", [])) if self.profile else set()
        new_tweet_ids = []
        
        for tweet_id in recommendations.keys():
            new_tweet_ids.append(tweet_id)
        
        # Add the new recommended content IDs to the recommended list
        if new_tweet_ids and self.profile:
            all_recommended = list(recommended_tweet_ids) + new_tweet_ids
            self.profile.update_data("recommended_tweet_ids", all_recommended)

    @staticmethod
    def _nested_tweet_graph_key(tw: Dict[str, Any]) -> str:
        """Nested tweet graph dedup key (aligned with enrich_tweet_quote_reply_chain subtree)."""
        k = tweet_ref_key(tw.get("tweet_id") or tw.get("id"))
        if k:
            return k
        return f"__obj_{id(tw)}"

    @staticmethod
    def _quote_reply_chain_hint_strip_replies(tw: Dict[str, Any]) -> Dict[str, Any]:
        """Remove the `replies` field from each layer of the quote/reply chain"""
        return {k: v for k, v in tw.items() if k != "replies"}

    @staticmethod
    def _quote_reply_chain_hint(
        tweet: Dict[str, Any],
        *,
        my_id: Optional[str],
        recommended_tweet_ids: Set[str],
        max_nodes: int = 64,
    ) -> str:
        """Detect if the quote/reply chain includes a tweet you posted or a tweet that appeared in your recommendation feed before"""
        if not isinstance(tweet, dict):
            return ""

        my_s = str(my_id).strip() if my_id else ""
        rec = {str(x).strip() for x in recommended_tweet_ids if x}

        has_self_post = False
        has_seen_before = False
        seen_nodes: Set[str] = set()
        dq: deque[Dict[str, Any]] = deque([UserAgent._quote_reply_chain_hint_strip_replies(tweet)])

        while dq and len(seen_nodes) < max_nodes:
            tw = dq.popleft()
            nk = UserAgent._nested_tweet_graph_key(tw)
            if nk in seen_nodes:
                continue
            seen_nodes.add(nk)

            tid_k = tweet_ref_key(tw.get("tweet_id") or tw.get("id"))
            if tid_k and tid_k in rec:
                has_seen_before = True

            uid = tw.get("user_id")
            if my_s and uid is not None and str(uid).strip() == my_s:
                has_self_post = True

            for child_key in ("replied_tweet", "quoted_tweet"):
                nested = tw.get(child_key)
                if isinstance(nested, dict):
                    dq.append(UserAgent._quote_reply_chain_hint_strip_replies(nested))

        parts: List[str] = []
        if has_self_post:
            parts.append(
                "[Hint: The quote/reply chain includes a tweet you posted; mind context and avoid talking to yourself.]"
            )
        if has_seen_before:
            parts.append(
                "[Hint: Some tweets on this chain appeared in your recommendation feed before—do not treat them as brand-new.]"
            )
        return "\n".join(parts)

    @staticmethod
    def _collect_chain_user_ids(
        tweet_or_id: Any,
        pool: Optional[Dict[str, Any]],
        *,
        max_nodes: int = 64,
    ) -> List[str]:
        """Collect the user_ids on the chain"""
        if pool is None:
            root_tweet = tweet_or_id if isinstance(tweet_or_id, dict) else None
        else:
            key = tweet_ref_key(tweet_or_id)
            if not key or not isinstance(pool, dict):
                return []
            root_tweet = pool.get(key)

        if not isinstance(root_tweet, dict):
            return []

        out: List[str] = []
        seen_uids: Set[str] = set()
        seen_nodes: Set[str] = set()
        dq: deque[Dict[str, Any]] = deque([root_tweet])
        while dq and len(seen_nodes) < max_nodes:
            tw = dq.popleft()
            nk = UserAgent._nested_tweet_graph_key(tw)
            if nk in seen_nodes:
                continue
            seen_nodes.add(nk)
            uid = tw.get("user_id")
            if uid is not None:
                s = str(uid).strip()
                if s and s not in seen_uids:
                    seen_uids.add(s)
                    out.append(s)
            for child_key in ("replied_tweet", "quoted_tweet"):
                nested = tw.get(child_key)
                if isinstance(nested, dict):
                    dq.append(nested)
        return out

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
            logger.warning(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} ""unknown recommendation event {evt_cls}, skip")
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
        prior_recommended_raw = set(self.profile.get_data("recommended_tweet_ids", [])) if self.profile else set()
        chain_hint_seen_tweet_ids = {str(x).strip() for x in prior_recommended_raw if x}
        # Record the viewed notes
        self._add_recommendations(recommendations)

        # Get user information and mentionable users
        user_id = await self.get_data("id")
        user_nickname = await self.get_data("nickname", "")
        user_username = await self.get_data("username", "")
        current_timestamp = event.timestamp
        current_ts = int(event.timestamp) if getattr(event, "timestamp", None) is not None else 0
        step_duration = int(getattr(event, "timestamp_duration", 0) or 0)
        follow_ids = await self.get_data("follow_ids", [])
        fan_ids = await self.get_data("fan_ids", [])
        mentionable_users = self._get_mentionable_users(follow_ids, fan_ids)
        mention_prompt_users = sample_mentionable_users(mentionable_users, limit=5)

        max_chunk_units = max(1, int(os.environ.get("ONESIM_REC_CHUNK_MAX_UNITS", "3")))
        chunks = pack_llm_input_chunks(recommendations, max_chunk_units)

        events_to_send = []
        has_search = False

        content_pool = await self.get_env_data("content_pool", {}) or {}
        if not isinstance(content_pool, dict):
            content_pool = {}
        seeds_raw = await self.get_env_data("seed_root_tweet_ids", []) or []
        seed_ids = {str(x).strip() for x in seeds_raw if str(x).strip()}

        for chunk in chunks:
            chunk_for_llm: Dict[str, Any] = {}
            for tid, tw in chunk.items():
                inner = prepare_tweet_for_llm(tw)
                tid_key = tweet_ref_key(tid) or (
                    tweet_ref_key(tw.get("tweet_id") or tw.get("id"))
                    if isinstance(tw, dict)
                    else None
                )
                dep = (
                    TweetDepthGate.hop_edges_to_env_seed_root(str(tid_key), content_pool, seed_ids)
                    if tid_key
                    else None
                )
                if isinstance(inner, dict):
                    inner = dict(inner)
                    inner["n_hop"] = dep
                chunk_for_llm[tid] = inner
            recommendations_str = json.dumps(chunk_for_llm, ensure_ascii=False, indent=2)
            mentionable_users_str = json.dumps(mention_prompt_users, ensure_ascii=False, indent=2)

            # Label recommendation source; if the recommendation contains content published by the user, label it as "自己发布"
            has_self_tweet = any(
                isinstance(tweet, dict) and tweet.get("user_id") == user_id
                for tweet in chunk.values()
            )
            if has_self_tweet:
                source_name = ("Your own post [Hint: you authored this tweet; mind context and avoid replying as if to yourself.]")
            else:
                source_name = (
                    "Following feed (from accounts you follow)"
                    if source_type == "social"
                    else "Algorithmic recommendations"
                )

            chain_hint_lines: List[str] = []
            for rec_tid, rec_tweet in chunk.items():
                if not isinstance(rec_tweet, dict):
                    continue
                tid_key = tweet_ref_key(rec_tid) or tweet_ref_key(
                    rec_tweet.get("tweet_id") or rec_tweet.get("id")
                )
                if not tid_key:
                    continue
                one_hint = self._quote_reply_chain_hint(
                    rec_tweet,
                    my_id=user_id,
                    recommended_tweet_ids=chain_hint_seen_tweet_ids,
                )
                if one_hint:
                    chain_hint_lines.append(f"(tweet_id={tid_key}) {one_hint}")
            quote_reply_chain_hint = "\n".join(chain_hint_lines)

            if quote_reply_chain_hint:
                source_name = f"{source_name}\nChain tweets (quote/reply):\n{quote_reply_chain_hint}"

            depth_coaching = TweetDepthGate.coaching_for_recommendation_chunk(
                chunk, content_pool, seed_ids
            )

            observation = f"""[Scenario] You are scrolling a phone feed: skim most items; only rarely stop to type a line. You are not running an experiment task or writing a media analysis.

            Feed source: {source_name}

            Recommended content:
            {recommendations_str}

            Users you may @:
            {mentionable_users_str}
            """

            gates = await UserAgentGates(self).build_recommendation_coaching(
                chunk, current_timestamp, content_pool=content_pool
            )

            instruction = f"""Using the user's Profile, historical_summary, memory, and the recommended tweets, produce tweet decisions and text.

            Step 1 — Default each recommendation
            - "propagation": false
            - "propagation_type": "" (empty string: no propagation type chosen yet, same as no engagement)
            - "propagation_content": ""

            **Only steps 2/3/4 decide propagation counts; do not change them because of step 6.**
            {gates.memory_coaching}

            Step 2 — Interest gate (still no propagation_type/mode)d
            - Read all items in this batch; set propagation=true only if **all** of the following hold:
                {gates.memory_rec}
                1. Relevant to Profile/historical_summary/memory;
                2. Relationship and scene fit (prefer follows);
                3. You see heat/reply value and clear intent to respond or amplify;
                4. If the tweet is already in a quote/reply chain (replied_tweet or quoted_tweet non-empty), lean toward propagation=false, further propagation needs clear new stance, explanation, context, or extension;
            - Items that fail interest: not in the candidate pool, propagation=false.

            Step 3 — At most {gates.k_diff_targets} target tweets
            - From the sorted candidate list (interest-passing items first), pick **at most** {gates.k_diff_targets} distinct tweet_ids to engage with; **zero is allowed** if nothing qualifies.
            - Never exceed {gates.k_diff_targets} different tweet_ids with propagation=true in this batch.

            Step 4 — At most {gates.k_same_target} engagement(s) per target tweet_id
            - For each target tweet_id you engage with, output **at most** {gates.k_same_target} propagation=true rows (fewer if one line is enough).
            - If {gates.k_same_target}=1: at most one engagement row per target tweet_id.
            - If {gates.k_same_target}>=2: you may use 2+ rows only when each adds something (new point / correction / stronger emotion); no duplicate lines; **never more than {gates.k_same_target}** rows for the same tweet_id.

            Step 5 — propagation_type per engagement
            {gates.propagation_type_coaching}

            If you choose empty-text forward, propagation_type must be "retweet"; skip steps 6 and 7.

            Step 6 — propagation_mode (persona)
            - propagation_mode ∈ {{analysis/event narrative/advice, emotion/memes/roast, question/chat, emoji/placeholder}}; reflect in expression_reason.
                - analysis/event narrative/advice: facts, mechanisms, details, calls to action; denser; max ~40 words.
                - emotion/memes/roast: attitude-first; max ~25 words.
                - question/chat: push the thread; max ~30 words.
                - emoji/placeholder: emoji or very short; max ~1 word. e.g. 🤯😇👌😅🤣😨😰😱🙈🤩😂
            - Output style:
                - Prefer the user's locale; default English.
                - Match Profile and historical_summary voice;
                - Avoid canned phrases ("you're so right", "indeed", "hope everyone…");
                - **No hashtags (#...) in propagation_content.**

            Step 7 — @ mentions
            - @only if the user is in the allow-list, strongly relevant, and not the tweet author / parent retweet author; else mention_reasoning is [].

            Step 8 — keep_following_tweet_ids
            - Only if highly interested and worth tracking → non-empty; else [].

            Step 9 — search
            - search=true only if highly interested, info clearly insufficient, and memory has nothing useful; else false.

            Return JSON (field order fixed):
            {{
            "persona_understanding": "1-2 sentences: role, interests, voice",
            "content_understanding": "1-2 sentences: relevance and angle",
            "source_understanding": "following feed = social ties; algorithm = unknown authors; your own post = self-authored",
            "memory_reflection": "{gates.memory_ref}",
            "decisions": [
                {{
                "tweet_id": "tweet_id",
                "propagation": false,
                "propagation_type": "",
                "propagation_content": "",
                "decision_reason": "why skip (1 short line, <20 chars)",
                "expression_reason": "",
                "mention_reasoning": []
                }},
                {{
                "tweet_id": "tweet_id",
                "propagation": true,
                "propagation_type": "retweet",
                "propagation_content": "",
                "decision_reason": "why engage (1 short line, <20 chars)",
                "expression_reason": "",
                "mention_reasoning": []
                }},
                {{
                "tweet_id": "tweet_id",
                "propagation": true,
                "propagation_type": "reply",
                "propagation_content": "short reply (quote text if quoting)",
                "decision_reason": "why engage (1 short line, <20 chars)",
                "expression_reason": "tone; substantive/micro if applicable (1 line)",
                "mention_reasoning": [
                    {{
                    "user_id": "mentioned user id",
                    "persona_understanding": "brief read of that user",
                    "mention_reason": "why ping them (1 short line, <20 chars)"
                    }}
                ]
                }}
            ],
            "keep_following_tweet_ids": [],
            "keep_following_reason": "why keep following (1 short line, <20 chars)",
            "search": false or true,
            "search_keyword": "keyword for search, <len<=20 chars>",
            "search_reason": "search or not and why (1 line)",
            }}

            Output rules:
            - Fill understanding fields first, then decisions;
            - Each decision row starts as propagation=false / propagation_type ""; flip to true only where rules pass (single type preferred; "reply|quote" allowed but not ideal);
            - If propagation=false, propagation_type must be ""; if propagation=true and propagation_type is exactly "retweet", propagation_content must be "".
            - Align propagation_content with propagation_mode and persona; avoid boilerplate.
            {gates.similarity_kw_coaching}{gates.similarity_emb_coaching}
            {depth_coaching}{gates.freshness_coaching}"""

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
                        raise ValueError(
                            f"Invalid algorithm type '{search_algorithm_type}'. "
                            f"Allowed types: {sorted(allowed_search_types)}"
                        )
                    mapped_id = self.search_map.get(search_algorithm_type, "")
                    if not mapped_id:
                        raise ValueError(
                            f"No recommender agent mapped for algorithm type '{search_algorithm_type}'. "
                            f"Current mapping keys: {sorted(self.search_map.keys())}"
                        )
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
                keep_ids = response.get("keep_following_tweet_ids", [])
                if isinstance(keep_ids, list) and keep_ids:
                    # Only allow tweet_ids within the current batch
                    valid_keep_ids = []
                    for keep_tweet_id in keep_ids:
                        if keep_tweet_id in chunk:
                            valid_keep_ids.append(keep_tweet_id)
                    if valid_keep_ids:
                        self.profile.update_data("keep_following_tweet_ids", valid_keep_ids[:1])
                    else:
                        self.profile.update_data("keep_following_tweet_ids", [])

            # Handle diffusion decisions
            decisions = response.get("decisions", [])
            if not isinstance(decisions, list):
                continue
       
            # Handle each decision: update propagation count and propagation content
            for decision in decisions:
                if not isinstance(decision, dict):
                    continue
                    
                tweet_id = decision.get("tweet_id")
                should_propagation = decision.get("propagation", False)
                propagation_type = decision.get("propagation_type", "")
                propagation_content = decision.get("propagation_content", "")

                if not tweet_id or not should_propagation:
                    continue

                # If the retweet has no content, propagation_content should be ""
                if propagation_type != "retweet" and not (propagation_content or "").strip():
                    continue

                # Check if the tweet_id is valid
                if tweet_id not in chunk:
                    logger.warning(f"Step {current_step}/{max_step}: Tweet {tweet_id} not found in recommendations")
                    continue

                tweet = chunk[tweet_id]
                if not isinstance(tweet, dict):
                    tweet = {}

                # Parse the @users in the content, replace @id with @nickname, and return the list of user IDs
                mentioned_user_ids = []
                mention_reasoning = decision.get("mention_reasoning", [])
                if isinstance(mention_reasoning, list):
                    for mention_reason in mention_reasoning:
                        if isinstance(mention_reason, dict):
                            muid = mention_reason.get("user_id")
                            if muid:
                                mentioned_user_ids.append(muid)

                path_author_user_ids = self._collect_chain_user_ids(tweet_id, chunk)
                path_author_set = {str(x).strip() for x in path_author_user_ids if x}
                
                # Remove the authors in the quote/reply chain from the @list to avoid duplicate counting and duplicate reminders
                filtered_mention_ids: Set[str] = set()
                for x in mentioned_user_ids:
                    if x is not None:
                        sx = str(x).strip()
                        if sx and sx not in path_author_set:
                            filtered_mention_ids.add(sx)
                filtered_mention_ids.discard(user_id)
                mention_count = len(filtered_mention_ids)
                
                # If the propagation is true, add the propagation
                propagation_id = generate_propagation_id()
                if propagation_type == "retweet":
                    success = await self.add_env_tweets(propagation_id, {
                        "tweet_id": propagation_id,
                        "content": "",
                        "time": generate_tweet_timestamp(tweet, current_ts, step_duration),
                        "user_id": user_id,
                        "nickname": user_nickname,
                        "username": user_username,
                        "mention_count": mention_count,
                        "retweeted_tweet_id": tweet_id
                    })
                    if not success:
                        logger.error(f"Failed to add retweet to tweet {tweet_id}")
                        continue
                elif propagation_type == "quote":
                    success = await self.add_env_tweets(propagation_id, {
                        "tweet_id": propagation_id,
                        "content": propagation_content,
                        "time": generate_tweet_timestamp(tweet, current_ts, step_duration),
                        "user_id": user_id,
                        "nickname": user_nickname,
                        "username": user_username,
                        "mention_count": mention_count,
                        "quoted_tweet_id": tweet_id
                    })
                    if not success:
                        logger.error(f"Failed to add quote to tweet {tweet_id}")
                        continue
                elif propagation_type == "reply":
                    success = await self.add_env_tweets(propagation_id, {
                        "tweet_id": propagation_id,
                        "content": propagation_content,
                        "time": generate_tweet_timestamp(tweet, current_ts, step_duration),
                        "user_id": user_id,
                        "nickname": user_nickname,
                        "username": user_username,
                        "mention_count": mention_count,
                        "replied_tweet_id": tweet_id
                    })
                    if not success:
                        logger.error(f"Failed to add reply to tweet {tweet_id}")
                        continue

                # Send reminders to the authors in the quote/reply chain 
                path_notify_ids = sorted(
                    {
                        s
                        for s in (str(x).strip() for x in path_author_user_ids if x)
                        if s and s != user_id
                    }
                )
                for target_uid in path_notify_ids:
                    success = await self.update_env_mention_pool(f"{target_uid}.{propagation_id}", {
                        "action": "add",
                        "mention_message": {
                            "tweet_id": tweet_id,
                            "mention_type": propagation_type
                        }
                    })
                    if not success:
                        logger.error(f"Failed to update mention pool for {propagation_type} {propagation_id} by {user_id} -> chain author {target_uid} on tweet {tweet_id}")
                    else:
                        logger.info(f"Step {current_step}/{max_step}: User {user_id} {propagation_type} to tweet {tweet_id}")

                # Send MentionEvent to the users that are @ed
                if filtered_mention_ids:
                    for mentioned_user_id in filtered_mention_ids:
                        if mentioned_user_id and mentioned_user_id != user_id:  
                            success = await self.update_env_mention_pool(f"{mentioned_user_id}.{propagation_id}", {
                                "action": "add",
                                "mention_message": {
                                    "tweet_id": tweet_id,
                                    "mention_type": "at"
                                }
                            })
                            if not success:
                                logger.error(f"Failed to update mention pool for tweet {tweet_id} by {user_id}")
                                continue
                            logger.info(f"Step {current_step}/{max_step}: User {user_id} mentioned {mentioned_user_id} in tweet {tweet_id}")
                    
       
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

        recommended_raw = set(self.profile.get_data("recommended_tweet_ids", [])) if self.profile else set()
        recommended_tweet_ids = {str(x).strip() for x in recommended_raw if x}

        content_pool = await self.get_env_data("content_pool", {}) or {}
        if not isinstance(content_pool, dict):
            content_pool = {}
        seeds_raw = await self.get_env_data("seed_root_tweet_ids", []) or []
        seed_ids = {str(x).strip() for x in seeds_raw if str(x).strip()}

        # Build the reminder information
        mention_entries: List[Dict[str, Any]] = []
        for mention_key, mention_message in mentions.items():
            mention_type = mention_message.get("mention_type", "retweet")
            if mention_type == "retweet":
                continue

            mention_tweet = mention_message.get("tweet")
            if not isinstance(mention_tweet, dict):
                continue
            mentioner_id = mention_tweet.get("user_id")
            mentioner_nickname = mention_tweet.get("nickname", "")
            mentioner_username = mention_tweet.get("username", "")
            
            relationship_type = self._check_relationship(my_id, mentioner_id, follow_ids, fan_ids)
            if relationship_type == "mutual":
                relationship_hint = " (mutual follow)"
            elif relationship_type == "follow":
                relationship_hint = " (you follow them)"
            else:
                relationship_hint = ""

            if mention_type in ("quote"):
                mention_action = f"{mentioner_nickname}{relationship_hint} quoted your tweet"
                content_label = "Quoted content"
            elif mention_type == "reply":
                mention_action = f"{mentioner_nickname}{relationship_hint} replied to your tweet"
                content_label = "Reply content"
            else:
                mention_action = f"{mentioner_nickname}{relationship_hint} mentioned your tweet ({mention_type})"
                content_label = "Related content"

            if relationship_hint:
                relationship_source = f"Relationship with {mentioner_nickname}: {relationship_hint}"
            else:
                relationship_source = f"Relationship with {mentioner_nickname}: stranger / no strong tie"

            tweet_id = mention_tweet.get("tweet_id") or (mention_key.split("_")[0] if "_" in str(mention_key) else str(mention_key))
            quote_reply_chain_hint = self._quote_reply_chain_hint(
                mention_tweet,
                my_id=my_id,
                recommended_tweet_ids=recommended_tweet_ids,
            )
            dep_m = TweetDepthGate.hop_edges_to_env_seed_root(str(tweet_id), content_pool, seed_ids)
            root_m = TweetDepthGate.resolve_tweet_to_env_seed_root(str(tweet_id), content_pool, seed_ids)
            env_propagation_depth_hint = TweetDepthGate.format_propagation_depth_hint(dep_m, root_m)
            mention_entries.append({
                "mention_key": mention_key,
                "mention_tweet": mention_tweet,
                "tweet_id": tweet_id,
                "mentioner_id": mentioner_id,
                "mentioner_nickname": mentioner_nickname,
                "mentioner_username": mentioner_username,
                "mention_type": mention_type,
                "mention_action": mention_action,
                "content_label": content_label,
                "relationship_source": relationship_source,
                "quote_reply_chain_hint": quote_reply_chain_hint,
                "env_propagation_depth_hint": env_propagation_depth_hint,
                "n_hop": dep_m,
            })

        if not mention_entries:
            return []

        # Dynamic chunking by LLM input weight
        max_chunk_units = max(1, int(os.environ.get("ONESIM_REC_CHUNK_MAX_UNITS", "3")))
        tw_map = {e["mention_key"]: e["mention_tweet"] for e in mention_entries}
        chunk_dicts = pack_llm_input_chunks(tw_map, max_chunk_units)
        key_to_entry = {e["mention_key"]: e for e in mention_entries}
        mention_chunks: List[List[Dict[str, Any]]] = []
        for ch in chunk_dicts:
            sub: List[Dict[str, Any]] = []
            for mk in ch.keys():
                ent = key_to_entry.get(mk)
                if ent is not None:
                    sub.append(ent)
            if sub:
                mention_chunks.append(sub)

        if not mention_chunks:
            return []

        record_mentioned_tweet_ids_by_channel(
            self.profile,
            self.profile_id,
            current_step,
            mention_entries,
            getattr(event, "timestamp", 0),
        )

        events_to_send: List[Event] = []
        has_reply = False
        user_id = await self.get_data("id")
        nickname = await self.get_data("nickname", "")
        username = await self.get_data("username", "")
        current_timestamp = event.timestamp
        current_ts = int(event.timestamp) if getattr(event, "timestamp", None) is not None else 0
        step_duration = int(getattr(event, "timestamp_duration", 0) or 0)

        total_batches = len(mention_chunks)

        for batch_idx, chunk_entries in enumerate(mention_chunks):
            batch_k = len(chunk_entries)
            batch_tweet_ids = {
                str(e["tweet_id"]).strip() for e in chunk_entries if e.get("tweet_id") is not None
            }

            observation_parts = []
            for j, entry in enumerate(chunk_entries):
                tw_for_llm = prepare_tweet_for_llm(entry["mention_tweet"])
                if isinstance(tw_for_llm, dict):
                    tw_for_llm = dict(tw_for_llm)
                    tw_for_llm["n_hop"] = entry.get("n_hop")
                observation_parts.append(f"""## Alert {j + 1}
            {entry["mention_action"]}

            Relationship with the commenter/poster (relationship_understanding must follow this; do not invent or invert):
            {entry["relationship_source"]}

            {entry["content_label"]}:
            {json.dumps(tw_for_llm, ensure_ascii=False, indent=2)}
            {entry.get("quote_reply_chain_hint", "")}
            """)

            depth_coaching = TweetDepthGate.coaching_for_mention_entries(
                chunk_entries, content_pool, seed_ids
            )

            observation = (
                f"[Batch] {batch_idx + 1}/{total_batches}; this batch has {batch_k} alerts. **decisions length must be {batch_k}** (matches Alert 1 … Alert {batch_k} below).\n\n"
                + "[Scene] You got a notification (quote/reply). You may ignore; reply only if worth it. Keep replies short and chat-like, not a briefing.\n\n"
            )
            observation += (
                f"Quotes/replies ({batch_k} alerts; decide in order):\n\n"
                + "\n".join(observation_parts)
                + "\nUsers you may @:\n"
                + mentionable_users_str
            )

            gates = await UserAgentGates(self).build_mention_coaching(
                chunk_entries, current_timestamp, content_pool=content_pool
            )

            instruction = f"""This batch has {batch_k} alerts (batch {batch_idx + 1} of {total_batches}). Decide each alert in order (reply or not, text, etc.). **decisions length must equal {batch_k}**; item i matches alert i.

            Use Profile, historical_summary, memory, and relationships in Observation to decide engagement and write text.

            Step 1 — Default each row
            - "propagation": false
            - "propagation_type": "" (empty: no propagation type chosen; same as no engagement)
            - "propagation_content": ""

            **Only steps 2/3 control propagation counts; do not override because of step 5.**
            {gates.memory_coaching}

            Step 2 — Interest gate (still no propagation_type/mode)d
            - Read all items in this batch; set propagation=true only if **all** of the following hold:
                {gates.memory_rec}
                1. Relevant to Profile/historical_summary/memory;
                2. Relationship and scene fit (prefer follows);
                3. You see heat/reply value and clear intent to respond or amplify;
                4. If the tweet is already in a quote/reply chain (replied_tweet or quoted_tweet non-empty), lean toward propagation=false, further propagation needs clear new stance, explanation, context, or extension;
            - Items that fail interest: not in the candidate pool, propagation=false.

            Step 3 — propagation_type per engagement
            -{gates.propagation_type_coaching}

            If empty-text forward, propagation_type must be "retweet"; skip step 4.

            Step 4 — propagation_mode (persona)
            - propagation_mode ∈ {{analysis/event narrative/advice, emotion/memes/roast, question/chat, emoji/placeholder}}; reflect in expression_reason.
                - analysis/event narrative/advice: facts, mechanisms, details; max ~40 words.
                - emotion/memes/roast: attitude-first; max ~25 words.
                - question/chat: move the thread; max ~30 words.
                - emoji/placeholder: emoji or very short; max ~1 word. e.g. 🤯😇👌😅🤣😨😰😱🙈🤩😂
            - Output style:
                - Prefer the user's locale; default English.
                - Match Profile and historical_summary voice;
                - Avoid canned phrases ("you're so right", "indeed", "hope everyone…");
                - **No hashtags (#...) in propagation_content.**

            Step 5 — @ mentions
            - @only if user is in the allow-list, strongly relevant, and not the tweet author / parent retweet author; else mention_reasoning [].

            Step 6 — keep_following_tweet_ids
            - Only if highly interested and worth tracking; else [].

            Step 7 — search
            - search=true only if highly interested, info insufficient, and memory unhelpful; else false.

            Return JSON (field order fixed):
            {{
            "persona_understanding": "1-2 sentences: role, interests, voice",
            "content_understanding": "1-2 sentences: relevance of this @/comment to you",
            "relationship_understanding": "Follow Observation exactly; do not invent relationship types",
            "memory_reflection": "{gates.memory_ref}",
            "decisions": [
                {{
                "tweet_id": "tweet_id",
                "propagation": false,
                "propagation_type": "",
                "propagation_content": "",
                "decision_reason": "why skip (1 short line, <20 chars)",
                "expression_reason": "",
                "mention_reasoning": []
                }},
                {{
                "tweet_id": "tweet_id",
                "propagation": true,
                "propagation_type": "retweet",
                "propagation_content": "",
                "decision_reason": "why engage (1 short line, <20 chars)",
                "expression_reason": "",
                "mention_reasoning": []
                }},
                {{
                "tweet_id": "tweet_id",
                "propagation": true,
                "propagation_type": "reply|quote",
                "propagation_content": "short reply or quote text",
                "decision_reason": "why engage (1 short line, <20 chars)",
                "expression_reason": "tone; substantive/micro if applicable (1 line)",
                "mention_reasoning": [
                    {{
                    "user_id": "mentioned user id",
                    "persona_understanding": "brief read of that user",
                    "mention_reason": "why ping them (1 short line, <20 chars)"
                    }}
                ]
                }}
            ],
            "keep_following_tweet_ids": [],
            "keep_following_reason": "why keep following (1 short line, <20 chars)",
            "search": false,
            "search_keyword": "keyword for search, <len<=20 chars>",
            "search_reason": "search or not and why (1 line)",
            }}
            Output rules:
            - Understanding fields first, then decisions;
            - Start each decision as propagation=false / propagation_type ""; flip to true only where rules pass (single type preferred; "reply|quote" allowed but not ideal);
            - If propagation=false, propagation_type must be ""; if propagation=true and propagation_type is exactly "retweet", propagation_content must be "".
            - Align propagation_content with propagation_mode and persona; avoid boilerplate.
            {gates.similarity_kw_coaching}{gates.similarity_emb_coaching}
            {depth_coaching}{gates.freshness_coaching}"""

            logger.info(
                f"Step {current_step}/{max_step}: UserAgent {self.profile_id} handle_mention batch "
                f"{batch_idx + 1}/{total_batches} ({batch_k} entries), max_chunk_units={max_chunk_units}"
            )

            # Call LLM to generate the decision
            response = await self.generate_reaction(instruction, observation)

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

                    # Get the user profile and send the event to the algorithm module
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
                keep_ids = response.get("keep_following_tweet_ids", [])
                if isinstance(keep_ids, list) and keep_ids:
                    valid_keep_ids = []
                    for keep_tweet_id in keep_ids:
                        sk = str(keep_tweet_id).strip() if keep_tweet_id is not None else ""
                        if sk and sk in batch_tweet_ids:
                            valid_keep_ids.append(keep_tweet_id)
                    if valid_keep_ids:
                        self.profile.update_data("keep_following_tweet_ids", valid_keep_ids[:1])
                    else:
                        self.profile.update_data("keep_following_tweet_ids", [])

            # Handle diffusion decisions
            decisions = response.get("decisions", [])
            if not isinstance(decisions, list):
                logger.warning(
                    f"Step {current_step}/{max_step}: UserAgent {self.profile_id} handle_mention batch "
                    f"{batch_idx + 1}/{total_batches}: invalid decisions (not a list), skip batch"
                )
                continue

            # Handle each decision: update propagation count and propagation content
            for i, decision in enumerate(decisions):
                if i >= len(chunk_entries):
                    break
                mention_entry = chunk_entries[i]
                mention_tweet = mention_entry["mention_tweet"]
                mention_tweet_id = mention_entry["tweet_id"]
                try:
                    if not isinstance(decision, dict) or not decision.get("propagation", False):
                        continue

                    tweet_id = decision.get("tweet_id")
                    should_propagation = decision.get("propagation", False)
                    propagation_type = decision.get("propagation_type", "")
                    propagation_content = decision.get("propagation_content", "")

                    if not tweet_id or not should_propagation:
                        continue

                    # If the retweet has no content, propagation_content should be ""
                    if propagation_type != "retweet" and not (propagation_content or "").strip():
                        continue

                    has_reply = True

                    # Check if the tweet_id is valid
                    if tweet_id != mention_tweet_id:
                        logger.warning(f"Tweet {tweet_id} does not match mention tweet {mention_tweet_id}, skipping")
                        continue

                    # Parse the @users in the propagation content, replace @id with @nickname, and return the list of user IDs
                    mentioned_user_ids = []
                    mention_reasoning = decision.get("mention_reasoning", [])
                    if isinstance(mention_reasoning, list):
                        for mention_reason in mention_reasoning:
                            if isinstance(mention_reason, dict):
                                muid = mention_reason.get("user_id")
                                if muid:
                                    mentioned_user_ids.append(muid)

                    path_author_user_ids = self._collect_chain_user_ids(
                        mention_tweet, None
                    )
                    path_author_set = {str(x).strip() for x in path_author_user_ids if x}

                    # Remove the authors in the quote/reply chain from the @list to avoid duplicate counting and duplicate reminders
                    filtered_mention_ids: Set[str] = set()
                    for x in mentioned_user_ids:
                        if x is not None:
                            sx = str(x).strip()
                            if sx and sx not in path_author_set:
                                filtered_mention_ids.add(sx)
                    filtered_mention_ids.discard(user_id)
                    mention_count = len(filtered_mention_ids)

                    propagation_id = generate_propagation_id()
                    ts = generate_tweet_timestamp(mention_tweet, current_ts, step_duration)

                    if propagation_type == "retweet":
                        success = await self.add_env_tweets(
                            propagation_id,
                            {
                                "tweet_id": propagation_id,
                                "content": propagation_content,
                                "time": ts,
                                "user_id": user_id,
                                "nickname": nickname,
                                "username": username,
                                "mention_count": mention_count,
                                "retweeted_tweet_id": tweet_id,
                            },
                        )
                        if not success:
                            logger.error(f"Failed to add retweet to tweet {tweet_id}")
                            continue
                    elif propagation_type == "quote":
                        success = await self.add_env_tweets(
                            propagation_id,
                            {
                                "tweet_id": propagation_id,
                                "content": propagation_content,
                                "time": ts,
                                "user_id": user_id,
                                "nickname": nickname,
                                "username": username,
                                "mention_count": mention_count,
                                "quoted_tweet_id": tweet_id,
                            },
                        )
                        if not success:
                            logger.error(f"Failed to add quote to tweet {tweet_id}")
                            continue
                    elif propagation_type == "reply":
                        success = await self.add_env_tweets(
                            propagation_id,
                            {
                                "tweet_id": propagation_id,
                                "content": propagation_content,
                                "time": ts,
                                "user_id": user_id,
                                "nickname": nickname,
                                "username": username,
                                "mention_count": mention_count,
                                "replied_tweet_id": tweet_id,
                            },
                        )
                        if not success:
                            logger.error(f"Failed to add reply to tweet {tweet_id}")
                            continue
                    else:
                        continue

                    # Send reminders to the authors in the quote/reply chain
                    if path_author_user_ids:
                        path_notify_ids = sorted(
                            {
                                s
                                for s in (str(x).strip() for x in path_author_user_ids if x)
                                if s and s != user_id
                            }
                        )
                        for target_uid in path_notify_ids:
                            success = await self.update_env_mention_pool(
                                f"{target_uid}.{propagation_id}",
                                {
                                    "action": "add",
                                    "mention_message": {
                                        "tweet_id": tweet_id,
                                        "mention_type": propagation_type,
                                    },
                                },
                            )
                            if not success:
                                logger.error(
                                    f"Failed to update mention pool for {propagation_type} {propagation_id} "
                                    f"by {user_id} -> chain author {target_uid} on tweet {tweet_id}"
                                )
                            else:
                                logger.info(
                                    f"Step {current_step}/{max_step}: User {user_id} {propagation_type} to tweet {tweet_id}"
                                )

                    # Send MentionEvent to the users that are @ed
                    if filtered_mention_ids:
                        for mentioned_user_id in filtered_mention_ids:
                            if mentioned_user_id and mentioned_user_id != user_id:
                                success = await self.update_env_mention_pool(
                                    f"{mentioned_user_id}.{propagation_id}",
                                    {
                                        "action": "add",
                                        "mention_message": {
                                            "tweet_id": tweet_id,
                                            "mention_type": "at",
                                        },
                                    },
                                )
                                if not success:
                                    logger.error(
                                        f"Failed to update mention pool for tweet {tweet_id} by {user_id}"
                                    )
                                    continue
                                logger.info(
                                    f"Step {current_step}/{max_step}: User {user_id} mentioned "
                                    f"{mentioned_user_id} in tweet {tweet_id}"
                                )
                finally:
                    success = await self.update_env_mention_pool(
                        f"{user_id}.{mention_tweet_id}",
                        {"action": "delete", "mention_message": None},
                    )
                    if not success:
                        logger.error(f"Failed to update mention pool for comment {mention_tweet_id}")
                    else:
                        logger.info(
                            f"Step {current_step}/{max_step}: User {user_id} deleted mention "
                            f"{mention_tweet_id} from pool (processed)"
                        )

        # Send MentionSpreadingEvent
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
    