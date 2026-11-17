from onesim.utils.comment_view import select_posts_for_agent
from .mention_address import resolve_address, address_key
from typing import Any, List, Optional, Dict, Set, Tuple, Union
import json
import asyncio
import os
import re
import secrets
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
from onesim.utils.midsim_params import MIDSimParams
from .events import *
import random
import math

from .gates import UserAgentGates
from .metrics.channel_snapshots import (
    record_mentioned_post_ids_by_channel,
    record_recommendations_by_source_step,
)
from .prompting.loader import format_prompt
from .prompting.builder import build_supplement, build_author_reply, with_official_policy
from .utils.time import generate_comment_timestamp, time_in_window, time_to_ms

_TIER_VALUES = ("low", "mid", "high")



class UserAgent(GeneralAgent):
    async def get_prompt_profile(self, observation: str) -> str:
        if self.profile is None:
            return await super().get_prompt_profile(observation)
        from .prompting.history import select_history
        data = dict(self.profile.get_profile(include_private=False))
        history = data.pop('historical_posts', {})
        selected, method = await asyncio.to_thread(select_history, history, observation)
        data['relevant_historical_posts'] = selected
        data['historical_evidence_usage'] = (
            'These are selected historical statements relevant to the current observation, '
            'not evidence that the current interaction has already been answered. '
            'Use historical_summary for overall style. Do not treat a quotation or shared post '
            'as a personal experience unless the text supports that interpretation.'
        )
        data['historical_retrieval_method'] = method
        return json.dumps(data, ensure_ascii=False, default=str)

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
        self.register_event("StartEvent", "generate_memory_from_own_posts")
        self.register_event("StartEvent", "supplement_own_posts")
        self.register_event("SocialRecommendationEvent", "receive_recommendation")
        self.register_event("AlgorithmRecommendationEvent", "receive_recommendation")
        self.register_event("SearchResultEvent", "receive_recommendation")
        self.register_event("BookmarkEvent", "receive_recommendation")
        self.register_event("MentionEvent", "handle_mention")
        self.register_event("AddCommentResponseEvent", "handle_add_comment_response")
        self.register_event("MentionPoolUpdateResponseEvent", "handle_update_mention_pool_response")
        self._comment_add_futures: Dict[str, Future] = {}
        self._mention_pool_update_futures: Dict[str, Future] = {}
        self._activation_step = None
        self._activated_this_step = False
        self._login_lock = asyncio.Lock()
        self._recommended_history_lock = asyncio.Lock()

    async def _is_official_by_agent_field(self) -> Optional[bool]:
        """Check if the account is official by agent field."""
        is_official = self.profile.get_data("is_official", False)

        return True if is_official else False

    def _profile_tier(self, field: str) -> str:
        raw = "low"
        if self.profile:
            raw = str(self.profile.get_data(field, "low") or "low").strip().lower()
        return raw if raw in _TIER_VALUES else "low"

    @staticmethod
    def _is_author_thread_mention(entry: Dict[str, Any], user_id: str) -> bool:
        post = entry.get("mention_post") or {}
        if not isinstance(post, dict):
            return False
        return str(post.get("user_id") or "").strip() == str(user_id or "").strip()

    async def _should_activate_this_round(self, current_step: int, max_step: int) -> bool:
        """Return the cached activation state for the current simulation step."""
        async with self._login_lock:
            if self._activation_step == current_step:
                return self._activated_this_step

            raw_activity = (
                self.profile.get_data("activity_level", 0.0)
                if self.profile
                else 0.0
            )
            try:
                activity_level = float(raw_activity)
            except (TypeError, ValueError):
                activity_level = 0.0

            normalized_activity = max(0.0, min(1.0, activity_level))
            activation_prob = 0.4 + 0.5 * normalized_activity
            activated = random.random() < activation_prob

            self._activation_step = current_step
            self._activated_this_step = activated
            return activated

    async def update_env_mention_pool(self, key: str, value: Any, parent_event_id: Optional[str] = None, *, target_user_id: Optional[str] = None, comment_id: Optional[str] = None) -> bool:
        """Update the mention_pool in the environment (using distributed lock)"""
        # Convert mentioner_id.comment_id to the full key format: mention_pool.mentioner_id.comment_id
        target_user_id, comment_id = resolve_address(
            f"mention_pool.{key}", value, target_user_id, comment_id)
        full_key = address_key(target_user_id, comment_id)
        
        # request_id is also the immutable operation ID across transport retries.
        request_id = f"mention_op_{secrets.token_hex(16)}"
        future = asyncio.get_running_loop().create_future()
        self._mention_pool_update_futures[request_id] = future
        # Freeze the payload so later caller mutations cannot change this operation.
        from copy import deepcopy
        payload = deepcopy(value)
        try:
            from onesim.events import get_event_bus
            event_bus = get_event_bus()
            attempt = 0
            while True:
                if future.done():
                    return future.result()
                attempt += 1
                event = MentionPoolUpdateEvent(
                    from_agent_id=self.profile_id, to_agent_id="ENV",
                    source_type="AGENT", target_type="ENV", key=full_key,
                    target_user_id=target_user_id, comment_id=comment_id,
                    value=deepcopy(payload), request_id=request_id,
                    parent_event_id=parent_event_id,
                )
                await event_bus.dispatch_event(event)
                try:
                    return await asyncio.wait_for(
                        asyncio.shield(future), timeout=_ENV_ASYNC_OP_TIMEOUT)
                except asyncio.TimeoutError:
                    logger.warning(
                        "Mention confirmation pending; retrying same operation: "
                        "request_id={} target={} action={} attempt={} timeout={}s",
                        request_id, target_user_id, payload.get("action"),
                        attempt, _ENV_ASYNC_OP_TIMEOUT,
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Mention update rejected: request_id={} error={!r}", request_id, exc)
            return False
        finally:
            self._mention_pool_update_futures.pop(request_id, None)
            if not future.done():
                future.cancel()

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
        current_time_ms = int(getattr(event, "timestamp", 0) or 0)
        step_duration_ms = int(getattr(event, "timestamp_duration", 0) or 0)

        # Check user activation status for this round
        if not await self._should_activate_this_round(current_step, max_step):
            logger.debug(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} is not activated in this round")
            return []
        logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} is activated in this round")

        events: List[Event] = []
        params = await MIDSimParams.load(self)

        # Get current available posts list
        current_posts = getattr(event, "current_posts", None) or {}
        if not isinstance(current_posts, dict):
            current_posts = {}

        # Algorithm recommendation stream
        raw_recommended_post_ids = self.profile.get_data("recommended_post_ids", []) if self.profile else []
        if not isinstance(raw_recommended_post_ids, list):
            raw_recommended_post_ids = []
        recommended_post_ids = {str(x).strip() for x in raw_recommended_post_ids if x is not None and str(x).strip()}

        profile_payload = {}
        if self.profile is not None:
            try:
                profile_payload = dict(self.profile.get_profile(include_private=True) or {})
            except Exception:
                logger.warning("Failed to serialize profile via get_profile(), fallback to empty payload.")

        algorithm_request_prob = params.user.algorithmic_recommendation_prob
        if random.random() <= algorithm_request_prob:
            events.append(
                GetAlgorithmRecommendationEvent(
                    from_agent_id=self.profile_id,
                    to_agent_id=PLATFORM_AGENT_ID,
                    timestamp=current_time_ms,
                    timestamp_duration=step_duration_ms,
                    current_step=current_step,
                    max_step=max_step,
                    user_profile=profile_payload,
                    current_posts=current_posts,
                    recommended_post_ids=recommended_post_ids,
                )
            )

        # Get mentions and send MentionEvent (if exceeds mention_budget, randomly retain and delete other items in the pool)
        mentions = getattr(event, "mentions", None) or {}
        if not isinstance(mentions, dict):
            mentions = {}
        mention_budget = params.user.attention_budget
        if len(mentions) > mention_budget:
            user_id = str(await self.get_data("id") or "").strip()
            all_keys = list(mentions.keys())
            sampled_keys = random.sample(all_keys, mention_budget)
            drop_n = len(all_keys) - len(sampled_keys)
            logger.info(
                f"Step {current_step}/{max_step}: UserAgent {self.profile_id} "
                f"mentions count {len(all_keys)} > {mention_budget}, sampling {mention_budget} for MentionEvent; "
                f"bulk-trimming {drop_n} others from mention_pool"
            )
            # One ENV round-trip: serial per-id deletes previously blocked round completion for hours.
            success = await self.update_env_mention_pool(
                user_id,
                {
                    "action": "keep_only",
                    "keep_comment_ids": list(sampled_keys),
                },
                target_user_id=user_id,
            )
            if not success:
                logger.error(
                    f"Failed to bulk-trim mention_pool for user {user_id} "
                    f"(keep={len(sampled_keys)}, drop={drop_n}); continuing with sampled mentions this round"
                )
            mentions = {k: mentions[k] for k in sampled_keys}
        if 0 < len(mentions) <= mention_budget:
            enriched_mentions: Dict[str, Any] = {}
            for mention_key, mention_message in mentions.items():
                if not isinstance(mention_message, dict):
                    enriched_mentions[mention_key] = mention_message
                    continue
                mm = dict(mention_message)
                post = mm.get("post")
                if isinstance(post, dict):
                    nid = post.get("post_id") or post.get("id")
                    if nid is not None:
                        post_id_key = str(nid).strip()
                        current_post = current_posts.get(post_id_key)
                        if isinstance(current_post, dict):
                            mm["post"] = dict(current_post)
                enriched_mentions[mention_key] = mm
            mentions = enriched_mentions

            logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} send MentionEvent, length of mentions: {len(mentions)}")
            events.append(MentionEvent(
                from_agent_id=self.profile_id,
                to_agent_id=self.profile_id,
                timestamp=current_time_ms,
                timestamp_duration=step_duration_ms,
                current_step=current_step,
                max_step=max_step,
                mentions=mentions,
            ))

        # Handle bookmark_post_ids list: if previous round decided to keep, re-add it to the recommendation in this round
        bookmark_posts = {}
        if self.profile:
            keep_ids = self.profile.get_data("bookmark_post_ids", []) or []
            if isinstance(keep_ids, (list, tuple)) and keep_ids:
                readded = 0
                for keep_post_id in keep_ids:
                    keep_post_id = str(keep_post_id).strip() if keep_post_id is not None else ""
                    if not keep_post_id or keep_post_id in bookmark_posts:
                        continue
                    post = current_posts.get(keep_post_id)
                    if not isinstance(post, dict):
                        continue
                    bookmark_posts[keep_post_id] = post
                    readded += 1
                if readded > 0:
                    logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} re-added bookmark posts: {readded}")
                self.profile.update_data("bookmark_post_ids", [])
                
        if bookmark_posts:
            events.append(BookmarkEvent(
                from_agent_id=self.profile_id,
                to_agent_id=self.profile_id,
                timestamp=current_time_ms,
                timestamp_duration=step_duration_ms,
                current_step=current_step,
                max_step=max_step,
                recommendations=bookmark_posts,
            ))

        # Social recommendation stream
        follow_ids = await self.get_data("follow_ids", [])
        follow_set = {
            str(x).strip() for x in follow_ids if x is not None and str(x).strip()
        } if isinstance(follow_ids, (list, tuple)) else set()

        # Get posts in the follow list that have not been viewed
        last_login = 0
        login_cap_days = params.user.social_feed_last_login_cap_days
        if self.profile:
            raw = self.profile.get_data("last_login_timestamp")
            logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} last_login_timestamp: {raw}")
            if raw is not None and isinstance(raw, (int, float)) and int(raw) > 0:
                last_login = int(raw)
                if current_time_ms > 0 and login_cap_days > 0:
                    lower = current_time_ms - int(login_cap_days * 86400000)
                    if last_login < lower:
                        last_login = lower
            else:
                last_login = 0

        window_end = current_time_ms + step_duration_ms

        recommendations = {}
        for post_id, post in current_posts.items():
            if not isinstance(post, dict):
                continue
            author_id = str(post.get("user_id") or post.get("author_id") or "").strip()
            if not author_id or author_id not in follow_set:
                continue
            t = post.get("time") or post.get("create_time")
            if t is None:
                continue
            try:
                t = int(t)
            except (TypeError, ValueError):
                continue
            if last_login <= t < window_end:
                post_id = str(post_id).strip() if post_id is not None else ""
                if not post_id or post_id in recommended_post_ids:
                    continue
                recommendations[post_id] = post

        # Update the last login timestamp of the agent
        if self.profile:
            self.profile.update_data("last_login_timestamp", current_time_ms)
            logger.info(
                f"Step {current_step}/{max_step}: UserAgent {self.profile_id} update last_login_timestamp: {current_time_ms}"
            )

        # Send SocialRecommendationEvent
        social_recommendation_prob = params.user.social_recommendation_prob
        if recommendations:
            if random.random() > social_recommendation_prob:
                logger.debug(
                    f"Step {current_step}/{max_step}: UserAgent {self.profile_id} "
                    f"skip SocialRecommendationEvent (n={len(recommendations)}, "
                    f"p={social_recommendation_prob:.0%} not drawn)"
                )
            else:
                logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} send SocialRecommendationEvent, length of recommendations: {len(recommendations)}")

                events.append(SocialRecommendationEvent(
                    from_agent_id=self.profile_id,
                    to_agent_id=self.profile_id,
                    timestamp=current_time_ms,
                    timestamp_duration=step_duration_ms,
                    current_step=current_step,
                    max_step=max_step,
                    recommendations=recommendations,
                ))
        return events

    async def add_env_comment(self, post_id: str, comment: Any, parent_event_id: Optional[str] = None) -> bool:
        """Add comment to the data in the environment (using distributed lock)"""
        full_key = f"content_pool.{post_id}.comments"
        
        # Create request and future
        request_id = f"agent_env_add_comment_req_{time.time()}_{id(self)}"
        future = asyncio.Future()
        self._comment_add_futures[request_id] = future

        try:
            from onesim.events import get_event_bus
            event_bus = get_event_bus()
            attempt = 0
            while True:
                if future.done():
                    return future.result()
                attempt += 1
                # A fresh transport event, but the SAME request and comment IDs.
                # SimEnv upserts by comment_id, making delayed retries idempotent.
                comment_add_event = AddCommentEvent(
                    from_agent_id=self.profile_id, to_agent_id="ENV",
                    source_type="AGENT", target_type="ENV", key=full_key,
                    value=comment, request_id=request_id,
                    parent_event_id=parent_event_id,
                )
                await event_bus.dispatch_event(comment_add_event)
                try:
                    # shield keeps a late acknowledgement valid after a timeout.
                    return await asyncio.wait_for(
                        asyncio.shield(future), timeout=_ENV_ASYNC_OP_TIMEOUT)
                except asyncio.TimeoutError:
                    logger.warning(
                        "Comment confirmation pending; retrying identical write: "
                        "request_id={} comment_id={} attempt={} timeout={}s",
                        request_id, comment.get("comment_id"), attempt,
                        _ENV_ASYNC_OP_TIMEOUT,
                    )
                    # Do not report failure or skip notifications on unknown status.
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Comment write rejected: request_id={} error={!r}", request_id, exc)
            return False
        finally:
            self._comment_add_futures.pop(request_id, None)
            if not future.done():
                future.cancel()

    async def handle_add_comment_response(self, event: AddCommentResponseEvent) -> None:
        """Handle the incoming comment add response event"""
        # Check if waiting for this response
        if event.request_id in self._comment_add_futures:
            future = self._comment_add_futures.pop(event.request_id)

            if not future.done():
                if event.success:
                    future.set_result(True)
                else:
                    future.set_exception(ValueError(event.error or "Unknown error"))

            # If there is a sync event, set it
            if hasattr(self, '_sync_event'):
                self._sync_event.set()
                self._sync_event.clear()

    async def generate_memory_from_own_posts(self, event: Event) -> List[Event]:
        """Generate self-memory from own posts published in the current step window."""
        current_step = getattr(event, "current_step", 1)
        max_step = getattr(event, "max_step", 8)
        user_id = str(await self.get_data("id") or "").strip()
        if not user_id:
            return []

        current_time_ms = int(getattr(event, "timestamp", 0) or 0)
        step_duration_ms = int(getattr(event, "timestamp_duration", 0) or 0)

        current_posts = getattr(event, "current_posts", None) or {}
        if not isinstance(current_posts, dict) or not current_posts:
            return []

        window_start_ms = current_time_ms
        window_end_ms = current_time_ms + step_duration_ms
        simulation_cap_ms = getattr(event, "simulation_cap_timestamp", None)
        if simulation_cap_ms is not None:
            window_end_ms = min(window_end_ms, int(simulation_cap_ms))
        if window_end_ms <= window_start_ms:
            return []

        own_posts_for_prompt = []
        for post_id, post in current_posts.items():
            if not isinstance(post, dict):
                continue
            post_id = str(post_id).strip() if post_id is not None else ""
            if not post_id:
                continue
            author_id = str(post.get("user_id") or post.get("author_id") or "").strip()
            if author_id != user_id:
                continue
            if not time_in_window(post, window_start_ms, window_end_ms):
                continue
            own_posts_for_prompt.append({
                "post_id": post_id,
                "title": post.get("title", ""),
                "desc": post.get("desc", ""),
                "tags_list": post.get("tags_list", []),
                "time": post.get("time", post.get("create_time", 0)),
            })

        if not own_posts_for_prompt:
            return []

        instruction = (
            "You are reviewing posts you recently published. From those posts, write one first-person memory. "
            "Focus on the topics you have been following, how you sound, and phrasing you can reuse in later interactions."
        )
        observation = f"Your recent posts:\n{json.dumps(own_posts_for_prompt, ensure_ascii=False, indent=2)}"
        reaction = {
            "task": "self_reflection_on_own_posts",
            "own_post_count": len(own_posts_for_prompt),
        }

        try:
            memory_text = await self.generate_memory(instruction, observation, reaction)
            if memory_text:
                logger.debug(
                    f"Step {current_step}/{max_step}: User {user_id} generated self-memory "
                    f"from {len(own_posts_for_prompt)} own posts"
                )
                logger.debug(f"Generated memory: {memory_text}")
        except Exception as e:
            logger.error(f"Step {current_step}/{max_step}: User {user_id} failed to generate memory from own posts: {e}")
        return []

    async def supplement_own_posts(self, event: Event) -> List[Event]:
        """Decide whether to add a supplementary comment on own posts from the last 14 days.

        A post is sent to the LLM only if it has never been reviewed, or its comment
        count changed since the last review. Own supplementary comments update the
        stored count so they do not trigger another review next step.
        """
        current_step = getattr(event, "current_step", 1)
        max_step = getattr(event, "max_step", 8)
        user_id = str(await self.get_data("id") or "").strip()
        if not user_id:
            return []

        current_time_ms = int(getattr(event, "timestamp", 0) or 0)
        step_duration_ms = int(getattr(event, "timestamp_duration", 0) or 0)

        current_posts = getattr(event, "current_posts", None) or {}
        if not isinstance(current_posts, dict) or not current_posts:
            return []

        window_start_ms = current_time_ms
        window_end_ms = current_time_ms + step_duration_ms
        simulation_cap_ms = getattr(event, "simulation_cap_timestamp", None)
        if simulation_cap_ms is not None:
            window_end_ms = min(window_end_ms, int(simulation_cap_ms))
        if window_end_ms <= window_start_ms:
            return []

        supplementary_window_ms = 14 * 24 * 3600 * 1000
        supplementary_start_ms = window_end_ms - supplementary_window_ms

        review_state: Dict[str, Any] = {}
        # if self.profile:
        #     raw_state = self.profile.get_data("supplement_review_state", {}) or {}
        #     if isinstance(raw_state, dict):
        #         review_state = {
        #             str(k).strip(): v
        #             for k, v in raw_state.items()
        #             if k is not None and str(k).strip()
        #         }

        own_posts_for_supplementary = []
        eligible_posts_by_id: Dict[str, Dict[str, Any]] = {}
        reviewed_comment_counts: Dict[str, int] = {}
        for post_id, post in current_posts.items():
            if not isinstance(post, dict):
                continue
            post_id = str(post_id).strip() if post_id is not None else ""
            if not post_id:
                continue
            author_id = str(post.get("user_id") or post.get("author_id") or "").strip()
            if author_id != user_id:
                continue
            if not time_in_window(post, supplementary_start_ms, window_end_ms):
                continue
            post = (await select_posts_for_agent(self, {post_id: post}, as_of=window_end_ms))[post_id]
            comments = post.get("comments") or {}
            if not isinstance(comments, dict):
                comments = {}
            comment_count = post.get("comment_count", len(comments))
            # prev = review_state.get(post_id)
            # if isinstance(prev, dict):
            #     try:
            #         prev_count = int(prev.get("last_comment_count"))
            #     except (TypeError, ValueError):
            #         prev_count = None
            #     if prev_count is not None and prev_count == comment_count:
            #         continue
            eligible_posts_by_id[post_id] = post
            comments_brief = []
            for cid, c in comments.items():
                if isinstance(c, dict):
                    prev_text = (c.get("content") or "")[:160]
                    comments_brief.append({
                        "comment_id": str(cid).strip() if cid is not None else "",
                        "user_id": str(c.get("user_id") or "").strip(),
                        "content_preview": prev_text,
                        "parent_comment_id": c.get("parent_comment_id"),
                    })
            reviewed_comment_counts[post_id] = comment_count
            own_posts_for_supplementary.append({
                "post_id": post_id,
                "content": post.get("content") or post.get("desc") or post.get("title") or "",
                "time": post.get("time", post.get("create_time", 0)),
                "existing_comments": comments_brief,
            })

        if not eligible_posts_by_id:
            return []

        supplement_tier = self._profile_tier("own_post_comment_tier")
        supplementary_instruction = build_supplement(
            supplement_tier, reply_tier=self._profile_tier("comment_reply_tier")
        )
        if (await self._is_official_by_agent_field()) is True:
            supplementary_instruction = with_official_policy(supplementary_instruction, "supplement")
            logger.info(
                f"Step {current_step}/{max_step}: User {user_id} own-post supplement "
                f"uses official priority constraint (tier={supplement_tier})"
            )

        observation_sup = (
            "Your own recent posts (with a short summary of existing comments, so you can decide whether to reply; "
            "treat the end of this simulation window as 'now'):\n"
            f"{json.dumps(own_posts_for_supplementary, ensure_ascii=False, indent=2)}"
        )

        try:
            sup_response = await self.generate_reaction(supplementary_instruction, observation_sup)
            sup_response = self._normalize_llm_reaction(sup_response)
        except Exception as e:
            logger.warning(
                f"Step {current_step}/{max_step}: User {user_id} own-post supplementary LLM failed: {e}"
            )
            return []

        decisions = sup_response.get("decisions", [])
        if not isinstance(decisions, list):
            logger.warning(
                f"Step {current_step}/{max_step}: User {user_id} own-post supplementary "
                f"decisions is not a list: {type(decisions).__name__}"
            )
            return []

        comment_decisions = []
        for decision in decisions:
            if not isinstance(decision, dict):
                continue
            post_id = str(decision.get("post_id") or "").strip()
            if not post_id or post_id not in eligible_posts_by_id:
                continue
            if decision.get("comment") is not True:
                continue
            if not str(decision.get("comment_content") or "").strip():
                continue
            comment_decisions.append(decision)

        added_comment_counts: Dict[str, int] = {}
        if not comment_decisions:
            # if self.profile:
            #     for post_id, comment_count in reviewed_comment_counts.items():
            #         review_state[post_id] = {
            #             "last_review_ms": window_end_ms,
            #             "last_comment_count": comment_count,
            #         }
            #     self.profile.update_data("supplement_review_state", review_state)
            return []

        nickname = await self.get_data("nickname", "")

        for decision in comment_decisions:
            post_id = str(decision.get("post_id") or "").strip()
            comment_content = str(decision.get("comment_content") or "").strip()
            parent_comment_id = decision.get("parent_comment_id")
            if parent_comment_id in ("", False):
                parent_comment_id = None
            post = eligible_posts_by_id[post_id]
            canonical_parent_id = None
            if parent_comment_id is not None:
                comments = post.get("comments") or {}
                if not isinstance(comments, dict):
                    comments = {}
                canonical_parent_id, _ = self._resolve_parent_comment_entry(
                    comments, parent_comment_id
                )
                if canonical_parent_id is None:
                    logger.warning(
                        f"Step {current_step}/{max_step}: Own-post supplementary skip: parent_comment_id {parent_comment_id} not in post {post_id}"
                    )
                    continue

            comment_id = self._generate_comment_id()
            success = await self.add_env_comment(
                post_id,
                {
                    "comment_id": comment_id,
                    "timestamp": generate_comment_timestamp(post, current_time_ms, step_duration_ms),
                    "post_id": post_id,
                    "user_id": user_id,
                    "nickname": nickname,
                    "parent_comment_id": canonical_parent_id,
                    "at_count": 0,
                    "content": comment_content,
                },
            )
            if not success:
                logger.error(f"Step {current_step}/{max_step}: Failed to add self supplementary comment on own post {post_id}")
                continue
            added_comment_counts[post_id] = added_comment_counts.get(post_id, 0) + 1
            logger.info(
                f"Step {current_step}/{max_step}: User {user_id} added self supplementary comment on own post {post_id} (no notifications)"
            )

        # if self.profile:
        #     for post_id, comment_count in reviewed_comment_counts.items():
        #         review_state[post_id] = {
        #             "last_review_ms": window_end_ms,
        #             "last_comment_count": comment_count + added_comment_counts.get(post_id, 0),
        #         }
        #     self.profile.update_data("supplement_review_state", review_state)

        return []

    def _filter_recommendations(self, recommendations: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Remove duplicate recommendations"""
        if not recommendations:
            return {}
        
        raw_recommended_ids = (
            self.profile.get_data("recommended_post_ids", [])
            if self.profile
            else []
        )
        if not isinstance(raw_recommended_ids, list):
            raw_recommended_ids = []
        recommended_post_ids = {
            str(x).strip()
            for x in raw_recommended_ids
            if x is not None and str(x).strip()
        }
        
        # Remove duplicate recommendations 
        filtered: Dict[str, Dict[str, Any]] = {}

        for post_id, rec in recommendations.items():
            if not isinstance(rec, dict):
                logger.warning(f"Recommendation {post_id} is not a dictionary")
                continue

            post_id = str(post_id).strip() if post_id is not None else ""
            if not post_id:
                continue
            if post_id in recommended_post_ids or post_id in filtered:
                continue
            filtered[post_id] = rec

        logger.debug(f"Filtered recommendations: {len(recommendations)} -> {len(filtered)}")
        return filtered

    def _record_recommended_history(self, recommendations: Dict[str, Any]) -> None:
        """Append newly seen post_ids to profile.recommended_post_ids, preserving order.

        Callers must hold ``_recommended_history_lock`` around this read-modify-write.
        """
        if not recommendations or not self.profile:
            return
        raw_recommended_ids = self.profile.get_data("recommended_post_ids", [])
        if not isinstance(raw_recommended_ids, list):
            raw_recommended_ids = []
        existing_ids = []
        existing_id_set: Set[str] = set()
        for x in raw_recommended_ids:
            if x is None:
                continue
            post_id = str(x).strip()
            if not post_id or post_id in existing_id_set:
                continue
            existing_id_set.add(post_id)
            existing_ids.append(post_id)
        added = False
        for post_id in recommendations.keys():
            post_id = str(post_id).strip() if post_id is not None else ""
            if not post_id or post_id in existing_id_set:
                continue
            existing_id_set.add(post_id)
            existing_ids.append(post_id)
            added = True
        if added:
            self.profile.update_data("recommended_post_ids", existing_ids)

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

        if not self.relationship_manager:
            logger.warning("RelationshipManager is not initialized")
            return mentionable_info

        follows_info = []
        fans_info = []
        mutual_info = []
        for user_id in follow_ids | fan_ids:
            rel = self.relationship_manager.get_relationship(user_id)
            if not (rel and rel.target_info and isinstance(rel.target_info, dict)):
                continue
            info = dict(rel.target_info)
            info.pop("historical_summary", None)
            info.pop("historical_posts", None)
            is_follow = user_id in follow_ids
            is_fan = user_id in fan_ids
            if is_follow:
                follows_info.append(info)
            if is_fan:
                fans_info.append(info)
            if is_follow and is_fan:
                mutual_info.append(info)

        mentionable_info["follows"] = follows_info
        mentionable_info["fans"] = fans_info
        mentionable_info["mutual"] = mutual_info
        
        return mentionable_info

    @staticmethod
    def _mentionable_id_set(mentionable_users: Dict[str, Any]) -> Set[str]:
        """IDs actually present in `_get_mentionable_users()` output."""
        ids: Set[str] = set()
        for group in mentionable_users.values():
            if not isinstance(group, list):
                continue
            for info in group:
                if not isinstance(info, dict):
                    continue
                uid = str(info.get("id") or info.get("user_id") or "").strip()
                if uid:
                    ids.add(uid)
        return ids

    def _build_search_event(
        self,
        response: Dict[str, Any],
        event: Event,
        current_step: int,
        max_step: int,
    ) -> Optional[Event]:
        if response.get("search") is not True:
            return None

        search_keyword = str(response.get("search_keyword") or "").strip()
        if not search_keyword:
            logger.debug(
                f"Step {current_step}/{max_step}: "
                f"UserAgent {self.profile_id} search=true but search_keyword is empty"
            )
            return None

        profile_payload = {}
        if self.profile:
            try:
                profile_payload = dict(self.profile.get_profile(include_private=True) or {})
            except Exception:
                logger.warning("Failed to serialize profile via get_profile(), fallback to empty payload.")

        return GetSearchResultEvent(
            from_agent_id=self.profile_id,
            to_agent_id=PLATFORM_AGENT_ID,
            timestamp=int(getattr(event, "timestamp", 0) or 0),
            timestamp_duration=int(getattr(event, "timestamp_duration", 0) or 0),
            current_step=current_step,
            max_step=max_step,
            user_profile=profile_payload,
            search_query=search_keyword,
        )

    @staticmethod
    def _decision_flag_on(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "y")
        return False

    def _build_bookmark_post_ids(
        self,
        response: Dict[str, Any],
        allowed_post_ids: Union[Set[str], Dict[str, Any]],
    ) -> Optional[List[str]]:
        decisions = response.get("decisions", [])
        if not isinstance(decisions, list):
            return None
        valid_keep_ids = []
        seen = set()
        for decision in decisions:
            if not isinstance(decision, dict) or not self._decision_flag_on(decision.get("bookmark")):
                continue
            keep_post_id = str(decision.get("post_id") or "").strip()
            if not keep_post_id or keep_post_id not in allowed_post_ids or keep_post_id in seen:
                continue
            seen.add(keep_post_id)
            valid_keep_ids.append(keep_post_id)
        return valid_keep_ids or None

    @staticmethod
    def _generate_comment_id() -> str:
        """24-char hex comment id, e.g. 693ba19b000000001702f98e."""
        return secrets.token_hex(12)

    @staticmethod
    def _resolve_parent_comment_entry(
        comments_map: Any,
        parent_raw: Any,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Map LLM parent_comment_id onto post['comments'] keys."""
        if not isinstance(comments_map, dict) or parent_raw is None:
            return None, None
        parent_id = str(parent_raw).strip()
        if not parent_id:
            return None, None
        entry = comments_map.get(parent_id)
        return (parent_id, entry) if isinstance(entry, dict) else (None, None)

    async def _add_comment_and_notify(
        self,
        decision: Dict[str, Any],
        post: Dict[str, Any],
        mentionable_id_set: Set[str],
        user_id: str,
        nickname: str,
        window_start_ms: int,
        window_duration_ms: int,
        current_step: int,
        max_step: int,
    ) -> bool:
        post_id = str(decision.get("post_id") or "").strip()
        comment_content = str(decision.get("comment_content") or "").strip()
        parent_comment_id = decision.get("parent_comment_id")

        mentioned_user_ids = []
        seen_mention_ids = set()
        mention_reasoning = decision.get("mention_reasoning", [])
        if isinstance(mention_reasoning, list):
            for mention_reason in mention_reasoning:
                if not isinstance(mention_reason, dict):
                    continue
                mentioned_uid = str(mention_reason.get("user_id") or "").strip()
                if (
                    mentioned_uid
                    and mentioned_uid in mentionable_id_set
                    and mentioned_uid not in seen_mention_ids
                ):
                    seen_mention_ids.add(mentioned_uid)
                    mentioned_user_ids.append(mentioned_uid)

        post_author_id = str(post.get("user_id") or "").strip()
        comments_map = post.get("comments")
        if not isinstance(comments_map, dict):
            comments_map = {}

        comment_author_id = None
        canonical_parent_id: Optional[str] = None
        if parent_comment_id is not None:
            canonical_parent_id, parent_entry = self._resolve_parent_comment_entry(
                comments_map, parent_comment_id
            )
            if canonical_parent_id is None:
                env = getattr(self, "env", None)
                if env is not None and hasattr(env, "get_committed_comment"):
                    parent_entry = await env.get_committed_comment(post_id, str(parent_comment_id))
                    if parent_entry is not None:
                        canonical_parent_id = str(parent_comment_id)
            if canonical_parent_id is None:
                logger.warning(
                    f"Parent comment {parent_comment_id} not found in post {post_id}, skipping reply"
                )
                return False
            if isinstance(parent_entry, dict):
                comment_author_id = str(parent_entry.get("user_id") or "").strip() or None

        mentioned_user_ids = [
            uid
            for uid in mentioned_user_ids
            if uid != user_id
            and uid != post_author_id
            and uid != comment_author_id
        ]

        comment_id = self._generate_comment_id()
        success = await self.add_env_comment(post_id, {
            "comment_id": comment_id,
            "timestamp": generate_comment_timestamp(
                post, window_start_ms, window_duration_ms
            ),
            "post_id": post_id,
            "user_id": user_id,
            "nickname": nickname,
            "parent_comment_id": canonical_parent_id,
            "at_count": len(mentioned_user_ids),
            "content": comment_content,
        })
        if not success:
            logger.error(f"Failed to add comment to post {post_id}")
            return False

        notify_targets: List[Tuple[str, str]] = []
        if comment_author_id and comment_author_id != user_id:
            notify_targets.append((comment_author_id, "reply"))
        if post_author_id and post_author_id != user_id and post_author_id != comment_author_id:
            notify_targets.append((post_author_id, "comment"))
        for mentioned_user_id in mentioned_user_ids:
            notify_targets.append((mentioned_user_id, "at"))

        for target_id, mention_type in notify_targets:
            success = await self.update_env_mention_pool(target_id, {
                "action": "add",
                "mention_message": {
                    "post_id": post_id,
                    "comment_id": comment_id,
                    "comment_content": comment_content,
                    "mentioner_id": user_id,
                    "mentioner_nickname": nickname,
                    "mention_type": mention_type,
                },
            }, target_user_id=target_id, comment_id=str(comment_id))
            if not success:
                logger.error(
                    f"Failed to update mention pool for comment {comment_id} by {user_id} on post {post_id}"
                )
            else:
                logger.debug(
                    f"Step {current_step}/{max_step}: User {user_id} "
                    f"{mention_type} notify {target_id} on post {post_id}"
                )
        return True

    async def receive_recommendation(self, event: Event) -> List[Event]:
        """Receive recommendations and decide whether to comment."""
        current_step = getattr(event, "current_step", 1)
        max_step = getattr(event, "max_step", 8)
        if not await self._should_activate_this_round(current_step, max_step):
            logger.debug(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} receive recommendation but is not activated this round")
            return []

        # Store recommendations by type
        if event.__class__.__name__ == "AlgorithmRecommendationEvent":
            source_type = "algorithm"
        elif event.__class__.__name__ == "SocialRecommendationEvent":
            source_type = "social"
        elif event.__class__.__name__ == "SearchResultEvent":
            source_type = "search"
        elif event.__class__.__name__ == "BookmarkEvent":
            source_type = "bookmark"
        else:
            evt_cls = event.__class__.__name__
            logger.warning(f"Step {current_step}/{max_step}: UserAgent {self.profile_id}, unknown recommendation event {evt_cls}, skip")
            return []

        logger.debug(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} receive recommendation event, source type: {source_type}, length of recommendations: {len(event.recommendations)}")

        # algorithm/social: remove duplicate recommendations from profile.recommended_post_ids
        recommendations = event.recommendations
        if not isinstance(recommendations, dict):
            recommendations = {}
        recommendations = {
            str(post_id).strip(): post
            for post_id, post in recommendations.items()
            if post_id is not None and str(post_id).strip()
        }
        record_recommendations_by_source_step(
            self.profile,
            self.profile_id,
            source_type,
            current_step,
            recommendations,
            getattr(event, "timestamp", 0),
        )
        async with self._recommended_history_lock:
            if source_type == "algorithm" or source_type == "social":
                recommendations = self._filter_recommendations(recommendations)
                logger.debug(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} after filter, length of recommendations: {len(recommendations)}")
            if recommendations:
                self._record_recommended_history(recommendations)
        current_timestamp = event.timestamp
        window_duration_ms = int(getattr(event, "timestamp_duration", 0) or 0)
        if not recommendations:
            return [
                RecommendationSpreadingEvent(
                    from_agent_id=self.profile_id,
                    to_agent_id="ENV",
                    timestamp=current_timestamp,
                    timestamp_duration=window_duration_ms,
                    current_step=current_step,
                    max_step=max_step,
                )
            ]

        # Get user information and mentionable users
        user_id = str(await self.get_data("id") or "").strip()
        user_nickname = await self.get_data("nickname", "")
        window_start_ms = int(current_timestamp) if isinstance(current_timestamp, (int, float)) else 0
        follow_ids = [str(x).strip() for x in (await self.get_data("follow_ids", []) or []) if x is not None and str(x).strip()]
        fan_ids = [str(x).strip() for x in (await self.get_data("fan_ids", []) or []) if x is not None and str(x).strip()]
        mentionable_users = self._get_mentionable_users(follow_ids, fan_ids)
        mentionable_id_set = self._mentionable_id_set(mentionable_users)
        
        # Build the observation and instruction, and process the recommendations in chunks
        rec_items = list(recommendations.items())
        chunk_size = 5
        if len(rec_items) <= chunk_size:
            chunks = [recommendations]
        else:
            chunks = [dict(rec_items[i:i + chunk_size]) for i in range(0, len(rec_items), chunk_size)]

        events_to_send = []
        has_search = False
        is_official = (await self._is_official_by_agent_field()) is True
        if is_official:
            logger.info(
                f"Step {current_step}/{max_step}: User {user_id} recommendation "
                f"uses official priority constraint"
            )

        for chunk in chunks:
            # Choose one simulated browse time; replies cannot predate this snapshot.
            window_end_ms = window_start_ms + max(window_duration_ms, 1) - 1
            latest_post_ms = max(
                [window_start_ms] + [int(time_to_ms(p.get("timestamp", p.get("time")), default=window_start_ms) or window_start_ms)
                                    for p in chunk.values()]
            )
            browse_ms = random.randint(min(latest_post_ms, window_end_ms), window_end_ms)

            async def prepare_recommendation_input():
                nonlocal chunk
                env = getattr(self, "env", None)
                if env is not None and hasattr(env, "snapshot_visible_comments"):
                    chunk = await env.snapshot_visible_comments(chunk, browse_ms)
                else:
                    logger.warning("Live comment snapshot unavailable; using delivered recommendation snapshot")
                recommendations_str = json.dumps(chunk, ensure_ascii=False, indent=2)
                mentionable_users_str = json.dumps(mentionable_users.get("follows", []), ensure_ascii=False, indent=2)

                # Label the feed; if the chunk includes the user's own post, mark it as own content.
                has_self_post = any(
                    isinstance(post, dict) and str(post.get("user_id") or "").strip() == user_id
                    for post in chunk.values()
                )
                if has_self_post:
                    source_name = "own post (your content)"
                elif source_type == "social":
                    source_name = "following feed (people you follow)"
                elif source_type == "search":
                    source_name = "search feed (from your search)"
                elif source_type == "bookmark":
                    source_name = "bookmark feed (posts you chose to see again last round)"
                else:
                    source_name = "recommendation feed (from the algorithm)"

                observation = f"""[Scene] You are scrolling on your phone. Most posts you just skip. Now and then you stop and type a short comment. This is not a lab task and not an opinion-analysis writeup.

                Simulated browse timestamp (milliseconds): {browse_ms}
                Feed source: {source_name}

                Posts you were shown:
                {recommendations_str}

                People you can @ :
                {mentionable_users_str}"""

                gates = await UserAgentGates(self).build_recommendation_coaching(chunk, current_timestamp)


                instruction = format_prompt(
                    "tasks/recommendation_reaction.txt",
                    memory_coaching=gates.memory_coaching,
                    memory_ref=gates.memory_ref,
                    similarity_kw_coaching=gates.similarity_kw_coaching,
                    similarity_emb_coaching=gates.similarity_emb_coaching,
                )
                if is_official:
                    instruction = with_official_policy(instruction, "recommendation")

                return instruction, observation

            # Call LLM to generate the decision
            logger.debug(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} generate reaction")
            env = getattr(self, "env", None)
            if env is not None and hasattr(env, "notify_agent_busy"):
                await env.notify_agent_busy()
            try:
                response = await self.generate_reaction("", prepare_input=prepare_recommendation_input)
            finally:
                if env is not None and hasattr(env, "notify_agent_idle"):
                    await env.notify_agent_idle()
            response = self._normalize_llm_reaction(response)
            logger.debug(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} generate reaction response")

            if not has_search:
                search_event = self._build_search_event(
                    response, event, current_step, max_step
                )
                if search_event is not None:
                    events_to_send.append(search_event)
                    has_search = True

            if self.profile:
                bookmark_ids = self._build_bookmark_post_ids(response, chunk)
                if bookmark_ids:
                    existing = [
                        str(x).strip()
                        for x in (self.profile.get_data("bookmark_post_ids") or [])
                        if x is not None and str(x).strip()
                    ]
                    self.profile.update_data(
                        "bookmark_post_ids",
                        list(dict.fromkeys(existing + bookmark_ids)),
                    )

            # Handle comment decisions
            decisions = response.get("decisions", [])
            if not isinstance(decisions, list):
                continue
       
            # Handle each decision: update comment count and comment content
            for decision in decisions:
                if not isinstance(decision, dict):
                    continue

                post_id = str(decision.get("post_id") or "").strip()
                comment_content = str(decision.get("comment_content") or "").strip()
                if not post_id or decision.get("comment") is not True or not comment_content:
                    continue

                if post_id not in chunk:
                    logger.warning(f"Step {current_step}/{max_step}: Post {post_id} not found in recommendations")
                    continue

                await self._add_comment_and_notify(
                    decision,
                    chunk[post_id],
                    mentionable_id_set,
                    user_id,
                    user_nickname,
                    browse_ms,
                    max(1, window_end_ms - browse_ms + 1),
                    current_step,
                    max_step,
                )
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

    def _check_relationship(self, user_id: str, other_user_id: str, follow_ids: List[str], fan_ids: List[str]) -> str:
        """Check the relationship type with another user"""
        if not other_user_id:
            return "none"
        
        if not user_id or user_id == other_user_id:
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

    def _mention_observation(
        self,
        mention_entries: List[Dict[str, Any]],
        mentionable_users_str: str,
        scene: str,
    ) -> str:
        observation_parts = []
        for i, entry in enumerate(mention_entries):
            observation_parts.append(f"""## Notification {i + 1}
            {entry["mention_action"]}

            Post that was commented on or that @ you:
            {json.dumps(entry["mention_post"], ensure_ascii=False, indent=2)}

            {entry["content_label"]}:
            {entry["mention_comment_content"]}

            {entry["content_label"]} ID: {entry["mention_comment_id"]}
            """)
        return (
            scene
            + "\n\n"
            + "Comments and mentions ({} notifications; give one decision per item, in order):\n\n".format(
                len(mention_entries)
            )
            + "\n".join(observation_parts)
            + "\nPeople you can @ :\n"
            + mentionable_users_str
        )

    async def _ack_mention_pool(self, mention_entries: List[Dict[str, Any]]) -> None:
        comment_ids: List[str] = []
        user_id = ""
        for entry in mention_entries:
            if not entry.get("_processed_successfully", False):
                continue
            pool_key = str(entry.get("mention_key") or "").strip()
            if not pool_key:
                continue
            if "." in pool_key:
                uid, cid = pool_key.split(".", 1)
            else:
                uid, cid = "", pool_key
            uid = uid.strip()
            cid = cid.strip()
            if not cid:
                continue
            if not user_id and uid:
                user_id = uid
            comment_ids.append(cid)
        if not comment_ids:
            return
        if not user_id:
            user_id = str(await self.get_data("id") or "").strip()
        if not user_id:
            logger.error("Failed to remove processed mentions: missing user_id")
            return
        success = await self.update_env_mention_pool(
            user_id,
            {
                "action": "delete_many",
                "comment_ids": comment_ids,
            },
            target_user_id=user_id,
        )
        if not success:
            logger.error(
                f"Failed to bulk-remove {len(comment_ids)} processed mentions for user {user_id}"
            )


    async def _react_to_mention_batch(
        self,
        mention_entries: List[Dict[str, Any]],
        event: MentionEvent,
        instruction: str,
        observation: str,
        mentionable_id_set: Set[str],
        user_id: str,
        nickname: str,
        current_step: int,
        max_step: int,
        allow_search_bookmark: bool,
        force_reply_to_mention: bool = False,
    ) -> Optional[Tuple[List[Event], bool]]:
        for entry in mention_entries:
            entry["_processed_successfully"] = False
        try:
            response = await self.generate_reaction(instruction, observation)
        except (ValueError, json.JSONDecodeError):
            logger.warning(
                f"Step {current_step}/{max_step}: "
                f"UserAgent {self.profile_id} failed to parse mention reaction"
            )
            return None
        except Exception:
            logger.exception(
                f"Step {current_step}/{max_step}: "
                f"UserAgent {self.profile_id} failed to handle mention"
            )
            return None

        response = self._normalize_llm_reaction(response)
        events_to_send: List[Event] = []
        mention_post_id_set: Set[str] = {
            str(e["post_id"]).strip()
            for e in mention_entries
            if isinstance(e, dict) and e.get("post_id")
        }
        if allow_search_bookmark:
            search_event = self._build_search_event(
                response, event, current_step, max_step
            )
            if search_event is not None:
                events_to_send.append(search_event)
            if self.profile:
                bookmark_ids = self._build_bookmark_post_ids(response, mention_post_id_set)
                if bookmark_ids:
                    existing = [
                        str(x).strip()
                        for x in (self.profile.get_data("bookmark_post_ids") or [])
                        if x is not None and str(x).strip()
                    ]
                    self.profile.update_data(
                        "bookmark_post_ids",
                        list(dict.fromkeys(existing + bookmark_ids)),
                    )

        has_reply = False
        current_timestamp = event.timestamp
        mention_window_start_ms = (
            int(current_timestamp) if isinstance(current_timestamp, (int, float)) else 0
        )
        mention_window_duration_ms = int(getattr(event, "timestamp_duration", 0) or 0)
        decisions = response.get("decisions", [])
        if isinstance(decisions, list):
            for i, decision in enumerate(decisions):
                if i >= len(mention_entries):
                    break
                mention_entry = mention_entries[i]
                mention_post = mention_entry["mention_post"]
                mention_post_id = str(mention_entry["post_id"] or "").strip()
                if not isinstance(decision, dict):
                    continue
                if decision.get("comment") is False:
                    if str(decision.get("post_id") or "").strip() == mention_post_id:
                        mention_entry["_processed_successfully"] = True
                    continue
                if decision.get("comment") is not True:
                    continue
                if force_reply_to_mention:
                    mention_comment_id = mention_entry.get("mention_comment_id")
                    if mention_comment_id in (None, ""):
                        continue
                    decision = dict(decision)
                    decision["parent_comment_id"] = mention_comment_id
                post_id = str(decision.get("post_id") or "").strip()
                comment_content = str(decision.get("comment_content") or "").strip()
                if not post_id or not comment_content:
                    continue
                if post_id != mention_post_id:
                    logger.warning(
                        f"Post {post_id} does not match mention post {mention_post_id}, skipping"
                    )
                    continue
                success = await self._add_comment_and_notify(
                    decision,
                    mention_post,
                    mentionable_id_set,
                    user_id,
                    nickname,
                    mention_window_start_ms,
                    mention_window_duration_ms,
                    current_step,
                    max_step,
                )
                if success:
                    has_reply = True
                    mention_entry["_processed_successfully"] = True
        return events_to_send, has_reply

    async def handle_mention(self, event: MentionEvent) -> List[Event]:
        """Handle the @/comment/reply reminder event"""
        current_step = getattr(event, "current_step", 1)
        max_step = getattr(event, "max_step", 8)
        if not await self._should_activate_this_round(current_step, max_step):
            logger.debug(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} handle mention but is not activated this round")
            return []

        # Get the reminder information
        mentions = getattr(event, "mentions", {}) or {}
        if not mentions:
            return []
        logger.debug(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} receive mention event, length of mentions: {len(mentions)}")

        # Get the list of users that can be @
        follow_ids = [str(x).strip() for x in (await self.get_data("follow_ids", []) or []) if x is not None and str(x).strip()]
        fan_ids = [str(x).strip() for x in (await self.get_data("fan_ids", []) or []) if x is not None and str(x).strip()]
        user_id = str(await self.get_data("id") or "").strip()
        mentionable_users = self._get_mentionable_users(follow_ids, fan_ids)
        mentionable_id_set = self._mentionable_id_set(mentionable_users)
        mentionable_users_str = json.dumps(mentionable_users.get("follows", []), ensure_ascii=False, indent=2)

        # Build the reminder information
        mention_entries: List[Dict[str, Any]] = []
        for mention_key, mention_message in mentions.items():
            mention_post = mention_message.get("post")
            if not isinstance(mention_post, dict):
                continue
            mention_comment_id = mention_message.get("comment_id")
            # Resolve this notice's post before fetching or selecting its thread.
            post_id = mention_post.get("post_id") or (mention_key.split("_")[0] if "_" in str(mention_key) else str(mention_key))
            post_id = str(post_id).strip() if post_id is not None else ""
            env = getattr(self, "env", None)
            if env is not None and hasattr(env, "get_committed_comment"):
                target = await env.get_committed_comment(
                    post_id, str(mention_comment_id)
                )
                if target is not None:
                    mention_post = dict(mention_post)
                    mention_post["comments"] = dict(mention_post.get("comments") or {})
                    mention_post["comments"][str(mention_comment_id)] = target
            if env is not None and hasattr(env, "_comment_view_embed"):
                view = await env.snapshot_visible_comments(
                    {post_id: mention_post}, getattr(event, "timestamp", 0),
                    anchors={post_id: [mention_comment_id]},
                )
                mention_post = view[post_id]
            mention_comment_content = mention_message.get("comment_content", "")
            mentioner_id = str(mention_message.get("mentioner_id") or "").strip()
            mentioner_nickname = mention_message.get("mentioner_nickname", "")
            mention_type = mention_message.get("mention_type", "at")
            mention_post_author_id = str(mention_post.get("user_id") or "").strip()
            is_author = (mention_post_author_id == mentioner_id)
            if is_author:
                relationship_hint = " (author)"
            else:
                relationship_type = self._check_relationship(user_id, mentioner_id, follow_ids, fan_ids)
                if relationship_type == "mutual":
                    relationship_hint = " (your friend)"
                elif relationship_type == "follow":
                    relationship_hint = " (someone you follow)"
                elif relationship_type == "fan":
                    relationship_hint = " (your follower)"
                else:
                    relationship_hint = " (a stranger)"
            if mention_type == "at":
                mention_action = f"{mentioner_nickname}{relationship_hint} mentioned you in a comment"
                content_label = "Comment that @ you"
            elif mention_type == "comment":
                mention_action = f"{mentioner_nickname}{relationship_hint} commented on your post"
                content_label = "Comment"
            elif mention_type == "reply":
                mention_action = f"{mentioner_nickname}{relationship_hint} replied to your comment"
                content_label = "Reply"
            else:
                mention_action = f"{mentioner_nickname}{relationship_hint} mentioned you"
                content_label = "Content"
            pool_key = str(mention_key).strip() if mention_key is not None else ""
            if "." not in pool_key:
                inner = pool_key or (
                    str(mention_comment_id).strip() if mention_comment_id is not None else ""
                )
                if not inner:
                    continue
                pool_key = f"{user_id}.{inner}"
            mention_entries.append({
                "mention_key": pool_key,
                "mention_post": mention_post,
                "post_id": post_id,
                "mention_comment_id": mention_comment_id,
                "mention_comment_content": mention_comment_content,
                "mentioner_id": mentioner_id,
                "mentioner_nickname": mentioner_nickname,
                "mention_type": mention_type,
                "mention_action": mention_action,
                "content_label": content_label,
            })

        if not mention_entries:
            return []

        record_mentioned_post_ids_by_channel(
            self.profile,
            self.profile_id,
            current_step,
            mention_entries,
            getattr(event, "timestamp", 0),
        )

        author_thread_entries = [
            entry
            for entry in mention_entries
            if self._is_author_thread_mention(entry, user_id)
        ]
        peer_entries = [
            entry
            for entry in mention_entries
            if not self._is_author_thread_mention(entry, user_id)
        ]

        nickname = await self.get_data("nickname", "")
        events_to_send: List[Event] = []
        has_reply = False

        if author_thread_entries:
            reply_tier = self._profile_tier("comment_reply_tier")
            gates = await UserAgentGates(self).build_author_coaching(
                author_thread_entries, getattr(event, "timestamp", None)
            )
            instruction = build_author_reply(reply_tier, gates)
            if (await self._is_official_by_agent_field()) is True:
                instruction = with_official_policy(instruction, "author_reply")
                logger.info(
                    f"Step {current_step}/{max_step}: User {user_id} author-thread replies "
                    f"use official priority constraint "
                    f"({len(author_thread_entries)} notifications, tier={reply_tier})"
                )
            observation = self._mention_observation(
                author_thread_entries,
                mentionable_users_str,
                "[Scene] Someone commented on your post. You can ignore it. "
                "Reply only when it is worth picking up the thread. "
                "Keep replies short, like a chat, not a briefing.",
            )
            author_result = await self._react_to_mention_batch(
                author_thread_entries,
                event,
                instruction,
                observation,
                mentionable_id_set,
                user_id,
                nickname,
                current_step,
                max_step,
                allow_search_bookmark=False,
                force_reply_to_mention=True,
            )
            if author_result is not None:
                batch_events, batch_reply = author_result
                events_to_send.extend(batch_events)
                has_reply = has_reply or batch_reply
                await self._ack_mention_pool(author_thread_entries)

        if peer_entries:
            gates = await UserAgentGates(self).build_mention_coaching(
                peer_entries, getattr(event, "timestamp", None)
            )
            instruction = format_prompt(
                "tasks/mention_reaction.txt",
                memory_coaching=gates.memory_coaching,
                memory_ref=gates.memory_ref,
                similarity_kw_coaching=gates.similarity_kw_coaching,
                similarity_emb_coaching=gates.similarity_emb_coaching,
            )
            if (await self._is_official_by_agent_field()) is True:
                instruction = with_official_policy(instruction, "mention")
                logger.info(
                    f"Step {current_step}/{max_step}: User {user_id} peer mentions "
                    f"use official priority constraint ({len(peer_entries)} notifications)"
                )
            observation = self._mention_observation(
                peer_entries,
                mentionable_users_str,
                "[Scene] You got notifications (@ / reply). You can ignore them. "
                "Reply only when it is worth picking up the thread. "
                "Keep replies short, like a chat, not a briefing.",
            )
            peer_result = await self._react_to_mention_batch(
                peer_entries,
                event,
                instruction,
                observation,
                mentionable_id_set,
                user_id,
                nickname,
                current_step,
                max_step,
                allow_search_bookmark=True,
            )
            if peer_result is not None:
                batch_events, batch_reply = peer_result
                events_to_send.extend(batch_events)
                has_reply = has_reply or batch_reply
                await self._ack_mention_pool(peer_entries)

        if has_reply:
            events_to_send.append(
                MentionSpreadingEvent(
                    from_agent_id=self.profile_id,
                    to_agent_id="ENV",
                    timestamp=event.timestamp,
                    timestamp_duration=int(getattr(event, "timestamp_duration", 0) or 0),
                    current_step=current_step,
                    max_step=max_step,
                )
            )

        return events_to_send
