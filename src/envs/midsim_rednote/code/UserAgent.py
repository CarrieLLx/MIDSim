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
    record_mentioned_note_ids_by_channel,
    record_recommendations_by_source_step,
)
from .utils import (
    format_historical_summary,
    generate_comment_id,
    time_in_window,
    generate_comment_timestamp,
    resolve_parent_comment_entry,
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
        self.register_event("StartEvent", "generate_memory_from_own_notes")
        self.register_event("SocialRecommendationEvent", "receive_recommendation")
        self.register_event("AlgorithmRecommendationEvent", "receive_recommendation")
        self.register_event("SearchResultEvent", "receive_recommendation")
        self.register_event("KeepFollowingEvent", "receive_recommendation")
        self.register_event("MentionEvent", "handle_mention")
        self.register_event("AddCommentResponseEvent", "handle_add_comment_response")
        self.register_event("MentionPoolUpdateResponseEvent", "handle_update_mention_pool_response")
        self._comment_add_futures: Dict[str, Future] = {}
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
            step_idx = current_step
            max_step = max_step
            p_t = activity_level

            activated = random.random() < p_t
            if self.profile:
                self.profile.update_data("login", 1 if activated else 0)

            return activated

    async def update_env_mention_pool(self, key: str, value: Any, parent_event_id: Optional[str] = None) -> bool:
        """Update the mention_pool in the environment (using distributed lock)"""
        # Convert mentioner_id.comment_id to the full key format: mention_pool.mentioner_id.comment_id
        full_key = f"mention_pool.{key}"
        
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
        lock_id = f"env_mention_pool_update_lock_{key}"
        lock = await get_lock(lock_id)

        try:
            async with lock:
                from onesim.events import get_event_bus
                event_bus = get_event_bus()
                await event_bus.dispatch_event(mention_pool_update_event)

            try:
                if hasattr(self, '_sync_event'):
                    await asyncio.wait_for(self._sync_event.wait(), timeout=_ENV_ASYNC_OP_TIMEOUT)
                    return await future
                else:
                    return await asyncio.wait_for(future, timeout=_ENV_ASYNC_OP_TIMEOUT)
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

        # Get current available notes list
        current_notes = getattr(event, "current_notes", None) or {}
        if not isinstance(current_notes, dict):
            current_notes = {}

        # Algorithm recommendation stream
        # Get specified algorithm types
        fixed_algorithm_types = await user_default_algorithm_types(self)
        if not isinstance(fixed_algorithm_types, list) or not fixed_algorithm_types:
            raise ValueError("default_algorithm_types must be a non-empty list")
        allowed_algorithm_types = set(self.recommender_map.keys())

        # Get recommended note IDs
        raw_rec = self.profile.get_data("recommended_note_ids", []) if self.profile else []
        if not isinstance(raw_rec, list):
            raw_rec = []
        recommended_note_ids = {str(x).strip() for x in raw_rec if x is not None and str(x).strip()}

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
                    current_notes=current_notes,
                    recommended_note_ids=recommended_note_ids,
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
                note = mm.get("note")
                if isinstance(note, dict):
                    nid = note.get("note_id") or note.get("id")
                    if nid is not None:
                        nid_key = str(nid).strip()
                        pool_note = current_notes.get(nid_key) if isinstance(current_notes, dict) else None
                        if isinstance(pool_note, dict):
                            mm["note"] = dict(pool_note)
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
        keep_following_notes = {}
        if self.profile and self.profile.get_data("keep_following_note_ids", []) is not None:
            keep_ids = self.profile.get_data("keep_following_note_ids", []) or []
            if isinstance(keep_ids, (list, tuple)) and keep_ids:
                readded = 0
                for keep_note_id in keep_ids:
                    if keep_note_id in keep_following_notes:
                        continue
                    note = current_notes.get(keep_note_id)
                    if not isinstance(note, dict):
                        continue
                    keep_following_notes[keep_note_id] = note
                    readded += 1
                if readded > 0:
                    logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} re-added keep_following notes: {readded}")
                self.profile.update_data("keep_following_note_ids", [])
                
        if len(keep_following_notes) > 0:
            events.append(KeepFollowingEvent(
                from_agent_id=self.profile_id,
                to_agent_id=self.profile_id,
                timestamp=current_ts,
                timestamp_duration=int(getattr(event, "timestamp_duration", 0) or 0),
                current_step=current_step,
                max_step=max_step,
                recommendations=keep_following_notes,
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
                    lower = current_ts - int(_login_cap_days * 86400000)
                    if last_login < lower:
                        last_login = lower
            else:
                last_login = 0

        step_duration = int(getattr(event, "timestamp_duration", 0) or 0)
        window_end = current_ts + step_duration

        recommendations = {}
        for note_id, note in current_notes.items():
            if not isinstance(note, dict):
                continue
            author_id = note.get("user_id") or note.get("author_id")
            if author_id not in follow_set:
                continue
            t = note.get("time") or note.get("create_time")
            if t is None:
                continue
            try:
                t = int(t)
            except (TypeError, ValueError):
                continue
            if last_login <= t < window_end:
                nid_key = str(note_id).strip() if note_id is not None else ""
                if nid_key and nid_key in recommended_note_ids:
                    continue
                recommendations[note_id] = note

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
                logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} send SocialRecommendationEvent, length of recommendations: {len(recommendations)}")

                events.append(SocialRecommendationEvent(
                    from_agent_id=self.profile_id,
                    to_agent_id=self.profile_id,
                    timestamp=current_ts,
                    timestamp_duration=int(getattr(event, "timestamp_duration", 0) or 0),
                    current_step=current_step,
                    max_step=max_step,
                    recommendations=recommendations,
                ))
        return events

    async def add_env_comments(self, key: str, value: Any, parent_event_id: Optional[str] = None) -> bool:
        """Add comments to the data in the environment (using distributed lock)"""
        # Convert note_id to the full key format: content_pool.note_id.comments
        full_key = f"content_pool.{key}.comments"
        
        # Create a unique request ID
        request_id = f"agent_env_add_comments_req_{time.time()}_{id(self)}"

        # Create a Future to receive the response
        future = asyncio.Future()
        self._comment_add_futures[request_id] = future

        # Create a comment add event
        comment_add_event = AddCommentEvent(
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
        lock_id = f"env_comment_add_lock_{key}"
        lock = await get_lock(lock_id)

        try:
            # Get the lock before sending the update
            async with lock:
                from onesim.events import get_event_bus
                event_bus = get_event_bus()
                await event_bus.dispatch_event(comment_add_event)

                try:
                    if hasattr(self, '_sync_event'):
                        await asyncio.wait_for(self._sync_event.wait(), timeout=_ENV_ASYNC_OP_TIMEOUT)
                        return await future
                    else:
                        return await asyncio.wait_for(future, timeout=_ENV_ASYNC_OP_TIMEOUT)
                except asyncio.TimeoutError:
                    logger.warning(f"Environment comment add timeout: {key}")
                    self._comment_add_futures.pop(request_id, None)
                    return False
                except Exception as e:
                    logger.error(f"Error adding environment comment: {e}")
                    self._comment_add_futures.pop(request_id, None)
                    return False
        except Exception as e:
            logger.error(f"Error getting environment comment add lock: {e}")
            return False

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

    async def generate_memory_from_own_notes(self, event: Event) -> List[Event]:
        """For user to generate memory from their own notes"""
        current_step = getattr(event, "current_step", 1)
        max_step = getattr(event, "max_step", 8)
        user_id = await self.get_data("id")
        if not user_id:
            return []

        current_ts = int(event.timestamp) if getattr(event, "timestamp", None) is not None else 0
        duration = int(getattr(event, "timestamp_duration", 0) or 0)

        current_notes = getattr(event, "current_notes", None) or {}
        if not isinstance(current_notes, dict) or not current_notes:
            return []

        # Filter own notes
        own_notes = []
        for note_id, note in current_notes.items():
            if not isinstance(note, dict):
                continue
            if (note.get("user_id") or note.get("author_id")) != user_id:
                continue
            own_notes.append((note_id, note))

        lo = float(current_ts)
        dur_ms = float(duration)
        cap = getattr(event, "simulation_cap_timestamp", None)
        if cap is not None:
            hi = min(lo + dur_ms, float(cap))
        else:
            hi = lo + dur_ms
        if hi <= lo:
            return []
        own_notes = [(nid, n) for nid, n in own_notes if time_in_window(n, lo, hi)]

        if not own_notes:
            return []

        hi_f = float(hi)
        seven_span = 14 * 24 * 3600 * (1000.0 if hi_f >= 1e11 else 1.0)
        cutoff_7d = hi_f - seven_span

        own_notes_for_prompt = []
        own_notes_for_supplementary = []
        supplementary_note_ids: Set[str] = set()
        for note_id, note in own_notes:
            own_notes_for_prompt.append({
                "note_id": note_id,
                "title": note.get("title", ""),
                "desc": note.get("desc", ""),
                "tags_list": note.get("tags_list", []),
                "time": note.get("time", note.get("create_time", 0)),
            })
            comments = note.get("comments") or {}
            if not isinstance(comments, dict):
                comments = {}
            comments_brief = []
            for cid, c in comments.items():
                if isinstance(c, dict):
                    prev = (c.get("content") or "")[:160]
                    comments_brief.append({
                        "comment_id": cid,
                        "user_id": c.get("user_id"),
                        "content_preview": prev,
                    })
            raw_t = note.get("time", note.get("create_time"))
            try:
                t_note = float(raw_t)
            except (TypeError, ValueError):
                t_note = None
            if t_note is not None and cutoff_7d <= t_note < hi_f:
                supplementary_note_ids.add(note_id)
                own_notes_for_supplementary.append({
                    "note_id": note_id,
                    "title": note.get("title", ""),
                    "desc": note.get("desc", ""),
                    "tags_list": note.get("tags_list", []),
                    "time": note.get("time", note.get("create_time", 0)),
                    "existing_comments": comments_brief,
                })

        # Build the instruction and observation for the memory generation
        instruction = (
            "你正在复盘自己最近发布的内容。请基于这些帖子，沉淀一条第一人称记忆，"
            "重点总结：你最近持续关注的话题、表达风格、以及后续互动中可复用的表达策略。"
        )
        observation = f"你最近发布的帖子：\n{json.dumps(own_notes_for_prompt, ensure_ascii=False, indent=2)}"
        reaction = {
            "task": "self_reflection_on_own_notes",
            "own_note_count": len(own_notes_for_prompt),
            "highlights": own_notes_for_prompt,
        }

        try:
            # Generate the memory from the own notes
            memory_text = await self.generate_memory(instruction, observation, reaction)
            if memory_text:
                logger.info(f"Step {current_step}/{max_step}: User {user_id} generated self-memory from {len(own_notes_for_prompt)} own notes, memory_text: {memory_text}")
        except Exception as e:
            logger.error(f"Step {current_step}/{max_step}: User {user_id} failed to generate memory from own notes: {e}")

        # Decide whether to add a comment to the own notes
        profile_snippet = ""
        if self.profile:
            try:
                p = self.profile.get_profile(include_private=False) or {}
                profile_snippet = json.dumps(
                    {
                        "nickname": p.get("nickname", ""),
                        "description": (p.get("description") or "")[:400],
                    },
                    ensure_ascii=False,
                )
            except Exception:
                profile_snippet = ""

        supplementary_instruction = """你正在查看自己最近发布的帖子（均为本人账号发布）。

        总体原则：默认多数帖子 comment 应为 false，避免为刷存在感、纯礼貌附和、复述正文或已有评论而补评。
        
        在默认克制的前提下，若出现下列**任一类**且**一句话就能说清楚**，可以 comment 为 true（仍须简短自然）：
        · 正文有明显事实/数据/引用错误需勘误，或关键信息遗漏可能误导读者；
        · 评论区有人提出共同的**具体问题**、误解或需要一句话澄清的点；
        · 有可执行的**小补充**（步骤、参数、链接类提示、边界条件），且正文未写全、对读者有明确帮助。
        
        不建议：纯表情/语气词、与正文完全重复、仅为「显得活跃」而评；若帖子与评论区已自洽，一律不补。
        
        数量感（软约束，写在决策里即可，不必单独字段）：同一批列表里，**通常 0 条**；确有多个强理由时，**优先只选最有价值的 1～2 条** comment=true，其余保持 false。不要为了「稍微宽松」而批量开评。
        
        若不需要补充：该条 comment 必须为 false，comment_content 为空字符串。
        若确需补充：comment 为 true，给出简短 comment_content；若要回复某条已有评论，填写 parent_comment_id（否则为 null）。

        请严格返回 JSON（可先写简短理解，再给出 decisions；decisions 中须覆盖下列列表中的每一个 note_id 各一条）：
        {
        "persona_understanding": "1句",
        "content_understanding": "1句",
        "decisions": [
            {
            "note_id": "必须与下方列表中的 note_id 一致",
            "comment": false,
            "parent_comment_id": null,
            "comment_content": "",
            "decision_reason": "不补充或补充的原因（≤25字）",
            "expression_reason": "",
            "mention_reasoning": []
            },
            {
            "note_id": "必须与下方列表中的 note_id 一致",
            "comment": true,
            "parent_comment_id": null,
            "comment_content": "补充的评论内容",
            "decision_reason": "补充评论的原因（≤25字）",
            "expression_reason": "",
            "mention_reasoning": []
            }
        ]
        }"""

        if not own_notes_for_supplementary:
            return []

        observation_sup = (
            f"你本人在最近发布的帖子（含已有评论概要，便于判断是否回复某条；时间以本轮仿真时间窗结束时为「现在」）：\n"
            f"{json.dumps(own_notes_for_supplementary, ensure_ascii=False, indent=2)}"
        )

        try:
            # Decide whether to add a comment to the own notes
            sup_response = await self.generate_reaction(supplementary_instruction, observation_sup)
            sup_response = self._normalize_llm_reaction(sup_response)
        except Exception as e:
            return []

        decisions = sup_response.get("decisions", [])
        if not isinstance(decisions, list):
            return []

        nickname = await self.get_data("nickname", "")
        ip_location = await self.get_data("ip_location", "")
        notes_by_id = {nid: n for nid, n in own_notes}

        # Add comments to the own notes
        for decision in decisions:
            if not isinstance(decision, dict):
                continue
            note_id = decision.get("note_id")
            if not note_id or note_id not in supplementary_note_ids:
                continue
            if not decision.get("comment", False):
                continue
            comment_content = (decision.get("comment_content") or "").strip()
            if not comment_content:
                continue
            parent_comment_id = decision.get("parent_comment_id")
            note = notes_by_id.get(note_id) or {}
            comments = note.get("comments") or {}
            if not isinstance(comments, dict):
                comments = {}
            canonical_parent_id, _ = resolve_parent_comment_entry(
                comments, parent_comment_id
            )
            if parent_comment_id is not None and canonical_parent_id is None:
                logger.warning(
                    f"Step {current_step}/{max_step}: Own-note supplementary skip: parent_comment_id {parent_comment_id} not in note {note_id}"
                )
                continue

            comment_id = generate_comment_id()
            success = await self.add_env_comments(
                note_id,
                {
                    "comment_id": comment_id,
                    "timestamp": generate_comment_timestamp(note, current_ts, duration),
                    "ip_location": ip_location,
                    "note_id": note_id,
                    "user_id": user_id,
                    "nickname": nickname,
                    "parent_comment_id": canonical_parent_id if parent_comment_id is not None else None,
                    "at_count": 0,
                    "content": comment_content,
                },
            )
            if not success:
                logger.error(f"Step {current_step}/{max_step}: Failed to add self supplementary comment on own note {note_id}")
                continue
            logger.info(
                f"Step {current_step}/{max_step}: User {user_id} added self supplementary comment on own note {note_id} (no notifications)"
            )

        return []

    def _filter_recommendations(self, recommendations: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Remove duplicate recommendations"""
        if not recommendations:
            return {}
        
        # Get the set of recommended content IDs (from profile)
        recommended_note_ids = set(self.profile.get_data("recommended_note_ids", [])) if self.profile else set()
        
        # Remove duplicate recommendations 
        filtered: Dict[str, Dict[str, Any]] = {}

        for note_id, rec in recommendations.items():
            if not isinstance(rec, dict):
                logger.warning(f"Recommendation {note_id} is not a dictionary")
                continue

            # If the note ID does not exist or has already been recommended, skip
            if not note_id or note_id in recommended_note_ids:
                logger.info(f"Recommendation {note_id} is already recommended")
                continue
            filtered[note_id] = rec
            logger.info(f"Recommendation {note_id} is added to filtered list")

        return filtered

    def _add_recommendations(self, recommendations: Dict[str, Any]) -> None:
        """Add the note_ids of the current recommendations to profile.recommended_note_ids."""
        if not recommendations:
            return
        recommended_note_ids = set(self.profile.get_data("recommended_note_ids", [])) if self.profile else set()
        new_note_ids = []
        for note_id in recommendations.keys():
            new_note_ids.append(note_id)
        
        # Add the new recommended content IDs to the recommended list
        if new_note_ids and self.profile:
            all_recommended = list(recommended_note_ids) + new_note_ids
            self.profile.update_data("recommended_note_ids", all_recommended)

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
                    info["historical_summary"] = format_historical_summary(info.get("historical_summary"))
                hn = info.get("historical_notes")
                if isinstance(hn, dict) and len(hn) > 2:
                    items = list(hn.items())[:2]
                    info["historical_notes"] = dict(items)
                follows_info.append(info)
        
        # Process the fan list (including mutual)
        for user_id in fan_ids:
            rel = self.relationship_manager.get_relationship(user_id)
            # Check if rel is None, then access target_info
            if rel and rel.target_info and isinstance(rel.target_info, dict):
                info = dict(rel.target_info)
                if "historical_summary" in info:
                    info["historical_summary"] = format_historical_summary(info.get("historical_summary"))
                hn = info.get("historical_notes")
                if isinstance(hn, dict) and len(hn) > 2:
                    items = list(hn.items())[:2]
                    info["historical_notes"] = dict(items)
                fans_info.append(info)
        
        # Process the mutual list (the historical notes of the mutual users are kept at most 2)
        for user_id in mutual_ids:
            rel = self.relationship_manager.get_relationship(user_id)
            # Check if rel is None, then access target_info
            if rel and rel.target_info and isinstance(rel.target_info, dict):
                info = dict(rel.target_info)
                if "historical_summary" in info:
                    info["historical_summary"] = format_historical_summary(info.get("historical_summary"))
                hn = info.get("historical_notes")
                if isinstance(hn, dict) and len(hn) > 2:
                    items = list(hn.items())[:2]
                    info["historical_notes"] = dict(items)
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
            logger.warning(f"Step {current_step}/{max_step}: UserAgent {self.profile_id}, unknown recommendation event {evt_cls}, skip")
            return []

        logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} receive recommendation event, source type: {source_type}, length of recommendations: {len(event.recommendations)}")

        # algorithm/social: remove duplicate recommendations from profile.recommended_note_ids
        recommendations = event.recommendations
        record_recommendations_by_source_step(
            self.profile,
            self.profile_id,
            source_type,
            current_step,
            recommendations,
            getattr(event, "timestamp", 0),
        )
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
        window_start_ms = int(current_timestamp) if isinstance(current_timestamp, (int, float)) else 0
        window_duration_ms = int(getattr(event, "timestamp_duration", 0) or 0)
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
            has_self_note = any(
                isinstance(note, dict) and note.get("user_id") == user_id
                for note in chunk.values()
            )
            if has_self_note:
                source_name = "自己发布（你的内容）"
            else:
                source_name = "关注流（来自你关注的用户）" if source_type == "social" else "推荐流（来自算法）"

            observation = f"""【场景】你正在用手机刷信息流：大部分内容划走即可，偶尔会停下来打一行字评论。你不是在完成实验任务，也不是写舆情分析。

            推荐来源：{source_name}

            收到的推荐内容：
            {recommendations_str}

            可@的用户列表：
            {mentionable_users_str}"""

            gates = await UserAgentGates(self).build_recommendation_coaching(chunk, current_timestamp)

            instruction = f"""根据用户 Profile、historical_summary、memory 和推荐内容，完成是否评论/回复及内容生成。

            步骤1：对每条推荐先设默认状态
            - "comment": false
            - "parent_comment_id": null
            - "comment_content": ""

           **comment 的数量决策仅由步骤2/3/4决定，不得因步骤5改变；**
            {gates.memory_coaching}

            步骤2：兴趣判断
            - 通读本批次所有推荐内容，判断哪些笔记能够进入候选池，当且仅当同时满足以下条件时，再将该笔记改为 comment=true：
                {gates.memory_rec}
                1. **对笔记感兴趣**：整帖与 Profile/historical_summary/memory 高度相关
                2. **对某条楼中评论感兴趣**：未必对整帖强兴趣，但**某条具体评论**与 Profile/historical_summary/memory 高度相关，且你愿意针对该评论做**增量回应**（补充、纠错、追问、共情、接梗）
                3. 互动对象关系与场景合适（关注关系优先）；
                4. 表达目的明确（补充事实、推进互动）
            - 进入候选池时，请在心中标记：本轮互动是「对帖」还是「对某条评论」；若选「对某条评论」，必须在后续 JSON 里使用非空的 parent_comment_id（见步骤4）。
            - 对不感兴趣的笔记：该笔记不进入候选池，comment=false。

            步骤3：选取 0～{gates.k_diff_targets} 目标笔记
            - {gates.k_diff_targets} 表示本轮**最多**对多少个**不同** note_id 置 comment=true；实际条数可为 0～{gates.k_diff_targets} 中任意值，**不得**为凑满条数而放宽步骤1.5/步骤2。
            - 仅在步骤2 的候选池内、按你的排序取前若干条作为“目标 note_id 集合”，且条数不超过 {gates.k_diff_targets}；若候选少于 {gates.k_diff_targets}，有多少算多少；其余笔记保持 comment=false。

            步骤4：为每个目标 note_id 生成 0～{gates.k_same_target} 条评论
            - **回帖 vs 回评论（务必二选一写进结构）**：
                · 对帖首评：`parent_comment_id` 必须为 null，comment_content 针对笔记正文。
                · 回复楼中某条：`parent_comment_id` 必须为推荐里给出的**真实 comment_id**（不得编造），comment_content 必须**承接该条评论的具体词句或论点**，禁止像对帖空泛表态。
            - 对每个目标 note_id：在 decisions 中写出 {gates.k_same_target} 个 comment=true 的条目（可用于追评链/补一句）。
            - 若 {gates.k_same_target}=1：每个目标 note_id 只生成 1 条评论。
            - 若 {gates.k_same_target}>=2：默认同时评论笔记和评论，也允许同帖多条，但第2条及之后必须是增量（补充新点/纠错/情绪加码），禁止复述同一句。

            步骤5：根据人设，为每条评论选择 comment_mode，默认为 attitude
            - comment_mode ∈ {{attitude, discussion, ultra_short}}，并在 expression_reason 体现。
            - 发生概率：attitude > discussion > ultra_short
            - 共同特征：口语化，以第一人称视角表达，不要求语法正确
                - attitude（态度）：以态度、吐槽、捧场、玩梗为主，信息增量很少。常见极短句和纯表情、语气词，上限7字
                - discussion（讨论）：提问、转述事实、讨论机制或后果，上限25字
                - ultra_short（超短互动）：仅有表情标签而无陈述，或仅有1个字
            - 输出风格
                - 使用用户所在地区常用语言；默认使用中文，允许中英文混合。
                - 表达应符合 Profile 与 historical_summary 的人设口吻；
                - 减少模板化套话，如“你说得对/确实/很有道理/希望大家…”；
                - 评论内容不包含标签（#...）。

            步骤6：决定是否需要 @ 用户
            - 仅当对方在可@列表且与内容强相关 + 不是笔记作者/父评论作者时，才在正文中额外 @user_id；否则 mention_reasoning 为 []。

            步骤7：keep_following_note_ids
            - 仅当：高度感兴趣 + 当前内容有持续关注价值→ keep_following_note_ids=true；否则为空列表 []

            步骤8：search
            - 仅当：高度感兴趣 + 当前信息明显不足 + memory 无相关内容 → search=true；否则 false。

            【平台表情】
            [doge][害羞R][飞吻R][哭惹R][汗颜R][捂脸R][笑哭R][偷笑R][生气R][赞R][棒R][嘻嘻R][石化R][暗中观察R][微笑R][大笑R][合十R][害羞R][呃R][失望R]

            请按以下 JSON 返回（字段顺序固定）：
            {{
            "persona_understanding": "1-2句。概括身份、兴趣、语言风格（简短）",
            "content_understanding": "1-2句。概括本次内容与你的相关性、你优先关注的角度",
            "source_understanding": "关注流=关注关系；推荐流=陌生关系，自己发布（你的内容）=自己发布",
            "memory_reflection": "{gates.memory_ref}",
            "decisions": [
                {{
                "note_id": "note_id",
                "comment": false,
                "parent_comment_id": null,
                "comment_content": "",
                "decision_reason": "不评论的原因（1句，<20字）",
                "expression_reason": "",
                "mention_reasoning": []
                }},
                {{
                "note_id": "note_id",
                "comment": true,
                "parent_comment_id": null,
                "comment_content": "口语短回复或轻互动或超短互动",
                "decision_reason": "评论的理由或原因（1句，<20字）",
                "expression_reason": "为何用这种语气；若为 substantive/micro 请体现（1句）",
                "mention_reasoning": [
                    {{
                    "user_id": "被@用户id",
                    "persona_understanding": "对该用户的简短理解",
                    "mention_reason": "为何需要额外提醒该用户（1句，<20字）"
                    }}
                ]
                }},
                {{
                "note_id": "note_id",
                "comment": true,
                "parent_comment_id": "comment_id_to_reply_to",
                "comment_content": "口语短回复或轻互动或超短互动",
                "decision_reason": "回复的理由（1句，<20字）",
                "expression_reason": "为何用这种语气（1句，<20字）",
                "mention_reasoning": []
                }}
            ],
            "keep_following_note_ids": [],
            "keep_following_reason": "为何保持关注（1句，<20字）",
            "search": false,
            "search_keyword": "搜索关键词（1句，≤20字）",
            "search_reason": "是否搜索及原因（1句）"
            }}

           输出规则：
            - 先给出理解字段，再给 decisions；
            - decisions 中每条先按默认状态填写 comment=false，再仅对满足条件的条目改为 comment=true；
            - comment=false 时，comment_content 必须为空字符串。
            - comment_content 建议根据 comment_mode 选择合适的表达方式，尽量符合人设口吻，减少模板化套话。
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
            response = self._normalize_llm_reaction(response)
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
                keep_ids = response.get("keep_following_note_ids", [])
                if isinstance(keep_ids, list) and keep_ids:
                    # Only allow note_ids within the current batch, and at most 1
                    valid_keep_ids = []
                    for keep_note_id in keep_ids:
                        if keep_note_id in chunk:
                            valid_keep_ids.append(keep_note_id)
                    if valid_keep_ids:
                        self.profile.update_data("keep_following_note_ids", valid_keep_ids[:1])
                    else:
                        self.profile.update_data("keep_following_note_ids", [])

            # Handle comment decisions
            decisions = response.get("decisions", [])
            if not isinstance(decisions, list):
                continue
       
            # Handle each decision: update comment count and comment content
            for decision in decisions:
                if not isinstance(decision, dict):
                    continue
                    
                note_id = decision.get("note_id")
                should_comment = decision.get("comment", False)
                parent_comment_id = decision.get("parent_comment_id")  # If it is a reply comment, specify the parent comment ID
                comment_content = decision.get("comment_content", "")
                    
                if not note_id or not should_comment:
                    continue

                # Check if the note_id is valid
                if note_id not in chunk:
                    logger.warning(f"Step {current_step}/{max_step}: Note {note_id} not found in recommendations")
                    continue
            
                # Parse the @users in the comment content, replace @id with @nickname, and return the list of user IDs
                mentioned_user_ids = []
                mention_reasoning = decision.get("mention_reasoning", [])
                if isinstance(mention_reasoning, list):
                    for mention_reason in mention_reasoning:
                        if isinstance(mention_reason, dict):
                            mentioned_uid = mention_reason.get("user_id")
                            if mentioned_uid:
                                mentioned_user_ids.append(mentioned_uid)

                note = chunk[note_id]
                note_author_id = note.get("user_id")
                comments_map = note.get("comments")
                if not isinstance(comments_map, dict):
                    comments_map = {}
                mentioned_user_ids = [
                    uid for uid in mentioned_user_ids if uid != note_author_id
                ]
                comment_author_id = None
                canonical_parent_id: Optional[str] = None
                if parent_comment_id is not None:
                    canonical_parent_id, parent_entry = resolve_parent_comment_entry(
                        comments_map, parent_comment_id
                    )
                    if isinstance(parent_entry, dict):
                        comment_author_id = parent_entry.get("user_id")
                    mentioned_user_ids = [
                        uid for uid in mentioned_user_ids if uid != comment_author_id
                    ]

                mention_count = len(mentioned_user_ids)
                
                # If it is a reply comment, check if the parent comment ID is valid
                if parent_comment_id is not None:
                    if canonical_parent_id is not None:
                        # Generate a unique comment ID
                        comment_id = generate_comment_id()
                        
                        success = await self.add_env_comments(note_id, {
                            "comment_id": comment_id,
                            "timestamp": generate_comment_timestamp(
                                note, window_start_ms, window_duration_ms
                            ),
                            "ip_location": ip_location,
                            "note_id": note_id,
                            "user_id": user_id,
                            "nickname": user_nickname,
                            "parent_comment_id": canonical_parent_id,
                            "at_count": mention_count,
                            "content": comment_content
                        })
                        if not success:
                            logger.error(f"Failed to add comment to note {note_id}")
                            continue
                        
                        # Send reminder to the comment author
                        if comment_author_id and comment_author_id != user_id:  # Do not send reminder to yourself
                            success = await self.update_env_mention_pool(f"{comment_author_id}.{comment_id}", {
                                "action": "add",
                                "mention_message": {
                                    "note_id": note_id,
                                    "comment_id": comment_id,
                                    "comment_content": comment_content,
                                    "mentioner_id": user_id,
                                    "mentioner_nickname": user_nickname,
                                    "mention_type": "reply"
                                }
                            })
                            if not success:
                                logger.error(f"Failed to update mention pool for comment {comment_id} by {user_id} on note {note_id}")
                                continue
                            logger.info(f"User {user_id} replied to comment {canonical_parent_id} by {comment_author_id} on note {note_id}")
                    
                        # Send reminder to the note author
                        if note_author_id and note_author_id != user_id and note_author_id != comment_author_id:  # 不给自己发提醒，也不给评论作者发提醒
                            success = await self.update_env_mention_pool(f"{note_author_id}.{comment_id}", {
                                "action": "add",
                                "mention_message": {
                                    "note_id": note_id,
                                    "comment_id": comment_id,
                                    "comment_content": comment_content,
                                    "mentioner_id": user_id,
                                    "mentioner_nickname": user_nickname,
                                    "mention_type": "comment"
                                }
                            })
                            if not success:
                                logger.error(f"Failed to update mention pool for comment {comment_id} by {user_id} on note {note_id}")
                                continue
                            logger.info(f"User {user_id} commented on note {note_id} by {note_author_id}")

                    else:
                        logger.warning(f"Parent comment {parent_comment_id} not found in note {note_id}, skipping reply")
                        continue
                else:
                    # If it is a comment on the note, add the comment
                    comment_id = generate_comment_id()
                    
                    success = await self.add_env_comments(note_id, {
                        "comment_id": comment_id,
                        "timestamp": generate_comment_timestamp(
                            note, window_start_ms, window_duration_ms
                        ),
                        "ip_location": ip_location,
                        "note_id": note_id,
                        "user_id": user_id,
                        "nickname": user_nickname,
                        "parent_comment_id": None,
                        "at_count": mention_count,
                        "content": comment_content
                    })
                    if not success:
                        logger.error(f"Failed to add comment to note {note_id}")
                        continue

                    # Send reminder to the note author
                    if note_author_id and note_author_id != user_id:  # Do not send reminder to yourself
                        success = await self.update_env_mention_pool(f"{note_author_id}.{comment_id}", {
                            "action": "add",
                            "mention_message": {
                                "note_id": note_id,
                                "comment_id": comment_id,
                                "comment_content": comment_content,
                                "mentioner_id": user_id,
                                "mentioner_nickname": user_nickname,
                                "mention_type": "comment"
                            }
                        })
                        if not success:
                            logger.error(f"Failed to update mention pool for comment {comment_id} by {user_id} on note {note_id}")
                            continue
                        logger.info(f"Step {current_step}/{max_step}: User {user_id} commented on note {note_id} by {note_author_id}")
                   

                # Send MentionEvent to the users mentioned
                if mentioned_user_ids:
                    for mentioned_user_id in mentioned_user_ids:
                        if mentioned_user_id and mentioned_user_id != user_id:  
                            success = await self.update_env_mention_pool(f"{mentioned_user_id}.{comment_id}", {
                                "action": "add",
                                "mention_message": {
                                    "note_id": note_id,
                                    "comment_id": comment_id,
                                    "comment_content": comment_content,
                                    "mentioner_id": user_id,
                                    "mentioner_nickname": user_nickname,
                                    "mention_type": "at"
                                }
                            })
                            if not success:
                                logger.error(f"Failed to update mention pool for comment {comment_id} by {user_id} on note {note_id}")
                                continue
                            logger.info(f"Step {current_step}/{max_step}: User {user_id} mentioned {mentioned_user_id} in comment on note {note_id}")
                    
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

        # Build the reminder information
        mention_entries: List[Dict[str, Any]] = []
        for mention_key, mention_message in mentions.items():
            mention_note = mention_message.get("note")
            if not isinstance(mention_note, dict):
                continue
            mention_comment_id = mention_message.get("comment_id")
            mention_comment_content = mention_message.get("comment_content", "")
            mentioner_id = mention_message.get("mentioner_id")
            mentioner_nickname = mention_message.get("mentioner_nickname", "")
            mention_type = mention_message.get("mention_type", "at")
            mention_note_author_id = mention_note.get("user_id")
            is_author = (mention_note_author_id == mentioner_id)
            if is_author:
                relationship_hint = "（作者）"
            else:
                relationship_type = self._check_relationship(my_id, mentioner_id, follow_ids, fan_ids)
                if relationship_type == "mutual":
                    relationship_hint = "（你的好友）"
                elif relationship_type == "follow":
                    relationship_hint = "（你的关注）"
                elif relationship_type == "fan":
                    relationship_hint = "（你的粉丝）"
                else:
                    relationship_hint = ""
            if mention_type == "at":
                mention_action = f"{mentioner_nickname}{relationship_hint}在评论中@了你"
                content_label = "@你的评论内容"
            elif mention_type == "comment":
                mention_action = f"{mentioner_nickname}{relationship_hint}评论了你的笔记"
                content_label = "评论内容"
            elif mention_type == "reply":
                mention_action = f"{mentioner_nickname}{relationship_hint}回复了你的评论"
                content_label = "回复内容"
            else:
                mention_action = f"{mentioner_nickname}{relationship_hint}提到了你"
                content_label = "内容"
            if relationship_hint and relationship_hint != "（作者）":
                relationship_source = f"{mentioner_nickname} 与你的关系：{relationship_hint}"
            else:
                relationship_source = f"{mentioner_nickname} 与你的关系：陌生人（既不在你的关注列表也不在你的粉丝列表中）"
            note_id = mention_note.get("note_id") or (mention_key.split("_")[0] if "_" in str(mention_key) else str(mention_key))
            mention_entries.append({
                "mention_key": mention_key,
                "mention_note": mention_note,
                "note_id": note_id,
                "mention_comment_id": mention_comment_id,
                "mention_comment_content": mention_comment_content,
                "mentioner_id": mentioner_id,
                "mentioner_nickname": mentioner_nickname,
                "mention_type": mention_type,
                "mention_action": mention_action,
                "content_label": content_label,
                "relationship_source": relationship_source,
            })

        if not mention_entries:
            return []

        record_mentioned_note_ids_by_channel(
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

            与评论者/发帖者的关系（relationship_understanding 必须据此填写，请勿臆测或颠倒）：
            {entry["relationship_source"]}

            被评论或@的笔记信息：
            {json.dumps(entry["mention_note"], ensure_ascii=False, indent=2)}

            {entry["content_label"]}：
            {entry["mention_comment_content"]}

            {entry["content_label"]} ID：{entry["mention_comment_id"]}
            """)

        observation = (
            "【场景】你收到了通知（@/评论/回复）。可直接忽略，在值得接话时回复。接话时也要短，像聊天，不像写通报。\n\n"
            + "评论和@（共 {} 条提醒，请按顺序对每条分别给出决策）：\n\n".format(len(mention_entries))
            + "\n".join(observation_parts)
            + "\n可@的用户列表：\n"
            + mentionable_users_str
        )

        gates = await UserAgentGates(self).build_mention_coaching(
            mention_entries, getattr(event, "timestamp", None)
        )

        instruction = f"""下面有多条提醒，请按顺序对每条提醒分别给出一个决策（是否回复、回复内容等）。decisions 数组与提醒顺序一致，第 i 个元素对应第 i 条提醒。

        请基于 Profile、historical_summary、memory 与 Observation 中的关系信息，完成“是否回复/回复内容”判断与生成。

        步骤1：对每条推荐先设默认状态
        - "comment": false
        - "parent_comment_id": null
        - "comment_content": ""

        **comment 的数量决策仅由步骤2/3/4决定，不得因步骤6改变；**
        {gates.memory_coaching}

        步骤2：兴趣判断
        - 通读本批次所有推荐内容，判断哪些笔记能够进入候选池，当且仅当同时满足以下条件时，再将该笔记改为 comment=true：
            {gates.memory_rec}
            1. **对笔记感兴趣**：整帖与 Profile/historical_summary/memory 高度相关
            2. **对某条楼中评论感兴趣**：未必对整帖强兴趣，但**某条具体评论**与 Profile/historical_summary/memory 高度相关，且你愿意针对该评论做**增量回应**（补充、纠错、追问、共情、接梗）
            3. 互动对象关系与场景合适（关注关系优先）；
            4. 表达目的明确（补充事实、推进互动）
        - 进入候选池时，请在心中标记：本轮互动是「对帖」还是「对某条评论」；若选「对某条评论」，必须在后续 JSON 里使用非空的 parent_comment_id（见步骤4）。
        - 对不感兴趣的笔记：该笔记不进入候选池，comment=false。

        步骤3：选取 0～{gates.k_diff_targets} 目标笔记
        - {gates.k_diff_targets} 表示本轮**最多**对多少个**不同** note_id 置 comment=true；实际条数可为 0～{gates.k_diff_targets} 中任意值，**不得**为凑满条数而放宽步骤1.5/步骤2。
        - 仅在步骤2 的候选池内、按你的排序取前若干条作为“目标 note_id 集合”，且条数不超过 {gates.k_diff_targets}；若候选少于 {gates.k_diff_targets}，有多少算多少；其余笔记保持 comment=false。

        步骤4：根据人设，为每条评论选择 comment_mode，默认为 attitude
        - comment_mode ∈ {{attitude,discussion,ultra_short}}，并在 expression_reason 体现。
        - 发生概率：attitude > discussion > ultra_short
        - 共同特征：口语化，以第一人称视角表达，不要求语法正确
            - attitude（态度）：以态度、吐槽、捧场、玩梗为主，信息增量很少。常见极短句和纯表情、语气词
            - discussion（讨论）：提问、对比、转述事实、讨论机制或后果
            - ultra_short（超短互动）：仅有表情标签，无陈述，或仅有一两个字
        - 输出风格
            - 使用用户所在地区常用语言；默认使用中文，允许中英文混合。
            - 表达应符合 Profile 与 historical_summary 的人设口吻；
            - 减少模板化套话，如“你说得对/确实/很有道理/希望大家…”；
            - 评论内容不包含标签（#...）。

        步骤5：决定是否需要 @ 用户
        - 仅当对方在可@列表且与内容强相关 + 不是笔记作者/父评论作者时，才在正文中额外 @user_id；否则 mention_reasoning 为 []。

        步骤6：keep_following_note_ids
        - 仅当：高度感兴趣 + 当前内容有持续关注价值→ keep_following_note_ids=true；否则为空列表 []

        步骤7：search
        - 仅当：高度感兴趣 + 当前信息明显不足 + memory 无相关内容 → search=true；否则 false。

        【平台表情】
       [doge][害羞R][飞吻R][哭惹R][汗颜R][捂脸R][笑哭R][偷笑R][生气R][赞R][棒R][嘻嘻R][石化R][暗中观察R][微笑R][大笑R][合十R][害羞R][呃R][失望R]

        请按以下 JSON 返回（字段顺序固定）：
        {{
        "persona_understanding": "1-2句。概括身份、兴趣、语言风格（简短）",
        "content_understanding": "1-2句。概括本次被@/被评论内容与你的相关性、你优先关注的角度",
        "relationship_understanding": "严格依据 Observation 的关系判定，不自行扩展关系类型",
        "memory_reflection": "{gates.memory_ref}",
        "decisions": [
            {{
            "note_id": "note_id",
            "comment": false,
            "parent_comment_id": null,
            "comment_content": "",
            "decision_reason": "不互动的核心原因（1句，≤20字）",
            "expression_reason": "",
            "mention_reasoning": []
            }},
            {{
            "note_id": "note_id",
            "comment": true,
            "parent_comment_id": "comment_id",
            "comment_content": "口语短回复或轻互动",
            "decision_reason": "为什么这条值得回复（1句，≤20字）",
            "expression_reason": "为何使用这种语气与措辞（1句，若为 substantive/micro 请体现）",
            "mention_reasoning": [
                {{
                "user_id": "被@用户id",
                "persona_understanding": "对该用户的简短理解",
                "mention_reason": "为何需要额外提醒该用户（1句）"
                }}
            ]
            }},
            {{
            "note_id": "note_id",
            "comment": true,
            "parent_comment_id": null,
            "comment_content": "口语短评论或轻互动",
            "decision_reason": "为什么这条值得评论（1句，≤20字）",
            "expression_reason": "为何使用这种语气与措辞（1句，若为 substantive/micro 请体现）",
            "mention_reasoning": []
            }}
        ],
            "keep_following_note_ids": [],
            "keep_following_reason": "为何保持关注（1句，<20字）",
        "search": false,
        "search_keyword": "搜索关键词（1句，≤20字）",
        "search_reason": "是否搜索及原因（1句，≤20字）"
        }}

        输出规则：
        - 先给出理解字段，再给 decisions；
        - decisions 中每条先按默认状态填写 comment=false，再仅对满足条件的条目改为 comment=true；
        - comment=false 时，comment_content 必须为空字符串。
        - comment_content 建议根据 comment_mode 选择合适的表达方式，尽量符合人设口吻，减少模板化套话。
        {gates.similarity_kw_coaching}{gates.similarity_emb_coaching}
        {gates.freshness_coaching}"""

        mention_note_id_set: Set[str] = {
            str(e["note_id"]).strip()
            for e in mention_entries
            if isinstance(e, dict) and e.get("note_id")
        }

        # Call LLM to generate the decision
        try:
            response = await self.generate_reaction(instruction, observation)
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} handle_mention")
        except Exception as e:
            logger.error(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} handle_mention ")

        response = self._normalize_llm_reaction(response)
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
            keep_ids = response.get("keep_following_note_ids", [])
            if isinstance(keep_ids, list) and keep_ids:
                # Only allow note_ids within the current batch, and at most 1
                valid_keep_ids = []
                for keep_note_id in keep_ids:
                    if str(keep_note_id).strip() in mention_note_id_set:
                        valid_keep_ids.append(keep_note_id)
                if valid_keep_ids:
                    self.profile.update_data("keep_following_note_ids", valid_keep_ids[:1])
                else:
                    self.profile.update_data("keep_following_note_ids", [])

        # Handle the comment decision
        decisions = response.get("decisions", [])
        if not isinstance(decisions, list):
            return events_to_send

        has_reply = False
        user_id = await self.get_data("id")
        nickname = await self.get_data("nickname", "")
        ip_location = await self.get_data("ip_location", "")
        current_timestamp = event.timestamp
        mention_window_start_ms = (
            int(current_timestamp) if isinstance(current_timestamp, (int, float)) else 0
        )
        mention_window_duration_ms = int(getattr(event, "timestamp_duration", 0) or 0)

        for i, decision in enumerate(decisions):
            if i >= len(mention_entries):
                break
            mention_entry = mention_entries[i]
            mention_note = mention_entry["mention_note"]
            mention_note_id = mention_entry["note_id"]
            mention_comment_id = mention_entry["mention_comment_id"]
            try:
                if not isinstance(decision, dict) or not decision.get("comment", False):
                    continue

                note_id = decision.get("note_id")
                should_comment = decision.get("comment", False)
                parent_comment_id = decision.get("parent_comment_id")
                comment_content = decision.get("comment_content", "")
                if not note_id or not should_comment:
                    continue

                has_reply = True

                # Check if the note_id is valid (should be the same as mention_note_id)
                if note_id != mention_note_id:
                    logger.warning(f"Note {note_id} does not match mention note {mention_note_id}, skipping")
                    continue

                # Parse the @users in the comment content, replace @id with @nickname, and return the user ID list
                mentioned_user_ids = []
                mention_reasoning = decision.get("mention_reasoning", [])
                if isinstance(mention_reasoning, list):
                    for mention_reason in mention_reasoning:
                        if isinstance(mention_reason, dict):
                            mentioned_uid = mention_reason.get("user_id")
                            if mentioned_uid:
                                mentioned_user_ids.append(mentioned_uid)


                note_author_id = mention_note.get("user_id")
                mentioned_user_ids = [
                    uid for uid in mentioned_user_ids if uid != note_author_id
                ]
                comment_author_id = None
                canonical_parent_id: Optional[str] = None
                if parent_comment_id is not None:
                    comments = mention_note.get("comments", {})
                    if not isinstance(comments, dict):
                        comments = {}
                    canonical_parent_id, parent_entry = resolve_parent_comment_entry(
                        comments, parent_comment_id
                    )
                    if isinstance(parent_entry, dict):
                        comment_author_id = parent_entry.get("user_id")
                    mentioned_user_ids = [
                        uid for uid in mentioned_user_ids if uid != comment_author_id
                    ]
                mention_count = len(mentioned_user_ids)

                # If it is a reply comment, check if the parent comment ID is valid
                if parent_comment_id is not None:
                    if canonical_parent_id is None:
                        logger.warning(
                            f"Parent comment {parent_comment_id} not found on note {note_id}, skipping reply"
                        )
                        continue

                    # Generate a unique comment ID
                    comment_id = generate_comment_id()
                    success = await self.add_env_comments(note_id, {
                        "comment_id": comment_id,
                        "timestamp": generate_comment_timestamp(
                            mention_note, mention_window_start_ms, mention_window_duration_ms
                        ),
                        "ip_location": ip_location,
                        "note_id": note_id,
                        "user_id": user_id,
                        "nickname": nickname,
                        "parent_comment_id": canonical_parent_id,
                        "at_count": mention_count,
                        "content": comment_content
                    })
                    if not success:
                        logger.error(f"Failed to add comment to note {note_id}")
                        continue
                    
                    # Send reminder to the comment author
                    if comment_author_id and comment_author_id != user_id:  
                        success = await self.update_env_mention_pool(f"{comment_author_id}.{comment_id}", {
                            "action": "add",
                            "mention_message": {
                                "note_id": note_id,
                                "comment_id": comment_id,
                                "comment_content": comment_content,
                                "mentioner_id": user_id,
                                "mentioner_nickname": nickname,
                                "mention_type": "reply"
                            }
                        })
                        if not success:
                            logger.error(f"Failed to update mention pool for comment {comment_id} by {user_id} on note {note_id}")
                            continue
                        logger.info(f"User {user_id} replied to comment {canonical_parent_id} by {comment_author_id} on note {note_id}")
                    
                    # Send reminder to the note author
                    if note_author_id and note_author_id != user_id and note_author_id != comment_author_id:  # 不给自己发提醒，也不给评论作者发提醒
                        success = await self.update_env_mention_pool(f"{note_author_id}.{comment_id}", {
                            "action": "add",
                            "mention_message": {
                                "note_id": note_id,
                                "comment_id": comment_id,
                                "comment_content": comment_content,
                                "mentioner_id": user_id,
                                "mentioner_nickname": nickname,
                                "mention_type": "comment"
                            }
                        })
                        if not success:
                            logger.error(f"Failed to update mention pool for comment {comment_id} by {user_id} on note {note_id}")
                            continue
                        logger.info(f"User {user_id} commented on note {note_id} by {note_author_id}")
                elif parent_comment_id is None:
                    # If it is a comment note, add a comment
                    comment_id = generate_comment_id()
                    success = await self.add_env_comments(note_id, {
                        "comment_id": comment_id,
                        "timestamp": generate_comment_timestamp(
                            mention_note, mention_window_start_ms, mention_window_duration_ms
                        ),
                        "ip_location": ip_location,
                        "note_id": note_id,
                        "user_id": user_id,
                        "nickname": nickname,
                        "parent_comment_id": parent_comment_id,
                        "at_count": mention_count,
                        "content": comment_content
                    })
                    if not success:
                        logger.error(f"Failed to add comment to note {note_id}")
                        continue

                    # Send reminder to the note author
                    if note_author_id and note_author_id != user_id:  
                        success = await self.update_env_mention_pool(f"{note_author_id}.{comment_id}", {
                            "action": "add",
                            "mention_message": {
                                "note_id": note_id,
                                "comment_id": comment_id,
                                "comment_content": comment_content,
                                "mentioner_id": user_id,
                                "mentioner_nickname": nickname,
                                "mention_type": "comment"
                            }
                        })
                        if not success:
                            logger.error(f"Failed to update mention pool for comment {comment_id} by {user_id} on note {note_id}")
                            continue
                        logger.info(f"Step {current_step}/{max_step}: User {user_id} commented on note {note_id} by {note_author_id}")

                # Send MentionEvent to the users that are mentioned
                if mentioned_user_ids:
                    for mentioned_user_id in mentioned_user_ids:
                        if mentioned_user_id and mentioned_user_id != user_id:  
                            success = await self.update_env_mention_pool(f"{mentioned_user_id}.{comment_id}", {
                                "action": "add",
                                "mention_message": {
                                    "note_id": note_id,
                                    "comment_id": comment_id,
                                    "comment_content": comment_content,
                                    "mentioner_id": user_id,
                                    "mentioner_nickname": nickname,
                                    "mention_type": "at"
                                }
                            })
                            if not success:
                                logger.error(f"Failed to update mention pool for comment {comment_id} by {user_id} on note {note_id}")
                                continue
                            logger.info(f"User {user_id} mentioned {mentioned_user_id} in comment on note {note_id}")
            finally:
                # Delete the mention from the mention pool regardless of whether it is a reply or not
                success = await self.update_env_mention_pool(f"{user_id}.{mention_comment_id}", {
                    "action": "delete",
                    "mention_message": None
                })
                if not success:
                    logger.error(f"Failed to update mention pool for comment {mention_comment_id}")
                else:
                    logger.info(f"Step {current_step}/{max_step}: User {user_id} deleted mention {mention_comment_id} from pool (processed)")

        # If there is a reply, notify the environment update the content pool via event
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
    