from onesim.utils.comment_view import snapshot_threads, select_for_env
from .mention_address import resolve_address
from onesim.simulator import BasicSimEnv
from onesim.events import Event
from onesim.events import DataUpdateEvent, DataUpdateResponseEvent
from onesim.distribution.distributed_lock import get_lock 
from collections import Counter
from typing import Any, Dict, Optional, Tuple
from loguru import logger
import json
import math
import os
import random
from copy import deepcopy
from .events import StartEvent, AddCommentEvent, AddCommentResponseEvent, MentionPoolUpdateEvent, MentionPoolUpdateResponseEvent, PLATFORM_AGENT_ID
from .metrics.channel_snapshots import save_channel_snapshots, save_content_pool_snapshot
from .utils.time import normalize_content_pool_times, time_to_ms
from onesim.utils.social_data_compat import add_runtime_post_aliases

MS_PER_DAY = 86_400_000


class SimEnv(BasicSimEnv):
    comment_view_field = 'comments'
    comment_view_linked = False

    def _comment_view_embed(self, texts):
        from .embedding import EmbeddingService
        params = self.data.get("midsim_params") or {}
        config = (params.get("user") or {}).get("memory_similarity") or {}
        return EmbeddingService(config.get("embedding_config_path") or None).embed(texts)


    def _load_profile_history(self):
        """Keep historical context out of recommendation feeds until actually revisited."""
        self._profile_history_posts = {}
        for actor in (self.agents.get("UserAgent") or {}).values():
            uid = str(actor.profile.get_data("id"))
            posts = actor.profile.get_data("historical_posts", {}) or {}
            rows = posts.items() if isinstance(posts, dict) else enumerate(posts)
            for key, row in rows:
                if not isinstance(row, dict):
                    continue
                pid = str(row.get("post_id") or row.get("id") or key)
                post = deepcopy(row)
                post.setdefault("user_id", uid)
                self._merge_history_post(pid, post)
        for actor in (self.agents.get("UserAgent") or {}).values():
            uid = str(actor.profile.get_data("id"))
            comments = actor.profile.get_data("historical_comments", {}) or {}
            rows = comments.items() if isinstance(comments, dict) else enumerate(comments)
            for key, row in rows:
                if not isinstance(row, dict):
                    continue
                cid = str(row.get("comment_id") or row.get("id") or key)
                root = row.get("post") or {}
                pid = str(row.get("post_id") or root.get("post_id") or root.get("id") or "")
                if not pid:
                    continue
                if root:
                    self._merge_history_post(pid, root)
                if pid not in self._profile_history_posts:
                    live = self.data.get("content_pool", {}).get(pid)
                    if not live:
                        logger.warning("X historical comment {} lacks root post {}; unavailable for visits", cid, pid)
                        continue
                    self._merge_history_post(pid, live)
                target = self._profile_history_posts[pid].setdefault("comments", {})
                ancestors = row.get("ancestor_comments") or {}
                ancestors = ancestors.items() if isinstance(ancestors, dict) else enumerate(ancestors)
                for ancestor_key, ancestor in ancestors:
                    if isinstance(ancestor, dict):
                        aid = str(ancestor.get("comment_id") or ancestor.get("id") or ancestor_key)
                        target.setdefault(aid, deepcopy(ancestor))
                comment = {k: deepcopy(v) for k, v in row.items() if k not in ("post", "ancestor_comments")}
                comment.setdefault("user_id", uid)
                comment.update(comment_id=cid, post_id=pid)
                target.setdefault(cid, comment)

    def _merge_history_post(self, pid, row):
        post = self._profile_history_posts.setdefault(pid, deepcopy(row))
        post.update(post_id=pid)
        post.setdefault("comments", {}).update(deepcopy(row.get("comments") or {}))

    def _profile_history_comment_limit(self):
        params = self.data.get("midsim_params") or {}
        exposure = params.get("exposure") or {}
        settings = exposure.get("profile_visit") or {}
        limit = settings.get("history_comment_limit", 2)
        if type(limit) is not int or limit < 0:
            raise ValueError("exposure.profile_visit.history_comment_limit must be a non-negative integer")
        return limit

    async def snapshot_profile_history(self, uid, as_of_ms, viewer_id, follow_ids):
        from .profile_browsing import history_snapshot, account_id
        limit = self._profile_history_comment_limit()
        async with self._lock:
            pool = deepcopy(getattr(self, "_profile_history_posts", {}))
            for pid, live in self.data.get("content_pool", {}).items():
                comments = {**pool.get(pid, {}).get("comments", {}), **live.get("comments", {})}
                pool[pid] = {**deepcopy(live), "comments": deepcopy(comments)}
            # Public replies only, or protected accounts explicitly visible to this follower.
            inaccessible = set()
            for actor in (self.agents.get("UserAgent") or {}).values():
                author = str(actor.profile.get_data("id"))
                if actor.profile.get_data("protected", False) and author != viewer_id and author not in follow_ids:
                    inaccessible.add(author)
            if uid in inaccessible:
                return {"user_id": uid, "as_of_ms": as_of_ms, "items": [], "status": "protected"}
            for pid in list(pool):
                post = pool[pid]
                author = account_id(post)
                if author in inaccessible or (post.get("protected") and author != viewer_id and author not in follow_ids):
                    del pool[pid]
                    continue
                post["comments"] = {cid: c for cid, c in post.get("comments", {}).items()
                                    if account_id(c) not in inaccessible}
            return history_snapshot(pool, uid, as_of_ms, limit=limit)

    async def retain_profile_posts(self, post_ids):
        """Make an explicitly bookmarked historical root available next round."""
        async with self._lock:
            pool = self.data.setdefault("content_pool", {})
            for pid in post_ids:
                if pid not in pool and pid in self._profile_history_posts:
                    pool[pid] = deepcopy(self._profile_history_posts[pid])

    async def get_committed_comment(self, post_id, comment_id):
        """Resolve an exact parent from authoritative state without an event round trip."""
        async with self._lock:
            comment = self.data.get("content_pool", {}).get(post_id, {}).get("comments", {}).get(str(comment_id))
            return deepcopy(comment) if comment is not None else None

    async def snapshot_visible_comments(self, posts, as_of_ms, *, limit_comments=True, anchors=None):
        return await snapshot_threads(self, posts, as_of_ms, limit_comments=limit_comments, anchors=anchors)


    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.register_event("AddCommentEvent", "handle_add_comment_event")
        self.register_event("MentionPoolUpdateEvent", "handle_update_mention_pool_event")
        self._prefer_platform_start_order()

    def _prefer_platform_start_order(self) -> None:
        """Dispatch Platform StartEvent before UserAgent exposure in this round."""
        targets = self.start_targets
        if not isinstance(targets, dict) or "Platform" not in targets:
            return
        self.start_targets = {
            "Platform": targets["Platform"],
            **{k: v for k, v in targets.items() if k != "Platform"},
        }

    def _platform_service_agent_ids(self) -> Tuple[str, str]:
        rec_id, search_id = "", ""
        for agent_id, agent in ((self.agents or {}).get("Algorithm") or {}).items():
            profile = getattr(agent, "profile", None)
            role = ""
            if profile is not None and hasattr(profile, "get_data"):
                role = str(profile.get_data("role", "") or "").strip().lower()
            if role == "recommendation":
                rec_id = str(agent_id)
            elif role == "search":
                search_id = str(agent_id)
        return rec_id, search_id

    async def load_initial_data(self) -> None:
        """Complete the derived fields, including current_posts."""
        await super().load_initial_data()
        self._prefer_platform_start_order()
        self._profile_history_comment_limit()

        async with self._lock:
            # Initialize user metadata
            self._collect_user_agent_ids()
            
            # Initialize comment statistics
            content_pool = self.data.get("content_pool", {})
            if isinstance(content_pool, dict):
                add_runtime_post_aliases(content_pool)
                n_time = normalize_content_pool_times(content_pool)
                if n_time:
                    logger.info(
                        f"load_initial_data: normalized post time fields to milliseconds on {n_time} posts"
                    )
            self._recalculate_comment_counts()
            # The original X seed data uses user_uid/user_sname; aliases stay in memory.
            for post in content_pool.values():
                post.setdefault("user_id", str(post.get("user_uid") or ""))
                post.setdefault("username", post.get("user_sname", ""))
            self._load_profile_history()

            # Initialize simulation time
            current_ts = self.data.get("current_timestamp", 1764255440000)
            sched = self._simulator_schedule_settings()

            if not self.data.get("simulation_start_timestamp"):
                self.data["simulation_start_timestamp"] = current_ts

            td = self.data.get("timestamp_duration")
            if td is None or td == 0:
                dur_days = self._calculate_step_duration(1, sched)
                self.data["timestamp_duration"] = int(dur_days * MS_PER_DAY)

            start_ts = int(self.data["simulation_start_timestamp"])
            dur_ms = int(self.data["timestamp_duration"])
            cap_ts = int(start_ts + sched["max_span_days"] * MS_PER_DAY)
            window_end = min(current_ts + dur_ms, cap_ts)

            # Initialize current posts
            self.data["current_posts"] = self._build_current_posts_subset(current_ts, window_end)
            if not isinstance(self.data.get("mention_pool"), dict):
                self.data["mention_pool"] = {}

            logger.info(
                f"load_initial_data: current_timestamp={current_ts}, "
                f"timestamp_duration={dur_ms}, "
                f"window_end={window_end}, "
                f"current_posts={len(self.data['current_posts'])}"
            )

    def _collect_user_agent_ids(self) -> None:
        if "user_agent_ids" in self.data:
            return
        self.data["user_agent_ids"] = sorted(
            self.agents.get("UserAgent", {}).keys()
        )

    def _recalculate_comment_counts(self) -> None:
        """Recalculate comment and reply counts for each post."""
        content_pool = self.data.get("content_pool", {})

        for post in content_pool.values():
            comments = post.get("comments", {})
            post["comment_count"] = len(comments)

            sub_comment_counts = Counter(
                comment.get("parent_comment_id")
                for comment in comments.values()
                if comment.get("parent_comment_id")
            )

            for comment_id, comment in comments.items():
                comment["sub_comment_count"] = sub_comment_counts[str(comment_id)]

    @staticmethod
    def _calculate_step_duration(step_num: int, sched: Dict[str, Any]) -> float:
        """Return days per step: uniform, or the original power/sigmoid schedule."""
        max_step = sched["max_step"]
        max_span_days = sched["max_span_days"]
        if max_step == 1:
            return max_span_days

        if sched["timestamp_schedule_type"] == "uniform":
            return max_span_days / max_step

        if sched["timestamp_schedule_type"] == "sigmoid":
            center = 1 + (max_step - 1) * sched["timestamp_sigmoid_center_ratio"]
            scale = sched["timestamp_sigmoid_scale"]
            weights = [
                0.05 + 1 / (1 + math.exp(-(i - center) / scale))
                for i in range(1, max_step + 1)
            ]
        else:
            power_p = sched["timestamp_power_p"]
            weights = [i ** power_p for i in range(1, max_step + 1)]

        step_idx = min(max(step_num, 1), max_step) - 1
        return max_span_days * weights[step_idx] / sum(weights)

    def _post_copy_limited_comments(self, post: dict, top_k: int = 5, random_k: int = 0) -> dict:
        return select_for_env(self, post, root_limit=top_k, field='comments')


    @staticmethod
    def _is_post_time_before_hi(post: Dict[str, Any], _lo: float, hi: float) -> bool:
        """The post time is strictly before hi."""
        raw_time = post.get("time")
        if raw_time is None or raw_time == "":
            raw_time = post.get("create_time", post.get("timestamp"))
        post_time = time_to_ms(raw_time)
        if post_time is None:
            return False
        return post_time < hi

    def _build_current_posts_subset(
        self, lo_ts: float, hi_ts: float
    ) -> Dict[str, Any]:
        """Build the current_posts subset."""
        content_pool = self.data.get("content_pool", {})
        if hi_ts <= lo_ts:
            return {}
        return {
            nid: deepcopy(post)
            for nid, post in content_pool.items()
            if isinstance(post, dict) and self._is_post_time_before_hi(post, lo_ts, hi_ts)
        }

    @staticmethod
    def _parse_memory_timestamp_ms(raw: Any) -> Optional[int]:
        """Parse the post ``time`` or comment ``timestamp`` to milliseconds."""
        ms = time_to_ms(raw)
        if ms is None:
            return None
        return int(round(ms))

    def _compute_window_popularity(self) -> None:
        content_pool = self.data["content_pool"]
        current_ts = self.data["current_timestamp"]
        window_days = self.data.get("popularity_window_days", 7.0)
        lo = current_ts - int((window_days if window_days > 0 else 7.0) * MS_PER_DAY)
        for post in content_pool.values():
            comments = post.get("comments", {})
            post["popularity"] = sum(
                1
                for comment in comments.values()
                if (ts := self._parse_memory_timestamp_ms(comment.get("timestamp")))
                and lo <= ts <= current_ts
            ) if current_ts > 0 else 0

    async def _save_step_data(self, step_num: int):
        """Update the environment data after each step."""
        async with self._lock:
            save_channel_snapshots(self, step_num)
        await super()._save_step_data(step_num)

        async with self._lock:
            # Update and save content state
            content_pool = self.data.get("content_pool", {})
            self._recalculate_comment_counts()
            self._compute_window_popularity()

            save_content_pool_snapshot(self, step_num, content_pool)

            # Advance simulation time
            current_ts = int(self.data["current_timestamp"])
            sched = self._simulator_schedule_settings()
            step_delta_ms = int(self._calculate_step_duration(step_num, sched) * MS_PER_DAY)
            start_ts = int(self.data["simulation_start_timestamp"])
            cap_ts = int(start_ts + sched["max_span_days"] * MS_PER_DAY)
            next_ts = min(int(current_ts) + step_delta_ms, cap_ts)
            next_step = step_num + 1
            if next_step <= sched["max_step"]:
                next_duration = int(self._calculate_step_duration(next_step, sched) * MS_PER_DAY)
            else:
                next_duration = 0

            self.data["current_timestamp"] = next_ts
            self.data["current_step"] = next_step
            self.data["timestamp_duration"] = next_duration

            # Build current posts for the next step
            window_end = min(next_ts + next_duration, cap_ts)
            self.data["current_posts"] = self._build_current_posts_subset(next_ts, window_end)

            logger.info(
                f"Step {step_num}: "
                f"{current_ts} -> {next_ts}, "
                f"next_duration={next_duration}, "
                f"current_posts={len(self.data['current_posts'])}"
            )

    async def _create_start_event(self, target_id: str) -> Event:
        """Create a StartEvent for the given target_id."""
        source_id = self.data.get('source_agent_id', 'default_source')
        timestamp = self.data.get('current_timestamp', 0)
        timestamp_duration = self.data.get('timestamp_duration', MS_PER_DAY)
        current_step = self.data.get("current_step", 1)
        current_posts = self.data.get('current_posts', {})
        
        sched = self._simulator_schedule_settings()
        max_step = sched["max_step"]

        start_ts = self.data.get("simulation_start_timestamp")
        simulation_cap_timestamp = int(start_ts + sched["max_span_days"] * MS_PER_DAY)

        logger.info(
            f"Step {current_step}/{max_step}: "
            f"timestamp={timestamp}, "
            f"timestamp_duration={timestamp_duration}"
        )
        
        mentions = {}
        rec_id, search_id = "", ""

        # UserAgent-specific initialization
        if not str(target_id).startswith("platform_agent_") and str(target_id) != PLATFORM_AGENT_ID:
            mention_pool = self.data.get("mention_pool", {}).get(target_id, {})
            for comment_id, mention_message in mention_pool.items():
                post_id = mention_message.get("post_id", "")
                if post_id in current_posts:
                    mentions[comment_id] = {
                        "post": current_posts[post_id],
                        "comment_id": comment_id,
                        "comment_content": mention_message.get("comment_content", ""),
                        "mentioner_id": mention_message.get("mentioner_id", ""),
                        "mentioner_nickname": mention_message.get("mentioner_nickname", ""),
                        "mention_type": mention_message.get("mention_type", "")
                    }
            logger.info(f"StartEvent: length of mentions for UserAgent {target_id}: {len(mentions)}")
        elif str(target_id) == PLATFORM_AGENT_ID:
            rec_id, search_id = self._platform_service_agent_ids()
        
        return StartEvent(
            from_agent_id=source_id,
            to_agent_id=target_id,
            timestamp=timestamp,
            timestamp_duration=timestamp_duration,
            current_step=current_step,
            max_step=max_step,
            current_posts=current_posts,
            mentions=mentions,
            simulation_cap_timestamp=simulation_cap_timestamp,
            recommendation_agent_id=rec_id,
            search_agent_id=search_id,
        )

    async def queue_event(self, event_data: Dict[str, Any]):
        """Add the event to the queue, and save and broadcast it when the step ends"""
        if event_data['event_type'] in [
            'AddCommentEvent',
            'AddCommentResponseEvent',
            'MentionPoolUpdateEvent',
            'MentionPoolUpdateResponseEvent',
        ]:
            return
        await super().queue_event(event_data)

    async def add_comment(self, key: str, data: Any) -> Any:
        """Update the shared data"""
        async with self._lock:
            parts = key.split(".")  

            if len(parts) != 3 or parts[0] != "content_pool" or parts[2] != "comments":
                raise ValueError(f"Invalid key: {key}, expected format: content_pool.post_id.comments")
            
            post_id = parts[1]  
            content_pool = self.data.get("content_pool", {})

            if post_id not in content_pool:
                historical = getattr(self, "_profile_history_posts", {}).get(post_id)
                if historical is None:
                    raise ValueError(f"post_id {post_id} not found in content_pool or profile history")
                content_pool[post_id] = deepcopy(historical)
            elif post_id in getattr(self, "_profile_history_posts", {}):
                # A reply may target an older ancestor absent from the seed snapshot.
                for cid, row in self._profile_history_posts[post_id].get("comments", {}).items():
                    content_pool[post_id].setdefault("comments", {}).setdefault(cid, deepcopy(row))
                
            comment_id = data["comment_id"]
            comments = content_pool[post_id].setdefault("comments", {})

            comments[comment_id] = {
                **comments.get(comment_id, {}),
                **data,
            }
            # self.data["content_pool"] = current_pool
                
            logger.debug(f"Added comment {comment_id} to post {post_id}")
            return True

    async def handle_add_comment_event(self, event: AddCommentEvent) -> None:
        """
        Handle the add comment event from the agent (using a distributed lock)
        Example:
            {
                "comment_id": "123",
                "create_time": 1764298329000,
                "post_id": "69290e59000000001e034ab4",
                "user_id": "5e2a573f0000000001002ddf",
                "nickname": "XiaoHongShu A",
                "parent_comment_id": None,
                "at_count": 0,
                "content": "Comment 1"
            }
        """
        try:
            logger.info(f"Received AddCommentEvent, request_id={event.request_id}, key={event.key}")
            lock_id = f"env_comment_add_lock_{event.key}"
            lock = await get_lock(lock_id)

            # Get the lock before updating the data
            async with lock:
                success = await self.add_comment(event.key, event.value)

                # Create and send the response event
                response_event = AddCommentResponseEvent(
                    from_agent_id=self.name,
                    to_agent_id=event.from_agent_id,
                    request_id=event.request_id,
                    key=event.key,
                    success=success
                )
                await self.event_bus.dispatch_event(response_event)

        except Exception as e:
            error_response = AddCommentResponseEvent(
                from_agent_id=self.name,
                to_agent_id=event.from_agent_id,
                request_id=event.request_id,
                key=event.key,
                success=False,
                error=str(e)
            )
            await self.event_bus.dispatch_event(error_response)

    async def update_mention_pool(self, key: str, data: Any, *, target_user_id=None, comment_id=None) -> Any:
        """Update the shared data (asynchronously, using a lock)"""
        async with self._lock:
            target_user_id, comment_id = resolve_address(key, data, target_user_id, comment_id)
            mention_pool = self.data.get("mention_pool")
            if not isinstance(mention_pool, dict):
                mention_pool = {}
            user_mentions = mention_pool.setdefault(target_user_id, {})
            action = data["action"]

            # Bulk keep: one round-trip instead of N serial deletes when trimming over budget.
            if comment_id is None:
                if action == "keep_only":
                    keep_ids = {
                        str(x).strip()
                        for x in (data.get("keep_comment_ids") or [])
                        if str(x).strip()
                    }
                    before = len(user_mentions)
                    mention_pool[target_user_id] = {
                        cid: msg
                        for cid, msg in user_mentions.items()
                        if cid in keep_ids
                    }
                    removed = before - len(mention_pool[target_user_id])
                    self.data["mention_pool"] = mention_pool
                    logger.info(
                        f"Kept {len(keep_ids)} mentions for user {target_user_id} "
                        f"(removed {removed})"
                    )
                    return True
                if action == "delete_many":
                    delete_ids = {
                        str(x).strip()
                        for x in (data.get("comment_ids") or [])
                        if str(x).strip()
                    }
                    removed = 0
                    for cid in delete_ids:
                        if cid in user_mentions:
                            del user_mentions[cid]
                            removed += 1
                    self.data["mention_pool"] = mention_pool
                    logger.info(
                        f"Deleted {removed}/{len(delete_ids)} mentions for user {target_user_id}"
                    )
                    return True
                raise ValueError(
                    f"Invalid bulk action: {action}, expected keep_only or delete_many"
                )

            # Add operation
            if action == "add":
                # A retried notification overwrites the same entry, never a comment.
                user_mentions[comment_id] = data["mention_message"]
                logger.info(f"Added mention {comment_id} for user {target_user_id}")

            # Delete operation
            elif action == "delete":
                if comment_id in user_mentions:
                    del user_mentions[comment_id]
                    logger.info(f"Deleted mention {comment_id} for user {target_user_id}")

            self.data["mention_pool"] = mention_pool
            return True
            
    async def handle_update_mention_pool_event(self, event: MentionPoolUpdateEvent) -> None:
        """
        Handle the update mention_pool event from the agent (using a distributed lock)
        Example:
            {
                "action": "add",
                "mention_message": {
                    "post_id": "69290e59000000001e034ab4",
                    "comment_id": 1764298329000,
                    "comment_content": "Comment 1",
                    "mentioner_id": "69290e59000000001e034ab4",
                    "mentioner_nickname": "XiaoHongShu A",
                    "mention_type": "reply"
                }
            }
        """
        try:
            logger.info(f"Received MentionPoolUpdateEvent, request_id={event.request_id}, key={event.key}")
            lock_id = f"env_mention_pool_update_lock_{event.key}"
            lock = await get_lock(lock_id)

            # Get the lock before updating the data
            async with lock:
                # Retain receipts for this environment's lifetime. A repeated
                # add must not resurrect a notification already consumed/deleted.
                if not hasattr(self, "_mention_operation_receipts"):
                    self._mention_operation_receipts = {}
                operation = (str(event.from_agent_id), str(event.request_id))
                receipts = self._mention_operation_receipts
                if operation in receipts:
                    success = receipts[operation]
                    logger.info("Mention operation already applied; returning receipt: {}", event.request_id)
                else:
                    success = await self.update_mention_pool(
                        event.key, event.value,
                        target_user_id=getattr(event, "target_user_id", None),
                        comment_id=getattr(event, "comment_id", None))
                    receipts[operation] = success

                # Create and send the response event
                response_event = MentionPoolUpdateResponseEvent(
                    from_agent_id=self.name,
                    to_agent_id=event.from_agent_id,
                    request_id=event.request_id,
                    key=event.key,
                    success=success
                )
                await self.event_bus.dispatch_event(response_event)

        except Exception as e:
            error_response = MentionPoolUpdateResponseEvent(
                from_agent_id=self.name,
                to_agent_id=event.from_agent_id,
                request_id=event.request_id,
                key=event.key,
                success=False,
                error=str(e)
            )
            await self.event_bus.dispatch_event(error_response)
