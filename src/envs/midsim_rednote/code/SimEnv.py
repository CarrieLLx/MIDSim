from onesim.simulator import BasicSimEnv
from onesim.events import Event
from onesim.events import DataUpdateEvent, DataUpdateResponseEvent
from onesim.distribution.distributed_lock import get_lock 
from typing import Any, Optional, Dict
from loguru import logger
import json
import math
import os
import random
from .events import StartEvent, AddCommentEvent, AddCommentResponseEvent, MentionPoolUpdateEvent, MentionPoolUpdateResponseEvent
from .metrics.channel_snapshots import save_channel_snapshots, save_content_pool_snapshot

class SimEnv(BasicSimEnv):
    def __init__(
        self,
        name: str,
        event_bus,
        data: Optional[Dict[str, Any]] = None,
        start_targets: Optional[Dict[str, Any]] = None,
        end_targets: Optional[Dict[str, Any]] = None,
        config: Optional[Any] = None,
        agents: Optional[Dict[str, Any]] = None,
        env_path: Optional[str] = None,
        trail_id: Optional[str] = None,
        output_dir: Optional[str] = None,
        **kwargs
    ) -> None:
        """Initialize SimEnv"""
        super().__init__(
            name=name,
            event_bus=event_bus,
            data=data,
            start_targets=start_targets,
            end_targets=end_targets,
            config=config,
            agents=agents,
            env_path=env_path,
            trail_id=trail_id,
            output_dir=output_dir,
            **kwargs
        )
        self.register_event("AddCommentEvent", "handle_add_comment_event")
        self.register_event("MentionPoolUpdateEvent", "handle_update_mention_pool_event")

    async def load_initial_data(self) -> None:
        """Complete the derived fields, including current_notes."""
        await super().load_initial_data()
        async with self._lock:
            self._populate_registered_user_agent_ids_for_metrics()
            
            # Update the comment count and sub-comment count in content_pool
            content_pool = self.data.get("content_pool", {})
            if not isinstance(content_pool, dict):
                logger.warning("load_initial_data: content_pool is not a dict, skip current_notes bootstrap")
                return

            self._normalize_content_pool_comment_meta(content_pool)

            # Update current_timestamp and timestamp_duration for the first step
            ts = self.data.get("current_timestamp", 1764255440000)
            if not isinstance(ts, (int, float)) or int(ts) <= 0:
                logger.warning(f"load_initial_data: invalid current_timestamp {ts}, set current_notes empty")
                self.data["current_notes"] = {}
                return

            sched = self._simulator_schedule_settings()
            day_ms = 86400000   # 1 day = 86400 seconds * 1000 milliseconds

            if not isinstance(self.data.get("simulation_start_timestamp"), (int, float)) or int(
                self.data.get("simulation_start_timestamp") or 0
            ) <= 0:
                self.data["simulation_start_timestamp"] = int(ts)

            td = self.data.get("timestamp_duration")
            if td is None or td == 0:
                # Only fill in the first segment: day -> milliseconds, for the first StartEvent
                dur_days = self._timestamp_duration_days_for_step(
                    1,
                    max_step=sched["max_step"],
                    max_span_days=sched["max_span_days"],
                    schedule_type=sched["timestamp_schedule_type"],
                    power_p=sched["timestamp_power_p"],
                    sigmoid_scale=sched["timestamp_sigmoid_scale"],
                    sigmoid_center_ratio=sched["timestamp_sigmoid_center_ratio"],
                )
                self.data["timestamp_duration"] = int(dur_days * day_ms)

            start_ts = int(self.data["simulation_start_timestamp"])
            cap_ts = int(start_ts + sched["max_span_days"] * day_ms)
            lo = float(self.data["current_timestamp"])
            dur_ms = int(self.data.get("timestamp_duration") or 0)
            hi = min(lo + dur_ms, float(cap_ts))

            logger.info(
                f"load_initial_data: current_timestamp={lo}, timestamp_duration(next window)={dur_ms}, "
                f"current_notes: time < {hi} ms (min(lo+duration, cap))"
            )

            # Build current_notes subset
            self.data["current_notes"] = self._build_current_notes_subset(content_pool, lo, hi)
            logger.info(f"load_initial_data: built current_notes count={len(self.data['current_notes'])}")

    def _populate_registered_user_agent_ids_for_metrics(self) -> None:
        """Count the number of UserAgents."""
        if self.data.get("registered_user_agent_ids"):
            return
        agents = getattr(self, "agents", None)
        if not isinstance(agents, dict):
            return
        ua = agents.get("UserAgent")
        if not isinstance(ua, dict):
            return
        ids = sorted({str(aid) for aid in ua.keys() if str(aid).strip()})
        if ids:
            self.data["registered_user_agent_ids"] = ids

    def _normalize_content_pool_comment_meta(self, content_pool: Dict[str, Any]) -> None:
        """Update the comment_count and sub_comment_count of each note in content_pool."""
        for _note_id, note in content_pool.items():
            if not isinstance(note, dict):
                continue
            comments = note.get("comments", {})
            if not isinstance(comments, dict):
                comments = {}
            note["comment_count"] = len(comments)
            sub_count: Dict[str, int] = {}
            for cid, c in comments.items():
                if not isinstance(c, dict):
                    continue
                pid = c.get("parent_comment_id")
                if pid is not None and pid:
                    sub_count[pid] = sub_count.get(pid, 0) + 1
            for cid, c in comments.items():
                if isinstance(c, dict):
                    c["sub_comment_count"] = sub_count.get(cid, 0)

    @staticmethod
    def _timestamp_duration_days_for_step(
        step_num: int,
        max_step: int = 8,
        max_span_days: float = 24.0,
        schedule_type: str = "power",
        power_p: float = 1.6,
        sigmoid_scale: float = 1.2,
        sigmoid_center_ratio: float = 0.5,
    ) -> float:
        """
        Allocate the time step (days) incrementally by step. Supports two strategies:
        1) power:   w_i = i^p
        2) sigmoid: w_i = eps + sigmoid((i-center)/scale)
        """
        try:
            s = int(step_num)
        except (TypeError, ValueError):
            s = 1
        s = max(1, s)

        T = max(1, int(max_step))
        try:
            span = float(max_span_days)
        except (TypeError, ValueError):
            span = 24.0
        span = max(0.0, span)
        if T == 1:
            return span

        mode = str(schedule_type).strip().lower() if schedule_type is not None else "power"
        weights = []

        if mode == "sigmoid":
            # The smaller the scale, the steeper the curve; center_ratio controls the inflection point position (0~1)
            try:
                scale = float(sigmoid_scale)
            except (TypeError, ValueError):
                scale = 1.2
            scale = max(1e-6, scale)
            try:
                center_ratio = float(sigmoid_center_ratio)
            except (TypeError, ValueError):
                center_ratio = 0.5
            center_ratio = min(1.0, max(0.0, center_ratio))
            center = 1.0 + (T - 1.0) * center_ratio
            eps = 0.05
            for i in range(1, T + 1):
                x = (i - center) / scale
                w = eps + (1.0 / (1.0 + math.exp(-x)))
                weights.append(w)
        else:
            # Default power strategy
            try:
                p = float(power_p)
            except (TypeError, ValueError):
                p = 1.6
            p = max(0.0, p)
            for i in range(1, T + 1):
                weights.append(float(i) ** p)

        total_w = sum(weights)
        if total_w <= 0.0:
            return span / float(T)

        idx = min(s, T) - 1
        return span * (weights[idx] / total_w)

    def _note_copy_limited_comments(self, note: dict, top_k: int = 5, random_k: int = 0) -> dict:
        """Copy the note and truncate the comments."""
        base = {k: v for k, v in note.items() if k != "comments"}
        comments = note.get("comments") or {}
        if not isinstance(comments, dict) or not comments:
            base["comments"] = {}
            return base

        def comment_score(c: dict) -> float:
            """The score of a comment is higher if its sub-comment count is higher."""
            if not isinstance(c, dict):
                return 0.0
            sub = c.get("sub_comment_count", 0) or 0
            return float(sub)

        # Sort by sub-comment count, and keep the top k highest comments
        items = list(comments.items())
        random.shuffle(items)
        sorted_items = sorted(items, key=lambda x: comment_score(x[1]), reverse=True)
        kept_ids = {cid for cid, _ in sorted_items[:top_k]}  

        # Randomly sample random_k comments
        rest = [(cid, _) for cid, _ in sorted_items[top_k:] if cid not in kept_ids]
        if rest and random_k > 0:
            n_rand = min(random_k, len(rest))
            for cid, _ in random.sample(rest, n_rand):
                kept_ids.add(cid)
        base["comments"] = {cid: comments[cid] for cid in kept_ids if cid in comments}
        return base

    @staticmethod
    def _is_note_time_before_hi(note: Dict[str, Any], _lo: float, hi: float) -> bool:
        """The note time is strictly before hi."""
        raw_time = note.get("time", None)
        try:
            note_time = float(raw_time)
        except (TypeError, ValueError):
            return False
        return note_time < hi

    def _build_current_notes_subset(
        self, content_pool: Dict[str, Any], lo_ts: float, hi_ts: float
    ) -> Dict[str, Any]:
        """Build the current_notes subset."""
        if hi_ts <= lo_ts:
            return {}
        return {
            nid: self._note_copy_limited_comments(note, top_k=5, random_k=0)
            for nid, note in content_pool.items()
            if isinstance(note, dict) and self._is_note_time_before_hi(note, lo_ts, hi_ts)
        }

    @staticmethod
    def _total_comments_in_content_pool(content_pool: Dict[str, Any]) -> int:
        """Count the total number of comments in content_pool."""
        total = 0
        for note in content_pool.values():
            if not isinstance(note, dict):
                continue
            comments = note.get("comments", {})
            if isinstance(comments, dict):
                total += len(comments)
        return total

    @staticmethod
    def _parse_memory_timestamp_ms(raw: Any) -> Optional[int]:
        """Parse the note ``time`` or comment ``timestamp`` to milliseconds."""
        if raw is None or isinstance(raw, bool):
            return None
        try:
            if isinstance(raw, (int, float)):
                x = float(raw)
            else:
                x = float(str(raw).strip())
        except (TypeError, ValueError):
            return None
        if x <= 0:
            return None
        return int(round(x * 1000.0)) if x < 1e11 else int(round(x))

    @staticmethod
    def _apply_note_popularity_window(
        content_pool: Dict[str, Any],
        ref_ts_ms: int,
        window_days: float,
    ) -> None:
        """Count the number of comments in each note within the window (default window 7 days)."""
        try:
            wd = float(window_days)
        except (TypeError, ValueError):
            wd = 7.0
        if wd <= 0:
            wd = 7.0
        window_ms = int(wd * 86400000)
        if ref_ts_ms <= 0:
            for note in content_pool.values():
                if isinstance(note, dict):
                    note["popularity"] = 0
            return
        lo = ref_ts_ms - window_ms
        for note in content_pool.values():
            if not isinstance(note, dict):
                continue
            comments = note.get("comments", {})
            if not isinstance(comments, dict):
                note["popularity"] = 0
                continue
            cnt = 0
            for c in comments.values():
                if not isinstance(c, dict):
                    continue
                ts = SimEnv._parse_memory_timestamp_ms(c.get("timestamp"))
                if ts is None:
                    continue
                if lo <= ts <= ref_ts_ms:
                    cnt += 1
            note["popularity"] = cnt

    async def _save_step_data(self, step_num: int):
        """Update the environment data after each step."""
        async with self._lock:
            save_channel_snapshots(self, step_num)

        await super()._save_step_data(step_num)

        async with self._lock:
            # Output the full mention_pool content
            mention_pool_snapshot = self.data.get("mention_pool", {})
            try:
                mention_pool_dump = json.dumps(mention_pool_snapshot, ensure_ascii=False, indent=2, default=str)
            except Exception:
                mention_pool_dump = str(mention_pool_snapshot)
            logger.info(f"Step {step_num}: mention_pool snapshot:\n{mention_pool_dump}")

            content_pool = self.data.get("content_pool", {})
            if not isinstance(content_pool, dict):
                logger.warning("content_pool is not a dict, skipping update")
                return

            # Update the comment_count and sub_comment_count of each note in content_pool
            self._normalize_content_pool_comment_meta(content_pool)

            total_comments = self._total_comments_in_content_pool(content_pool)
            try:
                ref_ts_ms = int(self.data.get("current_timestamp", 0) or 0)
            except (TypeError, ValueError):
                ref_ts_ms = 0
            pop_days = self.data.get("popularity_window_days", 7.0)
            self._apply_note_popularity_window(content_pool, ref_ts_ms, float(pop_days) if pop_days is not None else 7.0)
            pop_values = [
                int(n.get("popularity", 0) or 0)
                for n in content_pool.values()
                if isinstance(n, dict)
            ]
            pop_sum = sum(pop_values)
            pop_max = max(pop_values) if pop_values else 0
            logger.info(
                f"Step {step_num}: content_pool comment total (sum of comment counts in all notes) = {total_comments}; "
                f"note popularity (comment count in the last {pop_days} days, right endpoint current_timestamp={ref_ts_ms}): "
                f"sum={pop_sum}, max_per_note={pop_max}, notes={len(pop_values)}"
            )
            save_content_pool_snapshot(self, step_num, content_pool)

            # Update current_timestamp
            current_timestamp = self.data.get("current_timestamp", 1764255440000)
            if not isinstance(current_timestamp, (int, float)) or current_timestamp <= 0:
                logger.warning(f"Invalid current_timestamp: {current_timestamp}, skipping update")
                return

            day_ms = 86400000
            sched = self._simulator_schedule_settings()
            max_span_days = sched["max_span_days"]
            max_step = sched["max_step"]
            schedule_type = sched["timestamp_schedule_type"]
            power_p = sched["timestamp_power_p"]
            sigmoid_scale = sched["timestamp_sigmoid_scale"]
            sigmoid_center_ratio = sched["timestamp_sigmoid_center_ratio"]
            step_delta_ms = int(
                self._timestamp_duration_days_for_step(
                    step_num,
                    max_step=max_step,
                    max_span_days=max_span_days,
                    schedule_type=schedule_type,
                    power_p=power_p,
                    sigmoid_scale=sigmoid_scale,
                    sigmoid_center_ratio=sigmoid_center_ratio,
                )
                * day_ms
            )
            start_ts = self.data.get("simulation_start_timestamp")
            if not isinstance(start_ts, (int, float)) or start_ts <= 0:
                start_ts = current_timestamp
                self.data["simulation_start_timestamp"] = start_ts
            cap_ts = int(start_ts + max_span_days * day_ms)
            next_timestamp = min(int(current_timestamp) + step_delta_ms, cap_ts)
            self.data["current_timestamp"] = next_timestamp

            self.data["current_step"] = step_num + 1

            # Calculate the width of the next time window for the next StartEvent
            next_idx = step_num + 1
            try:
                max_step_i = int(max_step)
            except (TypeError, ValueError):
                max_step_i = 8
            if next_idx > max_step_i:
                dur_next_ms = 0
            else:
                dur_next_ms = int(
                    self._timestamp_duration_days_for_step(
                        next_idx,
                        max_step=max_step_i,
                        max_span_days=max_span_days,
                        schedule_type=schedule_type,
                        power_p=power_p,
                        sigmoid_scale=sigmoid_scale,
                        sigmoid_center_ratio=sigmoid_center_ratio,
                    )
                    * day_ms
                )
            # timestamp_duration represents the distance between the current timestamp and the next timestamp covered by current_notes
            self.data["timestamp_duration"] = dur_next_ms

            logger.info(
                f"Step {step_num}: step_delta_ms (completed slice)={step_delta_ms}, "
                f"dur_next_ms (next window for StartEvent)={dur_next_ms}"
            )
            logger.info(f"Step {step_num}: Updated current_timestamp from {current_timestamp} to {next_timestamp}")

            # Build current_notes: the notes with time < min(next_timestamp + dur_next_ms, cap_ts)
            lo = float(self.data["current_timestamp"])
            hi = min(lo + float(dur_next_ms), float(cap_ts))
            self.data["current_notes"] = self._build_current_notes_subset(content_pool, lo, hi)

            logger.info(f"Step {step_num}: Updated comment_count in content_pool, current_notes count: {len(self.data['current_notes'])}")

    async def _create_start_event(self, target_id: str) -> Event:
        """Create a StartEvent for the given target_id."""
        source_id = self.data.get('source_agent_id', 'default_source')
        timestamp = self.data.get('current_timestamp', 0)
        timestamp_duration = self.data.get('timestamp_duration', 86400000)

        current_step = self.data.get("current_step", 1)
        sched = self._simulator_schedule_settings()
        max_step = sched["max_step"]
        logger.info(f"Step {current_step}/{max_step}: timestamp: {timestamp}, timestamp_duration: {timestamp_duration}")

        current_notes = self.data.get('current_notes', {})

        day_ms = 86400000
        max_span_days = sched["max_span_days"]
        start_ts = self.data.get("simulation_start_timestamp")
        if isinstance(start_ts, (int, float)) and int(start_ts) > 0:
            simulation_cap_timestamp = int(start_ts + max_span_days * day_ms)
        else:
            ts0 = int(timestamp) if timestamp else 0
            simulation_cap_timestamp = int(ts0 + max_span_days * day_ms)

        if isinstance(target_id, str) and target_id.startswith('recomment_agent_'):
            return StartEvent(
                from_agent_id=source_id,
                to_agent_id=target_id,
                timestamp=timestamp,
                timestamp_duration=timestamp_duration,
                simulation_cap_timestamp=simulation_cap_timestamp,
                current_step=current_step,
                max_step=max_step,
                current_notes=current_notes,
            )
        
        # Set the login of the corresponding UserAgent to -1
        if hasattr(self, 'agents') and self.agents:
            user_agents = self.agents.get('UserAgent', {})
            if isinstance(user_agents, dict) and target_id in user_agents:
                agent = user_agents[target_id]
                if getattr(agent, 'profile', None) is not None:
                    agent.profile.update_data('login', -1)
        
        # Build mentions: the mentions of the current target
        mention_pool_raw = self.data.get('mention_pool', {})
        mention_pool = mention_pool_raw.get(target_id, {}) if isinstance(mention_pool_raw, dict) else {}
        mentions = {}
        for comment_id, mention_message in mention_pool.items():
            note_id = mention_message.get("note_id", "")
            if note_id in current_notes:
                mentions[comment_id] = {
                    "note": current_notes[note_id],
                    "comment_id": comment_id,
                    "comment_content": mention_message.get("comment_content", ""),
                    "mentioner_id": mention_message.get("mentioner_id", ""),
                    "mentioner_nickname": mention_message.get("mentioner_nickname", ""),
                    "mention_type": mention_message.get("mention_type", "")
                }

        logger.info(f"StartEvent: length of mentions for UserAgent {target_id}: {len(mentions)}")
        return StartEvent(
            from_agent_id=source_id,
            to_agent_id=target_id,
            timestamp=timestamp,
            timestamp_duration=timestamp_duration,
            current_step=current_step,
            max_step=max_step,
            current_notes=current_notes,
            mentions=mentions,
            simulation_cap_timestamp=simulation_cap_timestamp,
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
            if "." in key:
                # Parse the note_id and the field path
                parts = key.split(".")  
                if parts[0] != "content_pool" or parts[2] != "comments" or len(parts) != 3:
                    raise ValueError(f"Invalid key: {key}, expected format: content_pool.note_id.comments")
                note_id = parts[1]     # Note ID

                current_pool = self.data.get("content_pool", {})
                if not isinstance(current_pool, dict):
                    current_pool = {}
                    logger.warning("content_pool is not a dict, converting from list format")
            
                if note_id not in current_pool:
                    raise ValueError(f"note_id {note_id} not found in content_pool")
                note = current_pool[note_id]
                
                if "comments" not in note:
                    note["comments"] = {}
                if not isinstance(note["comments"], dict):
                    note["comments"] = {}
                
                if not isinstance(data, dict):
                    raise ValueError(f"data must be a dict: {data}")
                
                # If data contains comment_id, use comment_id as the key
                # Otherwise, merge the entire data dictionary
                if "comment_id" in data:
                    comment_id = data["comment_id"]
                    note["comments"][comment_id] = {**note["comments"].get(comment_id, {}), **data}
                else:
                    note["comments"] = {**note["comments"], **data}
                
                self.data["content_pool"] = current_pool
                
                logger.debug(f"Updated comments in {key}")
                return True
            else:
                raise ValueError(f"Invalid key: {key}, expected format: content_pool.note_id.comments")
            
    async def handle_add_comment_event(self, event: AddCommentEvent) -> None:
        """
        Handle the add comment event from the agent (using a distributed lock)
        Example:
            {
                "comment_id": "123",
                "create_time": 1764298329000,
                "ip_location": "Guangdong",
                "note_id": "69290e59000000001e034ab4",
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
                # Update the request data
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

    async def update_mention_pool(self, key: str, data: Any) -> Any:
        """Update the shared data (asynchronously, using a lock)"""
        async with self._lock:
            if "." in key:
                # Parse the note_id and the field path
                parts = key.split(".")
                if parts[0] != "mention_pool" or len(parts) != 3:
                    raise ValueError(f"Invalid key: {key}, expected format: content_pool.note_id.comments")
                mentioner_id = parts[1]     # Mentioner ID
                comment_id = parts[2]
                
                mention_pool = self.data.get("mention_pool", {})
                if not isinstance(mention_pool, dict):
                    mention_pool = {}
                    logger.warning("mention_pool is not a dict, converting from list format")

                if mentioner_id not in mention_pool:
                    mention_pool[mentioner_id] = {}

                mentioner_pool = mention_pool[mentioner_id]

                # Delete operation
                if isinstance(data, dict) and data.get("action") == "delete":
                    if comment_id in mentioner_pool:
                        del mentioner_pool[comment_id]
                        logger.info(f"Deleted comment {comment_id} from mentioner {mentioner_id}")
                    else:
                        logger.warning(f"comment {comment_id} not found in mentioner {mentioner_id}, skip delete")
                
                # Add operation
                elif isinstance(data, dict) and data.get("action") == "add":
                    if comment_id in mentioner_pool:
                        raise ValueError(f"comment {comment_id} already exists in mentioner {mentioner_id}")
                    mentioner_pool[comment_id] = data.get("mention_message", {})
                    logger.info(f"Added comment {comment_id} to mentioner {mentioner_id}")
              
                self.data["mention_pool"] = mention_pool
                return True
            else:
                raise ValueError(f"Invalid key: {key}, expected format: mention_pool.mentioner_id.comment_id")
            
    async def handle_update_mention_pool_event(self, event: MentionPoolUpdateEvent) -> None:
        """
        Handle the update mention_pool event from the agent (using a distributed lock)
        Example:
            {
                "action": "add",
                "mention_message": {
                    "note_id": "69290e59000000001e034ab4",
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
                # Update the request data
                success = await self.update_mention_pool(event.key, event.value)

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