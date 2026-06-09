from onesim.simulator import BasicSimEnv
from onesim.events import Event
from onesim.events import DataUpdateEvent, DataUpdateResponseEvent
from onesim.distribution.distributed_lock import get_lock 
from typing import Any, Dict, List, Optional, Set
from loguru import logger
import json
import math
import random
import os
from .events import StartEvent, AddRepostEvent, AddRepostResponseEvent, MentionPoolUpdateEvent, MentionPoolUpdateResponseEvent
from .metrics.channel_snapshots import save_channel_snapshots, save_content_pool_snapshot

# Simulation timestamps use Unix seconds
SEC_PER_DAY = 86400


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
        """
        Initialize SimEnv
        """
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
        self.register_event("AddRepostEvent", "handle_add_repost_event")
        self.register_event("MentionPoolUpdateEvent", "handle_update_mention_pool_event")

    def _normalize_sim_timestamps_to_seconds(self) -> None:
        """Normalize current_timestamp / timestamp_duration / simulation_start to Unix seconds; compatible with old version milliseconds."""
        def abs_to_sec(v: Any) -> Optional[int]:
            if v is None:
                return None
            try:
                x = int(float(v))
            except (TypeError, ValueError):
                return None
            if x <= 0:
                return None
            if x >= 10**12:
                return x // 1000
            return x

        def duration_to_sec(v: Any) -> Optional[int]:
            if v is None:
                return None
            try:
                x = int(float(v))
            except (TypeError, ValueError):
                return None
            if x <= 0:
                return None
            if x >= SEC_PER_DAY * 1000:
                return max(1, x // 1000)
            return x

        for k in ("current_timestamp", "simulation_start_timestamp"):
            if k not in self.data:
                continue
            nv = abs_to_sec(self.data.get(k))
            if nv is not None:
                self.data[k] = nv

        td = self.data.get("timestamp_duration")
        if td is not None:
            nv = duration_to_sec(td)
            if nv is not None:
                self.data["timestamp_duration"] = nv

    def _normalize_content_pool_times_to_seconds(self, content_pool: Dict[str, Any]) -> None:
        """Normalize content_pool[*].time to Unix seconds (int or string of digits); compatible with milliseconds."""
        for blog in content_pool.values():
            if not isinstance(blog, dict):
                continue
            raw = blog.get("time")
            if raw is None or raw == "":
                continue
            try:
                v = float(raw)
            except (TypeError, ValueError):
                continue
            if v >= 10**12:
                v = v / 1000.0
            blog["time"] = int(v)

    async def load_initial_data(self) -> None:
        """Complete the derived fields, including current_blogs."""
        await super().load_initial_data()
        async with self._lock:
            self._normalize_sim_timestamps_to_seconds()

            # Update the repost count of each blog in content_pool
            content_pool = self.data.get("content_pool", {})
            if not isinstance(content_pool, dict):
                logger.warning("load_initial_data: content_pool is not a dict, skip current_blogs bootstrap")
                return

            self._normalize_content_pool_times_to_seconds(content_pool)
            self._normalize_content_pool_repost_meta(content_pool)

            # Update current_timestamp and timestamp_duration for the first step
            ts = self.data.get("current_timestamp", 1764255440)
            if not isinstance(ts, (int, float)) or int(ts) <= 0:
                logger.warning(f"load_initial_data: invalid current_timestamp {ts}, set current_blogs empty")
                self.data["current_blogs"] = {}
                return

            sched = self._simulator_schedule_settings()
            max_span_days = sched["max_span_days"]
            max_step = sched["max_step"]
            schedule_type = sched["timestamp_schedule_type"]
            power_p = sched["timestamp_power_p"]
            sigmoid_scale = sched["timestamp_sigmoid_scale"]
            sigmoid_center_ratio = sched["timestamp_sigmoid_center_ratio"]

            if not isinstance(self.data.get("simulation_start_timestamp"), (int, float)) or int(
                self.data.get("simulation_start_timestamp") or 0
            ) <= 0:
                self.data["simulation_start_timestamp"] = int(ts)

            td = self.data.get("timestamp_duration")
            if td is None or td == 0:
                # Only fill in the "first segment" day → second, for the first StartEvent; do not advance current_timestamp here (still use the starting point in JSON)
                dur_days = self._timestamp_duration_days_for_step(
                    1,
                    max_step=max_step,
                    max_span_days=max_span_days,
                    schedule_type=schedule_type,
                    power_p=power_p,
                    sigmoid_scale=sigmoid_scale,
                    sigmoid_center_ratio=sigmoid_center_ratio,
                )
                self.data["timestamp_duration"] = int(dur_days * SEC_PER_DAY)

            start_ts = int(self.data["simulation_start_timestamp"])
            cap_ts = int(start_ts + max_span_days * SEC_PER_DAY)
            lo = float(self.data["current_timestamp"])
            dur_sec = int(self.data.get("timestamp_duration") or 0)
            hi = min(lo + dur_sec, float(cap_ts))

            logger.info(
                f"load_initial_data: current_timestamp={lo}, timestamp_duration(next window)={dur_sec} s, "
                f"current_blogs: time < {hi} s (min(lo+duration, cap))"
            )

            # Build current_blogs subset
            self.data["current_blogs"] = self._build_current_blogs_subset(content_pool, lo, hi)
            logger.info(f"load_initial_data: built current_blogs count={len(self.data['current_blogs'])}")

    def _normalize_content_pool_repost_meta(self, content_pool: Dict[str, Any]) -> None:
        """Update the repost_count of each blog in content_pool."""
        for blog_id, blog in content_pool.items():
            if not isinstance(blog, dict):
                continue
            reposted_path = blog.get("reposted_path", [])
            for reposted_blog_id in reposted_path:
                if reposted_blog_id not in content_pool:
                    continue
                reposted_blog = content_pool[reposted_blog_id]
                if not isinstance(reposted_blog, dict):
                    continue
                repost_count = reposted_blog.get("repost_count", 0)
                reposted_blog["repost_count"] = repost_count + 1
                rid_list = reposted_blog.get("repost_ids")
                if not isinstance(rid_list, list):
                    rid_list = []
                    reposted_blog["repost_ids"] = rid_list
                # 避免重复记录同一个转发ID
                if blog_id not in rid_list:
                    rid_list.append(blog_id)

    @staticmethod
    def _is_blog_time_before_hi(blog: Dict[str, Any], _lo: float, hi: float) -> bool:
        """The posting time of the blog is strictly before hi."""
        raw_time = blog.get("time", None)
        try:
            blog_time = float(raw_time)
        except (TypeError, ValueError):
            return False
        if blog_time >= 10**12:
            blog_time = blog_time / 1000.0
        return blog_time < hi

    def _build_current_blogs_subset(
        self, content_pool: Dict[str, Any], lo_ts: float, hi_ts: float
    ) -> Dict[str, Any]:
        """The blogs with posting time < hi_ts."""
        if hi_ts <= lo_ts:
            return {}
        return {
            bid: content_pool[bid]
            for bid, blog in content_pool.items()
            if isinstance(blog, dict) and self._is_blog_time_before_hi(blog, lo_ts, hi_ts)
        }

    @staticmethod
    def _total_reposts_in_content_pool(content_pool: Dict[str, Any]) -> int:
        """Count the total number of reposts in content_pool."""
        total = 0
        for blog in content_pool.values():
            if not isinstance(blog, dict):
                continue
            repost_count = blog.get("repost_count", 0)
            total += repost_count
        return total

    @staticmethod
    def _parse_event_time_seconds(raw: Any) -> Optional[int]:
        """Parse the ``time`` of the blog to Unix seconds (int); compatible with milliseconds."""
        if raw is None or isinstance(raw, bool):
            return None
        try:
            x = float(raw) if isinstance(raw, (int, float)) else float(str(raw).strip())
        except (TypeError, ValueError):
            return None
        if x <= 0:
            return None
        if x >= 10**12:
            x = x / 1000.0
        return int(round(x))

    @staticmethod
    def _apply_blog_popularity_window(
        content_pool: Dict[str, Any],
        ref_ts_sec: int,
        window_days: float,
    ) -> None:
        """Count the popularity of each blog in the popularity window (default 7 days)."""
        try:
            wd = float(window_days)
        except (TypeError, ValueError):
            wd = 7.0
        if wd <= 0:
            wd = 7.0
        window_sec = int(wd * SEC_PER_DAY)
        if ref_ts_sec <= 0:
            for b in content_pool.values():
                if isinstance(b, dict):
                    b["popularity"] = 0
            return
        lo = ref_ts_sec - window_sec
        for b in content_pool.values():
            if isinstance(b, dict):
                b["popularity"] = 0
        for _bid, blog in content_pool.items():
            if not isinstance(blog, dict):
                continue
            path = blog.get("reposted_path")
            keys: List[str] = []
            if isinstance(path, list) and path:
                seen: Set[str] = set()
                for x in path:
                    k = str(x).strip() if x is not None else ""
                    if k and k not in seen:
                        seen.add(k)
                        keys.append(k)
            else:
                pid = blog.get("reposted_blog_id")
                if not pid:
                    continue
                k = str(pid).strip()
                if not k:
                    continue
                keys.append(k)
            if not keys:
                continue
            ts = SimEnv._parse_event_time_seconds(blog.get("time"))
            if ts is None:
                continue
            if not (lo <= ts <= ref_ts_sec):
                continue
            for key in keys:
                ob = content_pool.get(key)
                if isinstance(ob, dict):
                    ob["popularity"] = int(ob.get("popularity", 0) or 0) + 1

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

            # Update the repost_count of each blog in content_pool
            self._normalize_content_pool_repost_meta(content_pool)
                    
            total_reposts = self._total_reposts_in_content_pool(content_pool)
            try:
                ref_ts_sec = int(self.data.get("current_timestamp", 0) or 0)
            except (TypeError, ValueError):
                ref_ts_sec = 0
            pop_days = self.data.get("popularity_window_days", 7.0)
            self._apply_blog_popularity_window(
                content_pool, ref_ts_sec, float(pop_days) if pop_days is not None else 7.0
            )
            pop_values = [
                int(b.get("popularity", 0) or 0)
                for b in content_pool.values()
                if isinstance(b, dict)
            ]
            pop_sum = sum(pop_values)
            pop_max = max(pop_values) if pop_values else 0
            logger.info(
                f"Step {step_num}: content_pool repost total (sum of repost counts in all blogs) = {total_reposts}; "
                f"blog popularity (chain reposts counted into ancestors in the last {pop_days} days, right endpoint current_timestamp={ref_ts_sec}s): "
                f"sum={pop_sum}, max_per_blog={pop_max}, blogs={len(pop_values)}"
            )
            save_content_pool_snapshot(self, step_num, content_pool)

            # Update current_timestamp
            current_timestamp = self.data.get("current_timestamp", 1764255440)
            if not isinstance(current_timestamp, (int, float)) or current_timestamp <= 0:
                logger.warning(f"Invalid current_timestamp: {current_timestamp}, skipping update")
                return
        
            sched = self._simulator_schedule_settings()
            max_span_days = sched["max_span_days"]
            max_step = sched["max_step"]
            schedule_type = sched["timestamp_schedule_type"]
            power_p = sched["timestamp_power_p"]
            sigmoid_scale = sched["timestamp_sigmoid_scale"]
            sigmoid_center_ratio = sched["timestamp_sigmoid_center_ratio"]
            step_delta_sec = int(
                self._timestamp_duration_days_for_step(
                    step_num,
                    max_step=max_step,
                    max_span_days=max_span_days,
                    schedule_type=schedule_type,
                    power_p=power_p,
                    sigmoid_scale=sigmoid_scale,
                    sigmoid_center_ratio=sigmoid_center_ratio,
                )
                * SEC_PER_DAY
            )
            start_ts = self.data.get("simulation_start_timestamp")
            if not isinstance(start_ts, (int, float)) or start_ts <= 0:
                start_ts = current_timestamp
                self.data["simulation_start_timestamp"] = start_ts
            cap_ts = int(start_ts + max_span_days * SEC_PER_DAY)
            next_timestamp = min(int(current_timestamp) + step_delta_sec, cap_ts)
            self.data["current_timestamp"] = next_timestamp

            self.data["current_step"] = step_num + 1

            # Calculate the width of the next time window for the next StartEvent
            next_idx = step_num + 1
            try:
                max_step_i = int(max_step)
            except (TypeError, ValueError):
                max_step_i = 8
            if next_idx > max_step_i:
                dur_next_sec = 0
            else:
                dur_next_sec = int(
                    self._timestamp_duration_days_for_step(
                        next_idx,
                        max_step=max_step_i,
                        max_span_days=max_span_days,
                        schedule_type=schedule_type,
                        power_p=power_p,
                        sigmoid_scale=sigmoid_scale,
                        sigmoid_center_ratio=sigmoid_center_ratio,
                    )
                    * SEC_PER_DAY
                )
            # timestamp_duration represents the distance between the current timestamp and the next timestamp covered by current_notes
            self.data["timestamp_duration"] = dur_next_sec

            logger.info(
                f"Step {step_num}: step_delta_sec (completed slice)={step_delta_sec}, "
                f"dur_next_sec (next window for StartEvent)={dur_next_sec}"
            )
            logger.info(f"Step {step_num}: Updated current_timestamp from {current_timestamp} to {next_timestamp}")

            # Build current_blogs: the blogs with time < min(next_timestamp + dur_next_ms, cap_ts)
            lo = float(self.data["current_timestamp"])
            hi = min(lo + float(dur_next_sec), float(cap_ts))
            self.data["current_blogs"] = self._build_current_blogs_subset(content_pool, lo, hi)

            logger.info(f"Step {step_num}: Updated repost_count in content_pool, current_blogs count: {len(self.data['current_blogs'])}")

    async def _create_start_event(self, target_id: str) -> Event:
        """Create a StartEvent for the given target_id."""
        source_id = self.data.get('source_agent_id', 'default_source')
        timestamp = self.data.get('current_timestamp', 0)
        timestamp_duration = self.data.get('timestamp_duration', SEC_PER_DAY)

        current_step = self.data.get("current_step", 1)
        sched = self._simulator_schedule_settings()
        max_step = sched["max_step"]
        logger.info(f"Step {current_step}/{max_step}: timestamp: {timestamp}, timestamp_duration: {timestamp_duration}")

        current_blogs = self.data.get('current_blogs', {})

        max_span_days = sched["max_span_days"]
        start_ts = self.data.get("simulation_start_timestamp")
        if isinstance(start_ts, (int, float)) and int(start_ts) > 0:
            simulation_cap_timestamp = int(start_ts + max_span_days * SEC_PER_DAY)
        else:
            ts0 = int(timestamp) if timestamp else 0
            simulation_cap_timestamp = int(ts0 + max_span_days * SEC_PER_DAY)

        if isinstance(target_id, str) and target_id.startswith('recomment_agent_'):
            return StartEvent(
                from_agent_id=source_id, 
                to_agent_id=target_id, 
                timestamp=timestamp, 
                timestamp_duration=timestamp_duration, 
                simulation_cap_timestamp=simulation_cap_timestamp,
                current_step=current_step, 
                max_step=max_step, 
                current_blogs=current_blogs
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
        for blog_id, mention_message in mention_pool.items():
            # 补充reposted_blog字段
            if blog_id in current_blogs:
                blog = current_blogs[blog_id]
                reposted_blog_id = blog.get("reposted_blog_id", "")
                if reposted_blog_id and reposted_blog_id in current_blogs:
                    blog["reposted_blog"] = current_blogs[reposted_blog_id]
                mentions[blog_id] = {
                    "blog": blog,
                    "mention_type": mention_message.get("mention_type", "")
                }

        logger.info(f"StartEvent: length of mentions for UserAgent {target_id}: {len(mentions)}")
        return StartEvent(
            from_agent_id=source_id, 
            to_agent_id=target_id, 
            timestamp=timestamp, 
            timestamp_duration=timestamp_duration, 
            simulation_cap_timestamp=simulation_cap_timestamp,
            current_step=current_step, 
            max_step=max_step, 
            current_blogs=current_blogs,
            mentions=mentions
        )

    async def queue_event(self, event_data: Dict[str, Any]):
        """Add the event to the queue, and save and broadcast it when the step ends"""
        if event_data['event_type'] in [
            'AddRepostEvent',
            'AddRepostResponseEvent',
            'MentionPoolUpdateEvent',
            'MentionPoolUpdateResponseEvent'
        ]:
            return
        await super().queue_event(event_data)

    async def add_repost(self, key: str, data: Any) -> Any:
        """Add the repost to the content_pool"""
        async with self._lock:
            current_pool = self.data.get("content_pool", {})
            if not isinstance(current_pool, dict):
                current_pool = {}
            blog_id = data.get("blog_id", "")
            if not blog_id:
                raise ValueError("Missing blog_id in repost data")
            if blog_id in current_pool:
                raise ValueError(f"blog_id {blog_id} already exists in content_pool")
            # Add a new repost: use blog_id as the key
            current_pool[blog_id] = dict(data)

            # Write back to content_pool
            self.data["content_pool"] = current_pool
            logger.debug(f"Added repost blog_id={blog_id} into content_pool")
            return True
            
    async def handle_add_repost_event(self, event: AddRepostEvent) -> None:
        """
        Handle the add repost event from the agent (using a distributed lock)
        Example:
            {
                "blog_id": "123",
                "content": "repost content",
                "time": 1764298329,
                "ip_location": "Guangdong",
                "user_id": "5e2a573f0000000001002ddf",
                "nickname": "User A",
                "at_count": 0,
                "reposted_blog_id": "69290e59000000001e034ab4",
                "reposted_path": ["69290e59000000001e034ab4", "69290e59000000001e034ab5"],
                "repost_count": 0,
                "reposts": {}
            }
        """
        try:
            logger.info(f"Received AddRepostEvent, request_id={event.request_id}, key={event.key}")
            lock_id = f"env_repost_add_lock_content_pool"
            lock = await get_lock(lock_id)

            # Get the lock before updating the data
            async with lock:
                # Update the request data
                success = await self.add_repost(event.key, event.value)

                # Create and send the response event
                response_event = AddRepostResponseEvent(
                    from_agent_id=self.name,
                    to_agent_id=event.from_agent_id,
                    request_id=event.request_id,
                    key=event.key,
                    success=success
                )
                await self.event_bus.dispatch_event(response_event)

        except Exception as e:
            error_response = AddRepostResponseEvent(
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
                # Parse the blog_id and the field path
                parts = key.split(".")
                if parts[0] != "mention_pool" or len(parts) != 3:
                    raise ValueError(f"Invalid key: {key}, expected format: content_pool.mentioner_id.blog_id")
                mentioner_id = parts[1]     # Mentioner ID
                blog_id = parts[2]
                
                mention_pool = self.data.get("mention_pool", {})
                if not isinstance(mention_pool, dict):
                    mention_pool = {}
                    logger.warning("mention_pool is not a dict, converting from list format")

                if mentioner_id not in mention_pool:
                    mention_pool[mentioner_id] = {}

                mentioner_pool = mention_pool[mentioner_id]

                # Delete operation
                if isinstance(data, dict) and data.get("action") == "delete":
                    if blog_id in mentioner_pool:
                        del mentioner_pool[blog_id]
                        logger.info(f"Deleted blog {blog_id} from mentioner {mentioner_id}")
                    else:
                        logger.warning(f"blog {blog_id} not found in mentioner {mentioner_id}, skip delete")
                
                # Add operation
                elif isinstance(data, dict) and data.get("action") == "add":
                    if blog_id in mentioner_pool:
                        raise ValueError(f"blog {blog_id} already exists in mentioner {mentioner_id}")
                    mentioner_pool[blog_id] = data.get("mention_message", {})
                    logger.info(f"Added blog {blog_id} to mentioner {mentioner_id}")
              
                self.data["mention_pool"] = mention_pool
                return True
            else:
                raise ValueError(f"Invalid key: {key}, expected format: mention_pool.mentioner_id.blog_id")
            
    async def handle_update_mention_pool_event(self, event: MentionPoolUpdateEvent) -> None:
        """
        Handle the update mention_pool event from the agent (using a distributed lock)
        Example:
            {
                "action": "add",
                "mention_message": {
                    "blog_id": "69290e59000000001e034ab4",
                    "mention_type": "repost"
                }
            }
        """
        try:
            logger.info(f"Received MentionPoolUpdateEvent, request_id={event.request_id}, key={event.key}")
            parts = str(event.key).split(".")
            lock_key = parts[1] if len(parts) >= 2 and parts[0] == "mention_pool" else (parts[0] if parts else "")
            lock_id = f"env_mention_pool_update_lock_{lock_key}"
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