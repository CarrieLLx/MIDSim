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

# 与微博 create_time 一致：Unix 秒（十位）。旧版 JSON 可能为毫秒，在 load 时归一化。
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
        **kwargs  # 允许额外的参数
    ) -> None:
        """
        初始化 SimEnv
        
        可以在这里添加自定义的初始化逻辑
        """
        # 调用基类的 __init__
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
        # 在这里添加你的自定义初始化逻辑
        # 例如：注册自定义事件、初始化自定义属性等
        # self.register_event("CustomEvent", "handle_custom_event")
        # self.custom_attribute = None

    def _normalize_sim_timestamps_to_seconds(self) -> None:
        """将 current_timestamp / timestamp_duration / simulation_start 等统一为 Unix 秒；兼容旧版毫秒。"""
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
            # 旧配置：一天 = 86400000 毫秒；多步长亦为毫秒量级
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
        """content_pool[*].time 统一为与微博一致的 Unix 秒（int 或数字字符串）；兼容毫秒。"""
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
        """在基类合并 env_data.json 之后，补全派生字段；current_blogs 为发帖时间 time < min(current_timestamp+首轮 duration, cap) 的微博。"""
        await super().load_initial_data()
        async with self._lock:
            self._normalize_sim_timestamps_to_seconds()

            # 1. 更新 content_pool 中每条微博的转发数
            content_pool = self.data.get("content_pool", {})
            if not isinstance(content_pool, dict):
                logger.warning("load_initial_data: content_pool is not a dict, skip current_blogs bootstrap")
                return

            self._normalize_content_pool_times_to_seconds(content_pool)
            self._normalize_content_pool_repost_meta(content_pool)

            # 2. 构建 current_blogs：发帖时间 time < min(current_timestamp+首轮 duration, cap) 的微博
            ts = self.data.get("current_timestamp", 1764255440)
            if not isinstance(ts, (int, float)) or int(ts) <= 0:
                logger.warning(f"load_initial_data: invalid current_timestamp {ts}, set current_blogs empty")
                self.data["current_blogs"] = {}
                return

            max_span_days = float(self.data.get("max_span_days", 24.0))
            max_step = self.data.get("max_step", 8)
            schedule_type = self.data.get("timestamp_schedule_type", "power")
            power_p = self.data.get("timestamp_power_p", 1.6)
            sigmoid_scale = self.data.get("timestamp_sigmoid_scale", 1.2)
            sigmoid_center_ratio = self.data.get("timestamp_sigmoid_center_ratio", 0.5)

            if not isinstance(self.data.get("simulation_start_timestamp"), (int, float)) or int(
                self.data.get("simulation_start_timestamp") or 0
            ) <= 0:
                self.data["simulation_start_timestamp"] = int(ts)

            td = self.data.get("timestamp_duration")
            if td is None or td == 0:
                # 仅补齐「第 1 段」日→秒，供首轮 StartEvent；不在此提前拨动 current_timestamp（仍用 JSON 里的起点）
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

            self.data["current_blogs"] = self._build_current_blogs_subset(content_pool, lo, hi)
            logger.info(f"load_initial_data: built current_blogs count={len(self.data['current_blogs'])}")

    def _normalize_content_pool_repost_meta(self, content_pool: Dict[str, Any]) -> None:
        """就地更新每条微博的 repost_count。"""
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
        """发帖时间 time 严格早于 hi（秒；兼容旧版毫秒 time）。"""
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
        """发帖时间 time < hi_ts 的微博 + 截断转发视图（hi_ts 已含 cap；lo_ts 仅用于零宽窗早退）。"""
        if hi_ts <= lo_ts:
            return {}
        return {
            bid: content_pool[bid]
            for bid, blog in content_pool.items()
            if isinstance(blog, dict) and self._is_blog_time_before_hi(blog, lo_ts, hi_ts)
        }

    @staticmethod
    def _total_reposts_in_content_pool(content_pool: Dict[str, Any]) -> int:
        """统计 content_pool 中所有博客的转发条数之和（每个 blog 的 repost_count）。"""
        total = 0
        for blog in content_pool.values():
            if not isinstance(blog, dict):
                continue
            repost_count = blog.get("repost_count", 0)
            total += repost_count
        return total

    @staticmethod
    def _parse_event_time_seconds(raw: Any) -> Optional[int]:
        """解析微博 ``time`` 等为 Unix 秒（int）；与全站一致：≥1e12 视为毫秒并除以 1000。"""
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
        """
        以 ref_ts_sec（Unix 秒）为右端点，统计每条微博在 [ref - window, ref] 内收到的转发链热度：
        对每条转发帖，若其 ``time`` 落在窗口内，则对 ``reposted_path`` 上每条祖先 id（若无则仅
        ``reposted_blog_id``）的 ``popularity`` 各 +1（与链式转发计数语义一致）。
        """
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

    def _save_content_pool_snapshot(self, step_num: int, content_pool: Dict[str, Any]) -> None:
        """
        每轮保存一份 content_pool 全量快照，便于离线排查与回放。
        """
        if not isinstance(content_pool, dict):
            return

        # 优先使用仿真输出目录；缺失时回退到工作目录下的快照目录
        base_dir = getattr(self, "output_dir", None) or self.data.get("output_dir")
        if not isinstance(base_dir, str) or not base_dir.strip():
            base_dir = os.path.join(os.getcwd(), "runs_content_pool_snapshots")

        step_dir = os.path.join(base_dir, "datasets", f"step_{step_num}")
        os.makedirs(step_dir, exist_ok=True)
        snapshot_path = os.path.join(step_dir, "content_pool_snapshot.json")

        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(content_pool, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"Step {step_num}: Saved content_pool snapshot to {snapshot_path}")

    def _save_user_recommended_blog_ids_by_channel_snapshot(self, step_num: int) -> None:
        """
        每轮遍历所有用户智能体，以用户 id 为 key，保存各用户 profile 中的 recommended_blog_ids_by_channel。
        目录与 content_pool 快照一致：{output_dir}/datasets/step_{step_num}/
        """
        base_dir = getattr(self, "output_dir", None) or self.data.get("output_dir")
        if not isinstance(base_dir, str) or not base_dir.strip():
            base_dir = os.path.join(os.getcwd(), "runs_content_pool_snapshots")

        step_dir = os.path.join(base_dir, "datasets", f"step_{step_num}")
        os.makedirs(step_dir, exist_ok=True)
        snapshot_path = os.path.join(step_dir, "user_recommended_blog_ids_by_channel.json")

        combined: Dict[str, Any] = {}
        agents_map = getattr(self, "agents", None) or {}
        user_map = agents_map.get("UserAgent", {}) if isinstance(agents_map, dict) else {}
        if isinstance(user_map, dict):
            for aid, agent in user_map.items():
                uid = str(aid).strip() if aid is not None else ""
                prof = getattr(agent, "profile", None)
                if prof is not None and not uid:
                    uid = str(prof.get_data("id", "") or "").strip()
                if not uid:
                    continue
                if prof is None:
                    combined[uid] = {"last_login_timestamp": 0, "recommended_blog_ids_by_channel": {}}
                    continue
                raw = prof.get_data("recommended_blog_ids_by_channel", {})
                ll = prof.get_data("last_login_timestamp", 0)
                try:
                    ll_int = int(ll) if ll is not None else 0
                except (TypeError, ValueError):
                    ll_int = 0
                combined[uid] = {
                    "last_login_timestamp": ll_int,
                    "recommended_blog_ids_by_channel": raw if isinstance(raw, dict) else {},
                }

        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2, default=str)
        # 供监控在线指标（calculate_comment_source_mix / recommendation_coverage 等）读取
        self.data["user_recommended_blog_ids_by_channel"] = combined
        # 与快照同时刻的仿真时间（本轮 StartEvent.timestamp，尚未在 _save_step_data 末尾推进 current_timestamp）
        # 供 calculate_recommendation_coverage_login_validity 与 last_login_timestamp 对齐；勿用已推进后的 current_timestamp
        try:
            self.data["recommendation_snapshot_login_timestamp"] = int(
                self.data.get("current_timestamp", 0) or 0
            )
        except (TypeError, ValueError):
            self.data["recommendation_snapshot_login_timestamp"] = 0
        logger.info(
            f"Step {step_num}: Saved user recommended_blog_ids_by_channel snapshot to {snapshot_path} "
            f"({len(combined)} user(s))"
        )

    def _save_user_mentioned_blog_ids_by_channel_snapshot(self, step_num: int) -> None:
        """
        每轮遍历所有用户智能体，以用户 id 为 key，保存各用户 profile 中的 mentioned_blog_ids_by_channel。
        目录与 content_pool / user_recommended_blog_ids_by_channel 快照一致。
        """
        base_dir = getattr(self, "output_dir", None) or self.data.get("output_dir")
        if not isinstance(base_dir, str) or not base_dir.strip():
            base_dir = os.path.join(os.getcwd(), "runs_content_pool_snapshots")

        step_dir = os.path.join(base_dir, "datasets", f"step_{step_num}")
        os.makedirs(step_dir, exist_ok=True)
        snapshot_path = os.path.join(step_dir, "user_mentioned_blog_ids_by_channel.json")

        combined: Dict[str, Any] = {}
        agents_map = getattr(self, "agents", None) or {}
        user_map = agents_map.get("UserAgent", {}) if isinstance(agents_map, dict) else {}
        if isinstance(user_map, dict):
            for aid, agent in user_map.items():
                uid = str(aid).strip() if aid is not None else ""
                prof = getattr(agent, "profile", None)
                if prof is not None and not uid:
                    uid = str(prof.get_data("id", "") or "").strip()
                if not uid:
                    continue
                if prof is None:
                    combined[uid] = {"last_login_timestamp": 0, "mentioned_blog_ids_by_channel": {}}
                    continue
                raw = prof.get_data("mentioned_blog_ids_by_channel", {})
                ll = prof.get_data("last_login_timestamp", 0)
                try:
                    ll_int = int(ll) if ll is not None else 0
                except (TypeError, ValueError):
                    ll_int = 0
                combined[uid] = {
                    "last_login_timestamp": ll_int,
                    "mentioned_blog_ids_by_channel": raw if isinstance(raw, dict) else {},
                }

        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2, default=str)
        self.data["user_mentioned_blog_ids_by_channel"] = combined
        logger.info(
            f"Step {step_num}: Saved user mentioned_blog_ids_by_channel snapshot to {snapshot_path} "
            f"({len(combined)} user(s))"
        )

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
        按轮次递增分配时间步长（天），并保证 1..max_step 的总和等于 max_span_days。
        与 multi_channel_information_diffusion/code/SimEnv 一致：
        - power: w_i = i^p
        - sigmoid: w_i = eps + sigmoid((i-center)/scale)
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
        """
        重写 _save_step_data 方法，在每个轮次结束后更新环境数据
        
        当前逻辑：更新 content_pool 中每条微博的转发数；输出本轮转发总量；
        为每条微博写入 popularity（近 popularity_window_days 天、以 current_timestamp 为右端点的链式转发热度）；
        并维护 current_blogs 副本（供 StartEvent 等使用）。
        
        Args:
            step_num: 当前轮次编号
        """
        async with self._lock:
            self._save_user_recommended_blog_ids_by_channel_snapshot(step_num)
            self._save_user_mentioned_blog_ids_by_channel_snapshot(step_num)

        await super()._save_step_data(step_num)

        async with self._lock:
            # 临时调试：每轮输出 mention_pool 全量内容，便于排查提醒增删问题
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

            # 1. 更新 content_pool 中每条微博的转发数
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
                f"Step {step_num}: content_pool 转发总量（所有博客下转发条数之和）= {total_reposts}; "
                f"微博热度 popularity（近 {pop_days} 天内链式转发计入祖先，右端点 current_timestamp={ref_ts_sec}s）: "
                f"sum={pop_sum}, max_per_blog={pop_max}, blogs={len(pop_values)}"
            )
            self._save_content_pool_snapshot(step_num, content_pool)

            # 2. 更新 current_timestamp（Unix 秒，与微博 create_time 一致）
            current_timestamp = self.data.get("current_timestamp", 1764255440)
            if not isinstance(current_timestamp, (int, float)) or current_timestamp <= 0:
                logger.warning(f"Invalid current_timestamp: {current_timestamp}, skipping update")
                return
        
            # --- 时间步长调度（总跨度固定，前小后大）：本轮刚结束的是 step_num，应用第 step_num 段切片（勿用已 +1 的 current_step） ---
            # 可在 env_data.json 中配置这些参数
            # - timestamp_schedule_type: power | sigmoid
            # - timestamp_power_p: 幂函数指数（默认 1.6）
            # - timestamp_sigmoid_scale: sigmoid 平滑参数（默认 1.2）
            # - timestamp_sigmoid_center_ratio: sigmoid 拐点位置比例（默认 0.5）
            max_span_days = float(self.data.get("max_span_days", 24.0))
            max_step = self.data.get("max_step", 8)
            schedule_type = self.data.get("timestamp_schedule_type", "power")
            power_p = self.data.get("timestamp_power_p", 1.6)
            sigmoid_scale = self.data.get("timestamp_sigmoid_scale", 1.2)
            sigmoid_center_ratio = self.data.get("timestamp_sigmoid_center_ratio", 0.5)
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

            # 下一时间窗宽度（供下一轮 StartEvent）：第 step_num+1 段；已超过 max_step 则为 0
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
            # env 中 timestamp_duration 表示「当前时刻起，current_blogs 所覆盖的下一仿真时刻间距」
            self.data["timestamp_duration"] = dur_next_sec

            logger.info(
                f"Step {step_num}: step_delta_sec (completed slice)={step_delta_sec}, "
                f"dur_next_sec (next window for StartEvent)={dur_next_sec}"
            )
            logger.info(f"Step {step_num}: Updated current_timestamp from {current_timestamp} to {next_timestamp}")

            # 3. current_blogs：发帖时间 time < min(next_timestamp + dur_next_sec, cap_ts)
            lo = float(self.data["current_timestamp"])
            hi = min(lo + float(dur_next_sec), float(cap_ts))
            self.data["current_blogs"] = self._build_current_blogs_subset(content_pool, lo, hi)

            logger.info(f"Step {step_num}: Updated repost_count in content_pool, current_blogs count: {len(self.data['current_blogs'])}")

    async def _create_start_event(self, target_id: str) -> Event:
        # Extract relevant information from self.data according to StartEvent
        source_id = self.data.get('source_agent_id', 'default_source')
        timestamp = self.data.get('current_timestamp', 0)
        timestamp_duration = self.data.get('timestamp_duration', SEC_PER_DAY)

        current_step = self.data.get("current_step", 1)
        max_step = self.data.get("max_step", 8)
        logger.info(f"Step {current_step}/{max_step}: timestamp: {timestamp}, timestamp_duration: {timestamp_duration}")

        current_blogs = self.data.get('current_blogs', {})

        max_span_days = float(self.data.get("max_span_days", 24.0))
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
        
        # 将对应 UserAgent 的 login 置为 -1，供本轮是否参与决策使用
        if hasattr(self, 'agents') and self.agents:
            user_agents = self.agents.get('UserAgent', {})
            if isinstance(user_agents, dict) and target_id in user_agents:
                agent = user_agents[target_id]
                if getattr(agent, 'profile', None) is not None:
                    agent.profile.update_data('login', -1)
        
        # mention_pool 按 target_id 存；取当前目标对应的 mention 数据
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
        """
        将事件加入队列，在步骤结束时保存并广播d
        """
        if event_data['event_type'] in [
            'AddRepostEvent',
            'AddRepostResponseEvent',
            'MentionPoolUpdateEvent',
            'MentionPoolUpdateResponseEvent'
        ]:
            return
        await super().queue_event(event_data)

    async def add_repost(self, key: str, data: Any) -> Any:
        """
        添加转发到环境中的数据（使用分布式锁）
        
        Args:
            key: 转发ID
            data: 转发数据字典，必须包含 repost_id 字段
        """
        # 使用异步锁
        async with self._lock:
            current_pool = self.data.get("content_pool", {})
            if not isinstance(current_pool, dict):
                current_pool = {}
            blog_id = data.get("blog_id", "")
            if not blog_id:
                raise ValueError("Missing blog_id in repost data")
            if blog_id in current_pool:
                raise ValueError(f"blog_id {blog_id} already exists in content_pool")
            # 新增一条转发：以 blog_id 作为 key
            current_pool[blog_id] = dict(data)

            # 显式回写 content_pool
            self.data["content_pool"] = current_pool
            logger.debug(f"Added repost blog_id={blog_id} into content_pool")
            return True
            
    async def handle_add_repost_event(self, event: AddRepostEvent) -> None:
        """
        处理来自代理的添加评论事件（使用分布式锁）
        示例：
            {
                "blog_id": "123",
                "content": "转发内容",
                "time": 1764298329,
                "ip_location": "广东",
                "user_id": "5e2a573f0000000001002ddf",
                "nickname": "阿卷",
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

            # 在更新数据前获取锁
            async with lock:
                # 更新请求的数据
                success = await self.add_repost(event.key, event.value)

                # 创建并发送响应事件
                response_event = AddRepostResponseEvent(
                    from_agent_id=self.name,
                    to_agent_id=event.from_agent_id,
                    request_id=event.request_id,
                    key=event.key,
                    success=success
                )

                # 通过事件总线分发响应
                await self.event_bus.dispatch_event(response_event)

        except Exception as e:
            # 发送错误响应
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
        """
        更新共享数据（异步，使用锁）
        """
        # 使用异步锁
        async with self._lock:
            if "." in key:
                # 解析 blog_id 和字段路径
                parts = key.split(".")
                if parts[0] != "mention_pool" or len(parts) != 3:
                    raise ValueError(f"Invalid key: {key}, expected format: content_pool.mentioner_id.blog_id")
                mentioner_id = parts[1]     # 用户ID
                blog_id = parts[2]
                
                mention_pool = self.data.get("mention_pool", {})
                # 确保 current_pool 是字典格式
                if not isinstance(mention_pool, dict):
                    mention_pool = {}
                    logger.warning("mention_pool is not a dict, converting from list format")

                if mentioner_id not in mention_pool:
                    mention_pool[mentioner_id] = {}

                mentioner_pool = mention_pool[mentioner_id]

                # 删除操作
                if isinstance(data, dict) and data.get("action") == "delete":
                    if blog_id in mentioner_pool:
                        del mentioner_pool[blog_id]
                        logger.info(f"Deleted blog {blog_id} from mentioner {mentioner_id}")
                    else:
                        logger.warning(f"blog {blog_id} not found in mentioner {mentioner_id}, skip delete")
                # 新增操作
                elif isinstance(data, dict) and data.get("action") == "add":
                    if blog_id in mentioner_pool:
                        raise ValueError(f"blog {blog_id} already exists in mentioner {mentioner_id}")
                    mentioner_pool[blog_id] = data.get("mention_message", {})
                    logger.info(f"Added blog {blog_id} to mentioner {mentioner_id}")
              
                # 显式更新回 self.data（虽然引用会生效，但为了明确性）
                self.data["mention_pool"] = mention_pool
                return True
            else:
                raise ValueError(f"Invalid key: {key}, expected format: mention_pool.mentioner_id.blog_id")
            
    async def handle_update_mention_pool_event(self, event: MentionPoolUpdateEvent) -> None:
        """
        处理来自代理的更新mention_pool事件（使用分布式锁）
        示例：
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

            # 在更新数据前获取锁
            async with lock:
                # 更新请求的数据
                success = await self.update_mention_pool(event.key, event.value)

                # 创建并发送响应事件
                response_event = MentionPoolUpdateResponseEvent(
                    from_agent_id=self.name,
                    to_agent_id=event.from_agent_id,
                    request_id=event.request_id,
                    key=event.key,
                    success=success
                )

                # 通过事件总线分发响应
                await self.event_bus.dispatch_event(response_event)

        except Exception as e:
            # 发送错误响应
            error_response = MentionPoolUpdateResponseEvent(
                from_agent_id=self.name,
                to_agent_id=event.from_agent_id,
                request_id=event.request_id,
                key=event.key,
                success=False,
                error=str(e)
            )
            await self.event_bus.dispatch_event(error_response)