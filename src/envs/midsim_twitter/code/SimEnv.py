from onesim.simulator import BasicSimEnv
from onesim.events import Event
from onesim.events import DataUpdateEvent, DataUpdateResponseEvent
from onesim.distribution.distributed_lock import get_lock 
from typing import Any, Optional, Dict
from loguru import logger
import asyncio
import json
import os
import math
import random
from .events import StartEvent, AddTweetEvent, AddTweetResponseEvent, MentionPoolUpdateEvent, MentionPoolUpdateResponseEvent

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
        self.register_event("AddCommentEvent", "handle_add_comment_event")
        self.register_event("AddTweetEvent", "handle_add_tweet_event")
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
        """content_pool[*].time 统一为与推特一致的 Unix 秒（int 或数字字符串）；兼容毫秒。"""
        for tweet in content_pool.values():
            if not isinstance(tweet, dict):
                continue
            raw = tweet.get("time")
            if raw is None or raw == "":
                continue
            try:
                v = float(raw)
            except (TypeError, ValueError):
                continue
            if v >= 10**12:
                v = v / 1000.0
            tweet["time"] = int(v)

    async def load_initial_data(self) -> None:
        """
        加载 env_data.json 后，冻结初始 content_pool 的 tweet id 列表为 seed_root_tweet_ids，
        供指标「Repost Count Frequency」将传播链归因到环境种子根推。
        若 env_data.json 已显式提供 seed_root_tweet_ids，则不再覆盖。
        """
        await super().load_initial_data()
        async with self._lock:
            if self.data.get("seed_root_tweet_ids") is not None:
                return
            cp = self.data.get("content_pool")
            if isinstance(cp, dict) and cp:
                self.data["seed_root_tweet_ids"] = [str(k) for k in cp.keys()]

            self._normalize_sim_timestamps_to_seconds()

            # 1. 更新 content_pool 中每条推特的转发数
            content_pool = self.data.get("content_pool", {})
            if not isinstance(content_pool, dict):
                logger.warning("load_initial_data: content_pool is not a dict, skip current_tweets bootstrap")
                return

            self._normalize_content_pool_times_to_seconds(content_pool)
            self._normalize_content_pool_propagation_meta(content_pool)

            # 2. 构建 current_tweets：发帖时间 time < min(current_timestamp+首轮 duration, cap) 的微博
            ts = self.data.get("current_timestamp", 1764255440)
            if not isinstance(ts, (int, float)) or int(ts) <= 0:
                logger.warning(f"load_initial_data: invalid current_timestamp {ts}, set current_tweets empty")
                self.data["current_tweets"] = {}
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
                f"current_tweets: time < {hi} s (min(lo+duration, cap))"
            )

            self.data["current_tweets"] = self._build_current_tweets_subset(content_pool, lo, hi)
            logger.info(f"load_initial_data: built current_tweets count={len(self.data['current_tweets'])}")

    def _clear_content_pool_propagation_aggregates(self, content_pool: Dict[str, Any]) -> None:
        """每轮归一化前清空回复/转推/引用的聚合字段，避免重复累加。

        须重置 count 与 *_ids，否则每轮 _normalize 会在旧值上再次累加、列表重复 append。
        """
        for tw in content_pool.values():
            if not isinstance(tw, dict):
                continue
            tw["replies"] = {}
            tw["reply_count"] = 0
            tw["reply_ids"] = []
            tw["retweet_count"] = 0
            tw["retweet_ids"] = []
            tw["quote_count"] = 0
            tw["quote_ids"] = []

    def _normalize_content_pool_propagation_meta(self, content_pool: Dict[str, Any]) -> None:
        """就地更新每条推文的 quote_count，reply_count，retweet_count。"""
        self._clear_content_pool_propagation_aggregates(content_pool)
        for tweet_id, tweet in content_pool.items():
            if not isinstance(tweet, dict):
                continue
            replied_tweet_id = tweet.get("replied_tweet_id") or tweet.get("replyed_tweet_id") or ""
            if replied_tweet_id:
                if replied_tweet_id in content_pool:
                    content_pool[replied_tweet_id]["reply_count"] = content_pool[replied_tweet_id].get("reply_count", 0) + 1
                    content_pool[replied_tweet_id]["reply_ids"] = content_pool[replied_tweet_id].get("reply_ids", [])
                    content_pool[replied_tweet_id]["reply_ids"].append(tweet_id)
                    content_pool[replied_tweet_id]["replies"][tweet_id] = tweet

            retweeted_tweet_id = tweet.get("retweeted_tweet_id", "")
            if retweeted_tweet_id:
                if retweeted_tweet_id in content_pool:
                    content_pool[retweeted_tweet_id]["retweet_count"] = content_pool[retweeted_tweet_id].get("retweet_count", 0) + 1
                    content_pool[retweeted_tweet_id]["retweet_ids"] = content_pool[retweeted_tweet_id].get("retweet_ids", [])
                    content_pool[retweeted_tweet_id]["retweet_ids"].append(tweet_id)

            quoted_tweet_id = tweet.get("quoted_tweet_id", "")
            if quoted_tweet_id:
                if quoted_tweet_id in content_pool:
                    content_pool[quoted_tweet_id]["quote_count"] = content_pool[quoted_tweet_id].get("quote_count", 0) + 1
                    content_pool[quoted_tweet_id]["quote_ids"] = content_pool[quoted_tweet_id].get("quote_ids", [])
                    content_pool[quoted_tweet_id]["quote_ids"].append(tweet_id)

    @staticmethod
    def _is_tweet_time_before_hi(tweet: Dict[str, Any], _lo: float, hi: float) -> bool:
        """发帖时间 time 严格早于 hi（秒；兼容旧版毫秒 time）。"""
        raw_time = tweet.get("time", None)
        try:
            tweet_time = float(raw_time)
        except (TypeError, ValueError):
            return False
        if tweet_time >= 10**12:
            tweet_time = tweet_time / 1000.0
        return tweet_time < hi

    def _tweet_copy_limited_replies(self, tweet: dict, top_k: int = 5, random_k: int = 0) -> dict:
        """截断回复视图，仅保留 top_k 条最高子回复，以及随机抽取 random_k 条子评论。"""
        base = {k: v for k, v in tweet.items() if k != "replies"}
        replies = tweet.get("replies") or {}
        if not isinstance(replies, dict) or not replies:
            base["replies"] = {}
            return base

        def reply_score(r: dict) -> float:
            """回复计数（reply_count + quote_count + retweet_count）越高，得分越高。"""
            if not isinstance(r, dict):
                return 0.0
            reply_count = r.get("reply_count", 0) or 0
            quote_count = r.get("quote_count", 0) or 0
            retweet_count = r.get("retweet_count", 0) or 0
            sub = reply_count + quote_count + retweet_count
            return float(sub)

        # 按回复计数排序
        items = list(replies.items())
        random.shuffle(items)
        sorted_items = sorted(items, key=lambda x: reply_score(x[1]), reverse=True)
        kept_ids = {rid for rid, _ in sorted_items[:top_k]}  # 保留 top_k 条最高回复

        # 随机抽取 random_k 条子评论
        rest = [(rid, _) for rid, _ in sorted_items[top_k:] if rid not in kept_ids]
        if rest and random_k > 0:
            n_rand = min(random_k, len(rest))
            for rid, _ in random.sample(rest, n_rand):
                kept_ids.add(rid)
        base["replies"] = {rid: replies[rid] for rid in kept_ids if rid in replies}
        return base

    def _build_current_tweets_subset(
        self, content_pool: Dict[str, Any], lo_ts: float, hi_ts: float
    ) -> Dict[str, Any]:
        """发帖时间 time < hi_ts 的推文 + 截断转发视图（hi_ts 已含 cap；lo_ts 仅用于零宽窗早退）。"""
        if hi_ts <= lo_ts:
            return {}
        return {
            tid: self._tweet_copy_limited_replies(tweet, top_k=5, random_k=0)
            for tid, tweet in content_pool.items()
            if isinstance(tweet, dict) and self._is_tweet_time_before_hi(tweet, lo_ts, hi_ts)
        }
        
    @staticmethod
    def _total_propagation_in_content_pool(content_pool: Dict[str, Any]) -> int:
        """统计 content_pool 中所有帖子的回复数、转推数、引用数之和。"""
        total = 0
        for tweet in content_pool.values():
            if not isinstance(tweet, dict):
                continue
            total += tweet.get("reply_count", 0) + tweet.get("retweet_count", 0) + tweet.get("quote_count", 0)
        return total

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

    @staticmethod
    def _profile_recommended_by_channel_for_snapshot(prof: Any) -> Dict[str, Any]:
        """
        Twitter 场景 UserAgent 写入 recommended_tweet_ids_by_channel；
        与小红书等场景使用的 recommended_note_ids_by_channel 二选一或兼容读取。
        """
        if prof is None:
            return {}
        tw = prof.get_data("recommended_tweet_ids_by_channel", None)
        nt = prof.get_data("recommended_note_ids_by_channel", None)
        if isinstance(tw, dict) and tw:
            return dict(tw)
        if isinstance(nt, dict) and nt:
            return dict(nt)
        if isinstance(tw, dict):
            return dict(tw)
        if isinstance(nt, dict):
            return dict(nt)
        return {}

    @staticmethod
    def _profile_mentioned_by_channel_for_snapshot(prof: Any) -> Dict[str, Any]:
        """
        Twitter 场景 UserAgent 写入 mentioned_tweet_ids_by_channel；
        兼容 mentioned_note_ids_by_channel。
        """
        if prof is None:
            return {}
        tw = prof.get_data("mentioned_tweet_ids_by_channel", None)
        nt = prof.get_data("mentioned_note_ids_by_channel", None)
        if isinstance(tw, dict) and tw:
            return dict(tw)
        if isinstance(nt, dict) and nt:
            return dict(nt)
        if isinstance(tw, dict):
            return dict(tw)
        if isinstance(nt, dict):
            return dict(nt)
        return {}

    def _save_user_recommended_note_ids_by_channel_snapshot(self, step_num: int) -> None:
        """
        每轮遍历所有用户智能体，以用户 id 为 key，保存各用户 profile 中的 recommended_note_ids_by_channel。
        目录与 content_pool 快照一致：{output_dir}/datasets/step_{step_num}/
        """
        base_dir = getattr(self, "output_dir", None) or self.data.get("output_dir")
        if not isinstance(base_dir, str) or not base_dir.strip():
            base_dir = os.path.join(os.getcwd(), "runs_content_pool_snapshots")

        step_dir = os.path.join(base_dir, "datasets", f"step_{step_num}")
        os.makedirs(step_dir, exist_ok=True)
        snapshot_path = os.path.join(step_dir, "user_recommended_note_ids_by_channel.json")

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
                    combined[uid] = {"last_login_timestamp": 0, "recommended_note_ids_by_channel": {}}
                    continue
                raw = self._profile_recommended_by_channel_for_snapshot(prof)
                ll = prof.get_data("last_login_timestamp", 0)
                try:
                    ll_int = int(ll) if ll is not None else 0
                except (TypeError, ValueError):
                    ll_int = 0
                combined[uid] = {
                    "last_login_timestamp": ll_int,
                    "recommended_note_ids_by_channel": raw if isinstance(raw, dict) else {},
                }

        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2, default=str)
        # 供监控在线指标（calculate_comment_source_mix / recommendation_coverage 等）读取
        self.data["user_recommended_note_ids_by_channel"] = combined
        # 与快照同时刻的仿真时间（本轮 StartEvent.timestamp，尚未在 _save_step_data 末尾推进 current_timestamp）
        # 供 calculate_recommendation_coverage_login_validity 与 last_login_timestamp 对齐；勿用已推进后的 current_timestamp
        try:
            self.data["recommendation_snapshot_login_timestamp"] = int(
                self.data.get("current_timestamp", 0) or 0
            )
        except (TypeError, ValueError):
            self.data["recommendation_snapshot_login_timestamp"] = 0
        logger.info(
            f"Step {step_num}: Saved user recommended_note_ids_by_channel snapshot to {snapshot_path} "
            f"({len(combined)} user(s))"
        )

    def _save_user_mentioned_note_ids_by_channel_snapshot(self, step_num: int) -> None:
        """
        每轮遍历所有用户智能体，以用户 id 为 key，保存各用户 profile 中的 mentioned_note_ids_by_channel。
        目录与 content_pool / user_recommended_note_ids_by_channel 快照一致。
        """
        base_dir = getattr(self, "output_dir", None) or self.data.get("output_dir")
        if not isinstance(base_dir, str) or not base_dir.strip():
            base_dir = os.path.join(os.getcwd(), "runs_content_pool_snapshots")

        step_dir = os.path.join(base_dir, "datasets", f"step_{step_num}")
        os.makedirs(step_dir, exist_ok=True)
        snapshot_path = os.path.join(step_dir, "user_mentioned_note_ids_by_channel.json")

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
                    combined[uid] = {"last_login_timestamp": 0, "mentioned_note_ids_by_channel": {}}
                    continue
                raw = self._profile_mentioned_by_channel_for_snapshot(prof)
                ll = prof.get_data("last_login_timestamp", 0)
                try:
                    ll_int = int(ll) if ll is not None else 0
                except (TypeError, ValueError):
                    ll_int = 0
                combined[uid] = {
                    "last_login_timestamp": ll_int,
                    "mentioned_note_ids_by_channel": raw if isinstance(raw, dict) else {},
                }

        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2, default=str)
        self.data["user_mentioned_note_ids_by_channel"] = combined
        logger.info(
            f"Step {step_num}: Saved user mentioned_note_ids_by_channel snapshot to {snapshot_path} "
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
        支持两种策略：
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
            # scale 越小曲线越陡；center_ratio 控制拐点位置（0~1）
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
            # 默认 power 策略
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
        
        当前逻辑：更新 content_pool 中每条推文的回复数、转推数、引用数；输出本轮 content_pool 传播总量；
        并维护 current_tweets 副本（供 StartEvent 等使用）。
        
        Args:
            step_num: 当前轮次编号
        """
        async with self._lock:
            self._save_user_recommended_note_ids_by_channel_snapshot(step_num)
            self._save_user_mentioned_note_ids_by_channel_snapshot(step_num)

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
            self._normalize_content_pool_propagation_meta(content_pool)
                    
            total_propagation = self._total_propagation_in_content_pool(content_pool)
            logger.info(
                f"Step {step_num}: content_pool 传播总量（所有帖子下回复数、转推数、引用数之和）= {total_propagation}"
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
            # env 中 timestamp_duration 表示「当前时刻起，current_notes 所覆盖的下一仿真时刻间距」
            self.data["timestamp_duration"] = dur_next_sec

            logger.info(
                f"Step {step_num}: step_delta_sec (completed slice)={step_delta_sec}, "
                f"dur_next_sec (next window for StartEvent)={dur_next_sec}"
            )
            logger.info(f"Step {step_num}: Updated current_timestamp from {current_timestamp} to {next_timestamp}")

            # 3. 保留一份 content_pool 副本（存为 current_tweets）：每条 tweet 只保留回复数最多的 2 条回复 + 3 条随机回复
            # 当前时间步之前的笔记
            lo = float(self.data["current_timestamp"])
            hi = min(lo + float(dur_next_sec), float(cap_ts))
            self.data["current_tweets"] = self._build_current_tweets_subset(content_pool, lo, hi)

            logger.info(f"Step {step_num}: Updated reply_count, retweet_count, quote_count in content_pool, current_tweets count: {len(self.data['current_tweets'])}")

    async def _create_start_event(self, target_id: str) -> Event:
        # Extract relevant information from self.data according to StartEvent
        source_id = self.data.get('source_agent_id', 'default_source')
        timestamp = self.data.get('current_timestamp', 0)
        timestamp_duration = self.data.get('timestamp_duration', 86400000)

        current_step = self.data.get("current_step", 1)
        max_step = self.data.get("max_step", 8)
        logger.info(f"Step {current_step}/{max_step}: timestamp: {timestamp}, timestamp_duration: {timestamp_duration}")

        current_tweets = self.data.get('current_tweets', {})

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
                current_tweets=current_tweets
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
        for tweet_id, mention_message in mention_pool.items():
            if tweet_id in current_tweets:
                mentions[tweet_id] = {
                    "tweet": current_tweets[tweet_id],
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
            current_tweets=current_tweets, 
            mentions=mentions
        )

    async def queue_event(self, event_data: Dict[str, Any]):
        """
        将事件加入队列，在步骤结束时保存并广播
        """
        if event_data['event_type'] in [
            'AddTweetEvent',
            'AddTweetResponseEvent',
            'MentionPoolUpdateEvent',
            'MentionPoolUpdateResponseEvent',
        ]:
            return
        await super().queue_event(event_data)

    async def add_tweet(self, key: str, data: Any) -> Any:
        """
        更新共享数据（异步，使用锁）
        """
        # 使用异步锁
        async with self._lock:
            content_pool = self.data.get("content_pool", {})
            if not isinstance(content_pool, dict):
                content_pool = {}
                logger.warning("content_pool is not a dict, converting from list format")

            if key in content_pool:
                raise ValueError(f"tweet_id {key} already exists in content_pool")

            content_pool[key] = data
            self.data["content_pool"] = content_pool
            return True
            
    async def handle_add_tweet_event(self, event: AddTweetEvent) -> None:
        """
        处理来自代理的添加推文事件（使用分布式锁）
        示例：
            {
                "tweet_id": "123",
                "content_id": "123",
                "time": 1764298329000,
                "user_id": "5e2a573f0000000001002ddf",
                "nickname": "阿卷",
                "username": "ajuan",
                "reply_count": 0,
                "retweet_count": 0,
                "quote_count": 0,
                "replied_tweet_id": None,
                "retweeted_tweet_id": None,
                "quoted_tweet_id": None
            }
        """
        try:
            logger.info(f"Received AddTweetEvent, request_id={event.request_id}, key={event.key}")
            lock_id = f"env_tweet_add_lock_content_pool"
            lock = await get_lock(lock_id)

            async with lock:
                success = await self.add_tweet(event.key, event.value)

                response_event = AddTweetResponseEvent(
                    from_agent_id=self.name,
                    to_agent_id=event.from_agent_id,
                    request_id=event.request_id,
                    key=event.key,
                    success=success,
                )

                await self.event_bus.dispatch_event(response_event)

        except Exception as e:
            error_response = AddTweetResponseEvent(
                from_agent_id=self.name,
                to_agent_id=event.from_agent_id,
                request_id=event.request_id,
                key=event.key,
                success=False,
                error=str(e),
            )
            await self.event_bus.dispatch_event(error_response)

    async def update_mention_pool(self, key: str, data: Any) -> Any:
        """
        更新共享数据（异步，使用锁）
        """
        # 使用异步锁
        async with self._lock:
            if "." in key:
                # 解析 mentioner_id 和字段路径
                parts = key.split(".")
                if parts[0] != "mention_pool" or len(parts) != 3:
                    raise ValueError(f"Invalid key: {key}, expected format: mention_pool.mentioner_id.tweet_id")
                mentioner_id = parts[1]     # mentioner_id
                tweet_id = parts[2]
                
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
                    if tweet_id in mentioner_pool:
                        del mentioner_pool[tweet_id]
                        logger.info(f"Deleted tweet {tweet_id} from mentioner {mentioner_id}")
                    else:
                        logger.warning(f"tweet {tweet_id} not found in mentioner {mentioner_id}, skip delete")
                # 新增操作
                elif isinstance(data, dict) and data.get("action") == "add":
                    if tweet_id in mentioner_pool:
                        raise ValueError(f"tweet {tweet_id} already exists in mentioner {mentioner_id}")
                    mentioner_pool[tweet_id] = data.get("mention_message", {})
                    logger.info(f"Added tweet {tweet_id} to mentioner {mentioner_id}")
              
                # 显式更新回 self.data（虽然引用会生效，但为了明确性）
                self.data["mention_pool"] = mention_pool
                return True
            else:
                raise ValueError(f"Invalid key: {key}, expected format: mention_pool.mentioner_id.tweet_id")
            
    async def handle_update_mention_pool_event(self, event: MentionPoolUpdateEvent) -> None:
        """
        处理来自代理的更新mention_pool事件（使用分布式锁）
        示例：
            {
                "action": "add",
                "mention_message": {
                    "tweet_id": "69290e59000000001e034ab4",
                    "mention_type": "reply"
                }
            }
        """
        try:
            logger.info(f"Received MentionPoolUpdateEvent, request_id={event.request_id}, key={event.key}")
            # key 形如 mention_pool.{mentioner_id}.{tweet_id}，与 UserAgent 侧 lock 的 mentioner_id 对齐
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