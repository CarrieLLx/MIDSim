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
        self.register_event("MentionPoolUpdateEvent", "handle_update_mention_pool_event")
        # 在这里添加你的自定义初始化逻辑
        # 例如：注册自定义事件、初始化自定义属性等
        # self.register_event("CustomEvent", "handle_custom_event")
        # self.custom_attribute = None

    async def load_initial_data(self) -> None:
        """在基类合并 env_data.json 之后，补全派生字段；current_notes 为发帖时间 time < min(current_timestamp+首轮 duration, cap) 的帖子。"""
        await super().load_initial_data()
        async with self._lock:
            self._populate_registered_user_agent_ids_for_metrics()
            # 1. 更新 content_pool 中每条帖子的评论数，以及每个评论的子评论数
            content_pool = self.data.get("content_pool", {})
            if not isinstance(content_pool, dict):
                logger.warning("load_initial_data: content_pool is not a dict, skip current_notes bootstrap")
                return

            self._normalize_content_pool_comment_meta(content_pool)

            # 2. 构建 current_notes：发帖时间 time < min(current_timestamp+首轮 duration, cap) 的帖子
            ts = self.data.get("current_timestamp", 1764255440000)
            if not isinstance(ts, (int, float)) or int(ts) <= 0:
                logger.warning(f"load_initial_data: invalid current_timestamp {ts}, set current_notes empty")
                self.data["current_notes"] = {}
                return

            max_span_days = float(self.data.get("max_span_days", 24.0))
            max_step = self.data.get("max_step", 8)
            schedule_type = self.data.get("timestamp_schedule_type", "power")
            power_p = self.data.get("timestamp_power_p", 1.6)
            sigmoid_scale = self.data.get("timestamp_sigmoid_scale", 1.2)
            sigmoid_center_ratio = self.data.get("timestamp_sigmoid_center_ratio", 0.5)
            day_ms = 86400000

            if not isinstance(self.data.get("simulation_start_timestamp"), (int, float)) or int(
                self.data.get("simulation_start_timestamp") or 0
            ) <= 0:
                self.data["simulation_start_timestamp"] = int(ts)

            td = self.data.get("timestamp_duration")
            if td is None or td == 0:
                # 仅补齐「第 1 段」日→毫秒，供首轮 StartEvent；不在此提前拨动 current_timestamp（仍用 JSON 里的起点）
                dur_days = self._timestamp_duration_days_for_step(
                    1,
                    max_step=max_step,
                    max_span_days=max_span_days,
                    schedule_type=schedule_type,
                    power_p=power_p,
                    sigmoid_scale=sigmoid_scale,
                    sigmoid_center_ratio=sigmoid_center_ratio,
                )
                self.data["timestamp_duration"] = int(dur_days * day_ms)

            start_ts = int(self.data["simulation_start_timestamp"])
            cap_ts = int(start_ts + max_span_days * day_ms)
            lo = float(self.data["current_timestamp"])
            dur_ms = int(self.data.get("timestamp_duration") or 0)
            hi = min(lo + dur_ms, float(cap_ts))

            logger.info(
                f"load_initial_data: current_timestamp={lo}, timestamp_duration(next window)={dur_ms}, "
                f"current_notes: time < {hi} ms (min(lo+duration, cap))"
            )

            self.data["current_notes"] = self._build_current_notes_subset(content_pool, lo, hi)
            logger.info(f"load_initial_data: built current_notes count={len(self.data['current_notes'])}")

    def _populate_registered_user_agent_ids_for_metrics(self) -> None:
        """供指标 comment_count_frequency：分母为全部 UserAgent（与 UserAgent.json 一致）。env 已含该字段则不覆盖。"""
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
            logger.info(
                f"SimEnv: 已写入 registered_user_agent_ids（{len(ids)} 个 UserAgent）供评论条数频率图分母使用"
            )

    def _normalize_content_pool_comment_meta(self, content_pool: Dict[str, Any]) -> None:
        """就地更新每条 note 的 comment_count 与每条评论的 sub_comment_count。"""
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
    def _is_note_time_before_hi(note: Dict[str, Any], _lo: float, hi: float) -> bool:
        """发帖时间 time 严格早于 hi（通常为 min(current_timestamp+timestamp_duration, cap)）。"""
        raw_time = note.get("time", None)
        try:
            note_time = float(raw_time)
        except (TypeError, ValueError):
            return False
        return note_time < hi

    def _note_copy_limited_comments(self, note: dict, top_k: int = 5, random_k: int = 0) -> dict:
        """截断评论视图，仅保留 top_k 条最高子评论，以及随机抽取 random_k 条子评论。"""
        base = {k: v for k, v in note.items() if k != "comments"}
        comments = note.get("comments") or {}
        if not isinstance(comments, dict) or not comments:
            base["comments"] = {}
            return base

        def comment_score(c: dict) -> float:
            """子评论计数（sub_comment_count）越高，得分越高。"""
            if not isinstance(c, dict):
                return 0.0
            sub = c.get("sub_comment_count", 0) or 0
            return float(sub)

        # 按子评论计数排序
        items = list(comments.items())
        random.shuffle(items)
        sorted_items = sorted(items, key=lambda x: comment_score(x[1]), reverse=True)
        kept_ids = {cid for cid, _ in sorted_items[:top_k]}  # 保留 top_k 条最高子评论

        # 随机抽取 random_k 条子评论
        rest = [(cid, _) for cid, _ in sorted_items[top_k:] if cid not in kept_ids]
        if rest and random_k > 0:
            n_rand = min(random_k, len(rest))
            for cid, _ in random.sample(rest, n_rand):
                kept_ids.add(cid)
        base["comments"] = {cid: comments[cid] for cid in kept_ids if cid in comments}
        return base

    def _build_current_notes_subset(
        self, content_pool: Dict[str, Any], lo_ts: float, hi_ts: float
    ) -> Dict[str, Any]:
        """发帖时间 time < hi_ts 的帖子 + 截断评论视图（hi_ts 已含 cap；lo_ts 仅用于零宽窗早退）。"""
        if hi_ts <= lo_ts:
            return {}
        return {
            nid: self._note_copy_limited_comments(note, top_k=5, random_k=0)
            for nid, note in content_pool.items()
            if isinstance(note, dict) and self._is_note_time_before_hi(note, lo_ts, hi_ts)
        }

    @staticmethod
    def _total_comments_in_content_pool(content_pool: Dict[str, Any]) -> int:
        """统计 content_pool 中所有帖子的评论条数之和（每个 note 的 comments 字典长度）。"""
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
        """解析帖子 ``time`` 或评论 ``timestamp`` 为毫秒（与笔记发帖时间规则一致：秒级乘 1000）。"""
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
        """
        以 ref_ts_ms 为时间轴右端点，统计每条帖子在 [ref - window, ref] 内的评论条数，
        写入 note['popularity'] 作为热度（默认窗口 7 天）。
        """
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

    def _save_current_notes_snapshot(self, step_num: int, current_notes: Dict[str, Any]) -> None:
        """保存本轮对应的 current_notes（note_id 子集），供离线指标判断帖子是否在当前可见窗内。"""
        if not isinstance(current_notes, dict):
            return
        base_dir = getattr(self, "output_dir", None) or self.data.get("output_dir")
        if not isinstance(base_dir, str) or not base_dir.strip():
            base_dir = os.path.join(os.getcwd(), "runs_content_pool_snapshots")

        step_dir = os.path.join(base_dir, "datasets", f"step_{step_num}")
        os.makedirs(step_dir, exist_ok=True)
        snapshot_path = os.path.join(step_dir, "current_notes_snapshot.json")
        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(current_notes, f, ensure_ascii=False, indent=2, default=str)
        logger.info(
            f"Step {step_num}: Saved current_notes snapshot ({len(current_notes)} notes) to {snapshot_path}"
        )

    def _save_step_metadata_snapshot(
        self, step_num: int, current_timestamp: int, timestamp_duration: int
    ) -> None:
        """
        保存本轮 StartEvent 对应的仿真时刻（写入 profile 前与 env 一致），供离线指标与 last_login_timestamp 对齐。
        """
        base_dir = getattr(self, "output_dir", None) or self.data.get("output_dir")
        if not isinstance(base_dir, str) or not base_dir.strip():
            base_dir = os.path.join(os.getcwd(), "runs_content_pool_snapshots")

        step_dir = os.path.join(base_dir, "datasets", f"step_{step_num}")
        os.makedirs(step_dir, exist_ok=True)
        snapshot_path = os.path.join(step_dir, "step_metadata.json")
        payload = {
            "step_num": int(step_num),
            "current_timestamp": int(current_timestamp),
            "timestamp_duration": int(timestamp_duration),
        }
        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        logger.info(
            f"Step {step_num}: Saved step_metadata (current_timestamp={current_timestamp}) to {snapshot_path}"
        )

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
                raw = prof.get_data("recommended_note_ids_by_channel", {})
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
                raw = prof.get_data("mentioned_note_ids_by_channel", {})
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
        
        当前逻辑：更新 content_pool 中每条帖子的评论数；输出本轮评论总量；
        为每条帖子写入 popularity（近 popularity_window_days 天、以 current_timestamp 为右端点的评论数）；
        并维护 current_notes 副本（供 StartEvent 等使用）。
        
        Args:
            step_num: 当前轮次编号
        """
        # 必须先于 BasicSimEnv._save_step_data 内的 collect_metrics：
        # 在线指标从 env.data 读取 user_recommended_note_ids_by_channel，若写在 collect_metrics 之后会全程为 None。
        async with self._lock:
            self._save_user_recommended_note_ids_by_channel_snapshot(step_num)
            self._save_user_mentioned_note_ids_by_channel_snapshot(step_num)

        await super()._save_step_data(step_num)

        async with self._lock:
            # 本轮结束时仍生效的 current_notes（时间窗尚未在下方推进），供离线指标与 current_notes 对齐
            current_notes_for_metric = self.data.get("current_notes", {})
            if isinstance(current_notes_for_metric, dict):
                self._save_current_notes_snapshot(step_num, current_notes_for_metric)

            # 与本轮 StartEvent.timestamp 对齐（尚未在下方推进 current_timestamp）
            try:
                _cts = int(self.data.get("current_timestamp", 0) or 0)
            except (TypeError, ValueError):
                _cts = 0
            try:
                _ctd = int(self.data.get("timestamp_duration", 0) or 0)
            except (TypeError, ValueError):
                _ctd = 0
            self._save_step_metadata_snapshot(step_num, _cts, _ctd)

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

            # 1. 更新 content_pool 中每条帖子的评论数，以及每个评论的子评论数
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
                f"Step {step_num}: content_pool 评论总量（所有帖子下评论条数之和）= {total_comments}; "
                f"帖子热度 popularity（近 {pop_days} 天内每条帖子下评论数，右端点 current_timestamp={ref_ts_ms}）: "
                f"sum={pop_sum}, max_per_note={pop_max}, notes={len(pop_values)}"
            )
            self._save_content_pool_snapshot(step_num, content_pool)

            # 2. 更新 current_timestamp
            current_timestamp = self.data.get("current_timestamp", 1764255440000)
            if not isinstance(current_timestamp, (int, float)) or current_timestamp <= 0:
                logger.warning(f"Invalid current_timestamp: {current_timestamp}, skipping update")
                return
            # --- 时间步长调度（总跨度固定，前小后大）：本轮刚结束的是 step_num，应用第 step_num 段切片（勿用已 +1 的 current_step） ---
            day_ms = 86400000
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

            # 下一时间窗宽度（供下一轮 StartEvent）：第 step_num+1 段；已超过 max_step 则为 0
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
            # env 中 timestamp_duration 表示「当前时刻起，current_notes 所覆盖的下一仿真时刻间距」
            self.data["timestamp_duration"] = dur_next_ms

            logger.info(
                f"Step {step_num}: step_delta_ms (completed slice)={step_delta_ms}, "
                f"dur_next_ms (next window for StartEvent)={dur_next_ms}"
            )
            logger.info(f"Step {step_num}: Updated current_timestamp from {current_timestamp} to {next_timestamp}")

            # 3. current_notes：发帖时间 time < min(next_timestamp + dur_next_ms, cap_ts)
            lo = float(self.data["current_timestamp"])
            hi = min(lo + float(dur_next_ms), float(cap_ts))
            self.data["current_notes"] = self._build_current_notes_subset(content_pool, lo, hi)

            logger.info(f"Step {step_num}: Updated comment_count in content_pool, current_notes count: {len(self.data['current_notes'])}")

    async def _create_start_event(self, target_id: str) -> Event:
        # Extract relevant information from self.data according to StartEvent
        source_id = self.data.get('source_agent_id', 'default_source')
        timestamp = self.data.get('current_timestamp', 0)
        timestamp_duration = self.data.get('timestamp_duration', 86400000)

        current_step = self.data.get("current_step", 1)
        max_step = self.data.get("max_step", 8)
        logger.info(f"Step {current_step}/{max_step}: timestamp: {timestamp}, timestamp_duration: {timestamp_duration}")

        current_notes = self.data.get('current_notes', {})

        day_ms = 86400000
        max_span_days = float(self.data.get("max_span_days", 24.0))
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
        """
        将事件加入队列，在步骤结束时保存并广播
        """
        if event_data['event_type'] in [
            'AddCommentEvent',
            'AddCommentResponseEvent',
            'MentionPoolUpdateEvent',
            'MentionPoolUpdateResponseEvent',
        ]:
            return
        await super().queue_event(event_data)

    async def add_comment(self, key: str, data: Any) -> Any:
        """
        更新共享数据（异步，使用锁）
        """
        # 使用异步锁
        async with self._lock:
            if "." in key:
                # 解析 note_id 和字段路径
                parts = key.split(".")  
                if parts[0] != "content_pool" or parts[2] != "comments" or len(parts) != 3:
                    raise ValueError(f"Invalid key: {key}, expected format: content_pool.note_id.comments")
                note_id = parts[1]     # 笔记ID

                current_pool = self.data.get("content_pool", {})
                # 确保 current_pool 是字典格式
                if not isinstance(current_pool, dict):
                    current_pool = {}
                    logger.warning("content_pool is not a dict, converting from list format")
            
                if note_id not in current_pool:
                    raise ValueError(f"note_id {note_id} not found in content_pool")
                note = current_pool[note_id]
                
                # 确保 comments 是字典格式
                if "comments" not in note:
                    note["comments"] = {}
                if not isinstance(note["comments"], dict):
                    note["comments"] = {}
                
                if not isinstance(data, dict):
                    raise ValueError(f"data must be a dict: {data}")
                
                # 如果 data 包含 comment_id，使用 comment_id 作为键
                # 否则直接合并整个 data 字典
                if "comment_id" in data:
                    comment_id = data["comment_id"]
                    # 使用 comment_id 作为键，合并评论数据
                    note["comments"][comment_id] = {**note["comments"].get(comment_id, {}), **data}
                else:
                    # 如果没有 comment_id，直接合并整个 data 字典到 comments
                    note["comments"] = {**note["comments"], **data}
                
                # 显式更新回 self.data（虽然引用会生效，但为了明确性）
                self.data["content_pool"] = current_pool
                
                logger.debug(f"Updated comments in {key}")
                return True
            else:
                raise ValueError(f"Invalid key: {key}, expected format: content_pool.note_id.comments")
            
    async def handle_add_comment_event(self, event: AddCommentEvent) -> None:
        """
        处理来自代理的添加评论事件（使用分布式锁）
        示例：
            {
                "comment_id": "123",
                "create_time": 1764298329000,
                "ip_location": "广东",
                "note_id": "69290e59000000001e034ab4",
                "user_id": "5e2a573f0000000001002ddf",
                "nickname": "阿卷",
                "parent_comment_id": None,
                "at_count": 0,
                "content": "评论1"
            }
        """
        try:
            logger.info(f"Received AddCommentEvent, request_id={event.request_id}, key={event.key}")
            lock_id = f"env_comment_add_lock_{event.key}"
            lock = await get_lock(lock_id)

            # 在更新数据前获取锁
            async with lock:
                # 更新请求的数据
                success = await self.add_comment(event.key, event.value)

                # 创建并发送响应事件
                response_event = AddCommentResponseEvent(
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
        """
        更新共享数据（异步，使用锁）
        """
        # 使用异步锁
        async with self._lock:
            if "." in key:
                # 解析 note_id 和字段路径
                parts = key.split(".")
                if parts[0] != "mention_pool" or len(parts) != 3:
                    raise ValueError(f"Invalid key: {key}, expected format: content_pool.note_id.comments")
                mentioner_id = parts[1]     # 笔记ID
                comment_id = parts[2]
                
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
                    if comment_id in mentioner_pool:
                        del mentioner_pool[comment_id]
                        logger.info(f"Deleted comment {comment_id} from mentioner {mentioner_id}")
                    else:
                        logger.warning(f"comment {comment_id} not found in mentioner {mentioner_id}, skip delete")
                # 新增操作
                elif isinstance(data, dict) and data.get("action") == "add":
                    if comment_id in mentioner_pool:
                        raise ValueError(f"comment {comment_id} already exists in mentioner {mentioner_id}")
                    mentioner_pool[comment_id] = data.get("mention_message", {})
                    logger.info(f"Added comment {comment_id} to mentioner {mentioner_id}")
              
                # 显式更新回 self.data（虽然引用会生效，但为了明确性）
                self.data["mention_pool"] = mention_pool
                return True
            else:
                raise ValueError(f"Invalid key: {key}, expected format: mention_pool.mentioner_id.comment_id")
            
    async def handle_update_mention_pool_event(self, event: MentionPoolUpdateEvent) -> None:
        """
        处理来自代理的更新mention_pool事件（使用分布式锁）
        示例：
            {
                "action": "add",
                "mention_message": {
                    "note_id": "69290e59000000001e034ab4",
                    "comment_id": 1764298329000,
                    "comment_content": "评论1",
                    "mentioner_id": "69290e59000000001e034ab4",
                    "mentioner_nickname": "阿卷",
                    "mention_type": "reply"
                }
            }
        """
        try:
            logger.info(f"Received MentionPoolUpdateEvent, request_id={event.request_id}, key={event.key}")
            lock_id = f"env_mention_pool_update_lock_{event.key}"
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