from collections import defaultdict
from typing import Any, List, Optional, Dict, Set, Tuple, Union
import json
import asyncio
import os
import re
import secrets
import time
from datetime import datetime, timezone

# SimEnv 对 mention_pool / 评论等更新使用全局锁串行化，高并发下排队可能超过 30s
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
from .events import *
import random
import math

from .InteractionThreshold import InteractionThreshold
from .step15_topic_gate import (
    load_step15_gate_config,
    should_inject_step15,
    evaluate_step15_policies as _gate_evaluate_step15_policies,
    topic_text_from_mention_entries,
    topic_text_from_notes_chunk,
)


class UserAgent(GeneralAgent):
    """
    依赖环境侧在加载 env_data.json 后提供一致的派生状态：
    `current_notes`、`comment_count` / `sub_comment_count` 等由本场景的 SimEnv.load_initial_data
    与每轮 _save_step_data 维护；UserAgent 只消费 StartEvent.current_notes。
    """

    def __init__(self,
                 sys_prompt: str | None = None,
                 model_config_name: str = None,
                 event_bus_queue: asyncio.Queue = None,
                 profile: AgentProfile=None,
                 memory: MemoryStrategy=None,
                 planning: PlanningBase=None,
                 relationship_manager: RelationshipManager=None) -> None:
        super().__init__(sys_prompt, model_config_name, event_bus_queue, profile, memory, planning, relationship_manager)
        self.register_event("StartEvent", "get_recommendations_and_mentions")  
        self.register_event("StartEvent", "generate_memory_from_own_notes")
        self.register_event("SocialRecommendationEvent", "receive_recommendation")
        self.register_event("AlgorithmRecommendationEvent", "receive_recommendation")
        self.register_event("SearchRecommendationEvent", "receive_recommendation")
        self.register_event("KeepFollowingEvent", "receive_recommendation")
        self.register_event("MentionEvent", "handle_mention")

        self.register_event("AddCommentResponseEvent", "handle_add_comment_response")
        self.register_event("MentionPoolUpdateResponseEvent", "handle_update_mention_pool_response")
        self._comment_add_futures: Dict[str, Future] = {}
        self._mention_pool_update_futures: Dict[str, Future] = {}
        self._login_lock = asyncio.Lock()  # 单个智能体级别的锁，用于本轮是否参与的决策

        # 系统内“算法类型 -> 推荐器智能体ID”显式映射（按环境约定维护）
        self.recommender_map: Dict[str, str] = {
            "Random Recommendation": "recomment_agent_0001",
            "Hot Recommendation": "recomment_agent_0002",
            "Interest Recommendation": "recomment_agent_0003",
        }
        # 用户侧默认请求的算法类型列表（由代码固定指定）
        self.default_algorithm_types: List[str] = ["Interest Recommendation"]

        # 系统内“算法类型 -> 推荐器智能体ID”显式映射（按环境约定维护）
        self.search_map: Dict[str, str] = {
            "Random Search": "search_agent_0001",
            "Hot Search": "search_agent_0002",
            "Relevant Search": "search_agent_0003",
            "LLM Search": "search_agent_0004",
        }
        self.default_search_types: List[str] = ["Relevant Search"]
        self._recommendation_earliest_post_anchor_ms: Optional[float] = None

    @staticmethod
    def _remap_activity_level(activity_level: float, out_min: float = 0.4, out_max: float = 0.8) -> float:
        """
        将 activity_level 先截断到 [0, 1]，再线性重映射到 [out_min, out_max]。
        """
        clamped = max(0.0, min(1.0, activity_level))
        return out_min + (out_max - out_min) * clamped

    async def _should_activate_this_round(self, current_step: int, max_step: int) -> bool:
        """
        使用“activity_level + 幂律上界”共同决定是否激活：
        p_t = min(activity_level, upper_t)
        """
        async with self._login_lock:
            flag = self.profile.get_data("login", -1) if self.profile else -1
            if flag != -1:
                return (flag == 1)

            raw_a = self.profile.get_data("activity_level", 0.0) if self.profile else 0.0
            try:
                activity_level = float(raw_a)
            except (TypeError, ValueError):
                activity_level = 0.0
            activity_level = self._remap_activity_level(activity_level)

            # 你环境里如果不是 current_step，可改成 round_idx / step / current_round
            step_idx = current_step
            max_step = max_step
            p_t = activity_level

            activated = random.random() < p_t
            if self.profile:
                self.profile.update_data("login", 1 if activated else 0)

            return activated

    async def get_recommendations_and_mentions(self, event: Event) -> List[Event]:
        """
        发送社交推荐。收到 StartEvent 时：先清空自己的 recommendations，
        再根据 activate_level 与随机数决定是否处理 mention 与推荐流。
        """
        current_step = getattr(event, "current_step", 1)
        max_step = getattr(event, "max_step", 8)
        if not await self._should_activate_this_round(current_step, max_step):
            logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} is not activated in this round")
            return []
        logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} is activated in this round")

        events: List[Event] = []

        # 从 event（StartEvent）的 current_notes 中筛选：关注用户的发帖，且时间在 [上次登录, 当前时间] 之间
        current_notes = getattr(event, "current_notes", None) or {}
        if not isinstance(current_notes, dict):
            current_notes = {}

        # 发送事件给推荐系统：请求“指定算法”的推荐（算法类型由代码固定指定）
        fixed_algorithm_types = self.default_algorithm_types
        if not isinstance(fixed_algorithm_types, list) or not fixed_algorithm_types:
            raise ValueError("default_algorithm_types must be a non-empty list")
        allowed_algorithm_types = set(self.recommender_map.keys())

        # 获取已推荐过的内容ID集合（从 profile 读取，统一为 strip 后的 str，便于与 current_notes 键比较）
        raw_rec = self.profile.get_data("recommended_note_ids", []) if self.profile else []
        if not isinstance(raw_rec, list):
            raw_rec = []
        recommended_note_ids = {str(x).strip() for x in raw_rec if x is not None and str(x).strip()}

        # 获取用户画像
        profile_payload = {}
        if self.profile is not None:
            try:
                profile_payload = dict(self.profile.get_profile(include_private=True) or {})
            except Exception:
                logger.warning("Failed to serialize profile via get_profile(), fallback to empty payload.")
        
        # 遍历所有指定算法类型，发送事件给推荐系统
        for fixed_algorithm_type in fixed_algorithm_types:
            if fixed_algorithm_type not in allowed_algorithm_types:
                raise ValueError(
                    f"Invalid algorithm type '{fixed_algorithm_type}'. "
                    f"Allowed types: {sorted(allowed_algorithm_types)}"
                )
            mapped_id = self.recommender_map.get(fixed_algorithm_type, "")
            if not mapped_id:
                raise ValueError(
                    f"No recommender agent mapped for algorithm type '{fixed_algorithm_type}'. "
                    f"Current mapping keys: {sorted(self.recommender_map.keys())}"
                )
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

        # 社交推荐
        current_ts = int(event.timestamp) if getattr(event, "timestamp", None) is not None else 0
        follow_ids = await self.get_data("follow_ids", [])
        follow_set = set(follow_ids) if isinstance(follow_ids, (list, tuple)) else set()

        # 上次登录时间：优先用 profile 记录，否则用 当前时间 - 一个时间步长。
        # 若长期未登录（记录早于当前时刻 3 天以上），社交推荐窗口仍只取「最近 3 天内」的帖子。
        last_login = 0
        three_days_ms = 86400000 * 1000000
        if self.profile:
            raw = self.profile.get_data("last_login_timestamp")
            logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} last_login_timestamp: {raw}")
            if raw is not None and isinstance(raw, (int, float)) and int(raw) > 0:
                last_login = int(raw)
                if current_ts > 0:
                    lower = current_ts - three_days_ms
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
            # current_notes：time < current_ts+duration；关注流再按 last_login 截断
            if last_login <= t < window_end:
                nid_key = str(note_id).strip() if note_id is not None else ""
                if nid_key and nid_key in recommended_note_ids:
                    continue
                recommendations[note_id] = note

        # 取 mentions 给自己发 MentionEvent（与 twitter 场景一致：>10 时随机保留 10 条并从 mention_pool 删除其余；≤10 时用 current_notes 刷新 note）
        mentions = getattr(event, "mentions", None) or {}
        if not isinstance(mentions, dict):
            mentions = {}
        if len(mentions) > 10:
            my_id = await self.get_data("id")
            all_keys = list(mentions.keys())
            sampled_keys = random.sample(all_keys, 10)
            logger.info(
                f"Step {current_step}/{max_step}: UserAgent {self.profile_id} "
                f"mentions count {len(all_keys)} > 10, sampling 10 for MentionEvent; dropping others from mention_pool"
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
                    logger.error(
                        f"Failed to delete non-sampled mention {k} for user {my_id} in mention_pool"
                    )
            mentions = {k: mentions[k] for k in sampled_keys}
        if 0 < len(mentions) <= 10:
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

        # 处理“保持关注/下轮再看”的一次性列表：若上轮决定 keep，则本轮将其重新塞回推荐一次
        keep_following_notes = {}
        if self.profile and self.profile.get_data("keep_following_note_ids", []) is not None:
            keep_ids = self.profile.get_data("keep_following_note_ids", []) or []
            if isinstance(keep_ids, (list, tuple)) and keep_ids:
                readded = 0
                for keep_note_id in keep_ids:
                    if keep_note_id in keep_following_notes or keep_note_id in recommendations:
                        continue
                    note = current_notes.get(keep_note_id)
                    if not isinstance(note, dict):
                        continue
                    author_id = note.get("user_id") or note.get("author_id")
                    if author_id not in follow_set:
                        continue
                    keep_following_notes[keep_note_id] = note
                    readded += 1
                if readded > 0:
                    logger.info(
                        f"Step {current_step}/{max_step}: UserAgent {self.profile_id} re-added keep_following notes: {readded}"
                    )
                # 一次性使用：无论是否成功塞回，都清空，避免无限循环
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

        # 更新「上次处理到的仿真时刻」为当前时间窗右端（与 time < ts+duration 对齐）
        if self.profile:
            _d = int(getattr(event, "timestamp_duration", 0) or 0)
            self.profile.update_data("last_login_timestamp", current_ts)
            logger.info(
                f"Step {current_step}/{max_step}: UserAgent {self.profile_id} update last_login_timestamp: {current_ts + _d}"
            )

        # 有关注流推荐时再发 SocialRecommendationEvent
        if recommendations and len(recommendations) > 0:
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

    @staticmethod
    def _note_post_time_in_window(note: Dict[str, Any], lo: float, hi: float) -> bool:
        """与 SimEnv._is_note_time_in_window 一致：发帖时间落在 [lo, hi) 内。"""
        if lo >= hi:
            return False
        raw = note.get("time", note.get("create_time"))
        try:
            return float(raw) >= lo and float(raw) < hi
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _parse_note_time_ms(raw: Any) -> Optional[float]:
        """统一为毫秒时间戳；秒级时间戳（<1e11）会乘 1000。"""
        if raw is None:
            return None
        try:
            x = float(raw)
        except (TypeError, ValueError):
            return None
        if x <= 0:
            return None
        if x < 1e11:
            x *= 1000.0
        return x

    @staticmethod
    def _format_sim_ms_utc(ms: float) -> str:
        if ms is None or ms <= 0:
            return "（未知）"
        dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M UTC")

    @staticmethod
    def _sim_ts_to_ms_for_time_module(ts: Any) -> int:
        """
        时间模块内统一用毫秒；与 multi_channel_information_diffusion / weibo_re 对齐。
        timestamp 可能为 Unix 秒（<1e12）或毫秒。
        """
        if ts is None or not isinstance(ts, (int, float)):
            return 0
        x = float(ts)
        if x < 1e12:
            return int(x * 1000)
        return int(x)

    @staticmethod
    def _content_dicts_for_time_module(chunk: Union[Dict[str, Any], List[Any]]) -> List[Dict[str, Any]]:
        """推荐 chunk（note_id -> note）或 mention_entries（mention_note / mention_blog）。"""
        items: List[Dict[str, Any]] = []
        if isinstance(chunk, dict):
            for v in chunk.values():
                if isinstance(v, dict):
                    items.append(v)
        elif isinstance(chunk, list):
            for entry in chunk:
                if not isinstance(entry, dict):
                    continue
                inner = entry.get("mention_note")
                if not isinstance(inner, dict):
                    inner = entry.get("mention_blog")
                if isinstance(inner, dict):
                    items.append(inner)
        return items

    def _time_module_for_recommendation_chunk(
        self, chunk: Union[Dict[str, Any], List[Any]], ref_ms: int
    ) -> str:
        """
        显式时间说明：
        - 首次：将本 chunk 内最早发帖时间写入 self._recommendation_earliest_post_anchor_ms（只写一次），
          并返回「当前仿真时刻 + 本批次最早发帖时刻」。
        - 之后：用当前 ref_ms 与锚点计算间隔天数；超过 7 天附加「时效弱、倾向不回复」类警告。
        """
        times_ms: List[float] = []
        for note in self._content_dicts_for_time_module(chunk):
            t = UserAgent._parse_note_time_ms(
                note.get("time", note.get("create_time"))
            )
            if t is not None:
                times_ms.append(t)
        if not times_ms:
            return None

        earliest_in_chunk_ms = min(times_ms)
        earliest_str = self._format_sim_ms_utc(earliest_in_chunk_ms)
        ref_ok = ref_ms and ref_ms > 0
        ref_str = self._format_sim_ms_utc(float(ref_ms)) if ref_ok else "（未知）"

        anchor = self._recommendation_earliest_post_anchor_ms
        if anchor is None:
            self._recommendation_earliest_post_anchor_ms = float(earliest_in_chunk_ms)
            return None

        if not ref_ok:
            anchor_str = self._format_sim_ms_utc(float(anchor))
            return None

        delta_ms = float(ref_ms) - float(anchor)
        days = delta_ms / 86400000.0
        anchor_str = self._format_sim_ms_utc(float(anchor))
        stale_days = 7.0
        if days >= 0:
            interval_txt = f"已过约 {days:.2f} 天（当前仿真时刻 − 锚点时刻，1 天 = 86400 秒）"
        else:
            interval_txt = (
                f"当前仿真时刻早于锚点约 {-days:.2f} 天（数据或时钟可能异常，请谨慎解读）"
            )
        lines = [
            f"【时间】当前仿真时刻（本轮窗口起点）：{ref_str}。",
            f"相对智能体内首次锚定的最早发帖时刻（{anchor_str}），{interval_txt}。",
        ]
        if days > stale_days:
            lines.append(
                f"【警告】间隔已超过约 {stale_days:.0f} 天，内容时效性通常已明显减弱；"
                "若无强动机或可追溯的新事实，请优先倾向不回复、不参与互动，保持沉默更合理。"
            )
        return "\n            ".join(lines)

    async def _memory_text_for_gate(self) -> str:
        """
        步骤1.5 门控专用：记忆条数少时不做「按 observation 检索 / 算相关性」，直接汇总各存储中的**全部**记忆文本，
        与 generate_reaction 里按 query retrieve 的 top_k 可能不同，但「是否有可对照记忆」以全量为准。
        """
        if not self.memory:
            return ""
        try:
            items: list = []
            gam = getattr(self.memory, "get_all_memory", None)
            if callable(gam):
                buckets = await gam()
                if isinstance(buckets, dict):
                    seen: Set[Any] = set()
                    for lst in buckets.values():
                        if not isinstance(lst, list):
                            continue
                        for msg in lst:
                            iid = getattr(msg, "id", None)
                            if iid is None:
                                iid = id(msg)
                            if iid in seen:
                                continue
                            seen.add(iid)
                            items.append(msg)
            if not items:
                memory_msgs = await self.memory.retrieve("")
                items = list(memory_msgs or [])
            text = ""
            for msg in items:
                text += getattr(msg, "content", "") or ""
            return text
        except Exception as e:
            logger.warning(
                f"UserAgent {self.profile_id} _memory_text_for_gate failed: {e}"
            )
            return ""

    def _historical_summary_text(self) -> str:
        if not self.profile:
            return ""
        try:
            p = self.profile.get_profile(include_private=False) or {}
            return str(p.get("historical_summary", "") or "")
        except Exception:
            return ""

    @staticmethod
    def _clip_historical_summary_for_mentionable(text: Any) -> str:
        """可@用户列表里的 historical_summary：超过 100 字时只保留前 50、后 50，中间用 … 连接。"""
        if text is None:
            return ""
        s = str(text)
        if len(s) <= 100:
            return s
        return s[:50] + "…" + s[-50:]

    async def _should_inject_step15(self, *, topic_text: str) -> bool:
        """
        是否注入「步骤1.5」长 prompt。策略由环境变量 ONESIM_STEP15_* 控制（见 step15_topic_gate.py）。
        记忆侧文本来自全量记忆（_memory_text_for_gate），与推荐 observation 无关；topic_text 仅用于 keyword/embedding 与当前批次比对。
        向量分支在线程中执行，避免阻塞事件循环。
        """
        cfg = load_step15_gate_config()
        mem_blob = await self._memory_text_for_gate()
        memory_nonempty = bool(mem_blob.strip())
        # hist = self._historical_summary_text()
        return await asyncio.to_thread(
            should_inject_step15,
            cfg,
            memory_nonempty=memory_nonempty,
            memory_blob=mem_blob,
            topic_text=topic_text or "",
        )

    async def evaluate_step15_policies(self, *, topic_text: str) -> Dict[str, Any]:
        """
        对 ONESIM_STEP15_POLICY 中每个策略求值，返回 dict（见 step15_topic_gate.evaluate_step15_policies）。
        总开关用键 _combined_inject；各策略名为键，值为至少含 inject: bool 的子 dict。
        向量分支在线程中执行，避免阻塞事件循环。
        """
        cfg = load_step15_gate_config()
        mem_blob = await self._memory_text_for_gate()
        memory_nonempty = bool(mem_blob.strip())
        return await asyncio.to_thread(
            _gate_evaluate_step15_policies,
            cfg,
            memory_nonempty=memory_nonempty,
            memory_blob=mem_blob,
            topic_text=topic_text or "",
        )

    async def generate_memory_from_own_notes(self, event: Event) -> List[Event]:
        """
        StartEvent 多播：从 current_notes 中筛本人帖，再按与 SimEnv 相同的规则做时间过滤：
        发帖时间 time < min(event.timestamp + timestamp_duration, simulation_cap_timestamp)（见 SimEnv._is_note_time_in_window）。
        """
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

        own_notes = []
        # 筛选本人发帖
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
        own_notes = [
            (nid, n)
            for nid, n in own_notes
            if self._note_post_time_in_window(n, lo, hi)
        ]

        if not own_notes:
            return []

        hi_f = float(hi)
        # 与 note.time 同量级：毫秒时间戳通常 ≥1e11，秒级则更小
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
            memory_text = await self.generate_memory(instruction, observation, reaction)
            if memory_text:
                logger.info(f"Step {current_step}/{max_step}: User {user_id} generated self-memory from {len(own_notes_for_prompt)} own notes, memory_text: {memory_text}")
        except Exception as e:
            logger.error(f"Step {current_step}/{max_step}: User {user_id} failed to generate memory from own notes: {e}")

        # 二次决策：是否给自己帖子补一条评论（仅写入 content_pool，不发任何提醒 / mention_pool）
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
            canonical_parent_id, _ = UserAgent._resolve_parent_comment_entry(
                comments, parent_comment_id
            )
            if parent_comment_id is not None and canonical_parent_id is None:
                logger.warning(
                    f"Step {current_step}/{max_step}: Own-note supplementary skip: parent_comment_id {parent_comment_id} not in note {note_id}"
                )
                continue

            comment_id = self._generate_comment_id()
            success = await self.add_env_comments(
                note_id,
                {
                    "comment_id": comment_id,
                    "timestamp": self._random_comment_timestamp(note, current_ts, duration),
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

    @staticmethod
    def _resolve_parent_comment_entry(
        comments_map: Any,
        parent_raw: Any,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Map ``parent_comment_id`` from LLM output onto ``note['comments']`` keys.
        Handles whitespace and str/int key mismatch. Returns ``(canonical_key, entry)`` or ``(None, None)``.
        """
        if parent_raw is None:
            return None, None
        if not isinstance(comments_map, dict) or not comments_map:
            return None, None
        s = str(parent_raw).strip()
        if not s:
            return None, None
        ent = comments_map.get(s)
        if isinstance(ent, dict):
            return s, ent
        for k, v in comments_map.items():
            if str(k).strip() == s and isinstance(v, dict):
                return str(k), v
        return None, None

    def _generate_comment_id(self) -> str:
        """
        生成唯一的评论ID，格式类似：693ba19b000000001702f98e（24位十六进制字符串）
        
        Returns:
            str: 24位十六进制字符串的唯一ID
        """
        return secrets.token_hex(12)  # 生成24位十六进制字符串

    @staticmethod
    def _note_post_time_ms(note: Dict[str, Any]) -> Optional[int]:
        """解析笔记发帖时间 ``time`` 为毫秒整数（支持 int/float/数字字符串；秒级自动乘 1000）。"""
        if not isinstance(note, dict):
            return None
        t = note.get("time")
        if t is None or isinstance(t, bool):
            return None
        if isinstance(t, (int, float)):
            x = float(t)
            if x <= 0:
                return None
            return int(round(x * 1000.0)) if x < 1e11 else int(round(x))
        if isinstance(t, str):
            s = t.strip()
            if not s:
                return None
            try:
                x = float(s)
            except ValueError:
                return None
            if x <= 0:
                return None
            return int(round(x * 1000.0)) if x < 1e11 else int(round(x))
        return None

    @staticmethod
    def _random_comment_timestamp(
        note: Dict[str, Any], window_start_ms: int, window_duration_ms: int
    ) -> int:
        """
        评论时间戳：落在 [max(发帖时间, 窗口起点), 窗口终点] 内均匀随机（毫秒）。
        与 SimEnv 一致：半开区间 [window_start_ms, window_start_ms + window_duration_ms)；
        window_duration_ms==0 时上界退化为 window_start_ms。
        若时间窗退化为单点（lo==hi），在 [0, ONESIM_COMMENT_TS_DEGENERATE_JITTER_MS] 内加毫秒抖动，
        避免「Comment Volume Real Time」等按 timestamp 汇总时全体落在同一毫秒。
        """
        if window_start_ms <= 0 and window_duration_ms <= 0:
            return 0
        lo_win = int(window_start_ms)
        if window_duration_ms > 0:
            hi_incl = lo_win + int(window_duration_ms) - 1
        else:
            hi_incl = lo_win
        post_ms = UserAgent._note_post_time_ms(note) if isinstance(note, dict) else None
        if post_ms is None:
            lo = lo_win
        else:
            lo = max(int(post_ms), lo_win)
        hi = hi_incl
        logger.info(f"lo: {lo}, hi: {hi}")
        if lo > hi:
            lo, hi = hi, lo
        if lo < hi:
            return random.randint(lo, hi)
        jitter_max = int(os.environ.get("ONESIM_COMMENT_TS_DEGENERATE_JITTER_MS", "600000"))
        if jitter_max <= 0:
            return lo
        return lo + random.randint(0, jitter_max)

    def _parse_mentions(self, comment_content: str, user_id_map: Dict[str, str], mentionable_users: Dict[str, Any]) -> Tuple[str, List[str]]:
        """
        解析评论中的@，提取被@的用户ID，并将 @id 替换为 @昵称
        
        Args:
            comment_content: 评论内容
            user_id_map: 用户ID/昵称到用户ID的映射（nickname -> user_id, user_id -> user_id）
            mentionable_users: 可@的用户列表，包含 follows、fans、mutual 三个列表
            
        Returns:
            tuple[str, List[str]]: (替换后的评论内容, 被@的用户ID列表)
        """
        # 构建 user_id 到 nickname 的反向映射
        user_id_to_nickname = {}
        all_mentionable = mentionable_users.get("follows", []) + \
                         mentionable_users.get("fans", []) + \
                         mentionable_users.get("mutual", [])
        for user_info in all_mentionable:
            if isinstance(user_info, dict):
                user_id = user_info.get("user_id", "") or user_info.get("id", "")
                nickname = user_info.get("nickname", "") 
                if user_id and nickname:
                    user_id_to_nickname[user_id] = nickname
        
        # 匹配@格式：@用户ID 或 @用户昵称
        mention_pattern = r'@(\w+)'
        mentions = re.findall(mention_pattern, comment_content)
        
        # 存储被@的用户ID列表
        mentioned_user_ids = []
        # 存储替换映射：原始内容 -> 替换后的内容
        replacements = {}
        
        # 为每个@找到对应的用户ID，并准备替换
        for mention in mentions:
            # 通过 user_id_map 找到对应的用户ID
            mentioned_user_id = user_id_map.get(mention)
            
            if mentioned_user_id:
                mentioned_user_ids.append(mentioned_user_id)
                # 获取对应的昵称
                nickname = user_id_to_nickname.get(mentioned_user_id, "")
                if nickname:
                    # 将 @id 或 @昵称 替换为 @昵称
                    replacements[f"@{mention}"] = f"@{nickname}"
                else:
                    # 如果找不到昵称，保持原样（但这种情况不应该发生）
                    logger.warning(f"Could not find nickname for user_id: {mentioned_user_id}")
            else:
                logger.warning(f"Could not find user_id for mention: {mention}")
        
        # 执行替换
        updated_content = comment_content
        for old_text, new_text in replacements.items():
            updated_content = updated_content.replace(old_text, new_text)
        
        return updated_content, mentioned_user_ids
    
    def _add_recommendations(self, recommendations: Dict[str, Any]) -> None:
        """
        将本轮推荐中的 note_id 追加写入 profile.recommended_note_ids（调用前应已按需过滤）。
        """
        if not recommendations:
            return
        
        # 获取已推荐过的内容ID集合（从profile中读取）
        recommended_note_ids = set(self.profile.get_data("recommended_note_ids", [])) if self.profile else set()
        
        new_note_ids = []
        
        for note_id in recommendations.keys():
            new_note_ids.append(note_id)
            # logger.info(f"Recommendation {note_id} is added to filtered list")
        
        # 将新推荐的内容ID添加到已推荐列表中
        if new_note_ids and self.profile:
            all_recommended = list(recommended_note_ids) + new_note_ids
            self.profile.update_data("recommended_note_ids", all_recommended)

    def _record_recommendations_by_source_step(
        self,
        source_type: str,
        current_step: int,
        recommendations: Dict[str, Any],
        event_timestamp: Any,
    ) -> None:
        """
        按推荐来源 source_type 与仿真轮次 current_step，把本轮 note_id 追加写入
        profile.recommended_note_ids_by_channel。

        结构：recommended_note_ids_by_channel[source_type][str(step)] = [note_id, ...]
        同一 (source_type, step) 下多次写入时合并列表并去重（保持顺序）。

        若 event_timestamp 可解析为大于 0 的整数，则同步更新 profile.last_login_timestamp。
        最后打日志输出 last_login_timestamp 与完整 recommended_note_ids_by_channel。
        """
        if not self.profile:
            return
        st = (source_type or "").strip()
        if not st or not recommendations:
            return

        try:
            step_key = str(int(current_step))
        except (TypeError, ValueError):
            step_key = str(current_step)

        raw_root = self.profile.get_data("recommended_note_ids_by_channel", {})
        by_ch: Dict[str, Any] = dict(raw_root) if isinstance(raw_root, dict) else {}

        step_map_raw = by_ch.get(st)
        step_map: Dict[str, Any] = (
            dict(step_map_raw) if isinstance(step_map_raw, dict) else {}
        )

        prev_ids = step_map.get(step_key)
        merged: List[str] = list(prev_ids) if isinstance(prev_ids, list) else []
        seen: Set[str] = {str(x).strip() for x in merged if str(x).strip()}
        for note_id in recommendations.keys():
            sid = str(note_id).strip()
            if not sid or sid in seen:
                continue
            merged.append(sid)
            seen.add(sid)

        step_map[step_key] = merged
        by_ch[st] = step_map
        self.profile.update_data("recommended_note_ids_by_channel", by_ch)

        ts_int = 0
        if event_timestamp is not None:
            try:
                ts_int = int(event_timestamp)
            except (TypeError, ValueError):
                ts_int = 0
        if ts_int > 0:
            self.profile.update_data("last_login_timestamp", ts_int)

        raw_ll = self.profile.get_data("last_login_timestamp", 0)
        try:
            ll_out = int(raw_ll) if raw_ll is not None else 0
        except (TypeError, ValueError):
            ll_out = 0

        logger.info(
            f"UserAgent {self.profile_id} recommended_by_channel: "
            f"last_login_timestamp={ll_out} "
            f"recommended_note_ids_by_channel={json.dumps(by_ch, ensure_ascii=False, default=str)}"
        )

    def _record_mentioned_note_ids_by_channel(
        self,
        current_step: int,
        mention_entries: List[Dict[str, Any]],
        event_timestamp: Any,
    ) -> None:
        """
        按 MentionEvent 中的 mention_type 与当前轮次，把相关 note_id 追加写入 profile.mentioned_note_ids_by_channel。

        结构：mentioned_note_ids_by_channel[mention_type][str(step)] = [note_id, ...]
        同一 (mention_type, step) 下多次写入时合并列表并去重（保持顺序）。

        若 event_timestamp 可解析为大于 0 的整数，则同步更新 profile.last_login_timestamp。
        最后打日志输出 last_login_timestamp、current_step 与完整 mentioned_note_ids_by_channel。
        """
        if not self.profile or not mention_entries:
            return

        try:
            step_key = str(int(current_step))
        except (TypeError, ValueError):
            step_key = str(current_step)

        batch: Dict[str, List[str]] = defaultdict(list)
        for entry in mention_entries:
            if not isinstance(entry, dict):
                continue
            mt = str(entry.get("mention_type") or "at").strip() or "at"
            nid = entry.get("note_id")
            sid = str(nid).strip() if nid is not None else ""
            if sid:
                batch[mt].append(sid)

        if not batch:
            return

        raw_root = self.profile.get_data("mentioned_note_ids_by_channel", {})
        by_ch: Dict[str, Any] = dict(raw_root) if isinstance(raw_root, dict) else {}

        for mt, ids in batch.items():
            prev = by_ch.get(mt)
            if isinstance(prev, dict):
                step_map: Dict[str, Any] = dict(prev)
            else:
                step_map = {}

            prev_ids = step_map.get(step_key)
            merged: List[str] = list(prev_ids) if isinstance(prev_ids, list) else []
            seen: Set[str] = {str(x).strip() for x in merged if str(x).strip()}
            for sid in ids:
                if sid not in seen:
                    merged.append(sid)
                    seen.add(sid)
            step_map[step_key] = merged
            by_ch[mt] = step_map

        self.profile.update_data("mentioned_note_ids_by_channel", by_ch)

        ts_int = 0
        if event_timestamp is not None:
            try:
                ts_int = int(event_timestamp)
            except (TypeError, ValueError):
                ts_int = 0
        if ts_int > 0:
            self.profile.update_data("last_login_timestamp", ts_int)

        raw_ll = self.profile.get_data("last_login_timestamp", 0)
        try:
            ll_out = int(raw_ll) if raw_ll is not None else 0
        except (TypeError, ValueError):
            ll_out = 0

        try:
            step_out = int(current_step)
        except (TypeError, ValueError):
            step_out = current_step

        logger.info(
            f"UserAgent {self.profile_id} mentioned_by_channel: "
            f"step={step_out} last_login_timestamp={ll_out} "
            f"mentioned_note_ids_by_channel={json.dumps(by_ch, ensure_ascii=False, default=str)}"
        )

    @staticmethod
    def _fallback_mention_reaction_dict(mention_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        LLM 返回非合法 JSON 或解析失败时，按提醒条数生成全 comment=false 的安全结构，避免协程未处理异常。
        """
        decisions: List[Dict[str, Any]] = []
        for entry in mention_entries:
            if not isinstance(entry, dict):
                continue
            nid = entry.get("note_id")
            sid = str(nid).strip() if nid is not None else ""
            if not sid:
                continue
            decisions.append(
                {
                    "note_id": sid,
                    "comment": False,
                    "parent_comment_id": None,
                    "comment_content": "",
                    "decision_reason": "模型输出非合法JSON，跳过",
                    "expression_reason": "",
                    "mention_reasoning": [],
                }
            )
        return {
            "persona_understanding": "",
            "content_understanding": "",
            "relationship_understanding": "",
            "memory_reflection": "",
            "decisions": decisions,
            "keep_following_note_ids": [],
            "keep_following_reason": "",
            "search": False,
            "search_keyword": "",
            "search_reason": "",
        }

    def _get_mentionable_users(
        self,
        follow_ids: List[str],
        fan_ids: List[str]
    ) -> Dict[str, Any]:
        """
        获取可@的用户列表（包括关注、粉丝、互关）
        
        Args:
            follow_ids: 关注列表
            fan_ids: 粉丝列表
        
        Returns:
            Dict[str, Any]: 包含 follows、fans、mutual 三个列表的字典
        """
        mentionable_info = {
            "follows": [],  # 关注列表
            "fans": [],  # 粉丝列表
            "mutual": []  # 互关列表
        }
        
        # 转换为集合以便计算
        follow_ids = set(follow_ids)
        fan_ids = set(fan_ids)
        
        # 计算互关（交集）
        mutual_ids = follow_ids & fan_ids
        
        # 构建用户ID到昵称的映射
        user_info = {}
        
        if not self.relationship_manager:
            logger.warning("RelationshipManager is not initialized")
            return mentionable_info
        
        # 处理所有列表，直接从 relationship_manager 获取用户信息（同步，快速）
        follows_info = []
        fans_info = []
        mutual_info = []
        
        # 处理关注列表（包含互关）
        for user_id in follow_ids:
            rel = self.relationship_manager.get_relationship(user_id)
            if rel and rel.target_info and isinstance(rel.target_info, dict):
                info = dict(rel.target_info)
                if "historical_summary" in info:
                    info["historical_summary"] = self._clip_historical_summary_for_mentionable(
                        info.get("historical_summary")
                    )
                hn = info.get("historical_notes")
                if isinstance(hn, dict) and len(hn) > 2:
                    items = list(hn.items())[:2]
                    info["historical_notes"] = dict(items)
                follows_info.append(info)
        
        # 处理粉丝列表（包含互关）
        for user_id in fan_ids:
            rel = self.relationship_manager.get_relationship(user_id)
            # 先检查 rel 是否为 None，再访问 target_info
            if rel and rel.target_info and isinstance(rel.target_info, dict):
                info = dict(rel.target_info)
                if "historical_summary" in info:
                    info["historical_summary"] = self._clip_historical_summary_for_mentionable(
                        info.get("historical_summary")
                    )
                hn = info.get("historical_notes")
                if isinstance(hn, dict) and len(hn) > 2:
                    items = list(hn.items())[:2]
                    info["historical_notes"] = dict(items)
                fans_info.append(info)
        
        # 处理互关（互关用户的历史发帖最多保留 2 条）
        for user_id in mutual_ids:
            rel = self.relationship_manager.get_relationship(user_id)
            # 先检查 rel 是否为 None，再访问 target_info
            if rel and rel.target_info and isinstance(rel.target_info, dict):
                info = dict(rel.target_info)
                if "historical_summary" in info:
                    info["historical_summary"] = self._clip_historical_summary_for_mentionable(
                        info.get("historical_summary")
                    )
                hn = info.get("historical_notes")
                if isinstance(hn, dict) and len(hn) > 2:
                    items = list(hn.items())[:2]
                    info["historical_notes"] = dict(items)
                mutual_info.append(info)

        mentionable_info["follows"] = follows_info
        mentionable_info["fans"] = fans_info
        mentionable_info["mutual"] = mutual_info
        
        return mentionable_info

    async def add_env_comments(self, key: str, value: Any, parent_event_id: Optional[str] = None) -> bool:
        """
        添加评论到环境中的数据（使用分布式锁）
        
        Args:
            key: note_id（例如 "69290e59000000001e034ab4"），会自动转换为 "content_pool.{note_id}.comments"
            value: 评论数据字典，必须包含 comment_id 字段
            parent_event_id: 父事件ID（可选）
        """
        # 将 note_id 转换为完整的 key 格式：content_pool.note_id.comments
        full_key = f"content_pool.{key}.comments"
        
        # 创建唯一的请求ID
        request_id = f"agent_env_add_comments_req_{time.time()}_{id(self)}"

        # 创建 Future 用于接收响应
        future = asyncio.Future()
        self._comment_add_futures[request_id] = future

        # 创建添加评论事件
        comment_add_event = AddCommentEvent(
            from_agent_id=self.profile_id,  # 请求来源：当前代理
            to_agent_id="ENV",              # 请求目标：环境（特殊目标）
            source_type="AGENT",            # 源类型：代理
            target_type="ENV",              # 目标类型：环境
            key=full_key,                   # 要更新的数据键（完整格式）
            value=value,                    # 新的数据值
            request_id=request_id,          # 请求ID，用于匹配响应
            parent_event_id=parent_event_id # 父事件ID，用于事件追踪
        )

        # 获取此键的分布式锁
        # 锁ID格式：env_data_add_comments_lock_{key}，确保每个键有独立的锁
        lock_id = f"env_comment_add_lock_{key}"
        lock = await get_lock(lock_id)

        try:
            # 在发送更新前获取锁
            # 使用 async with 确保锁在使用后自动释放
            async with lock:
                # 将更新请求事件放入事件总线队列
                from onesim.events import get_event_bus
                event_bus = get_event_bus()
                await event_bus.dispatch_event(comment_add_event)

                # 等待响应（带超时）
                try:
                    if hasattr(self, '_sync_event'):
                        await asyncio.wait_for(self._sync_event.wait(), timeout=_ENV_ASYNC_OP_TIMEOUT)
                        return await future
                    else:
                        return await asyncio.wait_for(future, timeout=_ENV_ASYNC_OP_TIMEOUT)
                except asyncio.TimeoutError:
                    logger.warning(f"等待环境评论添加超时: {key}")
                    self._comment_add_futures.pop(request_id, None)
                    return False
                except Exception as e:
                    logger.error(f"添加环境评论时出错: {e}")
                    self._comment_add_futures.pop(request_id, None)
                    return False
        except Exception as e:
            logger.error(f"获取环境评论添加锁时出错: {e}")
            return False

    async def handle_add_comment_response(self, event: AddCommentResponseEvent) -> None:
        """
        处理传入的评论添加响应事件
        """
        # 检查是否正在等待此响应
        if event.request_id in self._comment_add_futures:
            future = self._comment_add_futures.pop(event.request_id)

            if not future.done():
                if event.success:
                    future.set_result(True)
                else:
                    future.set_exception(ValueError(event.error or "未知错误"))

            # 如果有同步事件，设置它
            if hasattr(self, '_sync_event'):
                self._sync_event.set()
                # 为下次操作重置
                self._sync_event.clear()

    async def update_env_mention_pool(self, key: str, value: Any, parent_event_id: Optional[str] = None) -> bool:
        """
        更新环境中的mention_pool（使用分布式锁）
        
        Args:
            key: mentioner_id（例如 "69290e59000000001e034ab4"），会自动转换为 "mention_pool.mentioner_id.comment_id"
            value: mention_pool数据字典，必须包含 mention_key 字段
            parent_event_id: 父事件ID（可选）
        """
        # 将 mentioner_id.comment_id 转换为完整的 key 格式：mention_pool.mentioner_id.comment_id
        full_key = f"mention_pool.{key}"
        
        # 创建唯一的请求ID
        request_id = f"agent_env_update_mention_pool_req_{time.time()}_{id(self)}"

        # 创建 Future 用于接收响应
        future = asyncio.Future()
        self._mention_pool_update_futures[request_id] = future

        # 创建更新mention_pool事件
        mention_pool_update_event = MentionPoolUpdateEvent(
            from_agent_id=self.profile_id,  # 请求来源：当前代理
            to_agent_id="ENV",              # 请求目标：环境（特殊目标）
            source_type="AGENT",            # 源类型：代理
            target_type="ENV",              # 目标类型：环境
            key=full_key,                   # 要更新的数据键（完整格式）
            value=value,                    # 新的数据值
            request_id=request_id,          # 请求ID，用于匹配响应
            parent_event_id=parent_event_id # 父事件ID，用于事件追踪
        )

        # 获取此键的分布式锁
        # 锁ID格式：env_mention_pool_update_lock_{key}，确保每个键有独立的锁
        lock_id = f"env_mention_pool_update_lock_{key}"
        lock = await get_lock(lock_id)

        try:
            # 在发送更新前获取锁
            # 使用 async with 确保锁在使用后自动释放
            async with lock:
                # 将更新请求事件放入事件总线队列
                from onesim.events import get_event_bus
                event_bus = get_event_bus()
                await event_bus.dispatch_event(mention_pool_update_event)

                # 等待响应（带超时）
                try:
                    if hasattr(self, '_sync_event'):
                        await asyncio.wait_for(self._sync_event.wait(), timeout=_ENV_ASYNC_OP_TIMEOUT)
                        return await future
                    else:
                        return await asyncio.wait_for(future, timeout=_ENV_ASYNC_OP_TIMEOUT)
                except asyncio.TimeoutError:
                    logger.warning(
                        f"等待环境mention_pool更新超时: {key} "
                        f"(>{_ENV_ASYNC_OP_TIMEOUT}s，多为 SimEnv 全局锁排队；可调大环境变量 ONESIM_ENV_ASYNC_OP_TIMEOUT)"
                    )
                    self._mention_pool_update_futures.pop(request_id, None)
                    return False
                except Exception as e:
                    logger.error(f"更新环境mention_pool时出错: {e}")
                    self._mention_pool_update_futures.pop(request_id, None)
                    return False
        except Exception as e:
            logger.error(f"获取环境mention_pool更新锁时出错: {e}")
            return False

    async def handle_update_mention_pool_response(self, event: MentionPoolUpdateResponseEvent) -> None:
        """
        处理传入的更新mention_pool响应事件
        """
        # 检查是否正在等待此响应
        if event.request_id in self._mention_pool_update_futures:
            future = self._mention_pool_update_futures.pop(event.request_id)

            if not future.done():
                if event.success:
                    future.set_result(True)
                else:
                    future.set_exception(ValueError(event.error or "未知错误"))

            # 如果有同步事件，设置它
            if hasattr(self, '_sync_event'):
                self._sync_event.set()
                # 为下次操作重置
                self._sync_event.clear()

    def _filter_recommendations(self, recommendations: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        筛选推荐内容的接口
        
        过滤掉已经推荐过的内容，避免重复推荐
        """
        if not recommendations:
            return {}
        
        # 获取已推荐过的内容ID集合（从profile中读取）
        recommended_note_ids = set(self.profile.get_data("recommended_note_ids", [])) if self.profile else set()
        
        # 过滤掉已推荐过的内容（结果键集即待写入 profile 的 id，无需再维护平行列表）
        filtered: Dict[str, Dict[str, Any]] = {}

        for note_id, rec in recommendations.items():
            if not isinstance(rec, dict):
                logger.warning(f"Recommendation {note_id} is not a dictionary")
                continue

            # 如果笔记ID不存在或已经推荐过，跳过
            if not note_id or note_id in recommended_note_ids:
                logger.info(f"Recommendation {note_id} is already recommended")
                continue

            filtered[note_id] = rec
            logger.info(f"Recommendation {note_id} is added to filtered list")

        return filtered

    async def receive_recommendation(self, event: Event) -> List[Event]:
        """
        接收推荐并决定是否评论。必须本轮的「所有推荐系统 Event」都收到且存在社交推荐，才进行后续处理；
        否则按类型分别存入缓冲，等收齐后再处理。
        """
        current_step = getattr(event, "current_step", 1)
        max_step = getattr(event, "max_step", 8)
        if not await self._should_activate_this_round(current_step, max_step):
            logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} receive recommendation but is not activated this round")
            return []

        # 按推荐类型分别存入缓冲
        if event.__class__.__name__ == "AlgorithmRecommendationEvent":
            source_type = "algorithm"
        elif event.__class__.__name__ == "SocialRecommendationEvent":
            source_type = "social"
        elif event.__class__.__name__ == "SearchRecommendationEvent":
            source_type = "search"
        elif event.__class__.__name__ == "KeepFollowingEvent":
            source_type = "keep_following"
        else:
            evt_cls = event.__class__.__name__
            logger.warning(
                f"Step {current_step}/{max_step}: UserAgent {self.profile_id} "
                f"unknown recommendation event {evt_cls}, skip"
            )
            return []

        logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} receive recommendation event, source type: {source_type}, length of recommendations: {len(event.recommendations)}")

        # algorithm/social：先去掉已在 profile.recommended_note_ids 中的重复推荐；keep_following 不过滤
        recommendations = event.recommendations
        self._record_recommendations_by_source_step(
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
        # 将本轮保留下的 note_id 写入 profile.recommended_note_ids（与上方过滤分工不同）
        self._add_recommendations(recommendations)

        # 获取用户信息和可@的用户列表
        user_id = await self.get_data("id")
        user_nickname = await self.get_data("nickname", "")
        ip_location = await self.get_data("ip_location", "")
        current_timestamp = event.timestamp
        window_start_ms = int(current_timestamp) if isinstance(current_timestamp, (int, float)) else 0
        window_duration_ms = int(getattr(event, "timestamp_duration", 0) or 0)
        time_ref_ms = self._sim_ts_to_ms_for_time_module(current_timestamp)
        follow_ids = await self.get_data("follow_ids", [])
        fan_ids = await self.get_data("fan_ids", [])
        mentionable_users = self._get_mentionable_users(follow_ids, fan_ids)
        
        # 构建用户ID和昵称的映射，用于@匹配
        # user_id_map = {}  # nickname -> user_id
        # all_mentionable = mentionable_users.get("follows", []) + \
        #                  mentionable_users.get("fans", []) + \
        #                  mentionable_users.get("mutual", [])
        # for user_info in all_mentionable:
        #     if isinstance(user_info, dict):
        #         user_id_map[user_info.get("user_id", "")] = user_info.get("user_id", "")
        #         user_id_map[user_info.get("nickname", "")] = user_info.get("user_id", "")  # 注意：schema中是nickname
        
        # 构建observation和instruction
        # profile_str = self.get_profile_str(include_private=False) if self.profile else "No profile information"
        # 若推荐超过 15 条，按每 15 条一组循环处理
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

            # Label recommendation source；若推荐里包含自己发布的内容，优先标记为“自己发布”
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

            time_module_str = self._time_module_for_recommendation_chunk(
                chunk, time_ref_ms
            )

            interaction_threshold = InteractionThreshold.sample(random.Random())
            k_same_target = interaction_threshold.k_same_target
            k_diff_targets = interaction_threshold.k_diff_targets

            raw_act = self.profile.get_data("activity_level", 0.0) if self.profile else 0.0
            try:
                _activity = float(raw_act)
            except (TypeError, ValueError):
                _activity = 0.0
            _activity = max(0.0, min(1.0, _activity))
            
            topic_txt = topic_text_from_notes_chunk(chunk)
            s15_ev = await self.evaluate_step15_policies(topic_text=topic_txt)
            mem = s15_ev.get("memory_nonempty") or {}
            kw = s15_ev.get("keyword") or {}
            emb = s15_ev.get("embedding") or {}
            mem_ok = bool(mem.get("inject"))
            kw_ok = bool(kw.get("inject"))
            emb_ok = bool(emb.get("inject"))

            step15_kw_coaching = (
                "\n\n【话题与 memory — 关键词重叠】\n            "
                "- 当前批次与已存记忆在关键词层面判定为显著相关 → **强烈倾向 comment=false**（易与 memory 中已有表态或同题讨论重复；仅当步骤1.5 明确满足可核验新信息与强动机等破例条件时再考虑 comment=true）；"
            ) if kw_ok else ""
            step15_emb_coaching = (
                "\n\n【话题与 memory — 语义相似】\n            "
                "- 向量相似度达到设定阈值，本批话题与记忆中内容相近 → **强烈倾向 comment=false**（视同同脉络/易重复话题，须严格按步骤1.5 评估是否仍 comment=true）；"
            ) if emb_ok else ""

            if (_activity < 0.65 and emb_ok):
                step15_receive = """
                步骤1.5：对照 memory 做「重复话题」检查（在步骤2之前完成）
                - 默认 comment=false。若当前内容与 memory 指向**同一事件/同一问题/同一争议脉络**，或与 memory 中重叠的词超过1个，无论你是否**已在同类内容上转发、表态过**，则**几乎必须保持comment=false**。
                - **若要破例，须同时满足以下两项，缺一不可：**
                · **（1）可核对的新信息点**：`decision_reason` 须在**单句**内写清相对 memory、帖中**独有**且可指认的一条新增信息（须出现具体人/机构/日期/数字/规则名或链接类标识之一）；不得单独用「新细节」「新进展」「新讨论点」「又一例」「再关注」「同类再发酵」「略多一句」等空话充数。
                · **（2）强动机**：**强烈情绪动机**（同句或紧邻句须点明具体情绪落点，禁止空泛「有感触」「想说两句」）**或** **明确关系动机**（须点明为维护/帮扩/站队**具体的**互关、关注或好友，写清对象，禁止笼统「支持一下」）。
                -**同时**具备可核对新信息 **与**（强烈情绪 **或** 明确关系动机），仅有情绪/关系而无新信息、或仅有新信息而无强情绪/关系动机，均 comment=false，避免同题刷屏。
                - **memory 越多、越要克制**：即便（1）（2）在字面上都能凑上，仍应把「本条 comment=true」当成**小概率事件**——默认继续 comment=false；仅当新信息**明显升级**（例如改变事件阶段、推翻或修正你 memory 中的既有判断、或出现关键新主体/新规则）时才可破例，禁止「勉强达标就评一句」。
                - **memory_reflection 禁止自相矛盾**：先写「与 memory 重叠/同一话题/已讨论过」等，又用无（1）+（2）支撑的转折暗示可以评论——一律视为无效；若判定重叠或几乎 comment=false，memory_reflection 须**通篇**结论为倾向沉默或明确无新信息，不得以模糊语气自我放行。
                将上述结论简要写入 memory_reflection；不转发时 decision_reason 须点明「与 memory 重叠/已表态/无新信息/缺新信息或缺强动机」等。
                """
                step2_zero_rec = "0. 未落入步骤1.5 的「几乎确定comment=false」情形，或你已写出可核验的新增信息点（仅换说法不算）；"
                mem_refl_rec = "2-3句。有同题时说明是否重叠、是否几乎应comment=false"
            else:
                step15_receive = ""
                step2_zero_rec = "0. 未落入步骤1.5 的情形，或你已写出可核验的新增信息点（仅换说法不算）"
                mem_refl_rec = "1-2句。无已存储记忆可对照时写「无相关记忆/首次接触」即可"

            if time_module_str:
                time_coaching_block = (
                    "【仿真时间与时效】\n            "
                    + time_module_str
                    + " - 若上文含 **【警告】** 或写明时效已明显减弱、倾向不回复 → **强烈倾向 repost=false**（该条不进候选池）；"
                )
            else:
                time_coaching_block = ""

            instruction = f"""根据用户 Profile、historical_summary、memory 和推荐内容，完成是否评论/回复及内容生成。

            步骤1：对每条推荐先设默认状态
            - "comment": false
            - "parent_comment_id": null
            - "comment_content": ""

           **comment 的数量决策仅由步骤2/3/4决定，不得因步骤5改变；**
            {step15_receive}

            步骤2：兴趣判断
            - 通读本批次所有推荐内容，判断哪些笔记能够进入候选池，当且仅当同时满足以下条件时，再将该笔记改为 comment=true：
                {step2_zero_rec}
                1. **对笔记感兴趣**：整帖与 Profile/historical_summary/memory 高度相关
                2. **对某条楼中评论感兴趣**：未必对整帖强兴趣，但**某条具体评论**与 Profile/historical_summary/memory 高度相关，且你愿意针对该评论做**增量回应**（补充、纠错、追问、共情、接梗）
                3. 互动对象关系与场景合适（关注关系优先）；
                4. 表达目的明确（补充事实、推进互动）
            - 进入候选池时，请在心中标记：本轮互动是「对帖」还是「对某条评论」；若选「对某条评论」，必须在后续 JSON 里使用非空的 parent_comment_id（见步骤4）。
            - 对不感兴趣的笔记：该笔记不进入候选池，comment=false。

            步骤3：选取 0～{k_diff_targets} 目标笔记
            - {k_diff_targets} 表示本轮**最多**对多少个**不同** note_id 置 comment=true；实际条数可为 0～{k_diff_targets} 中任意值，**不得**为凑满条数而放宽步骤1.5/步骤2。
            - 仅在步骤2 的候选池内、按你的排序取前若干条作为“目标 note_id 集合”，且条数不超过 {k_diff_targets}；若候选少于 {k_diff_targets}，有多少算多少；其余笔记保持 comment=false。

            步骤4：为每个目标 note_id 生成 0～{k_same_target} 条评论
            - **回帖 vs 回评论（务必二选一写进结构）**：
                · 对帖首评：`parent_comment_id` 必须为 null，comment_content 针对笔记正文。
                · 回复楼中某条：`parent_comment_id` 必须为推荐里给出的**真实 comment_id**（不得编造），comment_content 必须**承接该条评论的具体词句或论点**，禁止像对帖空泛表态。
            - 对每个目标 note_id：在 decisions 中写出 {k_same_target} 个 comment=true 的条目（可用于追评链/补一句）。
            - 若 {k_same_target}=1：每个目标 note_id 只生成 1 条评论。
            - 若 {k_same_target}>=2：默认同时评论笔记和评论，也允许同帖多条，但第2条及之后必须是增量（补充新点/纠错/情绪加码），禁止复述同一句。

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
            "memory_reflection": "{mem_refl_rec}",
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
            {step15_kw_coaching}{step15_emb_coaching}
            {time_coaching_block}"""

            # 调用LLM生成决策
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

            # 处理搜索决策
            search = response.get("search", False)
            if search and not has_search:
                has_search = True
                # 发送事件给推荐系统：请求“指定算法”的推荐（算法类型由代码固定指定）
                search_algorithm_types = self.default_search_types
                if not isinstance(search_algorithm_types, list) or not search_algorithm_types:
                    raise ValueError("default_search_types must be a non-empty list")
                allowed_search_types = set(self.search_map.keys())

                # 遍历所有指定算法类型，发送事件给推荐系统
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

                    # 获取用户画像
                    profile_payload = {}
                    if self.profile is not None:
                        try:
                            profile_payload = dict(self.profile.get_profile(include_private=True) or {})
                        except Exception:
                            logger.warning("Failed to serialize profile via get_profile(), fallback to empty payload.")
                    logger.info(f"Sending SearchEvent to RecommenderAgent {mapped_id} for search algorithm type {search_algorithm_type}")
                    events_to_send.append(SearchEvent(
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

            # 处理保持关注决策
            if self.profile:
                keep_ids = response.get("keep_following_note_ids", [])
                if isinstance(keep_ids, list) and keep_ids:
                    # 只允许本批次内的 note_id，且最多 1 个
                    valid_keep_ids = []
                    for keep_note_id in keep_ids:
                        if keep_note_id in chunk:
                            valid_keep_ids.append(keep_note_id)
                    if valid_keep_ids:
                        self.profile.update_data("keep_following_note_ids", valid_keep_ids[:1])
                    else:
                        self.profile.update_data("keep_following_note_ids", [])

            # 处理评论决策
            decisions = response.get("decisions", [])
            if not isinstance(decisions, list):
                continue
       
            # 处理每个决策：更新评论数和评论内容
            for decision in decisions:
                if not isinstance(decision, dict):
                    continue
                    
                note_id = decision.get("note_id")
                should_comment = decision.get("comment", False)
                parent_comment_id = decision.get("parent_comment_id")  # 如果是回复评论，需要指定父评论ID
                comment_content = decision.get("comment_content", "")
                    
                if not note_id or not should_comment:
                    continue

                # 检查 note_id 是否合法
                if note_id not in chunk:
                    logger.warning(f"Step {current_step}/{max_step}: Note {note_id} not found in recommendations")
                    continue
            
                # 解析评论内容中的@用户，将 @id 替换为 @昵称，并返回用户ID列表
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
                    canonical_parent_id, parent_entry = UserAgent._resolve_parent_comment_entry(
                        comments_map, parent_comment_id
                    )
                    if isinstance(parent_entry, dict):
                        comment_author_id = parent_entry.get("user_id")
                    mentioned_user_ids = [
                        uid for uid in mentioned_user_ids if uid != comment_author_id
                    ]

                mention_count = len(mentioned_user_ids)
                
                # 如果为回复评论，检查父评论ID是否合法
                if parent_comment_id is not None:
                    if canonical_parent_id is not None:
                        # 生成唯一的评论ID
                        comment_id = self._generate_comment_id()
                        
                        success = await self.add_env_comments(note_id, {
                            "comment_id": comment_id,
                            "timestamp": self._random_comment_timestamp(
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
                        
                        # 向评论作者发送提醒
                        if comment_author_id and comment_author_id != user_id:  # 不给自己发提醒
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
                    
                        # 向笔记作者发送提醒
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
                    # 如果为评论笔记，添加评论
                    # 生成唯一的评论ID
                    comment_id = self._generate_comment_id()
                    
                    success = await self.add_env_comments(note_id, {
                        "comment_id": comment_id,
                        "timestamp": self._random_comment_timestamp(
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

                    # 向笔记作者发送提醒
                    if note_author_id and note_author_id != user_id:  # 不给自己发提醒
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
                   

                # 发送MentionEvent给被@的用户
                if mentioned_user_ids:
                    # 为每个被@的用户创建MentionEvent
                    for mentioned_user_id in mentioned_user_ids:
                        if mentioned_user_id and mentioned_user_id != user_id:  # 不给自己发提醒
                            # 创建@事件，发送给被@的用户
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
                    
        # 通过事件通知环境更新内容池
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
        """
        检查与另一个用户的关系类型

        Args:
            other_user_id: 另一个用户的ID
            
        Returns:
            str: 关系类型 - "mutual"（互关）、"follow"（关注）、"fan"（粉丝）、"none"（无关系）
        """
        if not other_user_id:
            return "none"
        
        if not my_id or my_id == other_user_id:
            return "none"
        
        # 检查是否是互关
        if other_user_id in follow_ids and other_user_id in fan_ids:
            return "mutual"
        
        # 检查是否是我关注的人
        if other_user_id in follow_ids:
            return "follow"
        
        # 检查是否是我的粉丝
        if other_user_id in fan_ids:
            return "fan"

        return "none"

    def _merge_mentions(
        self, 
        old_messages: Dict[str, Any], 
        new_messages: Dict[str, Any], 
        current_ts: int = 0,
        last_check: int = 0,
        period_ms: int = 24 * 3600 * 1000,
        mention_cap: int = 20,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        合并提醒消息
        """
        merged = {**old_messages, **new_messages}

        if current_ts >= last_check + period_ms:
            if len(merged) > mention_cap:
                return {}, {}
            return merged, {}
        else:
            return {}, merged

    async def handle_mention(self, event: MentionEvent) -> List[Event]:
        """
        处理@/评论/回复提醒事件
        
        当用户被@、被评论或被回复时，更容易进行回复。
        - @提醒（mention_type="at"）：回应概率最高，来自关注的@更容易评论
        - 评论提醒（mention_type="comment"）：回应概率中等
        - 回复提醒（mention_type="reply"）：回应概率中等
        
        Args:
            event: MentionEvent，包含@/评论/回复信息
            
        Returns:
            List[Event]: 返回要发送的事件列表（通常是评论事件）
        """
        current_step = getattr(event, "current_step", 1)
        max_step = getattr(event, "max_step", 8)
        if not await self._should_activate_this_round(current_step, max_step):
            logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} handle mention but is not activated this round")
            return []

        # 获取提醒信息
        mentions = getattr(event, "mentions", {}) or {}
        if not mentions:
            return []

        logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} receive mention event, length of mentions: {len(mentions)}")

        # 获取可@的用户列表
        follow_ids = await self.get_data("follow_ids", [])
        fan_ids = await self.get_data("fan_ids", [])
        my_id = await self.get_data("id")
        mentionable_users = self._get_mentionable_users(follow_ids, fan_ids)
        mentionable_users_str = json.dumps(mentionable_users.get("follows", []), ensure_ascii=False, indent=2)

        # 构建用户ID和昵称的映射，用于@匹配
        # user_id_map = {}
        # all_mentionable = mentionable_users.get("follows", []) + mentionable_users.get("fans", []) + mentionable_users.get("mutual", [])
        # for user_info in all_mentionable:
        #     if isinstance(user_info, dict):
        #         user_id_map[user_info.get("user_id", "")] = user_info.get("user_id", "")
        #         user_id_map[user_info.get("nickname", "")] = user_info.get("user_id", "")
        # profile_str = self.get_profile_str(include_private=False) if self.profile else "No profile information"

        # 构建提醒信息
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

        self._record_mentioned_note_ids_by_channel(
            current_step,
            mention_entries,
            getattr(event, "timestamp", 0),
        )

        # 构建观察信息
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

        time_ref_ms = self._sim_ts_to_ms_for_time_module(
            getattr(event, "timestamp", None)
        )
        time_module_str = self._time_module_for_recommendation_chunk(
            mention_entries, time_ref_ms
        )

        interaction_threshold = InteractionThreshold.sample(random.Random())
        k_diff_targets = interaction_threshold.k_diff_targets

        # 步骤 1.5：按用户活跃度 activity_level（Profile 原始值，截断到 [0,1]）> 0.7 用轻量版，否则严版
        raw_act = self.profile.get_data("activity_level", 0.0) if self.profile else 0.0
        try:
            _activity = float(raw_act)
        except (TypeError, ValueError):
            _activity = 0.0
        _activity = max(0.0, min(1.0, _activity))

        topic_txt = topic_text_from_mention_entries(mention_entries)
        s15_ev = await self.evaluate_step15_policies(topic_text=topic_txt)
        mem = s15_ev.get("memory_nonempty") or {}
        kw = s15_ev.get("keyword") or {}
        emb = s15_ev.get("embedding") or {}
        mem_ok = bool(mem.get("inject"))
        kw_ok = bool(kw.get("inject"))
        emb_ok = bool(emb.get("inject"))

        step15_kw_coaching = (
            "\n\n【话题与 memory — 关键词重叠】\n            "
            "- 当前批次与已存记忆在关键词层面判定为显著相关 → **强烈倾向 comment=false**（易与 memory 中已有表态或同题讨论重复；仅当步骤1.5 明确满足可核验新信息与强动机等破例条件时再考虑 comment=true）；"
        ) if kw_ok else ""
        step15_emb_coaching = (
            "\n\n【话题与 memory — 语义相似】\n            "
            "- 向量相似度达到设定阈值，本批话题与记忆中内容相近 → **强烈倾向 comment=false**（视同同脉络/易重复话题，须严格按步骤1.5 评估是否仍 comment=true）；"
        ) if emb_ok else ""

        if (_activity < 0.65 and emb_ok):
            step15_receive = """
            步骤1.5：对照 memory 做「重复话题」检查（在步骤2之前完成）
            - 默认 comment=false。若当前内容与 memory 指向**同一事件/同一问题/同一争议脉络**，或与 memory 中重叠的词超过1个，无论你是否**已在同类内容上转发、表态过**，则**几乎必须保持comment=false**。
            - **若要破例，须同时满足以下两项，缺一不可：**
              · **（1）可核对的新信息点**：`decision_reason` 须在**单句**内写清相对 memory、帖中**独有**且可指认的一条新增信息（须出现具体人/机构/日期/数字/规则名或链接类标识之一）；不得单独用「新细节」「新进展」「新讨论点」「又一例」「再关注」「同类再发酵」「略多一句」等空话充数。
              · **（2）强动机**：**强烈情绪动机**（同句或紧邻句须点明具体情绪落点，禁止空泛「有感触」「想说两句」）**或** **明确关系动机**（须点明为维护/帮扩/站队**具体的**互关、关注或好友，写清对象，禁止笼统「支持一下」）。
            -**同时**具备可核对新信息 **与**（强烈情绪 **或** 明确关系动机），仅有情绪/关系而无新信息、或仅有新信息而无强情绪/关系动机，均 comment=false，避免同题刷屏。
            - **memory 越多、越要克制**：即便（1）（2）在字面上都能凑上，仍应把「本条 comment=true」当成**小概率事件**——默认继续 comment=false；仅当新信息**明显升级**（例如改变事件阶段、推翻或修正你 memory 中的既有判断、或出现关键新主体/新规则）时才可破例，禁止「勉强达标就评一句」。
            - **memory_reflection 禁止自相矛盾**：先写「与 memory 重叠/同一话题/已讨论过」等，又用无（1）+（2）支撑的转折暗示可以评论——一律视为无效；若判定重叠或几乎 comment=false，memory_reflection 须**通篇**结论为倾向沉默或明确无新信息，不得以模糊语气自我放行。
            将上述结论简要写入 memory_reflection；不转发时 decision_reason 须点明「与 memory 重叠/已表态/无新信息/缺新信息或缺强动机」等。
            """
            step2_zero_rec = "0. 未落入步骤1.5 的「几乎确定comment=false」情形，或你已写出可核验的新增信息点（仅换说法不算）；"
            mem_refl_rec = "2-3句。有同题时说明是否重叠、是否几乎应comment=false"
        else:
            step15_receive = ""
            step2_zero_rec = "0. 未落入步骤1.5 的「几乎确定comment=false」情形，或你已写出可感知的新事实、新角度或新推理（仅换说法不算）"
            mem_refl_rec = "1-2句。无已存储记忆可对照时写「无相关记忆/首次接触」即可"

        step15_mention_block = step15_receive
        mem_refl_mention_hint = mem_refl_rec

        if time_module_str and _activity < 0.75:
            time_coaching_block = (
                "【仿真时间与时效】\n            "
                + time_module_str
                + " - 若上文含 **【警告】** 或写明时效已明显减弱、倾向不回复 → **强烈倾向 repost=false**（该条不进候选池）；"
            )
        else:
            time_coaching_block = ""

        instruction = f"""下面有多条提醒，请按顺序对每条提醒分别给出一个决策（是否回复、回复内容等）。decisions 数组与提醒顺序一致，第 i 个元素对应第 i 条提醒。

        请基于 Profile、historical_summary、memory 与 Observation 中的关系信息，完成“是否回复/回复内容”判断与生成。

        步骤1：对每条推荐先设默认状态
        - "comment": false
        - "parent_comment_id": null
        - "comment_content": ""

        **comment 的数量决策仅由步骤2/3/4决定，不得因步骤6改变；**
        {step15_mention_block}

        步骤2：兴趣判断
        - 通读本批次所有推荐内容，判断哪些笔记能够进入候选池，当且仅当同时满足以下条件时，再将该笔记改为 comment=true：
            {step2_zero_rec}
            1. **对笔记感兴趣**：整帖与 Profile/historical_summary/memory 高度相关
            2. **对某条楼中评论感兴趣**：未必对整帖强兴趣，但**某条具体评论**与 Profile/historical_summary/memory 高度相关，且你愿意针对该评论做**增量回应**（补充、纠错、追问、共情、接梗）
            3. 互动对象关系与场景合适（关注关系优先）；
            4. 表达目的明确（补充事实、推进互动）
        - 进入候选池时，请在心中标记：本轮互动是「对帖」还是「对某条评论」；若选「对某条评论」，必须在后续 JSON 里使用非空的 parent_comment_id（见步骤4）。
        - 对不感兴趣的笔记：该笔记不进入候选池，comment=false。

        步骤3：选取 0～{k_diff_targets} 目标笔记
        - {k_diff_targets} 表示本轮**最多**对多少个**不同** note_id 置 comment=true；实际条数可为 0～{k_diff_targets} 中任意值，**不得**为凑满条数而放宽步骤1.5/步骤2。
        - 仅在步骤2 的候选池内、按你的排序取前若干条作为“目标 note_id 集合”，且条数不超过 {k_diff_targets}；若候选少于 {k_diff_targets}，有多少算多少；其余笔记保持 comment=false。

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
        "memory_reflection": "{mem_refl_mention_hint}",
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
        {step15_kw_coaching}{step15_emb_coaching}
        {time_coaching_block}"""

        mention_note_id_set: Set[str] = {
            str(e["note_id"]).strip()
            for e in mention_entries
            if isinstance(e, dict) and e.get("note_id")
        }

        # 生成决策（模型偶发截断/非法 JSON；捕获异常避免 Task exception was never retrieved）
        try:
            response = await self.generate_reaction(instruction, observation)
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning(
                f"Step {current_step}/{max_step}: UserAgent {self.profile_id} handle_mention "
                f"generate_reaction JSON 解析失败: {e}，使用全 false 回退决策"
            )
            response = self._fallback_mention_reaction_dict(mention_entries)
        except Exception as e:
            logger.error(
                f"Step {current_step}/{max_step}: UserAgent {self.profile_id} handle_mention "
                f"generate_reaction 失败: {e}，使用全 false 回退决策"
            )
            response = self._fallback_mention_reaction_dict(mention_entries)

        response = self._normalize_llm_reaction(response)

        events_to_send = []
        # 处理搜索决策
        search = response.get("search", False)
        if search:
            # 发送事件给推荐系统：请求“指定算法”的推荐（算法类型由代码固定指定）
            search_algorithm_types = self.default_search_types
            if not isinstance(search_algorithm_types, list) or not search_algorithm_types:
                raise ValueError("default_search_types must be a non-empty list")
            allowed_search_types = set(self.search_map.keys())

            # 遍历所有指定算法类型，发送事件给推荐系统
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

                # 获取用户画像
                profile_payload = {}
                if self.profile is not None:
                    try:
                        profile_payload = dict(self.profile.get_profile(include_private=True) or {})
                    except Exception:
                        logger.warning("Failed to serialize profile via get_profile(), fallback to empty payload.")
                logger.info(f"Sending SearchEvent to RecommenderAgent {mapped_id} for search algorithm type {search_algorithm_type}")
                events_to_send.append(SearchEvent(
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

        # 处理保持关注决策
        if self.profile:
            keep_ids = response.get("keep_following_note_ids", [])
            if isinstance(keep_ids, list) and keep_ids:
                # 只允许本批次内的 note_id，且最多 1 个
                valid_keep_ids = []
                for keep_note_id in keep_ids:
                    if str(keep_note_id).strip() in mention_note_id_set:
                        valid_keep_ids.append(keep_note_id)
                if valid_keep_ids:
                    self.profile.update_data("keep_following_note_ids", valid_keep_ids[:1])
                else:
                    self.profile.update_data("keep_following_note_ids", [])

        # 处理回复决策
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

                # 检查 note_id 是否合法（应该与 mention_note_id 一致）
                if note_id != mention_note_id:
                    logger.warning(f"Note {note_id} does not match mention note {mention_note_id}, skipping")
                    continue

                # 解析评论内容中的@用户，将 @id 替换为 @昵称，并返回用户ID列表
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
                    canonical_parent_id, parent_entry = UserAgent._resolve_parent_comment_entry(
                        comments, parent_comment_id
                    )
                    if isinstance(parent_entry, dict):
                        comment_author_id = parent_entry.get("user_id")
                    mentioned_user_ids = [
                        uid for uid in mentioned_user_ids if uid != comment_author_id
                    ]
                mention_count = len(mentioned_user_ids)

                # 如果为回复评论，检查父评论ID是否合法
                if parent_comment_id is not None:
                    if canonical_parent_id is None:
                        logger.warning(
                            f"Parent comment {parent_comment_id} not found on note {note_id}, skipping reply"
                        )
                        continue
                    # 生成唯一的评论ID
                    comment_id = self._generate_comment_id()
                    
                    success = await self.add_env_comments(note_id, {
                        "comment_id": comment_id,
                        "timestamp": self._random_comment_timestamp(
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
                    
                    # 向评论作者发送提醒
                    if comment_author_id and comment_author_id != user_id:  # 不给自己发提醒
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
                    
                    # 向笔记作者发送提醒
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
                    # 如果为评论笔记，添加评论
                    comment_id = self._generate_comment_id()
                    success = await self.add_env_comments(note_id, {
                        "comment_id": comment_id,
                        "timestamp": self._random_comment_timestamp(
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

                    # 向笔记作者发送提醒
                    if note_author_id and note_author_id != user_id:    # 不给自己发提醒
                        # 生成提醒信息
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

                # 发送MentionEvent给被@的用户
                if mentioned_user_ids:
                    # 为每个被@的用户创建MentionEvent
                    for mentioned_user_id in mentioned_user_ids:
                        if mentioned_user_id and mentioned_user_id != user_id:  # 不给自己发提醒
                            # 创建@事件，发送给被@的用户
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
                # 无论本条是否回复、是否中途 continue，处理完都从 mention_pool 删除
                success = await self.update_env_mention_pool(f"{user_id}.{mention_comment_id}", {
                    "action": "delete",
                    "mention_message": None
                })
                if not success:
                    logger.error(f"Failed to update mention pool for comment {mention_comment_id}")
                else:
                    logger.info(f"Step {current_step}/{max_step}: User {user_id} deleted mention {mention_comment_id} from pool (processed)")

        # 如果有回复，通过事件通知环境更新内容池
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
    