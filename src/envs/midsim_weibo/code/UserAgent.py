from typing import Any, List, Optional, Dict, Set, Tuple, Union
from collections import defaultdict
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
    topic_text_from_blogs_chunk,
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
        self.register_event("StartEvent", "get_recommendations_and_mentions")  
        self.register_event("StartEvent", "generate_memory_from_own_blogs")
        self.register_event("SocialRecommendationEvent", "receive_recommendation")
        self.register_event("AlgorithmRecommendationEvent", "receive_recommendation")
        self.register_event("SearchRecommendationEvent", "receive_recommendation")
        self.register_event("KeepFollowingEvent", "receive_recommendation")
        self.register_event("MentionEvent", "handle_mention")

        self.register_event("AddRepostResponseEvent", "handle_add_repost_response")
        self.register_event("MentionPoolUpdateResponseEvent", "handle_update_mention_pool_response")
        self._repost_add_futures: Dict[str, Future] = {}
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

        # 系统内“搜索类型 -> 搜索器智能体ID”显式映射（按环境约定维护）
        self.search_map: Dict[str, str] = {
            "Random Search": "search_agent_0001",
            "Hot Search": "search_agent_0002",
            "Relevant Search": "search_agent_0003",
            "LLM Search": "search_agent_0004",
        }
        self.default_search_types: List[str] = ["Relevant Search"]

        # 推荐 observation 时间锚点：仅第一次成功解析到 chunk 内最早发帖时间时写入，之后不再覆盖
        self._recommendation_earliest_post_anchor_ms: Optional[float] = None

    @staticmethod
    def _remap_activity_level(activity_level: float, out_min: float = 0.4, out_max: float = 0.8) -> float:
        """
        将 activity_level 先截断到 [0, 1]，再线性重映射到 [out_min, out_max]。
        """
        clamped = max(0.0, min(1.0, activity_level))
        return out_min + (out_max - out_min) * clamped

    async def _is_official_by_agent_field(self) -> Optional[bool]:
        """
        通过 agent 字段判断是否官方号。
        仅识别 agent.is_official 这个属性：
        - 若 agent 为 dict 且包含 "is_official" 键：视为官方号（不依赖其值）
        - 若不包含该键或 agent 缺失：返回 None（未知/未标注）
        """
        is_official = self.profile.get_data("is_official", False)

        return True if is_official else False

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

        if (await self._is_official_by_agent_field()) is True:
            logger.info(
                f"Step {current_step}/{max_step}: UserAgent {self.profile_id} detected official account by agent field, skip recommendations and mentions"
            )
            return []

        events: List[Event] = []

        # 从 event（StartEvent）的 current_blogs 中筛选：关注用户的发帖，且时间在 [上次登录, 当前时间] 之间
        current_blogs = getattr(event, "current_blogs", None) or {}
        if not isinstance(current_blogs, dict):
            current_blogs = {}

        # 发送事件给推荐系统：请求“指定算法”的推荐（算法类型由代码固定指定）
        fixed_algorithm_types = self.default_algorithm_types
        if not isinstance(fixed_algorithm_types, list) or not fixed_algorithm_types:
            raise ValueError("default_algorithm_types must be a non-empty list")
        allowed_algorithm_types = set(self.recommender_map.keys())

        # 获取已推荐过的内容ID集合（从profile中读取）
        recommended_blog_ids = set(self.profile.get_data("recommended_blog_ids", [])) if self.profile else set()

        # 获取用户画像
        profile_payload = {}
        if self.profile is not None:
            try:
                profile_payload = dict(self.profile.get_profile(include_private=True) or {})
            except Exception:
                logger.warning("Failed to serialize profile via get_profile(), fallback to empty payload.")
        
        # 遍历所有指定算法类型，发送事件给推荐系统（每种算法独立以 80% 概率触发）
        _algorithm_request_prob = 1
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
            if random.random() > _algorithm_request_prob:
                logger.debug(
                    f"Step {current_step}/{max_step}: UserAgent {self.profile_id} "
                    f"skip GetAlgorithmRecomendationEvent for {fixed_algorithm_type!r} "
                    f"(p={_algorithm_request_prob:.0%} not drawn)"
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
                    current_blogs=current_blogs,
                    recommended_blog_ids=recommended_blog_ids,
                    algorithm_type=fixed_algorithm_type,
                )
            )

        # 社交推荐
        current_ts = int(event.timestamp) if getattr(event, "timestamp", None) is not None else 0
        follow_ids = await self.get_data("follow_ids", [])
        follow_set = set(follow_ids) if isinstance(follow_ids, (list, tuple)) else set()

        # 上次登录时间：优先用 profile 记录。无记录时不可用 0 作为下界（否则 last_login<=t 会从纪元起整表扫进关注流，与算法 limit=3 完全不对齐）。
        # 若记录早于当前时刻 3 天以上，社交窗口下界截断为「当前时刻前推 3 天」。
        last_login = 0
        three_days_sec = 7 * 86400
        if self.profile:
            raw = self.profile.get_data("last_login_timestamp")
            logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} last_login_timestamp: {raw}")
            if raw is not None and isinstance(raw, (int, float)) and int(raw) > 0:
                last_login = int(raw)
                if current_ts > 0:
                    lower = current_ts - three_days_sec
                    if last_login < lower:
                        last_login = lower
            else:
                last_login = 0

        step_duration = int(getattr(event, "timestamp_duration", 0) or 0)
        window_end = current_ts + step_duration

        # 与 receive_recommendation._filter_recommendations 对齐：已进过全局「已推荐」集合的不进入关注流
        seen_rec_ids: Set[str] = {
            str(x).strip() for x in recommended_blog_ids if x is not None and str(x).strip()
        }

        # 关注流条数不受兴趣推荐 limit=3 约束；若需与算法可比，请另行在下游统计时只计 AlgorithmRecommendationEvent。
        recommendations = {}
        for blog_id, blog in current_blogs.items():
            if not isinstance(blog, dict):
                continue
            author_id = blog.get("user_id") or blog.get("author_id")
            if author_id not in follow_set:
                continue
            t = blog.get("time") or blog.get("create_time")
            if t is None:
                continue
            try:
                t = int(t)
            except (TypeError, ValueError):
                continue
            if t >= 10**12:
                t //= 1000
            # current_blogs：time < current_ts+duration；关注流再按 last_login 截断（均为 Unix 秒）
            if last_login <= t < window_end:
                sid = str(blog_id).strip() if blog_id is not None else ""
                if not sid or sid in seen_rec_ids:
                    continue
                chain_ids: Set[str] = set()
                reposted_path = blog.get("reposted_path", [])
                if isinstance(reposted_path, list):
                    chain_ids.update(
                        str(x).strip() for x in reposted_path if x is not None and str(x).strip()
                    )
                reposted_blog_id = blog.get("reposted_blog_id")
                if reposted_blog_id is not None and str(reposted_blog_id).strip():
                    chain_ids.add(str(reposted_blog_id).strip())
                if chain_ids.intersection(seen_rec_ids):
                    continue
                recommendations[blog_id] = blog

        # 取 mentions 给自己发 MentionEvent（与 acl 场景一致：>10 时随机保留 10 条并从 mention_pool 删除其余；≤10 时用 current_blogs 刷新 blog）
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
                blog = mm.get("blog")
                if isinstance(blog, dict):
                    bid = blog.get("blog_id") or blog.get("id")
                    if bid is not None:
                        bid_key = str(bid).strip()
                        pool_blog = current_blogs.get(bid_key) if isinstance(current_blogs, dict) else None
                        if isinstance(pool_blog, dict):
                            pb = dict(pool_blog)
                            reposted_blog_id = pb.get("reposted_blog_id", "")
                            if reposted_blog_id and reposted_blog_id in current_blogs:
                                pb["reposted_blog"] = current_blogs[reposted_blog_id]
                            mm["blog"] = pb
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
        keep_following_blogs = {}
        if self.profile and self.profile.get_data("keep_following_blog_ids", []) is not None:
            keep_ids = self.profile.get_data("keep_following_blog_ids", []) or []
            if isinstance(keep_ids, (list, tuple)) and keep_ids:
                readded = 0
                for keep_blog_id in keep_ids:
                    if keep_blog_id in recommendations:
                        continue
                    blog = current_blogs.get(keep_blog_id)
                    if not isinstance(blog, dict):
                        continue
                    author_id = blog.get("user_id") or blog.get("author_id")
                    if author_id not in follow_set:
                        continue
                    reposted_blog_id = blog.get("reposted_blog_id", "")
                    if reposted_blog_id and reposted_blog_id in current_blogs:
                        blog["reposted_blog"] = current_blogs[reposted_blog_id]
                    keep_following_blogs[keep_blog_id] = blog
                    readded += 1
                if readded > 0:
                    logger.info(
                        f"Step {current_step}/{max_step}: UserAgent {self.profile_id} re-added keep_following blogs: {readded}"
                    )
                # 一次性使用：无论是否成功塞回，都清空，避免无限循环
                self.profile.update_data("keep_following_blog_ids", [])
                
        if len(keep_following_blogs) > 0:
            events.append(KeepFollowingEvent(
                from_agent_id=self.profile_id,
                to_agent_id=self.profile_id,
                timestamp=current_ts,
                timestamp_duration=int(getattr(event, "timestamp_duration", 0) or 0),
                current_step=current_step,
                max_step=max_step,
                recommendations=keep_following_blogs,
            ))

        # 更新「上次处理到的仿真时刻」为当前时间窗右端（与 time < ts+duration 对齐）
        if self.profile:
            _d = int(getattr(event, "timestamp_duration", 0) or 0)
            self.profile.update_data("last_login_timestamp", current_ts)
            logger.info(
                f"Step {current_step}/{max_step}: UserAgent {self.profile_id} update last_login_timestamp: {current_ts + _d}"
            )

        # 有关注流推荐时再发 SocialRecommendationEvent（80% 概率触发）
        _social_recommendation_prob = 1
        if recommendations and len(recommendations) > 0:
            if random.random() > _social_recommendation_prob:
                logger.debug(
                    f"Step {current_step}/{max_step}: UserAgent {self.profile_id} "
                    f"skip SocialRecommendationEvent (n={len(recommendations)}, "
                    f"p={_social_recommendation_prob:.0%} not drawn)"
                )
            else:
                logger.info(
                    f"Step {current_step}/{max_step}: UserAgent {self.profile_id} send SocialRecommendationEvent, length of recommendations: {len(recommendations)}"
                )

                # 补充reposted_blog字段
                for blog_id, blog in recommendations.items():
                    reposted_blog_id = blog.get("reposted_blog_id", "")
                    if reposted_blog_id and reposted_blog_id in current_blogs:
                        blog["reposted_blog"] = current_blogs[reposted_blog_id]

                events.append(
                    SocialRecommendationEvent(
                        from_agent_id=self.profile_id,
                        to_agent_id=self.profile_id,
                        timestamp=current_ts,
                        timestamp_duration=int(getattr(event, "timestamp_duration", 0) or 0),
                        current_step=current_step,
                        max_step=max_step,
                        recommendations=recommendations,
                    )
                )
        return events

    @staticmethod
    def _event_abs_ts_to_sec(raw: Any) -> float:
        """与 SimEnv._normalize_sim_timestamps_to_seconds 一致：StartEvent.timestamp 统一为 Unix 秒（兼容毫秒）。"""
        if raw is None:
            return 0.0
        try:
            x = int(float(raw))
        except (TypeError, ValueError):
            return 0.0
        if x <= 0:
            return 0.0
        if x >= 10**12:
            return float(x // 1000)
        return float(x)

    @staticmethod
    def _event_cap_ts_to_sec(raw: Any) -> Optional[float]:
        """simulation_cap_timestamp 与 lo 同单位（秒）；兼容毫秒。"""
        if raw is None:
            return None
        try:
            x = float(raw)
        except (TypeError, ValueError):
            return None
        if x <= 0:
            return None
        if x >= 10**12:
            return x / 1000.0
        return x

    @staticmethod
    def _blog_post_time_in_window(blog: Dict[str, Any], lo: float, hi: float) -> bool:
        """发帖时间落在 [lo, hi) 内（lo/hi 为 Unix 秒；兼容旧毫秒 time）。"""
        if lo >= hi:
            return False
        raw = blog.get("time", blog.get("create_time"))
        try:
            t = float(raw)
        except (TypeError, ValueError):
            return False
        if t >= 10**12:
            t = t / 1000.0
        return t >= lo and t < hi

    @staticmethod
    def _is_repost_of_other_blog(blog_id: str, blog: Dict[str, Any]) -> bool:
        pid = blog.get("reposted_blog_id")
        if pid is None:
            return False
        ps = str(pid).strip()
        if not ps:
            return False
        return ps != str(blog_id).strip()

    @staticmethod
    def _parse_blog_time_ms(raw: Any) -> Optional[float]:
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
        与 multi_channel_information_diffusion 一致：时间模块内统一用毫秒。
        微博环境 event.timestamp 常为 Unix 秒（<1e12）；笔记 time 可能为毫秒，由 _parse_blog_time_ms 统一。
        """
        if ts is None or not isinstance(ts, (int, float)):
            return 0
        x = float(ts)
        if x < 1e12:
            return int(x * 1000)
        return int(x)

    @staticmethod
    def _content_dicts_for_time_module(chunk: Union[Dict[str, Any], List[Any]]) -> List[Dict[str, Any]]:
        """推荐 chunk（blog_id -> blog）或 mention_entries（含 mention_blog）。"""
        items: List[Dict[str, Any]] = []
        if isinstance(chunk, dict):
            for v in chunk.values():
                if isinstance(v, dict):
                    items.append(v)
        elif isinstance(chunk, list):
            for entry in chunk:
                if not isinstance(entry, dict):
                    continue
                inner = entry.get("mention_blog")
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
        ref_ms 通常取本轮 event 的 current_timestamp（仿真窗口起点）。
        """
        times_ms: List[float] = []
        for blog in self._content_dicts_for_time_module(chunk):
            t = UserAgent._parse_blog_time_ms(
                blog.get("time", blog.get("create_time"))
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
        stale_days = 3.0
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
                "若无强动机，请优先倾向repost=false，保持沉默更合理。"
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

    async def generate_memory_from_own_blogs(self, event: Event) -> List[Event]:
        """
        StartEvent 多播处理：从 current_blogs 中筛选自己的微博，并生成一条反思记忆。
        复盘仅针对本人原创帖（reposted_blog_id 为空、或等于本帖 id；与 metrics 一致）；其后对时间窗内本人帖先按概率子采样，再 **一次** 大模型批量决策「追加转发」补评（含本人转发帖），与复盘内容独立。
        该逻辑不受 _should_activate_this_round() 影响。
        """
        current_step = getattr(event, "current_step", 1)
        max_step = getattr(event, "max_step", 8)
        user_id = await self.get_data("id")
        if not user_id:
            logger.error(
                f"Step {current_step}/{max_step}: User {user_id} failed to get user_id"
            )
            return []
        uid_norm = str(user_id).strip()

        current_ts = self._event_abs_ts_to_sec(getattr(event, "timestamp", None))
        duration = int(getattr(event, "timestamp_duration", 0) or 0)

        current_blogs = getattr(event, "current_blogs", None) or {}
        if not isinstance(current_blogs, dict) or not current_blogs:
            logger.error(
                f"Step {current_step}/{max_step}: User {user_id} failed to get current_blogs"
            )
            return []

        # 本人发的所有帖（不要求 reposted_blog_id 是否为空）。
        # current_blogs 与 SimEnv._build_current_blogs_subset 一致：仅要求发帖 time < hi，无下界 lo。
        # 不要再套 [lo,hi)，否则第 2 步起 lo 前移会把仍在本轮可见列表里的旧原创帖全部滤掉。
        own_entries: List[Tuple[str, Dict[str, Any]]] = []
        for blog_id, blog in current_blogs.items():
            if not isinstance(blog, dict):
                continue
            b_uid = str(blog.get("user_id") or blog.get("author_id") or "").strip()
            if b_uid != uid_norm:
                continue
            own_entries.append((blog_id, blog))

        lo = float(current_ts)
        dur_sec = float(duration)
        cap = getattr(event, "simulation_cap_timestamp", None)
        cap_f = self._event_cap_ts_to_sec(cap)
        if cap_f is not None:
            hi = min(lo + dur_sec, cap_f)
        else:
            hi = lo + dur_sec
        if hi <= lo:
            # 常见：最后一轮之后 SimEnv 将 timestamp_duration 置为 0，仍多播 StartEvent，时间窗退化为空。
            # 少见：timestamp 与 cap 单位不一致（未归一成秒）会导致 hi<<lo，上面已对 event 侧做了秒级归一。
            if dur_sec <= 0:
                logger.info(
                    f"Step {current_step}/{max_step}: User {user_id} skip own-blogs memory: "
                    f"empty time window (timestamp_duration<=0, e.g. after max_step). lo={lo}, hi={hi}"
                )
            else:
                logger.warning(
                    f"Step {current_step}/{max_step}: User {user_id} empty time window hi<=lo: "
                    f"lo={lo}, hi={hi}, duration={dur_sec}, simulation_cap_timestamp={cap!r}"
                )
            return []

        if not own_entries:
            logger.info(
                f"Step {current_step}/{max_step}: User {user_id} no own posts in current_blogs "
                f"(subset time < hi≈{hi}; lo={lo} 仅用于与 SimEnv 对齐空窗判断)"
            )
            return []

        # 仅原创帖用于复盘与 recommended 去重；所有本人帖 id 都加入推荐排除，避免再进候选
        own_originals = [
            (bid, b)
            for bid, b in own_entries
            if not self._is_repost_of_other_blog(str(bid), b)
        ]
        if self.profile:
            existing_recommended = self.profile.get_data("recommended_blog_ids", []) or []
            if not isinstance(existing_recommended, list):
                existing_recommended = list(existing_recommended) if existing_recommended else []
            own_blog_ids = [str(blog_id) for blog_id, _ in own_entries if blog_id]
            if own_blog_ids:
                merged = list(dict.fromkeys(existing_recommended + own_blog_ids))
                self.profile.update_data("recommended_blog_ids", merged)

        own_blogs_for_prompt: List[Dict[str, Any]] = []
        for blog_id, blog in own_originals:
            own_blogs_for_prompt.append({
                "blog_id": blog_id,
                "content": blog.get("content", ""),
                "time": blog.get("time", blog.get("create_time", 0)),
            })

        if own_entries and not own_originals:
            logger.info(
                f"Step {current_step}/{max_step}: User {user_id} skip self-reflection prompt: "
                f"{len(own_entries)} own post(s) in current_blogs but all forward others (reposted_blog_id≠self); "
                f"复盘仅对原创帖触发。"
            )

        if own_blogs_for_prompt:
            instruction = (
                "你正在复盘自己最近发布的内容。请基于这些帖子，沉淀一条第一人称记忆，"
                "重点总结：你最近持续关注的话题、表达风格、以及后续互动中可复用的表达策略。"
            )
            observation = f"你最近发布的微博：\n{json.dumps(own_blogs_for_prompt, ensure_ascii=False, indent=2)}"
            reaction = {
                "task": "self_reflection_on_own_blogs",
                "own_blog_count": len(own_blogs_for_prompt),
                "highlights": own_blogs_for_prompt,
            }
            try:
                memory_text = await self.generate_memory(instruction, observation, reaction)
                if memory_text:
                    logger.info(
                        f"Step {current_step}/{max_step}: User {user_id} generated self-memory from "
                        f"{len(own_blogs_for_prompt)} own blogs, memory_text: {memory_text}"
                    )
            except Exception as e:
                logger.error(
                    f"Step {current_step}/{max_step}: User {user_id} failed to generate memory from own blogs: {e}"
                )

        followup_prob = float(os.environ.get("WEIBO_SELF_FOLLOWUP_REPOST_PROB", "0.001"))
        if followup_prob <= 0.0:
            logger.error(
                f"Step {current_step}/{max_step}: User {user_id} failed to get followup_prob"
            )
            return []

        max_batch = max(1, int(os.environ.get("WEIBO_SELF_FOLLOWUP_MAX_BATCH", "16")))
        candidates = list(own_entries)
        random.shuffle(candidates)
        subs: List[Tuple[Any, Dict[str, Any]]] = [
            (bid, b) for bid, b in candidates if random.random() < followup_prob
        ]
        if len(subs) > max_batch:
            subs = random.sample(subs, max_batch)
        if not subs:
            return []

        user_nickname = await self.get_data("nickname", "") or ""
        ip_location = await self.get_data("ip_location", "") or ""
        current_ts = int(event.timestamp) if getattr(event, "timestamp", None) is not None else 0
        step_duration = int(getattr(event, "timestamp_duration", 0) or 0)
        env = getattr(self, "env", None)

        posts_payload = [
            {"blog_id": str(bid), "content": (b.get("content") or "")[:800]}
            for bid, b in subs
        ]
        n_posts = len(posts_payload)
        posts_json = json.dumps({"self_posts": posts_payload}, ensure_ascii=False, indent=2)
        follow_ids = await self.get_data("follow_ids", [])
        fan_ids = await self.get_data("fan_ids", [])
        mentionable_users = self._get_mentionable_users(follow_ids, fan_ids)
        mentionable_users_str = json.dumps(
            mentionable_users.get("follows", []), ensure_ascii=False, indent=2
        )
        observation = f"""【场景】对本人微博追加转发（罕见，批量一次决策）
        {posts_json}

        可@的用户列表（追加转发正文禁止出现「@」，仅作关系参考）：
        {mentionable_users_str}"""
        instruction = f"""这是一次罕见的「对本人历史微博追加转发」批量决策（共 {n_posts} 条）。
        请结合 Observation 中的 self_posts 与 memory：对**每一条**判断是否值得追加转发（新事实、新进展或态度变化）。
        若值得：repost=true 并填写 repost_content（≤40字，口语、第一人称）；否则 repost=false，repost_content 为空。
        repost_content 中不允许出现「@」。

        请按以下 JSON 返回（字段顺序固定），并用 ```json 代码块包裹。
        **decisions 必须恰好包含 {n_posts} 条**，且与 self_posts **顺序一致**；第 i 条的 blog_id 必须等于 self_posts[i].blog_id。

        ```json
        {{
        "persona_understanding": "1-2句。概括身份、兴趣、语言风格（简短）",
        "content_understanding": "1-2句。概括与这批本人帖的整体相关性",
        "memory_reflection": "2-3句。哪些值得追评、哪些应沉默",
        "decisions": [
        {{
        "blog_id": "<与 self_posts[0].blog_id 一致>",
        "repost": false,
        "repost_content": "",
        "decision_reason": "（1句，<20字）",
        "expression_reason": "",
        "mention_reasoning": []
        }}
        ]
        }}
        ```
        （decisions 内请展开为 {n_posts} 个对象，勿省略。）"""

        if env is not None and hasattr(env, "notify_agent_busy"):
            await env.notify_agent_busy()
        try:
            response = await self.generate_reaction(instruction, observation)
        finally:
            if env is not None and hasattr(env, "notify_agent_idle"):
                await env.notify_agent_idle()

        if not isinstance(response, dict):
            return []
        decisions_raw = response.get("decisions")
        if not isinstance(decisions_raw, list):
            return []

        by_blog_id: Dict[str, Dict[str, Any]] = {}
        for d in decisions_raw:
            if isinstance(d, dict):
                bk = str(d.get("blog_id") or "").strip()
                if bk:
                    by_blog_id[bk] = d

        for blog_id, blog in subs:
            bid = str(blog_id)
            dec = by_blog_id.get(bid)
            if not isinstance(dec, dict):
                continue
            if not dec.get("repost", False):
                continue
            repost_content = (dec.get("repost_content") or "").strip()
            repost_content = " ".join(repost_content.splitlines()).strip()[:200]
            if not repost_content:
                continue

            repost_id = self._generate_repost_id()

            if str(blog.get("reposted_blog_id") or "").strip():
                reposted_path = list(blog.get("reposted_path", []))
                reposted_path.append(str(blog_id))
                reposted_path = list(dict.fromkeys(reposted_path))
                reposted_blog_id = blog.get("reposted_blog_id")
                reposted_user_id = blog.get("user_id")
                blog_content = blog.get("content", "")
                emit_content = f"{repost_content}//@{reposted_user_id}: {blog_content}"
                blog_author_ids: List[str] = []
                for part in str(emit_content).split("//"):
                    match = re.search(r"@([^:：\s]+)\s*[:：]", part)
                    if match:
                        blog_author_ids.append(match.group(1))
            else:
                reposted_path = [str(blog_id)]
                reposted_blog_id = blog_id
                blog_author_ids = [str(blog.get("user_id") or user_id)]

            mention_count = 0
            success = await self.add_env_reposts(repost_id, {
                "blog_id": repost_id,
                "content": emit_content if str(blog.get("reposted_blog_id") or "").strip() else repost_content,
                "time": self._random_repost_timestamp(blog, current_ts, step_duration),
                "ip_location": ip_location,
                "user_id": user_id,
                "nickname": user_nickname,
                "at_count": mention_count,
                "reposted_blog_id": reposted_blog_id,
                "reposted_path": reposted_path,
                "repost_count": 0,
                "repost_ids": [],
            })
            if not success:
                logger.error(
                    f"Step {current_step}/{max_step}: add_env_reposts failed for self_followup_repost {repost_id}"
                )
                continue

            for blog_author_id in blog_author_ids:
                if blog_author_id and blog_author_id != user_id:
                    ok = await self.update_env_mention_pool(f"{blog_author_id}.{repost_id}", {
                        "action": "add",
                        "mention_message": {"blog_id": repost_id, "mention_type": "repost"},
                    })
                    if not ok:
                        logger.error(
                            f"Step {current_step}/{max_step}: mention_pool update failed for self_followup {repost_id}"
                        )
                    else:
                        logger.info(
                            f"Step {current_step}/{max_step}: User {user_id} self_followup_repost on {blog_id} "
                            f"-> {repost_id}"
                        )
        return []

    def _generate_repost_id(self) -> str:
        """
        生成 10 位十进制字符串（1_000_000_000～9_999_999_999），用作 content_pool 新转发键。
        """
        return str(secrets.randbelow(9_000_000_000) + 1_000_000_000)
    
    @staticmethod
    def _random_repost_timestamp(
        blog: Dict[str, Any], window_start_sec: int, window_duration_sec: int
    ) -> int:
        """
        转发时间戳：Unix 秒（与微博 create_time 语义一致），落在 [max(发帖时间, 窗口起点), 窗口终点] 内离散均匀随机（整秒，无小数）。
        与 SimEnv 一致：半开区间 [window_start_sec, window_start_sec + window_duration_sec)；
        window_duration_sec==0 时上界退化为 window_start_sec。
        发帖时间可为毫秒（>=1e12）或秒，均归一为秒再抽样。
        """
        if window_start_sec <= 0 and window_duration_sec <= 0:
            return 0
        lo_win = window_start_sec
        if window_duration_sec > 0:
            hi_incl = lo_win + window_duration_sec - 1
        else:
            hi_incl = lo_win
        post = (
            blog.get("time", blog.get("create_time"))
            if isinstance(blog, dict)
            else None
        )
        if not isinstance(post, (int, float)):
            return hi_incl
        pt = int(post)
        if pt >= 10**12:
            pt //= 1000
        lo = max(pt, lo_win)
        hi = hi_incl
        if lo > hi:
            lo, hi = hi, lo
        lo_i, hi_i = int(lo), int(hi)
        if hi_i < lo_i:
            return lo_i
        return random.randint(lo_i, hi_i)

    def _parse_mentions(self, repost_content: str, user_id_map: Dict[str, str], mentionable_users: Dict[str, Any]) -> Tuple[str, List[str]]:
        """
        解析评论中的@，提取被@的用户ID，并将 @id 替换为 @昵称
        
        Args:
            repost_content: 评论内容
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
        mentions = re.findall(mention_pattern, repost_content)
        
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
        updated_content = repost_content
        for old_text, new_text in replacements.items():
            updated_content = updated_content.replace(old_text, new_text)
        
        return updated_content, mentioned_user_ids
    
    def _add_recommendations(self, recommendations: Dict[str, Any]) -> None:
        """将给定 blog_id 集合并入 profile 的 recommended_blog_ids（去重保序：先历史后本次）。"""
        if not recommendations or not self.profile:
            return
        recommended_blog_ids = set(self.profile.get_data("recommended_blog_ids", [])) if self.profile else set()
        new_blog_ids = [str(bid) for bid in recommendations.keys() if bid]
        if not new_blog_ids:
            return
        all_recommended = list(recommended_blog_ids) + new_blog_ids
        self.profile.update_data("recommended_blog_ids", all_recommended)
        
    @staticmethod
    def _clip_historical_summary_for_mentionable(text: Any) -> str:
        """可@用户列表里的 historical_summary：超过 100 字时只保留前 50、后 50，中间用 … 连接。"""
        if text is None:
            return ""
        s = str(text)
        if len(s) <= 100:
            return s
        return s[:50] + "…" + s[-50:]

    def _record_recommendations_by_source_step(
        self,
        source_type: str,
        current_step: int,
        recommendations: Dict[str, Any],
        event_timestamp: Any,
    ) -> None:
        """
        按推荐来源 source_type 与仿真轮次 current_step，把本轮 blog_id 追加写入
        profile.recommended_blog_ids_by_channel。

        结构：recommended_blog_ids_by_channel[source_type][str(step)] = [blog_id, ...]
        同一 (source_type, step) 下多次写入时合并列表并去重（保持顺序）。

        若 event_timestamp 可解析为大于 0 的整数，则同步更新 profile.last_login_timestamp。
        最后打日志输出 last_login_timestamp 与完整 recommended_blog_ids_by_channel。
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

        raw_root = self.profile.get_data("recommended_blog_ids_by_channel", {})
        by_ch: Dict[str, Any] = dict(raw_root) if isinstance(raw_root, dict) else {}

        step_map_raw = by_ch.get(st)
        step_map: Dict[str, Any] = (
            dict(step_map_raw) if isinstance(step_map_raw, dict) else {}
        )

        prev_ids = step_map.get(step_key)
        merged: List[str] = list(prev_ids) if isinstance(prev_ids, list) else []
        seen: Set[str] = {str(x).strip() for x in merged if str(x).strip()}
        for blog_id in recommendations.keys():
            sid = str(blog_id).strip()
            if not sid or sid in seen:
                continue
            merged.append(sid)
            seen.add(sid)

        step_map[step_key] = merged
        by_ch[st] = step_map
        self.profile.update_data("recommended_blog_ids_by_channel", by_ch)

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
            f"recommended_blog_ids_by_channel={json.dumps(by_ch, ensure_ascii=False, default=str)}"
        )

    def _record_mentioned_blog_ids_by_channel(
        self,
        current_step: int,
        mention_entries: List[Dict[str, Any]],
        event_timestamp: Any,
    ) -> None:
        """
        按 MentionEvent 中的 mention_type 与当前轮次，把相关 blog_id 追加写入 profile.mentioned_blog_ids_by_channel。

        结构：mentioned_blog_ids_by_channel[mention_type][str(step)] = [blog_id, ...]
        同一 (mention_type, step) 下多次写入时合并列表并去重（保持顺序）。

        若 event_timestamp 可解析为大于 0 的整数，则同步更新 profile.last_login_timestamp。
        最后打日志输出 last_login_timestamp、current_step 与完整 mentioned_blog_ids_by_channel。
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
            nid = entry.get("blog_id")
            sid = str(nid).strip() if nid is not None else ""
            if sid:
                batch[mt].append(sid)

        if not batch:
            return

        raw_root = self.profile.get_data("mentioned_blog_ids_by_channel", {})
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

        self.profile.update_data("mentioned_blog_ids_by_channel", by_ch)

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
            f"mentioned_blog_ids_by_channel={json.dumps(by_ch, ensure_ascii=False, default=str)}"
        )

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
                hn = info.get("historical_blogs")
                if isinstance(hn, dict) and len(hn) > 2:
                    items = list(hn.items())[:2]
                    info["historical_blogs"] = dict(items)
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
                hn = info.get("historical_blogs")
                if isinstance(hn, dict) and len(hn) > 2:
                    items = list(hn.items())[:2]
                    info["historical_blogs"] = dict(items)
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
                hn = info.get("historical_blogs")
                if isinstance(hn, dict) and len(hn) > 2:
                    items = list(hn.items())[:2]
                    info["historical_blogs"] = dict(items)
                mutual_info.append(info)

        mentionable_info["follows"] = follows_info
        mentionable_info["fans"] = fans_info
        mentionable_info["mutual"] = mutual_info
        
        return mentionable_info

    async def add_env_reposts(self, key: str, value: Any, parent_event_id: Optional[str] = None) -> bool:
        """
        添加转发到环境中的数据（使用分布式锁）
        
        Args:
            key: blog_id
            value: 转发数据字典，必须包含 repost_id 字段
            parent_event_id: 父事件ID（可选）
        """        
        # 创建唯一的请求ID
        request_id = f"agent_env_add_reposts_req_{time.time()}_{id(self)}"

        # 创建 Future 用于接收响应
        future = asyncio.Future()
        self._repost_add_futures[request_id] = future

        # 创建添加评论事件
        repost_add_event = AddRepostEvent(
            from_agent_id=self.profile_id,  # 请求来源：当前代理
            to_agent_id="ENV",              # 请求目标：环境（特殊目标）
            source_type="AGENT",            # 源类型：代理
            target_type="ENV",              # 目标类型：环境
            key=key,                        # 数据键
            value=value,                    # 新的数据值
            request_id=request_id,          # 请求ID，用于匹配响应
            parent_event_id=parent_event_id # 父事件ID，用于事件追踪
        )

        # 获取此键的分布式锁
        # 锁ID格式：env_data_add_reposts_lock_{key}，确保每个键有独立的锁
        lock_id = f"env_repost_add_lock_content_pool"
        lock = await get_lock(lock_id)

        try:
            # 仅在与 SimEnv.handle_add_repost_event 相同的锁内派发；释放锁后再等待，避免死锁超时。
            async with lock:
                from onesim.events import get_event_bus
                event_bus = get_event_bus()
                await event_bus.dispatch_event(repost_add_event)

            try:
                if hasattr(self, '_sync_event'):
                    await asyncio.wait_for(self._sync_event.wait(), timeout=30.0)
                    return await future
                return await asyncio.wait_for(future, timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning(f"等待环境转发添加超时: {key}")
                self._repost_add_futures.pop(request_id, None)
                return False
            except Exception as e:
                logger.error(f"添加环境转发时出错: {e}")
                self._repost_add_futures.pop(request_id, None)
                return False
        except Exception as e:
            logger.error(f"获取环境转发添加锁时出错: {e}")
            return False

    async def handle_add_repost_response(self, event: AddRepostResponseEvent) -> None:
        """
        处理传入的转发添加响应事件
        """
        # 检查是否正在等待此响应
        if event.request_id in self._repost_add_futures:
            future = self._repost_add_futures.pop(event.request_id)

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
            key: mentioner_id（例如 "69290e59000000001e034ab4"），会自动转换为 "mention_pool.mentioner_id.blog_id"
            value: mention_pool数据字典，必须包含 mention_key 字段
            parent_event_id: 父事件ID（可选）
        """
        # 将 mentioner_id.blog_id 转换为完整的 key 格式：mention_pool.mentioner_id.blog_id
        full_key = f"mention_pool.{key}"
        lock_key = key.split(".")[0]
        
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
        # 锁ID格式：env_mention_pool_update_lock_{lock_key}，确保每个键有独立的锁
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
                logger.warning(f"等待环境mention_pool更新超时: {key}")
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
        recommended_blog_ids = set(self.profile.get_data("recommended_blog_ids", [])) if self.profile else set()
        
        filtered: Dict[str, Dict[str, Any]] = {}

        for blog_id, rec in recommendations.items():
            if not isinstance(rec, dict):
                logger.warning(f"Recommendation {blog_id} is not a dictionary")
                continue
            
            # 如果笔记ID不存在或已经推荐过，跳过
            if not blog_id or blog_id in recommended_blog_ids:
                logger.info(f"Recommendation {blog_id} is already recommended")
                continue

            # 若转发链上任一 blog_id 已推荐过，也跳过（避免链路重复曝光）
            chain_ids = set()
            reposted_path = rec.get("reposted_path", [])
            if isinstance(reposted_path, list):
                chain_ids.update(str(x) for x in reposted_path if x is not None and str(x).strip())
            reposted_blog_id = rec.get("reposted_blog_id")
            if reposted_blog_id is not None and str(reposted_blog_id).strip():
                chain_ids.add(str(reposted_blog_id))

            hit_chain_ids = chain_ids.intersection(recommended_blog_ids)
            if hit_chain_ids:
                logger.info(
                    f"Recommendation {blog_id} filtered by repost chain history, hit ids: {sorted(hit_chain_ids)}"
                )
                continue

            filtered[blog_id] = rec
            logger.info(f"Recommendation {blog_id} is added to filtered list")

        self._add_recommendations(filtered)
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

        # 将推荐内容加入已推荐列表
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

        # 获取用户信息和可@的用户列表
        user_id = await self.get_data("id")
        user_nickname = await self.get_data("nickname", "")
        ip_location = await self.get_data("ip_location", "")
        current_timestamp = event.timestamp
        current_ts = int(event.timestamp) if getattr(event, "timestamp", None) is not None else 0
        step_duration = int(getattr(event, "timestamp_duration", 0) or 0)
        # 与 diffusion 时间模块对齐：内部统一毫秒；微博 event.timestamp 多为 Unix 秒
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
            has_self_blog = any(
                isinstance(blog, dict) and blog.get("user_id") == user_id
                for blog in chunk.values()
            )
            if has_self_blog:
                source_name = "【⚠ 检测到该转发链包含你自己：默认repost=false】"
            else:
                source_name = "关注流（来自你关注的用户）" if source_type == "social" else "推荐流（来自算法）"

            observation = f"""【场景】你正在用手机刷信息流：大部分内容划走即可，只有偶尔才会停下来打一行字或只是点一下转发。你不是在完成实验任务，也不是写舆情分析。

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

            topic_txt = topic_text_from_blogs_chunk(chunk)
            s15_ev = await self.evaluate_step15_policies(topic_text=topic_txt)
            mem = s15_ev.get("memory_nonempty") or {}
            kw = s15_ev.get("keyword") or {}
            emb = s15_ev.get("embedding") or {}
            mem_ok = bool(mem.get("inject"))
            kw_ok = bool(kw.get("inject"))
            emb_ok = bool(emb.get("inject"))

            step15_kw_coaching = (
                "\n\n【话题与 memory — 关键词重叠】\n            "
                "- 当前批次与已存记忆在关键词层面判定为显著相关 → **强烈倾向 repost=false**（易与 memory 中已有表态或同题讨论重复；仅当步骤1.5 明确满足可核验新信息与强动机等破例条件时再考虑 repost=true）；"
            ) if kw_ok else ""
            step15_emb_coaching = (
                "\n\n【话题与 memory — 语义相似】\n            "
                "- 向量相似度达到设定阈值，本批话题与记忆中内容相近 → **强烈倾向 repost=false**（视同同脉络/易重复话题，须严格按步骤1.5 评估是否仍 repost=true）；"
            ) if emb_ok else ""

            raw_act = self.profile.get_data("activity_level", 0.0) if self.profile else 0.0
            try:
                _activity = float(raw_act)
            except (TypeError, ValueError):
                _activity = 0.0
            _activity = max(0.0, min(1.0, _activity))

            if (_activity < 0.7 and mem_ok):
                step15_receive = """
                步骤1.5：对照 memory 做「重复话题」检查（在步骤2之前完成）
                - 默认 repost=false。若当前内容与 memory 指向**同一事件/同一问题/同一争议脉络**，或与 memory 中重叠的词超过1个，无论你是否**已在同类内容上转发、表态过**，则**几乎必须保持repost=false**。
                - **若要破例，须同时满足以下三项，缺一不可：**
                · **（1）可核对的新信息点**：`decision_reason` 须在**单句**内写清相对 memory、帖中**独有**且可指认的一条新增事实（须出现具体人/机构/日期/数字/规则名或链接类标识之一）；不得单独用「新细节」「新进展」「新讨论点」「又一例」「再关注」「同类再发酵」「略多一句」等空话充数。
                · **（2）强动机**：**强烈情绪动机**（同句或紧邻句须点明具体情绪落点，禁止空泛「有感触」「想说两句」）。
                · **（3）明确扩散动机**：**明确扩散动机**（须点明为维护/帮扩/站队**具体的**互关、关注或好友，写清对象，禁止笼统「支持一下」）。
                - **同时**具备可核对新信息 **与**强烈情绪 **与** 明确扩散动机，仅有情绪/扩散而无新信息、或仅有新信息而无强情绪/扩散动机，均 repost=false，避免同题刷屏。
                - **memory 越多、越要克制**：即便（1）（2）（3）在字面上都能凑上，仍应把「本条 repost=true」当成**小概率事件**——默认继续 repost=false；仅当新信息**明显升级**（例如改变事件阶段、推翻或修正你 memory 中的既有判断、或出现关键新主体/新规则）时才可破例，禁止「勉强达标就评一句」。
                - **memory_reflection 禁止自相矛盾**：先写「与 memory 重叠/同一话题/已讨论过」等，又用无（1）+（2）支撑的转折暗示可以评论——一律视为无效；若判定重叠或几乎 repost=false，memory_reflection 须**通篇**结论为倾向沉默或明确无新信息，不得以模糊语气自我放行。
                将上述结论简要写入 memory_reflection；不转发时 decision_reason 须点明「与 memory 重叠/已表态/无新信息/缺新信息或缺强动机」等。
                """
                step2_rec = "0. 破例须**同时**具备（1）可核对新信息点 **与**（2）强烈情绪动机 **与** （3）明确扩散动机，缺一仍 repost=false；若不触发步骤1.5 的重叠情形，本条可视为已满足；"
                mem_refl_rec = "2-3句。无相关记忆可写「无相关记忆/首次接触」；有同题时说明是否重叠；若重叠倾向不转发，可说明是否仍有一句评论欲（仅评论不转发）"
            else:
                # step15_receive = """
                # 步骤1.5：对照 memory 做「重复话题」检查（在步骤2之前完成）
                # - 若当前内容与 memory 指向同一事件/同一争议点/同一问题，且你在 memory 中的立场、态度或结论与本次若回复会说的内容高度相近 → 将该条倾向repost=false，除非你能明确写出：相对 memory 中已有表达，本次将补充**读者可感知的新事实、新角度或新推理**（例如改变事件阶段、推翻或修正你 memory 中的既有判断、或出现关键新主体/新规则，仅换说法不算）。
                # - 推荐流陌生关系、或你与作者关系较弱时，在「是否破例」上应更保守。
                # 将上述结论简要写入 memory_reflection；不评论时 decision_reason 须点明「与 memory 重叠/已表态/无新信息」等。
                # """
                step15_receive = ""
                step2_rec = "2. 关系与场景合适（关注关系优先）；"
                mem_refl_rec = "2-3句。无相关记忆可写「无相关记忆/首次接触」；有同题时说明是否重叠；若重叠倾向不转发，可说明是否仍有一句评论欲（仅评论不转发）"

            if time_module_str:
                time_coaching_block = (
                    "【仿真时间与时效】\n            "
                    + time_module_str
                    + " - 若上文含 **【警告】** 或写明时效已明显减弱、倾向不回复 → **强烈倾向 repost=false**（该条不进候选池）；"
                )
            else:
                time_coaching_block = ""

            instruction = f"""根据用户的 Profile、historical_summary、memory 和推荐内容，完成转发决策与生成。

            步骤1：对每条推荐先设默认状态
            - "repost": false
            - "repost_content": ""

            **repost 的数量决策仅由步骤2/3决定，不得因步骤5、6改变；**
            {step15_receive}

            步骤2：兴趣判断
            - 默认 repost=false。通读本批次所有推荐内容：若无任何一条能同时通过步骤1.5 破例条件，则**本批全体 repost=false**，不得因「略有相关」放行。
            - 仅当**下列全部满足**且**不违反步骤2.2**，可将该条改为 repost=true：
                1. 内容与 Profile / historical_summary 所体现的身份、长期关切或口味**明显对路**，你刷到会有「想让人看到 / 想站队 / 想关注对象顶一下」的冲动——满足才可作为转发的**主因**，否则repost=false
                {step2_rec}
                3. decision_reason 须**一句话**点明是哪一种；**禁止**用泛化套话（如「值得关注」「再说两句」）代替具体动机，否则repost=false
            - 对不感兴趣的微博：该微博不进入候选池，repost=false。
            - 若无任何一条满足转发条件，本批可全部为 repost=false。

            步骤2.2：转发链自检（显著提示）
            - 【⚠ 自检提醒】若该条微博的转发链（如 reposted_path、链路作者）中出现你自己的 user_id，视为“可能重复传播自己链路”。
            - 命中该情形时：默认不进入候选池，**repost=false**；极小概率二次转发，此时不允许 repost_content=""，必须写新增评论且和前面的转发内容有明显区别。

            步骤3：选取 0～{k_diff_targets} 目标微博
            - {k_diff_targets} 表示本轮**最多**对多少个**不同** blog_id 置 repost=true；实际条数可为 0～{k_diff_targets} 中任意值，**不得**为凑满条数而放宽步骤1.5/步骤2。
            - 仅在步骤2 的候选池内、按你的排序取前若干条作为“目标 blog_id 集合”，且条数不超过 {k_diff_targets}；若候选少于 {k_diff_targets}，有多少算多少；其余笔记保持 repost=false。

            步骤4：为每个目标 blog_id 生成 0～{k_same_target} 条转发
            - 以个人视角出发，对每个目标 blog_id：在 decisions 中写出 {k_same_target} 个 repost=true 的条目（可用于追评链/补一句，或可仅转发两次）。
            - 若 {k_same_target}=1：每个目标 blog_id 只生成 1 条转发。
            - 若 {k_same_target}>=2：允许同帖多条，但第2条及之后必须是增量（补充新点/纠错/情绪加码），禁止复述同一句。允许不写转发内容。

            步骤5：判断是否“仅转发不加评”
            - 以个人视角出发，对每条 repost=true 的候选，做二选一
            - A) 空内容转发：repost_content=""，只想转发不想评论
            - B) 带评论转发：repost_content 非空，仅当有意愿表达个性化评论或想参与事件讨论，同时想帮忙扩散事件时才选择
            若步骤5选择选择 A (空内容转发)，则 repost_content 必须为 ""，跳过步骤6和步骤7。

            步骤6：若repost_content不为空，根据人设，为每条转发选择 repost_mode，默认为围观
            - repost_mode ∈ {{围观，玩梗吐槽，情绪抒发，分析转述}}，并在 expression_reason 体现。
            - 发生概率：围观 > 玩梗吐槽 > 情绪抒发 > 分析转述
            - 共同特征：口语化，以第一人称视角表达，不要求语法正确
                - 围观：字数极短、结构弱，多重复其他短评中的围观表达（如“哈哈”、“笑死”、“666”等）。上限 5 字，禁止使用标点
                - 玩梗吐槽：通过双关、戏仿、梗化或夸张形成轻松/讽刺表达。上限 15 字，禁止使用标点
                - 情绪抒发：情绪强度较高的主观表态，核心是惊讶、兴奋、无奈等感受输出。上限 25 字，上限 1个标点，或者以空格代替标点
                - 分析转述：信息密度高，仅在“确有新增信息”时使用。上限 50 字
            -输出风格：
                - 使用用户所在地区常用语言；默认使用中文；
                - 表达应符合 Profile 与 historical_summary 的人设口吻；
                - 减少模板化套话，如“你说得对/确实/很有道理/希望大家…”；
                - 评论内容不包含标签（#...）。
                - repost_content中不允许出现“@...”。

            步骤7：决定是否需要 @ 用户
            - 以个人视角出发，仅当对方在可@列表且与内容强相关 + 不是推文作者/父转发作者时，才在正文中额外 @user_id；否则 mention_reasoning 为 []。

            步骤8：keep_following_blog_ids
            - 仅当：高度感兴趣 + 当前内容有持续关注价值→ keep_following_blog_ids=true；否则为空列表 []

            步骤9：search
            - 仅当：高度感兴趣 + 当前信息明显不足 + memory 无相关内容 → search=true；否则 false。

            请按以下 JSON 返回（字段顺序固定）：
            {{
            "persona_understanding": "1-2句。概括身份、兴趣、语言风格（简短）",
            "content_understanding": "1-2句。概括本次内容与你的相关性、你优先关注的角度",
            "source_understanding": "关注流=关注关系；推荐流=陌生关系，自己发布（你的内容）=自己发布",
            "memory_reflection": "{mem_refl_rec}",
            "decisions": [
                {{
                "blog_id": "blog_id",
                "repost": false,
                "repost_content": "",
                "decision_reason": "repost=false的原因（1句，<20字）",
                "expression_reason": "",
                "mention_reasoning": []
                }},
                {{
                "blog_id": "blog_id",
                "repost": true,
                "repost_content": "",
                "decision_reason": "转发的理由（1句，<20字）",
                "expression_reason": "repost_mode 为 none 时，expression_reason 为空字符串",
                "mention_reasoning": []
                }},
                {{
                "blog_id": "blog_id",
                "repost": true,
                "repost_content": "个性化的转发内容",
                "decision_reason": "转发的理由（1句，<20字）",
                "expression_reason": "为何使用这种语气与措辞",
                "mention_reasoning": [
                    {{
                    "user_id": "被@用户id",
                    "persona_understanding": "对该用户的简短理解",
                    "mention_reason": "为何需要额外提醒该用户（1句，<20字）"
                    }}
                ]
                }}
            ],
            "keep_following_blog_ids": [],
            "keep_following_reason": "为何保持关注（1句，<20字）",
            "search": false,
            "search_keyword": "搜索关键词（1句，≤20字）",
            "search_reason": "是否搜索及原因（1句）",
            }}

            输出规则：
            - 先给出理解字段，再给 decisions；默认使用中文，可中英文混合
            - decisions 中每条先按默认状态填写 repost=false，再仅对满足条件的条目改为 repost=true；
            - repost=false 时，repost_content 必须为空字符串。repost=true 时，repost_content 可以为空字符串，也可以为转发内容。
            - 根据 repost_mode 重复检验选择的评论内容，是否符合模式要求。
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
                keep_ids = response.get("keep_following_blog_ids", [])
                if isinstance(keep_ids, list) and keep_ids:
                    # 只允许本批次内的 blog_id，且最多 1 个
                    valid_keep_ids = []
                    for keep_blog_id in keep_ids:
                        if keep_blog_id in chunk:
                            valid_keep_ids.append(keep_blog_id)
                    if valid_keep_ids:
                        self.profile.update_data("keep_following_blog_ids", valid_keep_ids[:1])
                    else:
                        self.profile.update_data("keep_following_blog_ids", [])

            # 处理评论决策
            decisions = response.get("decisions", [])
            if not isinstance(decisions, list):
                continue
       
            # 处理每个决策：更新转发数和转发内容
            for decision in decisions:
                if not isinstance(decision, dict):
                    continue
                    
                blog_id = decision.get("blog_id")
                should_repost = decision.get("repost", False)
                repost_content = decision.get("repost_content", "")
                    
                if not blog_id or not should_repost:
                    continue

                # 检查 blog_id 是否合法
                if blog_id not in chunk:
                    logger.warning(f"Step {current_step}/{max_step}: Blog {blog_id} not found in recommendations")
                    continue
                blog = chunk[blog_id]
            
                # 解析转发内容中的@用户，将 @id 替换为 @昵称，并返回用户ID列表
                mentioned_user_ids = []
                mention_reasoning = decision.get("mention_reasoning", [])
                if isinstance(mention_reasoning, list):
                    for mention_reason in mention_reasoning:
                        if isinstance(mention_reason, dict):
                            reason_user_id = mention_reason.get("user_id")
                            if reason_user_id:
                                mentioned_user_ids.append(reason_user_id)

                # 构建转发路径和转发内容，以及作者列表
                if blog.get("reposted_blog_id"):
                    reposted_path = list(blog.get("reposted_path", []))
                    reposted_path.append(blog_id)
                    # 保序去重，避免传播链中同一节点重复累计
                    reposted_path = list(dict.fromkeys(reposted_path))
                    reposted_blog_id = blog.get("reposted_blog_id")
                    reposted_user_id = blog.get("user_id")
                    blog_content = blog.get("content", "")
                    repost_content = f"{repost_content}//@{reposted_user_id}: {blog_content}"
                    # blog_author_ids: 先按 // 分段，再提取每段里 @ 和 : 之间的 user_id
                    blog_author_ids = []
                    for part in str(repost_content).split("//"):
                        match = re.search(r"@([^:：\\s]+)\\s*[:：]", part)
                        if match:
                            blog_author_ids.append(match.group(1))
                else:
                    reposted_path = [blog_id]
                    reposted_blog_id = blog_id
                    reposted_user_id = blog.get("user_id")
                    blog_author_ids = [reposted_user_id]

                # 若 @ 的用户已在转发链作者列表里，避免重复提醒
                overlap_user_ids = set(uid for uid in blog_author_ids if uid)
                mentioned_user_ids = [
                    uid for uid in mentioned_user_ids if uid not in overlap_user_ids
                ]

                mention_count = len(mentioned_user_ids)

                if repost_content == "":
                    repost_content = "转发微博"
              
                # 如果为转发微博，添加转发
                # 生成唯一的转发ID
                repost_id = self._generate_repost_id()
                    
                success = await self.add_env_reposts(repost_id, {
                    "blog_id": repost_id,
                    "content": repost_content,
                    "time": self._random_repost_timestamp(blog, current_ts, step_duration),
                    "ip_location": ip_location,
                    "user_id": user_id,
                    "nickname": user_nickname,
                    "at_count": mention_count,
                    "reposted_blog_id": reposted_blog_id,
                    "reposted_path": reposted_path,
                    "repost_count": 0,
                    "repost_ids": []
                })
                if not success:
                    logger.error(f"Failed to add repost to content pool, blog {blog_id}")
                    continue

                # 向笔记作者发送提醒
                for blog_author_id in blog_author_ids:
                    if blog_author_id and blog_author_id != user_id:  # 不给自己发提醒
                        success = await self.update_env_mention_pool(f"{blog_author_id}.{repost_id}", {
                            "action": "add",
                            "mention_message": {
                                "blog_id": repost_id,
                                "mention_type": "repost"
                            }
                        })
                        if not success:
                            logger.error(f"Failed to update mention pool for repost {repost_id} by {user_id} on blog {blog_id}")
                            continue
                        logger.info(f"Step {current_step}/{max_step}: User {user_id} reposted blog {blog_id} by {blog_author_id}")
                   
                # 发送MentionEvent给被@的用户
                if mentioned_user_ids:
                    # 为每个被@的用户创建MentionEvent
                    for mentioned_user_id in mentioned_user_ids:
                        if mentioned_user_id and mentioned_user_id != user_id:  # 不给自己发提醒
                            # 创建@事件，发送给被@的用户
                            success = await self.update_env_mention_pool(f"{mentioned_user_id}.{repost_id}", {
                                "action": "add",
                                "mention_message": {
                                    "blog_id": repost_id,
                                    "mention_type": "at"
                                }
                            })
                            if not success:
                                logger.error(f"Failed to update mention pool for repost {repost_id} by {user_id} on blog {blog_id}")
                                continue
                            logger.info(f"Step {current_step}/{max_step}: User {user_id} mentioned {mentioned_user_id} in repost {repost_id} on blog {blog_id}")
                    
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

    def _has_self_in_repost_chain(
        self,
        blog: Dict[str, Any],
        my_id: str,
        content_pool: Dict[str, Any],
    ) -> bool:
        """
        检查 repost 链上是否出现自己：
        - 当前微博作者 user_id
        - reposted_path（若历史数据混入 user_id 也识别）
        - 沿 reposted_blog_id 向上遍历父链作者
        """
        if not isinstance(blog, dict) or not my_id:
            return False
        my_id_str = str(my_id)

        author_id = blog.get("user_id")
        if author_id is not None and str(author_id) == my_id_str:
            return True

        path = blog.get("reposted_path", [])
        if isinstance(path, list) and any(str(x) == my_id_str for x in path):
            return True

        parent_id = blog.get("reposted_blog_id")
        visited: Set[str] = set()
        while parent_id:
            p = str(parent_id).strip()
            if not p or p in visited:
                break
            visited.add(p)
            parent_blog = content_pool.get(p)
            if not isinstance(parent_blog, dict):
                break
            p_uid = parent_blog.get("user_id")
            if p_uid is not None and str(p_uid) == my_id_str:
                return True
            parent_id = parent_blog.get("reposted_blog_id")

        return False

    def _merge_mentions(
        self, 
        old_messages: Dict[str, Any], 
        new_messages: Dict[str, Any], 
        current_ts: int = 0,
        last_check: int = 0,
        period_sec: int = 86400,
        mention_cap: int = 20,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        合并提醒消息（current_ts / last_check / period 均为 Unix 秒）
        """
        merged = {**old_messages, **new_messages}

        if current_ts >= last_check + period_sec:
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
        - 评论提醒（mention_type="repost"）：回应概率中等
        
        Args:
            event: MentionEvent，包含@/转发信息
            
        Returns:
            List[Event]: 返回要发送的事件列表（通常是转发事件）
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

        content_pool = await self.get_env_data("content_pool")
        if not isinstance(content_pool, dict):
            content_pool = {}

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
            mention_blog = mention_message.get("blog")
            if not isinstance(mention_blog, dict):
                continue
            mentioner_id = mention_blog.get("user_id")
            mentioner_nickname = mention_blog.get("nickname", "")
            mention_type = mention_message.get("mention_type", "at")

            relationship_type = self._check_relationship(my_id, mentioner_id, follow_ids, fan_ids)
            if relationship_type == "mutual":
                relationship_hint = "（你的好友）"
            elif relationship_type == "follow":
                relationship_hint = "（你的关注）"
            else:
                relationship_hint = ""

            if mention_type == "at":
                mention_action = f"{mentioner_nickname}{relationship_hint}在转发中@了你"
                content_label = "@你的转发内容"
            elif mention_type == "repost":
                mention_action = f"{mentioner_nickname}{relationship_hint}转发了你的笔记"
                content_label = "转发内容"
            if relationship_hint:
                relationship_source = f"{mentioner_nickname} 与你的关系：{relationship_hint}"
            else:
                relationship_source = f"{mentioner_nickname} 与你的关系：陌生人"

            self_chain_hit = self._has_self_in_repost_chain(mention_blog, my_id, content_pool)
            self_chain_hint = ""
            if self_chain_hit:
                self_chain_hint = "【⚠ 检测到该转发链包含你自己：默认repost=false】"

            blog_id = mention_blog.get("blog_id") or (mention_key.split("_")[0] if "_" in str(mention_key) else str(mention_key))
            mention_entries.append({
                "mention_key": mention_key,
                "mention_blog": mention_blog,
                "blog_id": blog_id,
                "self_chain_hit": self_chain_hit,
                "self_chain_hint": self_chain_hint,
                "mentioner_id": mentioner_id,
                "mentioner_nickname": mentioner_nickname,
                "mention_type": mention_type,
                "mention_action": mention_action,
                "content_label": content_label,
                "relationship_source": relationship_source,
            })

        if not mention_entries:
            return []

        self._record_mentioned_blog_ids_by_channel(
            current_step,
            mention_entries,
            getattr(event, "timestamp", 0),
        )

        # 构建观察信息
        observation_parts = []
        for i, entry in enumerate(mention_entries):
            observation_parts.append(f"""## 提醒 {i + 1}
            {entry["mention_action"]}

            与转发者的关系（relationship_understanding 必须据此填写，请勿臆测或颠倒）：
            {entry["relationship_source"]}

            {entry["content_label"]}：：
            {json.dumps(entry["mention_blog"], ensure_ascii=False, indent=2)}
            {entry.get("self_chain_hint", "")}
            """)

        observation = (
            "【场景】你收到了通知（@/转发）。仍可直接忽略；只有值得接话时才回复。接话时也要短，像聊天，不像写通报。\n\n"
            + "转发和@（共 {} 条提醒，请按顺序对每条分别给出决策）：\n\n".format(len(mention_entries))
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
            "- 当前批次与已存记忆在关键词层面判定为显著相关 → **强烈倾向 repost=false**（易与 memory 中已有表态或同题讨论重复；仅当步骤1.5 明确满足可核验新信息与强动机等破例条件时再考虑 repost=true）；"
        ) if kw_ok else ""
        step15_emb_coaching = (
            "\n\n【话题与 memory — 语义相似】\n            "
            "- 向量相似度达到设定阈值，本批话题与记忆中内容相近 → **强烈倾向 repost=false**（视同同脉络/易重复话题，须严格按步骤1.5 评估是否仍 repost=true）；"
        ) if emb_ok else ""

        raw_act = self.profile.get_data("activity_level", 0.0) if self.profile else 0.0
        try:
            _activity = float(raw_act)
        except (TypeError, ValueError):
            _activity = 0.0
        _activity = max(0.0, min(1.0, _activity))

        if (_activity < 0.7 and mem_ok):
            step15_receive = """
            步骤1.5：对照 memory 做「重复话题」检查（在步骤2之前完成）
            - 默认 repost=false。若当前内容与 memory 指向**同一事件/同一问题/同一争议脉络**，或与 memory 中重叠的词超过1个，无论你是否**已在同类内容上转发、表态过**，则**几乎必须保持repost=false**。
            - **若要破例，须同时满足以下三项，缺一不可：**
            · **（1）可核对的新信息点**：`decision_reason` 须在**单句**内写清相对 memory、帖中**独有**且可指认的一条新增事实（须出现具体人/机构/日期/数字/规则名或链接类标识之一）；不得单独用「新细节」「新进展」「新讨论点」「又一例」「再关注」「同类再发酵」「略多一句」等空话充数。
            · **（2）强动机**：**强烈情绪动机**（同句或紧邻句须点明具体情绪落点，禁止空泛「有感触」「想说两句」）。
            · **（3）明确扩散动机**：**明确扩散动机**（须点明为维护/帮扩/站队**具体的**互关、关注或好友，写清对象，禁止笼统「支持一下」）。
            - **同时**具备可核对新信息 **与**强烈情绪 **与** 明确扩散动机，仅有情绪/扩散而无新信息、或仅有新信息而无强情绪/扩散动机，均 repost=false，避免同题刷屏。
            - **memory 越多、越要克制**：即便（1）（2）（3）在字面上都能凑上，仍应把「本条 repost=true」当成**小概率事件**——默认继续 repost=false；仅当新信息**明显升级**（例如改变事件阶段、推翻或修正你 memory 中的既有判断、或出现关键新主体/新规则）时才可破例，禁止「勉强达标就评一句」。
            - **memory_reflection 禁止自相矛盾**：先写「与 memory 重叠/同一话题/已讨论过」等，又用无（1）+（2）支撑的转折暗示可以评论——一律视为无效；若判定重叠或几乎 repost=false，memory_reflection 须**通篇**结论为倾向沉默或明确无新信息，不得以模糊语气自我放行。
            将上述结论简要写入 memory_reflection；不转发时 decision_reason 须点明「与 memory 重叠/已表态/无新信息/缺新信息或缺强动机」等。
            """
            step2_rec = "0. 破例须**同时**具备（1）可核对新信息点 **与**（2）强烈情绪动机 **与** （3）明确扩散动机，缺一仍 repost=false；若不触发步骤1.5 的重叠情形，本条可视为已满足；"
            mem_refl_rec = "2-3句。无相关记忆可写「无相关记忆/首次接触」；有同题时说明是否重叠；若重叠倾向不转发，可说明是否仍有一句评论欲（仅评论不转发）"
        else:
            # step15_receive = """
            # 步骤1.5：对照 memory 做「重复话题」检查（在步骤2之前完成）
            # - 若当前内容与 memory 指向同一事件/同一争议点/同一问题，且你在 memory 中的立场、态度或结论与本次若回复会说的内容高度相近 → 将该条倾向repost=false，除非你能明确写出：相对 memory 中已有表达，本次将补充**读者可感知的新事实、新角度或新推理**（例如改变事件阶段、推翻或修正你 memory 中的既有判断、或出现关键新主体/新规则，仅换说法不算）。
            # - 推荐流陌生关系、或你与作者关系较弱时，在「是否破例」上应更保守。
            # 将上述结论简要写入 memory_reflection；不评论时 decision_reason 须点明「与 memory 重叠/已表态/无新信息」等。
            # """
            step15_receive = ""
            step2_rec = "2. 关系与场景合适（关注关系优先）；"
            mem_refl_rec = "2-3句。无相关记忆可写「无相关记忆/首次接触」；有同题时说明是否重叠；若重叠倾向不转发，可说明是否仍有一句评论欲（仅评论不转发）"

        if time_module_str:
            time_coaching_block = (
                "【仿真时间与时效】\n            "
                + time_module_str
                + " - 若上文含 **【警告】** 或写明时效已明显减弱、倾向不回复 → **强烈倾向 repost=false**（该条不进候选池）；"
            )
        else:
            time_coaching_block = ""

        instruction = f"""下面有多条提醒，请按顺序对每条提醒分别给出一个决策（是否回复、回复内容等）。decisions 数组与提醒顺序一致，第 i 个元素对应第 i 条提醒。

        请基于 Profile、historical_summary、memory 与 Observation 中的关系信息，完成“是否转发”判断与转发生成。

        步骤1：对每条推荐先设默认状态
        - "repost": false
        - "repost_content": ""

        ** repost 的数量决策仅由步骤2/3决定，不得因步骤5改变**
        {step15_receive}

        步骤2：兴趣判断
        - 默认 repost=false。通读本批次所有推荐内容：若无任何一条能同时通过步骤1.5 破例条件，则**本批全体 repost=false**，不得因「略有相关」放行。
        - 仅当**下列全部满足**且**不违反步骤2.2**，可将该条改为 repost=true：
            1. 内容与 Profile / historical_summary 所体现的身份、长期关切或口味**明显对路**，你刷到会有「想让人看到 / 想站队 / 想关注对象顶一下」的冲动——满足才可作为转发的**主因**，否则repost=false
            {step2_rec}
            3. decision_reason 须**一句话**点明是哪一种；**禁止**用泛化套话（如「值得关注」「再说两句」）代替具体动机，否则repost=false
        - 对不感兴趣的微博：该微博不进入候选池，repost=false。
        - 若无任何一条满足转发条件，本批可全部为 repost=false；。

        步骤2.2：转发链自检（显著提示）
        - 【⚠ 自检提醒】若该条微博的转发链（如 reposted_path、链路作者）中出现你自己的 user_id，视为“可能重复传播自己链路”。
        - 命中该情形时：默认不进入候选池，**repost=false**；极小概率二次转发，此时不允许 repost_content=""，必须写新增评论且和前面的转发内容有明显区别。

        步骤3：选取 0～{k_diff_targets} 目标微博
        - {k_diff_targets} 表示本轮**最多**对多少个**不同** blog_id 置 repost=true；实际条数可为 0～{k_diff_targets} 中任意值，**不得**为凑满条数而放宽步骤1.5/步骤2。
        - 仅在步骤2 的候选池内、按你的排序取前若干条作为“目标 blog_id 集合”，且条数不超过 {k_diff_targets}；若候选少于 {k_diff_targets}，有多少算多少；其余笔记保持 repost=false。

        步骤4：判断是否“仅转发不加评”
        - 以个人视角出发，对每条 repost=true 的候选，做二选一
        - A) 空内容转发：repost_content=""，只想转发不想评论
        - B) 带评论转发：repost_content 非空，仅当有意愿表达个性化评论或想参与事件讨论，同时想帮忙扩散事件时才选择
        若步骤4选择选择 A (空内容转发)，则 repost_content 必须为 ""，跳过步骤6和步骤7。

        步骤5：若repost_content不为空，根据人设，为每条转发选择 repost_mode，默认选择围观
        - 发生概率：围观 > 玩梗吐槽 > 情绪抒发 > 分析转述
        - 共同特征：口语化，以第一人称视角表达，不要求语法正确
            - 围观：字数极短、结构弱，多重复其他短评中的围观表达（如“哈哈”、“笑死”、“666”等）。上限 5 字，禁止使用标点
            - 玩梗吐槽：通过双关、戏仿、梗化或夸张形成轻松/讽刺表达。上限 15 字，禁标点
            - 情绪抒发：情绪强度较高的主观表态，核心是惊讶、兴奋、无奈等感受输出。上限 25 字，上限 1个标点，或者以空格代替标点
            - 分析转述：信息密度高，仅在“确有新增信息”时使用。上限 50 字
        -输出风格：
            - 使用用户所在地区常用语言；默认使用中文；
            - 表达应符合 Profile 与 historical_summary 的人设口吻；
            - 减少模板化套话，如“你说得对/确实/很有道理/希望大家…”；
            - 评论内容不包含标签（#...）。
            - repost_content中不允许出现“@...”。

        步骤6：决定是否需要 @ 用户
        - 仅当对方在可@列表且与内容强相关 + 不是推文作者/父转发作者时，才在正文中额外 @user_id；否则 mention_reasoning 为 []。

        步骤7：keep_following_blog_ids
        - 仅当：高度感兴趣 + 当前内容有持续关注价值→ keep_following_blog_ids=true；否则为空列表 []

        步骤8：search
        - 仅当：高度感兴趣 + 当前信息明显不足 + memory 无相关内容 → search=true；否则 false。

        请按以下 JSON 返回（字段顺序固定）：
        {{
        "persona_understanding": "1-2句。概括身份、兴趣、语言风格（简短）",
        "content_understanding": "1-2句。概括本次被@/被评论内容与你的相关性、你优先关注的角度",
        "relationship_understanding": "严格依据 Observation 的关系判定，不自行扩展关系类型",
        "memory_reflection": "{mem_refl_rec}",
        "decisions": [
            {{
            "blog_id": "blog_id",
            "repost": false,
            "repost_content": "",
            "decision_reason": "repost=false的原因（1句，≤20字）",
            "expression_reason": "",
            "mention_reasoning": []
            }},
            {{
            "blog_id": "blog_id",
            "repost": true,
            "repost_content": "",
            "decision_reason": "转发的理由（1句，≤20字）",
            "expression_reason": "repost_mode 为 none 时，expression_reason 为空字符串",
            "mention_reasoning": []
            }},
            {{
            "blog_id": "blog_id",
            "repost": true,
            "repost_content": "个性化的转发内容",
            "decision_reason": "转发的理由（1句，≤20字）",
            "expression_reason": "为何使用这种语气与措辞",
            "mention_reasoning": [
                {{
                "user_id": "被@用户id",
                "persona_understanding": "对该用户的简短理解",
                "mention_reason": "为何需要额外提醒该用户（1句）"
                }}
            ]
            }}
        ],
        "keep_following_blog_ids": [],
        "keep_following_reason": "为何保持关注（1句，<20字）",
        "search": false,
        "search_keyword": "搜索关键词（1句，≤20字）",
        "search_reason": "是否搜索及原因（1句）"
        }}

        输出规则：
        - 先给出理解字段，再给 decisions；默认使用中文，可中英文混合
        - decisions 中每条先按默认状态填写 repost=false，再仅对满足条件的条目改为 repost=true；
        - repost=false 时，repost_content 必须为空字符串。repost=true 时，repost_content 可以为空字符串，也可以为转发内容。
        - 根据 repost_mode 重复检验选择的评论内容，是否符合模式要求。
        {step15_kw_coaching}{step15_emb_coaching}
        {time_coaching_block}"""

        # 生成决策
        response = await self.generate_reaction(instruction, observation)

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
            keep_ids = response.get("keep_following_blog_ids", [])
            if isinstance(keep_ids, list) and keep_ids:
                # 只允许本批次内的 blog_id，且最多 1 个
                valid_keep_ids = []
                for keep_blog_id in keep_ids:
                    if keep_blog_id in chunk:
                        valid_keep_ids.append(keep_blog_id)
                if valid_keep_ids:
                    self.profile.update_data("keep_following_blog_ids", valid_keep_ids[:1])
                else:
                    self.profile.update_data("keep_following_blog_ids", [])

        # 处理转发决策
        decisions = response.get("decisions", [])
        if not isinstance(decisions, list):
            return events_to_send

        has_reply = False
        user_id = await self.get_data("id")
        nickname = await self.get_data("nickname", "")
        ip_location = await self.get_data("ip_location", "")
        current_timestamp = event.timestamp
        current_ts = int(event.timestamp) if getattr(event, "timestamp", None) is not None else 0
        step_duration = int(getattr(event, "timestamp_duration", 0) or 0)

        for i, decision in enumerate(decisions):
            if i >= len(mention_entries):
                break
            mention_entry = mention_entries[i]
            mention_blog = mention_entry["mention_blog"]
            mention_blog_id = mention_entry["blog_id"]
            try:
                if not isinstance(decision, dict) or not decision.get("repost", False):
                    continue

                blog_id = decision.get("blog_id")
                should_repost = decision.get("repost", False)
                repost_content = decision.get("repost_content", "")

                if not blog_id or not should_repost:
                    continue

                has_repost = True

                # 检查 blog_id 是否合法（应该与 mention_blog_id 一致）
                if blog_id != mention_blog_id:
                    logger.warning(f"Blog {blog_id} does not match mention blog {mention_blog_id}, skipping")
                    continue

                # 解析评论内容中的@用户，将 @id 替换为 @昵称，并返回用户ID列表
                mentioned_user_ids = []
                mention_reasoning = decision.get("mention_reasoning", [])
                if isinstance(mention_reasoning, list):
                    for mention_reason in mention_reasoning:
                        if isinstance(mention_reason, dict):
                            reason_user_id = mention_reason.get("user_id")
                            if reason_user_id:
                                mentioned_user_ids.append(reason_user_id)

                # 构建转发路径和转发内容，以及作者列表
                if mention_blog.get("reposted_blog_id"):
                    reposted_path = list(mention_blog.get("reposted_path", []))
                    reposted_path.append(blog_id)
                    # 保序去重，避免传播链中同一节点重复累计
                    reposted_path = list(dict.fromkeys(reposted_path))
                    reposted_blog_id = mention_blog.get("reposted_blog_id")
                    reposted_user_id = mention_blog.get("user_id")
                    mention_blog_content = mention_blog.get("content", "")
                    repost_content = f"{repost_content}//@{reposted_user_id}: {mention_blog_content}"
                    # blog_author_ids: 先按 // 分段，再提取每段里 @ 和 : 之间的 user_id
                    blog_author_ids = []
                    for part in str(repost_content).split("//"):
                        match = re.search(r"@([^:：\\s]+)\\s*[:：]", part)
                        if match:
                            blog_author_ids.append(match.group(1))
                else:
                    reposted_path = [blog_id]
                    reposted_blog_id = blog_id
                    reposted_user_id = mention_blog.get("user_id")
                    blog_author_ids = [reposted_user_id]

                # 若 @ 的用户已在转发链作者列表里，避免重复提醒
                overlap_user_ids = set(uid for uid in blog_author_ids if uid)
                mentioned_user_ids = [
                    uid for uid in mentioned_user_ids if uid not in overlap_user_ids
                ]

                mention_count = len(mentioned_user_ids)

                if repost_content == "":
                    repost_content = "转发微博"

                # 添加转发
                repost_id = self._generate_repost_id()
                success = await self.add_env_reposts(repost_id, {
                    "blog_id": repost_id,
                    "content": repost_content,
                    "time": self._random_repost_timestamp(mention_blog, current_ts, step_duration),
                    "ip_location": ip_location,
                    "user_id": user_id,
                    "nickname": nickname,
                    "at_count": mention_count,
                    "reposted_blog_id": reposted_blog_id,
                    "reposted_path": reposted_path,
                    "repost_count": 0,
                    "repost_ids": []
                })
                if not success:
                    logger.error(f"Failed to add repost to content pool (parent blog {blog_id}, new id {repost_id})")
                    continue

                for blog_author_id in blog_author_ids:
                    if blog_author_id and blog_author_id != user_id:    # 不给自己发提醒
                        # 生成提醒信息
                        success = await self.update_env_mention_pool(f"{blog_author_id}.{repost_id}", {
                            "action": "add",
                            "mention_message": {
                                "blog_id": repost_id,
                                "mention_type": "repost"
                            }
                        })
                        if not success:
                            logger.error(f"Failed to update mention pool for repost {repost_id} by {user_id} on blog {blog_id}")
                            continue
                        logger.info(f"Step {current_step}/{max_step}: User {user_id} reposted on blog {blog_id} by {blog_author_id}")

                # 发送MentionEvent给被@的用户
                if mentioned_user_ids:
                    # 为每个被@的用户创建MentionEvent
                    for mentioned_user_id in mentioned_user_ids:
                        if mentioned_user_id and mentioned_user_id != user_id:  # 不给自己发提醒
                            # 创建@事件，发送给被@的用户
                            success = await self.update_env_mention_pool(f"{mentioned_user_id}.{repost_id}", {
                                "action": "add",
                                "mention_message": {
                                    "blog_id": repost_id,
                                    "mention_type": "at"
                                }
                            })
                            if not success:
                                logger.error(f"Failed to update mention pool for repost {repost_id} by {user_id} on blog {blog_id}")
                                continue
                            logger.info(f"User {user_id} mentioned {mentioned_user_id} in repost on blog {blog_id}")
            finally:
                # 无论本条是否回复、是否中途 continue，处理完都从 mention_pool 删除
                success = await self.update_env_mention_pool(f"{user_id}.{mention_blog_id}", {
                    "action": "delete",
                    "mention_message": None
                })
                if not success:
                    logger.error(f"Failed to update mention pool for blog {mention_blog_id}")
                else:
                    logger.info(f"Step {current_step}/{max_step}: User {user_id} deleted repost {mention_blog_id} from pool (processed)")

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
    