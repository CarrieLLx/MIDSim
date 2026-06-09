# -*- coding: utf-8 -*-
"""UserAgent reaction gates: four gate classes + UserAgentGates aggregator."""
from __future__ import annotations

import asyncio
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

from loguru import logger
from onesim.utils.midsim_params import (
    memory_similarity_gate_params,
    user_low_activity_memory_gate_threshold,
    user_low_activity_time_module_threshold,
    user_stale_days,
)

from .embedding_client import cosine_similarity, default_embedding_config_path, get_embeddings, load_embedding_config
from .utils import time_to_format_utc, time_to_ms

ReactionContext = Literal["recommendation", "mention"]


@dataclass
class ReactionGateCoaching:
    """Combined prompt coaching injected into LLM instruction templates."""

    freshness_coaching: str = ""
    similarity_kw_coaching: str = ""
    similarity_emb_coaching: str = ""
    memory_coaching: str = ""
    memory_rec: str = ""
    memory_ref: str = ""
    k_same_target: int = 1
    k_diff_targets: int = 1


@dataclass(frozen=True)
class MemorySimilarityGateConfig:
    policies: Tuple[str, ...]
    policy_raw: str
    multi_combine: str
    keyword_enabled: bool
    embedding_enabled: bool
    min_common: int
    embed_threshold: float
    include_historical_summary: bool
    embedding_config_path: str
    embed_max_chars: int
    embed_max_chunks: int
    embed_chunk_agg: str


class MemorySimilarityGate:
    """Compare current topic text against the agent's stored memory."""

    KW_COACHING = (
        "\n\n【话题与 memory — 关键词重叠】\n            "
        "- 当前批次与已存记忆在关键词层面判定为显著相关 → **强烈倾向 repost=false**"
        "（易与 memory 中已有表态或同题讨论重复；仅当步骤1.5 明确满足可核验新信息与强动机等破例条件时再考虑 repost=true）；"
    )
    EMB_COACHING = (
        "\n\n【话题与 memory — 语义相似】\n            "
        "- 向量相似度达到设定阈值，本批话题与记忆中内容相近 → **强烈倾向 repost=false**"
        "（视同同脉络/易重复话题，须严格按步骤1.5 评估是否仍 repost=true）；"
    )
    STRICT_STEP15_COACHING = """
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
    _STOPWORDS: Set[str] = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
        "and", "or", "but", "not", "this", "that", "these", "those", "it", "its",
        "的", "了", "在", "是", "和", "与", "或", "及", "有", "为", "对", "中", "上", "下",
        "不", "吗", "嘛", "啊", "哦", "嗯", "吧", "呀", "么", "呢", "着", "过", "还", "就",
        "一个", "可以", "这样", "这个", "我们", "你们", "他们", "什么", "怎么", "没有",
        "自己", "如果", "因为", "所以", "但是", "然后", "而且", "或者",
    }
    _VALID_POLICIES = frozenset(("memory_nonempty", "keyword", "embedding"))

    def __init__(self, agent: Any) -> None:
        self._agent = agent

    @classmethod
    def tokenize(cls, text: str) -> Set[str]:
        if not text or not str(text).strip():
            return set()
        raw = re.findall(r"[\w\u4e00-\u9fff]+", str(text).lower())
        out: Set[str] = set()
        for w in raw:
            w = w.strip().lower()
            if len(w) < 2 or w in cls._STOPWORDS:
                continue
            out.add(w)
        return out

    @classmethod
    def keyword_overlap(
        cls,
        topic_text: str,
        memory_side_text: str,
        *,
        min_common: int = 2,
    ) -> bool:
        """Calculate the keyword overlap degree"""
        a = cls.tokenize(topic_text)
        b = cls.tokenize(memory_side_text)
        if not a or not b:
            return False
        return len(a & b) >= min_common

    @classmethod
    def _chunk_text_for_embedding(cls, text: str, max_chars: int) -> List[str]:
        """Chunk the text into a list of strings"""
        t = (text or "").strip()
        if not t:
            return []
        mc = max(32, int(max_chars))
        if len(t) <= mc:
            return [t]
        out: List[str] = []
        cur = ""
        for line in t.split("\n"):
            line = line.strip()
            if not line:
                continue
            cand = f"{cur}\n{line}".strip() if cur else line
            if len(cand) <= mc:
                cur = cand
            else:
                if cur:
                    out.append(cur)
                if len(line) <= mc:
                    cur = line
                else:
                    for i in range(0, len(line), mc):
                        out.append(line[i : i + mc])
                    cur = ""
        if cur:
            out.append(cur)
        return [c for c in out if c]

    @staticmethod
    def _mean_embedding(vectors: List[List[float]]) -> List[float]:
        """Calculate the mean of the embeddings"""
        if not vectors:
            return []
        dim = len(vectors[0])
        acc = [0.0] * dim
        for v in vectors:
            for i, x in enumerate(v):
                acc[i] += float(x)
        n = float(len(vectors))
        return [x / n for x in acc]

    @staticmethod
    def _embed_chunks_sequential(
        get_embeddings: Any,
        base_url: str,
        model_name: str,
        chunks: List[str],
    ) -> List[List[float]]:
        """Embed the chunks sequentially"""
        vecs: List[List[float]] = []
        for ch in chunks:
            batch = get_embeddings(base_url, model_name, [ch])
            if len(batch) != 1:
                raise ValueError("embedding API returns an abnormal number of chunks")
            vecs.append(batch[0])
        return vecs

    @classmethod
    def embedding_similarity(
        cls,
        topic_text: str,
        memory_side_text: str,
        *,
        cfg: MemorySimilarityGateConfig,
    ) -> Optional[float]:
        """Calculate the embedding similarity"""
        max_chars = max(64, min(int(cfg.embed_max_chars), 8000))
        max_chunks = max(1, min(int(cfg.embed_max_chunks), 64))
        agg = cfg.embed_chunk_agg if cfg.embed_chunk_agg in ("mean", "max") else "mean"

        chunks_t = cls._chunk_text_for_embedding(topic_text or "", max_chars)[:max_chunks]
        chunks_m = cls._chunk_text_for_embedding(memory_side_text or "", max_chars)[:max_chunks]
        if not chunks_t or not chunks_m:
            return None
        try:
            base_url, model_name = load_embedding_config(cfg.embedding_config_path)
            vt = cls._embed_chunks_sequential(get_embeddings, base_url, model_name, chunks_t)
            vm = cls._embed_chunks_sequential(get_embeddings, base_url, model_name, chunks_m)
            if not vt or not vm:
                return None
            if agg == "max":
                best = -1.0
                for a in vt:
                    for b in vm:
                        best = max(best, float(cosine_similarity(a, b)))
                return best
            mt = cls._mean_embedding(vt)
            mm = cls._mean_embedding(vm)
            return float(cosine_similarity(mt, mm))
        except Exception as e:
            detail = ""
            resp = getattr(e, "response", None)
            if resp is not None:
                try:
                    detail = f" | response={resp.text[:400]!r}"
                except Exception:
                    detail = f" | status={getattr(resp, 'status_code', '')}"
            logger.warning(
                f"MemorySimilarityGate embedding failed (treated as no similarity): {e}{detail}"
            )
            return None

    @classmethod
    def _parse_policy_list(cls, raw: str) -> Tuple[str, ...]:
        """Parse the policy list"""
        s = (raw or "").strip().lower()
        if not s:
            return ("memory_nonempty",)
        parts = re.split(r"[,|+]+", s)
        seen: Set[str] = set()
        out: List[str] = []
        for p in parts:
            p = p.strip().lower()
            if not p or p in seen:
                continue
            if p not in cls._VALID_POLICIES:
                logger.warning(
                    f"MemorySimilarityGate: skip unknown policy {p!r}, "
                    f"valid={sorted(cls._VALID_POLICIES)}"
                )
                continue
            seen.add(p)
            out.append(p)
        return tuple(out) if out else ("memory_nonempty",)

    @staticmethod
    def _parse_config_bool(value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        s = str(value).strip().lower()
        if s in ("0", "false", "no", ""):
            return False
        if s in ("1", "true", "yes"):
            return True
        return default

    @classmethod
    def load_config(cls, params: Optional[Dict[str, Any]] = None) -> MemorySimilarityGateConfig:
        """Load the configuration for the memory similarity gate"""
        # Get the policy 
        p = params or {}
        policy_raw = str(p.get("policy", "memory_nonempty,keyword,embedding")).strip()
        policies = cls._parse_policy_list(policy_raw)
        combine = str(p.get("multi_combine", "or")).strip().lower()
        if combine not in ("or", "and"):
            combine = "or"
        kw = cls._parse_config_bool(p.get("keyword_enabled"), True)
        emb = cls._parse_config_bool(p.get("embedding_enabled"), True)

        # Get the minimum number of common tokens
        try:
            min_common = int(p.get("min_common_tokens", 2) or 2)
        except (TypeError, ValueError):
            min_common = 2

        # Get the embedding threshold
        try:
            embed_th = float(p.get("embed_threshold", 0.65) or 0.65)
        except (TypeError, ValueError):
            embed_th = 0.65

        # Get the historical summary
        inc_hist = cls._parse_config_bool(p.get("include_historical_summary"), True)

        # Get the embedding configuration path
        cfg_path = str(p.get("embedding_config_path", "") or "").strip()
        if not cfg_path:
            cfg_path = default_embedding_config_path()

        # Get the embedding max chars
        try:
            embed_max_chars = int(p.get("embed_max_chars", 400) or 400)
        except (TypeError, ValueError):
            embed_max_chars = 400

        # Get the embedding max chunks
        try:
            embed_max_chunks = int(p.get("embed_max_chunks", 12) or 12)
        except (TypeError, ValueError):
            embed_max_chunks = 12

        agg = str(p.get("embed_chunk_agg", "mean")).strip().lower()
        if agg not in ("mean", "max"):
            agg = "mean"

        return MemorySimilarityGateConfig(
            policies=policies,
            policy_raw=policy_raw,
            multi_combine=combine,
            keyword_enabled=kw,
            embedding_enabled=emb,
            min_common=max(1, min_common),
            embed_threshold=max(0.0, min(1.0, embed_th)),
            include_historical_summary=inc_hist,
            embedding_config_path=cfg_path,
            embed_max_chars=max(64, min(embed_max_chars, 8000)),
            embed_max_chunks=max(1, min(embed_max_chunks, 64)),
            embed_chunk_agg=agg,
        )

    @classmethod
    def _inject_for_policy(
        cls,
        pol: str,
        cfg: MemorySimilarityGateConfig,
        *,
        memory_nonempty: bool,
        mem_side: str,
        topic_text: str,
    ) -> Dict[str, Any]:
        if pol == "memory_nonempty":
            return {"inject": bool(memory_nonempty)}
        if pol == "keyword":
            if not cfg.keyword_enabled:
                return {"inject": bool(memory_nonempty), "note": "keyword_disabled_fallback_memory_nonempty"}
            if not mem_side:
                return {"inject": False}
            hit = cls.keyword_overlap(topic_text, mem_side, min_common=cfg.min_common)
            return {"inject": bool(hit)}
        if pol == "embedding":
            if not cfg.embedding_enabled:
                return {"inject": bool(memory_nonempty), "note": "embedding_disabled_fallback_memory_nonempty"}
            if not mem_side:
                return {"inject": False, "similarity": None}
            sim = cls.embedding_similarity(topic_text, mem_side, cfg=cfg)
            if sim is None:
                return {"inject": False, "similarity": None}
            return {
                "inject": bool(sim >= cfg.embed_threshold),
                "similarity": float(sim),
                "threshold": float(cfg.embed_threshold),
            }
        logger.warning(
            f"MemorySimilarityGate: unknown policy {pol!r}, valid={sorted(cls._VALID_POLICIES)}"
        )
        return {"inject": False}

    @classmethod
    def evaluate_sync(
        cls,
        cfg: MemorySimilarityGateConfig,
        *,
        memory_nonempty: bool,
        memory_blob: str,
        topic_text: str,
        historical_summary: str = "",
    ) -> Dict[str, Any]:
        mem_side = (memory_blob or "").strip()
        if cfg.include_historical_summary and (historical_summary or "").strip():
            mem_side = f"{mem_side}\n{(historical_summary or '').strip()}".strip()
        per: Dict[str, Any] = {}
        for pol in cfg.policies:
            per[pol] = cls._inject_for_policy(
                pol, cfg, memory_nonempty=memory_nonempty, mem_side=mem_side, topic_text=topic_text
            )
        flags = [bool(per[p].get("inject")) for p in cfg.policies]
        combined = all(flags) if (cfg.multi_combine == "and" and flags) else any(flags) if flags else False
        out = dict(per)
        out["_combine_mode"] = cfg.multi_combine
        out["_combined_inject"] = combined
        return out

    @staticmethod
    def topic_text_from_blogs_chunk(chunk: Dict[str, Dict[str, Any]]) -> str:
        parts: list[str] = []
        for blog in (chunk or {}).values():
            if not isinstance(blog, dict):
                continue
            parts.append(str(blog.get("title", "") or ""))
            parts.append(str(blog.get("desc", "") or ""))
            tags = blog.get("tags_list") or []
            if isinstance(tags, (list, tuple)):
                parts.append(" ".join(str(t) for t in tags))
        return "\n".join(parts)

    @staticmethod
    def topic_text_from_mention_entries(entries: list) -> str:
        parts: list[str] = []
        for ent in entries or []:
            if not isinstance(ent, dict):
                continue
            n = ent.get("mention_blog")
            if isinstance(n, dict):
                parts.append(str(n.get("title", "") or ""))
                parts.append(str(n.get("desc", "") or ""))
                tags = n.get("tags_list") or []
                if isinstance(tags, (list, tuple)):
                    parts.append(" ".join(str(t) for t in tags))
                parts.append(str(n.get("content", "") or ""))
                parts.append(str(n.get("text", "") or ""))
                tags2 = n.get("tags") or []
                if isinstance(tags2, (list, tuple)):
                    parts.append(" ".join(str(t) for t in tags2))
            parts.append(str(ent.get("mention_comment_content", "") or ""))
        return "\n".join(parts)

    async def _memory_blob(self) -> str:
        memory = getattr(self._agent, "memory", None)
        if not memory:
            return ""
        try:
            items: list = []
            get_all = getattr(memory, "get_all_memory", None)
            if callable(get_all):
                buckets = await get_all()
                if isinstance(buckets, dict):
                    seen: Set[Any] = set()
                    for lst in buckets.values():
                        if not isinstance(lst, list):
                            continue
                        for msg in lst:
                            iid = getattr(msg, "id", None) or id(msg)
                            if iid in seen:
                                continue
                            seen.add(iid)
                            items.append(msg)
            if not items:
                items = list(await memory.retrieve("") or [])
            return "".join(getattr(msg, "content", "") or "" for msg in items)
        except Exception as e:
            logger.warning(
                f"UserAgent {getattr(self._agent, 'profile_id', '?')} memory blob failed: {e}"
            )
            return ""

    async def evaluate(self, topic_text: str) -> Dict[str, Any]:
        cfg = self.load_config(await memory_similarity_gate_params(self._agent))
        mem_blob = await self._memory_blob()
        return await asyncio.to_thread(
            self.evaluate_sync,
            cfg,
            memory_nonempty=bool(mem_blob.strip()),
            memory_blob=mem_blob,
            topic_text=topic_text or "",
        )

    def coaching(self, sim_result: Dict[str, Any]) -> Tuple[str, str, bool]:
        """Return (keyword_coaching, embedding_coaching, embedding_hit)."""
        kw_hit = bool((sim_result.get("keyword") or {}).get("inject"))
        emb_hit = bool((sim_result.get("embedding") or {}).get("inject"))
        return (
            self.KW_COACHING if kw_hit else "",
            self.EMB_COACHING if emb_hit else "",
            emb_hit,
        )

    async def run(self, topic_text: str) -> Tuple[str, str, bool]:
        sim = await self.evaluate(topic_text)
        return self.coaching(sim)

    async def memory_prompt_coaching(
        self,
        *,
        memory_nonempty_hit: bool,
        activity: float,
        context: ReactionContext,
    ) -> Tuple[str, str, str]:
        """Return (memory_coaching, memory_rec, memory_ref); strict step 1.5 needs low activity + memory_nonempty."""
        threshold = await user_low_activity_memory_gate_threshold(self._agent)
        if activity < threshold and memory_nonempty_hit:
            return (
                self.STRICT_STEP15_COACHING,
                "0. 破例须**同时**具备（1）可核对新信息点 **与**（2）强烈情绪动机 **与** （3）明确扩散动机，缺一仍 repost=false；若不触发步骤1.5 的重叠情形，本条可视为已满足；",
                "2-3句。无相关记忆可写「无相关记忆/首次接触」；有同题时说明是否重叠；若重叠倾向不转发，可说明是否仍有一句评论欲（仅评论不转发）",
            )
        _ = context
        return (
            "",
            "2. 关系与场景合适（关注关系优先）；",
            "2-3句。无相关记忆可写「无相关记忆/首次接触」；有同题时说明是否重叠；若重叠倾向不转发，可说明是否仍有一句评论欲（仅评论不转发）",
        )


class FreshnessGate:
    """Content age vs the agent's first-seen post anchor."""

    MS_PER_DAY = 86400000.0

    def __init__(self, agent: Any) -> None:
        self._agent = agent

    @staticmethod
    def _content_dicts(chunk: Union[Dict[str, Any], List[Any]]) -> List[Dict[str, Any]]:
        """Format the chunk into a list of content dictionaries"""
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
                    inner = entry.get("mention_note")
                if isinstance(inner, dict):
                    items.append(inner)
        return items

    def _evaluate_text(
        self,
        chunk: Union[Dict[str, Any], List[Any]],
        ref_ms: int,
        *,
        stale_days: float,
    ) -> Optional[str]:
        """Evaluate the freshness of the chunk"""
        # Get the earliest time in the chunk
        times_ms: List[float] = []
        for blog in self._content_dicts(chunk):
            t = time_to_ms(blog.get("time", blog.get("create_time")))
            if t is not None:
                times_ms.append(t)
        if not times_ms:
            return None

        earliest_ms = min(times_ms)
        ref_ok = bool(ref_ms and ref_ms > 0)
        ref_str = time_to_format_utc(float(ref_ms)) if ref_ok else "(Unknown)"

        anchor = getattr(self._agent, "_recommendation_earliest_post_anchor_ms", None)
        if anchor is None:
            self._agent._recommendation_earliest_post_anchor_ms = float(earliest_ms)
            return None
        if not ref_ok:
            return None

        # Calculate the time difference between the current reference time and the anchor time
        delta_ms = float(ref_ms) - float(anchor)
        days = delta_ms / self.MS_PER_DAY
        anchor_str = time_to_format_utc(float(anchor))
        interval_txt = (
            f"已过约 {days:.2f} 天（当前仿真时刻 − 锚点时刻，1 天 = 86400 秒）"
            if days >= 0
            else f"当前仿真时刻早于锚点约 {-days:.2f} 天（数据或时钟可能异常，请谨慎解读）"
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

    async def _coaching(
        self,
        time_text: Optional[str],
        context: ReactionContext,
        *,
        activity: float,
    ) -> str:
        if not time_text:
            return ""
        if context == "recommendation":
            return (
                f"【仿真时间与时效】\n            {time_text} - 若上文含 **警告** 或写明时效已明显减弱、"
                "倾向不回复 → **强烈倾向 repost=false**（该条不进候选池）；"
            )
        threshold = await user_low_activity_time_module_threshold(self._agent)
        if threshold is None or activity >= threshold:
            return ""
        return (
            "【仿真时间与时效】\n            "
            + time_text
            + " - 若上文含 **【警告】** 或写明时效已明显减弱、倾向不回复 → **强烈倾向 repost=false**（该条不进候选池）；"
        )

    async def run(
        self,
        chunk: Union[Dict[str, Any], List[Any]],
        current_timestamp: Any,
        context: ReactionContext,
        *,
        activity: float,
    ) -> str:
        ref_ms = int(time_to_ms(current_timestamp, default=0) or 0)
        stale_days = await user_stale_days(self._agent)
        time_text = self._evaluate_text(chunk, ref_ms, stale_days=stale_days)
        return await self._coaching(time_text, context, activity=activity)


class ActivityLevelGate:
    """Read agent activity_level from profile (clamped to [0, 1])."""

    def __init__(self, agent: Any) -> None:
        self._agent = agent

    def level(self) -> float:
        profile = getattr(self._agent, "profile", None)
        raw = profile.get_data("activity_level", 0.0) if profile else 0.0
        try:
            val = float(raw)
        except (TypeError, ValueError):
            val = 0.0
        return max(0.0, min(1.0, val))


class InteractionThreshold:
    """Sample per-round repost budget."""

    _DEFAULT_SAME = ([1, 2], [0.99, 0.01])
    _DEFAULT_DIFF = ([1, 2], [0.99, 0.01])
    _DEFAULT_KEEP = ([1, 2], [0.99, 0.01])

    def __init__(self, agent: Any) -> None:
        self._agent = agent
        self.k_same_target = 1
        self.k_diff_targets = 1
        self.k_keep_following = 1

    @staticmethod
    def _draw_discrete(rng: random.Random, support: List[int], probs: List[float]) -> int:
        u = rng.random()
        c = 0.0
        for x, p in zip(support, probs):
            c += p
            if u <= c:
                return x
        return support[-1]

    @classmethod
    def sample(cls, rng: random.Random) -> "InteractionThreshold":
        gate = cls(agent=None)  # type: ignore[arg-type]
        same_s, same_p = cls._DEFAULT_SAME
        diff_s, diff_p = cls._DEFAULT_DIFF
        keep_s, keep_p = cls._DEFAULT_KEEP
        gate.k_same_target = cls._draw_discrete(rng, same_s, same_p)
        gate.k_diff_targets = cls._draw_discrete(rng, diff_s, diff_p)
        gate.k_keep_following = cls._draw_discrete(rng, keep_s, keep_p)
        return gate

    @classmethod
    async def sample_from_agent(
        cls, agent: Any, rng: Optional[random.Random] = None
    ) -> "InteractionThreshold":
        return await cls(agent).run(rng)

    async def run(self, rng: Optional[random.Random] = None) -> "InteractionThreshold":
        from onesim.utils.midsim_params import draw_discrete, user_interaction_threshold_config

        cfg = await user_interaction_threshold_config(self._agent)
        rng = rng or random.Random()
        same_s, same_p = self._DEFAULT_SAME
        diff_s, diff_p = self._DEFAULT_DIFF
        keep_s, keep_p = self._DEFAULT_KEEP
        self.k_same_target = int(draw_discrete(rng, cfg.get("same_targets"), same_s, same_p))
        self.k_diff_targets = int(draw_discrete(rng, cfg.get("diff_targets"), diff_s, diff_p))
        self.k_keep_following = int(
            draw_discrete(rng, cfg.get("keep_following"), keep_s, keep_p)
        )
        return self


class UserAgentGates:
    """Run freshness, memory similarity, activity, and interaction gates."""

    def __init__(self, agent: Any) -> None:
        self._agent = agent
        self.freshness = FreshnessGate(agent)
        self.memory_similarity = MemorySimilarityGate(agent)
        self.activity = ActivityLevelGate(agent)
        self.interaction = InteractionThreshold(agent)

    async def build_recommendation_coaching(
        self,
        chunk: Dict[str, Dict[str, Any]],
        current_timestamp: Any,
    ) -> ReactionGateCoaching:
        return await self._build(
            chunk,
            current_timestamp,
            topic_text=MemorySimilarityGate.topic_text_from_blogs_chunk(chunk),
            context="recommendation",
            include_k_same_target=True,
        )

    async def build_mention_coaching(
        self,
        mention_entries: List[Dict[str, Any]],
        current_timestamp: Any,
    ) -> ReactionGateCoaching:
        return await self._build(
            mention_entries,
            current_timestamp,
            topic_text=MemorySimilarityGate.topic_text_from_mention_entries(mention_entries),
            context="mention",
            include_k_same_target=False,
        )

    async def _build(
        self,
        chunk: Union[Dict[str, Any], List[Any]],
        current_timestamp: Any,
        *,
        topic_text: str,
        context: ReactionContext,
        include_k_same_target: bool,
    ) -> ReactionGateCoaching:
        act = self.activity.level()

        freshness_coaching = await self.freshness.run(
            chunk, current_timestamp, context, activity=act
        )

        sim = await self.memory_similarity.evaluate(topic_text)
        kw_coaching, emb_coaching, _emb_hit = self.memory_similarity.coaching(sim)
        mem_nonempty_hit = bool((sim.get("memory_nonempty") or {}).get("inject"))

        memory_coaching, memory_rec, memory_ref = await self.memory_similarity.memory_prompt_coaching(
            memory_nonempty_hit=mem_nonempty_hit,
            activity=act,
            context=context,
        )

        threshold = await self.interaction.run()

        return ReactionGateCoaching(
            freshness_coaching=freshness_coaching,
            similarity_kw_coaching=kw_coaching,
            similarity_emb_coaching=emb_coaching,
            memory_coaching=memory_coaching,
            memory_rec=memory_rec,
            memory_ref=memory_ref,
            k_same_target=threshold.k_same_target if include_k_same_target else 1,
            k_diff_targets=threshold.k_diff_targets,
        )

