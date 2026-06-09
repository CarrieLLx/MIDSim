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
    user_stale_days,
)

from .embedding_client import (
    cosine_similarity,
    default_embedding_config_path,
    get_embeddings,
    load_embedding_config,
)
from .utils import time_to_format_utc, time_to_ms, tweet_ref_key

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
    propagation_type_coaching: str = ""


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
        "\n\n【Topic vs memory — keyword overlap】\n            "
        "- This batch is judged to be strongly keyword-related to stored memory → **strong bias toward propagation=false** "
        "(high risk of repeating stances or same-thread discussion already in memory; consider repost=true only if step 1.5 "
        "clearly meets break-out conditions such as verifiable new information and strong motivation);"
    )
    EMB_COACHING = (
        "\n\n【Topic vs memory — semantic similarity】\n            "
        "- Embedding similarity meets the configured threshold; this batch is close in meaning to memory → "
        "**strong bias toward propagation=false** (treat as same thread / easy duplicate topic; strictly re-check step 1.5 "
        "before repost=true);"
    )
    STRICT_STEP15_COACHING = """
                Step 1.5: Check for "duplicate topic" against memory (complete before step 2)
                - If the current content refers to the same incident / dispute / issue as memory, and your stance, tone, or conclusion in memory would closely match what you would say in this reply → treat this as "almost certainly keep the default of propagation=false", unless you can state clearly that, relative to what is already in memory, you will add **a new fact readers can perceive** (you must be able to name it: a specific person, event, time, rule, or number; rephrasing alone or vague "new angle" / "new reasoning" does not count).
                - If memory already has multiple entries on the same kind of topic, or you have recently engaged on similar content → lean toward propagation=false this round unless there is **materially new development** (new information, or the other party raises an argument not covered in memory).
                - On the recommendation feed with weak ties, or when your relationship with the author is weak, be more conservative about making an exception.
                - **memory_reflection must not contradict itself**: if you first say it overlaps with memory / same topic / already discussed / related to a prior event, then use "but" / "although" / "things may have updated" / "might still weigh in" / "still worth a brief comment" or similar turns **without a concrete new fact** to hint that you should comment — treat that as invalid; if you judge overlap or near–no engagement, memory_reflection must **throughout** conclude toward silence or clearly no new information, and must not use vague wording to let yourself off the hook.
                - **The more you have in memory, the more you should hold back**: even if (1) and (2) can be met on paper, still treat **propagation=true for this item as a low-probability event**—default stays propagation=false; break the rule only when new information **clearly escalates** the situation (e.g. changes the phase of the event, overturns or corrects a judgment already in your memory, or introduces a key new actor or rule). Do not repost just because you "barely" satisfy the bar with a throwaway line.
                - **Also disallowed**: admitting the same thread as memory, then using vague lines like "but this gives a new instance of the problem" / "new situation" / "another case" / "worth noting" / "worth further attention" / "new informational anchor" to imply you should comment — unless the same sentence states **one namable difference** beyond memory (e.g. a phenomenon, time, link, or rule name **unique** to this post).
                Summarize the above briefly in memory_reflection; if you do not comment, decision_reason must state overlap with memory / already stated / no new information, etc.
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
        a = cls.tokenize(topic_text)
        b = cls.tokenize(memory_side_text)
        if not a or not b:
            return False
        return len(a & b) >= min_common

    @classmethod
    def _chunk_text_for_embedding(cls, text: str, max_chars: int) -> List[str]:
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
    def _tweet_body_text(obj: Any) -> str:
        if not isinstance(obj, dict):
            return ""
        for key in ("content", "text", "full_text"):
            s = str(obj.get(key, "") or "").strip()
            if s:
                return s
        return ""

    @classmethod
    def _merge_tweet_with_pool(
        cls,
        tweet: Dict[str, Any],
        content_pool: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not isinstance(tweet, dict):
            return tweet
        if not content_pool or not isinstance(content_pool, dict):
            return tweet
        for k in (tweet.get("tweet_id"), tweet.get("id")):
            sid = str(k or "").strip()
            if not sid:
                continue
            raw = content_pool.get(sid)
            if isinstance(raw, dict):
                return {**raw, **tweet}
        return tweet

    @classmethod
    def _append_tweet_topic_parts(
        cls,
        obj: Any,
        parts: List[str],
        *,
        depth: int = 0,
        content_pool: Optional[Dict[str, Any]] = None,
        max_depth: int = 2,
    ) -> None:
        if depth > max_depth or not isinstance(obj, dict):
            return
        parts.append(str(obj.get("title", "") or ""))
        parts.append(str(obj.get("desc", "") or ""))
        tags = obj.get("tags_list") or []
        if isinstance(tags, (list, tuple)):
            parts.append(" ".join(str(t) for t in tags))
        bt = cls._tweet_body_text(obj)
        if bt:
            parts.append(bt)
        for k in ("quoted_tweet", "replied_tweet", "retweeted_tweet"):
            ch = obj.get(k)
            if isinstance(ch, dict):
                ch_m = cls._merge_tweet_with_pool(ch, content_pool)
                cls._append_tweet_topic_parts(
                    ch_m, parts, depth=depth + 1, content_pool=content_pool, max_depth=max_depth
                )
        rid = obj.get("retweeted_tweet_id")
        if rid and not isinstance(obj.get("retweeted_tweet"), dict) and content_pool:
            key = str(rid).strip()
            if key:
                parent = content_pool.get(key)
                if isinstance(parent, dict):
                    pm = cls._merge_tweet_with_pool(parent, content_pool)
                    cls._append_tweet_topic_parts(
                        pm, parts, depth=depth + 1, content_pool=content_pool, max_depth=max_depth
                    )

    @staticmethod
    def _mean_embedding(vectors: List[List[float]]) -> List[float]:
        if not vectors:
            return []
        dim = len(vectors[0])
        acc = [0.0] * dim
        for v in vectors:
            for i, x in enumerate(v):
                acc[i] += float(x)
        n = float(len(vectors))
        return [x / n for x in acc]

    @classmethod
    def _embed_chunks_sequential(
        cls,
        base_url: str,
        model_name: str,
        chunks: List[str],
    ) -> List[List[float]]:
        vecs: List[List[float]] = []
        for ch in chunks:
            batch = get_embeddings(base_url, model_name, [ch])
            if len(batch) != 1:
                raise ValueError("embedding API returned abnormal chunk count")
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
        max_chars = max(64, min(int(cfg.embed_max_chars), 8000))
        max_chunks = max(1, min(int(cfg.embed_max_chunks), 64))
        agg = cfg.embed_chunk_agg if cfg.embed_chunk_agg in ("mean", "max") else "mean"

        chunks_t = cls._chunk_text_for_embedding(topic_text or "", max_chars)[:max_chunks]
        chunks_m = cls._chunk_text_for_embedding(memory_side_text or "", max_chars)[:max_chunks]
        if not chunks_t or not chunks_m:
            return None
        try:
            base_url, model_name = load_embedding_config(cfg.embedding_config_path)
            vt = cls._embed_chunks_sequential(base_url, model_name, chunks_t)
            vm = cls._embed_chunks_sequential(base_url, model_name, chunks_m)
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
        p = params or {}
        policy_raw = str(p.get("policy", "memory_nonempty,keyword,embedding")).strip()
        policies = cls._parse_policy_list(policy_raw)
        combine = str(p.get("multi_combine", "or")).strip().lower()
        if combine not in ("or", "and"):
            combine = "or"
        kw = cls._parse_config_bool(p.get("keyword_enabled"), True)
        emb = cls._parse_config_bool(p.get("embedding_enabled"), True)
        try:
            min_common = int(p.get("min_common_tokens", 8) or 8)
        except (TypeError, ValueError):
            min_common = 8
        try:
            embed_th = float(p.get("embed_threshold", 0.65) or 0.65)
        except (TypeError, ValueError):
            embed_th = 0.65
        inc_hist = cls._parse_config_bool(p.get("include_historical_summary"), True)
        cfg_path = str(p.get("embedding_config_path", "") or "").strip()
        if not cfg_path:
            cfg_path = default_embedding_config_path()
        try:
            embed_max_chars = int(p.get("embed_max_chars", 400) or 400)
        except (TypeError, ValueError):
            embed_max_chars = 400
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

    @classmethod
    def should_inject(
        cls,
        cfg: MemorySimilarityGateConfig,
        *,
        memory_nonempty: bool,
        memory_blob: str,
        topic_text: str,
        historical_summary: str = "",
    ) -> bool:
        ev = cls.evaluate_sync(
            cfg,
            memory_nonempty=memory_nonempty,
            memory_blob=memory_blob,
            topic_text=topic_text,
            historical_summary=historical_summary,
        )
        return bool(ev.get("_combined_inject"))

    @classmethod
    def topic_text_from_tweets_chunk(
        cls,
        chunk: Dict[str, Dict[str, Any]],
        content_pool: Optional[Dict[str, Any]] = None,
    ) -> str:
        parts: list[str] = []
        for tweet in (chunk or {}).values():
            if not isinstance(tweet, dict):
                continue
            merged = cls._merge_tweet_with_pool(tweet, content_pool)
            cls._append_tweet_topic_parts(merged, parts, depth=0, content_pool=content_pool)
        return "\n".join(p for p in parts if p)

    @classmethod
    def topic_text_from_mention_entries(
        cls,
        entries: list,
        content_pool: Optional[Dict[str, Any]] = None,
    ) -> str:
        parts: list[str] = []
        for ent in entries or []:
            if not isinstance(ent, dict):
                continue
            n = ent.get("mention_tweet")
            if isinstance(n, dict):
                nm = cls._merge_tweet_with_pool(n, content_pool)
                cls._append_tweet_topic_parts(nm, parts, depth=0, content_pool=content_pool)
            parts.append(str(ent.get("mention_comment_content", "") or ""))
        return "\n".join(p for p in parts if p)

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
        kw_hit = bool((sim_result.get("keyword") or {}).get("inject"))
        emb_hit = bool((sim_result.get("embedding") or {}).get("inject"))
        return (
            self.KW_COACHING if kw_hit else "",
            self.EMB_COACHING if emb_hit else "",
            emb_hit,
        )

    async def memory_prompt_coaching(
        self,
        *,
        embedding_hit: bool,
        activity: float,
    ) -> Tuple[str, str, str]:
        threshold = await user_low_activity_memory_gate_threshold(self._agent)
        if activity < threshold and embedding_hit:
            return (
                self.STRICT_STEP15_COACHING,
                '0. Not trapped by step 1.5 "almost no engagement", or you have a verifiable new point;',
                "2-3 sentences: same event as memory? already stated? should stay silent? if exception, name the new info",
            )
        return (
            "",
            "",
            '1–2 sentences. When there is no stored memory to compare against, write "no relevant memory / first exposure".',
        )


class FreshnessGate:
    """Content age vs the agent's first-seen post anchor."""

    MS_PER_DAY = 86400000.0

    def __init__(self, agent: Any) -> None:
        self._agent = agent

    @staticmethod
    def _content_dicts(chunk: Union[Dict[str, Any], List[Any]]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        if isinstance(chunk, dict):
            for v in chunk.values():
                if isinstance(v, dict):
                    items.append(v)
        elif isinstance(chunk, list):
            for entry in chunk:
                if not isinstance(entry, dict):
                    continue
                inner = entry.get("mention_tweet")
                if not isinstance(inner, dict):
                    inner = entry.get("mention_blog")
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
        times_ms: List[float] = []
        for tweet in self._content_dicts(chunk):
            t = time_to_ms(tweet.get("time", tweet.get("create_time")))
            if t is not None:
                times_ms.append(t)
        if not times_ms:
            return None

        earliest_ms = min(times_ms)
        ref_ok = bool(ref_ms and ref_ms > 0)
        ref_str = time_to_format_utc(float(ref_ms)) if ref_ok else "unknown"

        anchor = getattr(self._agent, "_recommendation_earliest_post_anchor_ms", None)
        if anchor is None:
            self._agent._recommendation_earliest_post_anchor_ms = float(earliest_ms)
            return None
        if not ref_ok:
            return None

        delta_ms = float(ref_ms) - float(anchor)
        days = delta_ms / self.MS_PER_DAY
        anchor_str = time_to_format_utc(float(anchor))
        if days >= 0:
            interval_txt = (
                f"about {days:.2f} days have elapsed "
                "(current simulation time − anchor; 1 day = 86400 s)"
            )
        else:
            interval_txt = (
                f"current simulation time is about {-days:.2f} days earlier than the anchor "
                "(data or clock may be inconsistent; interpret cautiously)"
            )
        lines = [
            f"[Time] Current simulation window start: {ref_str}.",
            f"Relative to the agent's anchored earliest post time ({anchor_str}), {interval_txt}.",
        ]
        if days > stale_days:
            lines.append(
                f"[Warning] More than about {stale_days:.0f} days have elapsed; content is likely stale. "
                "Unless you have a strong reason or a traceable new fact, prefer **no engagement** (silence)."
            )
        return "\n            ".join(lines)

    @staticmethod
    def _coaching(time_text: Optional[str], context: ReactionContext) -> str:
        if not time_text:
            return ""
        suffix = (
            " - If the text above contains **【WARNING】** or states that recency has clearly faded / "
            "you lean toward propagation=false → **strong bias toward propagation=false**"
        )
        if context == "recommendation":
            suffix += " (do not put this item in the candidate pool);"
        return f"【Simulated time & recency】\n            {time_text}{suffix}"

    async def run(
        self,
        chunk: Union[Dict[str, Any], List[Any]],
        current_timestamp: Any,
        context: ReactionContext,
    ) -> str:
        ref_ms = int(time_to_ms(current_timestamp, default=0) or 0)
        stale_days = await user_stale_days(self._agent)
        time_text = self._evaluate_text(chunk, ref_ms, stale_days=stale_days)
        return self._coaching(time_text, context)


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
    """Sample per-round propagation budget and default propagation/mention type."""

    _DEFAULT_SAME = ([1, 2, 3], [0.6, 0.3, 0.1])
    _DEFAULT_DIFF = ([1, 2, 3, 4, 5], [0.4, 0.3, 0.2, 0.05, 0.05])
    _DEFAULT_KEEP = ([0, 1], [0.9961, 0.0039])
    _DEFAULT_PROPAGATION = (["retweet", "reply", "quote"], [0.7, 0.03, 0.27])
    _DEFAULT_MENTION = (["retweet", "reply", "quote"], [0.0, 0.9, 0.1])

    def __init__(self, agent: Any) -> None:
        self._agent = agent
        self.k_same_target = 1
        self.k_diff_targets = 1
        self.k_keep_following = 0
        self.propagation_type = "retweet"
        self.mention_type = "reply"

    @staticmethod
    def _draw_discrete(rng: random.Random, support: List, probs: List[float]):
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
        gate.k_same_target = int(cls._draw_discrete(rng, *cls._DEFAULT_SAME))
        gate.k_diff_targets = int(cls._draw_discrete(rng, *cls._DEFAULT_DIFF))
        gate.k_keep_following = int(cls._draw_discrete(rng, *cls._DEFAULT_KEEP))
        gate.propagation_type = str(cls._draw_discrete(rng, *cls._DEFAULT_PROPAGATION))
        gate.mention_type = str(cls._draw_discrete(rng, *cls._DEFAULT_MENTION))
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
        prop_s, prop_p = self._DEFAULT_PROPAGATION
        mention_s, mention_p = self._DEFAULT_MENTION
        self.k_same_target = int(draw_discrete(rng, cfg.get("same_targets"), same_s, same_p))
        self.k_diff_targets = int(draw_discrete(rng, cfg.get("diff_targets"), diff_s, diff_p))
        self.k_keep_following = int(draw_discrete(rng, cfg.get("keep_following"), keep_s, keep_p))
        self.propagation_type = str(draw_discrete(rng, cfg.get("propagation_type"), prop_s, prop_p))
        self.mention_type = str(draw_discrete(rng, cfg.get("mention_type"), mention_s, mention_p))
        return self

    @staticmethod
    def propagation_type_coaching(propagation_type: str) -> str:
        if propagation_type == "retweet":
            return """
                - propagation_type ∈ {{"retweet","reply","quote"}}. Default "retweet".
                · "retweet" — Pure amplify: you believe the post is worth surfacing (e.g. timely, credible, useful, funny, or important to your audience) and you want followers to see it **as-is**—no added stance, no thread commentary; propagation_content "".
                · "reply" — You address the author or a concrete point voice aimed at the author.
                · "quote" — You add judgment, context, or extension for the curious onlookers; voice aimed at the audience.
                """
        if propagation_type == "reply":
            return """
                - propagation_type ∈ {{"retweet","reply","quote"}}. Default "reply".
                · "retweet" — Pure amplify: you believe the post is worth surfacing (e.g. timely, credible, useful, funny, or important to your audience) and you want followers to see it **as-is**—no added stance, no thread commentary; propagation_content "".
                · "reply" — You address the author or a concrete point; voice aimed at the author.
                · "quote" — You add judgment, context, or extension for the curious onlookers; voice aimed at the audience.
                """
        if propagation_type == "quote":
            return """
                - propagation_type ∈ {{"retweet","reply","quote"}}. Default "quote".
                · "retweet" — You only agree/amplify without adding stance; propagation_content "".
                · "reply" — You address the author or a concrete point; voice aimed at the author.
                · "quote" — You add judgment, context, or extension for the curious onlookers; voice aimed at the audience.
                """
        return ""


class TweetDepthGate:
    """Env seed-root hop depth coaching for LLM instructions."""

    INJECTION_PROB = 0.0

    @staticmethod
    def _immediate_parent_tweet_id_env(tw: Dict[str, Any]) -> Optional[str]:
        if not isinstance(tw, dict):
            return None
        for key in ("retweeted_tweet_id", "quoted_tweet_id", "replied_tweet_id", "replyed_tweet_id"):
            s = tweet_ref_key(tw.get(key))
            if s:
                return s
        return None

    @classmethod
    def resolve_tweet_to_env_seed_root(
        cls,
        tweet_id: str,
        content_pool: Dict[str, Any],
        seed_ids: Set[str],
        max_hops: int = 512,
    ) -> Optional[str]:
        cur = str(tweet_id).strip()
        if not cur or not isinstance(content_pool, dict) or not seed_ids:
            return None
        for _ in range(max_hops):
            if cur in seed_ids:
                return cur
            tw = content_pool.get(cur)
            if not isinstance(tw, dict):
                return None
            pid = cls._immediate_parent_tweet_id_env(tw)
            if not pid or pid == cur:
                return None
            cur = str(pid).strip()
            if not cur:
                return None
        return None

    @classmethod
    def hop_edges_to_env_seed_root(
        cls,
        tweet_id: str,
        content_pool: Dict[str, Any],
        seed_ids: Set[str],
        max_hops: int = 512,
    ) -> Optional[int]:
        if not str(tweet_id).strip() or not isinstance(content_pool, dict) or not seed_ids:
            return None
        cur = str(tweet_id).strip()
        edges = 0
        for _ in range(max_hops):
            if cur in seed_ids:
                return edges
            tw = content_pool.get(cur)
            if not isinstance(tw, dict):
                return None
            pid = cls._immediate_parent_tweet_id_env(tw)
            if not pid or pid == cur:
                return None
            cur = str(pid).strip()
            if not cur:
                return None
            edges += 1
        return None

    @staticmethod
    def format_propagation_depth_hint(
        depth: Optional[int],
        resolved_root: Optional[str],
    ) -> str:
        if depth is None:
            return ""
        root_s = str(resolved_root).strip() if resolved_root else "?"
        if depth == 0:
            return ""
        base = (
            f"**{depth}** hop(s) from the env seed root (same as Repost Hop Depth / hop_{depth}); "
            f"resolved seed tweet_id={root_s}."
        )
        if depth >= 3:
            base += " [≥3 hops: **lean toward propagation=false** unless new fact, correction, or stance shift—not +1/pile-on.]"
            base += " Use [Env-computed propagation depth] per alert; do not estimate. **If depth ≥ 3:** lean toward propagation=false, engage only for new fact, error fix, or real stance change (not courtesy/emoji/pile-on). If propagation=false, decision_reason may say e.g. `d≥3 skip`."
        return base

    @staticmethod
    def format_coaching_block(lines: List[str]) -> str:
        if not lines:
            return ""
        body = "\n            ".join(lines)
        return "\n\n[Env propagation depth — coaching]\n            " + body

    @classmethod
    def depth_line(
        cls,
        tweet_id: str,
        content_pool: Dict[str, Any],
        seed_ids: Set[str],
    ) -> str:
        dep = cls.hop_edges_to_env_seed_root(tweet_id, content_pool, seed_ids)
        root = cls.resolve_tweet_to_env_seed_root(tweet_id, content_pool, seed_ids)
        return f"(tweet_id={tweet_id}) {cls.format_propagation_depth_hint(dep, root)}"

    @classmethod
    def _maybe_inject(cls, block: str, rng: Optional[random.Random] = None) -> str:
        if not block:
            return ""
        r = rng if rng is not None else random.Random()
        return block if r.random() < cls.INJECTION_PROB else ""

    @classmethod
    def coaching_for_recommendation_chunk(
        cls,
        chunk: Dict[str, Dict[str, Any]],
        content_pool: Dict[str, Any],
        seed_ids: Set[str],
        *,
        rng: Optional[random.Random] = None,
    ) -> str:
        lines: List[str] = []
        for rec_tid, rec_tweet in chunk.items():
            if not isinstance(rec_tweet, dict):
                continue
            tid_key = tweet_ref_key(rec_tid) or tweet_ref_key(
                rec_tweet.get("tweet_id") or rec_tweet.get("id")
            )
            if not tid_key:
                continue
            lines.append(cls.depth_line(tid_key, content_pool, seed_ids))
        return cls._maybe_inject(cls.format_coaching_block(lines), rng)

    @classmethod
    def coaching_for_mention_entries(
        cls,
        entries: List[Dict[str, Any]],
        content_pool: Dict[str, Any],
        seed_ids: Set[str],
        *,
        rng: Optional[random.Random] = None,
    ) -> str:
        lines: List[str] = []
        for entry in entries:
            tw = entry.get("mention_tweet") if isinstance(entry.get("mention_tweet"), dict) else {}
            tid_key = tweet_ref_key(entry.get("tweet_id")) or tweet_ref_key(
                tw.get("tweet_id") or tw.get("id")
            )
            if not tid_key:
                continue
            lines.append(cls.depth_line(tid_key, content_pool, seed_ids))
        return cls._maybe_inject(cls.format_coaching_block(lines), rng)


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
        *,
        content_pool: Optional[Dict[str, Any]] = None,
    ) -> ReactionGateCoaching:
        return await self._build(
            chunk,
            current_timestamp,
            topic_text=MemorySimilarityGate.topic_text_from_tweets_chunk(chunk, content_pool),
            context="recommendation",
            propagation_type_key="propagation_type",
        )

    async def build_mention_coaching(
        self,
        mention_entries: List[Dict[str, Any]],
        current_timestamp: Any,
        *,
        content_pool: Optional[Dict[str, Any]] = None,
    ) -> ReactionGateCoaching:
        chunk_for_time = [{"mention_tweet": e["mention_tweet"]} for e in mention_entries]
        return await self._build(
            chunk_for_time,
            current_timestamp,
            topic_text=MemorySimilarityGate.topic_text_from_mention_entries(
                mention_entries, content_pool
            ),
            context="mention",
            propagation_type_key="mention_type",
        )

    async def _build(
        self,
        chunk: Union[Dict[str, Any], List[Any]],
        current_timestamp: Any,
        *,
        topic_text: str,
        context: ReactionContext,
        propagation_type_key: str,
    ) -> ReactionGateCoaching:
        act = self.activity.level()

        freshness_coaching = await self.freshness.run(chunk, current_timestamp, context)

        sim = await self.memory_similarity.evaluate(topic_text)
        kw_coaching, emb_coaching, emb_hit = self.memory_similarity.coaching(sim)
        memory_coaching, memory_rec, memory_ref = await self.memory_similarity.memory_prompt_coaching(
            embedding_hit=emb_hit,
            activity=act,
        )

        threshold = await self.interaction.run()
        prop_type = (
            threshold.propagation_type
            if propagation_type_key == "propagation_type"
            else threshold.mention_type
        )

        return ReactionGateCoaching(
            freshness_coaching=freshness_coaching,
            similarity_kw_coaching=kw_coaching,
            similarity_emb_coaching=emb_coaching,
            memory_coaching=memory_coaching,
            memory_rec=memory_rec,
            memory_ref=memory_ref,
            k_same_target=threshold.k_same_target,
            k_diff_targets=threshold.k_diff_targets,
            propagation_type_coaching=InteractionThreshold.propagation_type_coaching(prop_type),
        )
