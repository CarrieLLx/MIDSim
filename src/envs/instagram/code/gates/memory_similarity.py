# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import json
from .target_similarity import compare_targets
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger
from onesim.utils.midsim_params import MIDSimParams

from ..prompting.loader import load_policy
from ..utils.text import keyword_set
from ..embedding import EmbeddingService, default_embedding_config_path


@dataclass(frozen=True)
class MemorySimilarityGateConfig:
    policies: Tuple[str, ...]
    policy_raw: str
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

    _VALID_POLICIES = frozenset(
        ("memory_nonempty", "keyword", "embedding")
    )

    def __init__(self, agent: Any) -> None:
        self._agent = agent

    @classmethod
    def keyword_overlap(
        cls,
        topic_text: str,
        memory_side_text: str,
        *,
        min_common: int = 2,
    ) -> bool:
        a = keyword_set(topic_text)
        b = keyword_set(memory_side_text)
        if not a or not b:
            return False
        return len(a & b) >= min_common

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
        kw = cls._parse_config_bool(p.get("keyword_enabled"), True)
        emb = cls._parse_config_bool(p.get("embedding_enabled"), True)

        try:
            min_common = int(p.get("min_common_tokens", 2) or 2)
        except (TypeError, ValueError):
            min_common = 2

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
                return {"inject": bool(memory_nonempty), "post": "keyword_disabled_fallback_memory_nonempty"}
            if not mem_side:
                return {"inject": False}
            hit = cls.keyword_overlap(topic_text, mem_side, min_common=cfg.min_common)
            return {"inject": bool(hit)}
        if pol == "embedding":
            if not cfg.embedding_enabled:
                return {"inject": bool(memory_nonempty), "post": "embedding_disabled_fallback_memory_nonempty"}
            if not mem_side:
                return {"inject": False, "similarity": None}
            try:
                return compare_targets(cfg, topic_text, mem_side)
            except Exception as exc:
                logger.warning("target memory similarity failed: {}", exc)
                return {"inject": False, "similarity": None, "error": str(exc)}
        logger.warning(f"MemorySimilarityGate: unknown policy {pol!r}, valid={sorted(cls._VALID_POLICIES)}")
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
        return per

    @staticmethod
    def topic_text_from_posts_chunk(chunk: Dict[str, Dict[str, Any]]) -> str:
        targets = []
        for pid, post in (chunk or {}).items():
            if not isinstance(post, dict):
                continue
            targets.append({"target": str(pid), "text": str(post.get("content") or "")})
            comments = post.get("comments", {})
            entries = comments.items() if isinstance(comments, dict) else enumerate(comments or [])
            for cid, comment in entries:
                if isinstance(comment, dict):
                    targets.append({"target": str(pid) + "/" + str(comment.get("comment_id", cid)),
                                    "text": str(comment.get("content") or "")})
        return json.dumps(targets, ensure_ascii=False)

    @staticmethod
    def topic_text_from_mention_entries(entries: list) -> str:
        targets = []
        for i, ent in enumerate(entries or []):
            post = ent.get("mention_post")
            if isinstance(post, dict):
                targets.append({"target": str(i) + "/post", "text": str(post.get("content") or "")})
            targets.append({"target": str(i) + "/comment", "text": str(ent.get("mention_comment_content") or "")})
        return json.dumps(targets, ensure_ascii=False)

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
            return "\n".join(getattr(msg, "content", "") or "" for msg in items)
        except Exception as e:
            logger.warning(
                f"UserAgent {getattr(self._agent, 'profile_id', '?')} memory blob failed: {e}"
            )
            return ""

    async def evaluate(self, topic_text: str) -> Dict[str, Any]:
        params = await MIDSimParams.load(self._agent)
        cfg = self.load_config(params.gates.memory_similarity)
        mem_blob = await self._memory_blob()
        result = await asyncio.to_thread(
            self.evaluate_sync,
            cfg,
            memory_nonempty=bool(mem_blob.strip()),
            memory_blob=mem_blob,
            topic_text=topic_text or "",
            historical_summary=str(self._agent.profile.get_data("historical_summary", "") or "") if getattr(self._agent, "profile", None) else "",
        )
        logger.info("Memory similarity audit user={} result={}", getattr(self._agent, "profile_id", "?"), json.dumps(result, ensure_ascii=False))
        return result

    def coaching(self, sim_result: Dict[str, Any]) -> Tuple[str, str, bool]:
        kw_hit = bool((sim_result.get("keyword") or {}).get("inject"))
        emb_hit = bool((sim_result.get("embedding") or {}).get("inject"))
        return (
            load_policy("similarity", "keyword_overlap", "guidance") if kw_hit else "",
            (load_policy("similarity", "embedding_similarity", "guidance") + "\nMatched targets (apply repetition checks to these targets individually; unmatched targets are not proven repeated): " + json.dumps([{k: r[k] for k in ("target", "similarity", "matched_input", "matched_memory")} for r in (sim_result.get("embedding") or {}).get("targets", []) if r["inject"]], ensure_ascii=False)) if emb_hit else "",
            emb_hit,
        )

    async def run(self, topic_text: str) -> Tuple[str, str, bool]:
        sim = await self.evaluate(topic_text)
        return self.coaching(sim)

    async def memory_prompt_coaching(
        self,
        *,
        embedding_hit: bool,
        activity: float,
        context: str,
    ) -> Tuple[str, str]:
        if context == "author":
            return (
                load_policy("memory", "author", "coaching"),
                load_policy("memory", "author", "reflection"),
            )
        params = await MIDSimParams.load(self._agent)
        threshold = params.gates.low_activity_memory_threshold
        if activity <= threshold and embedding_hit:
            coaching = load_policy("memory", "strict", "coaching")
            return (
                coaching,
                load_policy("memory", "strict", "reflection"),
            )
        return (
            load_policy("memory", "default", "coaching"),
            load_policy("memory", "default", "reflection"),
        )
