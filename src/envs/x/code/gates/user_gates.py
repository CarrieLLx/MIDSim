# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Union

from .interaction import ActivityLevelGate
from .memory_similarity import MemorySimilarityGate

ReactionContext = Literal["recommendation", "mention", "author", "profile_history"]


@dataclass
class ReactionGateCoaching:
    """Combined prompt coaching injected into LLM instruction templates."""

    similarity_kw_coaching: str = ""
    similarity_emb_coaching: str = ""
    memory_coaching: str = ""
    memory_ref: str = ""


class UserAgentGates:
    """Run memory similarity and activity gates."""

    def __init__(self, agent: Any) -> None:
        self._agent = agent
        self.memory_similarity = MemorySimilarityGate(agent)
        self.activity = ActivityLevelGate(agent)

    async def build_recommendation_coaching(
        self,
        chunk: Dict[str, Dict[str, Any]],
        current_timestamp: Any,
    ) -> ReactionGateCoaching:
        return await self._build(
            chunk,
            current_timestamp,
            topic_text=MemorySimilarityGate.topic_text_from_posts_chunk(chunk),
            context="recommendation",
        )

    async def build_profile_history_coaching(
        self,
        posts: Dict[str, Dict[str, Any]],
        current_timestamp: Any,
    ) -> ReactionGateCoaching:
        return await self._build(
            posts,
            current_timestamp,
            topic_text=MemorySimilarityGate.topic_text_from_posts_chunk(posts),
            context="profile_history",
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
        )

    async def build_author_coaching(
        self,
        mention_entries: List[Dict[str, Any]],
        current_timestamp: Any,
    ) -> ReactionGateCoaching:
        return await self._build(
            mention_entries,
            current_timestamp,
            topic_text=MemorySimilarityGate.topic_text_from_mention_entries(mention_entries),
            context="author",
        )

    async def _build(
        self,
        chunk: Union[Dict[str, Any], List[Any]],
        current_timestamp: Any,
        *,
        topic_text: str,
        context: ReactionContext,
    ) -> ReactionGateCoaching:
        act = self.activity.level()


        kw_coaching, emb_coaching, emb_hit = await self.memory_similarity.run(topic_text)
        if context == "author":
            kw_coaching = ""
            emb_coaching = ""

        memory_coaching, memory_ref = await self.memory_similarity.memory_prompt_coaching(
            embedding_hit=emb_hit,
            activity=act,
            context=context,
        )

        return ReactionGateCoaching(
            similarity_kw_coaching=kw_coaching,
            similarity_emb_coaching=emb_coaching,
            memory_coaching=memory_coaching,
            memory_ref=memory_ref,
        )
