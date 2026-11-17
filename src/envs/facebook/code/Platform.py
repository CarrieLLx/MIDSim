# -*- coding: utf-8 -*-
"""Thin platform entry: route user requests to recommendation / search Algorithm agents."""
from __future__ import annotations

from typing import List

from loguru import logger
from onesim.agent import GeneralAgent
from onesim.events import Event
from onesim.memory import MemoryStrategy
from onesim.planning import PlanningBase
from onesim.profile import AgentProfile
from onesim.relationship import RelationshipManager

from .events import GetAlgorithmRecommendationEvent, GetSearchResultEvent


class Platform(GeneralAgent):
    """Forward recommendation / search requests. No ranking."""

    def __init__(
        self,
        sys_prompt: str | None = None,
        model_config_name: str = None,
        event_bus_queue=None,
        profile: AgentProfile = None,
        memory: MemoryStrategy = None,
        planning: PlanningBase = None,
        relationship_manager: RelationshipManager = None,
    ) -> None:
        super().__init__(
            sys_prompt,
            model_config_name,
            event_bus_queue,
            profile,
            memory,
            planning,
            relationship_manager,
        )
        self.register_event("StartEvent", "register_services")
        self.register_event("GetAlgorithmRecommendationEvent", "route_recommendation")
        self.register_event("GetSearchResultEvent", "route_search")
        self._recommendation_agent_id = ""
        self._search_agent_id = ""

    async def register_services(self, event: Event) -> List[Event]:
        rec_id = str(getattr(event, "recommendation_agent_id", "") or "").strip()
        search_id = str(getattr(event, "search_agent_id", "") or "").strip()
        if rec_id:
            self._recommendation_agent_id = rec_id
        if search_id:
            self._search_agent_id = search_id
        return []

    async def route_recommendation(self, event: Event) -> List[Event]:
        if not self._recommendation_agent_id:
            logger.warning("Platform: recommendation Algorithm is not registered")
            return []
        return [
            GetAlgorithmRecommendationEvent(
                from_agent_id=event.from_agent_id,
                to_agent_id=self._recommendation_agent_id,
                timestamp=int(getattr(event, "timestamp", 0) or 0),
                timestamp_duration=int(getattr(event, "timestamp_duration", 0) or 0),
                current_step=int(getattr(event, "current_step", 1) or 1),
                max_step=int(getattr(event, "max_step", 8) or 8),
                user_profile=getattr(event, "user_profile", None) or {},
                current_posts=getattr(event, "current_posts", None) or {},
                recommended_post_ids=list(getattr(event, "recommended_post_ids", None) or []),
                algorithm_type=str(getattr(event, "type", "") or ""),
            )
        ]

    async def route_search(self, event: Event) -> List[Event]:
        if not self._search_agent_id:
            logger.warning("Platform: search Algorithm is not registered")
            return []
        return [
            GetSearchResultEvent(
                from_agent_id=event.from_agent_id,
                to_agent_id=self._search_agent_id,
                timestamp=int(getattr(event, "timestamp", 0) or 0),
                timestamp_duration=int(getattr(event, "timestamp_duration", 0) or 0),
                current_step=int(getattr(event, "current_step", 1) or 1),
                max_step=int(getattr(event, "max_step", 8) or 8),
                user_profile=getattr(event, "user_profile", None) or {},
                algorithm_type=str(getattr(event, "type", "") or ""),
                search_query=str(getattr(event, "search_query", "") or ""),
            )
        ]
