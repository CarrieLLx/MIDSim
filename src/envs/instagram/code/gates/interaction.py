# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any



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
