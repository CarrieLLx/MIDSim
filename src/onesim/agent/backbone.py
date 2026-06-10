"""Agent backbone profile (default / llama) for model-specific behavior."""
from __future__ import annotations

import json
import os
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from onesim.config import AgentConfig, SimulatorConfig

_PROFILE = "default"


def normalize_backbone_profile(profile: Optional[str]) -> str:
    if not profile:
        return "default"
    key = str(profile).strip().lower().replace("-", "_")
    if key in ("llama", "llama_zh", "llama_en", "llama31", "llama3"):
        return "llama"
    if key in ("default", "qwen", "standard"):
        return "default"
    return key


def set_backbone_profile(profile: Optional[str]) -> str:
    global _PROFILE
    _PROFILE = normalize_backbone_profile(profile)
    return _PROFILE


def get_backbone_profile() -> str:
    return _PROFILE


def is_llama_backbone() -> bool:
    return _PROFILE == "llama"


def _resolve_config_path(path: str, env_path: Optional[str] = None) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    cwd_path = os.path.join(os.getcwd(), path)
    if os.path.exists(cwd_path):
        return cwd_path
    if env_path:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(env_path)))
        root_path = os.path.join(project_root, path)
        if os.path.exists(root_path):
            return root_path
    return cwd_path


def resolve_backbone_profile(
    agent_config: Optional["AgentConfig"] = None,
    simulator_config: Optional["SimulatorConfig"] = None,
    env_path: Optional[str] = None,
) -> str:
    """Resolve backbone from config agent section, then midsim params file."""
    if agent_config is not None:
        profile = getattr(agent_config, "backbone_profile", None)
        if profile:
            return set_backbone_profile(profile)

    if simulator_config is not None:
        env_cfg = getattr(simulator_config, "environment", None) or {}
        additional = env_cfg.get("additional_config") or {}
        params_path = additional.get("params_path")
        if params_path:
            resolved = _resolve_config_path(params_path, env_path)
            try:
                with open(resolved, "r", encoding="utf-8") as f:
                    params = json.load(f)
                agent_section = params.get("agent") or {}
                profile = agent_section.get("backbone_profile")
                if profile:
                    return set_backbone_profile(profile)
            except (OSError, json.JSONDecodeError):
                pass

    return set_backbone_profile("default")


def resolve_planning_class(planning_config: str) -> Any:
    """Pick COT implementation based on backbone_profile when planning is COTPlanning."""
    if planning_config == "COTPlanning" and is_llama_backbone():
        from onesim.planning.cot_guarded import GuardedCOTPlanning

        return GuardedCOTPlanning

    import importlib

    planning_module = importlib.import_module("onesim.planning")
    return getattr(planning_module, planning_config)


def load_planning_instance(
    planning_config: Optional[str],
    model_config_name: str,
    sys_prompt: str,
):
    if not planning_config:
        return None
    from onesim.planning.base import PlanningBase

    PlanningClass = resolve_planning_class(planning_config)
    instance: PlanningBase = PlanningClass(model_config_name, sys_prompt)
    return instance
