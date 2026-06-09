"""GeneralAgent prompt locale (zh / en)."""
from __future__ import annotations

import json
import os
from typing import Any, Optional, TYPE_CHECKING

from .general_agent_prompts import normalize_prompt_locale

if TYPE_CHECKING:
    from onesim.config import AgentConfig, SimulatorConfig

_LOCALE = "zh"


def set_general_agent_locale(locale: Optional[str]) -> str:
    """Set active prompt locale for GeneralAgent instances."""
    global _LOCALE
    _LOCALE = normalize_prompt_locale(locale)
    return _LOCALE


def get_general_agent_locale() -> str:
    return _LOCALE


def get_general_agent_class():
    from .general_agent import GeneralAgent

    return GeneralAgent


def is_general_agent_instance(obj: Any) -> bool:
    from .general_agent import GeneralAgent

    return isinstance(obj, GeneralAgent)


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


def resolve_general_agent_locale(
    agent_config: Optional["AgentConfig"] = None,
    simulator_config: Optional["SimulatorConfig"] = None,
    env_path: Optional[str] = None,
) -> str:
    """Resolve locale from config agent section, then midsim params file."""
    if agent_config is not None:
        locale = getattr(agent_config, "general_agent_locale", None)
        if locale:
            return set_general_agent_locale(locale)

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
                locale = agent_section.get("general_agent_locale")
                if locale:
                    return set_general_agent_locale(locale)
            except (OSError, json.JSONDecodeError):
                pass

    return set_general_agent_locale("zh")
