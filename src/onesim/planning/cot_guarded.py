"""Guarded COT for Llama backbone — cot prompt matches llama-en/zh forks by locale."""
import os
import re

from loguru import logger
from onesim.agent.locale import get_general_agent_locale
from onesim.models.core.message import Message
from onesim.planning.base import PlanningBase

# Short internal note before Reaction JSON; long or regurgitated Instruction hurts downstream parse.
_PLANNING_MAX_CHARS = int(os.environ.get("ONESIM_PLANNING_MAX_CHARS", "3250"))
_PLANNING_INSTRUCTION_MAX = int(os.environ.get("ONESIM_PLANNING_INSTRUCTION_MAX", "6500"))
_COT_MAX_USER_CHARS = int(os.environ.get("ONESIM_COT_MAX_USER_CHARS", "32500"))
_COT_MAX_REASONING_STEPS = int(os.environ.get("ONESIM_COT_MAX_REASONING_STEPS", "13"))


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 40)] + "\n...[truncated for planning only]"


def _clip_user_prompt(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    suffix = "\n\n[COT input clipped — ONESIM_COT_MAX_USER_CHARS]\n"
    cap = max(0, max_chars - len(suffix))
    return text[:cap] + suffix


def _count_reasoning_steps(text: str) -> int:
    if not text:
        return 0
    patterns = (
        r"(?im)^\s*(?:step\s*)?\d+[\.\):\-]\s+",
        r"(?im)^\s*###\s*step\s*\d+",
        r"(?im)^\s*步骤\s*\d+",
    )
    hits: set[int] = set()
    for pat in patterns:
        for m in re.finditer(pat, text):
            hits.add(m.start())
    return len(hits)


def _is_degenerate_planning(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if len(t) > _PLANNING_MAX_CHARS * 3:
        return True
    if _count_reasoning_steps(t) > _COT_MAX_REASONING_STEPS:
        return True
    if t.count("### 步骤") >= 8:
        return True
    if t.count("```json") >= 2:
        return True
    if re.search(r"步骤\d{2,}", t):
        return True
    if re.search(r"(?i)step\s*\d{2,}", t):
        return True
    if t.count('"comment"') >= 6 and t.count("note_id") >= 4:
        return True
    return False


def _fallback_planning() -> str:
    return (
        "Planning unavailable. Use Memory and Observation in the reaction step; "
        "apply Instruction there without repeating its wording."
    )


def _sanitize_planning(raw: str) -> str:
    t = (raw or "").strip()
    if not t or _is_degenerate_planning(t):
        if t:
            logger.warning("COTPlanning: degenerate output; using fallback")
        return _fallback_planning()
    if len(t) > _PLANNING_MAX_CHARS:
        t = t[:_PLANNING_MAX_CHARS].rstrip() + "\n...[planning truncated]"
    return t


class GuardedCOTPlanning(PlanningBase):
    def __init__(self, model_config_name, sys_prompt):
        super().__init__(model_config_name, sys_prompt)

    async def plan(self, **kwargs) -> str:
        profile = kwargs.get("profile") or ""
        memory = kwargs.get("memory") or ""
        observation = kwargs.get("observation") or ""
        instruction = _truncate(kwargs.get("instruction") or "", _PLANNING_INSTRUCTION_MAX)

        # llama-en cot: opens with Ground...; llama-zh cot: leading role sentence first.
        if get_general_agent_locale() == "en":
            prompt_lead = "\n\n"
        else:
            prompt_lead = (
                "You are preparing a short internal note before the agent's next reaction.\n\n"
            )

        prompt = f"""{prompt_lead}Ground your note in Profile, Memory, and Observation. The Instruction block is background only: do not quote it, do not walk through its numbered steps.

### Agent Profile
{profile}

### Memory
{memory}

### Observation
{observation}

### Instruction
{instruction}

You may think step by step, but use at most {_COT_MAX_REASONING_STEPS} numbered steps and stay under {_PLANNING_MAX_CHARS} characters. Stop once the note is sufficient; the structured response comes later."""

        if _COT_MAX_USER_CHARS > 0:
            before = len(prompt)
            prompt = _clip_user_prompt(prompt, _COT_MAX_USER_CHARS)
            if before > len(prompt):
                logger.warning(
                    f"COT input clipped {before} -> {len(prompt)} chars "
                    f"(ONESIM_COT_MAX_USER_CHARS={_COT_MAX_USER_CHARS})"
                )

        prompt = self.model.format(
            Message("system", self.sys_prompt, role="system"),
            Message("user", prompt, role="user"),
        )
        logger.info(
            f"COTPlanning plan prompt entry (model={getattr(self.model, 'config_name', '?')})"
        )
        response = await self.model.acall(prompt)
        logger.info(
            f"COTPlanning plan response exit (model={getattr(self.model, 'config_name', '?')})"
        )
        return response.text
