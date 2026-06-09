"""Localized prompt fragments for GeneralAgent (zh / en)."""
from __future__ import annotations

from typing import Optional


def normalize_prompt_locale(locale: Optional[str]) -> str:
    loc = (locale or "zh").strip().lower()
    if loc in ("en", "english", "general_agent_en"):
        return "en"
    return "zh"


def memory_locale_instructions(locale: str) -> str:
    if normalize_prompt_locale(locale) == "en":
        return """
        [MANDATORY MEMORY] Whether or not Reaction involved comments, replies, likes, or other interactions, you **must** output exactly one non-empty memory. **Even if** there was no interaction at all (e.g. all `comment` fields false), still complete one sentence using the pattern: state truthfully "scrolled past", "did not comment", "remained silent", etc., and you may summarize from Reaction one reason for not interacting (e.g. duplicate topic, no new information); **do not** omit memory due to no interaction, return an empty string, or use generic filler unrelated to this event.

        Prescribed sentence pattern (fill the bracketed slots with concrete details from this event; keep first person):
        Through [channel] I encountered [topic]; the core is [key fact]. I engaged in [interaction type] and expressed [attitude direction].

        Memory quality (general; aim to satisfy all):
        - For [key fact], fold related information on the same topic **into one sentence**, as if retelling a friend "what happened"; include proper names, times, and numbers when available.
        - If Observation contains extractable anchors, prefer them: proper names, times, numbers, rules/product names; avoid evaluative filler only (e.g. "serious", "hope it is resolved soon", "worth attention").
        - If this round had no comment or you explicitly chose silence, state plainly in [interaction type] or [attitude direction] (e.g. browsed without commenting / remained silent); do not fabricate a comment that did not happen; you must still output this memory.
        - Single sentence only; total length ideally ≤120 characters (or a similarly concise sentence in the language you use).
        """
    return """
        【记忆为必填】无论 Reaction 是否包含评论、回复、点赞等，你**必须**输出恰好一条非空的 memory 字符串。**即便**结果完全无互动（例如所有 comment=false），仍须按下方句式完成一句记忆，例如「划过去了」「未评论」「保持沉默」，并结合 Reaction 简要说明未参与的原因（如与此前话题重复、没有新信息等）。**不得**因无互动而跳过记忆、返回空串，或使用与本事件无关的泛泛套话。

        规定句式（方括号内填入本事件中的具体信息；第一人称）：
        通过 [渠道] 我了解到/刷到了 [话题]；核心是 [关键事实]。我 [互动方式]，并表达了 [态度倾向]。

        记忆质量（尽量全部满足）：
        - 将 [关键事实] 及同一话题的相关细节**压缩为一句**，像给朋友口述现状；若有姓名、时间、数字尽量写上。
        - 若 Observation 中有可抓手的信息，优先使用：人名、时间、数字、规则/产品名等；避免只有评价性空话（如「很严重」「希望平台…」「值得关注」）。
        - 若未评论或明确选择沉默，在 [互动方式] 或 [态度倾向] 中直白写出（如仅浏览未评论/保持沉默）；不要编造并未发生的评论。仍须输出该条记忆。
        - 仅一句；总长度宜 ≤120 字（中文或英文）。
        """


def memory_fallback_sentence(locale: str) -> str:
    if normalize_prompt_locale(locale) == "en":
        return (
            "Processed this event and completed the decision; "
            "the model did not return a valid memory field."
        )
    return "已处理本次事件并完成决策；模型未返回符合格式的 memory 字段。"


def mem_planning_gate(locale: str) -> str:
    if normalize_prompt_locale(locale) == "en":
        return (
            "[Planning self-check · Memory non-empty] If you intend comment=true and believe there are new facts relative to Memory: "
            "you must verify that the **novelty judgment in ### Planning** (the part corresponding to memory_reflection in the final JSON) holds—"
            "i.e. whether the judgment about new fact vs. duplicate of Memory is correct and consistent with the full ### Memory. "
            "Cross-check ### Planning: is it only repeating Memory, or full of empty phrases like 'serious / hope the platform acts / worth attention'? "
            "If Planning's novelty judgment is wrong, or it only paraphrases Memory, or is filled with such empty phrases, you must set comment=false and leave comment_content empty; "
            "memory_reflection in the final JSON must reflect that, after cross-checking Planning with Memory, there are no new hard facts, hence silence.\n\n"
        )
    return (
        "【Planning 自检 · Memory 非空】若你打算 comment=true，并认为相对 ### Memory 有新信息："
        "请核对 **### Planning 中的新颖性判断**（最终 JSON 里将对应 memory_reflection 的那部分）是否成立——"
        "即是否确为新信息、未与 Memory 重复、且不与完整 ### Memory 正文矛盾。"
        "检查 ### Planning：是否只是在复述 Memory，或充斥「很严重」「希望平台…」「值得关注」等空话？"
        "若 Planning 误判新颖性、仅重复 Memory，或充满此类空洞表述，你必须设 comment=false，且 comment_content 留空；"
        "最终 JSON 的 memory_reflection 须写明：对照 Memory 复核 Planning 后，并无新的硬事实，故选择沉默。\n\n"
    )


def profile_tags_system_prompt(locale: str, sys_prompt: Optional[str]) -> str:
    if sys_prompt:
        return sys_prompt
    if normalize_prompt_locale(locale) == "en":
        return "You are a professional user profiling assistant."
    return "You are a professional user-profile analysis assistant."


def build_profile_tags_prompt_text(
    *,
    locale: str,
    nickname: str,
    gender: str,
    description: str,
    location: str,
    notes_str: str,
    existing_tags_info: str,
    following_count: int,
    follower_count: int,
    interaction: int,
    append_mode: bool,
    existing_tags: list,
) -> str:
    append_hint = (
        f' (Existing tags: {", ".join(existing_tags)}; generate new tags and avoid duplicates.)'
        if append_mode and existing_tags
        else ""
    )
    if normalize_prompt_locale(locale) == "en":
        return f"""Generate interest tags from the following user information:

            User profile:
            - Nickname: {nickname}
            - Gender: {gender}
            - Bio / description: {description}
            - Location: {location}
            - Historical notes: {notes_str}{existing_tags_info}

            Social stats:
            - Following count: {following_count}
            - Follower count: {follower_count}
            - Interactions (likes and favorites): {interaction}

            Tasks:

            1. From nickname, gender, bio, location, and historical notes, produce 5–10 interest_tags.
            Tags should be short keywords reflecting interests and preferences; use the same primary language as in the user's historical posts where possible.{append_hint}

            Return JSON in this shape:
            ```json
            {{
                "interest_tags": ["tag1", "tag2", "tag3", ...]
            }}
            ```

            Requirements:
            - interest_tags is a list of 5–10 strings
            - Tags are concise keywords in Chinese or English as appropriate
            """
    return f"""Based on the following user information, generate interest tags (interest_tags).

            User information:
            - Nickname: {nickname}
            - Gender: {gender}
            - Bio / description: {description}
            - Location: {location}
            - Historical notes: {notes_str}{existing_tags_info}

            Social statistics:
            - Following count: {following_count}
            - Follower count: {follower_count}
            - Interactions (likes and saves): {interaction}

            Tasks:

            1. From nickname, gender, description, location, and historical notes, produce 5–10 interest tags (interest_tags).
            Tags should be short keywords reflecting interests and preferences; **use the same primary language** as the user's historical posts/notes when possible.{append_hint}

            Return JSON in this form:
            ```json
            {{
                "interest_tags": ["Tag1", "Tag2", "Tag3", ...]
            }}
            ```

            Requirements:
            - interest_tags is a list of 5–10 strings
            - Tags are concise keywords in Chinese or English as appropriate
            """
