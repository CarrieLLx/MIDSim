"""Assemble task instructions; gates retain their existing selection logic."""
from .loader import format_prompt, load_policy


def normalize_tier(tier: str) -> str:
    value = str(tier or "low").strip().lower()
    return value if value in ("low", "mid", "high") else "low"


def build_supplement(tier: str, *, reply_tier: str = "low") -> str:
    return format_prompt(
        "tasks/supplement_own_posts.txt",
        supplement_policy=load_policy("supplement", normalize_tier(tier)).rstrip(),
        author_reply_policy=load_policy("author_reply", normalize_tier(reply_tier)).rstrip(),
        author_memory_coaching=load_policy("memory", "author", "coaching").rstrip(),
    )


def build_author_reply(tier: str, gates) -> str:
    policy = load_policy("author_reply", normalize_tier(tier)).rstrip()
    return format_prompt(
        "tasks/author_thread_reply.txt",
        author_reply_policy=policy,
        memory_coaching=gates.memory_coaching,
        memory_ref=gates.memory_ref,
        similarity_kw_coaching=gates.similarity_kw_coaching,
        similarity_emb_coaching=gates.similarity_emb_coaching,
    )



def with_official_policy(instruction: str, context: str) -> str:
    # Preserve the existing peer-mention policy without introducing a new behavior.
    section = "recommendation" if context == "mention" else context
    return load_policy("official", section).rstrip() + "\n\n" + instruction
