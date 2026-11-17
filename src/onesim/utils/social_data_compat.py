"""Read canonical social datasets with legacy runtime field aliases in memory.

The files themselves only use account IDs and display usernames. Facebook keeps
its legacy nickname alias; Instagram reads canonical profiles directly.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path


def profile_for_runtime(row, source_path):
    platform = next((p.name for p in Path(source_path).parents
                     if p.name in {"facebook", "instagram"} and p.parent.name == "envs"), None)
    if platform != "facebook" or "nickname" in row or "historical_posts" not in row:
        return row
    result = deepcopy(row)
    result["nickname"] = row.get("username", "")
    return result


def add_runtime_post_aliases(pool, *, canonical_names=False):
    """Add post/time aliases; canonical_names keeps Instagram identity fields intact."""
    if not isinstance(pool, dict):
        return
    for post in pool.values():
        if not isinstance(post, dict) or "id" not in post or "post_id" in post:
            continue
        post["post_id"] = post["id"]
        post.setdefault("timestamp", post.get("time", 0))
        if canonical_names:
            continue
        post.setdefault("nickname", post.get("username", ""))
        post.setdefault("user_uid", post.get("user_id", ""))
        post.setdefault("user_aname", post.get("user_id", ""))
        post.setdefault("user_sname", post.get("username", ""))
        for comment in post.get("comments", {}).values():
            comment.setdefault("nickname", comment.get("username", ""))
