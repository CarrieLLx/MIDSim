"""Unambiguous mention addresses; legacy strings remain readable."""
import json

def resolve_address(key, data, target_user_id=None, comment_id=None):
    action = data.get("action")
    if action not in ("add", "delete", "keep_only", "delete_many"):
        raise ValueError(f"Unsupported mention action: {action}")
    bulk = action in ("keep_only", "delete_many")
    if target_user_id is None:
        if not isinstance(key, str) or not key.startswith("mention_pool."):
            raise ValueError("Legacy mention key must start with mention_pool.")
        tail = key[len("mention_pool."):]
        if bulk:
            target_user_id = tail
        else:
            # Legacy add payload supplies the exact ID, even if it contains dots.
            known = (data.get("mention_message") or {}).get("comment_id")
            if known is not None and tail.endswith("." + str(known)):
                comment_id = str(known)
                target_user_id = tail[:-len(comment_id)-1]
            else:
                if "." not in tail:
                    raise ValueError("Legacy single mention requires a comment ID")
                target_user_id, comment_id = tail.rsplit(".", 1)
    if not isinstance(target_user_id, str) or not target_user_id:
        raise ValueError("Mention target_user_id must be a nonempty string")
    if bulk:
        if comment_id is not None:
            raise ValueError("Bulk mention operation cannot specify a comment ID")
    elif not isinstance(comment_id, str) or not comment_id:
        raise ValueError("Mention comment_id must be a nonempty string")
    return target_user_id, comment_id

def address_key(user_id, comment_id):
    return "mention_pool:" + json.dumps([user_id, comment_id], ensure_ascii=True, separators=(",", ":"))
