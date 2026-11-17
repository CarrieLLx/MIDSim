"""X profile discovery: explicit LLM intent, bounded history and exact reply paths."""
import asyncio
import json
import math
from copy import deepcopy

from loguru import logger
from onesim.utils.reaction_failure import SkippedReaction
from .embedding import EmbeddingService, cosine_similarity
from .events import ProfileVisitEvent, ProfileVisitCompletedEvent
from .prompting.loader import load_prompt
from .utils.time import time_to_ms


def account_id(row):
    return str(row.get("user_id") or row.get("author_id") or row.get("user_uid") or "").strip()


def public_card(uid, profile):
    return {"id": uid, **{k: profile.get(k) for k in
            ("username", "description", "interest_tags") if profile.get(k)}}


def visible_people(posts):
    """Only authors actually shown in this interaction are valid discovery targets."""
    people = {}
    for pid, post in posts.items():
        for cid, row in [(None, post), *list((post.get("comments") or {}).items())]:
            uid = account_id(row)
            if uid:
                people.setdefault(uid, []).append({"post_id": str(pid),
                    "comment_id": str(cid) if cid is not None else None,
                    "content": row.get("content", "")})
    return people


def profile_text(profile):
    return json.dumps({k: profile.get(k, "") for k in
                      ("description", "interest_tags", "historical_summary")},
                     ensure_ascii=False, sort_keys=True)


def rank_followed_profiles(viewer, candidates, embed):
    """No sampling/lexical fallback: >5 follows must be ranked by embeddings."""
    ids = sorted(candidates)
    if len(ids) <= 5:
        return ids
    vectors = embed([profile_text(viewer)] + [profile_text(candidates[uid]) for uid in ids])
    if len(vectors) != len(ids) + 1 or not vectors[0]:
        raise ValueError("Follow ranking returned incomplete embeddings")
    dimension = len(vectors[0])
    if any(len(v) != dimension or not all(math.isfinite(x) for x in v)
           or not any(v) for v in vectors):
        raise ValueError("Follow ranking returned invalid embeddings")
    scores = {uid: cosine_similarity(vectors[0], v) for uid, v in zip(ids, vectors[1:])}
    return sorted(ids, key=lambda uid: (-scores[uid], uid))[:5]


def validate_following_review(response, candidate_ids):
    """Validate the dedicated following prompt before allowing any profile visits."""
    if not isinstance(response, dict) or set(response) != {"has_profile_visit", "profile_visits"}:
        raise ValueError("Expected exactly has_profile_visit and profile_visits")
    if type(response["has_profile_visit"]) is not bool:
        raise ValueError("has_profile_visit must be a JSON boolean")
    rows = response["profile_visits"]
    if not isinstance(rows, list) or len(rows) != len(candidate_ids):
        raise ValueError("Expected one decision row for every supplied account")
    fields = {"user_id", "account_interest", "view_profile", "reason"}
    for uid, row in zip(candidate_ids, rows):
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError("Each row must contain exactly user_id, account_interest, view_profile and reason")
        if not isinstance(row["user_id"], str) or row["user_id"] != uid:
            raise ValueError("Account IDs must match the input order without omissions or duplicates")
        if row["account_interest"] not in ("low", "moderate", "high", "very_high"):
            raise ValueError("Invalid account_interest")
        if type(row["view_profile"]) is not bool:
            raise ValueError("view_profile must be a JSON boolean")
        if not isinstance(row["reason"], str) or not row["reason"].strip():
            raise ValueError("Every account requires a non-empty reason, including declined visits")
        if len(row["reason"].split()) > 30:
            raise ValueError("Each reason must contain at most 30 words")
        if row["view_profile"] and row["account_interest"] != "very_high":
            raise ValueError("A visit requires account_interest=very_high")
    if response["has_profile_visit"] != any(row["view_profile"] for row in rows):
        raise ValueError("has_profile_visit must match the per-account visit decisions")


def history_snapshot(pool, uid, as_of_ms, limit=2):
    """Return newest visible authored replies and root -> ancestor -> reply paths."""
    if type(limit) is not int or limit < 0:
        raise ValueError("History comment limit must be a non-negative integer")
    items = []
    for pid, post in pool.items():
        stamp = time_to_ms(post.get("timestamp", post.get("time")))
        if stamp is None or stamp > as_of_ms or post.get("deleted"):
            continue
        comments = post.get("comments") or {}
        if not isinstance(comments, dict):
            continue
        visible = {str(cid): c for cid, c in comments.items() if isinstance(c, dict)
                   and not c.get("deleted") and not c.get("protected")
                   and (t := time_to_ms(c.get("timestamp", c.get("time")))) is not None
                   and t <= as_of_ms}
        for cid, comment in visible.items():
            if account_id(comment) != uid:
                continue
            chain, seen, missing = [], set(), None
            cursor = cid
            while cursor:
                if cursor in seen:
                    missing = {"comment_id": cursor, "reason": "cycle"}
                    break
                seen.add(cursor)
                row = visible.get(cursor)
                if row is None:
                    missing = {"comment_id": cursor, "reason": "missing_or_not_visible"}
                    break
                chain.append({**deepcopy(row), "comment_id": cursor, "post_id": str(pid)})
                cursor = str(row.get("parent_comment_id") or "")
            chain.reverse()
            root = {k: deepcopy(v) for k, v in post.items() if k != "comments"}
            root.update(post_id=str(pid), comments={r["comment_id"]: r for r in chain})
            root["comment_count"] = len(chain)
            items.append({"post_id": str(pid), "comment_id": cid, "post": root,
                          "path": [{"type": "post", "id": str(pid)}] +
                                  [{"type": "comment", "id": r["comment_id"]} for r in chain],
                          "missing_ancestor": missing,
                          "timestamp": time_to_ms(comment.get("timestamp", comment.get("time")))})
    items.sort(key=lambda r: (-r["timestamp"], r["post_id"], r["comment_id"]))
    return {"user_id": uid, "as_of_ms": as_of_ms, "total_visible": len(items),
            "limit": limit, "truncated": len(items) > limit, "items": items[:limit]}


class ProfileBrowsingMixin:
    def _init_profile_browsing(self):
        self.register_event("StartEvent", "review_followed_profiles")
        self.register_event("ProfileVisitEvent", "visit_profile")
        self._profile_visit_lock = asyncio.Lock()
        self._profile_visit_step = None
        self._profile_visited = set()
        self._follow_review_step = None

    def _profile_audit(self, event, **record):
        rows = list(self.profile.get_data("profile_visit_audit", []) or [])
        rows.append({"step": event.current_step, "event_id": getattr(event, "event_id", None), **record})
        self.profile.update_data("profile_visit_audit", rows)

    def _known_profile(self, uid):
        env = getattr(self, "env", None)
        for actor in ((getattr(env, "agents", None) or {}).get("UserAgent") or {}).values():
            if str(actor.profile.get_data("id")) == uid:
                return actor.profile.get_profile(include_private=True)
        rel = self.relationship_manager.get_relationship(uid) if self.relationship_manager else None
        return dict(rel.target_info) if rel and isinstance(rel.target_info, dict) else {"id": uid}

    @staticmethod
    def _with_profile_policy(instruction):
        return instruction + "\n\n" + load_prompt("tasks/profile_discovery.txt")

    async def _profile_events(self, response, posts, event, *, candidates=None, as_of_ms=None):
        evidence = visible_people(posts)
        allowed = set(candidates) if candidates is not None else set(evidence)
        own_id = str(self.profile.get_data("id"))
        allowed.discard(own_id)
        rows = response.get("profile_visits", [])
        events, accepted = [], []
        async with self._profile_visit_lock:
            if self._profile_visit_step != event.current_step:
                self._profile_visit_step = event.current_step
                self._profile_visited = set()
            for row in rows if isinstance(rows, list) else []:
                if not isinstance(row, dict):
                    continue
                uid = str(row.get("user_id") or "").strip()
                if uid not in allowed:
                    continue
                if candidates is None:
                    trigger = (str(row.get("trigger_post_id") or ""), row.get("trigger_comment_id"))
                    matches = [r for r in evidence[uid] if (r["post_id"], r["comment_id"]) == trigger]
                    grounded = bool(matches and str(row.get("reason") or "").strip())
                    strong = row.get("content_interest") == "very_high"
                else:
                    grounded = bool(str(row.get("reason") or "").strip())
                    strong = row.get("account_interest") == "very_high"
                # A follow is independently gated by sustained interest in the whole account.
                if grounded and row.get("follow") is True and row.get("account_interest") == "very_high":
                    self._follow_account(uid)
                if (response.get("has_profile_visit") is not True or row.get("view_profile") is not True
                        or not grounded or not strong or uid in self._profile_visited):
                    continue
                self._profile_visited.add(uid)
                accepted.append(uid)
                events.append(ProfileVisitEvent(
                    from_agent_id=self.profile_id, to_agent_id=self.profile_id,
                    timestamp=event.timestamp, timestamp_duration=getattr(event, "timestamp_duration", 0),
                    current_step=event.current_step, max_step=event.max_step,
                    target_user_id=uid, as_of_ms=as_of_ms if as_of_ms is not None else event.timestamp,
                    reason=row["reason"], source="following_review" if candidates is not None else "interaction",
                    trigger_post_id=row.get("trigger_post_id"), trigger_comment_id=row.get("trigger_comment_id")))
        self._profile_audit(event, action="profile_discovery", has_profile_visit=bool(accepted),
                            accepted_user_ids=accepted, decision=response.get("profile_visits", []))
        return events

    def _follow_account(self, uid):
        # Protected accounts require acceptance; this simulator has no request/accept flow.
        if self._known_profile(uid).get("protected"):
            return
        follows = list(self.profile.get_data("follow_ids", []) or [])
        if uid in follows:
            return
        follows.append(uid)
        self.profile.update_data("follow_ids", follows)
        self.profile.update_data("follow_count", len(follows))
        if self.relationship_manager:
            self.relationship_manager.add_relationship(uid, "following", public_card(uid, self._known_profile(uid)))
        env = getattr(self, "env", None)
        for actor in ((getattr(env, "agents", None) or {}).get("UserAgent") or {}).values():
            if str(actor.profile.get_data("id")) == uid:
                fans = list(actor.profile.get_data("fan_ids", []) or [])
                own = str(self.profile.get_data("id"))
                if own not in fans:
                    actor.profile.update_data("fan_ids", fans + [own])
                    actor.profile.update_data("fan_count", len(fans) + 1)
                if actor.relationship_manager:
                    actor.relationship_manager.add_relationship(own, "follower", public_card(own, self.profile.get_profile()))

    async def review_followed_profiles(self, event):
        if not await self._should_activate_this_round(event.current_step, event.max_step):
            return []
        async with self._profile_visit_lock:
            if self._follow_review_step == event.current_step:
                return []
            self._follow_review_step = event.current_step
        ids = sorted({str(uid) for uid in (self.profile.get_data("follow_ids", []) or [])
                      if str(uid) != str(self.profile.get_data("id"))})
        profiles = {uid: self._known_profile(uid) for uid in ids}
        try:
            selected = await asyncio.to_thread(rank_followed_profiles,
                self.profile.get_profile(include_private=True), profiles, EmbeddingService().embed)
        except Exception as exc:
            # Do not silently replace the requested embedding ranking with arbitrary users.
            logger.error("X following profile ranking unavailable: {}", exc)
            self._profile_audit(event, action="following_review", status="embedding_unavailable",
                                has_profile_visit=False, error=type(exc).__name__)
            return []
        if not selected:
            self._profile_audit(event, action="following_review", has_profile_visit=False, candidates=[])
            return []
        observation = json.dumps({"followed_accounts": [public_card(uid, profiles[uid]) for uid in selected]}, ensure_ascii=False)
        instruction = load_prompt("tasks/following_profile_review.txt")
        for attempt in range(2):
            response = self._normalize_llm_reaction(await self.generate_reaction(instruction, observation))
            if isinstance(response, SkippedReaction):
                self._profile_audit(event, action="following_review", status="skipped_error",
                                    has_profile_visit=False, candidates=selected)
                return []
            try:
                validate_following_review(response, selected)
                break
            except ValueError as exc:
                self._profile_audit(event, action="following_review", has_profile_visit=False,
                    status="retrying_llm_review" if attempt == 0 else "invalid_llm_review",
                    validation_error=str(exc))
                if attempt == 1:
                    return []
                logger.warning("X following review has invalid output; retrying: {}", exc)
        self._profile_audit(event, action="following_review", candidates=selected,
                            total_followed=len(ids), method="embedding_top5" if len(ids) > 5 else "all")
        return await self._profile_events(response, {}, event, candidates=selected, as_of_ms=event.timestamp)

    async def visit_profile(self, event):
        env = getattr(self, "env", None)
        snapshot = await env.snapshot_profile_history(event.target_user_id, event.as_of_ms,
            str(self.profile.get_data("id")), self.profile.get_data("follow_ids", []))
        self._profile_audit(event, action="profile_opened", target_user_id=event.target_user_id,
                            snapshot=snapshot)
        completed = ProfileVisitCompletedEvent(from_agent_id=self.profile_id, to_agent_id="ENV",
            timestamp=event.timestamp, timestamp_duration=event.timestamp_duration,
            current_step=event.current_step, max_step=event.max_step)
        if not snapshot["items"]:
            return [completed]
        events = []
        from .gates import UserAgentGates
        # The snapshot already applies the per-visit limit; preserve its complete paths.
        for offset in range(0, len(snapshot["items"]), 5):
            items = snapshot["items"][offset:offset + 5]
            posts = {}
            for item in items:
                pid = item["post_id"]
                if pid not in posts:
                    posts[pid] = deepcopy(item["post"])
                else:
                    posts[pid]["comments"].update(item["post"]["comments"])
            from .prompting.loader import format_prompt
            from .prompting.builder import with_official_policy
            gates = await UserAgentGates(self).build_profile_history_coaching(posts, event.as_of_ms)
            instruction = format_prompt("tasks/profile_history_context.txt",
                memory_coaching=gates.memory_coaching, memory_ref=gates.memory_ref,
                similarity_kw_coaching=gates.similarity_kw_coaching,
                similarity_emb_coaching=gates.similarity_emb_coaching)
            if await self._is_official_by_agent_field():
                instruction = with_official_policy(instruction, "profile_history")
            observation = json.dumps({"source": "profile_replies", "user_id": event.target_user_id,
                                      "as_of_ms": event.as_of_ms, "history": items}, ensure_ascii=False)
            response = self._normalize_llm_reaction(await self.generate_reaction(self._with_profile_policy(instruction), observation))
            events.extend(await self._profile_events(response, posts, event, as_of_ms=event.as_of_ms))
            search = self._build_search_event(response, event, event.current_step, event.max_step)
            if search:
                events.append(search)
            bookmarks = self._build_bookmark_post_ids(response, posts)
            if bookmarks:
                await env.retain_profile_posts(bookmarks)
                self.profile.update_data("bookmark_post_ids", list(dict.fromkeys(
                    list(self.profile.get_data("bookmark_post_ids", []) or []) + bookmarks)))
            mentionable = self._get_mentionable_users(self.profile.get_data("follow_ids", []), self.profile.get_data("fan_ids", []))
            for decision in response.get("decisions", []) if isinstance(response.get("decisions"), list) else []:
                if not isinstance(decision, dict) or decision.get("comment") is not True or not str(decision.get("comment_content") or "").strip():
                    continue
                pid = str(decision.get("post_id") or "")
                parent = decision.get("parent_comment_id")
                if pid not in posts or (parent is not None and str(parent) not in posts[pid]["comments"]):
                    continue
                end = int(event.timestamp) + max(1, int(event.timestamp_duration))
                await self._add_comment_and_notify(decision, posts[pid], self._mentionable_id_set(mentionable),
                    str(self.profile.get_data("id")), self.profile.get_data("username", ""),
                    event.as_of_ms, max(1, end - event.as_of_ms), event.current_step, event.max_step)
        events.append(completed)
        return events
