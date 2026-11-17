# -*- coding: utf-8 -*-
from typing import Any, Collection, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import hashlib
import asyncio
import json
import os
import random

from loguru import logger
from onesim.utils.reaction_failure import SkippedRecommendation
from onesim.utils.midsim_params import MIDSimParams

from ..embedding import EmbeddingService, cosine_similarity, default_embedding_config_path
from ..utils.prompt import load_prompt
from ..utils.text import format_historical_summary, format_popularity_distribution

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def to_float(raw: Any, *, default: float) -> float:
    """Parse profile/config scalar to float; empty or invalid values use *default*."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


class RecommendationEngine:
    _RANDOM_REC_MAU_DEFAULT = 240_000_000
    _RANDOM_REC_AVG_POSTS_PER_USER_MONTH_DEFAULT = 1.1182

    def __init__(self, agent: Any) -> None:
        self._agent = agent
        self._coarse_rank_lock = asyncio.Lock()
        self._popularity_distribution_cache: Optional[Dict[int, int]] = None
        self._embedder: Optional[EmbeddingService] = None
        self._post_vec_cache: Dict[str, Tuple[str, List[float]]] = {}

    @property
    def profile(self):
        return getattr(self._agent, "profile", None)

    def reset_popularity_cache(self) -> None:
        self._popularity_distribution_cache = None

    def _get_embedder(self) -> EmbeddingService:
        if self._embedder is None:
            path = ""
            if self.profile is not None:
                path = str(self.profile.get_data("embedding_config_path", "") or "").strip()
            self._embedder = EmbeddingService(path or default_embedding_config_path())
        return self._embedder

    @staticmethod
    def _profile_interest_text(user_profile: Optional[Dict[str, Any]]) -> str:
        if not user_profile or not isinstance(user_profile, dict):
            return ""
        tags = user_profile.get("interest_tags") or []
        if isinstance(tags, list):
            tag_s = ", ".join(str(x).strip() for x in tags if str(x).strip())
        else:
            tag_s = str(tags).strip()
        desc = " ".join(str(user_profile.get("description") or "").split())
        summary = " ".join(str(user_profile.get("historical_summary") or "").split())
        if len(summary) > 1500:
            summary = summary[:1500]
        parts = [p for p in (tag_s, desc, summary) if p]
        if parts:
            return "\n".join(parts)
        return str(user_profile.get("username") or user_profile.get("nickname") or user_profile.get("id") or "").strip()

    @staticmethod
    def _post_desc(post: Optional[Dict[str, Any]]) -> str:
        if not isinstance(post, dict):
            return ""
        return " ".join(str(post.get("desc") or post.get("content") or post.get("text") or "").split())

    @staticmethod
    def _text_digest(kind: str, text: str) -> str:
        return hashlib.sha256(f"{kind}:{text}".encode("utf-8")).hexdigest()[:16]

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        out: List[List[float]] = []
        embedder = self._get_embedder()
        batch_size = 32
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vecs = embedder.embed(batch)
            if len(vecs) != len(batch):
                raise ValueError(f"embedding count mismatch: {len(vecs)} vs {len(batch)}")
            out.extend(vecs)
        return out

    def _ensure_post_vectors(
        self,
        posts: Dict[str, Dict[str, Any]],
        post_ids: List[str],
    ) -> Dict[str, List[float]]:
        missing_ids: List[str] = []
        missing_texts: List[str] = []
        result: Dict[str, List[float]] = {}
        for pid in post_ids:
            post = posts.get(pid)
            text = self._post_desc(post) or pid
            digest = self._text_digest("p", text)
            cached = self._post_vec_cache.get(pid)
            if cached and cached[0] == digest:
                result[pid] = cached[1]
                continue
            missing_ids.append(pid)
            missing_texts.append(text)
        if missing_texts:
            vecs = self._embed_texts(missing_texts)
            for pid, vec in zip(missing_ids, vecs):
                text = self._post_desc(posts.get(pid)) or pid
                digest = self._text_digest("p", text)
                self._post_vec_cache[pid] = (digest, vec)
                result[pid] = vec
        return result

    async def _coarse_rank_from_pool_async(self, **kwargs) -> List[str]:
        # One coarse-rank worker per engine protects the shared vector cache.
        # Waiting for embedding no longer blocks LLM replies or event delivery.
        async with self._coarse_rank_lock:
            worker = asyncio.create_task(asyncio.to_thread(self._coarse_rank_from_pool, **kwargs))
            try:
                return await asyncio.shield(worker)
            except asyncio.CancelledError:
                # Threads cannot be cancelled. Keep the cache lock until this
                # worker finishes, then propagate cancellation to the caller.
                try:
                    await worker
                except Exception:
                    pass
                raise

    def _coarse_rank_from_pool(
        self,
        *,
        user_profile: Optional[Dict[str, Any]],
        interest_pool: Dict[str, Dict[str, Any]],
        exclude: Set[str],
        coarse_k: int,
    ) -> List[str]:
        candidate_ids = [
            pid for pid, post in interest_pool.items()
            if pid not in exclude and isinstance(post, dict)
        ]
        if not candidate_ids or coarse_k <= 0:
            return []

        query_text = self._profile_interest_text(user_profile)
        ranked: List[str] = []
        try:
            if query_text:
                user_vecs = self._embed_texts([BGE_QUERY_INSTRUCTION + query_text])
                user_vec = user_vecs[0] if user_vecs else []
                post_vecs = self._ensure_post_vectors(interest_pool, candidate_ids)
                scored: List[Tuple[str, float]] = []
                for pid in candidate_ids:
                    vec = post_vecs.get(pid)
                    if not vec or not user_vec:
                        scored.append((pid, 0.0))
                    else:
                        scored.append((pid, float(cosine_similarity(user_vec, vec))))
                scored.sort(key=lambda x: x[1], reverse=True)
                ranked = [pid for pid, _score in scored[:coarse_k]]
                if scored:
                    logger.info(
                        "interest coarse-rank: keep={} / {} top_sim={:.3f}",
                        len(ranked),
                        len(candidate_ids),
                        scored[0][1],
                    )
            else:
                logger.warning("interest coarse-rank: empty user profile text, sampling pool")
        except Exception as exc:
            logger.warning(f"interest coarse-rank embedding failed, sampling pool: {exc}")

        if not ranked:
            shuffled = list(candidate_ids)
            random.shuffle(shuffled)
            ranked = shuffled[:coarse_k]
        return ranked

    def _random_rec_mau_avg_posts(self) -> Tuple[float, float]:
        mau = to_float(
            self.profile.get_data("random_rec_mau") if self.profile else None,
            default=self._RANDOM_REC_MAU_DEFAULT,
        )
        avg_posts = to_float(
            self.profile.get_data("random_rec_avg_posts_per_user_month") if self.profile else None,
            default=self._RANDOM_REC_AVG_POSTS_PER_USER_MONTH_DEFAULT,
        )
        if mau <= 0:
            mau = float(self._RANDOM_REC_MAU_DEFAULT)
        if avg_posts <= 0:
            avg_posts = float(self._RANDOM_REC_AVG_POSTS_PER_USER_MONTH_DEFAULT)
        return mau, avg_posts

    async def random(
        self,
        recommended_post_ids: Collection[str],
        contents: Dict[str, Dict[str, Any]],
        limit: int = 10,
        *,
        current_timestamp: Optional[float] = None,
        candidate_post_ids: Optional[List[str]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        if not contents:
            return {}
        post_ids = list(contents.keys())
        if not post_ids:
            return {}

        post_ids = [post_id for post_id in post_ids if post_id not in recommended_post_ids]
        if not post_ids:
            return {}

        mau, avg_posts = self._random_rec_mau_avg_posts()
        denom = mau * avg_posts
        if denom <= 0:
            p_hit = 0.0
        else:
            p_hit = min(1.0, 1.0 / denom)

        random.shuffle(post_ids)
        log_each = bool(
            self.profile is not None
            and self.profile.get_data("random_rec_log_per_post_prob_check", False)
        )

        selected: List[str] = []
        for nid in post_ids:
            if len(selected) >= limit:
                break
            ok = random.random() < p_hit
            if log_each:
                logger.debug(
                    "random_rec post {}: Bernoulli(p=1/(MAU×avg)={:.6e}) => {}",
                    nid, p_hit, "hit" if ok else "miss",
                )
            if ok:
                selected.append(nid)

        w = (avg_posts / mau) if mau > 0 else 0.0
        logger.debug(
            "random recommendation: MAU={:.4g} avg_posts={:.6g} => p=1/(MAU×avg)={:.6e}; weight=(1/MAU)×avg={:.6e}",
            mau, avg_posts, p_hit, len(post_ids), limit, len(selected), w,
        )
        return {post_id: contents[post_id] for post_id in selected}

    def _get_popularity_distribution_map(self, rows: List[Dict[str, Any]]) -> Dict[int, int]:
        if self._popularity_distribution_cache is not None:
            return self._popularity_distribution_cache

        self._popularity_distribution_cache = format_popularity_distribution(rows)
        return self._popularity_distribution_cache

    @staticmethod
    def _calculate_popularity(post: Dict[str, Any]) -> float:
        raw_popularity = post.get("popularity")
        if raw_popularity is not None:
            try:
                return max(0.0, float(raw_popularity))
            except (TypeError, ValueError):
                pass
        try:
            return max(0.0, float(post.get("comment_count", 0) or 0))
        except (TypeError, ValueError):
            return 0.0

    async def popularity(
        self,
        recommended_post_ids: Collection[str],
        contents: Dict[str, Dict[str, Any]],
        limit: int = 10,
        *,
        current_timestamp: Optional[float] = None,
        candidate_post_ids: Optional[List[str]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        if not contents:
            return {}
        post_ids = list(contents.keys())
        if not post_ids:
            return {}

        post_ids = [post_id for post_id in post_ids if post_id not in recommended_post_ids]
        if not post_ids:
            return {}

        params = await MIDSimParams.load(self._agent)
        dist_map = self._get_popularity_distribution_map(params.recommendation.popularity_distribution)
        if not dist_map:
            posts_with_popularity: List[Tuple[str, Dict[str, Any], float]] = []
            for post_id in post_ids:
                post = contents[post_id]
                popularity = self._calculate_popularity(post)
                posts_with_popularity.append((post_id, post, popularity))
            random.shuffle(posts_with_popularity)
            posts_with_popularity.sort(key=lambda x: x[2], reverse=True)
            return {post_id: post for post_id, post, _ in posts_with_popularity[:limit]}

        buckets: Dict[int, List[Optional[str]]] = defaultdict(list)
        for cc, cnt in dist_map.items():
            try:
                n = int(cnt)
            except (TypeError, ValueError):
                continue
            for _ in range(max(0, n)):
                buckets[cc].append(None)
        for post_id in post_ids:
            p = int(self._calculate_popularity(contents[post_id]))
            buckets[p].append(post_id)

        merged: List[Optional[str]] = []
        for cc in sorted(buckets.keys(), reverse=True):
            row = buckets[cc]
            random.shuffle(row)
            merged.extend(row)

        top_slice = merged[: max(0, limit)]
        chosen_ids = [x for x in top_slice if x is not None]
        return {nid: contents[nid] for nid in chosen_ids if nid in contents}

    @staticmethod
    def _interest_interleave_pool_for_prompt(pool_ids: List[str], pool_set: Set[str]) -> List[str]:
        ordered = [str(x).strip() for x in pool_ids if x]
        a = [x for x in ordered if x in pool_set]
        b = [x for x in ordered if x not in pool_set]
        if not b:
            return a
        if not a:
            return b
        out: List[str] = []
        i, j = 0, 0
        while i < len(a) or j < len(b):
            if i < len(a):
                out.append(a[i])
                i += 1
            if j < len(b):
                out.append(b[j])
                j += 1
        return out

    @staticmethod
    def _compact_user_profile_for_interest_rec(
        user_profile: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not user_profile or not isinstance(user_profile, dict):
            return {}
        keys = ("id", "username", "gender", "description", "location", "interest_tags", "historical_summary")
        out: Dict[str, Any] = {}
        for k in keys:
            if k not in user_profile:
                continue
            v = user_profile[k]
            if k == "historical_summary":
                v = format_historical_summary(v)
            elif k == "description" and isinstance(v, str) and len(v) > 400:
                v = v[:400] + "…"
            out[k] = v
        return out


    async def _llm_rank_interest_candidates(
        self,
        *,
        pool_ids: List[str],
        pool_set: Set[str],
        user_profile: Optional[Dict[str, Any]],
        recommended_id_set: Set[str],
        resolve_post,
    ) -> Optional[List[str]]:
        if not pool_ids:
            return []
        if os.environ.get("ONESIM_REC_INTERLEAVE_INPUT", "1").strip().lower() not in ("0", "false", "no", "off"):
            pool_ids = self._interest_interleave_pool_for_prompt(list(pool_ids), pool_set)

        def _make_candidate_payload(
            ids: List[str], desc_max: int
        ) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for post_id in ids:
                post = resolve_post(post_id)
                if not isinstance(post, dict):
                    continue
                text = str(post.get("content") or post.get("desc") or post.get("text") or "")
                out.append({
                    "post_id": post_id,
                    "content": text[:desc_max],
                })
            return out

        max_prompt_chars = int(
            os.environ.get("ONESIM_RECOMMENDER_INTEREST_MAX_PROMPT_CHARS", "48000")
        )
        profile_compact = self._compact_user_profile_for_interest_rec(user_profile)
        if profile_compact:
            user_profile_prefix = (
                "User profile for interest ranking. Prefer aligning with the themes and intent in "
                "interest_tags, description, and historical_summary:\n"
                f"{json.dumps(profile_compact, ensure_ascii=False)}\n\n"
            )
        else:
            user_profile_prefix = ""

        instruction = load_prompt("algorithm/interest_recommendation.txt")
        desc_max = 800
        candidate_payload: List[Dict[str, Any]] = []
        observation = ""
        total_len = 0
        for _ in range(512):
            candidate_payload = _make_candidate_payload(pool_ids, desc_max)
            pool_ids_json = json.dumps(pool_ids, ensure_ascii=False)
            observation = (
                user_profile_prefix
                + f"Full candidate post_id set (rank only inside this set): {pool_ids_json}\n\n"
                + f"Candidate post details:\n{json.dumps(candidate_payload, ensure_ascii=False)}"
            )
            total_len = len(instruction) + len(observation)
            if total_len <= max_prompt_chars:
                break
            if len(pool_ids) > 1:
                drop_at = None
                for idx in range(len(pool_ids) - 1, -1, -1):
                    pid = pool_ids[idx]
                    if pid not in pool_set:
                        drop_at = idx
                        break
                if drop_at is None:
                    drop_at = len(pool_ids) - 1
                pool_ids.pop(drop_at)
                logger.warning(
                    f"Interest recommendation prompt over budget ({total_len} > {max_prompt_chars} chars): "
                    f"dropped a candidate, {len(pool_ids)} left"
                )
                continue
            if desc_max > 200:
                desc_max = max(200, desc_max // 2)
                logger.warning(f"Interest recommendation prompt over budget: shrinking desc cap to {desc_max} chars")
                continue
            if desc_max > 80:
                desc_max = 80
                logger.warning("Interest recommendation prompt over budget: shrinking desc cap to 80 chars")
                continue
            logger.error(
                f"Interest recommendation: prompt still {total_len} chars (limit {max_prompt_chars}); proceeding anyway"
            )
            break

        ordered_ids: List[str] = []
        pool_set_all: Set[str] = set(pool_ids)
        try:
            response = await self._agent.generate_recommendation(instruction, observation)
            if isinstance(response, SkippedRecommendation):
                return None
            raw_ids = response.get("ranked_post_ids")
            if raw_ids is None:
                raw_ids = response.get("selected_post_ids", [])
            if isinstance(raw_ids, list):
                for post_id in raw_ids:
                    post_id = str(post_id).strip()
                    if (
                        post_id
                        and post_id in pool_set_all
                        and resolve_post(post_id) is not None
                        and post_id not in recommended_id_set
                        and post_id not in ordered_ids
                    ):
                        ordered_ids.append(post_id)
        except Exception as e:
            logger.warning(f"LLM interest recommendation failed: {e}")
            return None
        return ordered_ids

    async def interest(
        self,
        recommended_post_ids: Collection[str],
        contents: Dict[str, Dict[str, Any]],
        limit: int = 10,
        *,
        current_timestamp: Optional[float] = None,
        candidate_post_ids: Optional[List[str]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        contents = contents if isinstance(contents, dict) else {}
        contents_by_id = {
            str(post_id).strip(): post
            for post_id, post in contents.items()
            if (
                post_id is not None
                and str(post_id).strip()
                and isinstance(post, dict)
            )
        }
        recommended_id_set = {
            str(post_id).strip()
            for post_id in recommended_post_ids
            if post_id is not None and str(post_id).strip()
        }

        interest_pool: Dict[str, Dict[str, Any]] = {}
        if self.profile is not None:
            raw_ip = self.profile.get_data("interest_recommendation_content_pool", None)
            if isinstance(raw_ip, dict):
                interest_pool = {
                    str(post_id).strip(): post
                    for post_id, post in raw_ip.items()
                    if (
                        post_id is not None
                        and str(post_id).strip()
                        and isinstance(post, dict)
                    )
                }

        working_pool: Dict[str, Dict[str, Any]] = dict(interest_pool)
        remaining_window: Dict[str, Dict[str, Any]] = dict(contents_by_id)
        interest_hits: Dict[str, int] = {}
        exclude: Set[str] = set(recommended_id_set)
        delivered: Dict[str, Dict[str, Any]] = {}

        def _resolve_post(nid: str) -> Optional[Dict[str, Any]]:
            sid = str(nid).strip()
            post = working_pool.get(sid)
            if post is not None:
                return post
            post = remaining_window.get(sid)
            if post is not None:
                return post
            post = interest_pool.get(sid)
            if post is not None:
                return post
            return contents_by_id.get(sid)

        params = await MIDSimParams.load(self._agent)
        coarse_k = max(0, int(params.recommendation.coarse_k))
        current_posts_k = max(0, int(params.recommendation.current_posts_k))
        interest_hit_limit = 0
        group_count = max(0, int(limit))
        posts_per_group = params.recommendation.posts_per_group

        for group_i in range(group_count):
            if not remaining_window:
                logger.info(
                    "interest groups: stop early at {}/{} because current_posts is empty",
                    group_i,
                    group_count,
                )
                break

            pool_sample: List[str] = []
            if working_pool:
                pool_sample = await self._coarse_rank_from_pool_async(
                    user_profile=user_profile,
                    interest_pool=working_pool,
                    exclude=exclude,
                    coarse_k=coarse_k,
                )
            pool_sample = [nid for nid in pool_sample if _resolve_post(nid) is not None]
            pool_set: Set[str] = set(pool_sample)

            feed_eligible = [
                post_id for post_id in remaining_window
                if post_id not in exclude and post_id not in pool_set
            ]
            random.shuffle(feed_eligible)
            feed_sample: List[str] = feed_eligible[:current_posts_k]

            pool_ids: List[str] = list(pool_sample) + feed_sample
            if not pool_ids:
                logger.info(
                    "interest groups: stop early at {}/{} because ranking pool is empty",
                    group_i,
                    group_count,
                )
                break

            ordered_ids = await self._llm_rank_interest_candidates(
                pool_ids=pool_ids,
                pool_set=pool_set,
                user_profile=user_profile,
                recommended_id_set=exclude,
                resolve_post=_resolve_post,
            )
            if ordered_ids is None:
                logger.warning(
                    "interest group {}/{}: ranking unavailable; no selection was made",
                    group_i, group_count,
                )
                continue
            top_ids = ordered_ids[:posts_per_group]
            kept_env = 0
            evicted = 0
            for sid in top_ids:
                in_interest = sid in working_pool or sid in interest_pool
                env_post = remaining_window.pop(sid, None)
                if env_post is not None:
                    if sid not in delivered:
                        delivered[sid] = env_post
                    exclude.add(sid)
                    working_pool.pop(sid, None)
                    kept_env += 1
                if in_interest:
                    interest_hits[sid] = interest_hits.get(sid, 0) + 1
                    if interest_hits[sid] > interest_hit_limit:
                        if working_pool.pop(sid, None) is not None:
                            evicted += 1
            logger.info(
                "interest group {}/{}: coarse={} window_mix={} fine_top={} kept_env={} "
                "evicted={} remaining_window={}",
                group_i + 1,
                group_count,
                len(pool_sample),
                len(feed_sample),
                len(top_ids),
                kept_env,
                evicted,
                len(remaining_window),
            )

        return delivered
