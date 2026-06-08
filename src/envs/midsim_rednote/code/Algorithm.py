from typing import Any, List, Optional, Callable, Dict, Tuple, Set
from collections import defaultdict
import json
import os
import asyncio
import random
from loguru import logger
from onesim.models import JsonBlockParser
from onesim.agent import GeneralAgent
from onesim.profile import AgentProfile
from onesim.memory import MemoryStrategy
from onesim.planning import PlanningBase
from onesim.events import Event
from onesim.relationship import RelationshipManager
from .events import *
from .embedding_client import cosine_similarity, get_embeddings, load_embedding_config
from onesim.utils.midsim_params import recommender_sampling_params, interest_recommendation_candidate_limits, memory_similarity_gate_params
from .user_agent_gates import MemorySimilarityGate

class Algorithm(GeneralAgent):
    """Platform Algorithm"""

    def __init__(self,
                 sys_prompt: str | None = None,
                 model_config_name: str = None,
                 event_bus_queue: asyncio.Queue = None,
                 profile: AgentProfile=None,
                 memory: MemoryStrategy=None,
                 planning: PlanningBase=None,
                 relationship_manager: RelationshipManager=None) -> None:
        super().__init__(sys_prompt, model_config_name, event_bus_queue, profile, memory, planning, relationship_manager)
        self.register_event("StartEvent", "update_current_notes")
        self.register_event("GetAlgorithmRecomendationEvent", "send_recommendation_results")
        self.register_event("GetSearchResultEvent", "send_search_results")
        
        # Recommendation algorithm mapping
        self.type_to_algorithm: Dict[str, str] = {
            "Random Recommendation": "random",
            "Hot Recommendation": "hot",
            "Interest Recommendation": "interest",
        }
        
        self.recommendation_algorithms: Dict[str, Callable] = {
            "random": self._random_recommendation,
            "hot": self._hot_recommendation,
            "interest": self._interest_recommendation,
        }
        self.default_algorithm = "hot"
        self._comment_count_dist_7d_cache: Optional[Dict[int, int]] = None

        # Search algorithm mapping
        self.type_to_search: Dict[str, str] = {
            "Relevant Search": "relevant",
        }
        
        self.search_algorithms: Dict[str, Callable] = {
            "relevant": self._relevant_search,
        }
        self.default_search = "relevant"

    def _parse_comment_count_dist_rows(self, rows: Any) -> Dict[int, int]:
        """Parse a list (or similar structure) from profile into comment_count -> post_count."""
        out: Dict[int, int] = {}
        if not isinstance(rows, list):
            return out
        for row in rows:
            if not isinstance(row, dict):
                continue
            raw_cc = row.get("comment_count")
            raw_pc = row.get("post_count")
            if raw_cc is None or raw_pc is None:
                continue
            try:
                cc = int(float(str(raw_cc).strip()))
                pc = int(float(str(raw_pc).strip()))
            except (TypeError, ValueError):
                continue
            if pc <= 0:
                continue
            out[cc] = out.get(cc, 0) + pc
        return out

    async def update_current_notes(self, event: Event) -> None:
        """Update current notes."""
        current_notes = event.current_notes
        if not isinstance(current_notes, dict) or not current_notes:
            logger.warning("No current notes found")
        self.profile.update_data("current_notes", current_notes)
        self._comment_count_dist_7d_cache = None

    def _sample_recommendation_limit_bernoulli(self, alpha: float, max_limit: int = 3) -> int:
        """Sample recommendation limit with Bernoulli distribution."""
        try:
            alpha = float(alpha)
        except (TypeError, ValueError):
            alpha = 0.5

        # Constrain alpha to [0, 1]
        alpha = max(0.0, min(1.0, alpha))
        max_limit = max(1, int(max_limit))

        # At least 1 recommendation; then at most max_limit-1 recommendations
        limit = 1
        for _ in range(max_limit - 1):
            if random.random() < alpha:
                limit += 1
            else:
                break
        return limit

    # ---------- Random recommendation ----------
    _RANDOM_REC_MAU_DEFAULT = 240_000_000   # Default Monthly Active Users (MAU)
    _RANDOM_REC_AVG_POSTS_PER_USER_MONTH_DEFAULT = 1.1182   # Default avg posts per user per month

    def _parse_positive_float(self, raw: Any, fallback: float) -> float:
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            return fallback
        try:
            return float(raw)
        except (TypeError, ValueError):
            return fallback

    def _random_rec_mau_avg_posts(self) -> Tuple[float, float]:
        """profile: random_rec_mau and random_rec_avg_posts_per_user_month."""
        mau = float(Algorithm._RANDOM_REC_MAU_DEFAULT)
        avg_posts = float(Algorithm._RANDOM_REC_AVG_POSTS_PER_USER_MONTH_DEFAULT)
        if self.profile is not None:
            mau = self._parse_positive_float(
                self.profile.get_data("random_rec_mau", None),
                mau,
            )
            avg_posts = self._parse_positive_float(
                self.profile.get_data("random_rec_avg_posts_per_user_month", None),
                avg_posts,
            )
        if mau <= 0:
            mau = float(Algorithm._RANDOM_REC_MAU_DEFAULT)
        return mau, avg_posts

    def _random_rec_per_note_weight(self) -> float:
        mau, avg_posts = self._random_rec_mau_avg_posts()
        return (1.0 / mau) * avg_posts

    def _random_recommendation(self, recommended_note_ids: List[str], contents: Dict[str, Dict[str, Any]], current_timestamp: float, limit: int = 10) -> Dict[str, Dict[str, Any]]:
        """Random recommendation, sample with equal probability (1/(MAU×avg)) from candidate pool."""
        if not contents:
            return {}
        note_ids = list(contents.keys())
        if not note_ids:
            return {}
        
        # Filter out recommended notes
        note_ids = [note_id for note_id in note_ids if note_id not in recommended_note_ids]
        if not note_ids:
            return {}

        # Count denominator, p = 1 / (MAU × avg)
        mau, avg_posts = self._random_rec_mau_avg_posts()
        denom = mau * avg_posts
        if denom <= 0:
            p_hit = 0.0
        else:
            p_hit = min(1.0, 1.0 / denom)

        random.shuffle(note_ids)
        log_each = bool(
            self.profile is not None
            and self.profile.get_data("random_rec_log_per_note_prob_check", False)
        )

        selected: List[str] = []
        for nid in note_ids:
            if len(selected) >= limit:
                break
            ok = random.random() < p_hit
            if log_each:
                logger.debug("random_rec note {}: Bernoulli(p=1/(MAU×avg)={:.6e}) => {}", nid, p_hit, "hit" if ok else "miss")
            if ok:
                selected.append(nid)

        w = self._random_rec_per_note_weight()
        logger.debug( "random recommendation: MAU={:.4g} avg_posts={:.6g} => p=1/(MAU×avg)={:.6e}; weight=(1/MAU)×avg={:.6e}",
            mau, avg_posts, p_hit, len(note_ids), limit, len(selected), w)
        return {note_id: contents[note_id] for note_id in selected}
    
    # ---------- Hot recommendation ----------
    def _get_comment_count_dist_7d_map(self) -> Dict[int, int]:
        """Get comment count distribution map for hot recommendation."""
        if self._comment_count_dist_7d_cache is not None:
            return self._comment_count_dist_7d_cache

        merged: Dict[int, int] = {}
        if self.profile is not None:
            raw = self.profile.get_data("comment_count_dist_7d", [])
            merged.update(self._parse_comment_count_dist_rows(raw))
        self._comment_count_dist_7d_cache = merged
        return merged

    def _calculate_popularity(self, note: Dict[str, Any]) -> float:
        """Calculate note popularity: use SimEnv's popularity (comment count in sliding window) if available, otherwise use comment_count."""
        if isinstance(note, dict) and "popularity" in note:
            raw = note.get("popularity")
            try:
                return float(raw) if raw is not None else 0.0
            except (TypeError, ValueError):
                return 0.0
        comment_count = note.get("comment_count", 0)
        return float(comment_count) if isinstance(comment_count, (int, float)) else 0.0

    def _hot_recommendation(self, recommended_note_ids: List[str], contents: Dict[str, Dict[str, Any]], current_timestamp: float, limit: int = 10) -> Dict[str, Dict[str, Any]]:
        """Hot recommendation: mix candidate note popularity with experience comment count distribution, and take the high-popularity notes."""
        if not contents:
            return {}
        note_ids = list(contents.keys())
        if not note_ids:
            return {}

        # Filter out recommended notes
        note_ids = [note_id for note_id in note_ids if note_id not in recommended_note_ids]
        if not note_ids:
            return {}

        # Get comment count distribution map
        dist_map = self._get_comment_count_dist_7d_map()
        if not dist_map:
            notes_with_popularity: List[Tuple[str, Dict[str, Any], float]] = []
            for note_id in note_ids:
                note = contents[note_id]
                popularity = self._calculate_popularity(note)
                notes_with_popularity.append((note_id, note, popularity))
            random.shuffle(notes_with_popularity)
            notes_with_popularity.sort(key=lambda x: x[2], reverse=True)
            return {note_id: note for note_id, note, _ in notes_with_popularity[:limit]}

        # Mix candidate notes with experience comment count distribution, and sort by popularity
        buckets: Dict[int, List[Optional[str]]] = defaultdict(list)
        for cc, cnt in dist_map.items():
            try:
                n = int(cnt)
            except (TypeError, ValueError):
                continue
            for _ in range(max(0, n)):
                buckets[cc].append(None)
        for note_id in note_ids:
            p = int(self._calculate_popularity(contents[note_id]))
            buckets[p].append(note_id)

        merged: List[Optional[str]] = []
        for cc in sorted(buckets.keys(), reverse=True):
            row = buckets[cc]
            random.shuffle(row)
            merged.extend(row)

        # Get top notes by popularity
        top_slice = merged[: max(0, limit)]
        chosen_ids = [x for x in top_slice if x is not None]
        return {nid: contents[nid] for nid in chosen_ids if nid in contents}

    # ---------- Interest recommendation ----------
    @staticmethod
    def _interest_interleave_pool_for_prompt(pool_ids: List[str], pool_set: Set[str]) -> List[str]:
        """Interleave candidate_pool and current_feed ids."""
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
    def _historical_summary_head_tail(text: Any) -> str:
        """Truncate historical_summary to 100 characters, and keep the first 50 and last 50 characters with "…" in between."""
        if text is None:
            return ""
        s = str(text).strip()
        if len(s) > 100:
            return s[:50] + "…" + s[-50:]
        return s

    @staticmethod
    def _compact_user_profile_for_interest_rec(
        user_profile: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Construct input profile for interest recommendation."""
        if not user_profile or not isinstance(user_profile, dict):
            return {}
        keys = ("id", "nickname", "gender", "description", "location", "interest_tags", "historical_summary")
        out: Dict[str, Any] = {}
        for k in keys:
            if k not in user_profile:
                continue
            v = user_profile[k]
            if k == "historical_summary":
                v = Algorithm._historical_summary_head_tail(v)
            elif k == "description" and isinstance(v, str) and len(v) > 400:
                v = v[:400] + "…"
            out[k] = v
        return out

    async def _interest_recommendation(
        self,
        recommended_note_ids: List[str],
        contents: Dict[str, Dict[str, Any]],
        current_timestamp: float,
        limit: int = 10,
        candidate_note_ids: Optional[List[str]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Interest recommendation: mix target list with candidate pool, and use LLM to re-rank."""
        interest_pool: Dict[str, Dict[str, Any]] = {}
        if self.profile is not None:
            raw_ip = self.profile.get_data("interest_recommendation_content_pool", None)
            if isinstance(raw_ip, dict):
                interest_pool = {
                    str(k).strip(): v
                    for k, v in raw_ip.items()
                    if isinstance(v, dict)
                }

        def _resolve_note(nid: str) -> Optional[Dict[str, Any]]:
            sid = str(nid).strip()
            n = interest_pool.get(sid)
            if isinstance(n, dict):
                return n
            c = contents.get(sid) if isinstance(contents, dict) else None
            return c if isinstance(c, dict) else None

        # Get candidate note ids
        candidate_note_ids = list(candidate_note_ids) if candidate_note_ids else []
        interest_k, target_k = await interest_recommendation_candidate_limits(self)

        ids_for_user: List[str] = []
        if candidate_note_ids:
            ids_for_user = [str(x).strip() for x in candidate_note_ids if x]
        ids_for_user = [
            nid for nid in ids_for_user
            if _resolve_note(nid) is not None
            and nid not in recommended_note_ids
        ]
        random.shuffle(ids_for_user)
        pool_sample: List[str] = ids_for_user[:interest_k]
        pool_set: Set[str] = set(pool_sample)

        # Construct current feed note ids
        feed_note_ids = list(contents.keys()) if contents else []
        feed_note_ids = [note_id for note_id in feed_note_ids if note_id not in recommended_note_ids]
        feed_eligible = [note_id for note_id in feed_note_ids if note_id not in pool_set]
        random.shuffle(feed_eligible)
        feed_sample: List[str] = feed_eligible[:target_k]

        # Mix candidate pool and current feed
        pool_ids: List[str] = list(pool_sample) + feed_sample
        if not pool_ids:
            return {}
        if os.environ.get("ONESIM_REC_INTERLEAVE_INPUT", "1").strip().lower() not in ("0", "false", "no", "off"):
            pool_ids = self._interest_interleave_pool_for_prompt(pool_ids, pool_set)

        def _make_candidate_payload(
            ids: List[str], desc_max: int
        ) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for note_id in ids:
                note = _resolve_note(note_id)
                if not isinstance(note, dict):
                    continue
                out.append({
                    "note_id": note_id,
                    "title": note.get("title", ""),
                    "desc": (note.get("desc", "") or "")[:desc_max],
                    "tags_list": note.get("tags_list", []),
                })
            return out

        # Construct prompt and avoid context overflow
        max_prompt_chars = int(
            os.environ.get("ONESIM_RECOMMENDER_INTEREST_MAX_PROMPT_CHARS", "48000")
        )
        profile_compact = self._compact_user_profile_for_interest_rec(user_profile)
        if profile_compact:
            user_profile_prefix = (
                "请求用户画像（用于兴趣个性化排序；请优先对齐 interest_tags、description、"
                "historical_summary 等所体现的主题与意图）：\n"
                f"{json.dumps(profile_compact, ensure_ascii=False)}\n\n"
            )
        else:
            user_profile_prefix = ""

        instruction = f"""
        你是推荐系统助手。请对下方「候选 note_id 全集」输出一条**完整排序列表**（从高到低），并严格返回 JSON。

        【目标 — 与用户兴趣对齐】
        1) **主排序信号（权重最高）**：对照用户画像中的 **interest_tags、description、historical_summary**，估计每条笔记与该用户在**主题域、身份角色、长期关切、表达习惯**上的一致程度；越像「该用户会点进、会停留、会评论/转发」的内容越靠前。
        2) **语义与标签匹配**：在标题、摘要（desc 截断）、tags_list 中，与用户显式兴趣标签或摘要中反复出现的关切**直接命中或强相关**的笔记，优先于仅有弱联想的内容。
        3) **次排序信号（主信号接近时的 tie-break）**：信息是否完整可读（标题+摘要能判断主题，非空泛标题党）、标签是否与正文主题一致；同等相关下，**更具体、更有信息增量**的笔记优先于空洞或重复套话的笔记。
        4) **多样性轻约束**：主信号已排序后，若多条笔记主题极度雷同，可适当错开相邻位次，使前段列表覆盖略广的兴趣面（仍须满足下方「覆盖全集、不丢 id」的硬约束）。

        请严格返回 JSON：
        {{
        "ranked_note_ids": ["id1", "id2", ...],
        "per_note_category": ["类别归纳1", "类别归纳2", ...]
        }}

        字段说明：
        - per_note_category：与 ranked_note_ids **等长**；第 i 项是对排序中**第 i 条**笔记的**类别归纳**（2～12 字为宜），如「学术/投稿」「生活/美食」「情感/成长」等，体现主题域或意图，勿写成长句理由。
        约束：
        1) 每个 id 必须来自下方候选列表中的 note_id，且**覆盖全集**；
        2) 不要重复；
        3) per_note_category 条数必须与 ranked_note_ids 一致且顺序一一对应；
        """
        desc_max = 800
        candidate_payload: List[Dict[str, Any]] = []
        observation = ""
        total_len = 0
        for _ in range(512):
            candidate_payload = _make_candidate_payload(pool_ids, desc_max)
            pool_ids_json = json.dumps(pool_ids, ensure_ascii=False)
            observation = (
                user_profile_prefix
                + f"候选 note_id 全集（须只在此集合内排序）：{pool_ids_json}\n\n"
                + f"候选笔记详情：\n{json.dumps(candidate_payload, ensure_ascii=False)}"
            )
            total_len = len(instruction) + len(observation)
            if total_len <= max_prompt_chars:
                break
            if len(pool_ids) > 1:
                pool_ids.pop()
                logger.warning(f"Interest recommendation prompt over budget ({total_len} > {max_prompt_chars} chars): dropped last candidate, {len(pool_ids)} left")
                continue
            if desc_max > 200:
                desc_max = max(200, desc_max // 2)
                logger.warning(f"Interest recommendation prompt over budget: shrinking desc cap to {desc_max} chars")
                continue
            if desc_max > 80:
                desc_max = 80
                logger.warning("Interest recommendation prompt over budget: shrinking desc cap to 80 chars")
                continue
            logger.error(f"Interest recommendation: prompt still {total_len} chars (limit {max_prompt_chars}); proceeding anyway")
            break

        contents_d: Dict[str, Dict[str, Any]] = (
            contents if isinstance(contents, dict) else {}
        )

        ordered_ids: List[str] = []
        pool_set_all: Set[str] = set(pool_ids)
        try:
            # Generate recommendation
            response = await self.generate_recommendation(instruction, observation)

            # Get ranked note ids
            raw_ids = response.get("ranked_note_ids")
            if raw_ids is None:
                raw_ids = response.get("selected_note_ids", [])
            if isinstance(raw_ids, list):
                for note_id in raw_ids:
                    note_id = str(note_id).strip()
                    if (
                        note_id
                        and note_id in pool_set_all
                        and _resolve_note(note_id) is not None
                        and note_id not in recommended_note_ids
                        and note_id not in ordered_ids
                    ):
                        ordered_ids.append(note_id)

        except Exception as e:
            logger.warning(f"LLM interest recommendation failed: {e}")
            return {}

        # Get top note ids
        if not ordered_ids:
            return {}
        top_ids = ordered_ids[: max(0, int(limit))]
        logger.info(f"Algorithm length of top_ids: {len(top_ids)}")

        # Filter target notes from ranked result
        def _note_from_contents_only(sid: str) -> Optional[Dict[str, Any]]:
            if sid in contents_d:
                v = contents_d.get(sid)
                if isinstance(v, dict):
                    return v
            for k, v in contents_d.items():
                if str(k).strip() == sid and isinstance(v, dict):
                    return v
            return None

        out: Dict[str, Dict[str, Any]] = {}
        for sid in top_ids:
            n = _note_from_contents_only(sid)
            if n is not None:
                out[sid] = n
        logger.info(f"Algorithm length of out: {len(out)}")
        return out

    async def send_recommendation_results(self, event: Event) -> List[Event]:
        """Send recommendation results."""
        # Get current timestamp
        current_timestamp = event.timestamp

        # Get content pool
        current_notes = {}
        if self.profile is not None:
            current_notes = self.profile.get_data("current_notes", {})
        if not isinstance(current_notes, dict) or not current_notes:
            current_notes = getattr(event, "current_notes", {})
        if not isinstance(current_notes, dict) or not current_notes:
            logger.warning("No current notes found from profile/event")
            return []

        # Check algorithm type
        expected_type = getattr(event, "type", None)
        type_value = await self.get_data("type", "")
        exp_s = str(expected_type).strip() if expected_type is not None else ""
        got_s = str(type_value).strip() if type_value is not None else ""
        if exp_s != got_s:
            if not exp_s and got_s:
                logger.debug(
                    f"Algorithm: event.type is empty, process with type={got_s!r}"
                )
            else:
                raise ValueError(
                    f"Algorithm: recommendation algorithm type mismatch — "
                    f"this agent get_data('type')={got_s!r}, event.type={exp_s!r}, "
                    f"from_agent_id={getattr(event, 'from_agent_id', '')!r}"
                )

        # Get recommendation algorithm function
        algorithm_name = self.type_to_algorithm.get(type_value, self.default_algorithm)
        if algorithm_name not in self.recommendation_algorithms:
            algorithm_name = self.default_algorithm
            logger.warning(f"Unknown recommendation algorithm type '{type_value}', using default: {self.default_algorithm}")

        recommendation_func = self.recommendation_algorithms[algorithm_name]

        # Bernoulli sampling
        alpha, max_limit = await recommender_sampling_params(self, mode="recommendation")
        recommendation_limit = self._sample_recommendation_limit_bernoulli(alpha, max_limit=max_limit)
        logger.info(f"Algorithm recommendation_limit: {recommendation_limit}")

        # Get candidate note ids by user
        by_user: Dict[str, Any] = {}
        if self.profile is not None:
            raw_map = self.profile.get_data("candidate_note_ids_by_user", {})
            if isinstance(raw_map, dict):
                by_user = raw_map

        requester_id = str(getattr(event, "from_agent_id", "") or "").strip()
        candidate_note_ids: List[str] = []
        if requester_id:
            raw_list = by_user.get(requester_id)
            if raw_list is None:
                for k, v in by_user.items():
                    if str(k).strip() == requester_id and isinstance(v, list):
                        raw_list = v
                        break
            if isinstance(raw_list, list):
                candidate_note_ids = [str(x).strip() for x in raw_list if x]

        # Get recommended note ids
        recommended_note_ids = list(event.recommended_note_ids)

        # Get user profile
        user_profile_raw = getattr(event, "user_profile", None)
        user_profile: Dict[str, Any] = (
            user_profile_raw if isinstance(user_profile_raw, dict) else {}
        )

        if algorithm_name == "interest":
            recommended_contents = await recommendation_func(
                recommended_note_ids,
                current_notes,
                current_timestamp,
                recommendation_limit,
                candidate_note_ids=candidate_note_ids,
                user_profile=user_profile,
            )
        else:
            recommended_contents = recommendation_func(
                recommended_note_ids,
                current_notes,
                current_timestamp,
                recommendation_limit
            )
            
        if not recommended_contents:
            logger.debug(f"No recommended contents for user {getattr(event, 'from_agent_id', 'unknown')}")
            return []

        logger.info(
            f"Sending AlgorithmRecommendationEvent to UserAgent {event.from_agent_id}, "
            f"recommended contents: {len(recommended_contents)}"
        )

        recommendation_event = AlgorithmRecommendationEvent(
            from_agent_id=self.profile_id,
            to_agent_id=event.from_agent_id,
            timestamp=current_timestamp,
            timestamp_duration=int(getattr(event, "timestamp_duration", 0) or 0),
            current_step=getattr(event, "current_step", 1),
            max_step=getattr(event, "max_step", 8),
            recommendations=dict(recommended_contents)
        )
        return [recommendation_event]

    # ---------- Search algorithm ----------
    @staticmethod
    def _search_note_ids(current_notes: Dict[str, Dict[str, Any]]) -> List[str]:
        """Get note ids that can be used as search candidates from content pool."""
        out: List[str] = []
        for nid, note in (current_notes or {}).items():
            if not isinstance(note, dict):
                continue
            sid = str(nid).strip()
            if sid:
                out.append(sid)
        return out

    @staticmethod
    def _note_text_for_search(note: Dict[str, Any], max_len: int = 1200) -> str:
        """Concatenate title, description and tags for search, and truncate to max_len."""
        title = str(note.get("title", "") or "")
        desc = str(note.get("desc", "") or "")
        tags = note.get("tags_list")
        if isinstance(tags, list):
            tg = " ".join(str(t) for t in tags[:30])
        else:
            tg = str(tags or "")
        s = f"{title}\n{desc}\n{tg}".strip()
        return s if len(s) <= max_len else s[:max_len]

    def _random_search(
        self,
        current_notes: Dict[str, Dict[str, Any]],
        current_timestamp: float,
        limit: int,
    ) -> Dict[str, Dict[str, Any]]:
        """Random search：shuffle candidate notes and take top limit notes."""
        if not current_notes or limit <= 0:
            return {}
        note_ids = list(current_notes.keys())
        random.shuffle(note_ids)
        picked = note_ids[:limit]
        return {nid: current_notes[nid] for nid in picked if nid in current_notes}

    def _keyword_search(
        self,
        current_notes: Dict[str, Dict[str, Any]],
        current_timestamp: float,
        limit: int,
        *,
        search_query: str = "",
    ) -> Dict[str, Dict[str, Any]]:
        """Keyword search：query Jaccard similarity with note text."""
        if not current_notes or limit <= 0:
            return {}
        cands = self._search_note_ids(current_notes)
        if not cands:
            return {}

        query = (search_query or "").strip()[:4000]
        if not query:
            sub_notes = {nid: current_notes[nid] for nid in cands if nid in current_notes}
            return self._random_search(sub_notes, current_timestamp, limit)

        qt = MemorySimilarityGate.tokenize(query)
        scored: List[Tuple[str, float]] = []
        for nid in cands:
            note = current_notes.get(nid)
            if not isinstance(note, dict):
                continue
            text = self._note_text_for_search(note, max_len=2000)
            nt = MemorySimilarityGate.tokenize(text)
            if not qt or not nt:
                scored.append((nid, 0.0))
                continue
            inter = len(qt & nt)
            union = len(qt | nt)
            j = (inter / union) if union else 0.0
            scored.append((nid, j))
        random.shuffle(scored)
        scored.sort(key=lambda x: x[1], reverse=True)
        return {
            nid: current_notes[nid]
            for nid, _ in scored[:limit]
            if nid in current_notes
        }

    def _relevant_search_sync(
        self,
        current_notes: Dict[str, Dict[str, Any]],
        current_timestamp: float,
        limit: int,
        *,
        search_query: str,
        embedding_config_path: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Relevant search：query and note embedding, cosine similarity sort."""
        if not current_notes or limit <= 0:
            return {}

        # Get searchable candidate ids
        cands = self._search_note_ids(current_notes)
        if not cands:
            return {}

        # Get search query
        query = (search_query or "").strip()[:4000]
        if not query.strip():
            sub_notes = {nid: current_notes[nid] for nid in cands if nid in current_notes}
            return self._random_search(sub_notes, current_timestamp, limit)

        # Batch embedding, query vector vs note vector cosine similarity
        try:
            base_url, model_name = load_embedding_config(embedding_config_path)
            texts = [query[:2000]]
            for nid in cands:
                note = current_notes.get(nid)
                texts.append(
                    self._note_text_for_search(note, max_len=800)
                    if isinstance(note, dict)
                    else ""
                )
            vecs = get_embeddings(base_url, model_name, texts)
            if not vecs or len(vecs) != len(texts):
                raise ValueError("embedding return length does not match input")
            vq = vecs[0]
            scored: List[Tuple[str, float]] = []
            for i, nid in enumerate(cands):
                sim = float(cosine_similarity(vq, vecs[i + 1]))
                scored.append((nid, sim))
            random.shuffle(scored)
            scored.sort(key=lambda x: x[1], reverse=True)
            top = scored[:limit]
            return {nid: current_notes[nid] for nid, _ in top if nid in current_notes}
        except Exception as e:
            logger.warning(f"Relevant search embedding failed, fallback to keyword: {e}")
            return self._keyword_search(
                current_notes,
                current_timestamp,
                limit,
                search_query=query,
            )

    async def _relevant_search(
        self,
        current_notes: Dict[str, Dict[str, Any]],
        current_timestamp: float,
        limit: int,
        *,
        search_query: str,
        embedding_config_path: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Relevant search：query and note embedding, cosine similarity sort."""
        return await asyncio.to_thread(
            self._relevant_search_sync,
            current_notes,
            current_timestamp,
            limit,
            search_query=search_query,
            embedding_config_path=embedding_config_path,
        )

    async def send_search_results(self, event: Event) -> List[Event]:
        """Generate search results."""
        # Get current timestamp
        current_timestamp = event.timestamp

        # Get content pool
        current_notes = {}
        if self.profile is not None:
            current_notes = self.profile.get_data("current_notes", {})
        if not isinstance(current_notes, dict) or not current_notes:
            current_notes = getattr(event, "current_notes", {})
        if not isinstance(current_notes, dict) or not current_notes:
            logger.warning("No current notes found from profile/event")
            return []

        # Check search algorithm type
        expected_type = getattr(event, "type", None)
        type_value = await self.get_data("type", "")
        exp_s = str(expected_type).strip() if expected_type is not None else ""
        got_s = str(type_value).strip() if type_value is not None else ""
        if exp_s != got_s:
            if not exp_s and got_s:
                logger.debug(f"Algorithm: event.type is empty, process with type={got_s!r}")
            else:
                raise ValueError(
                    f"Algorithm: search algorithm type mismatch — "
                    f"this agent get_data('type')={got_s!r}, event.type={exp_s!r}, "
                    f"from_agent_id={getattr(event, 'from_agent_id', '')!r}"
                )

        # Map type value to search algorithm name
        algorithm_name = self.type_to_search.get(type_value, self.default_search)
        if algorithm_name not in self.search_algorithms:
            algorithm_name = self.default_search
            logger.warning(
                f"Unknown search algorithm type '{type_value}', using default: {self.default_search}"
            )

        search_func = self.search_algorithms[algorithm_name]

        # Bernoulli sampling
        alpha, max_limit = await recommender_sampling_params(self, mode="search")
        recommendation_limit = self._sample_recommendation_limit_bernoulli(alpha, max_limit=max_limit)
        top_k = max(1, min(max_limit, recommendation_limit))
        logger.info(
            f"Algorithm search algorithm={algorithm_name}, "
            f"max_limit={max_limit}, sampled top_k={top_k}"
        )

        search_query = str(getattr(event, "search_query", "") or "")
        cfg = MemorySimilarityGate.load_config(await memory_similarity_gate_params(self))

        search_results: Dict[str, Dict[str, Any]] = {}
        if algorithm_name == "relevant":
            search_results = await search_func(
                current_notes,
                current_timestamp,
                top_k,
                search_query=search_query,
                embedding_config_path=cfg.embedding_config_path,
            )

        if not search_results:
            logger.debug(f"No search results for user {getattr(event, 'from_agent_id', 'unknown')}")
            return []

        logger.info(
            f"Sending SearchRecommendationEvent (search) to UserAgent {event.from_agent_id}, "
            f"search results: {len(search_results)}"
        )

        search_event = SearchResultEvent(
            from_agent_id=self.profile_id,
            to_agent_id=event.from_agent_id,
            timestamp=current_timestamp,
            timestamp_duration=int(getattr(event, "timestamp_duration", 0) or 0),
            current_step=getattr(event, "current_step", 1),
            max_step=getattr(event, "max_step", 8),
            recommendations=dict(search_results),
        )
        return [search_event]