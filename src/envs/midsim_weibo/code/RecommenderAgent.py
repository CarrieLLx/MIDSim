from typing import Any, List, Optional, Callable, Dict, Tuple, Set
from collections import defaultdict
import csv
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
from .metrics.repost_similarity import (
    cosine_similarity,
    get_embeddings,
    load_embedding_config,
)
from .step15_topic_gate import _tokenize, load_step15_gate_config

# 默认 7 天评论数分布 CSV（与 schema 说明一致）
_ICLR_DEFAULT_REPOST_DIST_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "profile",
    "data",
    "creator_contents_shared_count_within_7d.csv",
)

class RecommenderAgent(GeneralAgent):
    """推荐算法智能体：支持随机推荐和热点推荐"""
    
    # 随机推荐：对每条候选帖以 p=1/(MAU×月均发帖) 做伯努利试验，抽中则收录直至 limit；见 _random_recommendation。
    # 另：w=(1/MAU)×avg_posts 仍可用于其它说明，与 p 数值不同（p=1/分母，w=avg/MAU）。
    # 月活、月均发帖量仅来自 profile：random_rec_mau、random_rec_avg_posts_per_user_month；不读环境变量。
    # 未配置或非法时回退到下方类常量（非环境变量）。
    _RANDOM_REC_MAU_DEFAULT = 588_000_000
    _RANDOM_REC_AVG_POSTS_PER_USER_MONTH_DEFAULT = 95.69

    def _parse_positive_float(self, raw: Any, fallback: float) -> float:
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            return fallback
        try:
            return float(raw)
        except (TypeError, ValueError):
            return fallback

    def _random_rec_mau_avg_posts(self) -> Tuple[float, float]:
        """profile: random_rec_mau、random_rec_avg_posts_per_user_month；与 w 公式共用。"""
        mau = float(RecommenderAgent._RANDOM_REC_MAU_DEFAULT)
        avg_posts = float(RecommenderAgent._RANDOM_REC_AVG_POSTS_PER_USER_MONTH_DEFAULT)
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
            mau = float(RecommenderAgent._RANDOM_REC_MAU_DEFAULT)
        return mau, avg_posts

    def _random_rec_per_blog_weight(self) -> float:
        mau, avg_posts = self._random_rec_mau_avg_posts()
        return (1.0 / mau) * avg_posts

    def __init__(self,
                 sys_prompt: str | None = None,
                 model_config_name: str = None,
                 event_bus_queue: asyncio.Queue = None,
                 profile: AgentProfile=None,
                 memory: MemoryStrategy=None,
                 planning: PlanningBase=None,
                 relationship_manager: RelationshipManager=None) -> None:
        super().__init__(sys_prompt, model_config_name, event_bus_queue, profile, memory, planning, relationship_manager)
        self.register_event("StartEvent", "update_current_blogs")
        self.register_event("GetAlgorithmRecomendationEvent", "send_algorithm_recommendations")
        self.register_event("SearchEvent", "send_search_recommendations")
        
        # type字段值到算法名称的映射
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
        self._repost_count_dist_7d_cache: Optional[Dict[int, int]] = None

    @staticmethod
    def _is_original_weibo(blog: Any) -> bool:
        """原创帖：无 reposted_blog_id 或为空字符串（非转发链上的节点）。"""
        if not isinstance(blog, dict):
            return False
        rid = blog.get("reposted_blog_id")
        if rid is None:
            return True
        return str(rid).strip() == ""

    def _parse_repost_count_dist_rows(self, rows: Any) -> Dict[int, int]:
        """将 profile 中的列表或类似结构解析为 repost_count -> post_count。"""
        out: Dict[int, int] = {}
        if not isinstance(rows, list):
            return out
        for row in rows:
            if not isinstance(row, dict):
                continue
            raw_rc = row.get("repost_count")
            raw_pc = row.get("post_count")
            if raw_rc is None or raw_pc is None:
                continue
            try:
                rc = int(float(str(raw_rc).strip()))
                pc = int(float(str(raw_pc).strip()))
            except (TypeError, ValueError):
                continue
            if pc <= 0:
                continue
            out[rc] = out.get(rc, 0) + pc
        return out

    def _read_repost_count_dist_csv(self, path: str) -> Dict[int, int]:
        out: Dict[int, int] = {}
        if not path or not os.path.isfile(path):
            return out
        try:
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not row:
                        continue
                    raw_rc = row.get("repost_count")
                    raw_pc = row.get("post_count")
                    if raw_rc is None or raw_pc is None:
                        continue
                    try:
                        rc = int(float(str(raw_rc).strip()))
                        pc = int(float(str(raw_pc).strip()))
                    except (TypeError, ValueError):
                        continue
                    if pc <= 0:
                        continue
                    out[rc] = out.get(rc, 0) + pc
        except OSError as e:
            logger.warning(f"Failed to read repost_count dist CSV {path}: {e}")
        return out
        
    def _get_repost_count_dist_7d_map(self) -> Dict[int, int]:
        """
        7 天去重帖子的转发数分布（与站外/历史分布一起参与热点排序）。
        优先使用 profile.repost_count_dist_7d；若为空则从 CSV 读取。
        CSV 路径：profile.repost_count_dist_7d_csv（非空则用之），否则用内置默认路径；不读环境变量。
        """
        if self._repost_count_dist_7d_cache is not None:
            return self._repost_count_dist_7d_cache

        merged: Dict[int, int] = {}
        if self.profile is not None:
            raw = self.profile.get_data("repost_count_dist_7d", [])
            merged.update(self._parse_repost_count_dist_rows(raw))
        if not merged:
            csv_path = _ICLR_DEFAULT_REPOST_DIST_CSV
            if self.profile is not None:
                raw_path = self.profile.get_data("repost_count_dist_7d_csv", None)
                if isinstance(raw_path, str) and raw_path.strip():
                    csv_path = os.path.abspath(os.path.expanduser(raw_path.strip()))
            merged = self._read_repost_count_dist_csv(csv_path)
        self._repost_count_dist_7d_cache = merged
        return merged

    async def update_current_blogs(self, event: Event) -> None:
        """
        更新当前微博
        """
        current_blogs = event.current_blogs
        if not isinstance(current_blogs, dict) or not current_blogs:
            logger.warning("No current blogs found")
        self.profile.update_data("current_blogs", current_blogs)

    def _calculate_popularity(self, blog: Dict[str, Any]) -> float:
        """
        内容热度：优先使用 SimEnv 在每步写入的 ``popularity``（滑动时间窗内链式转发热度）；
        无该字段时退化为 ``repost_count``（累计转发数，兼容 int/float 与数字字符串）。
        """
        if isinstance(blog, dict) and "popularity" in blog:
            rawp = blog.get("popularity")
            try:
                return float(rawp) if rawp is not None else 0.0
            except (TypeError, ValueError):
                return 0.0
        raw = blog.get("repost_count", 0)
        if raw is None:
            return 0.0
        if isinstance(raw, bool):
            return 0.0
        if isinstance(raw, (int, float)):
            return float(raw)
        if isinstance(raw, str):
            s = raw.strip()
            if not s:
                return 0.0
            try:
                return float(s)
            except ValueError:
                return 0.0
        return 0.0

    def _sample_recommendation_limit_bernoulli(self, alpha: float, max_limit: int = 3) -> int:
        """
        使用“继续概率为 alpha”的伯努利衰减，采样本轮推荐数量。

        建模含义：每次已经推荐 1 屏后，用户以概率 alpha 决定继续往下看下一屏；
        连续继续下去的次数决定最终推荐数量（上限为 max_limit）。

        例如 max_limit=3 时：
          P(1) = 1 - alpha
          P(2) = alpha * (1 - alpha)
          P(3) = alpha^2
        """
        try:
            alpha = float(alpha)
        except (TypeError, ValueError):
            alpha = 0.5

        # alpha 约束到 [0, 1]
        alpha = max(0.0, min(1.0, alpha))
        max_limit = max(1, int(max_limit))

        # 至少推 1 个；之后最多再推 max_limit-1 个
        limit = 1
        for _ in range(max_limit - 1):
            if random.random() < alpha:
                limit += 1
            else:
                break
        return limit

    def _random_recommendation(self, recommended_blog_ids: List[str], contents: Dict[str, Dict[str, Any]], current_timestamp: float, limit: int = 10) -> Dict[str, Dict[str, Any]]:
        """随机推荐算法"""
        if not contents:
            return {}
        
        # 从字典中随机选择 blog_id
        blog_ids = list(contents.keys())
        if not blog_ids:
            return {}
        
        blog_ids = [blog_id for blog_id in blog_ids if blog_id not in recommended_blog_ids]
        if not blog_ids:
            return {}
        
        mau, avg_posts = self._random_rec_mau_avg_posts()
        denom = mau * avg_posts
        if denom <= 0:
            p_hit = 0.0
        else:
            p_hit = min(1.0, 1.0 / denom)

        random.shuffle(blog_ids)
        log_each = bool(
            self.profile is not None
            and self.profile.get_data("random_rec_log_per_blog_prob_check", False)
        )

        selected: List[str] = []
        for bid in blog_ids:
            if len(selected) >= limit:
                break
            ok = random.random() < p_hit
            if log_each:
                logger.debug(
                    "random_rec 帖 {}: Bernoulli(p=1/(MAU×avg)={:.6e}) => {}",
                    bid,
                    p_hit,
                    "hit" if ok else "miss",
                )
            if ok:
                selected.append(bid)

        w = self._random_rec_per_blog_weight()
        logger.debug(
            "random recommendation: MAU={:.4g} avg_posts={:.6g} => p=1/(MAU×avg)={:.6e}; "
            "候选 N={} 上限 limit={} => 本轮收录 {} 条；w=(1/MAU)×avg={:.6e}",
            mau,
            avg_posts,
            p_hit,
            len(blog_ids),
            limit,
            len(selected),
            w,
        )
        return {blog_id: contents[blog_id] for blog_id in selected}
    
    def _hot_recommendation(self, recommended_blog_ids: List[str], contents: Dict[str, Dict[str, Any]], current_timestamp: float, limit: int = 10) -> Dict[str, Dict[str, Any]]:
        """
        热点推荐：考虑 ``contents`` 中全部候选（排除已推荐）。

        - 真实帖热度：``_calculate_popularity``（优先 SimEnv 的 ``popularity``，否则 ``repost_count``）。
        - 若配置了 ``repost_count_dist_7d``（或 CSV）：按转发数分桶放置「虚拟帖」占位，与真实帖
          在同一整数桶内 shuffle 后按桶键降序串联；前 ``limit`` 个槽位中只输出真实 blog_id。
          虚拟桶键来自经验转发数分布，真实帖桶键来自上述热度。
        - 若无分布配置：对全部候选打乱后按热度降序取前 ``limit`` 条。

        ``current_timestamp`` 保留签名以兼容调用方（Unix 秒）。
        """
        if not contents:
            return {}

        eligible: Dict[str, Dict[str, Any]] = {}
        for blog_id, blog in contents.items():
            if blog_id in recommended_blog_ids or not isinstance(blog, dict):
                continue
            eligible[blog_id] = blog

        if not eligible:
            return {}

        dist_map = self._get_repost_count_dist_7d_map()
        if not dist_map:
            blogs_with_popularity: List[Tuple[str, Dict[str, Any], float]] = []
            for blog_id, blog in eligible.items():
                popularity = self._calculate_popularity(blog)
                blogs_with_popularity.append((blog_id, blog, popularity))
            random.shuffle(blogs_with_popularity)
            blogs_with_popularity.sort(key=lambda x: x[2], reverse=True)
            return {blog_id: blog for blog_id, blog, _ in blogs_with_popularity[:limit]}

        buckets: Dict[int, List[Optional[str]]] = defaultdict(list)
        for rc, cnt in dist_map.items():
            try:
                n = int(cnt)
            except (TypeError, ValueError):
                continue
            for _ in range(max(0, n)):
                buckets[rc].append(None)
        for blog_id, blog in eligible.items():
            p = int(self._calculate_popularity(blog))
            buckets[p].append(blog_id)

        merged: List[Optional[str]] = []
        for rc in sorted(buckets.keys(), reverse=True):
            row = buckets[rc]
            random.shuffle(row)
            merged.extend(row)

        top_slice = merged[: max(0, limit)]
        chosen_ids = [x for x in top_slice if x is not None]
        return {bid: eligible[bid] for bid in chosen_ids if bid in eligible}
    
    def _top_hot_blogs(
        self,
        recommended_blog_ids: List[str],
        contents: Dict[str, Dict[str, Any]],
        top_k: int = 100,
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """取热度最高的 top_k 条（过滤掉已推荐过的）。"""
        if not contents:
            return []
        rows: List[Tuple[str, Dict[str, Any], float]] = []
        for blog_id, blog in contents.items():
            if blog_id in recommended_blog_ids or not isinstance(blog, dict):
                continue
            popularity = self._calculate_popularity(blog)
            rows.append((blog_id, blog, popularity))
        random.shuffle(rows)
        rows.sort(key=lambda x: x[2], reverse=True)
        return rows[:top_k]

    def _extract_interest_tags(self, user_profile: Dict[str, Any]) -> List[str]:
        tags = user_profile.get("interest_tags", [])
        if not isinstance(tags, list):
            return []
        return [str(t).strip() for t in tags if str(t).strip()]

    @staticmethod
    def _interest_interleave_pool_for_prompt(pool_ids: List[str], pool_set: Set[str]) -> List[str]:
        """
        在调用大模型**之前**，将 candidate_pool 与 current_feed 两路 id 交错排列，
        避免 observation 里同源微博成片连续；不改变集合内容，只改展示顺序。
        """
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
        """historical_summary 超过 100 字符时保留前 50 与后 50，中间「…」；否则返回 strip 后的全文。"""
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
        """从事件携带的 user_profile 中抽取兴趣排序相关字段，并截断过长文本以控制 prompt 体积。
        historical_summary 超过 100 字时保留前 50 字与后 50 字，中间以「…」连接。
        """
        if not user_profile or not isinstance(user_profile, dict):
            return {}
        keys = (
            "id",
            "nickname",
            "gender",
            "description",
            "location",
            "interest_tags",
            "historical_summary",
        )
        out: Dict[str, Any] = {}
        for k in keys:
            if k not in user_profile:
                continue
            v = user_profile[k]
            if k == "historical_summary":
                v = RecommenderAgent._historical_summary_head_tail(v)
            elif k == "description" and isinstance(v, str) and len(v) > 400:
                v = v[:400] + "…"
            out[k] = v
        return out

    async def _interest_recommendation(
        self,
        recommended_blog_ids: List[str],
        contents: Dict[str, Dict[str, Any]],
        current_timestamp: float,
        limit: int = 10,
        candidate_blog_ids: Optional[List[str]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        兴趣推荐（LLM 重排序）：两路候选合并后一次性排序。
        按该顺序取前 ``limit`` 个 id；再**仅**保留其中在 ``contents``（current_blogs）里存在的条目并返回（只在兴趣池、不在 current_blogs 的丢弃）。
        若最终没有任何条在 current_blogs 中命中，返回空字典即可。
        """
        interest_pool: Dict[str, Dict[str, Any]] = {}
        if self.profile is not None:
            raw_ip = self.profile.get_data("interest_recommendation_content_pool", None)
            if isinstance(raw_ip, dict):
                interest_pool = {
                    str(k).strip(): v
                    for k, v in raw_ip.items()
                    if isinstance(v, dict)
                }

        def _resolve_blog(bid: str) -> Optional[Dict[str, Any]]:
            sid = str(bid).strip()
            n = interest_pool.get(sid)
            if isinstance(n, dict):
                return n
            c = contents.get(sid) if isinstance(contents, dict) else None
            return c if isinstance(c, dict) else None

        candidate_blog_ids = list(candidate_blog_ids) if candidate_blog_ids else []
        recommended_set: Set[str] = {str(x).strip() for x in recommended_blog_ids if x}

        # 两路候选：candidate_pool 侧至多 interest_k 条；current_feed 侧至多 target_k 条（与 limit 解耦，供 LLM 全排序后再截断前 limit 条）。
        interest_k = 20
        target_k = 1

        # --- candidate_pool：来自 profile 侧候选 id，在 interest_pool 或 contents 中可解析且未在已推荐集合中 ---
        ids_for_user: List[str] = []
        if candidate_blog_ids:
            ids_for_user = [str(x).strip() for x in candidate_blog_ids if x]
        ids_for_user = [
            bid for bid in ids_for_user
            if _resolve_blog(bid) is not None
            and bid not in recommended_set
        ]
        random.shuffle(ids_for_user)
        pool_sample: List[str] = ids_for_user[:interest_k]
        pool_set: Set[str] = set(pool_sample)

        # --- current_feed：contents 全量中未推荐的微博，与候选池去重后至多 target_k 条 ---
        feed_candidates: List[str] = []
        for bid, blog in contents.items():
            sid = str(bid).strip()
            if sid in recommended_set:
                continue
            if not isinstance(blog, dict):
                continue
            feed_candidates.append(sid)
        seen: Set[str] = set(pool_sample)
        feed_eligible = [sid for sid in feed_candidates if sid not in seen]
        random.shuffle(feed_eligible)
        feed_sample: List[str] = feed_eligible[:target_k]

        pool_ids: List[str] = list(pool_sample) + feed_sample
        if not pool_ids:
            return {}
        # 仅在大模型输入前重排 observation 中的顺序，不在模型输出后改序（可用 ONESIM_REC_INTERLEAVE_INPUT=0 关闭）
        if os.environ.get("ONESIM_REC_INTERLEAVE_INPUT", "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        ):
            pool_ids = self._interest_interleave_pool_for_prompt(pool_ids, pool_set)

        def _make_candidate_payload(
            ids: List[str], desc_max: int
        ) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for blog_id in ids:
                blog = _resolve_blog(blog_id)
                if not isinstance(blog, dict):
                    continue
                popularity = self._calculate_popularity(blog)
                out.append({
                    "blog_id": blog_id,
                    "content": (blog.get("content", "") or "")[:desc_max],
                })
            return out

        # instruction + observation 总长度上限，避免超出模型上下文。
        # 默认按 Qwen2.5-7B 等约 32K token 上下文粗算（中文约 1.5–2 字/token，本段为主输入时约 4.5–5 万字符量级仍较安全；若同一请求还叠了很长 system/记忆请调低或设环境变量）。
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
        你是推荐系统助手。请对下方「候选 blog_id 全集」输出一条**完整排序列表**（从高到低），并严格返回 JSON。

        【目标 — 与用户兴趣对齐】
        1) **主排序信号（权重最高）**：对照用户画像中的 **interest_tags、description、historical_summary**，估计每条微博与该用户在**主题域、身份角色、长期关切、表达习惯**上的一致程度；越像「该用户会点进、会停留、会评论/转发」的内容越靠前。
        2) **语义与标签匹配**：在标题、摘要（desc 截断）、tags_list 中，与用户显式兴趣标签或摘要中反复出现的关切**直接命中或强相关**的微博，优先于仅有弱联想的内容。
        3) **次排序信号（主信号接近时的 tie-break）**：信息是否完整可读（标题+摘要能判断主题，非空泛标题党）、标签是否与正文主题一致；同等相关下，**更具体、更有信息增量**的微博优先于空洞或重复套话的微博。
        4) **多样性轻约束**：主信号已排序后，若多条微博主题极度雷同，可适当错开相邻位次，使前段列表覆盖略广的兴趣面（仍须满足下方「覆盖全集、不丢 id」的硬约束）。

        请严格返回 JSON：
        {{
        "ranked_blog_ids": ["id1", "id2", ...],
        "per_blog_category": ["类别归纳1", "类别归纳2", ...]
        }}

        字段说明：
        - per_blog_category：与 ranked_blog_ids **等长**；第 i 项是对排序中**第 i 条**微博的**类别归纳**（2～12 字为宜），如「学术/投稿」「生活/美食」「情感/成长」等，体现主题域或意图；有用户画像时，可侧写「与用户画像中哪类兴趣更贴近」，勿写成长句理由。
        约束：
        1) 每个 id 必须来自下方候选列表中的 blog_id，且**覆盖全集**；
        2) 不要重复；
        3) per_blog_category 条数必须与 ranked_blog_ids 一致且顺序一一对应；
        """
        desc_max = 800
        candidate_payload: List[Dict[str, Any]] = []
        observation = ""
        total_len = 0
        for _ in range(512):
            candidate_payload = _make_candidate_payload(pool_ids, desc_max)
            pool_ids_json = json.dumps(pool_ids, ensure_ascii=False)
            observation = (
                f"{user_profile_prefix}"
                f"候选 blog_id 全集（须只在此集合内排序）：{pool_ids_json}\n\n"
                f"候选微博详情：\n{json.dumps(candidate_payload, ensure_ascii=False)}"
            )
            total_len = len(instruction) + len(observation)
            if total_len <= max_prompt_chars:
                break
            if len(pool_ids) > 1:
                pool_ids.pop()
                logger.warning(
                    f"Interest recommendation prompt over budget ({total_len} > {max_prompt_chars} chars): "
                    f"dropped last candidate, {len(pool_ids)} left"
                )
                continue
            if desc_max > 200:
                desc_max = max(200, desc_max // 2)
                logger.warning(
                    f"Interest recommendation prompt over budget: shrinking desc cap to {desc_max} chars"
                )
                continue
            if desc_max > 80:
                desc_max = 80
                logger.warning(
                    "Interest recommendation prompt over budget: shrinking desc cap to 80 chars"
                )
                continue
            logger.error(
                f"Interest recommendation: prompt still {total_len} chars "
                f"(limit {max_prompt_chars}); proceeding anyway"
            )
            break

        contents_d: Dict[str, Dict[str, Any]] = (
            contents if isinstance(contents, dict) else {}
        )

        ordered_ids: List[str] = []
        pool_set_all: Set[str] = set(pool_ids)
        try:
            response = await self.generate_recommendation(instruction, observation)

            # 顺序与模型 ranked_blog_ids 一致；不要求同时出现在兴趣池与 current_blogs
            raw_ids = response.get("ranked_blog_ids")
            if raw_ids is None:
                raw_ids = response.get("selected_blog_ids", [])
            if isinstance(raw_ids, list):
                for blog_id in raw_ids:
                    blog_id = str(blog_id).strip()
                    if (
                        blog_id
                        and blog_id in pool_set_all
                        and _resolve_blog(blog_id) is not None
                        and blog_id not in recommended_set
                        and blog_id not in ordered_ids
                    ):
                        ordered_ids.append(blog_id)

        except Exception as e:
            logger.warning(f"LLM interest recommendation failed: {e}")
            return {}

        if not ordered_ids:
            return {}

        top_ids = ordered_ids[: max(0, int(limit))]

        def _blog_from_contents_only(sid: str) -> Optional[Dict[str, Any]]:
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
            n = _blog_from_contents_only(sid)
            if n is not None:
                out[sid] = n
        return out

    async def send_algorithm_recommendations(self, event: Event) -> List[Event]:
        """
        发送算法推荐
        """
        # 获取当前时间步
        current_timestamp = event.timestamp

        # 获取内容池
        logger.info(f"RecommenderAgent {self.profile_id} start getting previous and current blogs")

        current_blogs = {}
        if self.profile is not None:
            current_blogs = self.profile.get_data("current_blogs", {})
        if not isinstance(current_blogs, dict) or not current_blogs:
            current_blogs = getattr(event, "current_blogs", {})
        if not isinstance(current_blogs, dict) or not current_blogs:
            logger.warning("No current blogs found from profile/event")
            return []

        # 校验：本推荐器 profile 的 type 须与请求事件携带的 type 一致（用户侧期望的算法类型）
        expected_type = getattr(event, "type", None)
        type_value = await self.get_data("type", "")
        exp_s = str(expected_type).strip() if expected_type is not None else ""
        got_s = str(type_value).strip() if type_value is not None else ""
        if exp_s != got_s:
            # 旧事件未带 algorithm_type 时 event.type 为空，仅信任本推荐器 profile.type
            if not exp_s and got_s:
                logger.debug(
                    f"RecommenderAgent {self.profile_id}: event.type 为空，按本智能体 type={got_s!r} 处理"
                )
            else:
                raise ValueError(
                    f"RecommenderAgent {self.profile_id}: 推荐算法 type 不一致 — "
                    f"本智能体 get_data('type')={got_s!r}, 事件 event.type={exp_s!r}, "
                    f"from_agent_id={getattr(event, 'from_agent_id', '')!r}"
                )

        # 将type字段值映射到算法名称
        algorithm_name = self.type_to_algorithm.get(type_value, self.default_algorithm)
        if algorithm_name not in self.recommendation_algorithms:
            algorithm_name = self.default_algorithm
            logger.warning(f"Unknown recommendation algorithm type '{type_value}', using default: {self.default_algorithm}")
        
        # 获取推荐算法函数
        recommendation_func = self.recommendation_algorithms[algorithm_name]

        # 伯努利衰减采样参数：
        # - alpha：优先取每个 user_agent 的 activity_level（0~1），找不到则退化到全局默认
        # - max_limit：最多推荐多少条（默认 3，对应你说的 1~3 区间）
        if isinstance(event, SearchEvent):
            max_limit = 5
        else:
            # 与 profile 中未写 limit 时的预期一致（weibo 推荐器普遍 limit=3）；Search 仍走上面的 5
            max_limit = await self.get_data("limit", 15)
            try:
                max_limit = max(1, min(15, int(float(max_limit))))
            except (TypeError, ValueError):
                max_limit = 15

        # alpha = await self.get_data("alpha", 0.5)
        alpha = 0.2
        
        # 所有用户共用同一个 alpha -> 推荐数量服从同一分布
        recommendation_limit = self._sample_recommendation_limit_bernoulli(alpha, max_limit=max_limit)
        logger.info(f"RecommenderAgent {self.profile_id} recommendation_limit: {recommendation_limit}")

        # 调用推荐算法获取推荐内容
        by_user: Dict[str, Any] = {}
        if self.profile is not None:
            raw_map = self.profile.get_data("candidate_blog_ids_by_user", {})
            if isinstance(raw_map, dict):
                by_user = raw_map

        requester_id = str(getattr(event, "from_agent_id", "") or "").strip()
        candidate_blog_ids: List[str] = []
        if requester_id:
            raw_list = by_user.get(requester_id)
            if raw_list is None:
                for k, v in by_user.items():
                    if str(k).strip() == requester_id and isinstance(v, list):
                        raw_list = v
                        break
            if isinstance(raw_list, list):
                candidate_blog_ids = [str(x).strip() for x in raw_list if x]
        recommended_blog_ids = list(event.recommended_blog_ids)

        # 推荐给算法的候选池：仅原创帖（去掉转发），避免以转发帖为推荐基底；完整池仍用于下方 reposted_blog 回填
        n_blogs = len(current_blogs)
        current_blogs_for_algorithm: Dict[str, Any] = {
            bid: b
            for bid, b in current_blogs.items()
            if self._is_original_weibo(b)
        }
        if len(current_blogs_for_algorithm) < n_blogs:
            logger.info(
                f"RecommenderAgent {self.profile_id}: current_blogs 仅保留原创推文：{n_blogs} -> {len(current_blogs_for_algorithm)}"
            )

        user_profile_for_rec: Optional[Dict[str, Any]] = None
        raw_up = getattr(event, "user_profile", None)
        if isinstance(raw_up, dict):
            user_profile_for_rec = raw_up

        if algorithm_name == "interest":
            recommended_contents = await recommendation_func(
                recommended_blog_ids,
                current_blogs_for_algorithm,
                current_timestamp,
                recommendation_limit,
                candidate_blog_ids=candidate_blog_ids,
                user_profile=user_profile_for_rec,
            )
        else:
            recommended_contents = recommendation_func(
                recommended_blog_ids,
                current_blogs_for_algorithm,
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

        # 补充reposted_blog字段
        for blog_id, blog in recommended_contents.items():
            reposted_blog_id = blog.get("reposted_blog_id", "")
            if reposted_blog_id and reposted_blog_id in current_blogs:
                blog["reposted_blog"] = current_blogs[reposted_blog_id]

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

    @staticmethod
    def _search_blog_ids(current_blogs: Dict[str, Dict[str, Any]]) -> List[str]:
        """内容池中可作为搜索候选的 blog_id（值为 dict 的条目）。"""
        out: List[str] = []
        for bid, blog in (current_blogs or {}).items():
            if not isinstance(blog, dict):
                continue
            sid = str(bid).strip()
            if sid:
                out.append(sid)
        return out

    @staticmethod
    def _blog_text_for_search(blog: Dict[str, Any], max_len: int = 1200) -> str:
        """拼接标题、正文与标签为检索用文本，并截断到 max_len。"""
        content = str(blog.get("content", "") or "")
        return content if len(content) <= max_len else content[:max_len]

    def _derive_search_query(self, user_profile: Dict[str, Any], search_query: str) -> str:
        """LLM 搜索用的查询串：优先用户显式关键词，否则从画像兴趣/简介/历史摘要拼接（摘要过长时头尾各 50 字）。"""
        q = (search_query or "").strip()
        if q:
            return q[:100]
        parts: List[str] = []
        if isinstance(user_profile, dict):
            for k in ("interest_tags", "description", "historical_summary"):
                v = user_profile.get(k)
                if v is None:
                    continue
                if k == "historical_summary":
                    parts.append(self._historical_summary_head_tail(v))
                else:
                    parts.append(str(v).strip())
        joined = "\n".join(parts).strip()
        return joined[:4000]

    def _random_search(
        self,
        current_blogs: Dict[str, Dict[str, Any]],
        current_timestamp: float,
        limit: int,
    ) -> Dict[str, Dict[str, Any]]:
        """随机搜索：在候选笔记中打乱顺序后取前 limit 条（与热度、语义无关）。"""
        if not current_blogs or limit <= 0:
            return {}
        blog_ids = list(current_blogs.keys())
        random.shuffle(blog_ids)
        picked = blog_ids[:limit]
        return {bid: current_blogs[bid] for bid in picked if bid in current_blogs}

    def _hot_search(
        self,
        current_blogs: Dict[str, Dict[str, Any]],
        current_timestamp: float,
        limit: int,
    ) -> Dict[str, Dict[str, Any]]:
        """热度搜索：按 `_calculate_popularity` 得分降序取前 limit 条；同分顺序经 shuffle 后再 sort 打散。"""
        if not current_blogs or limit <= 0:
            return {}
        rows: List[Tuple[str, Dict[str, Any], float]] = []
        for bid in current_blogs.keys():
            blog = current_blogs.get(bid)
            if not isinstance(blog, dict):
                continue
            pop = self._calculate_popularity(blog)
            rows.append((bid, blog, pop))
        random.shuffle(rows)
        rows.sort(key=lambda x: x[2], reverse=True)
        out: Dict[str, Dict[str, Any]] = {}
        for bid, blog, _ in rows[:limit]:
            out[bid] = blog
        return out

    def _relevant_search_sync(
        self,
        current_blogs: Dict[str, Dict[str, Any]],
        current_timestamp: float,
        limit: int,
        *,
        search_query: str,
    ) -> Dict[str, Dict[str, Any]]:
        """相关搜索（同步）：查询与笔记分别 embedding，按余弦相似度排序；失败或空查询则走关键词 Jaccard 或退化为热度。"""
        if not current_blogs or limit <= 0:
            return {}
        cands = self._search_blog_ids(current_blogs)
        if not cands:
            return {}

        query = (search_query or "").strip()[:4000]
        if not query.strip():
            sub_blogs = {bid: current_blogs[bid] for bid in cands if bid in current_blogs}
            return self._hot_search(sub_blogs, current_timestamp, limit)

        def _keyword_scores() -> List[Tuple[str, float]]:
            qt = _tokenize(query)
            scored: List[Tuple[str, float]] = []
            for bid in cands:
                blog = current_blogs.get(bid)
                if not isinstance(blog, dict):
                    continue
                text = self._blog_text_for_search(blog, max_len=2000)
                nt = _tokenize(text)
                if not qt or not nt:
                    scored.append((bid, 0.0))
                    continue
                inter = len(qt & nt)
                union = len(qt | nt)
                j = (inter / union) if union else 0.0
                scored.append((bid, j))
            random.shuffle(scored)
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored

        try:
            cfg = load_step15_gate_config()
            base_url, model_name = load_embedding_config(cfg.embedding_config_path)
            texts = [query[:2000]]
            for bid in cands:
                blog = current_blogs.get(bid)
                texts.append(
                    self._blog_text_for_search(blog, max_len=800)
                    if isinstance(blog, dict)
                    else ""
                )
            vecs = get_embeddings(base_url, model_name, texts)
            if not vecs or len(vecs) != len(texts):
                raise ValueError("embedding 返回长度与输入不一致")
            vq = vecs[0]
            scored: List[Tuple[str, float]] = []
            for i, bid in enumerate(cands):
                sim = float(cosine_similarity(vq, vecs[i + 1]))
                scored.append((bid, sim))
            random.shuffle(scored)
            scored.sort(key=lambda x: x[1], reverse=True)
            top = scored[:limit]
            return {bid: current_blogs[bid] for bid, _ in top if bid in current_blogs}
        except Exception as e:
            logger.warning(f"Relevant search embedding failed, fallback to keyword: {e}")
            kw = _keyword_scores()
            return {
                bid: current_blogs[bid]
                for bid, _ in kw[:limit]
                if bid in current_blogs
            }

    async def _relevant_search(
        self,
        current_blogs: Dict[str, Dict[str, Any]],
        current_timestamp: float,
        limit: int,
        *,
        search_query: str,
    ) -> Dict[str, Dict[str, Any]]:
        """相关搜索（异步）：在线程池中执行 `_relevant_search_sync`，避免阻塞事件循环。"""
        return await asyncio.to_thread(
            self._relevant_search_sync,
            current_blogs,
            current_timestamp,
            limit,
            search_query=search_query,
        )

    async def _llm_search(
        self,
        current_blogs: Dict[str, Dict[str, Any]],
        current_timestamp: float,
        limit: int,
        *,
        user_profile: Dict[str, Any],
        search_query: str,
    ) -> Dict[str, Dict[str, Any]]:
        """LLM 搜索：将画像、查询意图与最多 300 条候选摘要交给模型，解析 `ranked_blog_ids` 取前 limit 条。"""
        if not current_blogs or limit <= 0:
            return {}
        cands = self._search_blog_ids(current_blogs)
        if not cands:
            return {}
        query = self._derive_search_query(user_profile, search_query)

        compact: List[Dict[str, Any]] = []
        for bid in cands[:300]:
            n = current_blogs.get(bid)
            if not isinstance(n, dict):
                continue
            compact.append({
                "blog_id": bid,
                "content": str(n.get("content", "") or ""),
            })

        profile_compact: Dict[str, Any] = {}
        if isinstance(user_profile, dict):
            for k in (
                "id",
                "nickname",
                "interest_tags",
                "description",
                "historical_summary",
            ):
                if k in user_profile:
                    val = user_profile.get(k)
                    if k == "historical_summary":
                        val = self._historical_summary_head_tail(val)
                    profile_compact[k] = val

        instruction = """你是搜索排序助手。根据「用户画像」「搜索关键词/意图」与候选微博列表，输出按**与本次搜索相关性**从高到低排序的 blog_id 列表（JSON）。
        只考虑语义相关性与用户意图匹配，不要编造列表外的 id。
        请严格返回 JSON：
        {
        "ranked_blog_ids": ["id1", "id2", ...]
        }
        约束：ranked_blog_ids 中的每个 id 必须来自候选中的 blog_id，不重复，不要求覆盖全部候选。"""

        observation = (
            f"搜索关键词/意图（若为空则结合画像理解）：\n{query or '(空，请结合用户画像推断可能兴趣)'}\n\n"
            f"用户画像（节选）：\n{json.dumps(profile_compact, ensure_ascii=False)}\n\n"
            f"候选微博：\n{json.dumps(compact, ensure_ascii=False)}"
        )

        pool_set = set(cands)
        ordered: List[str] = []
        try:
            response = await self.generate_recommendation(instruction, observation)
            raw_ids = response.get("ranked_blog_ids")
            if raw_ids is None:
                raw_ids = response.get("selected_blog_ids", [])
            if isinstance(raw_ids, list):
                for blog_id in raw_ids:
                    sid = str(blog_id).strip()
                    if sid and sid in pool_set and sid not in ordered:
                        ordered.append(sid)
        except Exception as e:
            logger.warning(f"LLM search ranking failed: {e}")
            return {}

        if not ordered:
            return {}
        top_ids = ordered[:limit]
        return {bid: current_blogs[bid] for bid in top_ids if bid in current_blogs}
  
    async def send_search_recommendations(self, event: Event) -> List[Event]:
        """
        处理 SearchEvent：按 profile.type 映射到 random / hot / relevant / llm 搜索，将结果封装为 AlgorithmRecommendationEvent 返回。
        """
        # 获取当前时间步
        current_timestamp = event.timestamp

        # 获取内容池
        logger.info(f"RecommenderAgent {self.profile_id} start getting previous and current blogs")
        
        current_blogs = {}
        if self.profile is not None:
            current_blogs = self.profile.get_data("current_blogs", {})
        if not isinstance(current_blogs, dict) or not current_blogs:
            current_blogs = getattr(event, "current_blogs", {})
        if not isinstance(current_blogs, dict) or not current_blogs:
            logger.warning("No current blogs found from profile/event")
            return []

        # 校验：本推荐器 profile 的 type 须与请求事件携带的 type 一致（用户侧期望的算法类型）
        expected_type = getattr(event, "type", None)
        type_value = await self.get_data("type", "")
        exp_s = str(expected_type).strip() if expected_type is not None else ""
        got_s = str(type_value).strip() if type_value is not None else ""
        if exp_s != got_s:
            # 旧事件未带 algorithm_type 时 event.type 为空，仅信任本推荐器 profile.type
            if not exp_s and got_s:
                logger.debug(
                    f"RecommenderAgent {self.profile_id}: event.type 为空，按本智能体 type={got_s!r} 处理"
                )
            else:
                raise ValueError(
                    f"RecommenderAgent {self.profile_id}: 推荐算法 type 不一致 — "
                    f"本智能体 get_data('type')={got_s!r}, 事件 event.type={exp_s!r}, "
                    f"from_agent_id={getattr(event, 'from_agent_id', '')!r}"
                )

        # 将 type 字段值映射到搜索算法名称（与推荐算法的 type 字符串不同，如 LLM Search）
        algorithm_name = self.type_to_search.get(type_value, self.default_search)
        if algorithm_name not in self.search_algorithms:
            algorithm_name = self.default_search
            logger.warning(
                f"Unknown search algorithm type '{type_value}', using default: {self.default_search}"
            )

        search_func = self.search_algorithms[algorithm_name]

        max_limit = await self.get_data("limit", 15)
        try:
            max_limit = max(1, min(50, int(float(max_limit))))
        except (TypeError, ValueError):
            max_limit = 15

        alpha = 0.5
        recommendation_limit = self._sample_recommendation_limit_bernoulli(alpha, max_limit=max_limit)
        top_k = max(1, min(max_limit, recommendation_limit))
        logger.info(
            f"RecommenderAgent {self.profile_id} search algorithm={algorithm_name}, "
            f"max_limit={max_limit}, sampled top_k={top_k}"
        )

        search_query = str(getattr(event, "search_query", "") or "")

        if algorithm_name == "llm":
            user_profile_raw = getattr(event, "user_profile", None)
            user_profile: Dict[str, Any] = (
                user_profile_raw if isinstance(user_profile_raw, dict) else {}
            )
            search_results = await search_func(
                current_blogs,
                current_timestamp,
                top_k,
                user_profile=user_profile,
                search_query=search_query,
            )
        elif algorithm_name == "relevant":
            search_results = await search_func(
                current_blogs,
                current_timestamp,
                top_k,
                search_query=search_query,
            )
        else:
            search_results = search_func(
                current_blogs,
                current_timestamp,
                top_k,
            )

        if not search_results:
            logger.debug(f"No search results for user {getattr(event, 'from_agent_id', 'unknown')}")
            return []

        logger.info(
            f"Sending SearchRecommendationEvent (search) to UserAgent {event.from_agent_id}, "
            f"search results: {len(search_results)}"
        )

        # 补充reposted_blog字段
        for blog_id, blog in recommended_contents.items():
            reposted_blog_id = blog.get("reposted_blog_id", "")
            if reposted_blog_id and reposted_blog_id in current_blogs:
                blog["reposted_blog"] = current_blogs[reposted_blog_id]

        search_event = SearchRecommendationEvent(
            from_agent_id=self.profile_id,
            to_agent_id=event.from_agent_id,
            timestamp=current_timestamp,
            timestamp_duration=int(getattr(event, "timestamp_duration", 0) or 0),
            current_step=getattr(event, "current_step", 1),
            max_step=getattr(event, "max_step", 8),
            recommendations=dict(search_results),
        )
        return [search_event]