from typing import Any, List, Optional, Callable, Dict, Set, Tuple
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
from .metrics.twitter_similarity import (
    cosine_similarity,
    get_embeddings,
    load_embedding_config,
)
from .step15_topic_gate import _tokenize, load_step15_gate_config

# 默认传播数分布 CSV（quote+retweet+reply 合计 → 推文数）
_TWITTER_DEFAULT_PROPAGATION_DIST_CSV = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "..",
        "datasets",
        "twitter-openreview",
        "propagation_distribution.csv",
    )
)
class RecommenderAgent(GeneralAgent):
    """推荐算法智能体：支持随机推荐和热点推荐"""
    
    # 随机推荐：对每条候选帖以 p=1/(MAU×月均发帖) 做伯努利试验，抽中则收录直至 limit；见 _random_recommendation。
    # 另：w=(1/MAU)×avg_posts 仍可用于其它说明，与 p 数值不同（p=1/分母，w=avg/MAU）。
    # 月活、月均发帖量仅来自 profile：random_rec_mau、random_rec_avg_posts_per_user_month；不读环境变量。
    # 未配置或非法时回退到下方类常量（非环境变量）。
    _RANDOM_REC_MAU_DEFAULT = 557_000_000
    _RANDOM_REC_AVG_POSTS_PER_USER_MONTH_DEFAULT = 74.31

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

    def _random_rec_per_tweet_weight(self) -> float:
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
        self.register_event("StartEvent", "update_current_tweets")
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
        self._propagation_count_dist_7d_cache: Optional[Dict[int, int]] = None

    @staticmethod
    def _is_original_tweet(tweet: Dict[str, Any]) -> bool:
        """无 retweeted_tweet_id 或 replied_tweet_id 为空字符串（非转发链上的节点）。"""
        if not isinstance(tweet, dict):
            return False
        rid = tweet.get("retweeted_tweet_id") or tweet.get("replied_tweet_id")
        if rid is None:
            return True
        return str(rid).strip() == ""

    def _parse_propagation_count_dist_rows(self, rows: Any) -> Dict[int, int]:
        """将 profile 中的列表解析为 propagation_count -> post_count（推文数）。"""
        out: Dict[int, int] = {}
        if not isinstance(rows, list):
            return out
        for row in rows:
            if not isinstance(row, dict):
                continue
            raw_pc = row.get("propagation_count")
            raw_cnt = row.get("post_count")
            if raw_pc is None or raw_cnt is None:
                continue
            try:
                pc = int(float(str(raw_pc).strip()))
                cnt = int(float(str(raw_cnt).strip()))
            except (TypeError, ValueError):
                continue
            if cnt <= 0:
                continue
            out[pc] = out.get(pc, 0) + cnt
        return out

    def _read_propagation_count_dist_csv(self, path: str) -> Dict[int, int]:
        out: Dict[int, int] = {}
        if not path or not os.path.isfile(path):
            return out
        try:
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not row:
                        continue
                    raw_prc = row.get("propagation_count")
                    raw_pc = row.get("post_count")
                    if raw_prc is None or raw_pc is None:
                        continue
                    try:
                        prc = int(float(str(raw_prc).strip()))
                        pc = int(float(str(raw_pc).strip()))
                    except (TypeError, ValueError):
                        continue
                    if pc <= 0:
                        continue
                    out[prc] = out.get(prc, 0) + pc
        except OSError as e:
            logger.warning(f"Failed to read propagation dist CSV {path}: {e}")
        return out

    def _get_propagation_count_dist_7d_map(self) -> Dict[int, int]:
        """
        传播数（quote+retweet+reply）分布，与站外分布一起参与热点桶排序（同 weibo 的 repost_count_dist_7d）。
        优先 profile.propagation_count_dist_7d；为空则读 CSV（profile.propagation_count_dist_7d_csv 或默认路径）。
        """
        if self._propagation_count_dist_7d_cache is not None:
            return self._propagation_count_dist_7d_cache

        merged: Dict[int, int] = {}
        if self.profile is not None:
            raw = self.profile.get_data("propagation_count_dist_7d", [])
            merged.update(self._parse_propagation_count_dist_rows(raw))
        if not merged:
            csv_path = _TWITTER_DEFAULT_PROPAGATION_DIST_CSV
            if self.profile is not None:
                raw_path = self.profile.get_data("propagation_count_dist_7d_csv", None)
                if isinstance(raw_path, str) and raw_path.strip():
                    csv_path = os.path.abspath(os.path.expanduser(raw_path.strip()))
            merged = self._read_propagation_count_dist_csv(csv_path)
        self._propagation_count_dist_7d_cache = merged
        return merged

    async def update_current_tweets(self, event: Event) -> None:
        """
        更新当前推文
        """
        current_tweets = event.current_tweets
        if not isinstance(current_tweets, dict) or not current_tweets:
            logger.warning("No current tweets found")
        self.profile.update_data("current_tweets", current_tweets)

    def _calculate_popularity(self, tweet: Dict[str, Any]) -> float:
        """计算内容热度：热度 = 回复数 + 转推数 + 引用数"""
        reply_count = tweet.get("reply_count", 0)
        retweet_count = tweet.get("retweet_count", 0)
        quote_count = tweet.get("quote_count", 0)
        popularity = float(reply_count) + float(retweet_count) + float(quote_count)
        return popularity

    # @staticmethod
    # def _retweeted_tweet_id_empty(tweet: Dict[str, Any]) -> bool:
    #     """原创推：retweeted_tweet_id 缺失、空串或仅空白。"""
    #     rid = tweet.get("retweeted_tweet_id")
    #     if rid is None:
    #         return True
    #     if isinstance(rid, str):
    #         return rid.strip() == ""
    #     return False

    @staticmethod
    def _tweet_ref_key(ref: Any) -> Optional[str]:
        if ref is None:
            return None
        if isinstance(ref, str):
            s = ref.strip()
            return s if s else None
        try:
            return str(ref)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _enrich_tweet_quote_reply_chain(
        tweet: Dict[str, Any],
        current_tweets: Dict[str, Any],
        *,
        tweet_ref: Optional[str] = None,
        max_depth: int = 48,
        _depth: int = 0,
        _visited: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """
        从 current_tweets 递归挂上 quoted_tweet / replied_tweet（与 UserAgent 逻辑一致）。
        """
        if _visited is None:
            _visited = set()
        key = tweet_ref or RecommenderAgent._tweet_ref_key(tweet.get("id")) or RecommenderAgent._tweet_ref_key(tweet.get("tweet_id"))
        if key is not None:
            if key in _visited:
                return dict(tweet)
            _visited.add(key)
        if _depth >= max_depth:
            return dict(tweet)

        out = dict(tweet)
        qid = RecommenderAgent._tweet_ref_key(out.get("quoted_tweet_id"))
        rid = RecommenderAgent._tweet_ref_key(out.get("replied_tweet_id"))

        if qid and qid in current_tweets:
            nested = current_tweets[qid]
            if isinstance(nested, dict):
                out["quoted_tweet"] = RecommenderAgent._enrich_tweet_quote_reply_chain(
                    nested,
                    current_tweets,
                    tweet_ref=qid,
                    max_depth=max_depth,
                    _depth=_depth + 1,
                    _visited=_visited,
                )
        if rid and rid in current_tweets:
            nested = current_tweets[rid]
            if isinstance(nested, dict):
                out["replied_tweet"] = RecommenderAgent._enrich_tweet_quote_reply_chain(
                    nested,
                    current_tweets,
                    tweet_ref=rid,
                    max_depth=max_depth,
                    _depth=_depth + 1,
                    _visited=_visited,
                )
        return out

    def _sample_recommendation_limit_bernoulli(self, alpha: float, max_limit: int = 3) -> int:
        """
        使用“继续概率为 alpha”的伯努利衰减，采样本轮推荐数量。

        建模含义：每次已经推荐 1 屏后，用户以概率 alpha 决定继续往下看下一屏；
        连续继续下去的次数决定最终推荐数量（上限为 max_limit）。

        例如 max_limit=3 时：
          P(1) = 1 - alpha
          P(2) = alpha 3 (1 - alpha)
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

    def _random_recommendation(self, recommended_tweet_ids: List[str], contents: Dict[str, Dict[str, Any]], current_timestamp: float, limit: int = 10) -> Dict[str, Dict[str, Any]]:
        """随机推荐算法"""
        if not contents:
            return {}
        
        # 从字典中随机选择 tweet_id
        tweet_ids = list(contents.keys())
        if not tweet_ids:
            return {}
        
        tweet_ids = [tweet_id for tweet_id in tweet_ids if tweet_id not in recommended_tweet_ids]
        if not tweet_ids:
            return {}
        
        mau, avg_posts = self._random_rec_mau_avg_posts()
        denom = mau * avg_posts
        if denom <= 0:
            p_hit = 0.0
        else:
            p_hit = min(1.0, 1.0 / denom)

        random.shuffle(tweet_ids)
        log_each = bool(
            self.profile is not None
            and self.profile.get_data("random_rec_log_per_tweet_prob_check", False)
        )

        selected: List[str] = []
        for tid in tweet_ids:
            if len(selected) >= limit:
                break
            ok = random.random() < p_hit
            if log_each:
                logger.debug(
                    "random_rec 帖 {}: Bernoulli(p=1/(MAU×avg)={:.6e}) => {}",
                    tid,
                    p_hit,
                    "hit" if ok else "miss",
                )
            if ok:
                selected.append(tid)

        w = self._random_rec_per_tweet_weight()
        logger.debug(
            "random recommendation: MAU={:.4g} avg_posts={:.6g} => p=1/(MAU×avg)={:.6e}; "
            "候选 N={} 上限 limit={} => 本轮收录 {} 条；w=(1/MAU)×avg={:.6e}",
            mau,
            avg_posts,
            p_hit,
            len(tweet_ids),
            limit,
            len(selected),
            w,
        )
        return {tweet_id: contents[tweet_id] for tweet_id in selected}
    
    def _hot_recommendation(self, recommended_tweet_ids: List[str], contents: Dict[str, Dict[str, Any]], current_timestamp: float, limit: int = 10) -> Dict[str, Dict[str, Any]]:
        """热点推荐算法：从一周内的内容中选择热度最高的（热度=回复数 + 转推数 + 引用数）"""
        if not contents:
            return {}
        
        # 计算一周前的时间戳（7天 = 7 * 24 * 3600 * 1000 毫秒）
        one_week_ago = current_timestamp - (7 * 24 * 3600 * 1000)
        recent_contents = {}
        for tweet_id, tweet in contents.items():
            if tweet_id in recommended_tweet_ids or not isinstance(tweet, dict):
                continue
            raw_time = tweet.get("time", 0)
            try:
                tweet_time = float(raw_time)
            except (TypeError, ValueError):
                continue
            if tweet_time >= one_week_ago:
                recent_contents[tweet_id] = tweet
        
        if not recent_contents:
            return {}

        dist_map = self._get_propagation_count_dist_7d_map()
        if not dist_map:
            tweets_with_popularity: List[Tuple[str, Dict[str, Any], float]] = []
            for tweet_id, tweet in recent_contents.items():
                popularity = self._calculate_popularity(tweet)
                tweets_with_popularity.append((tweet_id, tweet, popularity))
            random.shuffle(tweets_with_popularity)
            tweets_with_popularity.sort(key=lambda x: x[2], reverse=True)
            return {tweet_id: tweet for tweet_id, tweet, _ in tweets_with_popularity[:limit]}

        # 同一传播热度桶：虚拟位（None）与真实 tweet_id 一起 shuffle，再按热度降序串联（同 weibo 热点逻辑）
        buckets: Dict[int, List[Optional[str]]] = defaultdict(list)
        for pc, cnt in dist_map.items():
            try:
                n = int(cnt)
            except (TypeError, ValueError):
                continue
            for _ in range(max(0, n)):
                buckets[pc].append(None)
        for tweet_id, tweet in recent_contents.items():
            p = int(self._calculate_popularity(tweet))
            buckets[p].append(tweet_id)

        merged_ids: List[Optional[str]] = []
        for pc in sorted(buckets.keys(), reverse=True):
            row = buckets[pc]
            random.shuffle(row)
            merged_ids.extend(row)

        top_slice = merged_ids[: max(0, limit)]
        chosen_ids = [x for x in top_slice if x is not None]
        return {tid: recent_contents[tid] for tid in chosen_ids if tid in recent_contents}
    
    def _top_hot_tweets(
        self,
        recommended_tweet_ids: List[str],
        contents: Dict[str, Dict[str, Any]],
        top_k: int = 100,
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """取热度最高的 top_k 条（过滤掉已推荐过的）。"""
        if not contents:
            return []
        rows: List[Tuple[str, Dict[str, Any], float]] = []
        for tweet_id, tweet in contents.items():
            if tweet_id in recommended_tweet_ids or not isinstance(tweet, dict):
                continue
            popularity = self._calculate_popularity(tweet)
            rows.append((tweet_id, tweet, popularity))
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
            "interest_tags",
            "description",
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
                v = v[:50] + "…"
            out[k] = v
        return out

    async def _interest_recommendation(
        self,
        recommended_tweet_ids: List[str],
        contents: Dict[str, Dict[str, Any]],
        current_timestamp: float,
        limit: int = 10,
        candidate_tweet_ids: Optional[List[str]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        兴趣推荐（LLM 重排序）：两路候选合并后一次性排序。
        按该顺序取前 ``limit`` 个 id；再**仅**保留其中在 ``contents``（current_tweets）里存在的条目并返回（只在兴趣池、不在 current_tweets 的丢弃）。
        若最终没有任何条在 current_tweets 中命中，返回空字典即可。
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

        def _resolve_tweet(tid: str) -> Optional[Dict[str, Any]]:
            sid = str(tid).strip()
            n = interest_pool.get(sid)
            if isinstance(n, dict):
                return n
            c = contents.get(sid) if isinstance(contents, dict) else None
            return c if isinstance(c, dict) else None

        candidate_tweet_ids = list(candidate_tweet_ids) if candidate_tweet_ids else []
        recommended_set: Set[str] = {str(x).strip() for x in recommended_tweet_ids if x}

        # 两路候选：candidate_pool 侧至多 interest_k 条；current_feed 侧至多 target_k 条（与 limit 解耦，供 LLM 全排序后再截断前 limit 条）。
        interest_k = 20
        target_k = 3

        # --- candidate_pool：来自 profile 侧候选 id，在 interest_pool 或 contents 中可解析且未在已推荐集合中 ---
        ids_for_user: List[str] = []
        if candidate_tweet_ids:
            ids_for_user = [str(x).strip() for x in candidate_tweet_ids if x]
        ids_for_user = [
            tid for tid in ids_for_user
            if _resolve_tweet(tid) is not None
            and tid not in recommended_set
        ]
        random.shuffle(ids_for_user)
        pool_sample: List[str] = ids_for_user[:interest_k]
        pool_set: Set[str] = set(pool_sample)

        # --- current_feed：contents 全量中未推荐的微博，与候选池去重后至多 target_k 条 ---
        feed_candidates: List[str] = []
        for tid, tweet in contents.items():
            sid = str(tid).strip()
            if sid in recommended_set:
                continue
            if not isinstance(tweet, dict):
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
            for tweet_id in ids:
                tweet = _resolve_tweet(tweet_id)
                if not isinstance(tweet, dict):
                    continue
                popularity = self._calculate_popularity(tweet)
                out.append({
                    "tweet_id": tweet_id,
                    "content": (tweet.get("content", "") or "")[:desc_max],
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
        你是推荐系统助手。请对下方「候选 tweet_id 全集」输出一条**完整排序列表**（从高到低），并严格返回 JSON。

        【目标 — 与用户兴趣对齐】
        1) **主排序信号（权重最高）**：对照用户画像中的 **interest_tags、description、historical_summary**，估计每条推文与该用户在**主题域、身份角色、长期关切、表达习惯**上的一致程度；越像「该用户会点进、会停留、会评论/转发」的内容越靠前。
        2) **语义与标签匹配**：在标题、摘要（desc 截断）、tags_list 中，与用户显式兴趣标签或摘要中反复出现的关切**直接命中或强相关**的推文，优先于仅有弱联想的内容。
        3) **次排序信号（主信号接近时的 tie-break）**：信息是否完整可读（标题+摘要能判断主题，非空泛标题党）、标签是否与正文主题一致；同等相关下，**更具体、更有信息增量**的推文优先于空洞或重复套话的推文。
        4) **多样性轻约束**：主信号已排序后，若多条推文主题极度雷同，可适当错开相邻位次，使前段列表覆盖略广的兴趣面（仍须满足下方「覆盖全集、不丢 id」的硬约束）。

        请严格返回 JSON：
        {{
        "ranked_tweet_ids": ["id1", "id2", ...],
        "per_tweet_category": ["类别归纳1", "类别归纳2", ...]
        }}

        字段说明：
        - per_tweet_category：与 ranked_tweet_ids **等长**；第 i 项是对排序中**第 i 条**微博的**类别归纳**（2～12 字为宜），如「学术/投稿」「生活/美食」「情感/成长」等，体现主题域或意图；有用户画像时，可侧写「与用户画像中哪类兴趣更贴近」，勿写成长句理由。
        约束：
        1) 每个 id 必须来自下方候选列表中的 tweet_id，且**覆盖全集**；
        2) 不要重复；
        3) per_tweet_category 条数必须与 ranked_tweet_ids 一致且顺序一一对应；
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
                f"候选 tweet_id 全集（须只在此集合内排序）：{pool_ids_json}\n\n"
                f"候选 tweet 详情：\n{json.dumps(candidate_payload, ensure_ascii=False)}"
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

            # 顺序与模型 ranked_tweet_ids 一致；不要求同时出现在兴趣池与 current_tweets
            raw_ids = response.get("ranked_tweet_ids")
            if raw_ids is None:
                raw_ids = response.get("selected_tweet_ids", [])
            if isinstance(raw_ids, list):
                for tweet_id in raw_ids:
                    tweet_id = str(tweet_id).strip()
                    if (
                        tweet_id
                        and tweet_id in pool_set_all
                        and _resolve_tweet(tweet_id) is not None
                        and tweet_id not in recommended_set
                        and tweet_id not in ordered_ids
                    ):
                        ordered_ids.append(tweet_id)

        except Exception as e:
            logger.warning(f"LLM interest recommendation failed: {e}")
            return {}

        if not ordered_ids:
            return {}

        top_ids = ordered_ids[: max(0, int(limit))]

        def _tweet_from_contents_only(sid: str) -> Optional[Dict[str, Any]]:
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
            n = _tweet_from_contents_only(sid)
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
        logger.info(f"RecommenderAgent {self.profile_id} start getting previous and current tweets")
        
        current_tweets = {}
        if self.profile is not None:
            current_tweets = self.profile.get_data("current_tweets", {})
        if not isinstance(current_tweets, dict) or not current_tweets:
            current_tweets = getattr(event, "current_tweets", {})
        if not isinstance(current_tweets, dict) or not current_tweets:
            logger.warning("No current tweets found from profile/event")
            return []

        # 从智能体的type字段中获取推荐算法名称
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
        alpha = 0.5
        
        # 所有用户共用同一个 alpha -> 推荐数量服从同一分布
        recommendation_limit = self._sample_recommendation_limit_bernoulli(alpha, max_limit=max_limit)
        logger.info(f"RecommenderAgent {self.profile_id} recommendation_limit: {recommendation_limit}")

        # 调用推荐算法获取推荐内容
        by_user: Dict[str, Any] = {}
        if self.profile is not None:
            raw_map = self.profile.get_data("candidate_tweet_ids_by_user", {})
            if isinstance(raw_map, dict):
                by_user = raw_map

        requester_id = str(getattr(event, "from_agent_id", "") or "").strip()
        candidate_tweet_ids: List[str] = []
        if requester_id:
            raw_list = by_user.get(requester_id)
            if raw_list is None:
                for k, v in by_user.items():
                    if str(k).strip() == requester_id and isinstance(v, list):
                        raw_list = v
                        break
            if isinstance(raw_list, list):
                candidate_tweet_ids = [str(x).strip() for x in raw_list if x]
        recommended_tweet_ids = list(event.recommended_tweet_ids)

        # 推荐给算法的候选池：仅原创帖（去掉转发），避免以转发帖为推荐基底；完整池仍用于下方 reposted_tweet 回填
        n_tweets = len(current_tweets)
        current_tweets_for_algorithm: Dict[str, Any] = {
            tid: t
            for tid, t in current_tweets.items()
            if self._is_original_tweet(t)
        }
        if len(current_tweets_for_algorithm) < n_tweets:
            logger.info(
                f"RecommenderAgent {self.profile_id}: current_tweets 仅保留原创推文：{n_tweets} -> {len(current_tweets_for_algorithm)}"
            )

        user_profile_for_rec: Optional[Dict[str, Any]] = None
        raw_up = getattr(event, "user_profile", None)
        if isinstance(raw_up, dict):
            user_profile_for_rec = raw_up

        if algorithm_name == "interest":
            recommended_contents = await recommendation_func(
                recommended_tweet_ids,
                current_tweets_for_algorithm,
                current_timestamp,
                recommendation_limit,
                candidate_tweet_ids=candidate_tweet_ids,
                user_profile=user_profile_for_rec,
            )
        else:
            recommended_contents = recommendation_func(
                recommended_tweet_ids,
                current_tweets,
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

        for tweet_id, tweet in recommended_contents.items():
            tw_key = self._tweet_ref_key(tweet_id)
            recommended_contents[tweet_id] = self._enrich_tweet_quote_reply_chain(
                dict(tweet), current_tweets, tweet_ref=tw_key
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

    @staticmethod
    def _search_tweet_ids(current_tweets: Dict[str, Dict[str, Any]]) -> List[str]:
        """内容池中可作为搜索候选的 tweet_id（值为 dict 的条目）。"""
        out: List[str] = []
        for tid, tweet in (current_tweets or {}).items():
            if not isinstance(tweet, dict):
                continue
            sid = str(tid).strip()
            if sid:
                out.append(sid)
        return out

    @staticmethod
    def _tweet_text_for_search(tweet: Dict[str, Any], max_len: int = 1200) -> str:
        """拼接标题、正文与标签为检索用文本，并截断到 max_len。"""
        content = str(tweet.get("content", "") or "")
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
        current_tweets: Dict[str, Dict[str, Any]],
        current_timestamp: float,
        limit: int,
    ) -> Dict[str, Dict[str, Any]]:
        """随机搜索：在候选笔记中打乱顺序后取前 limit 条（与热度、语义无关）。"""
        if not current_tweets or limit <= 0:
            return {}
        tweet_ids = list(current_tweets.keys())
        random.shuffle(tweet_ids)
        picked = tweet_ids[:limit]
        return {tid: current_tweets[tid] for tid in picked if tid in current_tweets}

    def _hot_search(
        self,
        current_tweets: Dict[str, Dict[str, Any]],
        current_timestamp: float,
        limit: int,
    ) -> Dict[str, Dict[str, Any]]:
        """热度搜索：按 `_calculate_popularity` 得分降序取前 limit 条；同分顺序经 shuffle 后再 sort 打散。"""
        if not current_tweets or limit <= 0:
            return {}
        rows: List[Tuple[str, Dict[str, Any], float]] = []
        for tid in current_tweets.keys():
            tweet = current_tweets.get(tid)
            if not isinstance(tweet, dict):
                continue
            pop = self._calculate_popularity(tweet)
            rows.append((tid, tweet, pop))
        random.shuffle(rows)
        rows.sort(key=lambda x: x[2], reverse=True)
        out: Dict[str, Dict[str, Any]] = {}
        for tid, tweet, _ in rows[:limit]:
            out[tid] = tweet
        return out

    def _relevant_search_sync(
        self,
        current_tweets: Dict[str, Dict[str, Any]],
        current_timestamp: float,
        limit: int,
        *,
        search_query: str,
    ) -> Dict[str, Dict[str, Any]]:
        """相关搜索（同步）：查询与笔记分别 embedding，按余弦相似度排序；失败或空查询则走关键词 Jaccard 或退化为热度。"""
        if not current_tweets or limit <= 0:
            return {}
        cands = self._search_tweet_ids(current_tweets)
        if not cands:
            return {}

        query = (search_query or "").strip()[:4000]
        if not query.strip():
            sub_tweets = {tid: current_tweets[tid] for tid in cands if tid in current_tweets}
            return self._hot_search(sub_tweets, current_timestamp, limit)

        def _keyword_scores() -> List[Tuple[str, float]]:
            qt = _tokenize(query)
            scored: List[Tuple[str, float]] = []
            for tid in cands:
                tweet = current_tweets.get(tid)
                if not isinstance(tweet, dict):
                    continue
                text = self._tweet_text_for_search(tweet, max_len=2000)
                nt = _tokenize(text)
                if not qt or not nt:
                    scored.append((tid, 0.0))
                    continue
                inter = len(qt & nt)
                union = len(qt | nt)
                j = (inter / union) if union else 0.0
                scored.append((tid, j))
            random.shuffle(scored)
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored

        try:
            cfg = load_step15_gate_config()
            base_url, model_name = load_embedding_config(cfg.embedding_config_path)
            texts = [query[:2000]]
            for tid in cands:
                tweet = current_tweets.get(tid)
                texts.append(
                    self._tweet_text_for_search(tweet, max_len=800)
                    if isinstance(tweet, dict)
                    else ""
                )
            vecs = get_embeddings(base_url, model_name, texts)
            if not vecs or len(vecs) != len(texts):
                raise ValueError("embedding 返回长度与输入不一致")
            vq = vecs[0]
            scored: List[Tuple[str, float]] = []
            for i, tid in enumerate(cands):
                sim = float(cosine_similarity(vq, vecs[i + 1]))
                scored.append((tid, sim))
            random.shuffle(scored)
            scored.sort(key=lambda x: x[1], reverse=True)
            top = scored[:limit]
            return {tid: current_tweets[tid] for tid, _ in top if tid in current_tweets}
        except Exception as e:
            logger.warning(f"Relevant search embedding failed, fallback to keyword: {e}")
            kw = _keyword_scores()
            return {
                tid: current_tweets[tid]
                for tid, _ in kw[:limit]
                if tid in current_tweets
            }

    async def _relevant_search(
        self,
        current_tweets: Dict[str, Dict[str, Any]],
        current_timestamp: float,
        limit: int,
        *,
        search_query: str,
    ) -> Dict[str, Dict[str, Any]]:
        """相关搜索（异步）：在线程池中执行 `_relevant_search_sync`，避免阻塞事件循环。"""
        return await asyncio.to_thread(
            self._relevant_search_sync,
            current_tweets,
            current_timestamp,
            limit,
            search_query=search_query,
        )

    async def _llm_search(
        self,
        current_tweets: Dict[str, Dict[str, Any]],
        current_timestamp: float,
        limit: int,
        *,
        user_profile: Dict[str, Any],
        search_query: str,
    ) -> Dict[str, Dict[str, Any]]:
        """LLM 搜索：将画像、查询意图与最多 300 条候选摘要交给模型，解析 `ranked_tweet_ids` 取前 limit 条。"""
        if not current_tweets or limit <= 0:
            return {}
        cands = self._search_tweet_ids(current_tweets)
        if not cands:
            return {}
        query = self._derive_search_query(user_profile, search_query)

        compact: List[Dict[str, Any]] = []
        for tid in cands[:300]:
            n = current_tweets.get(tid)
            if not isinstance(n, dict):
                continue
            compact.append({
                "tweet_id": tid,
                "content": str(n.get("content", "") or "")[:100]
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

        instruction = """你是搜索排序助手。根据「用户画像」「搜索关键词/意图」与候选微博列表，输出按**与本次搜索相关性**从高到低排序的 tweet_id 列表（JSON）。
        只考虑语义相关性与用户意图匹配，不要编造列表外的 id。
        请严格返回 JSON：
        {
        "ranked_tweet_ids": ["id1", "id2", ...]
        }
        约束：ranked_tweet_ids 中的每个 id 必须来自候选中的 tweet_id，不重复，不要求覆盖全部候选。"""

        observation = (
            f"搜索关键词/意图（若为空则结合画像理解）：\n{query or '(空，请结合用户画像推断可能兴趣)'}\n\n"
            f"用户画像（节选）：\n{json.dumps(profile_compact, ensure_ascii=False)}\n\n"
            f"候选微博：\n{json.dumps(compact, ensure_ascii=False)}"
        )

        pool_set = set(cands)
        ordered: List[str] = []
        try:
            response = await self.generate_recommendation(instruction, observation)
            raw_ids = response.get("ranked_tweet_ids")
            if raw_ids is None:
                raw_ids = response.get("selected_tweet_ids", [])
            if isinstance(raw_ids, list):
                for tweet_id in raw_ids:
                    sid = str(tweet_id).strip()
                    if sid and sid in pool_set and sid not in ordered:
                        ordered.append(sid)
        except Exception as e:
            logger.warning(f"LLM search ranking failed: {e}")
            return {}

        if not ordered:
            return {}
        top_ids = ordered[:limit]
        return {tid: current_tweets[tid] for tid in top_ids if tid in current_tweets}
  
    async def send_search_recommendations(self, event: Event) -> List[Event]:
        """
        处理 SearchEvent：按 profile.type 映射到 random / hot / relevant / llm 搜索，将结果封装为 AlgorithmRecommendationEvent 返回。
        """
        # 获取当前时间步
        current_timestamp = event.timestamp

        # 获取内容池
        logger.info(f"RecommenderAgent {self.profile_id} start getting previous and current tweets")
        
        current_tweets = {}
        if self.profile is not None:
            current_tweets = self.profile.get_data("current_tweets", {})
        if not isinstance(current_tweets, dict) or not current_tweets:
            current_tweets = getattr(event, "current_tweets", {})
        if not isinstance(current_tweets, dict) or not current_tweets:
            logger.warning("No current tweets found from profile/event")
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
                current_tweets,
                current_timestamp,
                top_k,
                user_profile=user_profile,
                search_query=search_query,
            )
        elif algorithm_name == "relevant":
            search_results = await search_func(
                current_tweets,
                current_timestamp,
                top_k,
                search_query=search_query,
            )
        else:
            search_results = search_func(
                current_tweets,
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

        for tweet_id, tweet in search_results.items():
            tw_key = self._tweet_ref_key(tweet_id)
            search_results[tweet_id] = self._enrich_tweet_quote_reply_chain(
                dict(tweet), current_tweets, tweet_ref=tw_key
            )

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