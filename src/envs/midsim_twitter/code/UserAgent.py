from typing import Any, List, Optional, Dict, Set, Tuple, Union
from collections import deque
import json
import asyncio
import os
import re
import secrets
import time
from datetime import datetime, timezone

# SimEnv 对 mention_pool / 评论等更新使用全局锁串行化，高并发下排队可能超过 30s
_ENV_ASYNC_OP_TIMEOUT = float(os.environ.get("ONESIM_ENV_ASYNC_OP_TIMEOUT", "300"))

from grpc import Future
from loguru import logger
from onesim.models import JsonBlockParser
from onesim.agent import GeneralAgent
from onesim.profile import AgentProfile
from onesim.memory import MemoryStrategy
from onesim.planning import PlanningBase
from onesim.events import Event
from onesim.relationship import RelationshipManager
from onesim.distribution.distributed_lock import get_lock  
from .events import *
import random
import math
from collections import defaultdict

from .InteractionThreshold import InteractionThreshold
from .step15_topic_gate import (
    load_step15_gate_config,
    should_inject_step15,
    evaluate_step15_policies as _gate_evaluate_step15_policies,
    topic_text_from_mention_entries,
    topic_text_from_tweets_chunk,
)

class UserAgent(GeneralAgent):
    # 送入 LLM observation 的推文 JSON 中省略的字段（降低 prompt 体积）
    _TWEET_LLM_DROP_KEYS = frozenset({"quote_ids", "reply_ids", "retweet_ids", "time"})
    # instruction 末尾 Step 2.5（env_depth_block 传播深度 coaching）注入概率
    _STEP2_FIVE_ENV_DEPTH_PROB = 0

    def __init__(self,
                 sys_prompt: str | None = None,
                 model_config_name: str = None,
                 event_bus_queue: asyncio.Queue = None,
                 profile: AgentProfile=None,
                 memory: MemoryStrategy=None,
                 planning: PlanningBase=None,
                 relationship_manager: RelationshipManager=None) -> None:
        super().__init__(sys_prompt, model_config_name, event_bus_queue, profile, memory, planning, relationship_manager)
        self.register_event("StartEvent", "get_recommendations_and_mentions")  
        self.register_event("StartEvent", "generate_memory_from_own_tweets")
        self.register_event("SocialRecommendationEvent", "receive_recommendation")
        self.register_event("AlgorithmRecommendationEvent", "receive_recommendation")
        self.register_event("KeepFollowingEvent", "receive_recommendation")
        self.register_event("MentionEvent", "handle_mention")

        self.register_event("AddTweetResponseEvent", "handle_add_tweet_response")
        self.register_event("MentionPoolUpdateResponseEvent", "handle_update_mention_pool_response")
        self._tweet_add_futures: Dict[str, Future] = {}
        self._mention_pool_update_futures: Dict[str, Future] = {}
        self._login_lock = asyncio.Lock()  # 单个智能体级别的锁，用于本轮是否参与的决策

        # 系统内“算法类型 -> 推荐器智能体ID”显式映射（按环境约定维护）
        self.recommender_map: Dict[str, str] = {
            "Random Recommendation": "recomment_agent_0001",
            "Hot Recommendation": "recomment_agent_0002",
            "Interest Recommendation": "recomment_agent_0003",
        }
        # 用户侧默认请求的算法类型列表（由代码固定指定）
        self.default_algorithm_types: List[str] = ["Interest Recommendation"]

        # 系统内“搜索类型 -> 搜索器智能体ID”显式映射（按环境约定维护）
        self.search_map: Dict[str, str] = {
            "Random Search": "search_agent_0001",
            "Hot Search": "search_agent_0002",
            "Relevant Search": "search_agent_0003",
            "LLM Search": "search_agent_0004",
        }
        self.default_search_types: List[str] = ["Relevant Search"]
        self._recommendation_earliest_post_anchor_ms: Optional[float] = None

    @staticmethod
    def _remap_activity_level(activity_level: float, out_min: float = 0.4, out_max: float = 0.8) -> float:
        """
        将 activity_level 先截断到 [0, 1]，再线性重映射到 [out_min, out_max]。
        """
        clamped = max(0.0, min(1.0, activity_level))
        return out_min + (out_max - out_min) * clamped

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

    async def _is_official_by_agent_field(self) -> Optional[bool]:
        """
        通过 agent 字段判断是否官方号。
        仅识别 agent.is_official 这个属性：
        - 若 agent 为 dict 且包含 "is_official" 键：视为官方号（不依赖其值）
        - 若不包含该键或 agent 缺失：返回 None（未知/未标注）
        """
        is_official = self.profile.get_data("is_official", False) 

        return True if is_official else False

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
        从 current_tweets 递归挂上 quoted_tweet / replied_tweet；
        子节点若仍有 quoted_tweet_id 或 replied_tweet_id，继续向上构造直到无引用或池里缺键。
        """
        if _visited is None:
            _visited = set()
        key = tweet_ref or UserAgent._tweet_ref_key(tweet.get("id")) or UserAgent._tweet_ref_key(tweet.get("tweet_id"))
        if key is not None:
            if key in _visited:
                return dict(tweet)
            _visited.add(key)
        if _depth >= max_depth:
            return dict(tweet)

        out = dict(tweet)
        qid = UserAgent._tweet_ref_key(out.get("quoted_tweet_id"))
        rid = UserAgent._tweet_ref_key(out.get("replied_tweet_id"))

        if qid and qid in current_tweets:
            nested = current_tweets[qid]
            if isinstance(nested, dict):
                out["quoted_tweet"] = UserAgent._enrich_tweet_quote_reply_chain(
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
                out["replied_tweet"] = UserAgent._enrich_tweet_quote_reply_chain(
                    nested,
                    current_tweets,
                    tweet_ref=rid,
                    max_depth=max_depth,
                    _depth=_depth + 1,
                    _visited=_visited,
                )
        return out

    @staticmethod
    def _nested_tweet_graph_key(tw: Dict[str, Any]) -> str:
        """嵌套推文节点去重键（与 _enrich_tweet_quote_reply_chain 挂上的子树一致）。"""
        k = UserAgent._tweet_ref_key(tw.get("tweet_id") or tw.get("id"))
        if k:
            return k
        return f"__obj_{id(tw)}"

    @staticmethod
    def _strip_tweet_for_llm_observation(obj: Any) -> Any:
        """
        去掉 quote_ids / reply_ids / retweet_ids / time，并递归处理 replies、quoted_tweet、replied_tweet 等嵌套结构。
        """
        if isinstance(obj, dict):
            out: Dict[str, Any] = {}
            for k, v in obj.items():
                if k in UserAgent._TWEET_LLM_DROP_KEYS:
                    continue
                if k == "replies" and isinstance(v, dict):
                    out[k] = {
                        rk: UserAgent._strip_tweet_for_llm_observation(rv)
                        for rk, rv in v.items()
                    }
                elif isinstance(v, dict):
                    out[k] = UserAgent._strip_tweet_for_llm_observation(v)
                elif isinstance(v, list):
                    out[k] = [UserAgent._strip_tweet_for_llm_observation(x) for x in v]
                else:
                    out[k] = v
            return out
        if isinstance(obj, list):
            return [UserAgent._strip_tweet_for_llm_observation(x) for x in obj]
        return obj

    @staticmethod
    def _shrink_tweet_content_head_tail(
        obj: Any,
        *,
        head: int,
        tail: int,
    ) -> Any:
        """将嵌套结构中的 content 字符串截断为 前 head + … + 后 tail（仅当长度超过 head+tail）。"""
        if isinstance(obj, dict):
            out: Dict[str, Any] = {}
            for k, v in obj.items():
                if k == "content" and isinstance(v, str) and len(v) > head + tail:
                    out[k] = v[:head] + "…" + v[-tail:]
                elif isinstance(v, dict):
                    out[k] = UserAgent._shrink_tweet_content_head_tail(v, head=head, tail=tail)
                elif isinstance(v, list):
                    out[k] = [
                        UserAgent._shrink_tweet_content_head_tail(x, head=head, tail=tail)
                        for x in v
                    ]
                else:
                    out[k] = v
            return out
        if isinstance(obj, list):
            return [
                UserAgent._shrink_tweet_content_head_tail(x, head=head, tail=tail)
                for x in obj
            ]
        return obj

    @staticmethod
    def _clip_discussion_content(text: Any, head: int, tail: int) -> str:
        """讨论树各层 content：仅保留前 head、后 tail（超出时中间用 …）。"""
        s = text if isinstance(text, str) else ("" if text is None else str(text))
        if head <= 0 and tail <= 0:
            return s
        if len(s) <= head + tail:
            return s
        return s[:head] + "…" + s[-tail:]

    @staticmethod
    def _tweet_dict_for_llm_observation(obj: Any) -> Any:
        """
        序列化进 LLM 前：先剥离字段；若 JSON 序列化后超过字符上限，则对所有 content 做前/后截断（默认各 50 字符）。
        上限与截断长度可通过环境变量调整。
        """
        stripped = UserAgent._strip_tweet_for_llm_observation(obj)
        max_json_chars = int(os.environ.get("ONESIM_LLM_TWEET_JSON_MAX_CHARS", "40000"))
        head = max(0, int(os.environ.get("ONESIM_LLM_TWEET_CONTENT_HEAD_CHARS", "50")))
        tail = max(0, int(os.environ.get("ONESIM_LLM_TWEET_CONTENT_TAIL_CHARS", "50")))
        try:
            serialized = json.dumps(stripped, ensure_ascii=False)
        except (TypeError, ValueError):
            return stripped
        if len(serialized) <= max_json_chars:
            return stripped
        shrunk = UserAgent._shrink_tweet_content_head_tail(stripped, head=head, tail=tail)
        try:
            again = json.dumps(shrunk, ensure_ascii=False)
            if len(again) > max_json_chars:
                logger.warning(
                    f"LLM tweet JSON 在 content 截断后仍约 {len(again)} 字符 (> {max_json_chars})，"
                    f"可考虑调低 ONESIM_LLM_TWEET_JSON_MAX_CHARS 或进一步收缩 observation"
                )
        except (TypeError, ValueError):
            pass
        return shrunk

    @staticmethod
    def _tweet_discussion_tree_content_only(
        tweet: Any,
        *,
        _head: Optional[int] = None,
        _tail: Optional[int] = None,
    ) -> Any:
        """
        讨论树在进 LLM 前只保留各层 content（及 tweet_id 便于对齐）；
        quoted_tweet / replied_tweet / replies 递归收缩，减小 observation 体积。
        各层 content 默认仅保留前/后各 50 字符（与 ONESIM_LLM_TWEET_CONTENT_HEAD_CHARS / TAIL_CHARS 一致）。
        """
        if not isinstance(tweet, dict):
            return tweet
        if _head is None:
            _head = max(0, int(os.environ.get("ONESIM_LLM_TWEET_CONTENT_HEAD_CHARS", "50")))
            _tail = max(0, int(os.environ.get("ONESIM_LLM_TWEET_CONTENT_TAIL_CHARS", "50")))
        out: Dict[str, Any] = {}
        tid = UserAgent._tweet_ref_key(tweet.get("tweet_id") or tweet.get("id"))
        if tid:
            out["tweet_id"] = tid

        out["user_id"] = tweet.get("user_id", "")
        out["username"] = tweet.get("username", "")
        out["nickname"] = tweet.get("nickname", "")
        out["content"] = UserAgent._clip_discussion_content(tweet.get("content", ""), _head, _tail)
        for k in ("quoted_tweet", "replied_tweet"):
            ch = tweet.get(k)
            if isinstance(ch, dict):
                out[k] = UserAgent._tweet_discussion_tree_content_only(ch, _head=_head, _tail=_tail)
        reps = tweet.get("replies")
        if isinstance(reps, dict):
            out["replies"] = {
                rk: UserAgent._tweet_discussion_tree_content_only(rv, _head=_head, _tail=_tail)
                if isinstance(rv, dict)
                else rv
                for rk, rv in reps.items()
            }
        return out

    @staticmethod
    def _count_quote_reply_nested_nodes(tweet: Dict[str, Any]) -> int:
        """quoted_tweet / replied_tweet 链上的子节点个数（不含根帖自身）。"""
        if not isinstance(tweet, dict):
            return 0
        n = 0
        for k in ("quoted_tweet", "replied_tweet"):
            ch = tweet.get(k)
            if isinstance(ch, dict):
                n += 1 + UserAgent._count_quote_reply_nested_nodes(ch)
        return n

    @staticmethod
    def _tweet_recommendation_chunk_units(tweet: Dict[str, Any]) -> int:
        """
        单条推荐占用的 chunk 单位（用于可变分块）：
        - 讨论数（replies 条数）：每 2 条算 1 单位；
        - 嵌套（引用/回复链上的子节点数）：每 2 个算 1 单位；
        - 至少 1 单位。
        """
        if not isinstance(tweet, dict):
            return 1
        reps = tweet.get("replies")
        n_rep = len(reps) if isinstance(reps, dict) else 0
        nested = UserAgent._count_quote_reply_nested_nodes(tweet)
        u = (n_rep + 1) // 2 + (nested + 1) // 2
        return max(1, u)

    @staticmethod
    def _pack_recommendation_chunks(
        recommendations: Dict[str, Dict[str, Any]],
        max_units: int,
    ) -> List[Dict[str, Dict[str, Any]]]:
        """按 chunk 单位打包，每批单位之和不超过 max_units；单条超过 max_units 时单独成批。"""
        rec_items = list(recommendations.items())
        if not rec_items:
            return []
        max_units = max(1, int(max_units))
        chunks: List[Dict[str, Dict[str, Any]]] = []
        cur: Dict[str, Dict[str, Any]] = {}
        cur_u = 0
        for tid, tw in rec_items:
            need = UserAgent._tweet_recommendation_chunk_units(tw)
            if need > max_units:
                if cur:
                    chunks.append(cur)
                    cur = {}
                    cur_u = 0
                chunks.append({tid: tw})
                continue
            if cur and cur_u + need > max_units:
                chunks.append(cur)
                cur = {}
                cur_u = 0
            cur[tid] = tw
            cur_u += need
        if cur:
            chunks.append(cur)
        return chunks

    @staticmethod
    def _collect_quote_reply_path_author_user_ids(
        tweet_or_id: Any,
        pool: Optional[Dict[str, Any]],
        *,
        max_nodes: int = 64,
    ) -> List[str]:
        """
        沿嵌套字段 `replied_tweet`、`quoted_tweet` 做 BFS，收集链路上各层 user_id（去重，顺序为 BFS）。

        - `pool` 为 dict 时：`tweet_or_id` 视为 tweet id，从 `pool` 取根帖（如 current_tweets / 推荐 chunk）。
        - `pool` 为 None 时：`tweet_or_id` 视为根推文 dict（如 mention 里的整条 tweet），不查环境。

        若根帖未挂上嵌套对象，链路上可能少于 id 解析结果。
        """
        if pool is None:
            root_tweet = tweet_or_id if isinstance(tweet_or_id, dict) else None
        else:
            key = UserAgent._tweet_ref_key(tweet_or_id)
            if not key or not isinstance(pool, dict):
                return []
            root_tweet = pool.get(key)

        if not isinstance(root_tweet, dict):
            return []

        out: List[str] = []
        seen_uids: Set[str] = set()
        seen_nodes: Set[str] = set()
        dq: deque[Dict[str, Any]] = deque([root_tweet])
        while dq and len(seen_nodes) < max_nodes:
            tw = dq.popleft()
            nk = UserAgent._nested_tweet_graph_key(tw)
            if nk in seen_nodes:
                continue
            seen_nodes.add(nk)
            uid = tw.get("user_id")
            if uid is not None:
                s = str(uid).strip()
                if s and s not in seen_uids:
                    seen_uids.add(s)
                    out.append(s)
            for child_key in ("replied_tweet", "quoted_tweet"):
                nested = tw.get(child_key)
                if isinstance(nested, dict):
                    dq.append(nested)
        return out

    @staticmethod
    def _collect_quote_reply_path_author_user_ids_by_ids(
        tweet_id: Any,
        content_pool: Dict[str, Any],
        *,
        max_nodes: int = 64,
    ) -> List[str]:
        """
        从 content_pool 中按 replied_tweet_id / quoted_tweet_id 的「id 链」向上遍历，
        收集链路上每条 tweet 对应的 author user_id（去重，顺序为 BFS）。

        适用场景：
        - handle_mention：MentionEvent 不提供 enrich 过的 nested quoted_tweet/replied_tweet
        - 但 content_pool 中保存了 replied_tweet_id / quoted_tweet_id，因此可以按 id 查找父帖
        """
        key = UserAgent._tweet_ref_key(tweet_id)
        if not key or not isinstance(content_pool, dict):
            return []

        out: List[str] = []
        seen_uids: Set[str] = set()
        seen_tweets: Set[str] = set()
        dq: deque[str] = deque([key])

        while dq and len(seen_tweets) < max_nodes:
            cur = dq.popleft()
            if cur in seen_tweets:
                continue
            seen_tweets.add(cur)

            tw = content_pool.get(cur)
            if not isinstance(tw, dict):
                continue

            uid = tw.get("user_id")
            if uid is not None:
                s = str(uid).strip()
                if s and s not in seen_uids:
                    seen_uids.add(s)
                    out.append(s)

            # 只向上遍历 reply/quote 的父链（retweet 链不在本 helper 范围内）
            for ref in (tw.get("replied_tweet_id"), tw.get("quoted_tweet_id")):
                pid = UserAgent._tweet_ref_key(ref)
                if pid and pid in content_pool and pid not in seen_tweets:
                    dq.append(pid)

        return out

    @staticmethod
    def _immediate_parent_tweet_id_env(tw: Dict[str, Any]) -> Optional[str]:
        """与 metrics / 监控一致：retweet → quote → reply（及 replyed 拼写）取第一条非空父 id。"""
        if not isinstance(tw, dict):
            return None
        for key in ("retweeted_tweet_id", "quoted_tweet_id", "replied_tweet_id", "replyed_tweet_id"):
            s = UserAgent._tweet_ref_key(tw.get(key))
            if s:
                return s
        return None

    # 与 multi_channel_information_diffusion_twitter_rec 中 UserAgent 保持一致：转推沿链解析根帖 + 推荐 key 用根帖。

    @staticmethod
    def _resolve_retweet_id_to_root_in_pool(
        first_retweet_parent_id: Any,
        pool: Dict[str, Any],
        max_hops: int = 64,
    ) -> Optional[str]:
        """
        从「被转推的 tweet_id」出发，仅在 retweet 边上沿 pool 向上，
        直到某条无 retweeted_tweet_id（视为原创）或池里缺键为止。
        等价于真实 Twitter 上「展示根帖」：避免只展开一层落在中间空壳转推上。
        """
        cur = UserAgent._tweet_ref_key(first_retweet_parent_id)
        if not cur or not isinstance(pool, dict):
            return None
        for _ in range(max_hops):
            tw = pool.get(cur)
            if not isinstance(tw, dict):
                return cur
            nxt = UserAgent._tweet_ref_key(tw.get("retweeted_tweet_id") or "")
            if not nxt:
                return cur
            cur = nxt
        return cur

    @staticmethod
    def _resolve_tweet_to_env_seed_root(
        tweet_id: str,
        content_pool: Dict[str, Any],
        seed_ids: Set[str],
        max_hops: int = 512,
    ) -> Optional[str]:
        """沿父链向上直到命中 seed_root_tweet_ids 中的根；无法到达则 None。"""
        cur = str(tweet_id).strip()
        if not cur or not isinstance(content_pool, dict) or not seed_ids:
            return None
        for _ in range(max_hops):
            if cur in seed_ids:
                return cur
            tw = content_pool.get(cur)
            if not isinstance(tw, dict):
                return None
            pid = UserAgent._immediate_parent_tweet_id_env(tw)
            if not pid or pid == cur:
                return None
            cur = str(pid).strip()
            if not cur:
                return None
        return None

    @staticmethod
    def _hop_edges_to_env_seed_root(
        tweet_id: str,
        content_pool: Dict[str, Any],
        seed_ids: Set[str],
        max_hops: int = 512,
    ) -> Optional[int]:
        """
        当前帖到某一 env 种子根之间的边数（与 Repost Hop Depth / hop_k 一致）。
        若当前帖本身即为种子根，返回 0；无法溯源到任一种子则返回 None。
        """
        if not str(tweet_id).strip() or not isinstance(content_pool, dict) or not seed_ids:
            return None
        cur = str(tweet_id).strip()
        edges = 0
        for _ in range(max_hops):
            if cur in seed_ids:
                return edges
            tw = content_pool.get(cur)
            if not isinstance(tw, dict):
                return None
            pid = UserAgent._immediate_parent_tweet_id_env(tw)
            if not pid or pid == cur:
                return None
            cur = str(pid).strip()
            if not cur:
                return None
            edges += 1
        return None

    @staticmethod
    def _format_env_propagation_depth_hint(
        depth: Optional[int],
        resolved_root: Optional[str],
    ) -> str:
        """One-line hint for the model (Observation)."""
        if depth is None:
            return ""
        root_s = str(resolved_root).strip() if resolved_root else "?"
        if depth == 0:
            return ""
        base = (
            f"**{depth}** hop(s) from the env seed root (same as Repost Hop Depth / hop_{depth}); "
            f"resolved seed tweet_id={root_s}."
        )
        if depth >= 3:
            base += " [≥3 hops: **lean toward propagation=false** unless new fact, correction, or stance shift—not +1/pile-on.]"
            base += " Use [Env-computed propagation depth] per alert; do not estimate. **If depth ≥ 3:** lean toward propagation=false, engage only for new fact, error fix, or real stance change (not courtesy/emoji/pile-on). If propagation=false, decision_reason may say e.g. `d≥3 skip`."
        return base

    @staticmethod
    def _format_env_depth_coaching_block(lines: List[str]) -> str:
        """Join per-tweet env depth lines for instruction / observation footers (see receive_recommendation)."""
        if not lines:
            return ""
        body = "\n            ".join(lines)
        return (
            "\n\n[Env propagation depth — coaching]\n            "
            + body
        )

    @staticmethod
    def _quote_reply_chain_hint_strip_replies(tw: Dict[str, Any]) -> Dict[str, Any]:
        """引用链 hint 用：各层节点去掉 `replies`，仅保留 quoted/replied 嵌套（不沿讨论串展开）。"""
        return {k: v for k, v in tw.items() if k != "replies"}

    @staticmethod
    def _quote_reply_chain_hint(
        tweet: Dict[str, Any],
        *,
        my_id: Optional[str],
        recommended_tweet_ids: Set[str],
        max_nodes: int = 64,
    ) -> str:
        """
        从当前推文对象出发，仅沿嵌套字段 `replied_tweet`、`quoted_tweet` 做 BFS（与 enrich 后结构一致）
        —— 不查 content_pool；链上各节点剔除 `replies` 字段（不将讨论串纳入本 hint 的链路）。
        检测链路上是否包含本人发帖、是否有 tweet_id 落在 recommended_tweet_ids 中。
        返回写入 observation 的醒目标注（无则空字符串）。
        """
        if not isinstance(tweet, dict):
            return ""

        my_s = str(my_id).strip() if my_id else ""
        rec = {str(x).strip() for x in recommended_tweet_ids if x}

        has_self_post = False
        has_seen_before = False
        seen_nodes: Set[str] = set()
        dq: deque[Dict[str, Any]] = deque([UserAgent._quote_reply_chain_hint_strip_replies(tweet)])

        while dq and len(seen_nodes) < max_nodes:
            tw = dq.popleft()
            nk = UserAgent._nested_tweet_graph_key(tw)
            if nk in seen_nodes:
                continue
            seen_nodes.add(nk)

            tid_k = UserAgent._tweet_ref_key(tw.get("tweet_id") or tw.get("id"))
            if tid_k and tid_k in rec:
                has_seen_before = True

            uid = tw.get("user_id")
            if my_s and uid is not None and str(uid).strip() == my_s:
                has_self_post = True

            for child_key in ("replied_tweet", "quoted_tweet"):
                nested = tw.get(child_key)
                if isinstance(nested, dict):
                    dq.append(UserAgent._quote_reply_chain_hint_strip_replies(nested))

        parts: List[str] = []
        if has_self_post:
            parts.append(
                "[⚠ Hint: The quote/reply chain includes a tweet you posted; mind context and avoid talking to yourself.]"
            )
        if has_seen_before:
            parts.append(
                "[⚠ Hint: Some tweets on this chain appeared in your recommendation feed before—do not treat them as brand-new.]"
            )
        return "\n".join(parts)

    async def _should_activate_this_round(self, current_step: int, max_step: int) -> bool:
        """
        使用“activity_level + 幂律上界”共同决定是否激活：
        p_t = min(activity_level, upper_t)
        """
        async with self._login_lock:
            flag = self.profile.get_data("login", -1) if self.profile else -1
            if flag != -1:
                return (flag == 1)

            raw_a = self.profile.get_data("activity_level", 0.0) if self.profile else 0.0
            try:
                activity_level = float(raw_a)
            except (TypeError, ValueError):
                activity_level = 0.0
            activity_level = self._remap_activity_level(activity_level)

            # 你环境里如果不是 current_step，可改成 round_idx / step / current_round
            step_idx = current_step
            max_step = max_step
            p_t = activity_level

            activated = random.random() < p_t
            if self.profile:
                self.profile.update_data("login", 1 if activated else 0)

            return activated

    async def get_recommendations_and_mentions(self, event: Event) -> List[Event]:
        """
        发送社交推荐。收到 StartEvent 时：先清空自己的 recommendations，
        再根据 activate_level 与随机数决定是否处理 mention 与推荐流。
        """
        current_step = getattr(event, "current_step", 1)
        max_step = getattr(event, "max_step", 8)
        if not await self._should_activate_this_round(current_step, max_step):
            logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} is not activated in this round")
            return []
        logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} is activated in this round")

        if (await self._is_official_by_agent_field()) is True:
            logger.info(
                f"Step {current_step}/{max_step}: UserAgent {self.profile_id} detected official account by agent field, skip recommendations and mentions"
            )
            return []

        events: List[Event] = []

        # 从 event（StartEvent）的 current_tweets 中筛选：关注用户的发帖，且时间在 [上次登录, 当前时间] 之间
        current_tweets = getattr(event, "current_tweets", None) or {}
        if not isinstance(current_tweets, dict):
            current_tweets = {}

        # 发送事件给推荐系统：请求“指定算法”的推荐（算法类型由代码固定指定）
        fixed_algorithm_types = self.default_algorithm_types
        if not isinstance(fixed_algorithm_types, list) or not fixed_algorithm_types:
            raise ValueError("default_algorithm_types must be a non-empty list")
        allowed_algorithm_types = set(self.recommender_map.keys())

        # 获取已推荐过的内容ID集合（从profile中读取）
        recommended_tweet_ids = set(self.profile.get_data("recommended_tweet_ids", [])) if self.profile else set()
        # 与 receive_recommendation / _filter 一致：用 strip 后的字符串做成员判断
        recommended_seen: Set[str] = {
            str(x).strip() for x in recommended_tweet_ids if x is not None and str(x).strip()
        }

        # 获取用户画像
        profile_payload = {}
        if self.profile is not None:
            try:
                profile_payload = dict(self.profile.get_profile(include_private=True) or {})
            except Exception:
                logger.warning("Failed to serialize profile via get_profile(), fallback to empty payload.")
        
        # 遍历所有指定算法类型，发送事件给推荐系统
        profile_payload = {}
        if self.profile is not None:
            try:
                profile_payload = dict(self.profile.get_profile(include_private=True) or {})
            except Exception:
                logger.warning("Failed to serialize profile via get_profile(), fallback to empty payload.")
        
        # 遍历所有指定算法类型，发送事件给推荐系统
        for fixed_algorithm_type in fixed_algorithm_types:
            if fixed_algorithm_type not in allowed_algorithm_types:
                raise ValueError(
                    f"Invalid algorithm type '{fixed_algorithm_type}'. "
                    f"Allowed types: {sorted(allowed_algorithm_types)}"
                )
            mapped_id = self.recommender_map.get(fixed_algorithm_type, "")
            if not mapped_id:
                raise ValueError(
                    f"No recommender agent mapped for algorithm type '{fixed_algorithm_type}'. "
                    f"Current mapping keys: {sorted(self.recommender_map.keys())}"
                )
                continue
            events.append(
                GetAlgorithmRecomendationEvent(
                    from_agent_id=self.profile_id,
                    to_agent_id=mapped_id,
                    timestamp=int(getattr(event, "timestamp", 0) or 0),
                    timestamp_duration=int(getattr(event, "timestamp_duration", 0) or 0),
                    current_step=current_step,
                    max_step=max_step,
                    user_profile=profile_payload,
                    current_tweets=current_tweets,
                    recommended_tweet_ids=recommended_tweet_ids,
                    algorithm_type=fixed_algorithm_type,
                )
            )

        # 社交推荐
        current_ts = int(event.timestamp) if getattr(event, "timestamp", None) is not None else 0
        follow_ids = await self.get_data("follow_ids", [])
        follow_set = set(follow_ids) if isinstance(follow_ids, (list, tuple)) else set()

        # 上次登录时间：优先用 profile 记录，否则用 当前时间 - 一个时间步长。
        # 若长期未登录（记录早于当前时刻 3 天以上），社交推荐窗口仍只取「最近 3 天内」的帖子。
        last_login = 0
        post_days_ms = 1000000 * 86400
        reply_days_ms = 1000000 * 86400
        if self.profile:
            raw = self.profile.get_data("last_login_timestamp")
            logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} last_login_timestamp: {raw}")
            if raw is not None and isinstance(raw, (int, float)) and int(raw) > 0:
                last_login = int(raw)
                if current_ts > 0:
                    lower = current_ts - post_days_ms
                    if last_login < lower:
                        last_login = lower
            else:
                last_login = 0

        step_duration = int(getattr(event, "timestamp_duration", 0) or 0)
        window_end = current_ts + step_duration

        recommendations = {}
        for tweet_id, tweet in current_tweets.items():
            if not isinstance(tweet, dict):
                continue
            author_id = tweet.get("user_id") or tweet.get("author_id")
            if author_id not in follow_set:
                continue
            t = tweet.get("time") or tweet.get("create_time")
            if t is None:
                continue
            try:
                t = int(t)
            except (TypeError, ValueError):
                continue
            if t >= 10**12:
                t //= 1000
            if last_login <= t < window_end:
                # 回复帖：仅 0.5 概率进入社交推荐（减轻时间线被长串回复刷屏）
                if tweet.get("replied_tweet_id") and t < window_end - reply_days_ms:
                    continue
                if tweet.get("replied_tweet_id") and random.random() >= 1:
                    continue
                if tweet.get("retweeted_tweet_id"):
                    rt_key = UserAgent._resolve_retweet_id_to_root_in_pool(
                        tweet.get("retweeted_tweet_id"), current_tweets
                    )
                    inner = current_tweets.get(rt_key) if rt_key else None
                    if isinstance(inner, dict):
                        # key 用根帖 id，与正文一致；同一窗口多条转推指向同一根则只保留一条
                        if rt_key in recommendations:
                            continue
                        rt_sid = str(rt_key).strip() if rt_key is not None else ""
                        if rt_sid and rt_sid in recommended_seen:
                            continue
                        recommendations[rt_key] = self._enrich_tweet_quote_reply_chain(
                            inner, current_tweets, tweet_ref=rt_key
                        )
                    else:
                        tid = str(tweet_id).strip() if tweet_id is not None else ""
                        if tid and tid in recommended_seen:
                            continue
                        recommendations[tweet_id] = dict(tweet)
                else:
                    tw_key = self._tweet_ref_key(tweet_id)
                    tid = str(tweet_id).strip() if tweet_id is not None else ""
                    twk = str(tw_key).strip() if tw_key else ""
                    if (tid and tid in recommended_seen) or (twk and twk in recommended_seen):
                        continue
                    recommendations[tweet_id] = self._enrich_tweet_quote_reply_chain(
                        dict(tweet), current_tweets, tweet_ref=tw_key
                    )

        # 取 mentions 给自己发 MentionEvent（先沿 current_tweets 挂上 quote/reply 链，与推荐流一致）
        mentions = getattr(event, "mentions", None) or {}
        if not isinstance(mentions, dict):
            mentions = {}
        if len(mentions) > 10:
            my_id = await self.get_data("id")
            all_keys = list(mentions.keys())
            sampled_keys = random.sample(all_keys, 10)
            logger.info(
                f"Step {current_step}/{max_step}: UserAgent {self.profile_id} "
                f"mentions count {len(all_keys)} > 10, sampling 10 for MentionEvent; dropping others from mention_pool"
            )
            for k in all_keys:
                if k in sampled_keys:
                    continue
                success = await self.update_env_mention_pool(
                    f"{my_id}.{k}",
                    {
                        "action": "delete",
                        "mention_message": None,
                    },
                )
                if not success:
                    logger.error(
                        f"Failed to delete non-sampled mention {k} for user {my_id} in mention_pool"
                    )
            mentions = {k: mentions[k] for k in sampled_keys}
        if 0 < len(mentions) <= 10:
            enriched_mentions: Dict[str, Any] = {}
            for mention_key, mention_message in mentions.items():
                if not isinstance(mention_message, dict):
                    enriched_mentions[mention_key] = mention_message
                    continue
                mm = dict(mention_message)
                tw = mm.get("tweet")
                if not isinstance(tw, dict):
                    enriched_mentions[mention_key] = mm
                    continue
                tw_copy = dict(tw)
                tw_key = self._tweet_ref_key(tw_copy.get("tweet_id") or tw_copy.get("id"))
                if not tw_key:
                    tw_key = self._tweet_ref_key(
                        str(mention_key).split("_")[0] if "_" in str(mention_key) else str(mention_key)
                    )
                if tw_copy.get("retweeted_tweet_id"):
                    rt_key = UserAgent._resolve_retweet_id_to_root_in_pool(
                        tw_copy.get("retweeted_tweet_id"), current_tweets
                    )
                    inner = current_tweets.get(rt_key) if rt_key else None
                    if isinstance(inner, dict):
                        mm["tweet"] = self._enrich_tweet_quote_reply_chain(
                            inner, current_tweets, tweet_ref=rt_key
                        )
                    else:
                        mm["tweet"] = self._enrich_tweet_quote_reply_chain(
                            tw_copy, current_tweets, tweet_ref=tw_key
                        )
                else:
                    mm["tweet"] = self._enrich_tweet_quote_reply_chain(
                        tw_copy, current_tweets, tweet_ref=tw_key
                    )
                enriched_mentions[mention_key] = mm
            mentions = enriched_mentions

            logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} send MentionEvent, length of mentions: {len(mentions)}")
            events.append(MentionEvent(
                from_agent_id=self.profile_id,
                to_agent_id=self.profile_id,
                timestamp=current_ts,
                timestamp_duration=int(getattr(event, "timestamp_duration", 0) or 0),
                current_step=current_step,
                max_step=max_step,
                mentions=mentions,
            ))
        
        # 处理“保持关注/下轮再看”的一次性列表：若上轮决定 keep，则本轮将其重新塞回推荐一次
        keep_following_tweets = {}
        if self.profile and self.profile.get_data("keep_following_tweet_ids", []) is not None:
            keep_ids = self.profile.get_data("keep_following_tweet_ids", []) or []
            if isinstance(keep_ids, (list, tuple)) and keep_ids:
                readded = 0
                for keep_tweet_id in keep_ids:
                    if keep_tweet_id in keep_following_tweets:
                        continue
                    tweet = current_tweets.get(keep_tweet_id)
                    if not isinstance(tweet, dict):
                        continue
                    author_id = tweet.get("user_id") or tweet.get("author_id")
                    if author_id not in follow_set:
                        continue
                    tw_key = self._tweet_ref_key(keep_tweet_id)
                    keep_following_tweets[keep_tweet_id] = self._enrich_tweet_quote_reply_chain(
                        dict(tweet), current_tweets, tweet_ref=tw_key
                    )
                    readded += 1
                if readded > 0:
                    logger.info(
                        f"Step {current_step}/{max_step}: UserAgent {self.profile_id} re-added keep_following tweets: {readded}"
                    )
                # 一次性使用：无论是否成功塞回，都清空，避免无限循环
                self.profile.update_data("keep_following_tweet_ids", [])
                
        if len(keep_following_tweets) > 0:
            events.append(KeepFollowingEvent(
                from_agent_id=self.profile_id,
                to_agent_id=self.profile_id,
                timestamp=current_ts,
                timestamp_duration=int(getattr(event, "timestamp_duration", 0) or 0),
                current_step=current_step,
                max_step=max_step,
                recommendations=keep_following_tweets,
            ))

        # 更新「上次处理到的仿真时刻」为当前时间窗右端（与 time < ts+duration 对齐）
        if self.profile:
            _d = int(getattr(event, "timestamp_duration", 0) or 0)
            self.profile.update_data("last_login_timestamp", current_ts)
            logger.info(
                f"Step {current_step}/{max_step}: UserAgent {self.profile_id} update last_login_timestamp: {current_ts + _d}"
            )

        # 有关注流推荐时再发 SocialRecommendationEvent（50% 概率触发；超过 3 条时随机抽 3 条）
        if recommendations and len(recommendations) > 0:
            if random.random() < 0.055:
                rec_payload = recommendations
                n_rec = len(recommendations)
                if n_rec > 3:
                    sampled_keys = random.sample(list(recommendations.keys()), 3)
                    rec_payload = {k: recommendations[k] for k in sampled_keys}
                    logger.info(
                        f"Step {current_step}/{max_step}: UserAgent {self.profile_id} send SocialRecommendationEvent, "
                        f"sampled 3 from {n_rec} recommendations"
                    )
                else:
                    logger.info(
                        f"Step {current_step}/{max_step}: UserAgent {self.profile_id} send SocialRecommendationEvent, "
                        f"length of recommendations: {n_rec}"
                    )

                events.append(SocialRecommendationEvent(
                    from_agent_id=self.profile_id,
                    to_agent_id=self.profile_id,
                    timestamp=current_ts,
                    timestamp_duration=int(getattr(event, "timestamp_duration", 0) or 0),
                    current_step=current_step,
                    max_step=max_step,
                    recommendations=rec_payload,
                ))
            else:
                logger.debug(
                    f"Step {current_step}/{max_step}: UserAgent {self.profile_id} skip SocialRecommendationEvent (p=0.5), "
                    f"would have sent {len(recommendations)} recommendations"
                )
        return events

    @staticmethod
    def _tweet_post_time_in_window(tweet: Dict[str, Any], lo: float, hi: float) -> bool:
        """发帖时间落在 [lo, hi) 内（lo/hi 为 Unix 秒；兼容旧毫秒 time）。"""
        if lo >= hi:
            return False
        raw = tweet.get("time", tweet.get("create_time"))
        try:
            t = float(raw)
        except (TypeError, ValueError):
            return False
        if t >= 10**12:
            t = t / 1000.0
        return t >= lo and t < hi

    @staticmethod
    def _parse_tweet_time_ms(raw: Any) -> Optional[float]:
        """Normalize to milliseconds; values <1e11 are treated as seconds and scaled by 1000."""
        if raw is None:
            return None
        try:
            x = float(raw)
        except (TypeError, ValueError):
            return None
        if x <= 0:
            return None
        if x < 1e11:
            x *= 1000.0
        return x

    @staticmethod
    def _format_sim_ms_utc(ms: float) -> str:
        if ms is None or ms <= 0:
            return "unknown"
        dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M UTC")

    @staticmethod
    def _sim_ts_to_ms_for_time_module(ts: Any) -> int:
        """Milliseconds for time module; event timestamps may be Unix seconds (<1e12) or ms."""
        if ts is None or not isinstance(ts, (int, float)):
            return 0
        x = float(ts)
        if x < 1e12:
            return int(x * 1000)
        return int(x)

    @staticmethod
    def _content_dicts_for_time_module(chunk: Union[Dict[str, Any], List[Any]]) -> List[Dict[str, Any]]:
        """Recommendation chunk (tweet_id -> tweet) or mention entry list (mention_tweet / mention_tweet / mention_blog)."""
        items: List[Dict[str, Any]] = []
        if isinstance(chunk, dict):
            for v in chunk.values():
                if isinstance(v, dict):
                    items.append(v)
        elif isinstance(chunk, list):
            for entry in chunk:
                if not isinstance(entry, dict):
                    continue
                inner = entry.get("mention_tweet")
                if not isinstance(inner, dict):
                    inner = entry.get("mention_blog")
                if not isinstance(inner, dict):
                    inner = entry.get("mention_tweet")
                if isinstance(inner, dict):
                    items.append(inner)
        return items

    def _time_module_for_recommendation_chunk(
        self, chunk: Union[Dict[str, Any], List[Any]], ref_ms: int
    ) -> str:
        """
        Explicit simulation-time context for the LLM (English).
        First successful parse: set self._recommendation_earliest_post_anchor_ms once.
        Later: elapsed days vs anchor; warn if >7 days (prefer no engagement).
        """
        times_ms: List[float] = []
        for tweet in self._content_dicts_for_time_module(chunk):
            t = UserAgent._parse_tweet_time_ms(
                tweet.get("time", tweet.get("create_time"))
            )
            if t is not None:
                times_ms.append(t)
        if not times_ms:
            return None

        earliest_in_chunk_ms = min(times_ms)
        earliest_str = self._format_sim_ms_utc(earliest_in_chunk_ms)
        ref_ok = ref_ms and ref_ms > 0
        ref_str = self._format_sim_ms_utc(float(ref_ms)) if ref_ok else "unknown"

        anchor = self._recommendation_earliest_post_anchor_ms
        if anchor is None:
            self._recommendation_earliest_post_anchor_ms = float(earliest_in_chunk_ms)
            return None

        if not ref_ok:
            anchor_str = self._format_sim_ms_utc(float(anchor))
            return None

        delta_ms = float(ref_ms) - float(anchor)
        days = delta_ms / 86400000.0
        anchor_str = self._format_sim_ms_utc(float(anchor))
        stale_days = 7.0
        if days >= 0:
            interval_txt = (
                f"about {days:.2f} days have elapsed "
                "(current simulation time − anchor; 1 day = 86400 s)"
            )
        else:
            interval_txt = (
                f"current simulation time is about {-days:.2f} days earlier than the anchor "
                "(data or clock may be inconsistent; interpret cautiously)"
            )
        lines = [
            f"[Time] Current simulation window start: {ref_str}.",
            f"Relative to the agent's anchored earliest post time ({anchor_str}), {interval_txt}.",
        ]
        if days > stale_days:
            lines.append(
                f"[Warning] More than about {stale_days:.0f} days have elapsed; content is likely stale. "
                "Unless you have a strong reason or a traceable new fact, prefer **no engagement** (silence)."
            )
        return "\n            ".join(lines)

    async def _memory_text_for_gate(self) -> str:
        """
        步骤1.5 门控专用：记忆条数少时不做「按 observation 检索 / 算相关性」，直接汇总各存储中的**全部**记忆文本，
        与 generate_reaction 里按 query retrieve 的 top_k 可能不同，但「是否有可对照记忆」以全量为准。
        """
        if not self.memory:
            return ""
        try:
            items: list = []
            gam = getattr(self.memory, "get_all_memory", None)
            if callable(gam):
                buckets = await gam()
                if isinstance(buckets, dict):
                    seen: Set[Any] = set()
                    for lst in buckets.values():
                        if not isinstance(lst, list):
                            continue
                        for msg in lst:
                            iid = getattr(msg, "id", None)
                            if iid is None:
                                iid = id(msg)
                            if iid in seen:
                                continue
                            seen.add(iid)
                            items.append(msg)
            if not items:
                memory_msgs = await self.memory.retrieve("")
                items = list(memory_msgs or [])
            text = ""
            for msg in items:
                text += getattr(msg, "content", "") or ""
            return text
        except Exception as e:
            logger.warning(
                f"UserAgent {self.profile_id} _memory_text_for_gate failed: {e}"
            )
            return ""

    async def _should_inject_step15(self, *, topic_text: str) -> bool:
        """
        是否注入「步骤1.5」长 prompt。策略由环境变量 ONESIM_STEP15_* 控制（见 step15_topic_gate.py）。
        记忆侧文本来自全量记忆（_memory_text_for_gate），与推荐 observation 无关；topic_text 仅用于 keyword/embedding 与当前批次比对。
        向量分支在线程中执行，避免阻塞事件循环。
        """
        cfg = load_step15_gate_config()
        mem_blob = await self._memory_text_for_gate()
        memory_nonempty = bool(mem_blob.strip())
        return await asyncio.to_thread(
            should_inject_step15,
            cfg,
            memory_nonempty=memory_nonempty,
            memory_blob=mem_blob,
            topic_text=topic_text or "",
        )

    async def evaluate_step15_policies(self, *, topic_text: str) -> Dict[str, Any]:
        """
        对 ONESIM_STEP15_POLICY 中每个策略求值，返回 dict（见 step15_topic_gate.evaluate_step15_policies）。
        总开关用键 _combined_inject；各策略名为键，值为至少含 inject: bool 的子 dict。
        向量分支在线程中执行，避免阻塞事件循环。
        """
        cfg = load_step15_gate_config()
        mem_blob = await self._memory_text_for_gate()
        memory_nonempty = bool(mem_blob.strip())
        return await asyncio.to_thread(
            _gate_evaluate_step15_policies,
            cfg,
            memory_nonempty=memory_nonempty,
            memory_blob=mem_blob,
            topic_text=topic_text or "",
        )

    async def generate_memory_from_own_tweets(self, event: Event) -> List[Event]:
        """
        StartEvent 多播：对齐微博 generate_memory_from_own_tweets 的流程。

        - 时间窗：与微博一致，[lo, hi) 且 hi 受 simulation_cap_timestamp 截断（`_tweet_post_time_in_window`）。
        - 记忆复盘：仅**纯原创**帖（无转推/引用/回复父 id）。
        - 本人帖 id 合并进 `recommended_tweet_ids`，避免再进推荐候选。
        - 追评：对本人非转推帖先按概率子采样，再 **一次** `generate_reaction` 批量决策；执行阶段按 decisions 落库（无循环调用大模型）。
          使用与 `receive_recommendation` 同构的 JSON 与 `add_env_tweets` 传播逻辑。
        """
        current_step = getattr(event, "current_step", 1)
        max_step = getattr(event, "max_step", 8)

        user_id = await self.get_data("id")
        if not user_id:
            logger.error(f"Step {current_step}/{max_step}: User {user_id} failed to get user_id")
            return []

        current_ts = int(event.timestamp) if getattr(event, "timestamp", None) is not None else 0
        duration = int(getattr(event, "timestamp_duration", 0) or 0)

        current_tweets = getattr(event, "current_tweets", None) or {}
        if not isinstance(current_tweets, dict) or not current_tweets:
            logger.error(f"Step {current_step}/{max_step}: User {user_id} failed to get current_tweets")
            return []

        own_entries: List[Tuple[str, Dict[str, Any]]] = []
        for tweet_id, tweet in current_tweets.items():
            if not isinstance(tweet, dict):
                continue
            if (tweet.get("user_id") or tweet.get("author_id")) != user_id:
                continue
            own_entries.append((str(tweet_id), tweet))

        lo = float(current_ts)
        dur_sec = float(duration)
        cap = getattr(event, "simulation_cap_timestamp", None)
        if cap is not None:
            cap_f = float(cap)
            if cap_f >= 10**12:
                cap_f = cap_f / 1000.0
            hi = min(lo + dur_sec, cap_f)
        else:
            hi = lo + dur_sec
        if hi <= lo:
            logger.error(f"Step {current_step}/{max_step}: User {user_id} failed to get own_entries")
            return []

        own_entries = [
            (tid, tw)
            for tid, tw in own_entries
            if self._tweet_post_time_in_window(tw, lo, hi)
        ]
        if not own_entries:
            return []

        def _is_pure_original(tw: Dict[str, Any]) -> bool:
            if str(tw.get("retweeted_tweet_id") or "").strip():
                return False
            if str(tw.get("quoted_tweet_id") or "").strip():
                return False
            if str(tw.get("replied_tweet_id") or tw.get("replyed_tweet_id") or "").strip():
                return False
            return True

        own_originals = [(tid, tw) for tid, tw in own_entries if _is_pure_original(tw)]

        if self.profile:
            existing_recommended = self.profile.get_data("recommended_tweet_ids", []) or []
            if not isinstance(existing_recommended, list):
                existing_recommended = list(existing_recommended) if existing_recommended else []
            own_ids = [str(tid) for tid, _ in own_entries if tid]
            if own_ids:
                merged = list(dict.fromkeys(existing_recommended + own_ids))
                self.profile.update_data("recommended_tweet_ids", merged)

        own_tweets_for_prompt: List[Dict[str, Any]] = []
        for tweet_id, tweet in own_originals:
            own_tweets_for_prompt.append({
                "tweet_id": tweet_id,
                "content": tweet.get("content", ""),
            })

        current_step = getattr(event, "current_step", 1)
        max_step = getattr(event, "max_step", 8)
        if own_tweets_for_prompt:
            instruction = (
                "You are reflecting on your recent posts. From these tweets, distill one first-person memory entry "
                "summarizing: topics you keep following, your voice/style, and reusable phrasing for future interactions."
            )
            observation = f"Your recent posts:\n{json.dumps(own_tweets_for_prompt, ensure_ascii=False, indent=2)}"
            reaction = {
                "task": "self_reflection_on_own_tweets",
                "own_tweet_count": len(own_tweets_for_prompt),
                "highlights": own_tweets_for_prompt,
            }
            try:
                memory_text = await self.generate_memory(instruction, observation, reaction)
                if memory_text:
                    logger.info(
                        f"Step {current_step}/{max_step}: User {user_id} generated self-memory from "
                        f"{len(own_tweets_for_prompt)} own tweets, memory_text: {memory_text}"
                    )
            except Exception as e:
                logger.error(
                    f"Step {current_step}/{max_step}: User {user_id} failed to generate memory from own tweets: {e}"
                )

        followup_prob = float(os.environ.get("TWITTER_SELF_FOLLOWUP_PROB", "0.001"))
        if followup_prob <= 0.0:
            return []

        max_batch = max(1, int(os.environ.get("TWITTER_SELF_FOLLOWUP_MAX_BATCH", "16")))
        pool = [
            (tid, tw)
            for tid, tw in own_entries
            if not str(tw.get("retweeted_tweet_id") or "").strip()
        ]
        random.shuffle(pool)
        subs: List[Tuple[Any, Dict[str, Any]]] = [
            (tid, tw) for tid, tw in pool if random.random() < followup_prob
        ]
        if len(subs) > max_batch:
            subs = random.sample(subs, max_batch)
        if not subs:
            return []

        user_nickname = await self.get_data("nickname", "") or ""
        user_username = await self.get_data("username", "") or ""
        follow_ids = await self.get_data("follow_ids", [])
        fan_ids = await self.get_data("fan_ids", [])
        current_timestamp = event.timestamp
        mentionable_users = self._get_mentionable_users(follow_ids, fan_ids)
        mention_prompt_users = self._mentionable_for_prompt_mutual_and_official_follows(mentionable_users)
        mentionable_users_str = json.dumps(mention_prompt_users, ensure_ascii=False, indent=2)

        chunk: Dict[str, Dict[str, Any]] = {str(tid): tw for tid, tw in subs}
        posts_payload = [
            {
                "tweet_id": str(tid),
                "content": (tw.get("content") or "")[:800],
            }
            for tid, tw in subs
        ]
        n_posts = len(posts_payload)
        recommendations_str = json.dumps({"self_tweets": posts_payload}, ensure_ascii=False, indent=2)
        observation = f"""[Scenario] Rare self follow-up on your own tweets (one batched decision)
        {recommendations_str}

        Users you may @:
        {mentionable_users_str}"""

        instruction = f"""This is a rare **batched** decision to add follow-up engagement on your own past tweets ({n_posts} items).
        Use Observation.self_tweets and memory: for each item, decide if it is worth engaging again (new facts, new developments, or a change of stance).
        If yes: propagation=true, propagation_type is "reply" / "quote" / "retweet" (prefer reply or quote); for retweet, propagation_content must be "".
        If no: propagation=false, propagation_type "".
        Do not use hashtags (#...) in propagation_content.

        Same schema as the main feed decision; **decisions must contain exactly {n_posts} objects**, in the same order as self_tweets; item i's tweet_id must equal self_tweets[i].tweet_id.

        ```json
        {{
        "persona_understanding": "1-2 sentences",
        "content_understanding": "1-2 sentences",
        "source_understanding": "Your historical tweets (batched self follow-up)",
        "memory_reflection": "2-3 sentences",
        "decisions": [
            {{
            "tweet_id": "<must match self_tweets[0].tweet_id>",
            "propagation": false,
            "propagation_type": "",
            "propagation_content": "",
            "decision_reason": "",
            "expression_reason": "",
            "mention_reasoning": []
            }}
        ],
        "keep_following_tweet_ids": [],
        "keep_following_reason": "",
        "search": false,
        "search_keyword": "keyword for search, <len<=20 chars>",
        "search_reason": ""
        }}
        ```
        (Expand decisions to {n_posts} objects; do not omit.)
        When propagation=true, propagation_type ∈ {{"retweet","reply","quote"}}."""

        env = getattr(self, "env", None)
        if env is not None and hasattr(env, "notify_agent_busy"):
            await env.notify_agent_busy()
        try:
            response = await self.generate_reaction(instruction, observation)
        finally:
            if env is not None and hasattr(env, "notify_agent_idle"):
                await env.notify_agent_idle()

        if not isinstance(response, dict):
            return []
        decisions_raw = response.get("decisions", [])
        if not isinstance(decisions_raw, list):
            return []

        by_tid: Dict[str, Dict[str, Any]] = {}
        for d in decisions_raw:
            if isinstance(d, dict):
                tk = str(d.get("tweet_id") or "").strip()
                if tk:
                    by_tid[tk] = d

        for tweet_id, tweet in subs:
            tid_key = str(tweet_id)
            decision = by_tid.get(tid_key)
            if not isinstance(decision, dict):
                continue
            should_propagation = decision.get("propagation", False)
            propagation_type = decision.get("propagation_type", "")
            propagation_content = decision.get("propagation_content", "")
            if not should_propagation:
                continue
            if propagation_type != "retweet" and not (propagation_content or "").strip():
                continue
            if tid_key not in chunk:
                continue

            mentioned_user_ids: List[str] = []
            mention_reasoning = decision.get("mention_reasoning", [])
            if isinstance(mention_reasoning, list):
                for mention_reason in mention_reasoning:
                    if isinstance(mention_reason, dict):
                        muid = mention_reason.get("user_id")
                        if muid:
                            mentioned_user_ids.append(muid)

            path_author_user_ids = self._collect_quote_reply_path_author_user_ids(tid_key, chunk)
            path_author_set = {str(x).strip() for x in path_author_user_ids if x}
            filtered_mention_ids: Set[str] = set()
            for x in mentioned_user_ids:
                if x is not None:
                    sx = str(x).strip()
                    if sx and sx not in path_author_set:
                        filtered_mention_ids.add(sx)
            filtered_mention_ids.discard(user_id)
            mention_count = len(filtered_mention_ids)

            propagation_id = self._generate_propagation_id()

            if propagation_type == "retweet":
                success = await self.add_env_tweets(propagation_id, {
                    "tweet_id": propagation_id,
                    "content": "",
                    "time": current_timestamp if isinstance(current_timestamp, (int, float)) else 0,
                    "user_id": user_id,
                    "nickname": user_nickname,
                    "username": user_username,
                    "mention_count": mention_count,
                    "retweeted_tweet_id": tid_key,
                })
                if not success:
                    logger.error(
                        f"Step {current_step}/{max_step}: self_followup retweet failed for {tid_key}"
                    )
                    continue
            elif propagation_type == "quote":
                success = await self.add_env_tweets(propagation_id, {
                    "tweet_id": propagation_id,
                    "content": propagation_content,
                    "time": current_timestamp if isinstance(current_timestamp, (int, float)) else 0,
                    "user_id": user_id,
                    "nickname": user_nickname,
                    "username": user_username,
                    "mention_count": mention_count,
                    "quoted_tweet_id": tid_key,
                })
                if not success:
                    logger.error(
                        f"Step {current_step}/{max_step}: self_followup quote failed for {tid_key}"
                    )
                    continue
            elif propagation_type == "reply":
                success = await self.add_env_tweets(propagation_id, {
                    "tweet_id": propagation_id,
                    "content": propagation_content,
                    "time": current_timestamp if isinstance(current_timestamp, (int, float)) else 0,
                    "user_id": user_id,
                    "nickname": user_nickname,
                    "username": user_username,
                    "mention_count": mention_count,
                    "replied_tweet_id": tid_key,
                })
                if not success:
                    logger.error(
                        f"Step {current_step}/{max_step}: self_followup reply failed for {tid_key}"
                    )
                    continue
            else:
                continue

            path_notify_ids = sorted(
                {
                    s
                    for s in (str(x).strip() for x in path_author_user_ids if x)
                    if s and s != user_id
                }
            )
            for target_uid in path_notify_ids:
                ok = await self.update_env_mention_pool(f"{target_uid}.{propagation_id}", {
                    "action": "add",
                    "mention_message": {
                        "tweet_id": tid_key,
                        "mention_type": propagation_type,
                    },
                })
                if not ok:
                    logger.error(
                        f"Step {current_step}/{max_step}: mention_pool update failed for self_followup "
                        f"{propagation_id} -> {target_uid}"
                    )
                else:
                    logger.info(
                        f"Step {current_step}/{max_step}: User {user_id} self_followup {propagation_type} "
                        f"on own tweet {tid_key} -> {propagation_id}"
                    )

            if filtered_mention_ids:
                for mentioned_user_id in filtered_mention_ids:
                    if mentioned_user_id and mentioned_user_id != user_id:
                        ok = await self.update_env_mention_pool(
                            f"{mentioned_user_id}.{propagation_id}",
                            {
                                "action": "add",
                                "mention_message": {
                                    "tweet_id": tid_key,
                                    "mention_type": "at",
                                },
                            },
                        )
                        if not ok:
                            logger.error(
                                f"Step {current_step}/{max_step}: mention_pool @ failed for self_followup "
                                f"{propagation_id}"
                            )

        return []

    def _generate_propagation_id(self) -> str:
        """
        生成 content_pool 新推文键：毫秒时间戳 + 6 位随机十进制后缀。
        避免仅用 time()*1000 时同毫秒内多事件/多智能体撞号（曾导致 already exists in content_pool）。
        """
        ms = int(time.time() * 1000)
        suffix = secrets.randbelow(1_000_000)
        return f"{ms}{suffix:06d}"
    
    @staticmethod
    def _random_propagation_timestamp(
        tweet: Dict[str, Any], window_start_sec: int, window_duration_sec: int
    ) -> int:
        """
        转发时间戳：Unix 秒（与微博 create_time 语义一致），落在 [max(发帖时间, 窗口起点), 窗口终点] 内离散均匀随机（整秒，无小数）。
        与 SimEnv 一致：半开区间 [window_start_sec, window_start_sec + window_duration_sec)；
        window_duration_sec==0 时上界退化为 window_start_sec。
        发帖时间可为毫秒（>=1e12）或秒，均归一为秒再抽样。
        """
        if window_start_sec <= 0 and window_duration_sec <= 0:
            return 0
        lo_win = window_start_sec
        if window_duration_sec > 0:
            hi_incl = lo_win + window_duration_sec - 1
        else:
            hi_incl = lo_win
        post = (
            tweet.get("time", tweet.get("create_time"))
            if isinstance(tweet, dict)
            else None
        )
        if not isinstance(post, (int, float)):
            return hi_incl
        pt = int(post)
        if pt >= 10**12:
            pt //= 1000
        lo = max(pt, lo_win)
        hi = hi_incl
        if lo > hi:
            lo, hi = hi, lo
        lo_i, hi_i = int(lo), int(hi)
        if hi_i < lo_i:
            return lo_i
        return random.randint(lo_i, hi_i)
    
    def _parse_mentions(self, comment_content: str, user_id_map: Dict[str, str], mentionable_users: Dict[str, Any]) -> Tuple[str, List[str]]:
        """
        解析评论中的@，提取被@的用户ID，并将 @id 替换为 @昵称
        
        Args:
            comment_content: 评论内容
            user_id_map: 用户ID/昵称到用户ID的映射（nickname -> user_id, user_id -> user_id）
            mentionable_users: 可@的用户列表，包含 follows、fans、mutual 三个列表
            
        Returns:
            tuple[str, List[str]]: (替换后的评论内容, 被@的用户ID列表)
        """
        # 构建 user_id 到 nickname 的反向映射
        user_id_to_nickname = {}
        all_mentionable = mentionable_users.get("follows", []) + \
                         mentionable_users.get("fans", []) + \
                         mentionable_users.get("mutual", [])
        for user_info in all_mentionable:
            if isinstance(user_info, dict):
                user_id = user_info.get("user_id", "") or user_info.get("id", "")
                nickname = user_info.get("nickname", "") 
                if user_id and nickname:
                    user_id_to_nickname[user_id] = nickname
        
        # 匹配@格式：@用户ID 或 @用户昵称
        mention_pattern = r'@(\w+)'
        mentions = re.findall(mention_pattern, comment_content)
        
        # 存储被@的用户ID列表
        mentioned_user_ids = []
        # 存储替换映射：原始内容 -> 替换后的内容
        replacements = {}
        
        # 为每个@找到对应的用户ID，并准备替换
        for mention in mentions:
            # 通过 user_id_map 找到对应的用户ID
            mentioned_user_id = user_id_map.get(mention)
            
            if mentioned_user_id:
                mentioned_user_ids.append(mentioned_user_id)
                # 获取对应的昵称
                nickname = user_id_to_nickname.get(mentioned_user_id, "")
                if nickname:
                    # 将 @id 或 @昵称 替换为 @昵称
                    replacements[f"@{mention}"] = f"@{nickname}"
                else:
                    # 如果找不到昵称，保持原样（但这种情况不应该发生）
                    logger.warning(f"Could not find nickname for user_id: {mentioned_user_id}")
            else:
                logger.warning(f"Could not find user_id for mention: {mention}")
        
        # 执行替换
        updated_content = comment_content
        for old_text, new_text in replacements.items():
            updated_content = updated_content.replace(old_text, new_text)
        
        return updated_content, mentioned_user_ids
    
    def _add_recommendations(self, recommendations: Dict[str, Any]) -> None:
        """
        将推荐内容加入已推荐列表（键为 tweet_id）。
        """
        if not recommendations:
            return
        
        # 获取已推荐过的内容ID集合（从profile中读取）
        recommended_tweet_ids = set(self.profile.get_data("recommended_tweet_ids", [])) if self.profile else set()
        
        new_tweet_ids = []
        
        for tweet_id in recommendations.keys():
            new_tweet_ids.append(tweet_id)
            # logger.info(f"Recommendation {tweet_id} is added to filtered list")
        
        # 将新推荐的内容ID添加到已推荐列表中
        if new_tweet_ids and self.profile:
            all_recommended = list(recommended_tweet_ids) + new_tweet_ids
            self.profile.update_data("recommended_tweet_ids", all_recommended)
        
    @staticmethod
    def _clip_historical_summary_for_mentionable(text: Any) -> str:
        """可@用户列表里的 historical_summary：超过 100 字时只保留前 50、后 50，中间用 … 连接。"""
        if text is None:
            return ""
        s = str(text)
        if len(s) <= 100:
            return s
        return s[:50] + "…" + s[-50:]

    @staticmethod
    def _sample_interest_tags_for_mentionable(tags: Any) -> Any:
        """可@用户信息中的 interest_tags：超过 3 个时无放回随机保留 3 个。"""
        if not isinstance(tags, (list, tuple)):
            return tags
        lst = [str(t).strip() for t in tags if str(t).strip()]
        if len(lst) <= 3:
            return lst
        return random.sample(lst, 3)

    def _record_recommendations_by_source_step(
        self,
        source_type: str,
        current_step: int,
        recommendations: Dict[str, Any],
        event_timestamp: Any,
    ) -> None:
        """
        按推荐来源 source_type 与仿真轮次 current_step，把本轮 tweet_id 追加写入
        profile.recommended_tweet_ids_by_channel。

        结构：recommended_tweet_ids_by_channel[source_type][str(step)] = [tweet_id, ...]
        同一 (source_type, step) 下多次写入时合并列表并去重（保持顺序）。

        若 event_timestamp 可解析为大于 0 的整数，则同步更新 profile.last_login_timestamp。
        最后打日志输出 last_login_timestamp 与完整 recommended_tweet_ids_by_channel。
        """
        if not self.profile:
            return
        st = (source_type or "").strip()
        if not st or not recommendations:
            return

        try:
            step_key = str(int(current_step))
        except (TypeError, ValueError):
            step_key = str(current_step)

        raw_root = self.profile.get_data("recommended_tweet_ids_by_channel", {})
        by_ch: Dict[str, Any] = dict(raw_root) if isinstance(raw_root, dict) else {}

        step_map_raw = by_ch.get(st)
        step_map: Dict[str, Any] = (
            dict(step_map_raw) if isinstance(step_map_raw, dict) else {}
        )

        prev_ids = step_map.get(step_key)
        merged: List[str] = list(prev_ids) if isinstance(prev_ids, list) else []
        seen: Set[str] = {str(x).strip() for x in merged if str(x).strip()}
        for tweet_id in recommendations.keys():
            sid = str(tweet_id).strip()
            if not sid or sid in seen:
                continue
            merged.append(sid)
            seen.add(sid)

        step_map[step_key] = merged
        by_ch[st] = step_map
        self.profile.update_data("recommended_tweet_ids_by_channel", by_ch)

        ts_int = 0
        if event_timestamp is not None:
            try:
                ts_int = int(event_timestamp)
            except (TypeError, ValueError):
                ts_int = 0
        if ts_int > 0:
            self.profile.update_data("last_login_timestamp", ts_int)

        raw_ll = self.profile.get_data("last_login_timestamp", 0)
        try:
            ll_out = int(raw_ll) if raw_ll is not None else 0
        except (TypeError, ValueError):
            ll_out = 0

        logger.info(
            f"UserAgent {self.profile_id} recommended_by_channel: "
            f"last_login_timestamp={ll_out} "
            f"recommended_tweet_ids_by_channel={json.dumps(by_ch, ensure_ascii=False, default=str)}"
        )

    def _record_mentioned_tweet_ids_by_channel(
        self,
        current_step: int,
        mention_entries: List[Dict[str, Any]],
        event_timestamp: Any,
    ) -> None:
        """
        按 MentionEvent 中的 mention_type 与当前轮次，把相关 tweet_id 追加写入 profile.mentioned_tweet_ids_by_channel。

        结构：mentioned_tweet_ids_by_channel[mention_type][str(step)] = [tweet_id, ...]
        同一 (mention_type, step) 下多次写入时合并列表并去重（保持顺序）。

        若 event_timestamp 可解析为大于 0 的整数，则同步更新 profile.last_login_timestamp。
        最后打日志输出 last_login_timestamp、current_step 与完整 mentioned_tweet_ids_by_channel。
        """
        if not self.profile or not mention_entries:
            return

        try:
            step_key = str(int(current_step))
        except (TypeError, ValueError):
            step_key = str(current_step)

        batch: Dict[str, List[str]] = defaultdict(list)
        for entry in mention_entries:
            if not isinstance(entry, dict):
                continue
            mt = str(entry.get("mention_type") or "at").strip() or "at"
            nid = entry.get("tweet_id")
            sid = str(nid).strip() if nid is not None else ""
            if sid:
                batch[mt].append(sid)

        if not batch:
            return

        raw_root = self.profile.get_data("mentioned_tweet_ids_by_channel", {})
        by_ch: Dict[str, Any] = dict(raw_root) if isinstance(raw_root, dict) else {}

        for mt, ids in batch.items():
            prev = by_ch.get(mt)
            if isinstance(prev, dict):
                step_map: Dict[str, Any] = dict(prev)
            else:
                step_map = {}

            prev_ids = step_map.get(step_key)
            merged: List[str] = list(prev_ids) if isinstance(prev_ids, list) else []
            seen: Set[str] = {str(x).strip() for x in merged if str(x).strip()}
            for sid in ids:
                if sid not in seen:
                    merged.append(sid)
                    seen.add(sid)
            step_map[step_key] = merged
            by_ch[mt] = step_map

        self.profile.update_data("mentioned_tweet_ids_by_channel", by_ch)

        ts_int = 0
        if event_timestamp is not None:
            try:
                ts_int = int(event_timestamp)
            except (TypeError, ValueError):
                ts_int = 0
        if ts_int > 0:
            self.profile.update_data("last_login_timestamp", ts_int)

        raw_ll = self.profile.get_data("last_login_timestamp", 0)
        try:
            ll_out = int(raw_ll) if raw_ll is not None else 0
        except (TypeError, ValueError):
            ll_out = 0

        try:
            step_out = int(current_step)
        except (TypeError, ValueError):
            step_out = current_step

        logger.info(
            f"UserAgent {self.profile_id} mentioned_by_channel: "
            f"step={step_out} last_login_timestamp={ll_out} "
            f"mentioned_tweet_ids_by_channel={json.dumps(by_ch, ensure_ascii=False, default=str)}"
        )

    def _get_mentionable_users(
        self,
        follow_ids: List[str],
        fan_ids: List[str]
    ) -> Dict[str, Any]:
        """
        获取可@的用户列表（包括关注、粉丝、互关）
        
        Args:
            follow_ids: 关注列表
            fan_ids: 粉丝列表
        
        Returns:
            Dict[str, Any]: 包含 follows、fans、mutual 三个列表的字典
        """
        mentionable_info = {
            "follows": [],  # 关注列表
            "fans": [],  # 粉丝列表
            "mutual": []  # 互关列表
        }
        
        # 转换为集合以便计算
        follow_ids = set(follow_ids)
        fan_ids = set(fan_ids)
        
        # 计算互关（交集）
        mutual_ids = follow_ids & fan_ids
        
        # 构建用户ID到昵称的映射
        user_info = {}
        
        if not self.relationship_manager:
            logger.warning("RelationshipManager is not initialized")
            return mentionable_info
        
        # 处理所有列表，直接从 relationship_manager 获取用户信息（同步，快速）
        follows_info = []
        fans_info = []
        mutual_info = []
        
        # 处理关注列表（包含互关）
        for user_id in follow_ids:
            rel = self.relationship_manager.get_relationship(user_id)
            if rel and rel.target_info and isinstance(rel.target_info, dict):
                info = dict(rel.target_info)
                if "historical_summary" in info:
                    info["historical_summary"] = self._clip_historical_summary_for_mentionable(
                        info.get("historical_summary")
                    )
                if "interest_tags" in info:
                    info["interest_tags"] = self._sample_interest_tags_for_mentionable(
                        info.get("interest_tags")
                    )
                hn = info.get("historical_tweets")
                if isinstance(hn, dict) and len(hn) > 2:
                    items = list(hn.items())[:2]
                    info["historical_tweets"] = dict(items)
                follows_info.append(info)
        
        # 处理粉丝列表（包含互关）
        for user_id in fan_ids:
            rel = self.relationship_manager.get_relationship(user_id)
            # 先检查 rel 是否为 None，再访问 target_info
            if rel and rel.target_info and isinstance(rel.target_info, dict):
                info = dict(rel.target_info)
                if "historical_summary" in info:
                    info["historical_summary"] = self._clip_historical_summary_for_mentionable(
                        info.get("historical_summary")
                    )
                if "interest_tags" in info:
                    info["interest_tags"] = self._sample_interest_tags_for_mentionable(
                        info.get("interest_tags")
                    )
                hn = info.get("historical_tweets")
                if isinstance(hn, dict) and len(hn) > 2:
                    items = list(hn.items())[:2]
                    info["historical_tweets"] = dict(items)
                fans_info.append(info)
        
        # 处理互关（互关用户的历史发帖最多保留 2 条）
        for user_id in mutual_ids:
            rel = self.relationship_manager.get_relationship(user_id)
            # 先检查 rel 是否为 None，再访问 target_info
            if rel and rel.target_info and isinstance(rel.target_info, dict):
                info = dict(rel.target_info)
                if "historical_summary" in info:
                    info["historical_summary"] = self._clip_historical_summary_for_mentionable(
                        info.get("historical_summary")
                    )
                if "interest_tags" in info:
                    info["interest_tags"] = self._sample_interest_tags_for_mentionable(
                        info.get("interest_tags")
                    )
                hn = info.get("historical_tweets")
                if isinstance(hn, dict) and len(hn) > 2:
                    items = list(hn.items())[:2]
                    info["historical_tweets"] = dict(items)
                mutual_info.append(info)

        mentionable_info["follows"] = follows_info
        mentionable_info["fans"] = fans_info
        mentionable_info["mutual"] = mutual_info
        
        return mentionable_info

    def _mentionable_for_prompt_mutual_and_official_follows(
        self, mentionable_users: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        供 LLM prompt「Users you may @」使用：仅互关用户，以及 follows 中 is_official 为真的用户（按 user_id 去重）。
        去重后超过 5 个时随机保留 5 个。
        """
        def _uid(info: Dict[str, Any]) -> str:
            return str(info.get("user_id") or info.get("id") or "").strip()

        seen: Set[str] = set()
        out: List[Dict[str, Any]] = []

        for info in mentionable_users.get("mutual") or []:
            if not isinstance(info, dict):
                continue
            u = _uid(info)
            if not u or u in seen:
                continue
            seen.add(u)
            out.append(info)

        for info in mentionable_users.get("follows") or []:
            if not isinstance(info, dict):
                continue
            if not bool(info.get("is_official", False)):
                continue
            u = _uid(info)
            if not u or u in seen:
                continue
            seen.add(u)
            out.append(info)

        if len(out) > 5:
            out = random.sample(out, 5)
        return out

    async def add_env_tweets(self, key: str, value: Any, parent_event_id: Optional[str] = None) -> bool:
        """
        添加推文到环境中的数据（使用分布式锁）
        
        Args:
            key: 新推文 tweet_id（与 value["tweet_id"] 一致），作为 AddTweetEvent.key 传入环境
            value: 推文数据字典（须含 tweet_id 等字段）
            parent_event_id: 父事件ID（可选）
        """
        # 创建唯一的请求ID
        request_id = f"agent_env_add_tweets_req_{time.time()}_{id(self)}"

        # 创建 Future 用于接收响应
        future = asyncio.Future()
        self._tweet_add_futures[request_id] = future

        # 创建添加推文事件
        tweet_add_event = AddTweetEvent(
            from_agent_id=self.profile_id,  # 请求来源：当前代理
            to_agent_id="ENV",              # 请求目标：环境（特殊目标）
            source_type="AGENT",            # 源类型：代理
            target_type="ENV",              # 目标类型：环境
            key=key,                   # 要更新的数据键（完整格式）
            value=value,                    # 新的数据值
            request_id=request_id,          # 请求ID，用于匹配响应
            parent_event_id=parent_event_id # 父事件ID，用于事件追踪
        )

        # 获取此键的分布式锁
        # 锁ID格式：env_data_add_tweets_lock_{key}，确保每个键有独立的锁
        lock_id = f"env_tweet_add_lock_content_pool"
        lock = await get_lock(lock_id)

        try:
            # 仅在与环境 handler 相同的锁内派发事件，必须在释放锁后再等待响应。
            # 否则 SimEnv.handle_add_tweet_event 无法取得 env_tweet_add_lock_content_pool，会死锁直至超时。
            async with lock:
                from onesim.events import get_event_bus
                event_bus = get_event_bus()
                await event_bus.dispatch_event(tweet_add_event)

            try:
                if hasattr(self, '_sync_event'):
                    await asyncio.wait_for(self._sync_event.wait(), timeout=30.0)
                    return await future
                return await asyncio.wait_for(future, timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning(f"等待环境推文添加超时: {key}")
                self._tweet_add_futures.pop(request_id, None)
                return False
            except Exception as e:
                logger.error(f"添加环境推文时出错: {e}")
                self._tweet_add_futures.pop(request_id, None)
                return False
        except Exception as e:
            logger.error(f"获取环境推文添加锁时出错: {e}")
            return False

    async def handle_add_tweet_response(self, event: AddTweetResponseEvent) -> None:
        """
        处理传入的推文添加响应事件
        """
        # 检查是否正在等待此响应
        if event.request_id in self._tweet_add_futures:
            future = self._tweet_add_futures.pop(event.request_id)

            if not future.done():
                if event.success:
                    future.set_result(True)
                else:
                    future.set_exception(ValueError(event.error or "未知错误"))

            # 如果有同步事件，设置它
            if hasattr(self, '_sync_event'):
                self._sync_event.set()
                # 为下次操作重置
                self._sync_event.clear()

    async def update_env_mention_pool(self, key: str, value: Any, parent_event_id: Optional[str] = None) -> bool:
        """
        更新环境中的mention_pool（使用分布式锁）
        
        Args:
            key: mentioner_id（例如 "69290e59000000001e034ab4"），会自动转换为 "mention_pool.mentioner_id.tweet_id"
            value: mention_pool数据字典，必须包含 mention_key 字段
            parent_event_id: 父事件ID（可选）
        """
        # 将 mentioner_id.tweet_id 转换为完整的 key 格式：mention_pool.mentioner_id.tweet_id
        full_key = f"mention_pool.{key}"
        lock_key = key.split(".")[0]
        
        # 创建唯一的请求ID
        request_id = f"agent_env_update_mention_pool_req_{time.time()}_{id(self)}"

        # 创建 Future 用于接收响应
        future = asyncio.Future()
        self._mention_pool_update_futures[request_id] = future

        # 创建更新mention_pool事件
        mention_pool_update_event = MentionPoolUpdateEvent(
            from_agent_id=self.profile_id,  # 请求来源：当前代理
            to_agent_id="ENV",              # 请求目标：环境（特殊目标）
            source_type="AGENT",            # 源类型：代理
            target_type="ENV",              # 目标类型：环境
            key=full_key,                   # 要更新的数据键（完整格式）
            value=value,                    # 新的数据值
            request_id=request_id,          # 请求ID，用于匹配响应
            parent_event_id=parent_event_id # 父事件ID，用于事件追踪
        )

        # 获取此键的分布式锁
        # 锁ID格式：env_mention_pool_update_lock_{key}，确保每个键有独立的锁
        lock_id = f"env_mention_pool_update_lock_{lock_key}"
        lock = await get_lock(lock_id)

        try:
            async with lock:
                from onesim.events import get_event_bus
                event_bus = get_event_bus()
                await event_bus.dispatch_event(mention_pool_update_event)

            try:
                if hasattr(self, '_sync_event'):
                    await asyncio.wait_for(self._sync_event.wait(), timeout=30.0)
                    return await future
                return await asyncio.wait_for(future, timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning(f"等待环境mention_pool更新超时: {key}")
                self._mention_pool_update_futures.pop(request_id, None)
                return False
            except Exception as e:
                logger.error(f"更新环境mention_pool时出错: {e}")
                self._mention_pool_update_futures.pop(request_id, None)
                return False
        except Exception as e:
            logger.error(f"获取环境mention_pool更新锁时出错: {e}")
            return False

    async def handle_update_mention_pool_response(self, event: MentionPoolUpdateResponseEvent) -> None:
        """
        处理传入的更新mention_pool响应事件
        """
        # 检查是否正在等待此响应
        if event.request_id in self._mention_pool_update_futures:
            future = self._mention_pool_update_futures.pop(event.request_id)

            if not future.done():
                if event.success:
                    future.set_result(True)
                else:
                    future.set_exception(ValueError(event.error or "未知错误"))

            # 如果有同步事件，设置它
            if hasattr(self, '_sync_event'):
                self._sync_event.set()
                # 为下次操作重置
                self._sync_event.clear()

    def _filter_recommendations(self, recommendations: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        筛选推荐内容的接口
        
        过滤掉已经推荐过的内容，避免重复推荐
        """
        if not recommendations:
            return {}
        
        # 获取已推荐过的内容ID集合（从profile中读取）
        recommended_tweet_ids = set(self.profile.get_data("recommended_tweet_ids", [])) if self.profile else set()
        
        # 过滤掉已推荐过的内容
        filtered = {}
        new_tweet_ids = []
        
        for tweet_id, rec in recommendations.items():
            if not isinstance(rec, dict):
                logger.warning(f"Recommendation {tweet_id} is not a dictionary")
                continue
            
            # 如果笔记ID不存在或已经推荐过，跳过
            if not tweet_id or tweet_id in recommended_tweet_ids:
                logger.info(f"Recommendation {tweet_id} is already recommended")
                continue
            
            # 添加到过滤后的列表
            filtered[tweet_id] = rec
            new_tweet_ids.append(tweet_id)
            logger.info(f"Recommendation {tweet_id} is added to filtered list")
        
        return filtered

    async def receive_recommendation(self, event: Event) -> List[Event]:
        """
        接收推荐并决定是否评论。必须本轮的「所有推荐系统 Event」都收到且存在社交推荐，才进行后续处理；
        否则按类型分别存入缓冲，等收齐后再处理。
        """
        current_step = getattr(event, "current_step", 1)
        max_step = getattr(event, "max_step", 8)
        if not await self._should_activate_this_round(current_step, max_step):
            logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} receive recommendation but is not activated this round")
            return []

        # 按推荐类型分别存入缓冲
        if event.__class__.__name__ == "AlgorithmRecommendationEvent":
            source_type = "algorithm"
        elif event.__class__.__name__ == "SocialRecommendationEvent":
            source_type = "social"
        elif event.__class__.__name__ == "SearchRecommendationEvent":
            source_type = "search"
        elif event.__class__.__name__ == "KeepFollowingEvent":
            source_type = "keep_following"
        else:
            evt_cls = event.__class__.__name__
            logger.warning(
                f"Step {current_step}/{max_step}: UserAgent {self.profile_id} "
                f"unknown recommendation event {evt_cls}, skip"
            )
            return []
        
        logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} receive recommendation event, source type: {source_type}, length of recommendations: {len(event.recommendations)}")

        # 将推荐内容加入已推荐列表
        recommendations = event.recommendations
        self._record_recommendations_by_source_step(
            source_type,
            current_step,
            recommendations,
            getattr(event, "timestamp", 0),
        )
        if source_type == "algorithm" or source_type == "social":
            recommendations = self._filter_recommendations(recommendations)
            logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} after filter, length of recommendations: {len(recommendations)}")
        if len(recommendations) == 0:
            logger.info(f"Step {current_step}/{max_step}: No recommendations, skip")
            return []
        # 写入 profile 前快照，用于溯源「链路上是否在过往推荐中见过」（不含本批即将写入的 id）
        prior_recommended_raw = set(self.profile.get_data("recommended_tweet_ids", [])) if self.profile else set()
        chain_hint_seen_tweet_ids = {str(x).strip() for x in prior_recommended_raw if x}
        self._add_recommendations(recommendations)

        # 获取用户信息和可@的用户列表
        user_id = await self.get_data("id")
        user_nickname = await self.get_data("nickname", "")
        user_username = await self.get_data("username", "")
        current_timestamp = event.timestamp
        current_ts = int(event.timestamp) if getattr(event, "timestamp", None) is not None else 0
        step_duration = int(getattr(event, "timestamp_duration", 0) or 0)
        follow_ids = await self.get_data("follow_ids", [])
        fan_ids = await self.get_data("fan_ids", [])
        mentionable_users = self._get_mentionable_users(follow_ids, fan_ids)
        mention_prompt_users = self._mentionable_for_prompt_mutual_and_official_follows(mentionable_users)

        # 构建用户ID和昵称的映射，用于@匹配
        # user_id_map = {}  # nickname -> user_id
        # all_mentionable = mentionable_users.get("follows", []) + \
        #                  mentionable_users.get("fans", []) + \
        #                  mentionable_users.get("mutual", [])
        # for user_info in all_mentionable:
        #     if isinstance(user_info, dict):
        #         user_id_map[user_info.get("user_id", "")] = user_info.get("user_id", "")
        #         user_id_map[user_info.get("nickname", "")] = user_info.get("user_id", "")  # 注意：schema中是nickname
        
        # 构建observation和instruction
        # profile_str = self.get_profile_str(include_private=False) if self.profile else "No profile information"
        # 按「讨论数/嵌套数」折算 chunk 单位动态分块（见 _tweet_recommendation_chunk_units / _pack_recommendation_chunks）
        max_chunk_units = max(1, int(os.environ.get("ONESIM_REC_CHUNK_MAX_UNITS", "3")))
        chunks = UserAgent._pack_recommendation_chunks(recommendations, max_chunk_units)

        events_to_send = []
        has_search = False

        content_pool = await self.get_env_data("content_pool", {}) or {}
        if not isinstance(content_pool, dict):
            content_pool = {}
        seeds_raw = await self.get_env_data("seed_root_tweet_ids", []) or []
        seed_ids = {str(x).strip() for x in seeds_raw if str(x).strip()}

        time_ref_ms = self._sim_ts_to_ms_for_time_module(getattr(event, "timestamp", None))

        for chunk in chunks:
            # n_hop：与 _hop_edges_to_env_seed_root / _format_env_propagation_depth_hint 使用同一套 hop 计数
            chunk_for_llm: Dict[str, Any] = {}
            for tid, tw in chunk.items():
                inner = UserAgent._tweet_dict_for_llm_observation(
                    UserAgent._tweet_discussion_tree_content_only(tw)
                )
                tid_key = UserAgent._tweet_ref_key(tid) or (
                    UserAgent._tweet_ref_key(tw.get("tweet_id") or tw.get("id"))
                    if isinstance(tw, dict)
                    else None
                )
                dep = (
                    UserAgent._hop_edges_to_env_seed_root(str(tid_key), content_pool, seed_ids)
                    if tid_key
                    else None
                )
                if isinstance(inner, dict):
                    inner = dict(inner)
                    inner["n_hop"] = dep
                chunk_for_llm[tid] = inner
            recommendations_str = json.dumps(chunk_for_llm, ensure_ascii=False, indent=2)
            mentionable_users_str = json.dumps(mention_prompt_users, ensure_ascii=False, indent=2)

            # Label recommendation source；若推荐里包含自己发布的内容，优先标记为“自己发布”
            has_self_tweet = any(
                isinstance(tweet, dict) and tweet.get("user_id") == user_id
                for tweet in chunk.values()
            )
            if has_self_tweet:
                source_name = (
                    "Your own post [⚠ Hint: you authored this tweet; mind context and avoid replying as if to yourself.]"
                )
            else:
                source_name = (
                    "Following feed (from accounts you follow)"
                    if source_type == "social"
                    else "Algorithmic recommendations"
                )

            chain_hint_lines: List[str] = []
            for rec_tid, rec_tweet in chunk.items():
                if not isinstance(rec_tweet, dict):
                    continue
                tid_key = UserAgent._tweet_ref_key(rec_tid) or UserAgent._tweet_ref_key(
                    rec_tweet.get("tweet_id") or rec_tweet.get("id")
                )
                if not tid_key:
                    continue
                one_hint = self._quote_reply_chain_hint(
                    rec_tweet,
                    my_id=user_id,
                    recommended_tweet_ids=chain_hint_seen_tweet_ids,
                )
                if one_hint:
                    chain_hint_lines.append(f"(tweet_id={tid_key}) {one_hint}")
            quote_reply_chain_hint = "\n".join(chain_hint_lines)

            if quote_reply_chain_hint:
                source_name = f"{source_name}\nChain tweets (quote/reply):\n{quote_reply_chain_hint}"

            env_depth_lines: List[str] = []
            for rec_tid, rec_tweet in chunk.items():
                if not isinstance(rec_tweet, dict):
                    continue
                tid_key = UserAgent._tweet_ref_key(rec_tid) or UserAgent._tweet_ref_key(
                    rec_tweet.get("tweet_id") or rec_tweet.get("id")
                )
                if not tid_key:
                    continue
                dep = UserAgent._hop_edges_to_env_seed_root(tid_key, content_pool, seed_ids)
                root = UserAgent._resolve_tweet_to_env_seed_root(tid_key, content_pool, seed_ids)
                env_depth_lines.append(
                    f"(tweet_id={tid_key}) {UserAgent._format_env_propagation_depth_hint(dep, root)}"
                )
            env_depth_block = UserAgent._format_env_depth_coaching_block(env_depth_lines)
            step2_five_rec = (
                env_depth_block
                if env_depth_block and random.random() < UserAgent._STEP2_FIVE_ENV_DEPTH_PROB
                else ""
            )

            time_module_str = self._time_module_for_recommendation_chunk(chunk, time_ref_ms)

            observation = f"""[Scenario] You are scrolling a phone feed: skim most items; only rarely stop to type a line. You are not running an experiment task or writing a media analysis.

            Feed source: {source_name}

            Recommended content:
            {recommendations_str}

            Users you may @:
            {mentionable_users_str}
            """

            interaction_threshold = InteractionThreshold.sample(random.Random())
            k_same_target = interaction_threshold.k_same_target
            k_diff_targets = interaction_threshold.k_diff_targets
            k_propagation_type = interaction_threshold.propagation_type

            raw_act = self.profile.get_data("activity_level", 0.0) if self.profile else 0.0
            try:
                _activity = float(raw_act)
            except (TypeError, ValueError):
                _activity = 0.0
            _activity = max(0.0, min(1.0, _activity))

            topic_txt = topic_text_from_tweets_chunk(chunk, content_pool)
            s15_ev = await self.evaluate_step15_policies(topic_text=topic_txt)
            mem = s15_ev.get("memory_nonempty") or {}
            kw = s15_ev.get("keyword") or {}
            emb = s15_ev.get("embedding") or {}
            mem_ok = bool(mem.get("inject"))
            kw_ok = bool(kw.get("inject"))
            emb_ok = bool(emb.get("inject"))

            step15_kw_coaching = (
                "\n\n【Topic vs memory — keyword overlap】\n            "
                "- This batch is judged to be strongly keyword-related to stored memory → **strong bias toward propagation=false** (high risk of repeating stances or same-thread discussion already in memory; consider repost=true only if step 1.5 clearly meets break-out conditions such as verifiable new information and strong motivation);"
            ) if kw_ok else ""
            step15_emb_coaching = (
                "\n\n【Topic vs memory — semantic similarity】\n            "
                "- Embedding similarity meets the configured threshold; this batch is close in meaning to memory → **strong bias toward propagation=false** (treat as same thread / easy duplicate topic; strictly re-check step 1.5 before repost=true);"
            ) if emb_ok else ""

            if (_activity < 0.85 and emb_ok):
                step15_receive = """
                Step 1.5: Check for \"duplicate topic\" against memory (complete before step 2)
                - If the current content refers to the same incident / dispute / issue as memory, and your stance, tone, or conclusion in memory would closely match what you would say in this reply → treat this as \"almost certainly keep the default of propagation=false\", unless you can state clearly that, relative to what is already in memory, you will add **a new fact readers can perceive** (you must be able to name it: a specific person, event, time, rule, or number; rephrasing alone or vague \"new angle\" / \"new reasoning\" does not count).
                - If memory already has multiple entries on the same kind of topic, or you have recently engaged on similar content → lean toward propagation=false this round unless there is **materially new development** (new information, or the other party raises an argument not covered in memory).
                - On the recommendation feed with weak ties, or when your relationship with the author is weak, be more conservative about making an exception.
                - **memory_reflection must not contradict itself**: if you first say it overlaps with memory / same topic / already discussed / related to a prior event, then use \"but\" / \"although\" / \"things may have updated\" / \"might still weigh in\" / \"still worth a brief comment\" or similar turns **without a concrete new fact** to hint that you should comment — treat that as invalid; if you judge overlap or near–no engagement, memory_reflection must **throughout** conclude toward silence or clearly no new information, and must not use vague wording to let yourself off the hook.
                - **The more you have in memory, the more you should hold back**: even if (1) and (2) can be met on paper, still treat **propagation=true for this item as a low-probability event**—default stays propagation=false; break the rule only when new information **clearly escalates** the situation (e.g. changes the phase of the event, overturns or corrects a judgment already in your memory, or introduces a key new actor or rule). Do not repost just because you \"barely\" satisfy the bar with a throwaway line.
                - **Also disallowed**: admitting the same thread as memory, then using vague lines like \"but this gives a new instance of the problem\" / \"new situation\" / \"another case\" / \"worth noting\" / \"worth further attention\" / \"new informational anchor\" to imply you should comment — unless the same sentence states **one namable difference** beyond memory (e.g. a phenomenon, time, link, or rule name **unique** to this post).
                Summarize the above briefly in memory_reflection; if you do not comment, decision_reason must state overlap with memory / already stated / no new information, etc.
                """
                # step15_receive = """
                # Step 1.5: Check for \"duplicate topic\" against memory (complete before step 2)
                # - If the current content refers to the same incident / dispute / issue as memory, and your stance, tone, or conclusion in memory would closely match what you would say in this reply → treat this as \"almost certainly keep the default of propagation=false\", unless you can state clearly that, relative to what is already in memory, you will add **a new fact readers can perceive** (you must be able to name it: a specific person, event, time, rule, or number; rephrasing alone or vague \"new angle\" / \"new reasoning\" does not count).
                # - If memory already has multiple entries on the same kind of topic, or you have recently engaged on similar content → lean toward propagation=false this round unless there is **materially new development** (new information, or the other party raises an argument not covered in memory).
                # - On the recommendation feed with weak ties, or when your relationship with the author is weak, be more conservative about making an exception.
                # - **memory_reflection must not contradict itself**: if you first say it overlaps with memory / same topic / already discussed / related to a prior event, then use \"but\" / \"although\" / \"things may have updated\" / \"might still weigh in\" / \"still worth a brief comment\" or similar turns **without a concrete new fact** to hint that you should comment — treat that as invalid; if you judge overlap or near–no engagement, memory_reflection must **throughout** conclude toward silence or clearly no new information, and must not use vague wording to let yourself off the hook.
                # - **Also disallowed**: admitting the same thread as memory, then using vague lines like \"but this gives a new instance of the problem\" / \"new situation\" / \"another case\" / \"worth noting\" / \"worth further attention\" / \"new informational anchor\" to imply you should comment — unless the same sentence states **one namable difference** beyond memory (e.g. a phenomenon, time, link, or rule name **unique** to this post).
                # Summarize the above briefly in memory_reflection; if you do not comment, decision_reason must state overlap with memory / already stated / no new information, etc.
                # """
                step2_zero_rec = "0. Not trapped by step 1.5 \"almost no engagement\", or you have a verifiable new point;"
                mem_refl_rec = "2-3 sentences: same event as memory? already stated? should stay silent? if exception, name the new info"
            # elif mem_ok:
            #     step15_receive = """
            #     Step 1.5: \"Duplicate topic\" check against memory (complete before step 2)
            #     - If the current content refers to the same incident / dispute / issue as memory, and your stance, tone, or conclusion in memory would closely match what you would say if you engaged now → **do not treat propagation=false as mandatory**, but assume **propagation=true is low-probability** unless you can state clearly that, relative to what is already in memory, you will add **a new fact, angle, or line of reasoning readers can perceive** (rephrasing alone does not count).
            #     - If memory already has multiple entries on the same kind of topic, or you have recently interacted on similar content → **high prior** toward propagation=false (not a hard rule); propagation=true is reasonable only if there is **materially new development** (new information, or an argument not covered in memory).
            #     - If memory already shows you responded on this kind of content → again **high prior** toward propagation=false; you may still set propagation=true if there is **clear new information** or a **meaningful stance shift**—name it in memory_reflection / decision_reason.
            #     - On the algorithm feed with weak ties, or when your relationship with the author is weak, treat propagation=true as **even less likely** (same prior toward false, stricter bar for exceptions).
            #     Summarize the above briefly in memory_reflection; if you do not engage, decision_reason may state overlap with memory / already stated / no new information, etc.
            #     """
            #     step2_zero_rec = "0. Not trapped by step 1.5 \"almost no engagement\", or you have a verifiable new point;"
            #     mem_refl_rec = "2-3 sentences: same event as memory? already stated? should stay silent? if exception, name the new info"
            else:
                # step15_receive = """
                # Step 1.5: \"Duplicate topic\" check against memory (complete before step 2)
                # - If the current content refers to the same incident / dispute / issue as memory, and your stance, tone, or conclusion in memory would closely match what you would say if you engaged now → treat this as \"almost certainly keep the default propagation=false\", unless you can state clearly that, relative to what is already in memory, you will add **a new fact, angle, or line of reasoning readers can perceive** (rephrasing alone does not count).
                # - If memory already has multiple entries on the same kind of topic, or you have recently interacted on similar content → lean toward propagation=false this round unless there is **materially new development** (new information, or an argument not covered in memory).
                # - If memory already shows you responded on this kind of content, treat this as \"almost certainly keep the default propagation=false\".
                # - On the algorithm feed with weak ties, or when your relationship with the author is weak, be more conservative about making an exception.
                # Summarize the above briefly in memory_reflection; if you do not engage, decision_reason must state overlap with memory / already stated / no new information, etc.
                # """
                step15_receive = ""
                step2_zero_rec = ""
                mem_refl_rec = "1–2 sentences. When there is no stored memory to compare against, write \"no relevant memory / first exposure\"."

            if k_propagation_type == "retweet":
                step5_rec = """
                - propagation_type ∈ {{"retweet","reply","quote"}}. Default "retweet".
                · "retweet" — Pure amplify: you believe the post is worth surfacing (e.g. timely, credible, useful, funny, or important to your audience) and you want followers to see it **as-is**—no added stance, no thread commentary; propagation_content "".
                · "reply" — You address the author or a concrete point voice aimed at the author.
                · "quote" — You add judgment, context, or extension for the curious onlookers; voice aimed at the audience.
                """
            elif k_propagation_type == "reply":
                step5_rec = """
                - propagation_type ∈ {{"retweet","reply","quote"}}. Default "reply".
                · "retweet" — Pure amplify: you believe the post is worth surfacing (e.g. timely, credible, useful, funny, or important to your audience) and you want followers to see it **as-is**—no added stance, no thread commentary; propagation_content "".
                · "reply" — You address the author or a concrete point; voice aimed at the author.
                · "quote" — You add judgment, context, or extension for the curious onlookers; voice aimed at the audience.
                """            
            elif k_propagation_type == "quote":
                step5_rec = """
                - propagation_type ∈ {{"retweet","reply","quote"}}. Default "quote".
                · "retweet" — You only agree/amplify without adding stance; propagation_content "".
                · "reply" — You address the author or a concrete point; voice aimed at the author.
                · "quote" — You add judgment, context, or extension for the curious onlookers; voice aimed at the audience.
                """   

            if time_module_str:
                time_coaching_block = (
                    "【Simulated time & recency】\n            "
                    + time_module_str
                    + " - If the text above contains **【WARNING】** or states that recency has clearly faded / you lean toward propagation=false → **strong bias toward propagation=false** (do not put this item in the candidate pool);"
                )
            else:
                time_coaching_block = ""

            instruction = f"""Using the user's Profile, historical_summary, memory, and the recommended tweets, produce tweet decisions and text.

            Step 1 — Default each recommendation
            - "propagation": false
            - "propagation_type": "" (empty string: no propagation type chosen yet, same as no engagement)
            - "propagation_content": ""

            **Only steps 2/3/4 decide propagation counts; do not change them because of step 6.**
            {step15_receive}

            Step 2 — Interest gate (still no propagation_type/mode)d
            - Read all items in this batch; set propagation=true only if **all** of the following hold:
                {step2_zero_rec}
                1. Relevant to Profile/historical_summary/memory;
                2. Relationship and scene fit (prefer follows);
                3. You see heat/reply value and clear intent to respond or amplify;
                4. If the tweet is already in a quote/reply chain (replied_tweet or quoted_tweet non-empty), lean toward propagation=false, further propagation needs clear new stance, explanation, context, or extension;
            - Items that fail interest: not in the candidate pool, propagation=false.

            Step 3 — At most {k_diff_targets} target tweets
            - From the sorted candidate list (interest-passing items first), pick **at most** {k_diff_targets} distinct tweet_ids to engage with; **zero is allowed** if nothing qualifies.
            - Never exceed {k_diff_targets} different tweet_ids with propagation=true in this batch.

            Step 4 — At most {k_same_target} engagement(s) per target tweet_id
            - For each target tweet_id you engage with, output **at most** {k_same_target} propagation=true rows (fewer if one line is enough).
            - If {k_same_target}=1: at most one engagement row per target tweet_id.
            - If {k_same_target}>=2: you may use 2+ rows only when each adds something (new point / correction / stronger emotion); no duplicate lines; **never more than {k_same_target}** rows for the same tweet_id.

            Step 5 — propagation_type per engagement
            {step5_rec}

            If you choose empty-text forward, propagation_type must be "retweet"; skip steps 6 and 7.

            Step 6 — propagation_mode (persona)
            - propagation_mode ∈ {{analysis/event narrative/advice, emotion/memes/roast, question/chat, emoji/placeholder}}; reflect in expression_reason.
                - analysis/event narrative/advice: facts, mechanisms, details, calls to action; denser; max ~40 words.
                - emotion/memes/roast: attitude-first; max ~25 words.
                - question/chat: push the thread; max ~30 words.
                - emoji/placeholder: emoji or very short; max ~1 word. e.g. 🤯😇👌😅🤣😨😰😱🙈🤩😂
            - Output style:
                - Prefer the user's locale; default English.
                - Match Profile and historical_summary voice;
                - Avoid canned phrases ("you're so right", "indeed", "hope everyone…");
                - **No hashtags (#...) in propagation_content.**

            Step 7 — @ mentions
            - @only if the user is in the allow-list, strongly relevant, and not the tweet author / parent retweet author; else mention_reasoning is [].

            Step 8 — keep_following_tweet_ids
            - Only if highly interested and worth tracking → non-empty; else [].

            Step 9 — search
            - search=true only if highly interested, info clearly insufficient, and memory has nothing useful; else false.

            Return JSON (field order fixed):
            {{
            "persona_understanding": "1-2 sentences: role, interests, voice",
            "content_understanding": "1-2 sentences: relevance and angle",
            "source_understanding": "following feed = social ties; algorithm = unknown authors; your own post = self-authored",
            "memory_reflection": "{mem_refl_rec}",
            "decisions": [
                {{
                "tweet_id": "tweet_id",
                "propagation": false,
                "propagation_type": "",
                "propagation_content": "",
                "decision_reason": "why skip (1 short line, <20 chars)",
                "expression_reason": "",
                "mention_reasoning": []
                }},
                {{
                "tweet_id": "tweet_id",
                "propagation": true,
                "propagation_type": "retweet",
                "propagation_content": "",
                "decision_reason": "why engage (1 short line, <20 chars)",
                "expression_reason": "",
                "mention_reasoning": []
                }},
                {{
                "tweet_id": "tweet_id",
                "propagation": true,
                "propagation_type": "reply",
                "propagation_content": "short reply (quote text if quoting)",
                "decision_reason": "why engage (1 short line, <20 chars)",
                "expression_reason": "tone; substantive/micro if applicable (1 line)",
                "mention_reasoning": [
                    {{
                    "user_id": "mentioned user id",
                    "persona_understanding": "brief read of that user",
                    "mention_reason": "why ping them (1 short line, <20 chars)"
                    }}
                ]
                }}
            ],
            "keep_following_tweet_ids": [],
            "keep_following_reason": "why keep following (1 short line, <20 chars)",
            "search": false or true,
            "search_keyword": "keyword for search, <len<=20 chars>",
            "search_reason": "search or not and why (1 line)",
            }}

            Output rules:
            - Fill understanding fields first, then decisions;
            - Each decision row starts as propagation=false / propagation_type ""; flip to true only where rules pass (single type preferred; "reply|quote" allowed but not ideal);
            - If propagation=false, propagation_type must be ""; if propagation=true and propagation_type is exactly "retweet", propagation_content must be "".
            - Align propagation_content with propagation_mode and persona; avoid boilerplate.
            {step15_kw_coaching}{step15_emb_coaching}
            {step2_five_rec}{time_coaching_block}"""

            # 调用LLM生成决策
            logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} generate reaction")
            env = getattr(self, "env", None)
            if env is not None and hasattr(env, "notify_agent_busy"):
                await env.notify_agent_busy()
            try:
                response = await self.generate_reaction(instruction, observation)
            finally:
                if env is not None and hasattr(env, "notify_agent_idle"):
                    await env.notify_agent_idle()
            logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} generate reaction response")

            # 处理搜索决策
            search = response.get("search", False)
            if search and not has_search:
                has_search = True
                # 发送事件给推荐系统：请求“指定算法”的推荐（算法类型由代码固定指定）
                search_algorithm_types = self.default_search_types
                if not isinstance(search_algorithm_types, list) or not search_algorithm_types:
                    raise ValueError("default_search_types must be a non-empty list")
                allowed_search_types = set(self.search_map.keys())

                # 遍历所有指定算法类型，发送事件给推荐系统
                for search_algorithm_type in search_algorithm_types:
                    if search_algorithm_type not in allowed_search_types:
                        raise ValueError(
                            f"Invalid algorithm type '{search_algorithm_type}'. "
                            f"Allowed types: {sorted(allowed_search_types)}"
                        )
                    mapped_id = self.search_map.get(search_algorithm_type, "")
                    if not mapped_id:
                        raise ValueError(
                            f"No recommender agent mapped for algorithm type '{search_algorithm_type}'. "
                            f"Current mapping keys: {sorted(self.search_map.keys())}"
                        )
                        continue

                    # 获取用户画像
                    profile_payload = {}
                    if self.profile is not None:
                        try:
                            profile_payload = dict(self.profile.get_profile(include_private=True) or {})
                        except Exception:
                            logger.warning("Failed to serialize profile via get_profile(), fallback to empty payload.")
                    logger.info(f"Sending SearchEvent to RecommenderAgent {mapped_id} for search algorithm type {search_algorithm_type}")
                    events_to_send.append(SearchEvent(
                        from_agent_id=self.profile_id,
                        to_agent_id=mapped_id,
                        timestamp=event.timestamp,
                        timestamp_duration=int(getattr(event, "timestamp_duration", 0) or 0),
                        current_step=current_step,
                        max_step=max_step,
                        user_profile=profile_payload,
                        algorithm_type=search_algorithm_type,
                        search_query=str(response.get("search_keyword") or "").strip(),
                    ))

            # 处理保持关注决策
            if self.profile:
                keep_ids = response.get("keep_following_tweet_ids", [])
                if isinstance(keep_ids, list) and keep_ids:
                    # 只允许本批次内的 tweet_id，且最多 1 个
                    valid_keep_ids = []
                    for keep_tweet_id in keep_ids:
                        if keep_tweet_id in chunk:
                            valid_keep_ids.append(keep_tweet_id)
                    if valid_keep_ids:
                        self.profile.update_data("keep_following_tweet_ids", valid_keep_ids[:1])
                    else:
                        self.profile.update_data("keep_following_tweet_ids", [])

            # 处理评论决策
            decisions = response.get("decisions", [])
            if not isinstance(decisions, list):
                continue
       
            # 处理每个决策：更新传播数和传播内容
            for decision in decisions:
                if not isinstance(decision, dict):
                    continue
                    
                tweet_id = decision.get("tweet_id")
                should_propagation = decision.get("propagation", False)
                propagation_type = decision.get("propagation_type", "")
                propagation_content = decision.get("propagation_content", "")

                if not tweet_id or not should_propagation:
                    continue
                # 转推无正文，propagation_content 应为 ""；与 handle_mention 分支一致
                if propagation_type != "retweet" and not (propagation_content or "").strip():
                    continue

                # 检查 tweet_id 是否合法
                if tweet_id not in chunk:
                    logger.warning(f"Step {current_step}/{max_step}: Tweet {tweet_id} not found in recommendations")
                    continue

                tweet = chunk[tweet_id]
                if not isinstance(tweet, dict):
                    tweet = {}

                # 解析转发内容中的@用户并收集用户ID列表
                mentioned_user_ids = []
                mention_reasoning = decision.get("mention_reasoning", [])
                if isinstance(mention_reasoning, list):
                    for mention_reason in mention_reasoning:
                        if isinstance(mention_reason, dict):
                            muid = mention_reason.get("user_id")
                            if muid:
                                mentioned_user_ids.append(muid)

                path_author_user_ids = self._collect_quote_reply_path_author_user_ids(tweet_id, chunk)
                path_author_set = {str(x).strip() for x in path_author_user_ids if x}
                # LLM 给出的 @ 中，去掉已在引用/回复链路上的作者（避免重复计数与重复提醒）
                filtered_mention_ids: Set[str] = set()
                for x in mentioned_user_ids:
                    if x is not None:
                        sx = str(x).strip()
                        if sx and sx not in path_author_set:
                            filtered_mention_ids.add(sx)
                filtered_mention_ids.discard(user_id)
                mention_count = len(filtered_mention_ids)
                
                # 如果为传播，添加传播
                # 生成唯一的传播ID
                propagation_id = self._generate_propagation_id()
                    
                if propagation_type == "retweet":
                    success = await self.add_env_tweets(propagation_id, {
                        "tweet_id": propagation_id,
                        "content": "",
                        "time": self._random_propagation_timestamp(tweet, current_ts, step_duration),
                        "user_id": user_id,
                        "nickname": user_nickname,
                        "username": user_username,
                        "mention_count": mention_count,
                        "retweeted_tweet_id": tweet_id
                    })
                    if not success:
                        logger.error(f"Failed to add retweet to tweet {tweet_id}")
                        continue
                elif propagation_type == "quote":
                    success = await self.add_env_tweets(propagation_id, {
                        "tweet_id": propagation_id,
                        "content": propagation_content,
                        "time": self._random_propagation_timestamp(tweet, current_ts, step_duration),
                        "user_id": user_id,
                        "nickname": user_nickname,
                        "username": user_username,
                        "mention_count": mention_count,
                        "quoted_tweet_id": tweet_id
                    })
                    if not success:
                        logger.error(f"Failed to add quote to tweet {tweet_id}")
                        continue
                elif propagation_type == "reply":
                    success = await self.add_env_tweets(propagation_id, {
                        "tweet_id": propagation_id,
                        "content": propagation_content,
                        "time": self._random_propagation_timestamp(tweet, current_ts, step_duration),
                        "user_id": user_id,
                        "nickname": user_nickname,
                        "username": user_username,
                        "mention_count": mention_count,
                        "replied_tweet_id": tweet_id
                    })
                    if not success:
                        logger.error(f"Failed to add reply to tweet {tweet_id}")
                        continue

                # 向引用/回复嵌套链上的作者发提醒（不含自己）
                path_notify_ids = sorted(
                    {
                        s
                        for s in (str(x).strip() for x in path_author_user_ids if x)
                        if s and s != user_id
                    }
                )
                for target_uid in path_notify_ids:
                    success = await self.update_env_mention_pool(f"{target_uid}.{propagation_id}", {
                        "action": "add",
                        "mention_message": {
                            "tweet_id": tweet_id,
                            "mention_type": propagation_type
                        }
                    })
                    if not success:
                        logger.error(f"Failed to update mention pool for {propagation_type} {propagation_id} by {user_id} -> chain author {target_uid} on tweet {tweet_id}")
                    else:
                        logger.info(f"Step {current_step}/{max_step}: User {user_id} {propagation_type} to tweet {tweet_id}")

                # 发送MentionEvent给被@的用户
                if filtered_mention_ids:
                    # 为每个被@的用户创建MentionEvent
                    for mentioned_user_id in filtered_mention_ids:
                        if mentioned_user_id and mentioned_user_id != user_id:  # 不给自己发提醒
                            # 创建@事件，发送给被@的用户
                            success = await self.update_env_mention_pool(f"{mentioned_user_id}.{propagation_id}", {
                                "action": "add",
                                "mention_message": {
                                    "tweet_id": tweet_id,
                                    "mention_type": "at"
                                }
                            })
                            if not success:
                                logger.error(f"Failed to update mention pool for tweet {tweet_id} by {user_id}")
                                continue
                            logger.info(f"Step {current_step}/{max_step}: User {user_id} mentioned {mentioned_user_id} in tweet {tweet_id}")
                    
       
        # 通过事件通知环境更新内容池
        content_update_event = RecommendationSpreadingEvent(
            from_agent_id=self.profile_id,
            to_agent_id="ENV",
            timestamp=current_timestamp,
            timestamp_duration=int(getattr(event, "timestamp_duration", 0) or 0),
            current_step=current_step,
            max_step=max_step,
        )
        events_to_send.append(content_update_event)
        
        return events_to_send
    
    def _check_relationship(self, my_id: str,other_user_id: str, follow_ids: List[str], fan_ids: List[str]) -> str:
        """
        检查与另一个用户的关系类型

        Args:
            other_user_id: 另一个用户的ID
            
        Returns:
            str: 关系类型 - "mutual"（互关）、"follow"（关注）、"fan"（粉丝）、"none"（无关系）
        """
        if not other_user_id:
            return "none"
        
        if not my_id or my_id == other_user_id:
            return "none"
        
        # 检查是否是互关
        if other_user_id in follow_ids and other_user_id in fan_ids:
            return "mutual"
        
        # 检查是否是我关注的人
        if other_user_id in follow_ids:
            return "follow"
        
        # 检查是否是我的粉丝
        if other_user_id in fan_ids:
            return "fan"

        return "none"

    def _merge_mentions(
        self, 
        old_messages: Dict[str, Any], 
        new_messages: Dict[str, Any], 
        current_ts: int = 0,
        last_check: int = 0,
        period_ms: int = 24 * 3600 * 1000,
        mention_cap: int = 20,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        合并提醒消息
        """
        merged = {**old_messages, **new_messages}

        if current_ts >= last_check + period_ms:
            if len(merged) > mention_cap:
                return {}, {}
            return merged, {}
        else:
            return {}, merged

    async def handle_mention(self, event: MentionEvent) -> List[Event]:
        """
        处理转发/引用/回复提醒事件
        
        当用户被转发/引用/回复时，更容易进行转发/引用/回复。
        - 转发提醒（mention_type="retweet"）
        - 引用提醒（mention_type="quote"）
        - 回复提醒（mention_type="reply"）

        传播深度（hop）与 receive_recommendation 一致：写入每条帖子 JSON 的 n_hop（_hop_edges_to_env_seed_root；无法溯源则为 null）。
        observation 中不再重复粘贴纯深度文案；quote_reply_chain_hint 等链路透示（如已在转发/引用链上）仍保留。
        深度相关的策略说明主要在 instruction 末尾的 Step 2.5（env_depth_block / step2_five_rec），与帖内 n_hop 配合使用。
        
        Args:
            event: MentionEvent，包含转发/引用/回复信息
            
        Returns:
            List[Event]: 返回要发送的事件列表（通常是转发/引用/回复事件）
        """
        current_step = getattr(event, "current_step", 1)
        max_step = getattr(event, "max_step", 8)
        if not await self._should_activate_this_round(current_step, max_step):
            logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} handle mention but is not activated this round")
            return []

        # 获取提醒信息
        mentions = getattr(event, "mentions", {}) or {}
        if not mentions:
            return []

        logger.info(f"Step {current_step}/{max_step}: UserAgent {self.profile_id} receive mention event, length of mentions: {len(mentions)}")

        # 获取可@的用户列表
        follow_ids = await self.get_data("follow_ids", [])
        fan_ids = await self.get_data("fan_ids", [])
        my_id = await self.get_data("id")
        mentionable_users = self._get_mentionable_users(follow_ids, fan_ids)
        mentionable_users_str = json.dumps(mentionable_users.get("follows", []), ensure_ascii=False, indent=2)

        # 构建用户ID和昵称的映射，用于@匹配
        # user_id_map = {}
        # all_mentionable = mentionable_users.get("follows", []) + mentionable_users.get("fans", []) + mentionable_users.get("mutual", [])
        # for user_info in all_mentionable:
        #     if isinstance(user_info, dict):
        #         user_id_map[user_info.get("user_id", "")] = user_info.get("user_id", "")
        #         user_id_map[user_info.get("nickname", "")] = user_info.get("user_id", "")
        # profile_str = self.get_profile_str(include_private=False) if self.profile else "No profile information"

        recommended_raw = set(self.profile.get_data("recommended_tweet_ids", [])) if self.profile else set()
        recommended_tweet_ids = {str(x).strip() for x in recommended_raw if x}

        content_pool = await self.get_env_data("content_pool", {}) or {}
        if not isinstance(content_pool, dict):
            content_pool = {}
        seeds_raw = await self.get_env_data("seed_root_tweet_ids", []) or []
        seed_ids = {str(x).strip() for x in seeds_raw if str(x).strip()}

        # 构建提醒信息
        mention_entries: List[Dict[str, Any]] = []
        for mention_key, mention_message in mentions.items():
            mention_type = mention_message.get("mention_type", "retweet")
            if mention_type == "retweet":
                continue

            mention_tweet = mention_message.get("tweet")
            if not isinstance(mention_tweet, dict):
                continue
            mentioner_id = mention_tweet.get("user_id")
            mentioner_nickname = mention_tweet.get("nickname", "")
            mentioner_username = mention_tweet.get("username", "")
            
            relationship_type = self._check_relationship(my_id, mentioner_id, follow_ids, fan_ids)
            if relationship_type == "mutual":
                relationship_hint = " (mutual follow)"
            elif relationship_type == "follow":
                relationship_hint = " (you follow them)"
            else:
                relationship_hint = ""

            if mention_type in ("quote"):
                mention_action = f"{mentioner_nickname}{relationship_hint} quoted your tweet"
                content_label = "Quoted content"
            elif mention_type == "reply":
                mention_action = f"{mentioner_nickname}{relationship_hint} replied to your tweet"
                content_label = "Reply content"
            else:
                mention_action = f"{mentioner_nickname}{relationship_hint} mentioned your tweet ({mention_type})"
                content_label = "Related content"

            if relationship_hint:
                relationship_source = f"Relationship with {mentioner_nickname}: {relationship_hint}"
            else:
                relationship_source = f"Relationship with {mentioner_nickname}: stranger / no strong tie"

            tweet_id = mention_tweet.get("tweet_id") or (mention_key.split("_")[0] if "_" in str(mention_key) else str(mention_key))
            quote_reply_chain_hint = self._quote_reply_chain_hint(
                mention_tweet,
                my_id=my_id,
                recommended_tweet_ids=recommended_tweet_ids,
            )
            dep_m = UserAgent._hop_edges_to_env_seed_root(str(tweet_id), content_pool, seed_ids)
            root_m = UserAgent._resolve_tweet_to_env_seed_root(str(tweet_id), content_pool, seed_ids)
            env_propagation_depth_hint = UserAgent._format_env_propagation_depth_hint(dep_m, root_m)
            mention_entries.append({
                "mention_key": mention_key,
                "mention_tweet": mention_tweet,
                "tweet_id": tweet_id,
                "mentioner_id": mentioner_id,
                "mentioner_nickname": mentioner_nickname,
                "mentioner_username": mentioner_username,
                "mention_type": mention_type,
                "mention_action": mention_action,
                "content_label": content_label,
                "relationship_source": relationship_source,
                "quote_reply_chain_hint": quote_reply_chain_hint,
                "env_propagation_depth_hint": env_propagation_depth_hint,
                "n_hop": dep_m,
            })

        if not mention_entries:
            return []

        # 按「讨论数/嵌套数」折算 chunk 单位动态分块（与 receive_recommendation 一致，见 _tweet_recommendation_chunk_units / _pack_recommendation_chunks）
        max_chunk_units = max(1, int(os.environ.get("ONESIM_REC_CHUNK_MAX_UNITS", "3")))
        tw_map = {e["mention_key"]: e["mention_tweet"] for e in mention_entries}
        chunk_dicts = UserAgent._pack_recommendation_chunks(tw_map, max_chunk_units)
        key_to_entry = {e["mention_key"]: e for e in mention_entries}
        mention_chunks: List[List[Dict[str, Any]]] = []
        for ch in chunk_dicts:
            sub: List[Dict[str, Any]] = []
            for mk in ch.keys():
                ent = key_to_entry.get(mk)
                if ent is not None:
                    sub.append(ent)
            if sub:
                mention_chunks.append(sub)

        if not mention_chunks:
            return []

        self._record_mentioned_tweet_ids_by_channel(
            current_step,
            mention_entries,
            getattr(event, "timestamp", 0),
        )

        events_to_send: List[Event] = []
        has_reply = False
        user_id = await self.get_data("id")
        nickname = await self.get_data("nickname", "")
        username = await self.get_data("username", "")
        current_timestamp = event.timestamp
        current_ts = int(event.timestamp) if getattr(event, "timestamp", None) is not None else 0
        step_duration = int(getattr(event, "timestamp_duration", 0) or 0)

        total_batches = len(mention_chunks)
        time_ref_ms = self._sim_ts_to_ms_for_time_module(getattr(event, "timestamp", None))

        for batch_idx, chunk_entries in enumerate(mention_chunks):
            batch_k = len(chunk_entries)
            batch_tweet_ids = {
                str(e["tweet_id"]).strip() for e in chunk_entries if e.get("tweet_id") is not None
            }

            observation_parts = []
            for j, entry in enumerate(chunk_entries):
                tw_for_llm = UserAgent._tweet_dict_for_llm_observation(
                    UserAgent._tweet_discussion_tree_content_only(entry["mention_tweet"])
                )
                if isinstance(tw_for_llm, dict):
                    tw_for_llm = dict(tw_for_llm)
                    tw_for_llm["n_hop"] = entry.get("n_hop")
                observation_parts.append(f"""## Alert {j + 1}
            {entry["mention_action"]}

            Relationship with the commenter/poster (relationship_understanding must follow this; do not invent or invert):
            {entry["relationship_source"]}

            {entry["content_label"]}:
            {json.dumps(tw_for_llm, ensure_ascii=False, indent=2)}
            {entry.get("quote_reply_chain_hint", "")}
            """)

            chunk_for_time = [{"mention_tweet": e["mention_tweet"]} for e in chunk_entries]
            time_module_str = self._time_module_for_recommendation_chunk(chunk_for_time, time_ref_ms)

            env_depth_lines: List[str] = []
            for entry in chunk_entries:
                tw = entry.get("mention_tweet") if isinstance(entry.get("mention_tweet"), dict) else {}
                tid_key = UserAgent._tweet_ref_key(entry.get("tweet_id")) or UserAgent._tweet_ref_key(
                    tw.get("tweet_id") or tw.get("id")
                )
                if not tid_key:
                    continue
                dep = UserAgent._hop_edges_to_env_seed_root(tid_key, content_pool, seed_ids)
                root = UserAgent._resolve_tweet_to_env_seed_root(tid_key, content_pool, seed_ids)
                env_depth_lines.append(
                    f"(tweet_id={tid_key}) {UserAgent._format_env_propagation_depth_hint(dep, root)}"
                )
            env_depth_block = UserAgent._format_env_depth_coaching_block(env_depth_lines)
            step2_five_rec = (
                env_depth_block
                if env_depth_block and random.random() < UserAgent._STEP2_FIVE_ENV_DEPTH_PROB
                else ""
            )

            observation = (
                f"[Batch] {batch_idx + 1}/{total_batches}; this batch has {batch_k} alerts. **decisions length must be {batch_k}** (matches Alert 1 … Alert {batch_k} below).\n\n"
                + "[Scene] You got a notification (quote/reply). You may ignore; reply only if worth it. Keep replies short and chat-like, not a briefing.\n\n"
            )
            observation += (
                f"Quotes/replies ({batch_k} alerts; decide in order):\n\n"
                + "\n".join(observation_parts)
                + "\nUsers you may @:\n"
                + mentionable_users_str
            )

            interaction_threshold = InteractionThreshold.sample(random.Random())
            k_mention_type = interaction_threshold.mention_type

            raw_act = self.profile.get_data("activity_level", 0.0) if self.profile else 0.0
            try:
                _activity = float(raw_act)
            except (TypeError, ValueError):
                _activity = 0.0
            _activity = max(0.0, min(1.0, _activity))

            topic_txt = topic_text_from_mention_entries(chunk_entries, content_pool)
            s15_ev = await self.evaluate_step15_policies(topic_text=topic_txt)
            mem = s15_ev.get("memory_nonempty") or {}
            kw = s15_ev.get("keyword") or {}
            emb = s15_ev.get("embedding") or {}
            mem_ok = bool(mem.get("inject"))
            kw_ok = bool(kw.get("inject"))
            emb_ok = bool(emb.get("inject"))

            step15_kw_coaching = (
                "\n\n【Topic vs memory — keyword overlap】\n            "
                "- This batch is judged to be strongly keyword-related to stored memory → **strong bias toward propagation=false** (high risk of repeating stances or same-thread discussion already in memory; consider repost=true only if step 1.5 clearly meets break-out conditions such as verifiable new information and strong motivation);"
            ) if kw_ok else ""
            step15_emb_coaching = (
                "\n\n【Topic vs memory — semantic similarity】\n            "
                "- Embedding similarity meets the configured threshold; this batch is close in meaning to memory → **strong bias toward propagation=false** (treat as same thread / easy duplicate topic; strictly re-check step 1.5 before repost=true);"
            ) if emb_ok else ""

            if (_activity < 0.85 and emb_ok):
                # step15_receive = """
                # Step 1.5: Check for \"duplicate topic\" against memory (complete before step 2)
                # - If the current content refers to the same incident / dispute / issue as memory, and your stance, tone, or conclusion in memory would closely match what you would say in this reply → treat this as \"almost certainly keep the default of propagation=false\", unless you can state clearly that, relative to what is already in memory, you will add **a new fact readers can perceive** (you must be able to name it: a specific person, event, time, rule, or number; rephrasing alone or vague \"new angle\" / \"new reasoning\" does not count).
                # - If memory already has multiple entries on the same kind of topic, or you have recently engaged on similar content → lean toward propagation=false this round unless there is **materially new development** (new information, or the other party raises an argument not covered in memory).
                # - On the recommendation feed with weak ties, or when your relationship with the author is weak, be more conservative about making an exception.
                # - **memory_reflection must not contradict itself**: if you first say it overlaps with memory / same topic / already discussed / related to a prior event, then use \"but\" / \"although\" / \"things may have updated\" / \"might still weigh in\" / \"still worth a brief comment\" or similar turns **without a concrete new fact** to hint that you should comment — treat that as invalid; if you judge overlap or near–no engagement, memory_reflection must **throughout** conclude toward silence or clearly no new information, and must not use vague wording to let yourself off the hook.
                # - **The more you have in memory, the more you should hold back**: even if (1) and (2) can be met on paper, still treat **propagation=true for this item as a low-probability event**—default stays propagation=false; break the rule only when new information **clearly escalates** the situation (e.g. changes the phase of the event, overturns or corrects a judgment already in your memory, or introduces a key new actor or rule). Do not repost just because you \"barely\" satisfy the bar with a throwaway line.
                # - **Also disallowed**: admitting the same thread as memory, then using vague lines like \"but this gives a new instance of the problem\" / \"new situation\" / \"another case\" / \"worth noting\" / \"worth further attention\" / \"new informational anchor\" to imply you should comment — unless the same sentence states **one namable difference** beyond memory (e.g. a phenomenon, time, link, or rule name **unique** to this post).
                # Summarize the above briefly in memory_reflection; if you do not comment, decision_reason must state overlap with memory / already stated / no new information, etc.
                # """
                step15_receive = """
                Step 1.5: Check for \"duplicate topic\" against memory (complete before step 2)
                - If the current content refers to the same incident / dispute / issue as memory, and your stance, tone, or conclusion in memory would closely match what you would say in this reply → treat this as \"almost certainly keep the default of propagation=false\", unless you can state clearly that, relative to what is already in memory, you will add **a new fact readers can perceive** (you must be able to name it: a specific person, event, time, rule, or number; rephrasing alone or vague \"new angle\" / \"new reasoning\" does not count).
                - If memory already has multiple entries on the same kind of topic, or you have recently engaged on similar content → lean toward propagation=false this round unless there is **materially new development** (new information, or the other party raises an argument not covered in memory).
                - On the recommendation feed with weak ties, or when your relationship with the author is weak, be more conservative about making an exception.
                - **memory_reflection must not contradict itself**: if you first say it overlaps with memory / same topic / already discussed / related to a prior event, then use \"but\" / \"although\" / \"things may have updated\" / \"might still weigh in\" / \"still worth a brief comment\" or similar turns **without a concrete new fact** to hint that you should comment — treat that as invalid; if you judge overlap or near–no engagement, memory_reflection must **throughout** conclude toward silence or clearly no new information, and must not use vague wording to let yourself off the hook.
                - **Also disallowed**: admitting the same thread as memory, then using vague lines like \"but this gives a new instance of the problem\" / \"new situation\" / \"another case\" / \"worth noting\" / \"worth further attention\" / \"new informational anchor\" to imply you should comment — unless the same sentence states **one namable difference** beyond memory (e.g. a phenomenon, time, link, or rule name **unique** to this post).
                Summarize the above briefly in memory_reflection; if you do not comment, decision_reason must state overlap with memory / already stated / no new information, etc.
                """
                step2_zero_rec = "0. Not trapped by step 1.5 \"almost no engagement\", or you have a verifiable new point;"
                mem_refl_rec = "2-3 sentences: same event as memory? already stated? should stay silent? if exception, name the new info"
            # elif mem_ok:
            #     step15_receive = """
            #     Step 1.5: \"Duplicate topic\" check against memory (complete before step 2)
            #     - If the current content refers to the same incident / dispute / issue as memory, and your stance, tone, or conclusion in memory would closely match what you would say if you engaged now → **do not treat propagation=false as mandatory**, but assume **propagation=true is low-probability** unless you can state clearly that, relative to what is already in memory, you will add **a new fact, angle, or line of reasoning readers can perceive** (rephrasing alone does not count).
            #     - If memory already has multiple entries on the same kind of topic, or you have recently interacted on similar content → **high prior** toward propagation=false (not a hard rule); propagation=true is reasonable only if there is **materially new development** (new information, or an argument not covered in memory).
            #     - If memory already shows you responded on this kind of content → again **high prior** toward propagation=false; you may still set propagation=true if there is **clear new information** or a **meaningful stance shift**—name it in memory_reflection / decision_reason.
            #     - On the algorithm feed with weak ties, or when your relationship with the author is weak, treat propagation=true as **even less likely** (same prior toward false, stricter bar for exceptions).
            #     Summarize the above briefly in memory_reflection; if you do not engage, decision_reason may state overlap with memory / already stated / no new information, etc.
            #     """
            #     step2_zero_rec = "0. Not trapped by step 1.5 \"almost no engagement\", or you have a verifiable new point;"
            #     mem_refl_rec = "2-3 sentences: same event as memory? already stated? should stay silent? if exception, name the new info"
            else:
                # step15_receive = """
                # Step 1.5: \"Duplicate topic\" check against memory (complete before step 2)
                # - If the current content refers to the same incident / dispute / issue as memory, and your stance, tone, or conclusion in memory would closely match what you would say if you engaged now → treat this as \"almost certainly keep the default propagation=false\", unless you can state clearly that, relative to what is already in memory, you will add **a new fact, angle, or line of reasoning readers can perceive** (rephrasing alone does not count).
                # - If memory already has multiple entries on the same kind of topic, or you have recently interacted on similar content → lean toward propagation=false this round unless there is **materially new development** (new information, or an argument not covered in memory).
                # - If memory already shows you responded on this kind of content, treat this as \"almost certainly keep the default propagation=false\".
                # - On the algorithm feed with weak ties, or when your relationship with the author is weak, be more conservative about making an exception.
                # Summarize the above briefly in memory_reflection; if you do not engage, decision_reason must state overlap with memory / already stated / no new information, etc.
                # """
                step15_receive = ""
                step2_zero_rec = ""
                mem_refl_rec = "1–2 sentences. When there is no stored memory to compare against, write \"no relevant memory / first exposure\"."

            if k_mention_type == "retweet":
                step3_rec = """
                - propagation_type ∈ {{"retweet","reply","quote"}}. Default "retweet".
                · "retweet" — Pure amplify: you believe the post is worth surfacing (e.g. timely, credible, useful, funny, or important to your audience) and you want followers to see it **as-is**—no added stance, no thread commentary; propagation_content "".
                · "reply" — You address the author or a concrete point; voice aimed at the author.
                · "quote" — You add judgment, context, or extension for the curious onlookers; voice aimed at the audience.
                """
            elif k_mention_type == "reply":
                step3_rec = """
                - propagation_type ∈ {{"retweet","reply","quote"}}. Default "reply".
                · "retweet" — You only agree/amplify without adding stance; propagation_content "".
                · "reply" — You address the author or a concrete point; voice aimed at the author.
                · "quote" — You add judgment, context, or extension for the curious onlookers; voice aimed at the audience.
                """            
            elif k_mention_type == "quote":
                step3_rec = """
                - propagation_type ∈ {{"retweet","reply","quote"}}. Default "quote".
                · "retweet" — You only agree/amplify without adding stance; propagation_content "".
                · "reply" — You address the author or a concrete point voice aimed at the author.
                · "quote" — You add judgment, context, or extension for the curious onlookers; voice aimed at the audience.
                """   
            if time_module_str:
                time_coaching_block = (
                    "【Simulated time & recency】\n            "
                    + time_module_str
                    + " - If the text above contains **【WARNING】** or states that recency has clearly faded / you lean toward propagation=false → **strong bias toward propagation=false**;"
                )
            else:
                time_coaching_block = ""

            instruction = f"""This batch has {batch_k} alerts (batch {batch_idx + 1} of {total_batches}). Decide each alert in order (reply or not, text, etc.). **decisions length must equal {batch_k}**; item i matches alert i.

            Use Profile, historical_summary, memory, and relationships in Observation to decide engagement and write text.

            Step 1 — Default each row
            - "propagation": false
            - "propagation_type": "" (empty: no propagation type chosen; same as no engagement)
            - "propagation_content": ""

            **Only steps 2/3 control propagation counts; do not override because of step 5.**
            {step15_receive}

            Step 2 — Interest gate (still no propagation_type/mode)d
            - Read all items in this batch; set propagation=true only if **all** of the following hold:
                {step2_zero_rec}
                1. Relevant to Profile/historical_summary/memory;
                2. Relationship and scene fit (prefer follows);
                3. You see heat/reply value and clear intent to respond or amplify;
                4. If the tweet is already in a quote/reply chain (replied_tweet or quoted_tweet non-empty), lean toward propagation=false, further propagation needs clear new stance, explanation, context, or extension;
            - Items that fail interest: not in the candidate pool, propagation=false.

            Step 3 — propagation_type per engagement
            -{step3_rec}

            If empty-text forward, propagation_type must be "retweet"; skip step 4.

            Step 4 — propagation_mode (persona)
            - propagation_mode ∈ {{analysis/event narrative/advice, emotion/memes/roast, question/chat, emoji/placeholder}}; reflect in expression_reason.
                - analysis/event narrative/advice: facts, mechanisms, details; max ~40 words.
                - emotion/memes/roast: attitude-first; max ~25 words.
                - question/chat: move the thread; max ~30 words.
                - emoji/placeholder: emoji or very short; max ~1 word. e.g. 🤯😇👌😅🤣😨😰😱🙈🤩😂
            - Output style:
                - Prefer the user's locale; default English.
                - Match Profile and historical_summary voice;
                - Avoid canned phrases ("you're so right", "indeed", "hope everyone…");
                - **No hashtags (#...) in propagation_content.**

            Step 5 — @ mentions
            - @only if user is in the allow-list, strongly relevant, and not the tweet author / parent retweet author; else mention_reasoning [].

            Step 6 — keep_following_tweet_ids
            - Only if highly interested and worth tracking; else [].

            Step 7 — search
            - search=true only if highly interested, info insufficient, and memory unhelpful; else false.

            Return JSON (field order fixed):
            {{
            "persona_understanding": "1-2 sentences: role, interests, voice",
            "content_understanding": "1-2 sentences: relevance of this @/comment to you",
            "relationship_understanding": "Follow Observation exactly; do not invent relationship types",
            "memory_reflection": "{mem_refl_rec}",
            "decisions": [
                {{
                "tweet_id": "tweet_id",
                "propagation": false,
                "propagation_type": "",
                "propagation_content": "",
                "decision_reason": "why skip (1 short line, <20 chars)",
                "expression_reason": "",
                "mention_reasoning": []
                }},
                {{
                "tweet_id": "tweet_id",
                "propagation": true,
                "propagation_type": "retweet",
                "propagation_content": "",
                "decision_reason": "why engage (1 short line, <20 chars)",
                "expression_reason": "",
                "mention_reasoning": []
                }},
                {{
                "tweet_id": "tweet_id",
                "propagation": true,
                "propagation_type": "reply|quote",
                "propagation_content": "short reply or quote text",
                "decision_reason": "why engage (1 short line, <20 chars)",
                "expression_reason": "tone; substantive/micro if applicable (1 line)",
                "mention_reasoning": [
                    {{
                    "user_id": "mentioned user id",
                    "persona_understanding": "brief read of that user",
                    "mention_reason": "why ping them (1 short line, <20 chars)"
                    }}
                ]
                }}
            ],
            "keep_following_tweet_ids": [],
            "keep_following_reason": "why keep following (1 short line, <20 chars)",
            "search": false,
            search
            "search_reason": "search or not and why (1 line)",
            }}
            Output rules:
            - Understanding fields first, then decisions;
            - Start each decision as propagation=false / propagation_type ""; flip to true only where rules pass (single type preferred; "reply|quote" allowed but not ideal);
            - If propagation=false, propagation_type must be ""; if propagation=true and propagation_type is exactly "retweet", propagation_content must be "".
            - Align propagation_content with propagation_mode and persona; avoid boilerplate.
            {step15_kw_coaching}{step15_emb_coaching}
            {step2_five_rec}{time_coaching_block}"""

            logger.info(
                f"Step {current_step}/{max_step}: UserAgent {self.profile_id} handle_mention batch "
                f"{batch_idx + 1}/{total_batches} ({batch_k} entries), max_chunk_units={max_chunk_units}"
            )
            response = await self.generate_reaction(instruction, observation)

            search = response.get("search", False)
            if search:
                # 发送事件给推荐系统：请求“指定算法”的推荐（算法类型由代码固定指定）
                search_algorithm_types = self.default_search_types
                if not isinstance(search_algorithm_types, list) or not search_algorithm_types:
                    raise ValueError("default_search_types must be a non-empty list")
                allowed_search_types = set(self.search_map.keys())

                # 遍历所有指定算法类型，发送事件给推荐系统
                for search_algorithm_type in search_algorithm_types:
                    if search_algorithm_type not in allowed_search_types:
                        raise ValueError(
                            f"Invalid algorithm type '{search_algorithm_type}'. "
                            f"Allowed types: {sorted(allowed_search_types)}"
                        )
                    mapped_id = self.search_map.get(search_algorithm_type, "")
                    if not mapped_id:
                        raise ValueError(
                            f"No recommender agent mapped for algorithm type '{search_algorithm_type}'. "
                            f"Current mapping keys: {sorted(self.search_map.keys())}"
                        )
                        continue

                    # 获取用户画像
                    profile_payload = {}
                    if self.profile is not None:
                        try:
                            profile_payload = dict(self.profile.get_profile(include_private=True) or {})
                        except Exception:
                            logger.warning("Failed to serialize profile via get_profile(), fallback to empty payload.")
                    logger.info(f"Sending SearchEvent to RecommenderAgent {mapped_id} for search algorithm type {search_algorithm_type}")
                    events_to_send.append(SearchEvent(
                        from_agent_id=self.profile_id,
                        to_agent_id=mapped_id,
                        timestamp=event.timestamp,
                        timestamp_duration=int(getattr(event, "timestamp_duration", 0) or 0),
                        current_step=current_step,
                        max_step=max_step,
                        user_profile=profile_payload,
                        algorithm_type=search_algorithm_type,
                        search_query=str(response.get("search_keyword") or "").strip(),
                    ))

            if self.profile:
                keep_ids = response.get("keep_following_tweet_ids", [])
                if isinstance(keep_ids, list) and keep_ids:
                    valid_keep_ids = []
                    for keep_tweet_id in keep_ids:
                        sk = str(keep_tweet_id).strip() if keep_tweet_id is not None else ""
                        if sk and sk in batch_tweet_ids:
                            valid_keep_ids.append(keep_tweet_id)
                    if valid_keep_ids:
                        self.profile.update_data("keep_following_tweet_ids", valid_keep_ids[:1])
                    else:
                        self.profile.update_data("keep_following_tweet_ids", [])

            decisions = response.get("decisions", [])
            if not isinstance(decisions, list):
                logger.warning(
                    f"Step {current_step}/{max_step}: UserAgent {self.profile_id} handle_mention batch "
                    f"{batch_idx + 1}/{total_batches}: invalid decisions (not a list), skip batch"
                )
                continue

            for i, decision in enumerate(decisions):
                if i >= len(chunk_entries):
                    break
                mention_entry = chunk_entries[i]
                mention_tweet = mention_entry["mention_tweet"]
                mention_tweet_id = mention_entry["tweet_id"]
                try:
                    if not isinstance(decision, dict) or not decision.get("propagation", False):
                        continue

                    tweet_id = decision.get("tweet_id")
                    should_propagation = decision.get("propagation", False)
                    propagation_type = decision.get("propagation_type", "")
                    propagation_content = decision.get("propagation_content", "")

                    if not tweet_id or not should_propagation:
                        continue
                    if propagation_type != "retweet" and not (propagation_content or "").strip():
                        continue

                    has_reply = True

                    if tweet_id != mention_tweet_id:
                        logger.warning(
                            f"Tweet {tweet_id} does not match mention tweet {mention_tweet_id}, skipping"
                        )
                        continue

                    mentioned_user_ids = []
                    mention_reasoning = decision.get("mention_reasoning", [])
                    if isinstance(mention_reasoning, list):
                        for mention_reason in mention_reasoning:
                            if isinstance(mention_reason, dict):
                                muid = mention_reason.get("user_id")
                                if muid:
                                    mentioned_user_ids.append(muid)

                    path_author_user_ids = self._collect_quote_reply_path_author_user_ids(
                        mention_tweet, None
                    )
                    path_author_set = {str(x).strip() for x in path_author_user_ids if x}
                    filtered_mention_ids: Set[str] = set()
                    for x in mentioned_user_ids:
                        if x is not None:
                            sx = str(x).strip()
                            if sx and sx not in path_author_set:
                                filtered_mention_ids.add(sx)
                    filtered_mention_ids.discard(user_id)
                    mention_count = len(filtered_mention_ids)

                    propagation_id = self._generate_propagation_id()
                    ts = self._random_propagation_timestamp(mention_tweet, current_ts, step_duration)

                    if propagation_type == "retweet":
                        success = await self.add_env_tweets(
                            propagation_id,
                            {
                                "tweet_id": propagation_id,
                                "content": propagation_content,
                                "time": ts,
                                "user_id": user_id,
                                "nickname": nickname,
                                "username": username,
                                "mention_count": mention_count,
                                "retweeted_tweet_id": tweet_id,
                            },
                        )
                        if not success:
                            logger.error(f"Failed to add retweet to tweet {tweet_id}")
                            continue
                    elif propagation_type == "quote":
                        success = await self.add_env_tweets(
                            propagation_id,
                            {
                                "tweet_id": propagation_id,
                                "content": propagation_content,
                                "time": ts,
                                "user_id": user_id,
                                "nickname": nickname,
                                "username": username,
                                "mention_count": mention_count,
                                "quoted_tweet_id": tweet_id,
                            },
                        )
                        if not success:
                            logger.error(f"Failed to add quote to tweet {tweet_id}")
                            continue
                    elif propagation_type == "reply":
                        success = await self.add_env_tweets(
                            propagation_id,
                            {
                                "tweet_id": propagation_id,
                                "content": propagation_content,
                                "time": ts,
                                "user_id": user_id,
                                "nickname": nickname,
                                "username": username,
                                "mention_count": mention_count,
                                "replied_tweet_id": tweet_id,
                            },
                        )
                        if not success:
                            logger.error(f"Failed to add reply to tweet {tweet_id}")
                            continue
                    else:
                        continue

                    path_notify_ids = sorted(
                        {
                            s
                            for s in (str(x).strip() for x in path_author_user_ids if x)
                            if s and s != user_id
                        }
                    )
                    for target_uid in path_notify_ids:
                        success = await self.update_env_mention_pool(
                            f"{target_uid}.{propagation_id}",
                            {
                                "action": "add",
                                "mention_message": {
                                    "tweet_id": tweet_id,
                                    "mention_type": propagation_type,
                                },
                            },
                        )
                        if not success:
                            logger.error(
                                f"Failed to update mention pool for {propagation_type} {propagation_id} "
                                f"by {user_id} -> chain author {target_uid} on tweet {tweet_id}"
                            )
                        else:
                            logger.info(
                                f"Step {current_step}/{max_step}: User {user_id} {propagation_type} to tweet {tweet_id}"
                            )

                    if filtered_mention_ids:
                        for mentioned_user_id in filtered_mention_ids:
                            if mentioned_user_id and mentioned_user_id != user_id:
                                success = await self.update_env_mention_pool(
                                    f"{mentioned_user_id}.{propagation_id}",
                                    {
                                        "action": "add",
                                        "mention_message": {
                                            "tweet_id": tweet_id,
                                            "mention_type": "at",
                                        },
                                    },
                                )
                                if not success:
                                    logger.error(
                                        f"Failed to update mention pool for tweet {tweet_id} by {user_id}"
                                    )
                                    continue
                                logger.info(
                                    f"Step {current_step}/{max_step}: User {user_id} mentioned "
                                    f"{mentioned_user_id} in tweet {tweet_id}"
                                )
                finally:
                    success = await self.update_env_mention_pool(
                        f"{user_id}.{mention_tweet_id}",
                        {"action": "delete", "mention_message": None},
                    )
                    if not success:
                        logger.error(f"Failed to update mention pool for comment {mention_tweet_id}")
                    else:
                        logger.info(
                            f"Step {current_step}/{max_step}: User {user_id} deleted mention "
                            f"{mention_tweet_id} from pool (processed)"
                        )

        if has_reply:
            content_update_event = MentionSpreadingEvent(
                from_agent_id=self.profile_id,
                to_agent_id="ENV",
                timestamp=current_timestamp,
                timestamp_duration=int(getattr(event, "timestamp_duration", 0) or 0),
                current_step=current_step,
                max_step=max_step,
            )
            events_to_send.append(content_update_event)

        return events_to_send
    