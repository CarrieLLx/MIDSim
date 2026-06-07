# -*- coding: utf-8 -*-
"""
步骤 1.5（重复话题 / memory 对齐）是否注入 instruction：可配置策略。

环境变量（均在运行时读取）：
- ONESIM_STEP15_POLICY: 单个或**多个**策略，用英文逗号 `,`、竖线 `|` 或加号 `+` 分隔，例如
    `memory_nonempty,keyword`、`keyword|embedding`、`memory_nonempty+keyword+embedding`。
    合法 token：memory_nonempty | keyword | embedding | hybrid_or | hybrid_and（非法片段会跳过）。
    默认 `memory_nonempty`（兼容旧行为）。
- ONESIM_STEP15_MULTI_COMBINE: 多策略时如何合成最终是否注入步骤 1.5：`or`（任一子策略为真则注入，默认）
    或 `and`（全部子策略为真才注入）。
- ONESIM_STEP15_KEYWORD_ENABLED: 1/0，是否在 keyword / hybrid_* 中启用关键词重叠（默认 1）。
- ONESIM_STEP15_EMBEDDING_ENABLED: 1/0，是否在 embedding / hybrid_* 中启用向量相似度（默认 0，避免额外 HTTP）。
- ONESIM_STEP15_MIN_COMMON_TOKENS: 话题与 memory 侧共同词下限（默认 2）。
- ONESIM_STEP15_MIN_JACCARD: Jaccard 下限，0 表示仅用共同词（默认 0）。
- ONESIM_STEP15_EMBED_THRESHOLD: 余弦相似度阈值（默认 0.65）。
- ONESIM_STEP15_INCLUDE_HISTORICAL_SUMMARY: 1/0，memory 侧是否拼上 profile.historical_summary（默认 1）。
- ONESIM_STEP15_EMBEDDING_CONFIG_PATH: 覆盖 model_config.json 路径（默认可空，用项目根 config/model_config.json）。
- ONESIM_STEP15_EMBED_MAX_CHARS: 单段最大字符数（默认 400）；超长文本会切成多段分别 embedding。
- ONESIM_STEP15_EMBED_CHUNK_AGG: 多段聚合方式 mean（各侧向量取平均再比余弦，默认）| max（取跨段最大余弦）。
- ONESIM_STEP15_EMBED_MAX_CHUNKS: 每一侧最多嵌入几段（默认 12，防止 memory 极长时请求过多）。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

# 轻量停用词（中英混合场景下仅作粗筛）
_STOPWORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "and", "or", "but", "not", "this", "that", "these", "those", "it", "its",
    "的", "了", "在", "是", "和", "与", "或", "及", "有", "为", "对", "中", "上", "下",
    "不", "吗", "嘛", "啊", "哦", "嗯", "吧", "呀", "么", "呢", "着", "过", "还", "就",
    "一个", "可以", "这样", "这个", "我们", "你们", "他们", "什么", "怎么", "没有",
    "自己", "如果", "因为", "所以", "但是", "然后", "而且", "或者",
}


def _find_project_root() -> str:
    path = os.path.abspath(os.path.dirname(__file__))
    for _ in range(12):
        cfg = os.path.join(path, "config", "model_config.json")
        if os.path.isfile(cfg):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return os.getcwd()


def _tokenize(text: str) -> Set[str]:
    if not text or not str(text).strip():
        return set()
    raw = re.findall(r"[\w\u4e00-\u9fff]+", str(text).lower())
    out: Set[str] = set()
    for w in raw:
        w = w.strip().lower()
        if len(w) < 2:
            continue
        if w in _STOPWORDS:
            continue
        out.add(w)
    return out


def keyword_topic_overlap(
    topic_text: str,
    memory_side_text: str,
    *,
    min_common: int = 2,
    min_jaccard: float = 0.0,
) -> bool:
    """话题文本与 memory 侧文本的词重叠：共同词数或 Jaccard 过阈则 True。"""
    a = _tokenize(topic_text)
    b = _tokenize(memory_side_text)
    if not a or not b:
        return False
    inter = a & b
    if len(inter) >= min_common:
        return True
    if min_jaccard > 0.0:
        union = a | b
        if not union:
            return False
        j = len(inter) / len(union)
        return j >= min_jaccard
    return False


def _chunk_text_for_embedding(text: str, max_chars: int) -> List[str]:
    """将长文本切成多段，每段不超过 max_chars；优先在换行处合并，超长行再硬切。"""
    t = (text or "").strip()
    if not t:
        return []
    mc = max(32, int(max_chars))
    if len(t) <= mc:
        return [t]
    out: List[str] = []
    cur = ""
    for line in t.split("\n"):
        line = line.strip()
        if not line:
            continue
        cand = f"{cur}\n{line}".strip() if cur else line
        if len(cand) <= mc:
            cur = cand
        else:
            if cur:
                out.append(cur)
            if len(line) <= mc:
                cur = line
            else:
                for i in range(0, len(line), mc):
                    out.append(line[i : i + mc])
                cur = ""
    if cur:
        out.append(cur)
    return [c for c in out if c]


def _mean_embedding(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    acc = [0.0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            acc[i] += float(x)
    n = float(len(vectors))
    return [x / n for x in acc]


def _embed_chunks_sequential(
    get_embeddings: Any,
    base_url: str,
    model_name: str,
    chunks: List[str],
) -> List[List[float]]:
    """每段单独请求，避免服务端把批量 input 的 token 加总后超上下文。"""
    vecs: List[List[float]] = []
    for ch in chunks:
        batch = get_embeddings(base_url, model_name, [ch])
        if len(batch) != 1:
            raise ValueError("embedding API 单段返回数量异常")
        vecs.append(batch[0])
    return vecs


def embedding_topic_similarity(
    topic_text: str,
    memory_side_text: str,
    *,
    config_path: str,
) -> Optional[float]:
    """
    返回 topic 与 memory 侧文本 embedding 后的相似度；失败返回 None。
    超长文本按 ONESIM_STEP15_EMBED_MAX_CHARS 切成多段，逐段请求后用 mean 或 max 聚合（见环境变量）。
    同步函数，建议在 asyncio.to_thread 中调用。
    """
    try:
        from .metrics.repost_similarity import (
            cosine_similarity,
            get_embeddings,
            load_embedding_config,
        )
    except ImportError:
        from metrics.repost_similarity import (
            cosine_similarity,
            get_embeddings,
            load_embedding_config,
        )

    max_chars = int(os.environ.get("ONESIM_STEP15_EMBED_MAX_CHARS", "400") or "400")
    max_chars = max(64, min(max_chars, 8000))
    max_chunks = int(os.environ.get("ONESIM_STEP15_EMBED_MAX_CHUNKS", "12") or "12")
    max_chunks = max(1, min(max_chunks, 64))
    agg = os.environ.get("ONESIM_STEP15_EMBED_CHUNK_AGG", "mean").strip().lower()
    if agg not in ("mean", "max"):
        agg = "mean"

    chunks_t = _chunk_text_for_embedding(topic_text or "", max_chars)[:max_chunks]
    chunks_m = _chunk_text_for_embedding(memory_side_text or "", max_chars)[:max_chunks]
    if not chunks_t or not chunks_m:
        return None
    try:
        base_url, model_name = load_embedding_config(config_path)
        vt = _embed_chunks_sequential(get_embeddings, base_url, model_name, chunks_t)
        vm = _embed_chunks_sequential(get_embeddings, base_url, model_name, chunks_m)
        if not vt or not vm:
            return None
        if agg == "max":
            best = -1.0
            for a in vt:
                for b in vm:
                    best = max(best, float(cosine_similarity(a, b)))
            return best
        mt = _mean_embedding(vt)
        mm = _mean_embedding(vm)
        return float(cosine_similarity(mt, mm))
    except Exception as e:
        try:
            from loguru import logger

            detail = ""
            resp = getattr(e, "response", None)
            if resp is not None:
                try:
                    detail = f" | response={resp.text[:400]!r}"
                except Exception:
                    detail = f" | status={getattr(resp, 'status_code', '')}"
            logger.warning(
                f"step15 embedding_topic_similarity 失败（将视为无向量相似度）: {e}{detail}"
            )
        except Exception:
            pass
        return None


_VALID_POLICIES = frozenset(
    ("memory_nonempty", "keyword", "embedding", "hybrid_or", "hybrid_and")
)


def _parse_policy_list(raw: str) -> Tuple[str, ...]:
    """从环境变量解析策略列表，去重、保序。"""
    s = (raw or "").strip().lower()
    if not s:
        return ("memory_nonempty",)
    parts = re.split(r"[,|+]+", s)
    seen: Set[str] = set()
    out: List[str] = []
    for p in parts:
        p = p.strip().lower()
        if not p or p in seen:
            continue
        if p not in _VALID_POLICIES:
            try:
                from loguru import logger

                logger.warning(
                    f"ONESIM_STEP15_POLICY: skip unknown token {p!r}, "
                    f"valid={sorted(_VALID_POLICIES)}"
                )
            except Exception:
                pass
            continue
        seen.add(p)
        out.append(p)
    return tuple(out) if out else ("memory_nonempty",)


@dataclass(frozen=True)
class Step15GateConfig:
    """policy_raw: 环境变量原串；policies: 归一化后的多策略（有序）。"""
    policies: Tuple[str, ...]
    policy_raw: str
    multi_combine: str
    keyword_enabled: bool
    embedding_enabled: bool
    min_common: int
    min_jaccard: float
    embed_threshold: float
    include_historical_summary: bool
    embedding_config_path: str


def load_step15_gate_config() -> Step15GateConfig:
    policy_raw = os.environ.get("ONESIM_STEP15_POLICY", "memory_nonempty,keyword,embedding").strip()
    policies = _parse_policy_list(policy_raw)

    combine = os.environ.get("ONESIM_STEP15_MULTI_COMBINE", "or").strip().lower()
    if combine not in ("or", "and"):
        combine = "or"

    kw = os.environ.get("ONESIM_STEP15_KEYWORD_ENABLED", "1").strip() not in ("0", "false", "False", "")
    emb = os.environ.get("ONESIM_STEP15_EMBEDDING_ENABLED", "1").strip() in ("1", "true", "True", "yes", "Yes")

    min_common = int(os.environ.get("ONESIM_STEP15_MIN_COMMON_TOKENS", "2") or "2")
    min_jaccard = float(os.environ.get("ONESIM_STEP15_MIN_JACCARD", "0") or "0")
    embed_th = float(os.environ.get("ONESIM_STEP15_EMBED_THRESHOLD", "0.65") or "0.65")
    inc_hist = os.environ.get("ONESIM_STEP15_INCLUDE_HISTORICAL_SUMMARY", "1").strip() not in ("0", "false", "False", "")

    cfg_path = os.environ.get("ONESIM_STEP15_EMBEDDING_CONFIG_PATH", "").strip()
    if not cfg_path:
        cfg_path = os.path.join(_find_project_root(), "config", "model_config.json")

    return Step15GateConfig(
        policies=policies,
        policy_raw=policy_raw,
        multi_combine=combine,
        keyword_enabled=kw,
        embedding_enabled=emb,
        min_common=max(1, min_common),
        min_jaccard=max(0.0, min_jaccard),
        embed_threshold=max(0.0, min(1.0, embed_th)),
        include_historical_summary=inc_hist,
        embedding_config_path=cfg_path,
    )


def _inject_for_single_policy(
    pol: str,
    cfg: Step15GateConfig,
    *,
    memory_nonempty: bool,
    mem_side: str,
    topic_text: str,
) -> Dict[str, Any]:
    """
    对单个策略名计算是否建议注入步骤 1.5，并返回结构化结果（便于日志与调试）。
    返回值至少含键 inject: bool；embedding 含 similarity；hybrid 含 keyword_hit、embedding_hit。
    """
    if pol == "memory_nonempty":
        return {"inject": bool(memory_nonempty)}

    if pol == "keyword":
        if not cfg.keyword_enabled:
            return {"inject": bool(memory_nonempty), "blog": "keyword_disabled_fallback_memory_nonempty"}
        if not mem_side:
            return {"inject": False}
        hit = keyword_topic_overlap(
            topic_text,
            mem_side,
            min_common=cfg.min_common,
            min_jaccard=cfg.min_jaccard,
        )
        return {"inject": bool(hit)}

    if pol == "embedding":
        if not cfg.embedding_enabled:
            return {"inject": bool(memory_nonempty), "blog": "embedding_disabled_fallback_memory_nonempty"}
        if not mem_side:
            return {"inject": False, "similarity": None}
        sim = embedding_topic_similarity(
            topic_text,
            mem_side,
            config_path=cfg.embedding_config_path,
        )
        if sim is None:
            return {"inject": False, "similarity": None}
        return {
            "inject": bool(sim >= cfg.embed_threshold),
            "similarity": float(sim),
            "threshold": float(cfg.embed_threshold),
        }

    # hybrid_or / hybrid_and
    kw_ok = False
    if cfg.keyword_enabled and mem_side:
        kw_ok = keyword_topic_overlap(
            topic_text,
            mem_side,
            min_common=cfg.min_common,
            min_jaccard=cfg.min_jaccard,
        )

    emb_ok = False
    sim_val: Optional[float] = None
    if cfg.embedding_enabled and mem_side:
        sim = embedding_topic_similarity(
            topic_text,
            mem_side,
            config_path=cfg.embedding_config_path,
        )
        if sim is not None:
            sim_val = float(sim)
            emb_ok = sim_val >= cfg.embed_threshold

    if not cfg.keyword_enabled and not cfg.embedding_enabled:
        return {
            "inject": bool(memory_nonempty),
            "blog": "hybrid_but_kw_emb_disabled_fallback_memory_nonempty",
        }

    if pol == "hybrid_or":
        if cfg.keyword_enabled and not cfg.embedding_enabled:
            inj = kw_ok
        elif not cfg.keyword_enabled and cfg.embedding_enabled:
            inj = emb_ok
        else:
            inj = kw_ok or emb_ok
        return {
            "inject": inj,
            "keyword_hit": kw_ok,
            "embedding_hit": emb_ok,
            "similarity": sim_val,
        }

    # hybrid_and
    if cfg.keyword_enabled and not cfg.embedding_enabled:
        inj = kw_ok
    elif not cfg.keyword_enabled and cfg.embedding_enabled:
        inj = emb_ok
    else:
        inj = kw_ok and emb_ok
    return {
        "inject": inj,
        "keyword_hit": kw_ok,
        "embedding_hit": emb_ok,
        "similarity": sim_val,
    }


def evaluate_step15_policies(
    cfg: Step15GateConfig,
    *,
    memory_nonempty: bool,
    memory_blob: str,
    topic_text: str,
    historical_summary: str = "",
) -> Dict[str, Any]:
    """
    对 cfg.policies 中**每一个**策略各算一次，返回 dict：
    - 键为策略名，值为该策略的结果 dict（至少含 inject）。
    - 另含 _combine_mode、_combined_inject（按 multi_combine 合成后的总开关）。

    memory_nonempty / memory_blob / topic_text / historical_summary 语义同 should_inject_step15。
    """
    mem_side = (memory_blob or "").strip()
    if cfg.include_historical_summary and (historical_summary or "").strip():
        mem_side = f"{mem_side}\n{(historical_summary or '').strip()}".strip()

    per: Dict[str, Any] = {}
    for pol in cfg.policies:
        per[pol] = _inject_for_single_policy(
            pol,
            cfg,
            memory_nonempty=memory_nonempty,
            mem_side=mem_side,
            topic_text=topic_text,
        )

    flags = [bool(per[p].get("inject")) for p in cfg.policies]
    if cfg.multi_combine == "and":
        combined = all(flags) if flags else False
    else:
        combined = any(flags) if flags else False

    out = dict(per)
    out["_combine_mode"] = cfg.multi_combine
    out["_combined_inject"] = combined
    return out


def should_inject_step15(
    cfg: Step15GateConfig,
    *,
    memory_nonempty: bool,
    memory_blob: str,
    topic_text: str,
    historical_summary: str = "",
) -> bool:
    """
    是否向 instruction 注入「步骤 1.5」长 prompt。

    多策略时：先对每项求值，再按 cfg.multi_combine（or/and）合成。

    memory_nonempty: retrieve 后拼接文本非空（与 generate_reaction 一致）。
    memory_blob: 同上拼接串，可为空。
    historical_summary: profile 中摘要，可选。
    topic_text: 当前批次话题（推荐 chunk 或 mention 汇总出的 title/desc/tags/评论等）。
    """
    ev = evaluate_step15_policies(
        cfg,
        memory_nonempty=memory_nonempty,
        memory_blob=memory_blob,
        topic_text=topic_text,
        historical_summary=historical_summary,
    )
    return bool(ev.get("_combined_inject"))


def topic_text_from_blogs_chunk(chunk: Dict[str, Dict[str, Any]]) -> str:
    """从推荐 batch（blog_id -> blog dict）拼话题文本。"""
    parts: list[str] = []
    for blog in (chunk or {}).values():
        if not isinstance(blog, dict):
            continue
        parts.append(str(blog.get("title", "") or ""))
        parts.append(str(blog.get("desc", "") or ""))
        tags = blog.get("tags_list") or []
        if isinstance(tags, (list, tuple)):
            parts.append(" ".join(str(t) for t in tags))
    return "\n".join(parts)


def topic_text_from_mention_entries(entries: list) -> str:
    """从 handle_mention 的 mention_entries 拼话题文本（blog 场景用 mention_blog；微博场景用 mention_blog）。"""
    parts: list[str] = []
    for ent in entries or []:
        if not isinstance(ent, dict):
            continue
        n = ent.get("mention_blog")
        if isinstance(n, dict):
            parts.append(str(n.get("title", "") or ""))
            parts.append(str(n.get("desc", "") or ""))
            tags = n.get("tags_list") or []
            if isinstance(tags, (list, tuple)):
                parts.append(" ".join(str(t) for t in tags))
        b = ent.get("mention_blog")
        if isinstance(b, dict):
            parts.append(str(b.get("title", "") or ""))
            parts.append(str(b.get("content", "") or ""))
            parts.append(str(b.get("text", "") or ""))
            tags = b.get("tags_list") or b.get("tags") or []
            if isinstance(tags, (list, tuple)):
                parts.append(" ".join(str(t) for t in tags))
        parts.append(str(ent.get("mention_comment_content", "") or ""))
    return "\n".join(parts)
