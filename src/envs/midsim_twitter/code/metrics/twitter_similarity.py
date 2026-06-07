# -*- coding: utf-8 -*-
"""
推文生成文本与参考推文向量的语义相似度：离线脚本 / 在线监控。

- 在线：对 content_pool 中符合条件的生成文本做 embedding，与参考 tweet embedding 求余弦相似度均值。
- 离线：从 output JSON 抽取生成文本，同样对齐根推 id 后算相似度，并输出按 tweet 聚合表。

说明：
1) 评估的是「生成文本」与「根推语义向量」的贴合程度，不是文本两两匹配。
2) 监控入口：`calculate_tweet_text_similarity`（由 scene_info / metrics 注册）。
3) Twitter：仅引用/回复链上的生成内容按根推 id 对齐；转推与纯原创跳过。
4) 离线可传 content_pool 快照，将 output 里的 note_id 归并为根推 id。
5) `count_content_pool_tweets_for_monitor` 供 metrics 统计「传播类」推文：原创推一律不计，转推/引用/回复计入。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    requests = None

# ---------- 项目根与 output 解析（own_scripts 内解析模块，供离线使用）----------
def _find_project_root() -> Optional[str]:
    """从当前文件向上查找包含 config/model_config.json 的目录作为项目根。"""
    path = os.path.abspath(__file__)
    for _ in range(10):
        path = os.path.dirname(path)
        if not path or path == os.path.dirname(path):
            break
        config_file = os.path.join(path, "config", "model_config.json")
        if os.path.isfile(config_file):
            return path
    return None

_project_root: Optional[str] = None

def _ensure_project_root() -> str:
    global _project_root
    if _project_root is None:
        _project_root = _find_project_root()
    if _project_root and _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    return _project_root or os.getcwd()

extract_decisions_from_output = None
extract_prompt_and_output_from_file = None


def _lazy_import_output_decisions_helpers():
    """
    延迟导入离线解析工具，避免在线监控路径不必要依赖。
    实现位于仓库 own_scripts/comments_to_csv.py（模块名历史遗留）。
    """
    global extract_decisions_from_output, extract_prompt_and_output_from_file
    if extract_decisions_from_output is not None:
        return
    root = _ensure_project_root()
    own_scripts = os.path.join(root, "own_scripts")
    if os.path.isdir(own_scripts) and own_scripts not in sys.path:
        sys.path.insert(0, own_scripts)
    try:
        from comments_to_csv import (  # noqa: F401
            extract_decisions_from_output as _extract_decisions,
            extract_prompt_and_output_from_file as _extract_prompt_output,
        )
        extract_decisions_from_output = _extract_decisions
        extract_prompt_and_output_from_file = _extract_prompt_output
    except ImportError:
        pass

# ---------- 配置与向量 ----------
def load_embedding_config(config_path: str) -> Tuple[str, str]:
    """
    从 model_config.json 读取 embedding 服务配置。

    约定读取 embedding 数组的第一个配置项：
    - client_args.base_url: embedding 服务地址（会自动去掉尾部 /）
    - model_name: 模型名
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    embeddings = config.get("embedding", [])
    if not embeddings:
        raise ValueError("model_config.json 中未找到 embedding 配置")
    emb = embeddings[0]
    base_url = emb.get("client_args", {}).get("base_url", "").rstrip("/")
    model_name = emb.get("model_name", "bge-base-en-v1.5")
    return base_url, model_name


def get_embeddings(base_url: str, model_name: str, texts: List[str]) -> List[List[float]]:
    """
    调用 embedding HTTP API，批量获取文本向量。

    参数:
    - base_url: 服务根地址，如 http://host:port/v1
    - model_name: embedding 模型名
    - texts: 输入文本列表

    返回:
    - 与输入顺序一致的 embedding 列表
    """
    if not texts:
        return []
    if requests is None:
        raise RuntimeError("需要安装 requests: pip install requests")
    url = f"{base_url}/embeddings"
    payload = {"model": model_name, "input": texts}
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if "data" not in data or not data["data"]:
        raise ValueError("API 返回格式异常")
    items = sorted(data["data"], key=lambda x: x.get("index", 0))
    return [item["embedding"] for item in items]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """计算两个向量的余弦相似度（范围通常在 [-1, 1]）。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------- 数据构建 ----------
def _tweet_ref_str(val: Any) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip()
    return s or None


def _resolve_quote_reply_root_tweet_id(
    start_id: str,
    content_pool: Dict[str, Any],
    *,
    max_hops: int = 64,
) -> str:
    """
    沿 quoted_tweet_id / replied_tweet_id 向上走，直到当前节点两者皆无（根推文）。
    优先沿回复链（replied），否则沿引用链（quoted）；防环与缺键则停在当前 id。
    """
    visited: set[str] = set()
    cur = str(start_id)
    for _ in range(max_hops):
        if cur in visited:
            return cur
        visited.add(cur)
        tw = content_pool.get(cur)
        if not isinstance(tw, dict):
            return cur
        qid = _tweet_ref_str(tw.get("quoted_tweet_id"))
        rid = _tweet_ref_str(tw.get("replied_tweet_id")) or _tweet_ref_str(
            tw.get("replyed_tweet_id")
        )
        if not qid and not rid:
            return cur
        nxt = rid or qid
        if nxt not in content_pool:
            return cur
        cur = nxt
    return cur


def _twitter_similarity_root_tweet_id(
    tweet_id: str,
    tweet: Optional[Dict[str, Any]],
    content_pool: Dict[str, Any],
) -> Optional[str]:
    """
    判断是否应参与 Twitter 文本相似度对齐，并返回用于匹配 embedding 的根推 id。

    - 池内为转推：不参与（返回 None）。
    - 池内为无引用、无回复的原创推：不参与（返回 None）。
    - 池内为引用/回复链上的推文：返回解析后的根推 id。
    - tweet 为 None（池内无该键）：仍做向上解析；无法继续解析时返回原 id。
    """
    tid = str(tweet_id).strip()
    if not tid:
        return None
    if tweet is None:
        return _resolve_quote_reply_root_tweet_id(tid, content_pool)
    if not isinstance(tweet, dict):
        return None
    if _tweet_ref_str(tweet.get("retweeted_tweet_id")):
        return None
    qid = _tweet_ref_str(tweet.get("quoted_tweet_id"))
    rid = _tweet_ref_str(tweet.get("replied_tweet_id")) or _tweet_ref_str(
        tweet.get("replyed_tweet_id")
    )
    if not qid and not rid:
        return None
    return _resolve_quote_reply_root_tweet_id(tid, content_pool)


def is_twitter_original_tweet(tweet: Dict[str, Any]) -> bool:
    """
    原创推：无转推、无引用、无回复父帖（与 build_generated_contents 中「跳过原创」的判定一致）。
    """
    if not isinstance(tweet, dict):
        return False
    if _tweet_ref_str(tweet.get("retweeted_tweet_id")):
        return False
    qid = _tweet_ref_str(tweet.get("quoted_tweet_id"))
    rid = _tweet_ref_str(tweet.get("replied_tweet_id")) or _tweet_ref_str(
        tweet.get("replyed_tweet_id")
    )
    return not qid and not rid


def count_direct_content_pool_tweets_for_monitor(
    content_pool: Dict[str, Any],
    _current_timestamp: Optional[float] = None,
) -> int:
    """
    统计 content_pool 中「非二级传播」推文条数（仅直接传播类：转推 / 引用 / 回复）。

    规则:
    - 遍历值为 dict 的条目。
    - 「原创推」（无 retweeted_tweet_id，且无 quoted_tweet_id，且无 replied_tweet_id /
      replyed_tweet_id）：一律不计入。
    - 「直接传播类」定义为：当前推文为传播类，且其父推文（retweeted_tweet / quoted_tweet /
      replied_tweet）是原创推（即父推文的 retweeted_tweet_id、quoted_tweet_id、
      replied_tweet_id / replyed_tweet_id 均为空）。
    - 第二参数保留以兼容旧调用（如传入 current_timestamp），当前不参与计算。
    """
    if not isinstance(content_pool, dict):
        return 0

    n = 0
    for tweet in content_pool.values():
        if not isinstance(tweet, dict):
            continue
        if is_twitter_original_tweet(tweet):
            continue

        parent_tweet = None
        if _tweet_ref_str(tweet.get("retweeted_tweet_id")):
            retweeted_id = _tweet_ref_str(tweet.get("retweeted_tweet_id"))
            parent_tweet = tweet.get("retweeted_tweet")
            if not isinstance(parent_tweet, dict) and retweeted_id:
                parent_tweet = content_pool.get(retweeted_id)
        elif _tweet_ref_str(tweet.get("quoted_tweet_id")):
            quoted_id = _tweet_ref_str(tweet.get("quoted_tweet_id"))
            parent_tweet = tweet.get("quoted_tweet")
            if not isinstance(parent_tweet, dict) and quoted_id:
                parent_tweet = content_pool.get(quoted_id)
        else:
            replied_id = _tweet_ref_str(tweet.get("replied_tweet_id")) or _tweet_ref_str(
                tweet.get("replyed_tweet_id")
            )
            if replied_id:
                parent_tweet = tweet.get("replied_tweet")
                if not isinstance(parent_tweet, dict):
                    parent_tweet = content_pool.get(replied_id)

        if isinstance(parent_tweet, dict) and is_twitter_original_tweet(parent_tweet):
            n += 1

    return n


def count_direct_propagation_by_type(
    content_pool: Dict[str, Any],
    propagation_type: str,
) -> int:
    """
    按传播形态分别统计「非二级传播」条数（与 `count_direct_content_pool_tweets_for_monitor`
    的父帖判定一致：父帖须为原创推，即 retweeted_tweet_id / quoted_tweet_id /
    replied_tweet_id / replyed_tweet_id 在父对象上均为空）。

    `propagation_type`: ``"retweet"`` | ``"quote"`` | ``"reply"``。
    分类优先级与聚合函数相同：先转推，再引用，再回复（互斥）。
    """
    if not isinstance(content_pool, dict):
        return 0
    if propagation_type not in {"retweet", "quote", "reply"}:
        return 0

    n = 0
    for tweet in content_pool.values():
        if not isinstance(tweet, dict):
            continue
        if is_twitter_original_tweet(tweet):
            continue

        parent_tweet = None

        if propagation_type == "retweet":
            if not _tweet_ref_str(tweet.get("retweeted_tweet_id")):
                continue
            retweeted_id = _tweet_ref_str(tweet.get("retweeted_tweet_id"))
            parent_tweet = tweet.get("retweeted_tweet")
            if not isinstance(parent_tweet, dict) and retweeted_id:
                parent_tweet = content_pool.get(retweeted_id)
        elif propagation_type == "quote":
            if _tweet_ref_str(tweet.get("retweeted_tweet_id")):
                continue
            if not _tweet_ref_str(tweet.get("quoted_tweet_id")):
                continue
            quoted_id = _tweet_ref_str(tweet.get("quoted_tweet_id"))
            parent_tweet = tweet.get("quoted_tweet")
            if not isinstance(parent_tweet, dict) and quoted_id:
                parent_tweet = content_pool.get(quoted_id)
        else:
            if _tweet_ref_str(tweet.get("retweeted_tweet_id")):
                continue
            if _tweet_ref_str(tweet.get("quoted_tweet_id")):
                continue
            replied_id = _tweet_ref_str(tweet.get("replied_tweet_id")) or _tweet_ref_str(
                tweet.get("replyed_tweet_id")
            )
            if not replied_id:
                continue
            parent_tweet = tweet.get("replied_tweet")
            if not isinstance(parent_tweet, dict):
                parent_tweet = content_pool.get(replied_id)

        if isinstance(parent_tweet, dict) and is_twitter_original_tweet(parent_tweet):
            n += 1

    return n


def count_content_pool_tweets_for_monitor(
    content_pool: Dict[str, Any],
    _current_timestamp: Optional[float] = None,
) -> int:
    """
    统计 content_pool 中计入监控的推文条数（仅传播类：转推 / 引用 / 回复）。

    规则:
    - 遍历值为 dict 的条目。
    - 「原创推」（无 retweeted_tweet_id，且无 quoted_tweet_id，且无 replied_tweet_id /
      replyed_tweet_id）：一律不计入。
    - 否则（任一类引用或转推字段有值）：计 1，不做时间过滤。
    - 第二参数保留以兼容旧调用（如传入 current_timestamp），当前不参与计算。
    """
    if not isinstance(content_pool, dict):
        return 0
    n = 0
    for tweet in content_pool.values():
        if not isinstance(tweet, dict):
            continue
        if is_twitter_original_tweet(tweet):
            continue
        n += 1
    return n

def build_generated_contents_from_content_pool(content_pool: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    从环境 content_pool 提取生成内容。

    规则:
    - 无 retweeted_tweet_id、无 quoted_tweet_id、无 replied_tweet_id 的原创推：跳过。
    - 有 retweeted_tweet_id 的转推：跳过。
    - 仅有引用/回复链的推文：向上解析到根推（无 quote、无 reply 父 id），用根推 id 作为聚合键。

    返回: [(root_tweet_id, content), ...]
    """
    generated_rows: List[Tuple[str, str]] = []
    if not isinstance(content_pool, dict):
        return generated_rows
    for tweet_id, tweet in content_pool.items():
        if not isinstance(tweet, dict):
            continue
        # 这里必须使用独立变量名，避免覆盖上面的列表变量。
        content = tweet.get("content") or {}
        root_id = _twitter_similarity_root_tweet_id(str(tweet_id), tweet, content_pool)
        if root_id and content:
            generated_rows.append((root_id, content))
    return generated_rows


def load_tweet_embeddings(embeddings_json_path: str) -> Dict[str, List[float]]:
    """
    加载参考推文 embedding 列表 JSON:
    [
      {"tweet_id": "...", "embedding": [...]},
      或 {"note_id": "...", "embedding": [...]}（与 tweet_id 等价）,
      ...
    ]

    返回:
    - 字典 `tweet_id -> embedding`
    - 对缺失字段/空向量做容错跳过
    """
    with open(embeddings_json_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    result: Dict[str, List[float]] = {}
    if not isinstance(items, list):
        return result
    for it in items:
        if not isinstance(it, dict):
            continue
        tweet_id = str(
            it.get("tweet_id") or it.get("note_id") or ""
        ).strip()
        emb = it.get("embedding")
        if not tweet_id or not isinstance(emb, list) or not emb:
            continue
        result[tweet_id] = emb
    return result


def load_generated_texts_from_output(
    output_path: str,
    content_pool: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, str]]:
    """
    从 output JSON 中提取生成文本（离线用）。

    数据来源:
    - 使用 own_scripts 内解析器抽取每条 entry 的 output
    - 再从 decisions 数组中取文本；字段优先 tweet_text / generated_text，其次为管线遗留键名。

    若传入 content_pool，则对 note_id 做与 build_generated_contents_from_content_pool 相同的根推 id 归并。

    返回: [(tweet_id, generated_text), ...]
    """
    _lazy_import_output_decisions_helpers()
    if extract_prompt_and_output_from_file is None or extract_decisions_from_output is None:
        raise RuntimeError("离线评估需要 output 解析模块，请从项目根运行或设置 PYTHONPATH")
    with open(output_path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    entries = extract_prompt_and_output_from_file(raw)
    out_rows: List[Tuple[str, str]] = []
    pool = content_pool if isinstance(content_pool, dict) else None
    for entry in entries:
        output = entry.get("output", "")
        decisions = extract_decisions_from_output(output)
        for d in decisions:
            if not isinstance(d, dict):
                continue
            tid = str(d.get("note_id") or d.get("tweet_id") or "").strip()
            text = (
                d.get("tweet_text")
                or d.get("generated_text")
                or d.get("comment_content")
                or ""
            )
            text = str(text).strip()
            if not tid or not text:
                continue
            if pool is not None:
                tw = pool.get(tid)
                mapped = _twitter_similarity_root_tweet_id(tid, tw, pool)
                if mapped is None:
                    continue
                tid = mapped
            out_rows.append((tid, text))
    return out_rows


def compute_tweet_text_cosine_scores(
    base_url: str,
    model_name: str,
    generated_pairs: List[Tuple[str, str]],
    reference_embeddings: Dict[str, List[float]],
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    """
    对每条生成文本求 embedding，并与对应根推 id 的参考向量算余弦相似度。

    generated_pairs: [(tweet_id, generated_text), ...]

    返回:
    [{"tweet_id", "generated_text", "cosine_similarity"}, ...]
    """
    valid_items: List[Tuple[str, str]] = [
        (tid, text)
        for tid, text in generated_pairs
        if tid in reference_embeddings and text
    ]
    if not valid_items:
        return []

    texts = [it[1] for it in valid_items]
    text_embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        text_embeddings.extend(get_embeddings(base_url, model_name, texts[i : i + batch_size]))

    results: List[Dict[str, Any]] = []
    for i, (tid, gen_text) in enumerate(valid_items):
        sim = cosine_similarity(text_embeddings[i], reference_embeddings[tid])
        results.append(
            {
                "tweet_id": tid,
                "generated_text": gen_text,
                "cosine_similarity": float(sim),
            }
        )
    return results


def aggregate_scores_by_tweet(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    按 tweet_id（通常为根推 id）聚合逐条相似度。

    输出字段:
    - sample_count: 有效样本条数
    - mean/min/max_cosine_similarity
    """
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        tid = str(row.get("tweet_id") or "").strip()
        score = row.get("cosine_similarity")
        if tid and isinstance(score, (int, float)):
            grouped[tid].append(float(score))

    table: List[Dict[str, Any]] = []
    for tid, vals in grouped.items():
        if not vals:
            continue
        table.append(
            {
                "tweet_id": tid,
                "sample_count": len(vals),
                "mean_cosine_similarity": sum(vals) / len(vals),
                "min_cosine_similarity": min(vals),
                "max_cosine_similarity": max(vals),
            }
        )
    table.sort(key=lambda x: x["tweet_id"])
    return table


# ---------- 离线评估 ----------
def run_offline_evaluation(
    output_json: str,
    tweet_embeddings_path: str,
    config_path: Optional[str] = None,
    batch_size: int = 32,
    content_pool: Optional[Dict[str, Any]] = None,
) -> Tuple[float, int, List[Dict[str, Any]]]:
    """
    离线评估入口。

    步骤:
    1) 读取 embedding 配置与参考 tweet 向量
    2) 从 output_json 提取生成文本（可选 content_pool 做根推 id 归并）
    3) 计算逐条 cosine
    4) 聚合为 tweet 级统计

    返回: (全量均值, 有效样本数, 按 tweet 聚合表)。
    """
    if not config_path:
        root = _ensure_project_root()
        config_path = os.path.join(root, "config", "model_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    if not os.path.isfile(tweet_embeddings_path):
        raise FileNotFoundError(f"tweet embeddings 文件不存在: {tweet_embeddings_path}")
    base_url, model_name = load_embedding_config(config_path)
    reference_embeddings = load_tweet_embeddings(tweet_embeddings_path)
    generated_pairs = load_generated_texts_from_output(
        output_json, content_pool=content_pool
    )
    rows = compute_tweet_text_cosine_scores(
        base_url,
        model_name,
        generated_pairs,
        reference_embeddings,
        batch_size=batch_size,
    )
    if not rows:
        return 0.0, 0, []
    mean_cos = sum(r["cosine_similarity"] for r in rows) / len(rows)
    return float(mean_cos), len(rows), aggregate_scores_by_tweet(rows)


# ---------- 在线评估（监控指标）----------
def calculate_tweet_text_similarity(data: Dict[str, Any]) -> Any:
    """
    在线监控指标：生成文本 vs 参考 tweet 向量的平均余弦相似度。

    需要 data.content_pool；embedding 服务由 model_config 或 data 内字段指定；
    参考向量文件：tweet_embeddings_path / reference_embedding_path / reference_csv_path（Twitter env 常用），否则默认 datasets/twitter-openreview/embeddings/...。
    无有效样本或无法加载参考向量时返回 None。
    """
    try:
        from onesim.monitor.utils import safe_get, log_metric_error
    except ImportError:
        try:
            from onesim_cn.monitor.utils import safe_get, log_metric_error
        except ImportError:
            def safe_get(d, k, default=None):
                return (d or {}).get(k, default) if isinstance(d, dict) else default
            def log_metric_error(name, e, ctx):
                pass

    if not data or not isinstance(data, dict):
        log_metric_error("tweet_text_similarity", ValueError("无效的数据输入"), {"data": data})
        return None

    content_pool = safe_get(data, "content_pool", {})
    generated_contents = build_generated_contents_from_content_pool(content_pool)
    if not generated_contents:
        return None

    base_url = data.get("embedding_base_url")
    model_name = data.get("embedding_model_name")
    if not base_url or not model_name:
        config_path = data.get("embedding_config_path")
        if not config_path:
            root = _ensure_project_root()
            config_path = os.path.join(root, "config", "model_config.json")
        if os.path.isfile(config_path):
            base_url, model_name = load_embedding_config(config_path)
        else:
            log_metric_error(
                "tweet_text_similarity",
                FileNotFoundError("未找到 embedding 配置"),
                {"data_keys": list(data.keys())},
            )
            return None

    # 路径优先级：tweet_embeddings_path → reference_embedding_path → reference_csv_path（Twitter env 常用此字段存 json）
    tweet_embeddings_path = (
        data.get("tweet_embeddings_path")
        or data.get("reference_embedding_path")
        or data.get("reference_csv_path")
    )
    if not tweet_embeddings_path:
        root = _ensure_project_root()
        tweet_embeddings_path = os.path.join(
            root,
            "datasets",
            "twitter-openreview",
            "embeddings",
            "bge-base-zh-v1.5_embeddings.json",
        )
    if not os.path.isfile(tweet_embeddings_path):
        log_metric_error(
            "tweet_text_similarity",
            FileNotFoundError("未找到 tweet embeddings 文件"),
            {"path": tweet_embeddings_path},
        )
        return None
    reference_embeddings = load_tweet_embeddings(tweet_embeddings_path)
    if not reference_embeddings:
        return None

    try:
        rows = compute_tweet_text_cosine_scores(
            base_url,
            model_name,
            generated_contents,
            reference_embeddings,
            batch_size=32,
        )
        if not rows:
            return None
        mean_cos = sum(r["cosine_similarity"] for r in rows) / len(rows)
        return float(mean_cos)
    except Exception as e:
        log_metric_error("tweet_text_similarity", e, {"data_keys": list(data.keys())})
        return None


def load_content_pool_from_json(path: str) -> Dict[str, Any]:
    """
    从 JSON 加载 content_pool，供离线评估与 output 中的 note_id 对齐。

    支持:
    - 整文件即为 tweet_id -> tweet 对象 的字典
    - 或顶层含 "content_pool" 键的字典
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    inner = data.get("content_pool")
    if isinstance(inner, dict):
        return inner
    return data


# ---------- CLI（离线）----------
def main() -> None:
    """
    离线 CLI：
    - 输入 output JSON 与 tweet embeddings JSON
    - 输出全体样本平均相似度 + 按 tweet 聚合的 CSV
    """
    parser = argparse.ArgumentParser(
        description="推文生成文本相似度离线评估：生成文本向量 vs 参考 tweet 向量余弦相似度"
    )
    parser.add_argument("output_json", help="仿真 output JSON 路径")
    parser.add_argument(
        "tweet_embeddings_json",
        help="参考 tweet embeddings JSON（含 tweet_id 或 note_id、embedding）",
    )
    parser.add_argument("--config", default="", help="model_config.json 路径，默认项目根 config/model_config.json")
    parser.add_argument("--batch-size", type=int, default=32, help="embedding 批大小")
    parser.add_argument(
        "--output-table",
        default="",
        help="离线按 tweet 聚合结果 CSV 路径（默认: 与 output_json 同目录下 tweet_text_similarity.csv）",
    )
    parser.add_argument(
        "--content-pool-json",
        default="",
        help="可选：仿真结束时的 content_pool JSON（或含 content_pool 键的快照），用于将 output 中 note_id 归并为根推 id",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.output_json):
        print(f"错误: output JSON 不存在 {args.output_json}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.tweet_embeddings_json):
        print(f"错误: tweet embeddings JSON 不存在 {args.tweet_embeddings_json}", file=sys.stderr)
        sys.exit(1)

    config_path = args.config.strip() or None
    pool_arg = args.content_pool_json.strip()
    content_pool: Optional[Dict[str, Any]] = None
    if pool_arg:
        if not os.path.isfile(pool_arg):
            print(f"错误: content_pool JSON 不存在 {pool_arg}", file=sys.stderr)
            sys.exit(1)
        content_pool = load_content_pool_from_json(pool_arg)
    try:
        mean_cos, n_rows, tweet_table = run_offline_evaluation(
            args.output_json,
            args.tweet_embeddings_json,
            config_path=config_path,
            batch_size=args.batch_size,
            content_pool=content_pool,
        )
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

    if args.output_table.strip():
        output_table = args.output_table.strip()
    else:
        output_table = os.path.join(
            os.path.dirname(os.path.abspath(args.output_json)),
            "tweet_text_similarity.csv",
        )

    with open(output_table, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tweet_id",
                "sample_count",
                "mean_cosine_similarity",
                "min_cosine_similarity",
                "max_cosine_similarity",
            ],
        )
        writer.writeheader()
        for row in tweet_table:
            writer.writerow(row)

    print(f"有效样本数: {n_rows}")
    print(f"Mean Cosine Similarity: {mean_cos:.6f}")
    print(f"按 tweet 聚合表已输出: {output_table}")


if __name__ == "__main__":
    main()
