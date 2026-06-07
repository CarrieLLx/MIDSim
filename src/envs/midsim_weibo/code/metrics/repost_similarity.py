# -*- coding: utf-8 -*-
"""
转发语义相似度评估：离线评估（脚本/API）与在线评估（监控指标）。
- 在线：对生成转发做 embedding，与 blog_id 对应 ACL blog embedding 算余弦相似度，返回均值。
- 离线：默认从 SimEnv 的 ``content_pool_snapshot.json`` 读取条目，仅保留 ``reposted_blog_id`` 非空的转发帖；
  用 ``reposted_blog_id`` 作为被转原帖 ``blog_id`` 查 embedding，转发正文经 ``weibo_content_for_similarity``（``//`` 首段）后算余弦。
  可选 ``--from-llm-output-json`` 使用旧逻辑（从 LLM output JSON + reposts_to_csv 解析）。

实现说明（当前版本）：
1) 评估对象是“生成转发文本”与“对应 blog 的语义向量”的贴合程度，而非转发-转发匹配。
1.1) 微博正文含 `//` 链式拼接时，相似度仅用首段（`weibo_content_for_similarity`）。
2) 在线函数 `calculate_repost_similarity` 为了兼容历史指标名保留原函数名，
   但返回值语义已变为 mean cosine similarity。
3) 离线 CLI 会输出 blog 级聚合表，便于定位哪些 blog 的转发更贴题或偏题。
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

# ---------- 项目根与 reposts_to_csv 导入（供离线使用）----------
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

# 仅 --from-llm-output-json 时复用 reposts_to_csv
extract_decisions_from_output = None
extract_prompt_and_output_from_file = None

def _lazy_import_reposts_to_csv():
    """
    延迟导入离线解析工具，避免在线监控路径不必要依赖。
    仅离线从 LLM output JSON 提取 decisions 时需要该依赖。
    """
    global extract_decisions_from_output, extract_prompt_and_output_from_file
    if extract_decisions_from_output is not None:
        return
    root = _ensure_project_root()
    own_scripts = os.path.join(root, "own_scripts")
    if os.path.isdir(own_scripts) and own_scripts not in sys.path:
        sys.path.insert(0, own_scripts)
    try:
        from reposts_to_csv import (  # noqa: F401
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


def weibo_content_for_similarity(raw: Any) -> str:
    """
    微博转发正文常为「当前用户评语 + // + 链上嵌套」；算相似度时只用用户自己写的第一段（按 // 分割取第 0 段）。
    """
    s = str(raw or "").strip()
    if not s:
        return ""
    return s.split("//", 1)[0].strip()


# ---------- 数据构建 ----------
def build_generated_reposts_from_content_pool(content_pool: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    从环境 content_pool 提取生成转发。

    仅统计顶层条目：`reposted_blog_id` 非空（与 calculate_repost_generation / SimEnv.add_repost 一致）；
    该字段即被转原帖的 ``blog_id``，用于在 embedding 表中查找原帖向量。
    正文用 `weibo_content_for_similarity`：按 ``//`` 分割后取第 0 段，再 strip。
    返回: [(原帖 blog_id, 首段正文), ...]，原帖 id 为 reposted_blog_id，供 ACL embedding 查表。
    """
    generated_rows: List[Tuple[str, str]] = []
    if not isinstance(content_pool, dict):
        return generated_rows
    for blog in content_pool.values():
        if not isinstance(blog, dict):
            continue
        rpid = blog.get("reposted_blog_id")
        if rpid is None or not str(rpid).strip():
            continue
        ref_blog_id = str(rpid).strip()
        content = weibo_content_for_similarity(blog.get("content"))
        if content:
            generated_rows.append((ref_blog_id, content))
    return generated_rows


def load_content_pool_from_snapshot_file(snapshot_path: str) -> Dict[str, Any]:
    """
    从 SimEnv 保存的 ``content_pool_snapshot.json`` 解析出 content_pool 字典。

    支持：
    - ``{ blog_id: { ..., \"reposted_blog_id\": \"...\", ... }, ... }``
    - ``{ \"content_pool\": { ... } }``
    """
    with open(snapshot_path, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"快照 JSON 顶层应为 object，实际为 {type(data).__name__}")
    inner = data.get("content_pool")
    if isinstance(inner, dict):
        return inner
    return data


def load_generated_reposts_from_content_pool_snapshot(snapshot_path: str) -> List[Tuple[str, str]]:
    """从快照提取 ``(被转原帖 blog_id, 转发首段正文)``，逻辑同 ``build_generated_reposts_from_content_pool``。"""
    pool = load_content_pool_from_snapshot_file(snapshot_path)
    return build_generated_reposts_from_content_pool(pool)


def load_blog_embeddings(blog_embeddings_path: str) -> Dict[str, List[float]]:
    """
    加载 ACL 侧 blog embedding 数据:
    [
      {"blog_id": "...", "embedding": [...]},
      ...
    ]

    返回:
    - 字典 `blog_id -> embedding`
    - 对缺失字段/空向量做容错跳过
    """
    with open(blog_embeddings_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    result: Dict[str, List[float]] = {}
    if not isinstance(items, list):
        return result
    for it in items:
        if not isinstance(it, dict):
            continue
        # 历史/数据集兼容：
        # - 有的 embedding 文件用 blog_id 字段
        # - 你当前数据集 bge-base-zh-v1.5_embeddings.json 用 note_id 字段
        blog_id = str(it.get("blog_id") or it.get("note_id") or "").strip()
        emb = it.get("embedding")
        if not blog_id or not isinstance(emb, list) or not emb:
            continue
        result[blog_id] = emb
    return result


def load_generated_reposts_from_output(output_path: str) -> List[Tuple[str, str]]:
    """
    从 output JSON 中提取生成转发（离线用）。

    数据来源:
    - 先用 reposts_to_csv 中的解析函数抽取每条 entry 的 output
    - 再从 decisions 数组中提取 (blog_id, repost_content)

    repost_content 同样按 // 取首段后再参与相似度（与线上一致）。

    返回: [(blog_id, repost_content), ...]
    """
    _lazy_import_reposts_to_csv()
    if extract_prompt_and_output_from_file is None or extract_decisions_from_output is None:
        raise RuntimeError("离线评估需要 reposts_to_csv 中的提取函数，请从项目根运行或设置 PYTHONPATH")
    with open(output_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    entries = extract_prompt_and_output_from_file(content)
    reposts: List[Tuple[str, str]] = []
    for entry in entries:
        output = entry.get("output", "")
        decisions = extract_decisions_from_output(output)
        for d in decisions:
            blog_id = str(d.get("blog_id") or "").strip()
            content_text = weibo_content_for_similarity(d.get("repost_content"))
            if blog_id and content_text:
                reposts.append((blog_id, content_text))
    return reposts


def compute_repost_blog_cosine_scores(
    base_url: str,
    model_name: str,
    generated_reposts: List[Tuple[str, str]],
    blog_embeddings: Dict[str, List[float]],
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    """
    对每条生成转发计算其 embedding，并与对应 blog_id 的 embedding 计算余弦相似度。

    处理流程:
    1) 过滤掉找不到 blog embedding 的转发
    2) 按 batch 调用 embedding 接口，降低请求开销
    3) 逐条计算 cosine，相同索引一一对应

    返回逐转发结果列表:
    [{"blog_id","repost_content","cosine_similarity"}, ...]
    """
    valid_items: List[Tuple[str, str]] = [
        (blog_id, text)
        for blog_id, text in generated_reposts
        if blog_id in blog_embeddings and text
    ]
    if not valid_items:
        return []

    texts = [it[1] for it in valid_items]
    text_embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        text_embeddings.extend(get_embeddings(base_url, model_name, texts[i : i + batch_size]))

    results: List[Dict[str, Any]] = []
    for i, (blog_id, repost_text) in enumerate(valid_items):
        sim = cosine_similarity(text_embeddings[i], blog_embeddings[blog_id])
        results.append(
            {
                "blog_id": blog_id,
                "repost_content": repost_text,
                "cosine_similarity": float(sim),
            }
        )
    return results


def aggregate_scores_by_blog(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    按 blog_id 聚合逐转发相似度，输出统计表。

    每个 blog 输出:
    - repost_count: 有效转发条数
    - mean/min/max_cosine_similarity: 余弦相似度统计
    """
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        blog_id = str(row.get("blog_id") or "").strip()
        score = row.get("cosine_similarity")
        if blog_id and isinstance(score, (int, float)):
            grouped[blog_id].append(float(score))

    table: List[Dict[str, Any]] = []
    for blog_id, vals in grouped.items():
        if not vals:
            continue
        table.append(
            {
                "blog_id": blog_id,
                "repost_count": len(vals),
                "mean_cosine_similarity": sum(vals) / len(vals),
                "min_cosine_similarity": min(vals),
                "max_cosine_similarity": max(vals),
            }
        )
    table.sort(key=lambda x: x["blog_id"])
    return table


# ---------- 离线评估 ----------
def run_offline_evaluation(
    reposts_source_path: str,
    blog_embeddings_path: str,
    config_path: Optional[str] = None,
    batch_size: int = 32,
    *,
    from_llm_output_json: bool = False,
) -> Tuple[float, int, List[Dict[str, Any]]]:
    """
    离线评估入口。

    步骤:
    1) 读取 embedding 配置与 ACL blog 向量
    2) 从 ``content_pool_snapshot.json``（默认）或 LLM output JSON 提取生成转发
    3) 计算逐转发 cosine
    4) 聚合为 blog 级统计

    参数:
    - ``reposts_source_path``: 默认视为 ``content_pool_snapshot.json``；若 ``from_llm_output_json=True`` 则为旧版 output JSON。

    返回: (全量均值, 有效转发数, 按 blog 聚合表)。
    """
    if not config_path:
        root = _ensure_project_root()
        config_path = os.path.join(root, "config", "model_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    if not os.path.isfile(blog_embeddings_path):
        raise FileNotFoundError(f"blog embeddings 文件不存在: {blog_embeddings_path}")
    if not os.path.isfile(reposts_source_path):
        raise FileNotFoundError(f"转发来源文件不存在: {reposts_source_path}")
    base_url, model_name = load_embedding_config(config_path)
    blog_embeddings = load_blog_embeddings(blog_embeddings_path)
    if from_llm_output_json:
        generated_reposts = load_generated_reposts_from_output(reposts_source_path)
    else:
        generated_reposts = load_generated_reposts_from_content_pool_snapshot(reposts_source_path)
    rows = compute_repost_blog_cosine_scores(
        base_url, model_name, generated_reposts, blog_embeddings, batch_size=batch_size
    )
    if not rows:
        return 0.0, 0, []
    mean_cos = sum(r["cosine_similarity"] for r in rows) / len(rows)
    return float(mean_cos), len(rows), aggregate_scores_by_blog(rows)


# ---------- 在线评估（监控指标）----------
def calculate_repost_similarity(data: Dict[str, Any]) -> Any:
    """
    在线监控指标入口（函数名为历史兼容保留）。

    语义:
    - 返回值是 mean cosine similarity

    在线指标：根据当前 content_pool 生成转发，计算其与对应 blog embedding 的余弦相似度均值。
    需要 data 中包含 content_pool。
    可选 data.embedding_base_url / data.embedding_model_name 或 data.embedding_config_path 指定 embedding；
    可选 data.blog_embeddings_path 指定 ACL embedding 文件路径；
    若未指定，依次尝试 data.reference_embedding_path、data.reference_csv_path（微博 env_data 常用）；
    若仍未提供，再使用默认路径:
    src/envs/multi_channel_information_diffusion/profile/data/acl/embeddings/bge-base-zh-v1.5_embeddings.json
    否则使用项目根下 config/model_config.json。
    若无有效转发或无可匹配 blog embedding 则返回 None（监控可忽略该点）。
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
        log_metric_error("repost_similarity_mse", ValueError("无效的数据输入"), {"data": data})
        return None

    content_pool = safe_get(data, "content_pool", {})
    generated_reposts = build_generated_reposts_from_content_pool(content_pool)
    # 没有转发时返回 None，避免用 0.0 误导监控含义。
    if not generated_reposts:
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
            log_metric_error("repost_similarity_mse", FileNotFoundError("未找到 embedding 配置"), {"data_keys": list(data.keys())})
            return None

    # 路径优先级：
    # 1) blog_embeddings_path（新字段）
    # 2) reference_embedding_path（兼容旧 env_data）
    # 3) reference_csv_path（微博 env_data.json 中常用，实际可为 embeddings 的 json）
    # 4) 代码内默认路径
    blog_embeddings_path = (
        data.get("blog_embeddings_path")
        or data.get("reference_embedding_path")
        or data.get("reference_csv_path")
    )
    if not blog_embeddings_path:
        root = _ensure_project_root()
        blog_embeddings_path = os.path.join(
            root,
            "src",
            "envs",
            "multi_channel_information_diffusion",
            "profile",
            "data",
            "acl",
            "embeddings",
            "bge-base-zh-v1.5_embeddings.json",
        )
    if not os.path.isfile(blog_embeddings_path):
        log_metric_error("repost_similarity_mse", FileNotFoundError("未找到 blog embeddings 文件"), {"path": blog_embeddings_path})
        return None
    blog_embeddings = load_blog_embeddings(blog_embeddings_path)
    # blog embedding 为空时无法评估。
    if not blog_embeddings:
        return None

    try:
        rows = compute_repost_blog_cosine_scores(
            base_url, model_name, generated_reposts, blog_embeddings, batch_size=32
        )
        if not rows:
            return None
        mean_cos = sum(r["cosine_similarity"] for r in rows) / len(rows)
        return float(mean_cos)
    except Exception as e:
        log_metric_error("repost_similarity_mse", e, {"data_keys": list(data.keys())})
        return None


# scene_info.json 中 Repost Similarity 的 function_name 与此对齐
calculate_repost_similarity = calculate_repost_similarity


# ---------- CLI（离线）----------
def main() -> None:
    """
    离线 CLI：
    - 默认输入 content_pool_snapshot.json 与 blog embeddings JSON
    - 输出全体转发平均相似度 + blog 级 CSV 表
    """
    parser = argparse.ArgumentParser(description="转发相似度离线评估：转发向量 vs blog 向量余弦相似度")
    parser.add_argument(
        "content_pool_snapshot",
        help="SimEnv 导出的 content_pool_snapshot.json（仅 reposted_blog_id 非空条目参与）",
    )
    parser.add_argument("blog_embeddings_json", help="ACL blog embeddings JSON 路径（含 blog_id / note_id, embedding）")
    parser.add_argument(
        "--from-llm-output-json",
        action="store_true",
        help="改为从 LLM 对话 output JSON 解析转发（旧行为，依赖 own_scripts/reposts_to_csv）",
    )
    parser.add_argument("--config", default="", help="model_config.json 路径，默认项目根 config/model_config.json")
    parser.add_argument("--batch-size", type=int, default=32, help="embedding 批大小")
    parser.add_argument(
        "--output-table",
        default="",
        help="按 blog 聚合结果 CSV（默认: 与 content_pool_snapshot 同目录下 blog_repost_similarity.csv）",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.content_pool_snapshot):
        print(f"错误: 文件不存在 {args.content_pool_snapshot}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.blog_embeddings_json):
        print(f"错误: blog embeddings JSON 不存在 {args.blog_embeddings_json}", file=sys.stderr)
        sys.exit(1)

    config_path = args.config.strip() or None
    try:
        mean_cos, n_rows, blog_table = run_offline_evaluation(
            args.content_pool_snapshot,
            args.blog_embeddings_json,
            config_path=config_path,
            batch_size=args.batch_size,
            from_llm_output_json=args.from_llm_output_json,
        )
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

    if args.output_table.strip():
        output_table = args.output_table.strip()
    else:
        output_table = os.path.join(
            os.path.dirname(os.path.abspath(args.content_pool_snapshot)),
            "blog_repost_similarity.csv",
        )

    with open(output_table, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "blog_id",
                "repost_count",
                "mean_cosine_similarity",
                "min_cosine_similarity",
                "max_cosine_similarity",
            ],
        )
        writer.writeheader()
        for row in blog_table:
            writer.writerow(row)

    print(f"有效转发数: {n_rows}")
    print(f"Mean Cosine Similarity: {mean_cos:.6f}")
    print(f"按 blog 聚合表已输出: {output_table}")


if __name__ == "__main__":
    main()
