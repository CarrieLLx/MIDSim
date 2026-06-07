# -*- coding: utf-8 -*-
"""
评论语义相似度评估：离线评估（脚本/API）与在线评估（监控指标）。
- 在线：对生成评论做 embedding，与 note_id 对应 ACL note embedding 算余弦相似度，返回均值。
- 离线：对每条生成评论算与对应 note embedding 的余弦相似度，并按 note_id 输出统计表。

实现说明（当前版本）：
1) 评估对象是“生成评论文本”与“对应 note 的语义向量”的贴合程度，而非评论-评论匹配。
2) 在线函数 `calculate_comment_similarity` 为了兼容历史指标名保留原函数名，
   但返回值语义已变为 mean cosine similarity。
3) 离线 CLI 会输出 note 级聚合表，便于定位哪些 note 的评论更贴题或偏题。
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

# ---------- 项目根与 comments_to_csv 导入（供离线使用）----------
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

# 离线解析 output 时复用 comments_to_csv
extract_decisions_from_output = None
extract_prompt_and_output_from_file = None

def _lazy_import_comments_to_csv():
    """
    延迟导入离线解析工具，避免在线监控路径不必要依赖。
    仅离线从 output JSON 提取 decisions 时需要该依赖。
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
def build_generated_comments_from_content_pool(content_pool: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    从环境 content_pool 提取生成评论。
    content_pool[note_id]["comments"] 为 comment_id -> { "user_id", "content" }。
    返回: [(note_id, comment_content), ...]
    """
    generated_rows: List[Tuple[str, str]] = []
    if not isinstance(content_pool, dict):
        return generated_rows
    for note_id, note in content_pool.items():
        if not isinstance(note, dict):
            continue
        # 这里必须使用独立变量名，避免覆盖上面的列表变量。
        note_comments = note.get("comments") or {}
        if not isinstance(note_comments, dict):
            continue
        for comment in note_comments.values():
            if not isinstance(comment, dict):
                continue
            content = (comment.get("content") or "").strip()
            if content:
                generated_rows.append((str(note_id), content))
    return generated_rows


def load_note_embeddings(note_embeddings_path: str) -> Dict[str, List[float]]:
    """
    加载 ACL 侧 note embedding 数据:
    [
      {"note_id": "...", "embedding": [...]},
      ...
    ]

    返回:
    - 字典 `note_id -> embedding`
    - 对缺失字段/空向量做容错跳过
    """
    with open(note_embeddings_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    result: Dict[str, List[float]] = {}
    if not isinstance(items, list):
        return result
    for it in items:
        if not isinstance(it, dict):
            continue
        note_id = str(it.get("note_id") or "").strip()
        emb = it.get("embedding")
        if not note_id or not isinstance(emb, list) or not emb:
            continue
        result[note_id] = emb
    return result


def load_generated_comments_from_output(output_path: str) -> List[Tuple[str, str]]:
    """
    从 output JSON 中提取生成评论（离线用）。

    数据来源:
    - 先用 comments_to_csv 中的解析函数抽取每条 entry 的 output
    - 再从 decisions 数组中提取 (note_id, comment_content)

    返回: [(note_id, comment_content), ...]
    """
    _lazy_import_comments_to_csv()
    if extract_prompt_and_output_from_file is None or extract_decisions_from_output is None:
        raise RuntimeError("离线评估需要 comments_to_csv 中的提取函数，请从项目根运行或设置 PYTHONPATH")
    with open(output_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    entries = extract_prompt_and_output_from_file(content)
    comments: List[Tuple[str, str]] = []
    for entry in entries:
        output = entry.get("output", "")
        decisions = extract_decisions_from_output(output)
        for d in decisions:
            note_id = str(d.get("note_id") or "").strip()
            content_text = (d.get("comment_content") or "").strip()
            if note_id and content_text:
                comments.append((note_id, content_text))
    return comments


def compute_comment_note_cosine_scores(
    base_url: str,
    model_name: str,
    generated_comments: List[Tuple[str, str]],
    note_embeddings: Dict[str, List[float]],
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    """
    对每条生成评论计算其 embedding，并与对应 note_id 的 embedding 计算余弦相似度。

    处理流程:
    1) 过滤掉找不到 note embedding 的评论
    2) 按 batch 调用 embedding 接口，降低请求开销
    3) 逐条计算 cosine，相同索引一一对应

    返回逐评论结果列表:
    [{"note_id","comment_content","cosine_similarity"}, ...]
    """
    valid_items: List[Tuple[str, str]] = [
        (note_id, text)
        for note_id, text in generated_comments
        if note_id in note_embeddings and text
    ]
    if not valid_items:
        return []

    texts = [it[1] for it in valid_items]
    text_embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        text_embeddings.extend(get_embeddings(base_url, model_name, texts[i : i + batch_size]))

    results: List[Dict[str, Any]] = []
    for i, (note_id, comment_text) in enumerate(valid_items):
        sim = cosine_similarity(text_embeddings[i], note_embeddings[note_id])
        results.append(
            {
                "note_id": note_id,
                "comment_content": comment_text,
                "cosine_similarity": float(sim),
            }
        )
    return results


def aggregate_scores_by_note(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    按 note_id 聚合逐评论相似度，输出统计表。

    每个 note 输出:
    - comment_count: 有效评论条数
    - mean/min/max_cosine_similarity: 余弦相似度统计
    """
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        note_id = str(row.get("note_id") or "").strip()
        score = row.get("cosine_similarity")
        if note_id and isinstance(score, (int, float)):
            grouped[note_id].append(float(score))

    table: List[Dict[str, Any]] = []
    for note_id, vals in grouped.items():
        if not vals:
            continue
        table.append(
            {
                "note_id": note_id,
                "comment_count": len(vals),
                "mean_cosine_similarity": sum(vals) / len(vals),
                "min_cosine_similarity": min(vals),
                "max_cosine_similarity": max(vals),
            }
        )
    table.sort(key=lambda x: x["note_id"])
    return table


# ---------- 离线评估 ----------
def run_offline_evaluation(
    output_json: str,
    note_embeddings_path: str,
    config_path: Optional[str] = None,
    batch_size: int = 32,
) -> Tuple[float, int, List[Dict[str, Any]]]:
    """
    离线评估入口。

    步骤:
    1) 读取 embedding 配置与 ACL note 向量
    2) 从 output_json 提取生成评论
    3) 计算逐评论 cosine
    4) 聚合为 note 级统计

    返回: (全量均值, 有效评论数, 按 note 聚合表)。
    """
    if not config_path:
        root = _ensure_project_root()
        config_path = os.path.join(root, "config", "model_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    if not os.path.isfile(note_embeddings_path):
        raise FileNotFoundError(f"note embeddings 文件不存在: {note_embeddings_path}")
    base_url, model_name = load_embedding_config(config_path)
    note_embeddings = load_note_embeddings(note_embeddings_path)
    generated_comments = load_generated_comments_from_output(output_json)
    rows = compute_comment_note_cosine_scores(
        base_url, model_name, generated_comments, note_embeddings, batch_size=batch_size
    )
    if not rows:
        return 0.0, 0, []
    mean_cos = sum(r["cosine_similarity"] for r in rows) / len(rows)
    return float(mean_cos), len(rows), aggregate_scores_by_note(rows)


# ---------- 在线评估（监控指标）----------
def calculate_comment_similarity(data: Dict[str, Any]) -> Any:
    """
    在线监控指标入口（函数名为历史兼容保留）。

    语义:
    - 返回值是 mean cosine similarity

    在线指标：根据当前 content_pool 生成评论，计算其与对应 note embedding 的余弦相似度均值。
    需要 data 中包含 content_pool。
    可选 data.embedding_base_url / data.embedding_model_name 或 data.embedding_config_path 指定 embedding；
    可选 data.note_embeddings_path 指定 ACL embedding 文件路径；
    若未指定，优先尝试 data.reference_embedding_path（为兼容既有 env_data 字段）；
    若仍未提供，再使用默认路径:
    datasets/openreview/embeddings/bge-base-zh-v1.5_embeddings.json（与 env_data reference_embedding_path 一致）
    否则使用项目根下 config/model_config.json。
    若无有效评论或无可匹配 note embedding 则返回 None（监控可忽略该点）。
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
        log_metric_error("comment_similarity_mse", ValueError("无效的数据输入"), {"data": data})
        return None

    content_pool = safe_get(data, "content_pool", {})
    generated_comments = build_generated_comments_from_content_pool(content_pool)
    # 没有评论时返回 None，避免用 0.0 误导监控含义。
    if not generated_comments:
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
            log_metric_error("comment_similarity_mse", FileNotFoundError("未找到 embedding 配置"), {"data_keys": list(data.keys())})
            return None

    # 路径优先级：
    # 1) note_embeddings_path（新字段）
    # 2) reference_embedding_path（兼容旧 env_data 命名，实际可指向 embeddings json）
    # 3) 代码内默认路径
    note_embeddings_path = data.get("note_embeddings_path") or data.get("reference_embedding_path")
    if not note_embeddings_path:
        root = _ensure_project_root()
        note_embeddings_path = os.path.join(
            root,
            "datasets",
            "openreview",
            "embeddings",
            "bge-base-zh-v1.5_embeddings.json",
        )
    if not os.path.isfile(note_embeddings_path):
        log_metric_error("comment_similarity_mse", FileNotFoundError("未找到 note embeddings 文件"), {"path": note_embeddings_path})
        return None
    note_embeddings = load_note_embeddings(note_embeddings_path)
    # note embedding 为空时无法评估。
    if not note_embeddings:
        return None

    try:
        rows = compute_comment_note_cosine_scores(
            base_url, model_name, generated_comments, note_embeddings, batch_size=32
        )
        if not rows:
            return None
        mean_cos = sum(r["cosine_similarity"] for r in rows) / len(rows)
        return float(mean_cos)
    except Exception as e:
        log_metric_error("comment_similarity_mse", e, {"data_keys": list(data.keys())})
        return None


# ---------- CLI（离线）----------
def main() -> None:
    """
    离线 CLI：
    - 输入 output JSON 与 note embeddings JSON
    - 输出全体评论平均相似度 + note 级 CSV 表
    """
    parser = argparse.ArgumentParser(description="评论相似度离线评估：评论向量 vs note 向量余弦相似度")
    parser.add_argument("output_json", help="生成评论的 output JSON 路径")
    parser.add_argument("note_embeddings_json", help="ACL note embeddings JSON 路径（含 note_id, embedding）")
    parser.add_argument("--config", default="", help="model_config.json 路径，默认项目根 config/model_config.json")
    parser.add_argument("--batch-size", type=int, default=32, help="embedding 批大小")
    parser.add_argument(
        "--output-table",
        default="",
        help="离线按 note 聚合结果 CSV 输出路径（默认: 与 output_json 同目录下 note_comment_similarity.csv）",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.output_json):
        print(f"错误: output JSON 不存在 {args.output_json}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.note_embeddings_json):
        print(f"错误: note embeddings JSON 不存在 {args.note_embeddings_json}", file=sys.stderr)
        sys.exit(1)

    config_path = args.config.strip() or None
    try:
        mean_cos, n_rows, note_table = run_offline_evaluation(
            args.output_json,
            args.note_embeddings_json,
            config_path=config_path,
            batch_size=args.batch_size,
        )
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

    if args.output_table.strip():
        output_table = args.output_table.strip()
    else:
        output_table = os.path.join(
            os.path.dirname(os.path.abspath(args.output_json)),
            "note_comment_similarity.csv",
        )

    with open(output_table, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "note_id",
                "comment_count",
                "mean_cosine_similarity",
                "min_cosine_similarity",
                "max_cosine_similarity",
            ],
        )
        writer.writeheader()
        for row in note_table:
            writer.writerow(row)

    print(f"有效评论数: {n_rows}")
    print(f"Mean Cosine Similarity: {mean_cos:.6f}")
    print(f"按 note 聚合表已输出: {output_table}")


if __name__ == "__main__":
    main()
