# -*- coding: utf-8 -*-
"""Repost embedding helpers and online text similarity metrics for Weibo."""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from ..embedding_client import (
        cosine_similarity,
        default_embedding_config_path,
        find_project_root,
        get_embeddings,
        load_embedding_config,
    )
    from ..utils import is_repost_of_other_blog
    from .repost_text_utils import (
        format_repost_text,
        get_blog_id,
        load_reference_reposts_by_root,
        original_root_ids,
        resolve_root_blog_id,
    )
except ImportError:
    import sys

    _metrics_dir = os.path.dirname(os.path.abspath(__file__))
    _code_dir = os.path.dirname(_metrics_dir)
    if _metrics_dir not in sys.path:
        sys.path.insert(0, _metrics_dir)
    if _code_dir not in sys.path:
        sys.path.insert(0, _code_dir)
    from embedding_client import (
        cosine_similarity,
        default_embedding_config_path,
        find_project_root,
        get_embeddings,
        load_embedding_config,
    )
    from utils import is_repost_of_other_blog
    from repost_text_utils import (
        format_repost_text,
        get_blog_id,
        load_reference_reposts_by_root,
        original_root_ids,
        resolve_root_blog_id,
    )


@dataclass(frozen=True)
class RepostRecord:
    root_blog_id: str
    ref_blog_id: str
    blog_id: str
    text: str
    embedding: Optional[List[float]] = None


class EmbeddingCache:
    """Text-level embedding cache with optional batch HTTP calls."""

    def __init__(self, base_url: str, model_name: str, batch_size: int = 32) -> None:
        self.base_url = base_url
        self.model_name = model_name
        self.batch_size = max(1, int(batch_size))
        self._cache: Dict[str, List[float]] = {}

    def ensure_embedded(self, texts: Sequence[str]) -> None:
        pending: List[str] = []
        seen: set[str] = set()
        for raw in texts:
            t = (raw or "").strip()
            if not t or t in self._cache or t in seen:
                continue
            seen.add(t)
            pending.append(t)
        if not pending:
            return
        for i in range(0, len(pending), self.batch_size):
            batch = pending[i : i + self.batch_size]
            vectors = get_embeddings(self.base_url, self.model_name, batch)
            for text, vec in zip(batch, vectors):
                self._cache[text] = vec

    def vector(self, text: str) -> List[float]:
        self.ensure_embedded([text])
        return self._cache[(text or "").strip()]

    def max_cosine_against(self, query_text: str, candidate_texts: Sequence[str]) -> Optional[float]:
        candidates = [(t or "").strip() for t in candidate_texts if (t or "").strip()]
        q = (query_text or "").strip()
        if not q or not candidates:
            return None
        self.ensure_embedded([q, *candidates])
        qv = self._cache[q]
        best = max(cosine_similarity(qv, self._cache[c]) for c in candidates)
        return float(best)

    def __len__(self) -> int:
        return len(self._cache)


def _monitor_utils():
    try:
        from onesim.monitor.utils import log_metric_error, safe_get
    except ImportError:
        try:
            from onesim_cn.monitor.utils import log_metric_error, safe_get
        except ImportError:

            def safe_get(d, k, default=None):
                return (d or {}).get(k, default) if isinstance(d, dict) else default

            def log_metric_error(name, e, ctx):
                pass

    return safe_get, log_metric_error


def resolve_embedding_api(data: Dict[str, Any]) -> Tuple[str, str]:
    base_url = data.get("embedding_base_url")
    model_name = data.get("embedding_model_name")
    if base_url and model_name:
        return str(base_url), str(model_name)
    config_path = data.get("embedding_config_path") or default_embedding_config_path()
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"embedding config not found: {config_path}")
    return load_embedding_config(config_path)


def collect_repost_records(
    content_pool: Dict[str, Any],
    *,
    reuse_existing: bool = True,
) -> List[RepostRecord]:
    rows: List[RepostRecord] = []
    if not isinstance(content_pool, dict):
        return rows
    cache: Dict[str, str] = {}
    for bid, blog in content_pool.items():
        if not isinstance(blog, dict):
            continue
        blog_id = get_blog_id(blog, str(bid))
        if not blog_id or not is_repost_of_other_blog(blog_id, blog):
            continue
        text = format_repost_text(blog.get("content") or "")
        if not text:
            continue
        root_id = resolve_root_blog_id(blog_id, blog, content_pool, cache)
        ref_id = str(blog.get("reposted_blog_id") or "").strip()
        emb = None
        if reuse_existing:
            existing = blog.get("embedding")
            if isinstance(existing, list) and existing:
                emb = [float(x) for x in existing]
        rows.append(
            RepostRecord(
                root_blog_id=str(root_id).strip(),
                ref_blog_id=ref_id,
                blog_id=blog_id,
                text=text,
                embedding=emb,
            )
        )
    return rows


def embed_repost_records(records: List[RepostRecord], cache: EmbeddingCache) -> List[RepostRecord]:
    pending_texts = [r.text for r in records if r.embedding is None]
    cache.ensure_embedded(pending_texts)
    out: List[RepostRecord] = []
    for r in records:
        if r.embedding is not None:
            out.append(r)
        else:
            out.append(
                RepostRecord(
                    root_blog_id=r.root_blog_id,
                    ref_blog_id=r.ref_blog_id,
                    blog_id=r.blog_id,
                    text=r.text,
                    embedding=list(cache.vector(r.text)),
                )
            )
    return out


def attach_embeddings_to_content_pool(
    content_pool: Dict[str, Any],
    records: Sequence[RepostRecord],
) -> Dict[str, Any]:
    if not isinstance(content_pool, dict):
        return {}
    lookup = {r.blog_id: r.embedding for r in records if r.embedding}
    pool: Dict[str, Any] = {}
    for bid, blog in content_pool.items():
        if not isinstance(blog, dict):
            pool[bid] = blog
            continue
        blog_copy = dict(blog)
        blog_id = get_blog_id(blog, str(bid))
        emb = lookup.get(blog_id)
        if emb is not None:
            blog_copy["embedding"] = emb
        pool[bid] = blog_copy
    return pool


def records_to_embedding_payload(records: Sequence[RepostRecord]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for r in records:
        if not r.embedding:
            continue
        payload.append(
            {
                "root_blog_id": r.root_blog_id,
                "ref_blog_id": r.ref_blog_id,
                "blog_id": r.blog_id,
                "content": r.text,
                "embedding": r.embedding,
            }
        )
    return payload


def embed_content_pool_reposts(
    content_pool: Dict[str, Any],
    data: Dict[str, Any],
    *,
    batch_size: int = 32,
) -> Tuple[List[RepostRecord], EmbeddingCache]:
    base_url, model_name = resolve_embedding_api(data)
    cache = EmbeddingCache(base_url, model_name, batch_size=batch_size)
    records = collect_repost_records(content_pool, reuse_existing=True)
    records = embed_repost_records(records, cache)
    return records, cache


def default_reference_reposts_csv_path() -> str:
    root = find_project_root()
    candidates = [
        os.path.join(root, "datasets", "weibo", "reposts.csv"),
        os.path.join(root, "datasets", "weibo-openreview", "reposts.csv"),
        os.path.join(root, "datasets", "openreview", "reposts.csv"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return candidates[0]


def _group_generated_by_root(records: Sequence[RepostRecord]) -> Dict[str, List[str]]:
    by_root: Dict[str, List[str]] = defaultdict(list)
    for r in records:
        by_root[str(r.root_blog_id).strip()].append(r.text)
    return dict(by_root)


def compute_max_reference_cosine_mean(
    generated_by_root: Dict[str, List[str]],
    reference_by_root: Dict[str, List[str]],
    cache: EmbeddingCache,
) -> Tuple[Optional[float], Dict[str, Any]]:
    ref_items: List[Tuple[str, str]] = []
    unique_texts: set[str] = set()
    skipped_no_simulated = 0

    for root_id, ref_texts in reference_by_root.items():
        sim_texts = generated_by_root.get(root_id, [])
        if not sim_texts:
            skipped_no_simulated += len(ref_texts)
            continue
        unique_texts.update(sim_texts)
        for ref_text in ref_texts:
            if not ref_text:
                continue
            ref_items.append((root_id, ref_text))
            unique_texts.add(ref_text)

    if not ref_items:
        return None, {
            "n_reference_matched": 0,
            "n_reference_total": sum(len(v) for v in reference_by_root.values()),
            "n_skipped_no_simulated": skipped_no_simulated,
            "n_simulated_roots": len(generated_by_root),
        }

    cache.ensure_embedded(unique_texts)
    max_scores: List[float] = []
    for root_id, ref_text in ref_items:
        sim_texts = generated_by_root.get(root_id, [])
        score = cache.max_cosine_against(ref_text, sim_texts)
        if score is not None:
            max_scores.append(score)

    mean_score = float(sum(max_scores) / len(max_scores)) if max_scores else None
    meta = {
        "mean_max_cosine_similarity": mean_score,
        "direction": "reference_to_simulated",
        "n_reference_matched": len(max_scores),
        "n_reference_total": sum(len(v) for v in reference_by_root.values()),
        "n_skipped_no_simulated": skipped_no_simulated,
        "n_simulated_roots": len(generated_by_root),
        "n_unique_embedded": len(cache),
    }
    return mean_score, meta


def default_blog_embeddings_path(data: Optional[Dict[str, Any]] = None) -> str:
    data = data or {}
    path = (
        data.get("blog_embeddings_path")
        or data.get("reference_embedding_path")
        or data.get("reference_csv_path")
    )
    if path and os.path.isfile(str(path)):
        return str(path)
    root = find_project_root()
    for rel in (
        "src/envs/multi_channel_information_diffusion/profile/data/acl/embeddings/bge-base-zh-v1.5_embeddings.json",
        "datasets/weibo/embeddings/bge-base-zh-v1.5_embeddings.json",
        "datasets/weibo-openreview/embeddings/bge-base-zh-v1.5_embeddings.json",
    ):
        candidate = os.path.join(root, *rel.split("/"))
        if os.path.isfile(candidate):
            return candidate
    return os.path.join(
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


def load_blog_embeddings(blog_embeddings_path: str) -> Dict[str, List[float]]:
    with open(blog_embeddings_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    result: Dict[str, List[float]] = {}
    if not isinstance(items, list):
        return result
    for it in items:
        if not isinstance(it, dict):
            continue
        blog_id = str(it.get("blog_id") or it.get("note_id") or "").strip()
        emb = it.get("embedding")
        if not blog_id or not isinstance(emb, list) or not emb:
            continue
        result[blog_id] = emb
    return result


def compute_repost_ref_cosine_scores(
    records: Sequence[RepostRecord],
    blog_embeddings: Dict[str, List[float]],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for r in records:
        if not r.embedding:
            continue
        ref_emb = blog_embeddings.get(r.ref_blog_id)
        if not ref_emb:
            continue
        sim = cosine_similarity(r.embedding, ref_emb)
        results.append(
            {
                "ref_blog_id": r.ref_blog_id,
                "blog_id": r.blog_id,
                "repost_content": r.text,
                "cosine_similarity": float(sim),
            }
        )
    return results


def calculate_text_similarity(data: Dict[str, Any]) -> Any:
    """Mean cosine similarity between repost embeddings and reposted blog embeddings."""
    safe_get, log_metric_error = _monitor_utils()
    metric_id = "text_similarity"
    try:
        if not data or not isinstance(data, dict):
            log_metric_error(metric_id, ValueError("Invalid data input"), {"data": data})
            return None

        content_pool = safe_get(data, "content_pool", {})
        if not isinstance(content_pool, dict):
            log_metric_error(metric_id, ValueError("content_pool is not a dict"), {})
            return None

        blog_embeddings_path = default_blog_embeddings_path(data)
        if not os.path.isfile(blog_embeddings_path):
            log_metric_error(
                metric_id,
                FileNotFoundError("blog embeddings file not found"),
                {"path": blog_embeddings_path},
            )
            return None
        blog_embeddings = load_blog_embeddings(blog_embeddings_path)
        if not blog_embeddings:
            return None

        records, _cache = embed_content_pool_reposts(content_pool, data)
        if not records:
            return None

        rows = compute_repost_ref_cosine_scores(records, blog_embeddings)
        if not rows:
            return None
        return float(sum(r["cosine_similarity"] for r in rows) / len(rows))
    except FileNotFoundError as e:
        log_metric_error(metric_id, e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return None
    except Exception as e:
        log_metric_error(metric_id, e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return None


def calculate_text_max_reference_similarity(data: Dict[str, Any]) -> Any:
    """Max-reference cosine: each reference repost vs same-root simulated reposts, then mean."""
    safe_get, log_metric_error = _monitor_utils()
    metric_id = "text_max_reference_similarity"
    try:
        if not data or not isinstance(data, dict):
            log_metric_error(metric_id, ValueError("Invalid data input"), {"data": data})
            return None

        content_pool = safe_get(data, "content_pool", {})
        if not isinstance(content_pool, dict):
            log_metric_error(metric_id, ValueError("content_pool is not a dict"), {})
            return None

        csv_path = (
            safe_get(data, "reference_reposts_csv_path", None)
            or safe_get(data, "reference_csv_path", None)
            or default_reference_reposts_csv_path()
        )
        sim_roots = original_root_ids(content_pool)
        reference_by_root = load_reference_reposts_by_root(str(csv_path), sim_roots or None)
        if not reference_by_root:
            log_metric_error(
                metric_id,
                FileNotFoundError("reference reposts CSV missing or empty"),
                {"path": csv_path},
            )
            return None

        base_url, model_name = resolve_embedding_api(data)
        cache = EmbeddingCache(base_url, model_name)
        records = collect_repost_records(content_pool, reuse_existing=True)
        if not records:
            return None
        records = embed_repost_records(records, cache)
        generated_by_root = _group_generated_by_root(records)

        mean_score, meta = compute_max_reference_cosine_mean(
            generated_by_root, reference_by_root, cache
        )
        if mean_score is None:
            return None

        return {
            "_viz_kind": "repost_max_reference_similarity",
            "mean_max_cosine_similarity": float(mean_score),
            **meta,
            "_repost_embeddings": records_to_embedding_payload(records),
            "reference_csv_path": str(csv_path),
        }
    except Exception as e:
        log_metric_error(
            metric_id,
            e,
            {"data_keys": list(data.keys()) if isinstance(data, dict) else None},
        )
        return None
