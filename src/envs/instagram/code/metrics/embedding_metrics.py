# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from ..embedding import (
        cosine_similarity,
        default_embedding_config_path,
        find_project_root,
        get_embeddings,
        load_embedding_config,
    )
except ImportError:
    import sys

    _code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _code_dir not in sys.path:
        sys.path.insert(0, _code_dir)
    from embedding import (
        cosine_similarity,
        default_embedding_config_path,
        find_project_root,
        get_embeddings,
        load_embedding_config,
    )

try:
    from ..utils.text import format_real_text
except ImportError:
    from utils.text import format_real_text


@dataclass(frozen=True)
class CommentRecord:
    post_id: str
    comment_id: str
    text: str
    embedding: Optional[List[float]] = None


def resolve_embedding_api(data: Dict[str, Any]) -> Tuple[str, str]:
    base_url = data.get("embedding_base_url")
    model_name = data.get("embedding_model_name")
    if base_url and model_name:
        return str(base_url), str(model_name)
    config_path = data.get("embedding_config_path") or default_embedding_config_path()
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"embedding config not found: {config_path}")
    return load_embedding_config(config_path)


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


def collect_comment_records(
    content_pool: Dict[str, Any],
    *,
    strip_mentions: bool = True,
    reuse_existing: bool = True,
) -> List[CommentRecord]:
    rows: List[CommentRecord] = []
    if not isinstance(content_pool, dict):
        return rows
    for post_id, post in content_pool.items():
        if not isinstance(post, dict):
            continue
        nid = str(post.get("post_id") or post_id).strip()
        comments = post.get("comments") or {}
        if not isinstance(comments, dict):
            continue
        for cid_key, comment in comments.items():
            if not isinstance(comment, dict):
                continue
            cid = str(comment.get("comment_id") or cid_key).strip()
            raw_text = (comment.get("content") or "").strip()
            if not raw_text:
                continue
            text = format_real_text(raw_text) if strip_mentions else raw_text
            if not text:
                continue
            emb = None
            if reuse_existing:
                existing = comment.get("embedding")
                if isinstance(existing, list) and existing:
                    emb = [float(x) for x in existing]
            rows.append(CommentRecord(post_id=nid, comment_id=cid, text=text, embedding=emb))
    return rows


def embed_comment_records(
    records: List[CommentRecord],
    cache: EmbeddingCache,
) -> List[CommentRecord]:
    pending_texts = [r.text for r in records if r.embedding is None]
    cache.ensure_embedded(pending_texts)
    out: List[CommentRecord] = []
    for r in records:
        if r.embedding is not None:
            out.append(r)
        else:
            out.append(
                CommentRecord(
                    post_id=r.post_id,
                    comment_id=r.comment_id,
                    text=r.text,
                    embedding=list(cache.vector(r.text)),
                )
            )
    return out


def attach_embeddings_to_content_pool(
    content_pool: Dict[str, Any],
    records: Sequence[CommentRecord],
) -> Dict[str, Any]:
    """Return a shallow copy of content_pool with comment.embedding filled."""
    if not isinstance(content_pool, dict):
        return {}
    lookup = {f"{r.post_id}.{r.comment_id}": r.embedding for r in records if r.embedding}
    pool = {}
    for post_id, post in content_pool.items():
        if not isinstance(post, dict):
            pool[post_id] = post
            continue
        post_copy = dict(post)
        comments = post.get("comments") or {}
        if isinstance(comments, dict):
            new_comments = {}
            for cid_key, comment in comments.items():
                if not isinstance(comment, dict):
                    new_comments[cid_key] = comment
                    continue
                ccopy = dict(comment)
                nid = str(post.get("post_id") or post_id).strip()
                cid = str(comment.get("comment_id") or cid_key).strip()
                emb = lookup.get(f"{nid}.{cid}")
                if emb is not None:
                    ccopy["embedding"] = emb
                new_comments[cid_key] = ccopy
            post_copy["comments"] = new_comments
        pool[post_id] = post_copy
    return pool


def records_to_embedding_payload(records: Sequence[CommentRecord]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for r in records:
        if not r.embedding:
            continue
        payload.append(
            {
                "post_id": r.post_id,
                "comment_id": r.comment_id,
                "content": r.text,
                "embedding": r.embedding,
            }
        )
    return payload


def embed_content_pool_comments(
    content_pool: Dict[str, Any],
    data: Dict[str, Any],
    *,
    batch_size: int = 32,
) -> Tuple[List[CommentRecord], EmbeddingCache]:
    base_url, model_name = resolve_embedding_api(data)
    cache = EmbeddingCache(base_url, model_name, batch_size=batch_size)
    records = collect_comment_records(content_pool, reuse_existing=True)
    records = embed_comment_records(records, cache)
    return records, cache


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


def compute_max_reference_from_embeddings(
    records: Sequence[CommentRecord],
    post_embeddings: Dict[str, List[float]],
) -> Tuple[Optional[float], Dict[str, Any]]:
    generated_by_post: Dict[str, List[List[float]]] = defaultdict(list)
    for r in records:
        if r.embedding:
            generated_by_post[str(r.post_id).strip()].append(r.embedding)

    skipped_no_simulated = 0
    max_scores: List[float] = []
    for post_id, ref_emb in post_embeddings.items():
        sim_embs = generated_by_post.get(post_id, [])
        if not sim_embs:
            skipped_no_simulated += 1
            continue
        score = max(cosine_similarity(ref_emb, emb) for emb in sim_embs)
        max_scores.append(float(score))

    mean_score = float(sum(max_scores) / len(max_scores)) if max_scores else None
    meta = {
        "mean_max_cosine_similarity": mean_score,
        "direction": "reference_to_simulated",
        "n_reference_matched": len(max_scores),
        "n_reference_total": len(post_embeddings),
        "n_skipped_no_simulated": skipped_no_simulated,
        "n_simulated_posts": len(generated_by_post),
    }
    return mean_score, meta


def calculate_text_max_reference_similarity(data: Dict[str, Any]) -> Any:
    """Max-reference cosine: each reference post embedding vs same-post simulated comments, then mean."""
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
        if not collect_comment_records(content_pool):
            return None

        embeddings_path = default_post_embeddings_path(data)
        if not os.path.isfile(embeddings_path):
            log_metric_error(
                metric_id,
                FileNotFoundError("reference embedding JSON missing"),
                {"path": embeddings_path},
            )
            return None
        post_embeddings = load_post_embeddings(embeddings_path)
        if not post_embeddings:
            log_metric_error(
                metric_id,
                FileNotFoundError("reference embedding JSON empty"),
                {"path": embeddings_path},
            )
            return None

        records, _cache = embed_content_pool_comments(content_pool, data)
        if not records:
            return None

        mean_score, meta = compute_max_reference_from_embeddings(records, post_embeddings)
        if mean_score is None:
            return None

        return {
            "_viz_kind": "comment_max_reference_similarity",
            "mean_max_cosine_similarity": float(mean_score),
            **meta,
            "_comment_embeddings": records_to_embedding_payload(records),
            "reference_embedding_path": embeddings_path,
        }
    except Exception as e:
        log_metric_error(
            metric_id,
            e,
            {"data_keys": list(data.keys()) if isinstance(data, dict) else None},
        )
        return None


def default_post_embeddings_path(data: Optional[Dict[str, Any]] = None) -> str:
    data = data or {}
    path = data.get("post_embeddings_path") or data.get("reference_embedding_path")
    root = find_project_root()
    if path:
        path = str(path)
        if os.path.isfile(path):
            return path
        relative = path[1:] if path.startswith("/") else path
        candidate = os.path.join(root, *relative.replace("\\", "/").split("/"))
        if os.path.isfile(candidate):
            return candidate
    return os.path.join(root, "datasets", "ins", "embeddings", "bge-base-en-v1.5_embeddings.json")


_POST_EMB_CACHE: Dict[str, Dict[str, List[float]]] = {}


def load_post_embeddings(post_embeddings_path: str) -> Dict[str, List[float]]:
    cached = _POST_EMB_CACHE.get(post_embeddings_path)
    if cached is not None:
        return cached
    with open(post_embeddings_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    result: Dict[str, List[float]] = {}
    if not isinstance(items, list):
        _POST_EMB_CACHE[post_embeddings_path] = result
        return result
    for it in items:
        if not isinstance(it, dict):
            continue
        post_id = str(it.get("post_id") or it.get("note_id") or "").strip()
        emb = it.get("embedding")
        if not post_id or not isinstance(emb, list) or not emb:
            continue
        result[post_id] = emb
    _POST_EMB_CACHE[post_embeddings_path] = result
    return result


def compute_comment_post_cosine_scores(
    records: Sequence[CommentRecord],
    post_embeddings: Dict[str, List[float]],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for r in records:
        if not r.embedding:
            continue
        post_emb = post_embeddings.get(r.post_id)
        if not post_emb:
            continue
        sim = cosine_similarity(r.embedding, post_emb)
        results.append(
            {
                "post_id": r.post_id,
                "comment_id": r.comment_id,
                "comment_content": r.text,
                "cosine_similarity": float(sim),
            }
        )
    return results


def calculate_text_similarity(data: Dict[str, Any]) -> Any:
    """Mean cosine similarity between comment embeddings and precomputed post embeddings."""
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
        if not collect_comment_records(content_pool):
            return None

        post_embeddings_path = default_post_embeddings_path(data)
        if not os.path.isfile(post_embeddings_path):
            log_metric_error(
                metric_id,
                FileNotFoundError("post embeddings file not found"),
                {"path": post_embeddings_path},
            )
            return None
        post_embeddings = load_post_embeddings(post_embeddings_path)
        if not post_embeddings:
            return None

        records, _cache = embed_content_pool_comments(content_pool, data)
        if not records:
            return None

        rows = compute_comment_post_cosine_scores(records, post_embeddings)
        if not rows:
            return None
        return float(sum(r["cosine_similarity"] for r in rows) / len(rows))
    except FileNotFoundError as e:
        log_metric_error(metric_id, e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return None
    except Exception as e:
        log_metric_error(metric_id, e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return None
