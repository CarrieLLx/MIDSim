# -*- coding: utf-8 -*-
"""Propagation embedding helpers and max-reference cosine metric for Twitter."""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from ..embedding_client import (
        cosine_similarity,
        default_embedding_config_path,
        find_project_root,
        get_embeddings,
        load_embedding_config,
    )
    from .propagation_text_utils import (
        PropagationRecord,
        collect_propagation_records,
        load_reference_propagations_by_root,
        original_root_ids,
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
    from propagation_text_utils import (
        PropagationRecord,
        collect_propagation_records,
        load_reference_propagations_by_root,
        original_root_ids,
    )


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


def embed_propagation_records(
    records: List[PropagationRecord],
    cache: EmbeddingCache,
) -> List[PropagationRecord]:
    pending_texts = [r.text for r in records if r.embedding is None]
    cache.ensure_embedded(pending_texts)
    out: List[PropagationRecord] = []
    for r in records:
        if r.embedding is not None:
            out.append(r)
        else:
            out.append(
                PropagationRecord(
                    root_tweet_id=r.root_tweet_id,
                    tweet_id=r.tweet_id,
                    text=r.text,
                    embedding=list(cache.vector(r.text)),
                )
            )
    return out


def attach_embeddings_to_content_pool(
    content_pool: Dict[str, Any],
    records: Sequence[PropagationRecord],
) -> Dict[str, Any]:
    if not isinstance(content_pool, dict):
        return {}
    lookup = {r.tweet_id: r.embedding for r in records if r.embedding}

    def attach_tweet(tweet: Dict[str, Any]) -> Dict[str, Any]:
        blog_copy = dict(tweet)
        tid = str(tweet.get("tweet_id") or tweet.get("note_id") or "").strip()
        emb = lookup.get(tid)
        if emb is not None:
            blog_copy["embedding"] = emb
        for key in ("retweets", "qoutes", "quotes", "reply", "replies"):
            nested = tweet.get(key)
            if isinstance(nested, dict):
                blog_copy[key] = {
                    nk: attach_tweet(item) if isinstance(item, dict) else item
                    for nk, item in nested.items()
                }
        return blog_copy

    pool: Dict[str, Any] = {}
    for bid, tweet in content_pool.items():
        if isinstance(tweet, dict):
            pool[bid] = attach_tweet(tweet)
        else:
            pool[bid] = tweet
    return pool


def records_to_embedding_payload(records: Sequence[PropagationRecord]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for r in records:
        if not r.embedding:
            continue
        payload.append(
            {
                "root_tweet_id": r.root_tweet_id,
                "tweet_id": r.tweet_id,
                "content": r.text,
                "embedding": r.embedding,
            }
        )
    return payload


def embed_content_pool_propagations(
    content_pool: Dict[str, Any],
    data: Dict[str, Any],
    *,
    batch_size: int = 32,
) -> Tuple[List[PropagationRecord], EmbeddingCache]:
    base_url, model_name = resolve_embedding_api(data)
    cache = EmbeddingCache(base_url, model_name, batch_size=batch_size)
    records = collect_propagation_records(content_pool, reuse_existing=True)
    records = embed_propagation_records(records, cache)
    return records, cache


def default_reference_propagations_csv_path() -> str:
    root = find_project_root()
    candidates = [
        os.path.join(root, "datasets", "twitter", "reposts.csv"),
        os.path.join(root, "datasets", "twitter-openreview", "reposts.csv"),
        os.path.join(root, "datasets", "openreview", "reposts.csv"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return candidates[0]


def _group_generated_by_root(records: Sequence[PropagationRecord]) -> Dict[str, List[str]]:
    by_root: Dict[str, List[str]] = defaultdict(list)
    for r in records:
        by_root[str(r.root_tweet_id).strip()].append(r.text)
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


def calculate_text_max_reference_similarity(data: Dict[str, Any]) -> Any:
    """Max-reference cosine: each reference quote/reply vs same-root simulated texts, then mean."""
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
            safe_get(data, "reference_propagations_csv_path", None)
            or safe_get(data, "reference_csv_path", None)
            or default_reference_propagations_csv_path()
        )
        sim_roots = original_root_ids(content_pool)
        reference_by_root = load_reference_propagations_by_root(str(csv_path), sim_roots or None)
        if not reference_by_root:
            log_metric_error(
                metric_id,
                FileNotFoundError("reference reposts CSV missing or empty"),
                {"path": csv_path},
            )
            return None

        base_url, model_name = resolve_embedding_api(data)
        cache = EmbeddingCache(base_url, model_name)
        records = collect_propagation_records(content_pool, reuse_existing=True)
        if not records:
            return None
        records = embed_propagation_records(records, cache)
        generated_by_root = _group_generated_by_root(records)

        mean_score, meta = compute_max_reference_cosine_mean(
            generated_by_root, reference_by_root, cache
        )
        if mean_score is None:
            return None

        return {
            "_viz_kind": "propagation_max_reference_similarity",
            "mean_max_cosine_similarity": float(mean_score),
            **meta,
            "_propagation_embeddings": records_to_embedding_payload(records),
            "reference_csv_path": str(csv_path),
        }
    except Exception as e:
        log_metric_error(
            metric_id,
            e,
            {"data_keys": list(data.keys()) if isinstance(data, dict) else None},
        )
        return None
