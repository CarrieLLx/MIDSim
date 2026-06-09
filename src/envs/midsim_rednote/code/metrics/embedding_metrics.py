# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
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
except ImportError:
    import sys

    _code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _code_dir not in sys.path:
        sys.path.insert(0, _code_dir)
    from embedding_client import (
        cosine_similarity,
        default_embedding_config_path,
        find_project_root,
        get_embeddings,
        load_embedding_config,
    )

try:
    from ..utils import format_real_text
except ImportError:
    from utils import format_real_text


@dataclass(frozen=True)
class CommentRecord:
    note_id: str
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
    for note_id, note in content_pool.items():
        if not isinstance(note, dict):
            continue
        nid = str(note.get("note_id") or note_id).strip()
        comments = note.get("comments") or {}
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
            rows.append(CommentRecord(note_id=nid, comment_id=cid, text=text, embedding=emb))
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
                    note_id=r.note_id,
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
    lookup = {f"{r.note_id}.{r.comment_id}": r.embedding for r in records if r.embedding}
    pool = {}
    for note_id, note in content_pool.items():
        if not isinstance(note, dict):
            pool[note_id] = note
            continue
        note_copy = dict(note)
        comments = note.get("comments") or {}
        if isinstance(comments, dict):
            new_comments = {}
            for cid_key, comment in comments.items():
                if not isinstance(comment, dict):
                    new_comments[cid_key] = comment
                    continue
                ccopy = dict(comment)
                nid = str(note.get("note_id") or note_id).strip()
                cid = str(comment.get("comment_id") or cid_key).strip()
                emb = lookup.get(f"{nid}.{cid}")
                if emb is not None:
                    ccopy["embedding"] = emb
                new_comments[cid_key] = ccopy
            note_copy["comments"] = new_comments
        pool[note_id] = note_copy
    return pool


def records_to_embedding_payload(records: Sequence[CommentRecord]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for r in records:
        if not r.embedding:
            continue
        payload.append(
            {
                "note_id": r.note_id,
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


def default_reference_csv_path() -> str:
    root = find_project_root()
    candidates = [
        os.path.join(root, "datasets", "rednote", "comments.csv"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return candidates[0]


def load_reference_comments_by_note(csv_path: str) -> Dict[str, List[str]]:
    by_note: Dict[str, List[str]] = defaultdict(list)
    if not csv_path or not os.path.isfile(csv_path):
        return {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            note_id = str(row.get("note_id") or "").strip()
            text = format_real_text(row.get("content") or "")
            if note_id and text:
                by_note[note_id].append(text)
    return dict(by_note)


def _group_generated_by_note(records: Sequence[CommentRecord]) -> Dict[str, List[str]]:
    by_note: Dict[str, List[str]] = defaultdict(list)
    for r in records:
        by_note[str(r.note_id).strip()].append(r.text)
    return dict(by_note)


def compute_max_reference_cosine_mean(
    generated_by_note: Dict[str, List[str]],
    reference_by_note: Dict[str, List[str]],
    cache: EmbeddingCache,
) -> Tuple[Optional[float], Dict[str, Any]]:
    ref_items: List[Tuple[str, str]] = []
    unique_texts: set[str] = set()
    skipped_no_simulated = 0

    for note_id, ref_texts in reference_by_note.items():
        sim_texts = generated_by_note.get(note_id, [])
        if not sim_texts:
            skipped_no_simulated += len(ref_texts)
            continue
        unique_texts.update(sim_texts)
        for ref_text in ref_texts:
            if not ref_text:
                continue
            ref_items.append((note_id, ref_text))
            unique_texts.add(ref_text)

    if not ref_items:
        return None, {
            "n_reference_matched": 0,
            "n_reference_total": sum(len(v) for v in reference_by_note.values()),
            "n_skipped_no_simulated": skipped_no_simulated,
            "n_simulated_notes": len(generated_by_note),
        }

    cache.ensure_embedded(unique_texts)
    max_scores: List[float] = []
    for note_id, ref_text in ref_items:
        sim_texts = generated_by_note.get(note_id, [])
        score = cache.max_cosine_against(ref_text, sim_texts)
        if score is not None:
            max_scores.append(score)

    mean_score = float(sum(max_scores) / len(max_scores)) if max_scores else None
    meta = {
        "mean_max_cosine_similarity": mean_score,
        "direction": "reference_to_simulated",
        "n_reference_matched": len(max_scores),
        "n_reference_total": sum(len(v) for v in reference_by_note.values()),
        "n_skipped_no_simulated": skipped_no_simulated,
        "n_simulated_notes": len(generated_by_note),
        "n_unique_embedded": len(cache),
    }
    return mean_score, meta


def calculate_text_max_reference_similarity(data: Dict[str, Any]) -> Any:
    """Max-reference cosine: each reference comment vs same-note simulated comments, then mean."""
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
            safe_get(data, "reference_csv_path", None)
            or default_reference_csv_path()
        )
        reference_by_note = load_reference_comments_by_note(str(csv_path))
        if not reference_by_note:
            log_metric_error(
                metric_id,
                FileNotFoundError("reference comments CSV missing or empty"),
                {"path": csv_path},
            )
            return None

        base_url, model_name = resolve_embedding_api(data)
        cache = EmbeddingCache(base_url, model_name)
        records = collect_comment_records(content_pool, reuse_existing=True)
        if not records:
            return None
        records = embed_comment_records(records, cache)
        generated_by_note = _group_generated_by_note(records)

        mean_score, meta = compute_max_reference_cosine_mean(
            generated_by_note, reference_by_note, cache
        )
        if mean_score is None:
            return None

        return {
            "_viz_kind": "comment_max_reference_similarity",
            "mean_max_cosine_similarity": float(mean_score),
            **meta,
            "_comment_embeddings": records_to_embedding_payload(records),
            "reference_csv_path": str(csv_path),
        }
    except Exception as e:
        log_metric_error(
            metric_id,
            e,
            {"data_keys": list(data.keys()) if isinstance(data, dict) else None},
        )
        return None


def default_note_embeddings_path(data: Optional[Dict[str, Any]] = None) -> str:
    data = data or {}
    path = data.get("note_embeddings_path") or data.get("reference_embedding_path")
    if path and os.path.isfile(str(path)):
        return str(path)
    root = find_project_root()
    for rel in (
        "datasets/rednote/embeddings/bge-base-zh-v1.5_embeddings.json",
        "datasets/openreview/embeddings/bge-base-zh-v1.5_embeddings.json",
    ):
        candidate = os.path.join(root, *rel.split("/"))
        if os.path.isfile(candidate):
            return candidate
    return os.path.join(root, "datasets", "rednote", "embeddings", "bge-base-zh-v1.5_embeddings.json")


def load_note_embeddings(note_embeddings_path: str) -> Dict[str, List[float]]:
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


def compute_comment_note_cosine_scores(
    records: Sequence[CommentRecord],
    note_embeddings: Dict[str, List[float]],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for r in records:
        if not r.embedding:
            continue
        note_emb = note_embeddings.get(r.note_id)
        if not note_emb:
            continue
        sim = cosine_similarity(r.embedding, note_emb)
        results.append(
            {
                "note_id": r.note_id,
                "comment_id": r.comment_id,
                "comment_content": r.text,
                "cosine_similarity": float(sim),
            }
        )
    return results


def calculate_text_similarity(data: Dict[str, Any]) -> Any:
    """Mean cosine similarity between comment embeddings and precomputed note embeddings."""
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

        note_embeddings_path = default_note_embeddings_path(data)
        if not os.path.isfile(note_embeddings_path):
            log_metric_error(
                metric_id,
                FileNotFoundError("note embeddings file not found"),
                {"path": note_embeddings_path},
            )
            return None
        note_embeddings = load_note_embeddings(note_embeddings_path)
        if not note_embeddings:
            return None

        records, _cache = embed_content_pool_comments(content_pool, data)
        if not records:
            return None

        rows = compute_comment_note_cosine_scores(records, note_embeddings)
        if not rows:
            return None
        return float(sum(r["cosine_similarity"] for r in rows) / len(rows))
    except FileNotFoundError as e:
        log_metric_error(metric_id, e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return None
    except Exception as e:
        log_metric_error(metric_id, e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return None
