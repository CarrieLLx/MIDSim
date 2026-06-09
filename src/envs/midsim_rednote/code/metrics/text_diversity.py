# -*- coding: utf-8 -*-
"""
Text diversity (online metric): TTR, Distinct-2/3, 1-Self-BLEU, Div_sem.
"""
from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, Sequence

try:
    from ..embedding_client import cosine_similarity
except ImportError:
    import sys

    _code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _code_dir not in sys.path:
        sys.path.insert(0, _code_dir)
    from embedding_client import cosine_similarity

try:
    from .embedding_metrics import (
        attach_embeddings_to_content_pool,
        embed_content_pool_comments,
        records_to_embedding_payload,
    )
    from ..utils import tokenize
except ImportError:
    import sys

    _metrics_dir = os.path.dirname(os.path.abspath(__file__))
    _code_dir = os.path.dirname(_metrics_dir)
    if _metrics_dir not in sys.path:
        sys.path.insert(0, _metrics_dir)
    if _code_dir not in sys.path:
        sys.path.insert(0, _code_dir)
    from embedding_metrics import (
        attach_embeddings_to_content_pool,
        embed_content_pool_comments,
        records_to_embedding_payload,
    )
    from utils import tokenize

DEFAULT_TRUNCATE_TOKENS = 32
DEFAULT_DISTINCT_SAMPLE_SIZE = 1000
DEFAULT_SELF_BLEU_MAX = 500
DEFAULT_SEMANTIC_MAX = 800
DEFAULT_SEED = 42


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


def truncate_tokens(text: str, max_tokens: int) -> str:
    toks = tokenize(text)
    if len(toks) <= max_tokens:
        return text.strip()
    return "".join(toks[:max_tokens])


def corpus_ttr(texts: Sequence[str], max_tokens: int) -> float:
    types: set[str] = set()
    tokens = 0
    for text in texts:
        toks = tokenize(truncate_tokens(text, max_tokens))
        types.update(toks)
        tokens += len(toks)
    if tokens == 0:
        return 0.0
    return len(types) / tokens


def corpus_distinct_n(texts: Sequence[str], n: int, max_tokens: int) -> float:
    ngrams: List[str] = []
    for text in texts:
        toks = tokenize(truncate_tokens(text, max_tokens))
        if len(toks) < n:
            continue
        ngrams.extend("".join(toks[i : i + n]) for i in range(len(toks) - n + 1))
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def one_minus_self_bleu(texts: Sequence[str], max_samples: int, seed: int) -> Optional[float]:
    cleaned = [t.strip() for t in texts if (t or "").strip()]
    if len(cleaned) < 2:
        return 0.0
    try:
        from sacrebleu import corpus_bleu
    except ImportError:
        return None
    if len(cleaned) > max_samples:
        cleaned = random.Random(seed).sample(cleaned, max_samples)
    bleu_texts = [" ".join(tokenize(t)) for t in cleaned]
    scores: List[float] = []
    for i, hyp in enumerate(bleu_texts):
        refs = [bleu_texts[j] for j in range(len(bleu_texts)) if j != i]
        scores.append(corpus_bleu([hyp], [refs]).score / 100.0)
    return 1.0 - float(sum(scores) / len(scores))


def pairwise_mean_cosine_distance(vectors: Sequence[Sequence[float]]) -> float:
    vecs = [list(v) for v in vectors if v]
    n = len(vecs)
    if n < 2:
        return 0.0
    dists: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(vecs[i], vecs[j])
            dists.append(1.0 - sim)
    return float(sum(dists) / len(dists))


def div_semantic_from_vectors(
    vectors: Sequence[Sequence[float]],
    max_samples: int,
    seed: int,
) -> float:
    vecs = [list(v) for v in vectors if v]
    if len(vecs) < 2:
        return 0.0
    if len(vecs) > max_samples:
        vecs = random.Random(seed).sample(vecs, max_samples)
    return pairwise_mean_cosine_distance(vecs)


def compute_diversity_metrics(
    texts: Sequence[str],
    vectors: Sequence[Sequence[float]],
    *,
    truncate_tokens_n: int = DEFAULT_TRUNCATE_TOKENS,
    distinct_sample_size: int = DEFAULT_DISTINCT_SAMPLE_SIZE,
    self_bleu_max: int = DEFAULT_SELF_BLEU_MAX,
    semantic_max: int = DEFAULT_SEMANTIC_MAX,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Optional[float]]:
    cleaned = [t.strip() for t in texts if (t or "").strip()]
    if not cleaned:
        raise ValueError("no valid comment texts")

    rng = random.Random(seed)
    distinct_texts = cleaned
    if len(cleaned) > distinct_sample_size:
        distinct_texts = rng.sample(cleaned, distinct_sample_size)

    omsb = one_minus_self_bleu(cleaned, self_bleu_max, seed)
    div_sem = div_semantic_from_vectors(vectors, semantic_max, seed) if vectors else None

    return {
        "ttr": corpus_ttr(cleaned, truncate_tokens_n),
        "distinct_2": corpus_distinct_n(distinct_texts, 2, truncate_tokens_n),
        "distinct_3": corpus_distinct_n(distinct_texts, 3, truncate_tokens_n),
        "one_minus_self_bleu": omsb,
        "div_sem": div_sem,
        "n_comments": float(len(cleaned)),
    }


def calculate_text_diversity(data: Dict[str, Any]) -> Any:
    safe_get, log_metric_error = _monitor_utils()
    empty: Dict[str, Any] = {
        "_viz_kind": "comment_diversity",
        "ttr": None,
        "distinct_2": None,
        "distinct_3": None,
        "one_minus_self_bleu": None,
        "div_sem": None,
        "n_comments": 0,
        "_comment_embeddings": [],
        "content_pool_with_embeddings": {},
    }
    metric_id = "text_diversity"
    try:
        if not data or not isinstance(data, dict):
            log_metric_error(metric_id, ValueError("Invalid data input"), {"data": data})
            return empty

        content_pool = safe_get(data, "content_pool", {})
        if not isinstance(content_pool, dict):
            log_metric_error(metric_id, ValueError("content_pool is not a dict"), {})
            return empty

        records, _cache = embed_content_pool_comments(content_pool, data)
        if not records:
            return empty

        texts = [r.text for r in records]
        vectors = [r.embedding for r in records if r.embedding]
        metrics = compute_diversity_metrics(texts, vectors)

        pool_with_emb = attach_embeddings_to_content_pool(content_pool, records)
        return {
            "_viz_kind": "comment_diversity",
            **{k: metrics[k] for k in metrics},
            "_comment_embeddings": records_to_embedding_payload(records),
            "content_pool_with_embeddings": pool_with_emb,
        }
    except Exception as e:
        log_metric_error(
            metric_id,
            e,
            {"data_keys": list(data.keys()) if isinstance(data, dict) else None},
        )
        return empty
