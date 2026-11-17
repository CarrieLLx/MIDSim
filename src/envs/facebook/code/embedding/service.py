# -*- coding: utf-8 -*-
"""Embedding HTTP client, vector helpers, and text-similarity pipeline."""
from __future__ import annotations

import json
import hashlib
import sqlite3
from contextlib import closing
import math
import os
from pathlib import Path
from functools import lru_cache
from typing import List, Optional, Tuple

from loguru import logger

try:
    import requests
except ImportError:
    requests = None


def find_project_root(start: Optional[str] = None) -> str:
    """Walk up from *start* (or this file) until config/model_config.json is found."""
    path = os.path.abspath(start or os.path.dirname(__file__))
    for _ in range(12):
        cfg = os.path.join(path, "config", "model_config.json")
        if os.path.isfile(cfg):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return os.getcwd()


def default_embedding_config_path() -> str:
    return os.path.join(find_project_root(), "config", "model_config.json")


def load_embedding_config(config_path: str) -> Tuple[str, str]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    embeddings = config.get("embedding", [])
    if not embeddings:
        raise ValueError("model_config.json not found embedding config")
    emb = embeddings[0]
    base_url = emb.get("client_args", {}).get("base_url", "").rstrip("/")
    model_name = emb.get("model_name", "bge-base-en-v1.5")
    return base_url, model_name



@lru_cache(maxsize=4)
def _embedding_tokenizer(model_name: str):
    """Use the deployed model's local tokenizer, including special tokens."""
    from tokenizers import Tokenizer
    name = str(model_name).replace("\\", "/").rstrip("/").split("/")[-1]
    candidates = [Path(model_name)] + [p / "models" / name for p in Path(__file__).resolve().parents]
    folder = next((p for p in candidates if (p / "tokenizer.json").is_file()), None)
    if folder is None:
        raise ValueError(f"No local tokenizer for embedding model {model_name!r}; cannot enforce token limit")
    cfg = json.loads((folder / "tokenizer_config.json").read_text(encoding="utf-8"))
    model_cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    limits = [int(x) for x in (cfg.get("model_max_length"), model_cfg.get("max_position_embeddings"))
              if x is not None and 0 < int(x) < 1000000]
    if not limits:
        raise ValueError(f"No finite embedding token limit for {model_name!r}")
    tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer, min(limits)


def _bounded_embedding_texts(model_name: str, texts: List[str]) -> List[str]:
    tokenizer, limit = _embedding_tokenizer(model_name)
    bounded = []
    for index, text in enumerate(texts):
        count = len(tokenizer.encode(text, add_special_tokens=True).ids)
        if count <= limit:
            bounded.append(text)
            continue
        # Preserve the original string prefix (including a query instruction).
        # Re-tokenize each prefix: character counts are not token counts.
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if len(tokenizer.encode(text[:mid], add_special_tokens=True).ids) <= limit:
                lo = mid
            else:
                hi = mid - 1
        clipped = text[:lo]
        after = len(tokenizer.encode(clipped, add_special_tokens=True).ids)
        if after > limit:
            raise ValueError("Embedding truncation failed to satisfy token limit")
        logger.warning("Embedding input truncated: model={} item={} tokens={} -> {} limit={}",
                       model_name, index, count, after, limit)
        bounded.append(clipped)
    return bounded


def _request_embeddings(base_url: str, model_name: str, texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    if requests is None:
        raise RuntimeError("Install requests: pip install requests")
    url = f"{base_url}/embeddings"
    payload = {"model": model_name, "input": _bounded_embedding_texts(model_name, texts)}
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if "data" not in data or not data["data"]:
        raise ValueError("API returned invalid format")
    items = sorted(data["data"], key=lambda x: x.get("index", 0))
    return [item["embedding"] for item in items]



@lru_cache(maxsize=8)
def _vector_cache_namespace(base_url, model_name):
    tokenizer, limit = _embedding_tokenizer(model_name)
    identity = [base_url, model_name, tokenizer.to_str(), limit,
                "prefix-token-limit-v1", os.environ.get("ONESIM_EMBEDDING_CACHE_REVISION", "1")]
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()


def get_embeddings(base_url: str, model_name: str, texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    namespace = _vector_cache_namespace(base_url, model_name)
    # Keep full original text in the key: a changed profile must be re-evaluated.
    keys = [hashlib.sha256((namespace + "\0" + t).encode()).hexdigest() for t in texts]
    folder = Path(os.environ.get("ONESIM_EMBEDDING_CACHE_DIR") or
                  str(Path(find_project_root()) / "temp" / "embedding_cache"))
    folder.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(str(folder / "vectors.sqlite3"), timeout=60)) as db:
        db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vector TEXT NOT NULL)")
        found = {}
        for key in set(keys):
            row = db.execute("SELECT vector FROM vectors WHERE key=?", (key,)).fetchone()
            if row:
                found[key] = json.loads(row[0])
        missing = dict((k,t) for k,t in zip(keys,texts) if k not in found)
        if missing:
            # Do not hold a database write transaction while waiting on the API.
            vectors = _request_embeddings(base_url, model_name, list(missing.values()))
            if len(vectors) != len(missing):
                raise ValueError("Embedding response count differs from input count")
            dims = {len(v) for v in vectors}
            if len(dims) != 1 or 0 in dims or any(not math.isfinite(float(x)) for v in vectors for x in v):
                raise ValueError("Invalid embedding vectors; not cached")
            found.update(zip(missing,vectors))
            db.executemany("INSERT OR REPLACE INTO vectors VALUES (?,?)",
                           [(k,json.dumps(found[k])) for k in missing])
            db.commit()
            logger.info("Embedding cache: hits={} missing={} unique={}", len(set(keys))-len(missing), len(missing), len(set(keys)))
        return [found[k] for k in keys]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _chunk_text(text: str, max_chars: int) -> List[str]:
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


class EmbeddingService:
    """Load embedding config once, then embed texts or score text pairs."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        path = str(config_path or "").strip()
        self.config_path = path or default_embedding_config_path()
        self._base_url: Optional[str] = None
        self._model_name: Optional[str] = None

    def _client(self) -> Tuple[str, str]:
        if self._base_url is None or self._model_name is None:
            self._base_url, self._model_name = load_embedding_config(self.config_path)
        return self._base_url, self._model_name

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        base_url, model_name = self._client()
        return get_embeddings(base_url, model_name, texts)

    def embed_chunks(
        self,
        text: str,
        *,
        max_chars: int = 400,
        max_chunks: int = 12,
    ) -> List[List[float]]:
        max_chars = max(64, min(int(max_chars), 8000))
        max_chunks = max(1, min(int(max_chunks), 64))
        chunks = _chunk_text(text or "", max_chars)[:max_chunks]
        if not chunks:
            return []
        vecs: List[List[float]] = []
        base_url, model_name = self._client()
        for ch in chunks:
            batch = get_embeddings(base_url, model_name, [ch])
            if len(batch) != 1:
                raise ValueError("embedding API returns an abnormal number of chunks")
            vecs.append(batch[0])
        return vecs

    def similarity(
        self,
        text_a: str,
        text_b: str,
        *,
        max_chars: int = 400,
        max_chunks: int = 12,
        aggregation: str = "mean",
    ) -> Optional[float]:
        agg = aggregation if aggregation in ("mean", "max") else "mean"
        try:
            va = self.embed_chunks(text_a, max_chars=max_chars, max_chunks=max_chunks)
            vb = self.embed_chunks(text_b, max_chars=max_chars, max_chunks=max_chunks)
            if not va or not vb:
                return None
            if agg == "max":
                best = -1.0
                for a in va:
                    for b in vb:
                        best = max(best, float(cosine_similarity(a, b)))
                return best
            return float(cosine_similarity(_mean_embedding(va), _mean_embedding(vb)))
        except Exception as e:
            detail = ""
            resp = getattr(e, "response", None)
            if resp is not None:
                try:
                    detail = f" | response={resp.text[:400]!r}"
                except Exception:
                    detail = f" | status={getattr(resp, 'status_code', '')}"
            logger.warning(f"embedding similarity failed (treated as no similarity): {e}{detail}")
            return None


def embed_texts(texts: List[str], *, config_path: Optional[str] = None) -> List[List[float]]:
    return EmbeddingService(config_path).embed(texts)


def embed_chunks(
    text: str,
    *,
    config_path: Optional[str] = None,
    max_chars: int = 400,
    max_chunks: int = 12,
) -> List[List[float]]:
    return EmbeddingService(config_path).embed_chunks(
        text, max_chars=max_chars, max_chunks=max_chunks
    )


def text_similarity(
    text_a: str,
    text_b: str,
    *,
    config_path: Optional[str] = None,
    max_chars: int = 400,
    max_chunks: int = 12,
    aggregation: str = "mean",
) -> Optional[float]:
    return EmbeddingService(config_path).similarity(
        text_a,
        text_b,
        max_chars=max_chars,
        max_chunks=max_chunks,
        aggregation=aggregation,
    )
