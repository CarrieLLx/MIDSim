# -*- coding: utf-8 -*-
"""Shared embedding HTTP client and vector utilities for Weibo env."""
from __future__ import annotations

import json
import math
import os
from typing import List, Optional, Tuple

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


def get_embeddings(base_url: str, model_name: str, texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    if requests is None:
        raise RuntimeError("Install requests: pip install requests")
    url = f"{base_url}/embeddings"
    payload = {"model": model_name, "input": texts}
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if "data" not in data or not data["data"]:
        raise ValueError("API returned invalid format")
    items = sorted(data["data"], key=lambda x: x.get("index", 0))
    return [item["embedding"] for item in items]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
