"""Shared optional BM25 search over the current environment post pool."""
from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from dataclasses import dataclass
import heapq
import math
import re
import threading
from typing import Any, Dict, List, Optional, Set

from .midsim_params import BM25Params

_KEYWORD_STOPWORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "and", "or", "but", "not", "this", "that", "these", "those", "it", "its",
    '\u7684', '\u4e86', '\u5728', '\u662f', '\u548c', '\u4e0e', '\u6216', '\u53ca', '\u6709', '\u4e3a', '\u5bf9', '\u4e2d', '\u4e0a', '\u4e0b',
    '\u4e0d', '\u5417', '\u561b', '\u554a', '\u54e6', '\u55ef', '\u5427', '\u5440', '\u4e48', '\u5462', '\u7740', '\u8fc7', '\u8fd8', '\u5c31',
    '\u4e00\u4e2a', '\u53ef\u4ee5', '\u8fd9\u6837', '\u8fd9\u4e2a', '\u6211\u4eec', '\u4f60\u4eec', '\u4ed6\u4eec', '\u4ec0\u4e48', '\u600e\u4e48', '\u6ca1\u6709',
    '\u81ea\u5df1', '\u5982\u679c', '\u56e0\u4e3a', '\u6240\u4ee5', '\u4f46\u662f', '\u7136\u540e', '\u800c\u4e14', '\u6216\u8005',
}


def _tokenize_without_jieba(text: str) -> List[str]:
    """Latin/digit words plus overlapping CJK bigrams."""
    out: List[str] = []
    for run in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", text):
        if re.fullmatch(r"[a-z0-9_]+", run):
            out.append(run)
        elif len(run) == 1:
            out.append(run)
        else:
            out.extend(run[i : i + 2] for i in range(len(run) - 1))
    return out


def tokenize(text: str) -> List[str]:
    """Chinese word segmentation via jieba when available; else CJK bigrams."""
    text = (text or "").strip()
    if not text:
        return []
    try:
        import jieba

        return [w for w in jieba.lcut(text) if w.strip()]
    except ImportError:
        return _tokenize_without_jieba(text.lower())


def keyword_tokens(text: str) -> List[str]:
    """Filtered tokens, preserving repetitions for term-frequency scoring."""
    if not text or not str(text).strip():
        return []
    out: List[str] = []
    for w in tokenize(str(text).lower()):
        w = w.strip().lower()
        if len(w) < 2 or w in _KEYWORD_STOPWORDS:
            continue
        if not re.search(r"[0-9a-z_\u4e00-\u9fff]", w):
            continue
        out.append(w)
    return out



@dataclass(frozen=True)
class _BM25Index:
    lengths: Dict[str, int]
    postings: Dict[str, Dict[str, int]]
    average_length: float

    @classmethod
    def build(cls, documents: Dict[str, str]) -> "_BM25Index":
        lengths: Dict[str, int] = {}
        postings: Dict[str, Dict[str, int]] = defaultdict(dict)
        for post_id, text in documents.items():
            counts = Counter(keyword_tokens(text))
            lengths[post_id] = sum(counts.values())
            for term, count in counts.items():
                postings[term][post_id] = count
        average = sum(lengths.values()) / len(lengths) if lengths else 0.0
        return cls(lengths, dict(postings), average)

    def rank(self, terms: List[str], limit: int, params: BM25Params) -> List[str]:
        if not self.lengths or not self.average_length or limit <= 0:
            return []
        scores: Dict[str, float] = defaultdict(float)
        for term in dict.fromkeys(terms):
            hits = self.postings.get(term, {})
            if not hits:
                continue
            # Positive IDF (Lucene BM25 variant), even for very common terms.
            idf = math.log1p((len(self.lengths) - len(hits) + 0.5) / (len(hits) + 0.5))
            for post_id, tf in hits.items():
                norm = 1 - params.b + params.b * self.lengths[post_id] / self.average_length
                scores[post_id] += idf * tf * (params.k1 + 1) / (tf + params.k1 * norm)
        # Only matching posts; stable ties do not consume simulation randomness.
        return heapq.nsmallest(limit, scores, key=lambda pid: (-scores[pid], pid))


class BM25SearchEngine:
    def __init__(self) -> None:
        self._bm25_documents: Dict[str, str] = {}
        self._bm25_index: Optional[_BM25Index] = None
        self._bm25_lock = threading.Lock()

    @staticmethod
    def _post_text_for_bm25(post: Dict[str, Any]) -> str:
        """Index post text only; content/desc are aliases, not separate fields."""
        body = post.get("content") or post.get("desc") or ""
        tags = post.get("tags_list") or []
        tag_text = " ".join(str(tag) for tag in tags) if isinstance(tags, list) else str(tags)
        return f"{post.get('title') or ''}\n{body}\n{tag_text}".strip()

    def _bm25_sync(
        self,
        current_posts: Dict[str, Dict[str, Any]],
        limit: int,
        search_query: str,
        params: BM25Params,
    ) -> Dict[str, Dict[str, Any]]:
        terms = keyword_tokens(search_query)
        if not terms:
            return {}
        posts = {
            str(pid).strip(): post
            for pid, post in current_posts.items()
            if pid is not None and str(pid).strip() and isinstance(post, dict)
        }
        documents = {pid: self._post_text_for_bm25(post) for pid, post in posts.items()}
        # Compare text snapshots to catch in-place edits and pool replacement.
        # Comment/count updates do not affect the index. One index per Algorithm
        # agent is reused by all searching users; build only when BM25 is called.
        with self._bm25_lock:
            if self._bm25_index is None or documents != self._bm25_documents:
                index = _BM25Index.build(documents)
                self._bm25_documents = documents
                self._bm25_index = index
            index = self._bm25_index
        return {pid: posts[pid] for pid in index.rank(terms, limit, params)}

    async def bm25(
        self,
        current_posts: Dict[str, Dict[str, Any]],
        limit: int,
        *,
        current_timestamp: Optional[float] = None,
        search_query: str = "",
        embedding_config_path: Optional[str] = None,
        bm25_params: Optional[BM25Params] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Optional CPU-only BM25 search over the supplied current_posts pool.

        Select with exposure.search.types = ["BM25 Search"]. Parameters live
        under exposure.search.bm25: k1=1.2 (>=0), b=0.75 (0..1). Selection
        is opt-in; importing or constructing this engine builds no index.
        No training, embeddings, or random fallback.
        The cached index is shared by users of this Algorithm agent; each
        query still compares an O(total text length) snapshot for invalidation.
        """
        query = (search_query or "").strip()[:4000]
        if not current_posts or limit <= 0 or not query:
            return {}
        return await asyncio.to_thread(
            self._bm25_sync, current_posts, limit, query, bm25_params or BM25Params()
        )
