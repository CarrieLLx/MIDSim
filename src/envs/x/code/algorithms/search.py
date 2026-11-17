# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional, Tuple
import asyncio
import random

from loguru import logger
from onesim.utils.bm25 import BM25SearchEngine

from ..embedding import EmbeddingService, cosine_similarity
from ..utils.text import keyword_set


class SearchEngine(BM25SearchEngine):
    def __init__(self, agent: Any) -> None:
        super().__init__()
        self._agent = agent

    @staticmethod
    def _search_post_ids(current_posts: Dict[str, Dict[str, Any]]) -> List[str]:
        out: List[str] = []
        for nid, post in (current_posts or {}).items():
            if not isinstance(post, dict):
                continue
            sid = str(nid).strip()
            if sid:
                out.append(sid)
        return out

    @staticmethod
    def _post_text_for_search(post: Dict[str, Any], max_len: int = 1200) -> str:
        title = str(post.get("title", "") or "")
        desc = str(post.get("desc", "") or "")
        tags = post.get("tags_list")
        if isinstance(tags, list):
            tg = " ".join(str(t) for t in tags[:30])
        else:
            tg = str(tags or "")
        s = f"{title}\n{desc}\n{tg}".strip()
        return s if len(s) <= max_len else s[:max_len]

    async def random(
        self,
        current_posts: Dict[str, Dict[str, Any]],
        limit: int,
        *,
        current_timestamp: Optional[float] = None,
        search_query: str = "",
        embedding_config_path: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        if not current_posts or limit <= 0:
            return {}
        post_ids = list(current_posts.keys())
        random.shuffle(post_ids)
        picked = post_ids[:limit]
        return {nid: current_posts[nid] for nid in picked if nid in current_posts}

    async def keyword(
        self,
        current_posts: Dict[str, Dict[str, Any]],
        limit: int,
        *,
        current_timestamp: Optional[float] = None,
        search_query: str = "",
        embedding_config_path: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        if not current_posts or limit <= 0:
            return {}
        cands = self._search_post_ids(current_posts)
        if not cands:
            return {}

        query = (search_query or "").strip()[:4000]
        if not query:
            sub_posts = {nid: current_posts[nid] for nid in cands if nid in current_posts}
            return await self.random(sub_posts, limit)

        qt = keyword_set(query)
        scored: List[Tuple[str, float]] = []
        for nid in cands:
            post = current_posts.get(nid)
            if not isinstance(post, dict):
                continue
            text = self._post_text_for_search(post, max_len=2000)
            nt = keyword_set(text)
            if not qt or not nt:
                scored.append((nid, 0.0))
                continue
            inter = len(qt & nt)
            union = len(qt | nt)
            j = (inter / union) if union else 0.0
            scored.append((nid, j))
        random.shuffle(scored)
        scored.sort(key=lambda x: x[1], reverse=True)
        return {
            nid: current_posts[nid]
            for nid, _ in scored[:limit]
            if nid in current_posts
        }

    def _relevant_sync(
        self,
        current_posts: Dict[str, Dict[str, Any]],
        limit: int,
        *,
        search_query: str,
        embedding_config_path: str,
    ) -> Dict[str, Dict[str, Any]]:
        cands = self._search_post_ids(current_posts)
        if not cands:
            return {}
        query = (search_query or "").strip()[:4000]
        texts = [query[:2000]]
        for nid in cands:
            post = current_posts.get(nid)
            texts.append(
                self._post_text_for_search(post, max_len=800)
                if isinstance(post, dict)
                else ""
            )
        vecs = EmbeddingService(embedding_config_path).embed(texts)
        if not vecs or len(vecs) != len(texts):
            raise ValueError("embedding return length does not match input")
        vq = vecs[0]
        scored: List[Tuple[str, float]] = []
        for i, nid in enumerate(cands):
            sim = float(cosine_similarity(vq, vecs[i + 1]))
            scored.append((nid, sim))
        random.shuffle(scored)
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:limit]
        return {nid: current_posts[nid] for nid, _ in top if nid in current_posts}

    async def relevant(
        self,
        current_posts: Dict[str, Dict[str, Any]],
        limit: int,
        *,
        current_timestamp: Optional[float] = None,
        search_query: str = "",
        embedding_config_path: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        if not current_posts or limit <= 0:
            return {}
        cands = self._search_post_ids(current_posts)
        if not cands:
            return {}
        query = (search_query or "").strip()[:4000]
        if not query:
            sub_posts = {nid: current_posts[nid] for nid in cands if nid in current_posts}
            return await self.random(sub_posts, limit)
        try:
            return await asyncio.to_thread(
                self._relevant_sync,
                current_posts,
                limit,
                search_query=query,
                embedding_config_path=embedding_config_path or "",
            )
        except Exception as e:
            logger.warning(f"Relevant search embedding failed, fallback to keyword: {e}")
            return await self.keyword(
                current_posts,
                limit,
                search_query=query,
            )
