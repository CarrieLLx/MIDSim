from abc import ABC, abstractmethod
import time
import numpy as np
from datetime import datetime
from loguru import logger
from onesim.models import ModelManager
from onesim.models.core.message import Message
from onesim.models.parsers import TagParser

class MemoryMetric(ABC):
    def __init__(self, config):
        pass

    @abstractmethod
    async def calculate(self, memory_item, query=None):
        pass


class ImportanceMetric(MemoryMetric):
    def __init__(self, config):
        model_config_name = config.get("model_config_name")
        model_manager = ModelManager.get_instance()
        self.llm_model = model_manager.get_model(
            model_config_name,
        )  # LLM model instance
        # self.parser = TagParser(
        #     tag_begin="[SCORE]",
        #     content_hint="the importance score",
        #     tag_end="[/SCORE]"
        # )
        self.parser = TagParser(
            tag_start="[SCORE]",
            tag_end="[/SCORE]",
            content_hint="the importance score (a number from 1 to 10)",
        )
        # Add cache to reduce LLM calls
        self.cache = {}
        
  
    async def calculate(self, memory_item, query=None):
        # 检查缓存
        if memory_item.id in self.cache:
            return self.cache[memory_item.id]
            
        if 'importance' in memory_item.attributes and memory_item.attributes['importance'] is not None:
            return memory_item.attributes['importance']
            
        # Use LLM to compute importance
        prompt=f"Evaluate the importance of this memory based on its relevance, context, and potential impact. Provide a score from 1 to 10, where 1 is the least important and 10 is the most important.\nMemory Content: {memory_item.content}\n"+self.parser.format_instruction
        prompt = self.llm_model.format(
            Message("user",prompt, role="user")
        )
        try:
            model_name = getattr(self.llm_model, "config_name", type(self.llm_model).__name__)
            logger.debug(
                f"[ImportanceMetric] Call LLM to evaluate importance memory_id={memory_item.id} "
                f"model={model_name}"
            )
            response = await self.llm_model.acall(prompt)
            res = self.parser.parse(response)
            # importance = float(res.parsed['score'])
            raw = res.parsed
            if isinstance(raw, dict):
                score_val = raw.get("score")
                importance = float(str(score_val).strip()) if score_val is not None else 5.0
            else:
                importance = float(str(raw).strip())
            logger.info(
                f"[ImportanceMetric] memory_id={memory_item.id} importance={importance} "
                f"raw_parsed={raw!r}"
            )
            # Save to cache
            self.cache[memory_item.id] = importance
            return importance
        except Exception as e:
            error_msg = f"Error parsing importance score: {e}\nResponse: {response if 'response' in locals() else 'No response'}"
            importance = 5.0
            return importance

class RecencyMetric(MemoryMetric):
    @staticmethod
    async def calculate(memory_item, query=None):
        # Calculate recency based on timestamp
        try:
            if isinstance(memory_item.timestamp, str):
                dt = datetime.strptime(memory_item.timestamp, "%Y-%m-%d %H:%M:%S")
                memory_timestamp = dt.timestamp()
            else:
                # If timestamp is already float or int, use directly
                memory_timestamp = float(memory_item.timestamp)

            recency = 1 / (time.time() - memory_timestamp + 1)
            logger.debug(
                f"[RecencyMetric] memory_id={getattr(memory_item, 'id', None)} "
                f"recency={recency:.6f}"
            )
            return recency
        except Exception:
            return 0.1

class RelevanceMetric(MemoryMetric):
    def __init__(self, config):
        model_config_name = config.get("model_config_name")
        model_manager = ModelManager.get_instance()
        self.embedding_model = model_manager.get_model(
            model_config_name,
        )
        # Add cache to reduce repeated calculations
        self.embedding_cache = {}
           
    async def calculate(self, memory_item, query=None):
        if query is None or self.embedding_model is None:
            return 1.0
            
        # Use cache to reduce repeated calculations
        cache_key = f"{memory_item.id}:{query}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
            
        try:
            emb_model = getattr(
                self.embedding_model, "config_name", type(self.embedding_model).__name__
            )
            mid = getattr(memory_item, "id", None)
            # Check if memory_item already has embedding vector
            if hasattr(memory_item, 'embedding') and memory_item.embedding is not None:
                memory_embedding = memory_item.embedding
            else:
                # Calculate and cache
                logger.debug(
                    f"[RelevanceMetric] acall embed memory memory_id={mid} model={emb_model} "
                    f"content_len={len(str(memory_item.content or ''))}"
                )
                embedding_result = await self.embedding_model.acall(memory_item.content)
                memory_embedding = embedding_result.embedding
                memory_item.embedding = memory_embedding
                mem_dim = len(memory_embedding) if memory_embedding is not None else 0
                logger.debug(
                    f"[RelevanceMetric] memory embedding ok memory_id={mid} dim={mem_dim}"
                )

            q_preview_len = len(str(query))
            logger.debug(
                f"[RelevanceMetric] acall embed query memory_id={mid} model={emb_model} "
                f"query_len={q_preview_len}"
            )
            query_embedding = await self.embedding_model.acall(query)
            query_embedding = query_embedding.embedding
            qdim = len(query_embedding) if query_embedding is not None else 0
            logger.debug(
                f"[RelevanceMetric] query embedding ok memory_id={mid} dim={qdim}"
            )

            similarity = self.cosine_similarity(memory_embedding, query_embedding)
            # Save to cache
            self.embedding_cache[cache_key] = similarity
            return similarity
        except Exception as e:
            return 0.5
    
    def cosine_similarity(self, vec1, vec2):
        try:
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        except:
            return 0.0
