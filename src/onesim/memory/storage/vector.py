from abc import ABC, abstractmethod
import asyncio
import numpy as np
import faiss  
from .storage import MemoryStorage
from onesim.models import ModelManager
from loguru import logger

class VectorMemoryStorage(MemoryStorage):
    def __init__(self, config):
        self.config = config
        model_config_name = config.get("model_config_name")
        self._embedding_config_name = model_config_name
        model_manager = ModelManager.get_instance()
        self.embedding_model = model_manager.get_model(
            config_name=model_config_name, model_type="embedding"
        )
        self.memory_items = []
        self.embeddings = []
        self.index = None  
        self.batch_size = config.get("batch_size", 10)  
        self.pending_updates = 0  
        self.index_dimension = None  
        self.max_index_size = config.get("max_index_size", 10000)  

    def _embedding_model_log_tag(self) -> str:
        """Log prefix, e.g. [vLLM] vllm-embedding-bert (consistent with chat side [vLLM] vllm-qwen-9889)"""
        m = self.embedding_model
        cfg = getattr(m, "config_name", None) or self._embedding_config_name or "unknown"
        prov = getattr(m, "provider", None)
        if prov:
            p = str(prov).lower()
            label = "vLLM" if p == "vllm" else str(prov).capitalize()
        else:
            cls = type(m).__name__
            if "VLLM" in cls:
                label = "vLLM"
            elif "OpenAI" in cls:
                label = "OpenAI"
            elif "LoadBalancer" in cls:
                label = "LoadBalancer"
            else:
                label = cls.replace("EmbeddingAdapter", "").replace("Adapter", "") or "Embedding"
        return f"[{label}] {cfg}"

    @staticmethod
    def _coerce_embedding_to_1d(embedding) -> np.ndarray:
        """
        Coerce various embedding outputs to 1D float32 (FAISS requires single row dimension d in (n, d))
        """
        if embedding is None:
            raise ValueError("embedding is None")
        arr = np.asarray(embedding, dtype=np.float32)
        if arr.ndim == 2:
            if arr.shape[0] == 1:
                arr = arr.reshape(-1)
            elif arr.shape[1] == 1:
                arr = arr.reshape(-1)
            else:
                logger.warning(
                    f"Embedding matrix shape {arr.shape}; using first row as single vector"
                )
                arr = np.asarray(arr[0], dtype=np.float32).reshape(-1)
        elif arr.ndim > 1:
            arr = arr.reshape(-1)
        if arr.ndim != 1:
            raise ValueError(f"Cannot coerce embedding to 1d, got shape {arr.shape}")
        return np.ascontiguousarray(arr, dtype=np.float32)

    async def add(self, memory_item):
        """Add memory item and generate embedding"""
        try:
            # Check if there is already an embedding vector
            if not hasattr(memory_item, 'embedding') or memory_item.embedding is None:
                try:
                    logger.debug(
                        f"{self._embedding_model_log_tag()} Computing embedding for new memory item: {memory_item.id}"
                    )
                    embedding_result = await self.embedding_model.acall(memory_item.content)
                    embedding = self._coerce_embedding_to_1d(embedding_result.embedding)
                    memory_item.embedding = embedding
                except Exception as e:
                    logger.error(f"Error computing embedding for item {memory_item.id}: {e}")
                    # Create a zero vector as fallback, avoid complete failure
                    if self.embeddings and self.index_dimension:
                        embedding = np.zeros(self.index_dimension, dtype=np.float32)
                        memory_item.embedding = embedding
                    else:
                        raise ValueError(f"Cannot create fallback embedding: no index dimension determined yet")
            else:
                embedding = self._coerce_embedding_to_1d(memory_item.embedding)

            # Determine dimension and initialize index
            if self.index is None and embedding is not None:
                self.index_dimension = int(embedding.shape[0])
                self._initialize_index()

            # Add to memory and embedding list
            self.memory_items.append(memory_item)
            self.embeddings.append(embedding)

            # Increase pending updates count
            self.pending_updates += 1

            # Check if index needs to be updated
            if self.pending_updates >= self.batch_size:
                await self._update_index_batch()

            return memory_item.id
        except Exception as e:
            logger.error(f"Error adding item to vector storage: {e}")
            raise

    async def get_all(self):
        return self.memory_items.copy()

    async def delete(self, memory_item):
        try:
            idx = -1
            # Find by ID or object
            if hasattr(memory_item, 'id'):
                for i, item in enumerate(self.memory_items):
                    if item.id == memory_item.id:
                        idx = i
                        break
            else:
                idx = self.memory_items.index(memory_item)

            if idx >= 0:
                self.memory_items.pop(idx)
                self.embeddings.pop(idx)
                # Deletion operation also needs to update index
                self.pending_updates += 1

                # Check if index needs to be updated
                if self.pending_updates >= self.batch_size:
                    await self._update_index_batch()
            else:
                logger.warning(f"Memory item not found for deletion: {memory_item}")
        except Exception as e:
            logger.error(f"Error deleting item from vector storage: {e}")
            raise

    async def update(self, memory_item):
        """Update memory item, only recalculate embedding when content changes"""
        try:
            idx = -1
            if hasattr(memory_item, 'id'):
                for i, item in enumerate(self.memory_items):
                    if item.id == memory_item.id:
                        idx = i
                        break

            if idx >= 0:
                old_item = self.memory_items[idx]
                content_changed = old_item.content != memory_item.content

                # Update memory item
                self.memory_items[idx] = memory_item

                # Only recalculate embedding when content changes or embedding is missing
                if content_changed or not hasattr(memory_item, 'embedding') or memory_item.embedding is None:
                    logger.debug(
                        f"{self._embedding_model_log_tag()} Content changed or embedding missing, "
                        f"recalculating embedding for item {memory_item.id}"
                    )
                    embedding_result = await self.embedding_model.acall(memory_item.content)
                    embedding = self._coerce_embedding_to_1d(embedding_result.embedding)
                    memory_item.embedding = embedding
                    self.embeddings[idx] = embedding

                    # Mark need to update index
                    self.pending_updates += 1
                elif hasattr(old_item, 'embedding') and old_item.embedding is not None:
                    # If content didn't change and old item has embedding, keep old embedding
                    coerced = self._coerce_embedding_to_1d(old_item.embedding)
                    memory_item.embedding = coerced
                    self.embeddings[idx] = coerced
                else:
                    # Use new provided embedding
                    coerced = self._coerce_embedding_to_1d(memory_item.embedding)
                    memory_item.embedding = coerced
                    self.embeddings[idx] = coerced

                # Check if index needs to be updated
                if self.pending_updates >= self.batch_size:
                    await self._update_index_batch()
            else:
                logger.warning(f"Memory item not found for update: {memory_item}")
        except Exception as e:
            logger.error(f"Error updating item in vector storage: {e}")
            raise

    def _initialize_index(self):
        """Initialize FAISS index"""
        try:
            if self.index_dimension is None:
                logger.warning("Cannot initialize index: dimension not yet determined")
                return

            self.index = faiss.IndexFlatL2(self.index_dimension)
            logger.info(f"Initialized FAISS index with dimension {self.index_dimension}")
        except Exception as e:
            logger.error(f"Error initializing FAISS index: {e}")
            raise

    async def _update_index_batch(self):
        """Batch update index instead of rebuilding each time"""
        try:
            if not self.embeddings:
                logger.debug("No embeddings to update index")
                self.index = None
                self.pending_updates = 0
                return

            if self.index is None and len(self.embeddings) > 0:
                # If index doesn't exist but has embeddings, initialize index
                first = self._coerce_embedding_to_1d(self.embeddings[0])
                self.index_dimension = int(first.shape[0])
                self._initialize_index()

            # Completely rebuild index (coerce each embedding to 1D, compatible with historical batch lists)
            rows = [self._coerce_embedding_to_1d(e) for e in self.embeddings]
            embeddings_array = np.stack(rows, axis=0).astype(np.float32)
            if self.index is not None:
                self.index.reset()  # Clear index
                self.index.add(embeddings_array)  # Add all vectors

            self.pending_updates = 0  # Reset pending updates count
            logger.debug(f"Updated FAISS index with {len(self.embeddings)} vectors")
        except Exception as e:
            logger.error(f"Error updating FAISS index: {e}")
            raise

    async def query(self, query, top_k=5):
        """Query most similar memory items"""
        try:
            # If no data or index not initialized, return empty list
            if not self.memory_items or self.index is None:
                return []

            # Ensure index is up to date
            if self.pending_updates > 0:
                await self._update_index_batch()

            # Process query
            if query is None:
                return self.memory_items[:min(top_k, len(self.memory_items))]

            # Convert query to string
            if isinstance(query, list):
                query_string = ".".join(query)
            else:
                query_string = str(query)

            # Get query embedding
            try:
                embedding_res = await self.embedding_model.acall(query_string[:500])
                q1 = self._coerce_embedding_to_1d(embedding_res.embedding)
                query_vector = q1.reshape(1, -1).astype(np.float32)

                # Perform search
                distances, indices = self.index.search(query_vector, min(top_k, len(self.memory_items)))

                # Collect results
                retrieved_items = []
                for idx in indices[0]:
                    if 0 <= idx < len(self.memory_items):  # Ensure index is valid
                        retrieved_items.append(self.memory_items[idx])

                return retrieved_items
            except Exception as e:
                logger.error(f"Error during vector search: {e}")
                return self.memory_items[:min(top_k, len(self.memory_items))]
        except Exception as e:
            logger.error(f"Error in query operation: {e}")
            return []

    async def get_size(self):
        return len(self.memory_items)

    async def clear(self):
        """Clear storage"""
        self.memory_items = []
        self.embeddings = []
        if self.index is not None:
            self.index.reset()
        self.pending_updates = 0

    async def batch_add(self, memory_items):
        """
        Batch add multiple memory items, efficiently calculate embeddings
        """
        added_ids = []

        if not memory_items:
            logger.debug("No items to add in batch operation")
            return added_ids

        try:
            # Collect items to calculate embeddings
            items_to_embed = []

            for item in memory_items:
                if not hasattr(item, 'embedding') or item.embedding is None:
                    items_to_embed.append(item)

            # Batch calculate embeddings
            if items_to_embed:
                logger.info(
                    f"{self._embedding_model_log_tag()} Computing embeddings for {len(items_to_embed)} items in batch"
                )
                contents = [item.content for item in items_to_embed]

                # To avoid resource issues with large batches, we limit concurrent requests
                max_concurrent = min(len(contents), 10)  # Concurrent requests limited to 10
                semaphore = asyncio.Semaphore(max_concurrent)

                async def get_embedding_with_semaphore(content, item_idx):
                    try:
                        async with semaphore:
                            result = await self.embedding_model.acall(content)
                            return result, item_idx
                    except Exception as e:
                        logger.error(f"Error computing embedding for item {items_to_embed[item_idx].id}: {e}")
                        return None, item_idx

                # Concurrent execution, but control concurrent requests
                batch_tasks = [get_embedding_with_semaphore(content, i) for i, content in enumerate(contents)]
                batch_results = await asyncio.gather(*batch_tasks)

                # Process results, including error handling
                for result, idx in batch_results:
                    if result is not None:
                        items_to_embed[idx].embedding = self._coerce_embedding_to_1d(
                            result.embedding
                        )
                    elif self.embeddings and self.index_dimension:
                        # If failed, use zero vector fallback
                        items_to_embed[idx].embedding = np.zeros(self.index_dimension, dtype=np.float32)
                        logger.warning(f"Using zero vector fallback for item {items_to_embed[idx].id}")

            # Add all items
            for item in memory_items:
                try:
                    if hasattr(item, 'embedding') and item.embedding is not None:
                        # If already has embedding, add directly
                        embedding = self._coerce_embedding_to_1d(item.embedding)
                        item.embedding = embedding

                        # Initialize index (if needed)
                        if self.index is None and embedding is not None:
                            self.index_dimension = int(embedding.shape[0])
                            self._initialize_index()

                        self.memory_items.append(item)
                        self.embeddings.append(embedding)
                        added_ids.append(item.id)
                        self.pending_updates += 1
                    else:
                        logger.warning(f"Skipping item {item.id} with no embedding")
                except Exception as e:
                    logger.error(f"Error adding individual item in batch: {e}")
                    # Continue processing other items, rather than complete failure
                    continue

            # Update index
            if self.pending_updates > 0:
                await self._update_index_batch()

            return added_ids
        except Exception as e:
            logger.error(f"Error in batch_add operation: {e}")
            # Return successfully added IDs, rather than complete failure
            return added_ids
