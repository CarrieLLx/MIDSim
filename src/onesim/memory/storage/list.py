from abc import ABC, abstractmethod
from .storage import MemoryStorage
from loguru import logger

class ListMemoryStorage(MemoryStorage):
    def __init__(self, config):
        self.config = config
        # Validate capacity parameters
        try:
            self.capacity = int(config.get('capacity', 100))
            if self.capacity <= 0:
                logger.warning(f"Invalid capacity value: {self.capacity}, using default 100")
                self.capacity = 100
        except (ValueError, TypeError):
            logger.warning(f"Invalid capacity format in config, using default 100")
            self.capacity = 100
            
        self.memory_list = []
        self.eviction_policy = config.get('eviction_policy', 'fifo')  # 'fifo', 'lru', 'importance'

    async def add(self, memory_item):
        # Check capacity and clear old memories if necessary
        if len(self.memory_list) >= self.capacity:
            await self._evict_memory()
            
        self.memory_list.append(memory_item)
        return memory_item.id  # Return added item ID for tracking

    async def get_all(self):
        return self.memory_list.copy()

    async def delete(self, memory_item):
        try:
            self.memory_list.remove(memory_item)
        except ValueError:
            # If deleted by ID
            if hasattr(memory_item, 'id'):
                for idx, item in enumerate(self.memory_list):
                    if item.id == memory_item.id:
                        del self.memory_list[idx]
                        return
            logger.warning(f"Memory item not found for deletion: {memory_item}")

    async def query(self, query=None, top_k=None):
        try:
            if query is None:
                result = self.memory_list
            else:
                if callable(query):
                    # If query is a function, use as a filter
                    result = [item for item in self.memory_list if query(item)]
                elif isinstance(query, list) and all(callable(q) for q in query):
                    # If query is a list of callable objects, apply all filters
                    result = self.memory_list
                    for condition in query:
                        result = [item for item in result if condition(item)]
                else:
                    # By default, return all content
                    result = self.memory_list
            
            # Apply limit
            if top_k is not None and top_k > 0:
                return result[:min(top_k, len(result))]
            return result
        except Exception as e:
            logger.error(f"Error during query operation: {e}")
            return []
    
    async def get_size(self):
        return len(self.memory_list)
        
    async def clear(self):
        """Clear all memory items"""
        self.memory_list.clear()
        
    async def merge(self):
        """Stub implementation of merge function - should be overridden in actual application"""
        logger.warning("Default merge operation called - no action taken")
        return self.memory_list.copy()
        
    async def forget(self, criteria):
        """Forget certain memories based on criteria"""
        try:
            if callable(criteria):
                # Delete items that satisfy the criteria
                self.memory_list = [item for item in self.memory_list if not criteria(item)]
            else:
                logger.warning(f"Invalid criteria for forget operation: {criteria}")
        except Exception as e:
            logger.error(f"Error during forget operation: {e}")
            
    async def batch_add(self, memory_items):
        """Batch add multiple memory items"""
        added_ids = []
        for item in memory_items:
            item_id = await self.add(item)
            added_ids.append(item_id)
        return added_ids
        
    async def _evict_memory(self):
        """Remove memory based on eviction policy"""
        if not self.memory_list:
            return
            
        if self.eviction_policy == 'fifo':
            # FIFO policy
            self.memory_list.pop(0)
        elif self.eviction_policy == 'lru':
            # LRU policy - need to track access time
            # Here we simply use timestamp as an approximation
            self.memory_list.sort(key=lambda x: x.timestamp)
            self.memory_list.pop(0)
        elif self.eviction_policy == 'importance':
            # Remove least important based on importance
            self.memory_list.sort(key=lambda x: x.attributes.get('importance', 0))
            self.memory_list.pop(0)
        else:
            # Default to FIFO
            self.memory_list.pop(0)