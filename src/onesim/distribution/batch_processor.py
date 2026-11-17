import asyncio
import time
from typing import List, Dict, Any
from loguru import logger

class BatchProcessor:
    """
    Batching utility that reduces network overhead and improves throughput.
    Useful in highly concurrent multi-agent simulations to combine event and decision-record delivery requests.
    """
    def __init__(self, batch_size=50, max_wait_time=0.1):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.storage_events = []
        self.decision_records = []
        self.last_flush_time = time.time()
        self.flush_lock = asyncio.Lock()
        self.master_address = None
        self.master_port = None
        self._task = None
        self._stopped = False
    
    def start(self, master_address, master_port):
        """Start the batch processor and its background flush task."""
        if self._task is not None:
            return
            
        self.master_address = master_address
        self.master_port = master_port
        self._stopped = False
        self._task = asyncio.create_task(self._periodic_flush())
        logger.info(f"Started batch processor for {master_address}:{master_port}")
    
    def stop(self):
        """Stop the batch processor and flush all remaining items."""
        self._stopped = True
        if self._task:
            self._task.cancel()
            self._task = None
        
        # Final flush.
        asyncio.create_task(self.flush())
    
    async def _periodic_flush(self):
        """Background task that periodically flushes batches."""
        try:
            while not self._stopped:
                now = time.time()
                if now - self.last_flush_time >= self.max_wait_time:
                    await self.flush()
                await asyncio.sleep(0.05)  # Sleep briefly to avoid busy waiting.
        except asyncio.CancelledError:
            logger.debug("Batch processor task cancelled")
        except Exception as e:
            logger.error(f"Error in batch processor: {e}")
    
    async def add_storage_event(self, event_data):
        """Add a storage event to the batch."""
        # Avoid circular import by importing here
        from onesim.distribution.grpc_impl import send_storage_event_to_master
        
        if not self.master_address:
            logger.warning("Batch processor not started yet")
            # Send immediately if the processor has not started.
            return await send_storage_event_to_master(
                self.master_address, 
                self.master_port, 
                event_data
            )
        
        async with self.flush_lock:
            self.storage_events.append(event_data)
        
        # Flush when the batch size is reached.
        if len(self.storage_events) >= self.batch_size:
            await self.flush_storage_events()
            
        return True
    
    async def add_decision_record(self, decision_data):
        """Add a decision record to the batch."""
        # Avoid circular import by importing here
        from onesim.distribution.grpc_impl import send_decision_record_to_master
        
        if not self.master_address:
            logger.warning("Batch processor not started yet")
            # Send immediately if the processor has not started.
            return await send_decision_record_to_master(
                self.master_address, 
                self.master_port, 
                decision_data
            )
        
        async with self.flush_lock:
            self.decision_records.append(decision_data)
        
        # Flush when the batch size is reached.
        if len(self.decision_records) >= self.batch_size:
            await self.flush_decision_records()
            
        return True
    
    async def flush_storage_events(self):
        """Flush all pending storage events."""
        # Avoid circular import by importing here
        from onesim.distribution.grpc_impl import send_storage_event_batch_to_master
        
        async with self.flush_lock:
            if not self.storage_events:
                return True
                
            events_to_send = self.storage_events
            self.storage_events = []
        
        success = await send_storage_event_batch_to_master(
            self.master_address,
            self.master_port,
            events_to_send
        )
        
        return success
    
    async def flush_decision_records(self):
        """Flush all pending decision records."""
        # Avoid circular import by importing here
        from onesim.distribution.grpc_impl import send_decision_record_batch_to_master
        
        async with self.flush_lock:
            if not self.decision_records:
                return True
                
            records_to_send = self.decision_records
            self.decision_records = []
        
        success = await send_decision_record_batch_to_master(
            self.master_address,
            self.master_port,
            records_to_send
        )
        
        return success
    
    async def flush(self):
        """Flush all pending items."""
        results = []
        if self.storage_events:
            results.append(await self.flush_storage_events())
        if self.decision_records:
            results.append(await self.flush_decision_records())
        
        self.last_flush_time = time.time()
        return all(results) if results else True

# Create the global batch processor instance.
batch_processor = BatchProcessor() 