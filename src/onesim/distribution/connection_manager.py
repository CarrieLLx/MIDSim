import asyncio
import grpc
import time
import json
import weakref
import random
from typing import Dict, Any, Optional, Tuple, List, Set
from loguru import logger

class CircuitBreaker:
    """Circuit breaker that prevents repeated requests to a failing service."""
    
    def __init__(self, failure_threshold=5, recovery_timeout=30.0, half_open_success=2):
        self.failure_threshold = failure_threshold  # Failure count that triggers the circuit breaker.
        self.recovery_timeout = recovery_timeout    # Seconds before attempting recovery after the circuit opens.
        self.half_open_success = half_open_success  # Successful requests required in the half-open state.
        
        self.failures = 0                # Current failure count.
        self.state = "closed"            # Circuit state: closed (normal), open (blocked), half-open (testing recovery).
        self.last_failure_time = 0       # Time of the most recent failure.
        self.successes_since_half_open = 0  # Successful requests in the half-open state.
    
    def record_failure(self):
        """Record a failed request."""
        self.failures += 1
        self.last_failure_time = time.time()
        
        # Check whether to open the circuit.
        if self.state == "closed" and self.failures >= self.failure_threshold:
            logger.warning(f"Circuit breaker tripped after {self.failures} failures")
            self.state = "open"
        
        # Reopen a half-open circuit after a failure.
        elif self.state == "half-open":
            logger.warning("Circuit breaker reopened after failure in half-open state")
            self.state = "open"
            self.successes_since_half_open = 0
    
    def record_success(self):
        """Record a successful request."""
        # Only count successes in the half-open state.
        if self.state == "half-open":
            self.successes_since_half_open += 1
            
            # Check whether the recovery threshold has been reached.
            if self.successes_since_half_open >= self.half_open_success:
                logger.info(f"Circuit breaker reset after {self.successes_since_half_open} consecutive successes")
                self.reset()
    
    def reset(self):
        """Reset the circuit breaker state."""
        self.failures = 0
        self.state = "closed"
        self.last_failure_time = 0
        self.successes_since_half_open = 0
    
    def is_allowed(self):
        """Check whether a request is currently allowed through."""
        # Allow requests while the circuit is closed (normal state).
        if self.state == "closed":
            return True
        
        # For an open circuit, check whether the recovery timeout has elapsed.
        if self.state == "open":
            elapsed = time.time() - self.last_failure_time
            if elapsed >= self.recovery_timeout:
                # Enter the half-open state and allow a few trial requests.
                logger.info(f"Circuit half-open after {elapsed:.1f}s timeout")
                self.state = "half-open"
                self.successes_since_half_open = 0
                return True
            return False
        
        # Limit half-open concurrency by randomly allowing about 20% of requests.
        if self.state == "half-open":
            return random.random() < 0.2
        
        return True  # Safe default.


class ConnectionShard:
    """
    Connection-management shard responsible for creating and maintaining a subset of connections.
    """
    
    def __init__(self, 
                shard_id: int, 
                semaphore_size: int = 100, 
                lock_timeout: float = 5.0,
                max_idle_time: float = 3600.0):
        """
        Initialize the connection shard.
        
        Args:
            shard_id: Shard identifier.
            semaphore_size: Semaphore capacity controlling concurrent access.
            lock_timeout: Timeout in seconds for acquiring the lock.
            max_idle_time: Maximum retention time in seconds for idle connections.
        """
        self.shard_id = shard_id
        
        # Connection pool state.
        self.channels = {}  # {(address, port): (channel, last_used_time, use_count)}
        self.stubs = {}     # {(address, port, stub_type): stub}
        
        # Concurrency control.
        self.semaphore = asyncio.Semaphore(semaphore_size)
        self.lock_timeout = lock_timeout
        
        # Connection management.
        self.max_idle_time = max_idle_time
        self.cleanup_task = None
        
        # Circuit breakers: {(address, port): CircuitBreaker}.
        self.circuit_breakers = {}
        
        # Track known healthy connections to avoid repeated checks.
        self.known_good_connections = set()  # {(address, port)}
        
        # Preconnection flag.
        self.preconnected = False
    
    async def get_channel(self, address: str, port: int) -> grpc.aio.Channel:
        """
        Get or create a channel for the specified address and port.
        
        Args:
            address: Destination address.
            port: Destination port.
            
        Returns:
            Asynchronous gRPC channel.
        """
        key = (address, port)
        
        # Reject immediately if the circuit breaker blocks this connection.
        if key in self.circuit_breakers and not self.circuit_breakers[key].is_allowed():
            raise RuntimeError(f"Circuit breaker open for {key}, service may be unavailable")
        
        # Control concurrency with a semaphore.
        acquired = False
        try:
            # Acquire the semaphore with a timeout.
            try:
                acquired = await asyncio.wait_for(self.semaphore.acquire(), timeout=self.lock_timeout)
            except asyncio.TimeoutError:
                logger.warning(f"Shard {self.shard_id}: Timeout acquiring semaphore for {key}, creating uncached channel")
                # On timeout, return a temporary channel without caching it.
                return self._create_channel(address, port, cache=False)
                
            if acquired:
                try:
                    # First check for a cached channel that is known to be healthy.
                    if key in self.channels and key in self.known_good_connections:
                        channel, last_used, use_count = self.channels[key]
                        
                        # Update usage statistics and return.
                        self.channels[key] = (channel, time.time(), use_count + 1)
                        return channel
                    
                    # Check cached channels whose state still needs verification.
                    if key in self.channels:
                        channel, last_used, use_count = self.channels[key]
                        
                        # Check channel state to ensure it is usable.
                        current_state = channel.get_state(try_to_connect=False)
                        
                        # Use the channel directly if it is ready.
                        if current_state == grpc.ChannelConnectivity.READY:
                            self.channels[key] = (channel, time.time(), use_count + 1)
                            self.known_good_connections.add(key)
                            return channel
                            
                        # Try to activate an IDLE channel if it was used recently.
                        elif current_state == grpc.ChannelConnectivity.IDLE and time.time() - last_used < 300:
                            # Wait for the channel to become ready with a shorter timeout.
                            try:
                                is_ready = await self._wait_for_ready_state(channel, timeout=10.0)
                                if is_ready:
                                    # The channel is now ready.
                                    self.channels[key] = (channel, time.time(), use_count + 1)
                                    self.known_good_connections.add(key)
                                    return channel
                            except Exception as e:
                                logger.debug(f"Shard {self.shard_id}: Error checking idle channel: {e}")
                        
                        # Create a new channel for other states or failed activation.
                        logger.debug(f"Shard {self.shard_id}: Channel for {key} in state {current_state}, creating new one")
                        
                        # Remove the old channel without waiting for closure to avoid blocking.
                        old_channel = self.channels.pop(key)[0]
                        asyncio.create_task(old_channel.close())
                    
                    # Create and cache a new channel.
                    channel = self._create_channel(address, port)
                    
                    # Check channel state with a short timeout to avoid prolonged blocking.
                    is_ready = False
                    try:
                        is_ready = await self._wait_for_ready_state(channel, timeout=10.0)
                    except Exception as e:
                        logger.debug(f"Shard {self.shard_id}: Error checking new channel: {e}")
                    
                    # Cache regardless of readiness; check again on the next use.
                    self.channels[key] = (channel, time.time(), 1)
                    
                    if is_ready:
                        self.known_good_connections.add(key)
                        
                        # Register a weak-reference close callback.
                        weak_self = weakref.ref(self)
                        if hasattr(channel, "add_on_close_callback"):
                            channel.add_on_close_callback(
                                lambda: asyncio.create_task(
                                    weak_self()._on_channel_close(key) if weak_self() else None
                                )
                            )
                    
                    # Reset the circuit breaker, if present, after channel creation succeeds.
                    if key in self.circuit_breakers:
                        self.circuit_breakers[key].record_success()
                        
                    return channel
                    
                finally:
                    # Ensure the semaphore is released.
                    self.semaphore.release()
                    
        except Exception as e:
            logger.error(f"Shard {self.shard_id}: Unexpected error in get_channel for {key}: {e}")
            
            # Record the failure in the circuit breaker.
            if key in self.circuit_breakers:
                self.circuit_breakers[key].record_failure()
            else:
                # Create a new circuit breaker.
                self.circuit_breakers[key] = CircuitBreaker()
                self.circuit_breakers[key].record_failure()
                
            # Ensure the semaphore is released.
            if acquired:
                self.semaphore.release()
                
            # Create a temporary channel on error.
            return self._create_channel(address, port, cache=False)
    
    def _create_channel(self, address: str, port: int, cache: bool = True) -> grpc.aio.Channel:
        """
        Create a gRPC channel with optimized settings.
        
        Args:
            address: Destination address.
            port: Destination port.
            cache: Whether to cache this channel.
            
        Returns:
            Asynchronous gRPC channel.
        """
        # Configure a new channel for improved stability.
        options = [
            ('grpc.keepalive_time_ms', 10000),           # Send a keepalive every 10 seconds.
            ('grpc.keepalive_timeout_ms', 5000),         # Wait up to 5 seconds for a keepalive response.
            ('grpc.http2.max_pings_without_data', 5),    # Allow pings without data.
            ('grpc.http2.max_ping_strikes', 0),          # Do not disconnect because of ping strikes.
            ('grpc.http2.min_ping_interval_without_data_ms', 30000),
            ('grpc.keepalive_permit_without_calls', 1),  # Allow keepalive while idle.
            ('grpc.max_connection_idle_ms', 600000),     # Idle timeout of 10 minutes.
            ('grpc.max_connection_age_ms', 1800000),     # Maximum connection age of 30 minutes.
            ('grpc.max_connection_age_grace_ms', 60000), # Shutdown grace period of 1 minute.
            ('grpc.enable_retries', 1),                  # Enable retries.
            ('grpc.min_reconnect_backoff_ms', 100),      # Minimum reconnection backoff of 100 ms.
            ('grpc.max_reconnect_backoff_ms', 5000),     # Maximum reconnection backoff of 5 seconds.
            ('grpc.initial_reconnect_backoff_ms', 100),  # Initial reconnection backoff.
            ('grpc.client_idle_timeout_ms', -1),         # Disable the client idle timeout.
            ('grpc.service_config', json.dumps({
                'methodConfig': [{
                    'name': [{'service': 'agent.AgentService'}],
                    'retryPolicy': {
                        'maxAttempts': 5,
                        'initialBackoff': '0.1s',
                        'maxBackoff': '30s',
                        'backoffMultiplier': 2.0,
                        'retryableStatusCodes': [
                            'UNAVAILABLE', 'DEADLINE_EXCEEDED', 
                            'RESOURCE_EXHAUSTED', 'CANCELLED'
                        ]
                    }
                }]
            }))
        ]
        
        return grpc.aio.insecure_channel(f"{address}:{port}", options=options)
                
    async def _wait_for_ready_state(self, channel, timeout=10.0):
        """
        Wait for the channel to reach READY using shorter timeouts and polling intervals.
        
        Args:
            channel: gRPC channel to wait for.
            timeout: Maximum wait in seconds.
            
        Returns:
            bool: True if the channel becomes READY, otherwise False.
        """
        start_time = time.time()
        end_time = start_time + timeout
        
        # Check the current state first.
        current_state = channel.get_state(try_to_connect=True)
        if current_state == grpc.ChannelConnectivity.READY:
            return True
        
        # Wait until the channel reaches READY or the timeout expires.
        while time.time() < end_time:
            try:
                # Compute the remaining time.
                remaining = end_time - time.time()
                if remaining <= 0:
                    break
                    
                # Wait at most 0.5 seconds for each state change to avoid prolonged blocking.
                await asyncio.wait_for(
                    channel.wait_for_state_change(current_state),
                    timeout=min(0.5, remaining)
                )
                
                # Check the new state.
                new_state = channel.get_state(try_to_connect=False)
                
                if new_state == grpc.ChannelConnectivity.READY:
                    return True
                elif new_state in (grpc.ChannelConnectivity.TRANSIENT_FAILURE, grpc.ChannelConnectivity.SHUTDOWN):
                    return False
                    
                # Update the current state and continue waiting.
                current_state = new_state
                
            except asyncio.TimeoutError:
                # Continue checking after a short timeout while the overall deadline has not expired.
                current_state = channel.get_state(try_to_connect=True)
                
                if current_state == grpc.ChannelConnectivity.READY:
                    return True
                elif current_state in (grpc.ChannelConnectivity.TRANSIENT_FAILURE, grpc.ChannelConnectivity.SHUTDOWN):
                    return False
        
        # Check the final state.
        final_state = channel.get_state(try_to_connect=False)
        return final_state == grpc.ChannelConnectivity.READY
    
    async def _on_channel_close(self, key):
        """Handle the channel close callback."""
        # Use an asynchronous task to avoid blocking.
        try:
            # Remove the connection from the known-healthy set if present.
            if key in self.known_good_connections:
                self.known_good_connections.remove(key)
                
            async with self.semaphore:
                if key in self.channels:
                    self.channels.pop(key, None)
                    
                    # Remove associated stubs.
                    stub_keys = [k for k in self.stubs.keys() if k[0] == key[0] and k[1] == key[1]]
                    for stub_key in stub_keys:
                        self.stubs.pop(stub_key, None)
                        
        except Exception as e:
            logger.error(f"Shard {self.shard_id}: Error in _on_channel_close for {key}: {e}")
    
    async def get_stub(self, address: str, port: int, stub_type) -> Any:
        """
        Get or create a stub for the specified address, port and type.
        
        Args:
            address: Destination address.
            port: Destination port.
            stub_type: Stub class.
            
        Returns:
            gRPC stub.
        """
        key = (address, port, stub_type)
        
        # Control concurrency with a semaphore.
        acquired = False
        try:
            try:
                acquired = await asyncio.wait_for(self.semaphore.acquire(), timeout=self.lock_timeout)
            except asyncio.TimeoutError:
                logger.warning(f"Shard {self.shard_id}: Timeout acquiring semaphore in get_stub for {key}")
                # Create a temporary channel and stub on timeout.
                temp_channel = self._create_channel(address, port, cache=False)
                return stub_type(temp_channel)
                
            if acquired:
                try:
                    # Check the cache.
                    if key in self.stubs:
                        return self.stubs[key]
                    
                    # Release the semaphore to avoid recursive deadlock.
                    self.semaphore.release()
                    acquired = False
                    
                    # Get or create a channel.
                    channel = await self.get_channel(address, port)
                    
                    # Acquire the semaphore again.
                    try:
                        acquired = await asyncio.wait_for(self.semaphore.acquire(), timeout=self.lock_timeout)
                    except asyncio.TimeoutError:
                        # Return an uncached stub directly on timeout.
                        return stub_type(channel)
                    
                    if acquired:
                        # Create and cache the stub.
                        stub = stub_type(channel)
                        self.stubs[key] = stub
                        return stub
                finally:
                    if acquired:
                        self.semaphore.release()
        except Exception as e:
            logger.error(f"Shard {self.shard_id}: Error in get_stub for {key}: {e}")
            
            # Ensure the semaphore is released.
            if acquired:
                self.semaphore.release()
                
            # Create a temporary stub on error.
            temp_channel = self._create_channel(address, port, cache=False)
            return stub_type(temp_channel)
    
    async def with_stub(self, address: str, port: int, stub_type, func, *args, **kwargs):
        """
        Call a function through the stub with retry handling.
        
        Args:
            address: Destination address.
            port: Destination port.
            stub_type: Stub class.
            func: Name of the function to call.
            *args, **kwargs: Arguments passed to the function.
            
        Returns:
            Result of the function call.
        """
        key = (address, port)
        max_retries = 3  # Limit retries to avoid prolonged waits.
        retry_count = 0
        last_exception = None
        
        # Check the circuit breaker.
        if key in self.circuit_breakers and not self.circuit_breakers[key].is_allowed():
            raise RuntimeError(f"Circuit breaker open for {key}, service may be unavailable")
        
        while retry_count < max_retries:
            try:
                # Get the stub.
                stub = await self.get_stub(address, port, stub_type)
                method = getattr(stub, func)
                
                # Set a timeout to avoid prolonged blocking.
                timeout = 600.0  # Default timeout in seconds.
                
                # Execute the RPC call.
                try:
                    result = await asyncio.wait_for(
                        method(*args, **kwargs),
                        timeout=timeout
                    )
                    
                    # Record successful calls in the circuit breaker.
                    if key in self.circuit_breakers:
                        self.circuit_breakers[key].record_success()
                        
                    return result
                    
                except asyncio.TimeoutError:
                    error_msg = f"RPC call {func} timed out after {timeout} seconds"
                    logger.warning(f"Shard {self.shard_id}: {error_msg}")
                    raise RuntimeError(error_msg)
                
            except asyncio.TimeoutError:
                # RPC call timed out.
                last_exception = RuntimeError(f"RPC call {func} timed out")
                logger.warning(f"Shard {self.shard_id}: Timeout calling {func} on {address}:{port}")
                
                # Record the failure in the circuit breaker.
                if key in self.circuit_breakers:
                    self.circuit_breakers[key].record_failure()
                else:
                    self.circuit_breakers[key] = CircuitBreaker()
                    self.circuit_breakers[key].record_failure()
                
                retry_count += 1
                
                # Retry quickly to avoid prolonged waits.
                if retry_count < max_retries:
                    wait_time = 0.1 * (2 ** retry_count)  # Exponential backoff with a smaller base delay.
                    await asyncio.sleep(wait_time)
                
            except grpc.aio.AioRpcError as e:
                last_exception = e
                
                # Log error details.
                logger.warning(f"Shard {self.shard_id}: gRPC error in {func} on {address}:{port}: {e.code()}: {e.details()}")
                
                # Record the failure in the circuit breaker.
                if key in self.circuit_breakers:
                    self.circuit_breakers[key].record_failure()
                else:
                    self.circuit_breakers[key] = CircuitBreaker()
                    self.circuit_breakers[key].record_failure()
                
                # Check whether to retry.
                retriable_codes = (
                    grpc.StatusCode.UNAVAILABLE, 
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    grpc.StatusCode.RESOURCE_EXHAUSTED,
                    grpc.StatusCode.CANCELLED,
                    grpc.StatusCode.ABORTED,
                    grpc.StatusCode.UNKNOWN
                )
                
                if e.code() in retriable_codes:
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = 0.1 * (2 ** retry_count)
                        await asyncio.sleep(wait_time)
                        continue
                
                # Propagate errors that cannot be retried.
                logger.error(f"Shard {self.shard_id}: Non-retriable gRPC error: {e.code()}: {e.details()}")
                raise
                
            except Exception as e:
                logger.error(f"Shard {self.shard_id}: Unexpected error in with_stub {address}:{port}: {e}")
                last_exception = e
                
                # Record the failure in the circuit breaker.
                if key in self.circuit_breakers:
                    self.circuit_breakers[key].record_failure()
                else:
                    self.circuit_breakers[key] = CircuitBreaker()
                    self.circuit_breakers[key].record_failure()
                
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = 0.1 * (2 ** retry_count)
                    await asyncio.sleep(wait_time)
                    continue
                raise
        
        # Retries exhausted.
        if last_exception:
            logger.error(f"Shard {self.shard_id}: Failed after {max_retries} attempts: {last_exception}")
            raise last_exception
        else:
            raise RuntimeError(f"Failed to execute {func} after {max_retries} retries")

    async def start_cleanup_task(self):
        """Start the cleanup task."""
        if self.cleanup_task is None or self.cleanup_task.done():
            self.cleanup_task = asyncio.create_task(self._cleanup_idle_channels())
            logger.debug(f"Shard {self.shard_id}: Cleanup task started")
    
    async def _cleanup_idle_channels(self):
        """Periodically clean up idle channels."""
        try:
            while True:
                await asyncio.sleep(300)  # Run cleanup every 5 minutes.
                
                # Take a snapshot and process it without holding the semaphore for too long.
                to_check = []
                
                async with self.semaphore:
                    current_time = time.time()
                    for key, (channel, last_used, _) in self.channels.items():
                        if current_time - last_used > self.max_idle_time:
                            to_check.append((key, channel))
                
                # Close channels individually outside the lock.
                for key, channel in to_check:
                    try:
                        # Check again whether the channel still needs to close.
                        should_close = False
                        async with self.semaphore:
                            if key in self.channels:
                                _, last_used, _ = self.channels[key]
                                if time.time() - last_used > self.max_idle_time:
                                    self.channels.pop(key)
                                    should_close = True
                                    
                                    # Remove associated stubs.
                                    stub_keys = [k for k in self.stubs.keys() if k[0] == key[0] and k[1] == key[1]]
                                    for stub_key in stub_keys:
                                        self.stubs.pop(stub_key, None)
                                    
                                    # Remove it from the known-healthy connection set.
                                    if key in self.known_good_connections:
                                        self.known_good_connections.remove(key)
                        
                        # Once closure is confirmed, execute it outside the lock.
                        if should_close:
                            await channel.close()
                    except Exception as e:
                        logger.warning(f"Shard {self.shard_id}: Error closing idle channel for {key}: {e}")
                        
        except asyncio.CancelledError:
            logger.debug(f"Shard {self.shard_id}: Cleanup task cancelled")
        except Exception as e:
            logger.error(f"Shard {self.shard_id}: Error in cleanup task: {e}")
            
    async def preconnect(self, workers: List[Tuple[str, int]]):
        """
        Preconnect to all worker nodes.
        
        Args:
            workers: List of (address, port) tuples for worker nodes.
        """
        if self.preconnected:
            return
            
        logger.info(f"Shard {self.shard_id}: Pre-connecting to {len(workers)} workers")
        
        # Limit concurrent connections with a semaphore.
        sem = asyncio.Semaphore(min(10, len(workers)))
        
        async def connect_one(addr, port):
            async with sem:
                try:
                    channel = await self.get_channel(addr, port)
                    return (addr, port), True
                except Exception as e:
                    logger.warning(f"Shard {self.shard_id}: Failed to pre-connect to {addr}:{port}: {e}")
                    return (addr, port), False
        
        # Create all connection tasks.
        tasks = [connect_one(addr, port) for addr, port in workers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results.
        success_count = sum(1 for result in results if isinstance(result, tuple) and result[1])
        logger.info(f"Shard {self.shard_id}: Successfully pre-connected to {success_count}/{len(workers)} workers")
        
        self.preconnected = True


class ShardedConnectionManager:
    """
    Connection manager that reduces lock contention by hashing connections into shards.
    """
    
    _instance = None
    _init_lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ShardedConnectionManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # Configuration.
        self.shard_count = 32 # Number of shards.
        self.semaphore_size = 10 # Semaphore capacity per shard.
        self.lock_timeout = 5.0  # Lock timeout in seconds (reduced from 30.0).
        self.max_idle_time = 600.0  # Idle channel timeout in seconds (10 minutes).
        
        # Create shards.
        self.shards = [
            ConnectionShard(
                shard_id=i, 
                semaphore_size=self.semaphore_size,
                lock_timeout=self.lock_timeout,
                max_idle_time=self.max_idle_time
            ) 
            for i in range(self.shard_count)
        ]
        
        # Preconnection state.
        self.preconnected = False
        self._initialized = True
    
    def get_shard(self, address: str, port: int) -> ConnectionShard:
        """
        Select the shard for an address and port.
        
        Args:
            address: Destination address.
            port: Destination port.
            
        Returns:
            Corresponding connection shard.
        """
        # Use consistent hashing to avoid hotspots.
        key = f"{address}:{port}"
        shard_index = hash(key) % self.shard_count
        return self.shards[shard_index]
    
    async def initialize(self):
        """Initialize the connection manager once."""
        async with self._init_lock:
            # Start the cleanup task for each shard.
            for shard in self.shards:
                await shard.start_cleanup_task()
            
            logger.info(f"Initialized connection manager with {self.shard_count} shards")
    
    async def get_channel(self, address: str, port: int) -> grpc.aio.Channel:
        """
        Get or create a channel.
        
        Args:
            address: Destination address.
            port: Destination port.
            
        Returns:
            Asynchronous gRPC channel.
        """
        shard = self.get_shard(address, port)
        return await shard.get_channel(address, port)
    
    async def get_stub(self, address: str, port: int, stub_type) -> Any:
        """
        Get or create a stub.
        
        Args:
            address: Destination address.
            port: Destination port.
            stub_type: Stub class.
            
        Returns:
            gRPC stub.
        """
        shard = self.get_shard(address, port)
        return await shard.get_stub(address, port, stub_type)
    
    async def with_stub(self, address: str, port: int, stub_type, func, *args, **kwargs):
        """
        Call a function through the stub.
        
        Args:
            address: Destination address.
            port: Destination port.
            stub_type: Stub class.
            func: Name of the function to call.
            *args, **kwargs: Arguments passed to the function.
            
        Returns:
            Result of the function call.
        """
        shard = self.get_shard(address, port)
        return await shard.with_stub(address, port, stub_type, func, *args, **kwargs)
    
    async def preconnect(self, workers: List[Tuple[str, int]]):
        """
        Preconnect to all worker nodes.
        
        Args:
            workers: List of worker nodes.
        """
        if self.preconnected:
            return
            
        logger.info(f"Pre-connecting to {len(workers)} workers across {self.shard_count} shards")
        
        # Assign worker nodes to shards.
        shard_workers = [[] for _ in range(self.shard_count)]
        for addr, port in workers:
            shard_index = hash(f"{addr}:{port}") % self.shard_count
            shard_workers[shard_index].append((addr, port))
        
        # Preconnect all shards concurrently.
        tasks = []
        for i, worker_list in enumerate(shard_workers):
            if worker_list:
                tasks.append(self.shards[i].preconnect(worker_list))
        
        await asyncio.gather(*tasks)
        self.preconnected = True
        logger.info("Pre-connection to all workers completed")
    
    async def stop(self):
        """Stop the connection manager and close all channels."""
        logger.info("Stopping connection manager")
        
        # Collect all channels to close.
        all_channels = []
        for shard in self.shards:
            async with shard.semaphore:
                for key, (channel, _, _) in list(shard.channels.items()):
                    all_channels.append((key, channel))
                shard.channels.clear()
                shard.stubs.clear()
                shard.known_good_connections.clear()
                
                if shard.cleanup_task and not shard.cleanup_task.done():
                    shard.cleanup_task.cancel()
        
        # Close all channels without holding locks.
        for key, channel in all_channels:
            try:
                await asyncio.wait_for(channel.close(), timeout=1.0)
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"Error closing channel for {key}: {e}")
        
        logger.info("Connection manager stopped")
    
    def sync_stop(self):
        """Synchronous shutdown for non-async contexts."""
        logger.info("Sync stopping connection manager")
        for shard in self.shards:
            if shard.cleanup_task and not shard.cleanup_task.done():
                shard.cleanup_task.cancel()


# Create the global singleton.
connection_manager = ShardedConnectionManager()


# Provide a convenience function for global initialization.
async def initialize_connection_manager(workers: List[Tuple[str, int]] = None):
    """
    Initialize the global connection manager and optionally preconnect to workers.
    
    Args:
        workers: Optional list of worker nodes to preconnect to.
    """
    await connection_manager.initialize()
    
    if workers:
        await connection_manager.preconnect(workers)

    return connection_manager