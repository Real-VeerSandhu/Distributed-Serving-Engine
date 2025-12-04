import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Awaitable, Set
from collections import defaultdict
import uuid
import contextlib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BatchedRequest:
    request_id: str
    model_name: str
    version: str
    inputs: Any
    created_at: float
    future: asyncio.Future = field(default_factory=lambda: asyncio.Future())


@dataclass
class Batch:
    model_name: str
    version: str
    requests: List[BatchedRequest]
    created_at: float
    max_batch_size: int
    max_latency: float


class Batcher:
    def __init__(
        self,
        max_batch_size: int = 32,
        max_latency_ms: float = 100.0,
        batch_callback: Optional[Callable[[str, str, List[Any]], Awaitable[List[Any]]]] = None
    ):
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        if max_latency_ms <= 0:
            raise ValueError("max_latency_ms must be greater than 0")
            
        self.max_batch_size = max_batch_size
        self.max_latency = max_latency_ms / 1000.0
        self.batch_callback = batch_callback
        
        self._batches: Dict[str, Batch] = {}
        self._flush_tasks: Dict[str, asyncio.Task] = {}
        self._running = False
        self._lock = asyncio.Lock()
        
        self.total_batches = 0
        self.total_requests = 0
        self.total_batched_requests = 0
    
    async def start(self) -> None:
        if self._running:
            logger.warning("Batcher is already running")
            return
        
        self._running = True
        logger.info("Batcher started")
    
    async def stop(self) -> None:
        if not self._running:
            return
            
        self._running = False
        
        # Make a copy of the items to avoid modification during iteration
        async with self._lock:
            tasks_to_cancel = list(self._flush_tasks.values())
            batches_to_process = list(self._batches.items())
            
            # Clear the dictionaries
            self._flush_tasks.clear()
            self._batches.clear()
        
        # Cancel all pending flush tasks
        for task in tasks_to_cancel:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        
        # Process any remaining batches
        for batch_key, batch in batches_to_process:
            if batch.requests:
                try:
                    await self._process_batch(batch_key, batch, batch.requests.copy())
                except Exception as e:
                    logger.exception(f"Error processing batch {batch_key} during shutdown")
        
        logger.info("Batcher stopped")
    
    async def add_request(
        self,
        model_name: str,
        version: str,
        inputs: Any,
        request_id: Optional[str] = None
    ) -> asyncio.Future:
        if not self._running:
            raise RuntimeError("Batcher is not running")
        
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        batch_key = f"{model_name}:{version}"
        
        batched_request = BatchedRequest(
            request_id=request_id,
            model_name=model_name,
            version=version,
            inputs=inputs,
            created_at=time.time()
        )
        
        async with self._lock:
            if batch_key not in self._batches:
                batch = Batch(
                    model_name=model_name,
                    version=version,
                    requests=[],
                    created_at=time.time(),
                    max_batch_size=self.max_batch_size,
                    max_latency=self.max_latency
                )
                self._batches[batch_key] = batch
                self._flush_tasks[batch_key] = asyncio.create_task(
                    self._schedule_flush(batch_key, batch)
                )
            
            batch = self._batches[batch_key]
            batch.requests.append(batched_request)
            self.total_requests += 1
            
            should_flush = len(batch.requests) >= self.max_batch_size
        
        if should_flush:
            await self._flush_batch(batch_key, batch)
        
        return batched_request.future
    
    async def _schedule_flush(self, batch_key: str, batch: Batch) -> None:
        try:
            await asyncio.sleep(batch.max_latency)
            
            async with self._lock:
                # Only proceed if this batch still exists and hasn't been flushed
                if batch_key in self._batches and self._batches[batch_key] is batch and batch.requests:
                    # Create a copy of the requests and clear them
                    requests_to_process = batch.requests.copy()
                    batch.requests.clear()
                    
                    # Remove the batch from tracking
                    del self._batches[batch_key]
                    
                    # Process the batch
                    await self._process_batch(batch_key, batch, requests_to_process)
        except asyncio.CancelledError:
            # Task was cancelled, which is expected during normal operation
            pass
        except Exception as e:
            logger.exception(f"Unexpected error in scheduled flush for batch {batch_key}")
            
            # Make sure to clean up the batch on error
            async with self._lock:
                if batch_key in self._batches:
                    del self._batches[batch_key]
                if batch_key in self._flush_tasks:
                    del self._flush_tasks[batch_key]
    
    async def _flush_batch(self, batch_key: str, batch: Batch) -> None:
        async with self._lock:
            if not batch.requests:
                return
                
            # Get a copy of the requests and clear them
            requests_to_process = batch.requests.copy()
            batch.requests.clear()
            
            # Remove the batch from tracking
            if batch_key in self._batches and self._batches[batch_key] is batch:
                del self._batches[batch_key]
            
            # Cancel and remove the flush task
            if batch_key in self._flush_tasks:
                task = self._flush_tasks.pop(batch_key)
                if not task.done():
                    task.cancel()
        
        # Process the batch outside the lock
        await self._process_batch(batch_key, batch, requests_to_process)
    
    async def _process_batch(self, batch_key: str, batch: Batch, requests_to_process: List[BatchedRequest]) -> None:
        if not requests_to_process:
            return
            
        self.total_batches += 1
        self.total_batched_requests += len(requests_to_process)
        
        try:
            if self.batch_callback:
                inputs_list = [req.inputs for req in requests_to_process]
                results = await self.batch_callback(
                    batch.model_name,
                    batch.version,
                    inputs_list
                )
                
                if len(results) != len(requests_to_process):
                    raise ValueError(
                        f"Batch callback returned {len(results)} results "
                        f"but expected {len(requests_to_process)}"
                    )
                
                # Set results for all requests
                for req, result in zip(requests_to_process, results):
                    if not req.future.done():
                        req.future.set_result(result)
            else:
                # No callback configured, set an exception on all requests
                error = RuntimeError("No batch callback configured")
                for req in requests_to_process:
                    if not req.future.done():
                        req.future.set_exception(error)
                        
        except Exception as e:
            logger.exception(f"Error processing batch {batch_key}")
            # Set the exception on all requests in the batch
            for req in requests_to_process:
                if not req.future.done():
                    req.future.set_exception(e)
    
    async def get_stats(self) -> Dict[str, Any]:
        avg_batch_size = (
            self.total_batched_requests / self.total_batches
            if self.total_batches > 0
            else 0
        )
        
        pending_batches = 0
        pending_requests = 0
        
        # Get a snapshot of the current state
        async with self._lock:
            for batch in self._batches.values():
                if batch.requests:
                    pending_batches += 1
                    pending_requests += len(batch.requests)
        
        return {
            "total_batches": self.total_batches,
            "total_requests": self.total_requests,
            "total_batched_requests": self.total_batched_requests,
            "pending_batches": pending_batches,
            "pending_requests": pending_requests,
            "avg_batch_size": avg_batch_size,
            "max_batch_size": self.max_batch_size,
            "max_latency_ms": self.max_latency * 1000.0,
            "pending_batches": len(self._batches)
        }

