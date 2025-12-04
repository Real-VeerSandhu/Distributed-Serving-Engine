import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Awaitable
from collections import defaultdict
import uuid

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
        self.max_batch_size = max_batch_size
        self.max_latency = max_latency_ms / 1000.0
        self.batch_callback = batch_callback
        
        self._batches: Dict[str, Batch] = {}
        self._flush_tasks: Dict[str, asyncio.Task] = {}
        self._running = False
        
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
        self._running = False
        
        for task in self._flush_tasks.values():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        for batch_key, batch in list(self._batches.items()):
            await self._flush_batch(batch_key, batch)
        
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
        
        if len(batch.requests) >= self.max_batch_size:
            await self._flush_batch(batch_key, batch)
        
        return batched_request.future
    
    async def _schedule_flush(self, batch_key: str, batch: Batch) -> None:
        try:
            await asyncio.sleep(batch.max_latency)
            
            if batch_key in self._batches and self._batches[batch_key] is batch:
                if batch.requests:
                    await self._flush_batch(batch_key, batch)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in scheduled flush: {e}")
    
    async def _flush_batch(self, batch_key: str, batch: Batch) -> None:
        if not batch.requests:
            if batch_key in self._batches:
                del self._batches[batch_key]
            if batch_key in self._flush_tasks:
                task = self._flush_tasks.pop(batch_key)
                if not task.done():
                    task.cancel()
            return
        
        requests_to_process = batch.requests[:]
        batch.requests.clear()
        
        if batch_key in self._batches:
            del self._batches[batch_key]
        if batch_key in self._flush_tasks:
            task = self._flush_tasks.pop(batch_key)
            if not task.done():
                task.cancel()
        
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
                
                for req, result in zip(requests_to_process, results):
                    if not req.future.done():
                        req.future.set_result(result)
            else:
                for req in requests_to_process:
                    if not req.future.done():
                        req.future.set_exception(
                            RuntimeError("No batch callback configured")
                        )
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            for req in requests_to_process:
                if not req.future.done():
                    req.future.set_exception(e)
    
    def get_stats(self) -> Dict[str, Any]:
        avg_batch_size = (
            self.total_batched_requests / self.total_batches
            if self.total_batches > 0
            else 0
        )
        
        return {
            "total_batches": self.total_batches,
            "total_requests": self.total_requests,
            "total_batched_requests": self.total_batched_requests,
            "avg_batch_size": avg_batch_size,
            "max_batch_size": self.max_batch_size,
            "max_latency_ms": self.max_latency * 1000.0,
            "pending_batches": len(self._batches)
        }

