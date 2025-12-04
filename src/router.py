"""
Router for the Distributed Inference Engine.

This module provides routing and sharding logic for distributing inference
requests across worker nodes. It handles model-to-worker mapping, hash-based
sharding, worker health checks, and failover.
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum, auto

from src.model_registry import ModelRegistry, ModelShard, ModelStatus

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WorkerHealth(Enum):
    """Health status of a worker."""
    HEALTHY = auto()
    UNHEALTHY = auto()
    UNKNOWN = auto()


@dataclass
class WorkerInfo:
    """Information about a worker node."""
    worker_id: str
    address: str  # e.g., "127.0.0.1:9001"
    health: WorkerHealth = WorkerHealth.UNKNOWN
    last_health_check: float = 0.0
    consecutive_failures: int = 0
    last_success: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class Router:
    """
    Router for distributing inference requests across worker nodes.
    
    Features:
    - Hash-based sharding for consistent request routing
    - Worker health monitoring and automatic failover
    - Load-aware routing (optional)
    - Integration with model registry for shard information
    """
    
    def __init__(
        self,
        registry: ModelRegistry,
        health_check_interval: float = 5.0,
        health_check_timeout: float = 2.0,
        max_consecutive_failures: int = 3,
        failover_enabled: bool = True
    ):
        """
        Initialize the router.
        
        Args:
            registry: Model registry instance
            health_check_interval: Seconds between health checks
            health_check_timeout: Timeout for health check requests
            max_consecutive_failures: Mark worker unhealthy after N failures
            failover_enabled: Enable automatic failover to backup workers
        """
        self.registry = registry
        self.health_check_interval = health_check_interval
        self.health_check_timeout = health_check_timeout
        self.max_consecutive_failures = max_consecutive_failures
        self.failover_enabled = failover_enabled
        
        # Worker tracking
        self.workers: Dict[str, WorkerInfo] = {}  # worker_id -> WorkerInfo
        
        # Health check task
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False
        
    async def start(self) -> None:
        """Start the router and begin health checking."""
        if self._running:
            logger.warning("Router is already running")
            return
            
        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Router started")
    
    async def stop(self) -> None:
        """Stop the router and health checking."""
        self._running = False
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        logger.info("Router stopped")
    
    def register_worker(
        self,
        worker_id: str,
        address: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a worker with the router.
        
        Args:
            worker_id: Unique identifier for the worker
            address: Network address (host:port)
            metadata: Optional metadata about the worker
        """
        if worker_id in self.workers:
            logger.warning(f"Worker {worker_id} already registered, updating...")
        
        self.workers[worker_id] = WorkerInfo(
            worker_id=worker_id,
            address=address,
            health=WorkerHealth.UNKNOWN,
            metadata=metadata or {}
        )
        logger.info(f"Registered worker {worker_id} at {address}")
    
    def unregister_worker(self, worker_id: str) -> None:
        """Unregister a worker from the router."""
        if worker_id in self.workers:
            del self.workers[worker_id]
            logger.info(f"Unregistered worker {worker_id}")
    
    def route_request(
        self,
        model_name: str,
        version: str,
        request_key: str,
        prefer_healthy: bool = True
    ) -> Optional[ModelShard]:
        """
        Route a request to the appropriate worker shard.
        
        Args:
            model_name: Name of the model
            version: Model version
            request_key: Key for consistent hashing (e.g., user_id, session_id)
            prefer_healthy: Only route to healthy workers
            
        Returns:
            ModelShard if routing successful, None otherwise
        """
        # Get shard from registry using hash-based sharding
        shard = self.registry.get_shard_for_key(model_name, version, request_key)
        
        if shard is None:
            logger.warning(f"No shard found for model {model_name}:{version}")
            return None
        
        # Check if worker is healthy
        worker_info = self.workers.get(shard.worker_id)
        if worker_info is None:
            logger.warning(f"Worker {shard.worker_id} not registered")
            if not self.failover_enabled:
                return None
            # Try to find an alternative shard
            return self._find_alternative_shard(model_name, version, request_key)
        
        if prefer_healthy and worker_info.health != WorkerHealth.HEALTHY:
            logger.warning(
                f"Worker {shard.worker_id} is {worker_info.health.name}, "
                f"attempting failover"
            )
            if self.failover_enabled:
                return self._find_alternative_shard(model_name, version, request_key)
            return None
        
        return shard
    
    def _find_alternative_shard(
        self,
        model_name: str,
        version: str,
        request_key: str
    ) -> Optional[ModelShard]:
        """
        Find an alternative shard when primary shard is unavailable.
        
        Uses round-robin or load-based selection from available healthy shards.
        """
        model_version = self.registry.get_model_version(model_name, version)
        if model_version is None:
            return None
        
        # Get all healthy shards for this model
        healthy_shards = []
        for shard in model_version.shards:
            worker_info = self.workers.get(shard.worker_id)
            if worker_info and worker_info.health == WorkerHealth.HEALTHY:
                healthy_shards.append(shard)
        
        if not healthy_shards:
            logger.error(f"No healthy shards available for {model_name}:{version}")
            return None
        
        # Simple round-robin based on request key hash
        # This ensures same key goes to same backup shard
        key_hash = int(hashlib.md5(request_key.encode()).hexdigest(), 16)
        selected = healthy_shards[key_hash % len(healthy_shards)]
        
        logger.info(
            f"Failover: routing {request_key} to shard {selected.shard_id} "
            f"on worker {selected.worker_id}"
        )
        return selected
    
    def mark_worker_success(self, worker_id: str) -> None:
        """Mark a worker request as successful."""
        if worker_id in self.workers:
            worker = self.workers[worker_id]
            worker.last_success = time.time()
            worker.consecutive_failures = 0
            if worker.health != WorkerHealth.HEALTHY:
                worker.health = WorkerHealth.HEALTHY
                logger.info(f"Worker {worker_id} marked as healthy")
    
    def mark_worker_failure(self, worker_id: str) -> None:
        """Mark a worker request as failed."""
        if worker_id in self.workers:
            worker = self.workers[worker_id]
            worker.consecutive_failures += 1
            
            if worker.consecutive_failures >= self.max_consecutive_failures:
                if worker.health != WorkerHealth.UNHEALTHY:
                    worker.health = WorkerHealth.UNHEALTHY
                    logger.warning(
                        f"Worker {worker_id} marked as unhealthy "
                        f"({worker.consecutive_failures} consecutive failures)"
                    )
    
    async def _health_check_loop(self) -> None:
        """Background task to periodically check worker health."""
        while self._running:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._check_all_workers()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
    
    async def _check_all_workers(self) -> None:
        """Check health of all registered workers."""
        if not self.workers:
            return
        
        tasks = [
            self._check_worker_health(worker_id)
            for worker_id in self.workers.keys()
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_worker_health(self, worker_id: str) -> None:
        """
        Check the health of a single worker.
        
        This is a simple TCP connection check. In production, you might
        want to send an actual health check request.
        """
        worker = self.workers.get(worker_id)
        if worker is None:
            return
        
        try:
            # Simple TCP connection check
            host, port = worker.address.split(':')
            port = int(port)
            
            # Try to connect with timeout
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port),
                    timeout=self.health_check_timeout
                )
                writer.close()
                await writer.wait_closed()
                
                # Connection successful
                self.mark_worker_success(worker_id)
                worker.last_health_check = time.time()
                
            except (asyncio.TimeoutError, OSError) as e:
                # Connection failed
                self.mark_worker_failure(worker_id)
                worker.last_health_check = time.time()
                logger.debug(f"Health check failed for {worker_id}: {e}")
                
        except Exception as e:
            logger.error(f"Error checking health of {worker_id}: {e}")
            self.mark_worker_failure(worker_id)
    
    def get_worker_address(self, worker_id: str) -> Optional[str]:
        """Get the network address for a worker."""
        worker = self.workers.get(worker_id)
        return worker.address if worker else None
    
    def get_healthy_workers(self) -> List[str]:
        """Get list of healthy worker IDs."""
        return [
            worker_id
            for worker_id, worker in self.workers.items()
            if worker.health == WorkerHealth.HEALTHY
        ]
    
    def get_unhealthy_workers(self) -> List[str]:
        """Get list of unhealthy worker IDs."""
        return [
            worker_id
            for worker_id, worker in self.workers.items()
            if worker.health == WorkerHealth.UNHEALTHY
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics."""
        total_workers = len(self.workers)
        healthy_count = len(self.get_healthy_workers())
        unhealthy_count = len(self.get_unhealthy_workers())
        
        return {
            "total_workers": total_workers,
            "healthy_workers": healthy_count,
            "unhealthy_workers": unhealthy_count,
            "unknown_workers": total_workers - healthy_count - unhealthy_count,
            "failover_enabled": self.failover_enabled,
            "health_check_interval": self.health_check_interval
        }
    
    def get_worker_info(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a worker."""
        worker = self.workers.get(worker_id)
        if worker is None:
            return None
        
        return {
            "worker_id": worker.worker_id,
            "address": worker.address,
            "health": worker.health.name,
            "consecutive_failures": worker.consecutive_failures,
            "last_health_check": worker.last_health_check,
            "last_success": worker.last_success,
            "metadata": worker.metadata
        }

