"""
Load Balancer

This module implements a load balancer that distributes inference requests
across multiple worker nodes based on various strategies.
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable, Awaitable, Any

logger = logging.getLogger(__name__)

class LoadBalancerStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RANDOM = "random"
    LEAST_LATENCY = "least_latency"

@dataclass
class WorkerStats:
    """Statistics for a worker."""
    request_count: int = 0
    error_count: int = 0
    active_connections: int = 0
    total_latency: float = 0.0
    last_seen: float = field(default_factory=time.time)

    @property
    def avg_latency(self) -> float:
        """Calculate average request latency."""
        return self.total_latency / self.request_count if self.request_count > 0 else 0.0

class LoadBalancer:
    """Load balancer for distributing requests across workers."""
    
    def __init__(
        self,
        strategy: LoadBalancerStrategy = LoadBalancerStrategy.ROUND_ROBIN,
        health_check_interval: float = 5.0,
        health_check_timeout: float = 2.0,
        max_failures: int = 3
    ):
        """Initialize the load balancer.
        
        Args:
            strategy: Load balancing strategy to use
            health_check_interval: Seconds between health checks
            health_check_timeout: Timeout for health check requests
            max_failures: Mark worker as unhealthy after this many consecutive failures
        """
        self.strategy = strategy
        self.health_check_interval = health_check_interval
        self.health_check_timeout = health_check_timeout
        self.max_failures = max_failures
        
        self.workers: Dict[str, str] = {}  # worker_id -> address
        self.worker_stats: Dict[str, WorkerStats] = {}
        self.health_checks: Dict[str, int] = {}  # worker_id -> failure_count
        
        self._strategy_fns = {
            LoadBalancerStrategy.ROUND_ROBIN: self._round_robin,
            LoadBalancerStrategy.LEAST_CONNECTIONS: self._least_connections,
            LoadBalancerStrategy.RANDOM: self._random,
            LoadBalancerStrategy.LEAST_LATENCY: self._least_latency
        }
        
        self._current_worker_idx = 0
        self._running = False
        self._health_check_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the load balancer and health checks."""
        if self._running:
            return
            
        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Load balancer started")
    
    async def stop(self) -> None:
        """Stop the load balancer and health checks."""
        self._running = False
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        logger.info("Load balancer stopped")
    
    def register_worker(self, worker_id: str, address: str) -> None:
        """Register a worker with the load balancer.
        
        Args:
            worker_id: Unique identifier for the worker
            address: Network address (host:port) of the worker
        """
        self.workers[worker_id] = address
        if worker_id not in self.worker_stats:
            self.worker_stats[worker_id] = WorkerStats()
            self.health_checks[worker_id] = 0
        logger.info(f"Registered worker {worker_id} at {address}")
    
    def unregister_worker(self, worker_id: str) -> bool:
        """Unregister a worker from the load balancer.
        
        Args:
            worker_id: ID of the worker to unregister
            
        Returns:
            bool: True if worker was found and removed, False otherwise
        """
        if worker_id in self.workers:
            del self.workers[worker_id]
            del self.worker_stats[worker_id]
            if worker_id in self.health_checks:
                del self.health_checks[worker_id]
            logger.info(f"Unregistered worker {worker_id}")
            return True
        return False
    
    async def get_worker(
        self,
        worker_id: Optional[str] = None
    ) -> Optional[tuple[str, str]]:
        """Get the next available worker based on the load balancing strategy.
        
        Args:
            worker_id: If specified, return this worker if available
            
        Returns:
            Tuple of (worker_id, address) or None if no workers available
        """
        if not self.workers:
            return None
            
        # If worker_id is specified, try to return that worker
        if worker_id and worker_id in self.workers:
            if self.health_checks.get(worker_id, 0) < self.max_failures:
                return worker_id, self.workers[worker_id]
            return None
            
        # Get healthy workers
        healthy_workers = {
            wid: addr for wid, addr in self.workers.items()
            if self.health_checks.get(wid, 0) < self.max_failures
        }
        
        if not healthy_workers:
            return None
            
        # Select worker based on strategy
        strategy_fn = self._strategy_fns.get(self.strategy, self._round_robin)
        worker_id = await strategy_fn(list(healthy_workers.keys()))
        
        if worker_id:
            return worker_id, self.workers[worker_id]
        return None
    
    async def update_stats(
        self,
        worker_id: str,
        success: bool,
        latency: float
    ) -> None:
        """Update statistics for a worker.
        
        Args:
            worker_id: ID of the worker
            success: Whether the request was successful
            latency: Request latency in seconds
        """
        if worker_id not in self.worker_stats:
            self.worker_stats[worker_id] = WorkerStats()
            
        stats = self.worker_stats[worker_id]
        stats.request_count += 1
        stats.total_latency += latency
        stats.last_seen = time.time()
        
        if not success:
            stats.error_count += 1
            self.health_checks[worker_id] = self.health_checks.get(worker_id, 0) + 1
        else:
            self.health_checks[worker_id] = 0
    
    def get_worker_stats(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a worker.
        
        Args:
            worker_id: ID of the worker
            
        Returns:
            Dictionary of worker statistics or None if worker not found
        """
        if worker_id not in self.worker_stats:
            return None
            
        stats = self.worker_stats[worker_id]
        return {
            "worker_id": worker_id,
            "address": self.workers.get(worker_id, "unknown"),
            "request_count": stats.request_count,
            "error_count": stats.error_count,
            "active_connections": stats.active_connections,
            "avg_latency": stats.avg_latency,
            "last_seen": stats.last_seen,
            "healthy": self.health_checks.get(worker_id, 0) < self.max_failures
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all workers.
        
        Returns:
            Dictionary mapping worker IDs to their statistics
        """
        return {
            wid: self.get_worker_stats(wid)
            for wid in self.workers
        }
    
    def __repr__(self) -> str:
        return f"LoadBalancer(workers={self.workers})"
    
    async def _round_robin(self, worker_ids: List[str]) -> str:
        """Round-robin worker selection strategy.
        
        Args:
            worker_ids: List of available worker IDs
            
        Returns:
            Selected worker ID
        """
        if not worker_ids:
            return None
            
        self._current_worker_idx = (self._current_worker_idx + 1) % len(worker_ids)
        return worker_ids[self._current_worker_idx]
    
    async def _least_connections(self, worker_ids: List[str]) -> str:
        """Select the worker with the fewest active connections.
        
        Args:
            worker_ids: List of available worker IDs
            
        Returns:
            Worker ID with fewest connections
        """
        if not worker_ids:
            return None
            
        return min(
            worker_ids,
            key=lambda wid: self.worker_stats[wid].active_connections
        )
    
    async def _random(self, worker_ids: List[str]) -> str:
        """Random worker selection strategy.
        
        Args:
            worker_ids: List of available worker IDs
            
        Returns:
            Randomly selected worker ID
        """
        if not worker_ids:
            return None
        return random.choice(worker_ids)
    
    async def _least_latency(self, worker_ids: List[str]) -> str:
        """Select the worker with the lowest average latency.
        
        Args:
            worker_ids: List of available worker IDs
            
        Returns:
            Worker ID with lowest average latency
        """
        if not worker_ids:
            return None
            
        return min(
            worker_ids,
            key=lambda wid: self.worker_stats[wid].avg_latency
        )
    
    async def _health_check_loop(self) -> None:
        """Background task to check worker health."""
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
            self._check_worker(worker_id)
            for worker_id in self.workers
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_worker(self, worker_id: str) -> None:
        """Check health of a single worker."""
        if worker_id not in self.workers:
            return
            
        worker_addr = self.workers[worker_id]
        host, port = worker_addr.split(":")
        port = int(port)
        
        try:
            start_time = time.time()
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=self.health_check_timeout
            )
            writer.close()
            await writer.wait_closed()
            
            # Update stats
            latency = time.time() - start_time
            self.health_checks[worker_id] = 0
            stats = self.worker_stats[worker_id]
            stats.last_seen = time.time()
            stats.total_latency += latency
            stats.request_count += 1
            
        except (asyncio.TimeoutError, ConnectionError) as e:
            logger.warning(f"Health check failed for worker {worker_id}: {e}")
            self.health_checks[worker_id] = self.health_checks.get(worker_id, 0) + 1
            if self.health_checks[worker_id] >= self.max_failures:
                logger.error(f"Worker {worker_id} marked as unhealthy")
        except Exception as e:
            logger.error(f"Error checking health of worker {worker_id}: {e}")
            self.health_checks[worker_id] = self.health_checks.get(worker_id, 0) + 1