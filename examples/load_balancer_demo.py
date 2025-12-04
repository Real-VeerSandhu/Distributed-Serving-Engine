"""
Load Balancer Demo for the Distributed Inference Engine.

This script demonstrates the load balancer by:
1. Starting multiple worker instances
2. Registering them with the load balancer
3. Sending requests through the load balancer
4. Demonstrating different load balancing strategies
"""

import asyncio
import random
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

# Add the src directory to the path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.load_balancer import LoadBalancer, LoadBalancerStrategy
from src.worker import Worker
from src.config import ModelConfig

@dataclass
class DemoWorker:
    """Wrapper for a worker process in the demo."""
    worker_id: str
    host: str
    port: int
    process: Optional[asyncio.subprocess.Process] = None
    worker: Optional[Worker] = None

class DemoMode(Enum):
    """Demo modes."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RANDOM = "random"
    LEAST_LATENCY = "least_latency"

async def start_worker(worker_id: str, host: str = "0.0.0.0", port: int = 0) -> DemoWorker:
    """Start a worker process.
    
    Args:
        worker_id: Unique ID for the worker
        host: Host to bind to
        port: Port to listen on (0 for random)
        
    Returns:
        DemoWorker instance
    """
    # In a real implementation, we would start a subprocess here
    # For the demo, we'll just create a worker instance
    worker = Worker(worker_id=worker_id, host=host, port=port)
    port = await worker.start()
    
    # Load a test model
    config = ModelConfig(
        model_name="test-model",
        model_path=f"models/{worker_id}",
        batch_size=8,
        max_batch_size=32,
        input_schema={"input": "string"},
        output_schema={"output": "string"}
    )
    worker.load_model(config)
    
    return DemoWorker(
        worker_id=worker_id,
        host=host,
        port=port,
        worker=worker
    )

async def stop_worker(worker: DemoWorker) -> None:
    """Stop a worker process."""
    if worker.worker:
        await worker.worker.shutdown()
    if worker.process:
        worker.process.terminate()
        try:
            await asyncio.wait_for(worker.process.wait(), timeout=5)
        except asyncio.TimeoutError:
            worker.process.kill()
            await worker.process.wait()

async def run_demo(
    num_workers: int = 3,
    num_requests: int = 20,
    mode: DemoMode = DemoMode.ROUND_ROBIN
) -> None:
    """Run the load balancer demo.
    
    Args:
        num_workers: Number of worker processes to start
        num_requests: Number of requests to send
        mode: Load balancing strategy to demonstrate
    """
    print(f"\n{'='*50}")
    print(f"Load Balancer Demo - {mode.value.upper()} Mode")
    print("="*50)
    
    # Start workers
    print(f"\nStarting {num_workers} workers...")
    workers: List[DemoWorker] = []
    for i in range(num_workers):
        worker = await start_worker(
            worker_id=f"worker-{i+1}",
            host="0.0.0.0",
            port=8000 + i
        )
        workers.append(worker)
        print(f"  Started worker {worker.worker_id} on port {worker.port}")
    
    # Initialize load balancer
    lb = LoadBalancer(strategy=LoadBalancerStrategy(mode.value))
    for worker in workers:
        lb.register_worker(
            worker_id=worker.worker_id,
            address=f"localhost:{worker.port}"
        )
    
    await lb.start()
    
    try:
        print(f"\nSending {num_requests} requests with {mode.value} strategy...")
        print("\nWorker distribution:")
        
        request_stats = {worker.worker_id: 0 for worker in workers}
        start_time = time.time()
        
        for i in range(num_requests):
            # Get a worker from the load balancer
            worker_info = await lb.get_worker()
            if not worker_info:
                print("No healthy workers available!")
                break
                
            worker_id, _ = worker_info
            request_stats[worker_id] += 1
            
            # Simulate request processing
            process_time = 0.05 + random.random() * 0.1  # 50-150ms
            await asyncio.sleep(process_time)
            
            # Update stats with success
            await lb.update_stats(
                worker_id=worker_id,
                success=True,
                latency=process_time
            )
            
            # For least latency demo, occasionally make a worker slower
            if mode == DemoMode.LEAST_LATENCY and i == num_requests // 2:
                print("\nMaking worker-1 slower...")
                # Simulate a slow worker by adding latency to its stats
                for _ in range(10):
                    await lb.update_stats(
                        worker_id="worker-1",
                        success=True,
                        latency=0.5  # 500ms
                    )
            
            # For round-robin demo, kill a worker halfway through to show failover
            if mode == DemoMode.ROUND_ROBIN and i == num_requests // 2 and len(workers) > 1:
                print("\nSimulating worker failure...")
                failed_worker = workers[0]
                await stop_worker(failed_worker)
                lb.unregister_worker(failed_worker.worker_id)
                print(f"  Worker {failed_worker.worker_id} failed")
                workers = workers[1:]
                print(f"  Remaining workers: {[w.worker_id for w in workers]}")
            
            print(f"\r  Request {i+1}/{num_requests} - Worker: {worker_id}", end="")
        
        # Print results
        total_time = time.time() - start_time
        print(f"\n\nCompleted {num_requests} requests in {total_time:.2f} seconds")
        print(f"Average requests per second: {num_requests / total_time:.2f}")
        
        print("\nRequest distribution:")
        for worker_id, count in request_stats.items():
            print(f"  {worker_id}: {count} requests")
        
        # Show final stats
        print("\nFinal worker stats:")
        stats = lb.get_all_stats()
        for worker_id, stat in stats.items():
            if stat:
                print(f"\n{worker_id}:")
                print(f"  Address: {stat['address']}")
                print(f"  Requests: {stat['request_count']}")
                print(f"  Errors: {stat['error_count']}")
                print(f"  Avg Latency: {stat['avg_latency']*1000:.2f}ms")
                print(f"  Healthy: {stat['healthy']}")
    
    finally:
        # Clean up
        print("\nCleaning up...")
        await lb.stop()
        for worker in workers:
            await stop_worker(worker)

async def main():
    """Run the load balancer demo with different strategies."""
    parser = argparse.ArgumentParser(description="Load Balancer Demo")
    parser.add_argument(
        "--workers", 
        type=int, 
        default=3,
        help="Number of worker processes to start"
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=20,
        help="Number of requests to send per demo"
    )
    args = parser.parse_args()
    
    # Run demos for each strategy
    for mode in DemoMode:
        try:
            await run_demo(
                num_workers=args.workers,
                num_requests=args.requests,
                mode=mode
            )
            print("\n" + "="*50 + "\n")
        except Exception as e:
            print(f"Error in {mode.value} demo: {e}")
            continue

if __name__ == "__main__":
    import argparse
    asyncio.run(main())