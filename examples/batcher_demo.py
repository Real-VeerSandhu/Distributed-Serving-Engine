import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.batcher import Batcher


async def mock_batch_inference(model_name: str, version: str, inputs: list) -> list:
    await asyncio.sleep(0.1)
    
    results = []
    for inp in inputs:
        results.append({
            "model": model_name,
            "version": version,
            "input": inp,
            "output": f"processed_{inp}",
            "timestamp": time.time()
        })
    
    return results


async def demonstrate_batching():
    print("=" * 70)
    print("Batcher Demo")
    print("=" * 70)
    print()
    
    batcher = Batcher(
        max_batch_size=5,
        max_latency_ms=500.0,
        batch_callback=mock_batch_inference
    )
    
    await batcher.start()
    
    print("Sending 12 requests (batch size: 5, max latency: 500ms)")
    print("Expected: 3 batches (5, 5, 2 requests)")
    print()
    
    futures = []
    start_time = time.time()
    
    for i in range(12):
        future = await batcher.add_request(
            model_name="test-model",
            version="1.0",
            inputs=f"request_{i}"
        )
        futures.append((i, future))
        print(f"Added request {i}")
        await asyncio.sleep(0.05)
    
    print("\nWaiting for all requests to complete...")
    
    results = []
    for i, future in futures:
        result = await future
        results.append((i, result))
        elapsed = (time.time() - start_time) * 1000
        print(f"Request {i} completed at {elapsed:.2f}ms")
    
    print("\nResults:")
    for i, result in sorted(results):
        print(f"  Request {i}: {result['output']}")
    
    stats = batcher.get_stats()
    print("\nBatcher Statistics:")
    print(f"  Total Batches: {stats['total_batches']}")
    print(f"  Total Requests: {stats['total_requests']}")
    print(f"  Total Batched Requests: {stats['total_batched_requests']}")
    print(f"  Average Batch Size: {stats['avg_batch_size']:.2f}")
    print(f"  Pending Batches: {stats['pending_batches']}")
    
    await batcher.stop()


async def demonstrate_latency_flush():
    print("\n" + "=" * 70)
    print("Latency-Based Flush Demo")
    print("=" * 70)
    print()
    
    batcher = Batcher(
        max_batch_size=10,
        max_latency_ms=200.0,
        batch_callback=mock_batch_inference
    )
    
    await batcher.start()
    
    print("Sending 3 requests slowly (max latency: 200ms)")
    print("Expected: Batch flushes due to latency before reaching max size")
    print()
    
    futures = []
    
    for i in range(3):
        future = await batcher.add_request(
            model_name="test-model",
            version="1.0",
            inputs=f"slow_request_{i}"
        )
        futures.append(future)
        print(f"Added request {i}")
        await asyncio.sleep(0.25)
    
    print("\nWaiting for batch to flush...")
    
    results = []
    for i, future in enumerate(futures):
        result = await future
        results.append(result)
        print(f"Request {i} completed")
    
    stats = batcher.get_stats()
    print(f"\nTotal batches created: {stats['total_batches']}")
    print("(Should be 1 batch that flushed due to latency)")
    
    await batcher.stop()


if __name__ == "__main__":
    asyncio.run(demonstrate_batching())
    asyncio.run(demonstrate_latency_flush())

