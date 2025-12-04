import asyncio
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add parent directory to path to import the batcher module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.batcher import Batcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("batcher_demo")



async def mock_batch_inference(model_name: str, version: str, inputs: List[Any]) -> List[Dict[str, Any]]:
    """Mock batch inference function that simulates processing a batch of inputs.
    
    Args:
        model_name: Name of the model to use for inference
        version: Version of the model
        inputs: List of inputs to process
        
    Returns:
        List of processed results, one for each input
    """
    start_time = time.time()
    batch_size = len(inputs)
    logger.info(f"Processing batch of {batch_size} requests with {model_name}:{version}")
    
    # Simulate processing time
    await asyncio.sleep(0.1)
    
    results = []
    for i, inp in enumerate(inputs):
        # Simulate occasional failures (1 in 20 requests)
        if i % 20 == 0 and i > 0:
            results.append({
                "model": model_name,
                "version": version,
                "input": inp,
                "output": None,
                "error": "Simulated processing error",
                "timestamp": time.time()
            })
        else:
            results.append({
                "model": model_name,
                "version": version,
                "input": inp,
                "output": f"processed_{inp}",
                "timestamp": time.time()
            })
    
    total_time = (time.time() - start_time) * 1000
    logger.info(f"Processed batch of {batch_size} in {total_time:.2f}ms")
    
    return results


async def demonstrate_batching() -> None:
    """Demonstrate basic batching functionality with a fixed number of requests."""
    print("\n" + "=" * 70)
    print("Basic Batching Demo")
    print("=" * 70)
    print()
    
    batcher = Batcher(
        max_batch_size=5,
        max_latency_ms=500.0,
        batch_callback=mock_batch_inference
    )
    
    try:
        await batcher.start()
        
        print("Sending 12 requests (batch size: 5, max latency: 500ms)")
        print("Expected: 3 batches (5, 5, 2 requests)")
        print()
        
        futures = []
        start_time = time.time()
        
        # Add requests
        for i in range(12):
            request_time = time.time()
            future = await batcher.add_request(
                model_name="test-model",
                version="1.0",
                inputs=f"request_{i}"
            )
            futures.append((i, future, request_time))
            logger.info(f"Added request {i}")
            await asyncio.sleep(0.05)  # Small delay between requests
        
        print("\nWaiting for all requests to complete...")
        
        # Process results
        results = []
        success_count = 0
        error_count = 0
        
        for i, future, req_time in futures:
            try:
                result = await future
                elapsed = (time.time() - start_time) * 1000
                req_elapsed = (time.time() - req_time) * 1000
                
                if "error" in result:
                    logger.error(f"Request {i} failed: {result['error']} (took {req_elapsed:.2f}ms)")
                    error_count += 1
                else:
                    logger.info(f"Request {i} completed in {req_elapsed:.2f}ms")
                    success_count += 1
                
                results.append((i, result))
            except Exception as e:
                logger.error(f"Error processing request {i}: {str(e)}")
                error_count += 1
        
        # Print summary
        print("\n" + "-" * 50)
        print("Batch Processing Summary:")
        print(f"  Total Requests: {len(futures)}")
        print(f"  Successful: {success_count}")
        print(f"  Failed: {error_count}")
        
        # Print detailed stats
        stats = await batcher.get_stats()
        print("\nBatcher Statistics:")
        print(f"  Total Batches: {stats['total_batches']}")
        print(f"  Total Requests: {stats['total_requests']}")
        print(f"  Total Batched Requests: {stats['total_batched_requests']}")
        print(f"  Average Batch Size: {stats['avg_batch_size']:.2f}")
        print(f"  Pending Batches: {stats['pending_batches']}")
        print(f"  Pending Requests: {stats['pending_requests']}")
        
    except Exception as e:
        logger.error(f"Error in batching demo: {e}")
        raise
    finally:
        await batcher.stop()


async def demonstrate_latency_flush() -> None:
    """Demonstrate latency-based flushing of batches."""
    print("\n" + "=" * 70)
    print("Latency-Based Flush Demo")
    print("=" * 70)
    print()
    
    batcher = Batcher(
        max_batch_size=10,  # Will never be reached due to latency
        max_latency_ms=200.0,  # 200ms max latency
        batch_callback=mock_batch_inference
    )
    
    try:
        await batcher.start()
        
        print("Sending 3 requests slowly (max latency: 200ms)")
        print("Expected: Batch flushes due to latency before reaching max size")
        print()
        
        futures = []
        start_time = time.time()
        
        # Add requests with delays longer than the max latency
        for i in range(3):
            request_time = time.time()
            future = await batcher.add_request(
                model_name="latency-test",
                version="1.0",
                inputs=f"slow_request_{i}"
            )
            futures.append((i, future, request_time))
            logger.info(f"Added request {i}")
            await asyncio.sleep(0.25)  # Longer than max_latency_ms
        
        print("\nWaiting for all requests to complete...")
        
        # Process results
        results = []
        success_count = 0
        
        for i, future, req_time in futures:
            try:
                result = await future
                req_elapsed = (time.time() - req_time) * 1000
                
                if "error" in result:
                    logger.error(f"Request {i} failed: {result['error']} (took {req_elapsed:.2f}ms)")
                else:
                    logger.info(f"Request {i} completed in {req_elapsed:.2f}ms")
                    success_count += 1
                
                results.append((i, result))
            except Exception as e:
                logger.error(f"Error processing request {i}: {str(e)}")
        
        # Print summary
        print("\n" + "-" * 50)
        print("Latency Flush Summary:")
        print(f"  Total Requests: {len(futures)}")
        print(f"  Successful: {success_count}")
        print(f"  Failed: {len(futures) - success_count}")
        
        # Print detailed stats
        stats = await batcher.get_stats()
        print("\nBatcher Statistics:")
        print(f"  Total Batches: {stats['total_batches']} (should be 3 - one for each request due to latency)")
        print(f"  Pending Batches: {stats['pending_batches']} (should be 0)")
        print(f"  Pending Requests: {stats['pending_requests']} (should be 0)")
        
    except Exception as e:
        logger.error(f"Error in latency flush demo: {e}")
        raise
    finally:
        await batcher.stop()


async def run_demos() -> None:
    """Run all demos in sequence."""
    try:
        # Run batching demo
        await demonstrate_batching()
        
        # Add a small delay between demos
        await asyncio.sleep(1.0)
        
        # Run latency flush demo
        await demonstrate_latency_flush()
        
    except Exception as e:
        logger.exception("Error running demos")
        sys.exit(1)


if __name__ == "__main__":
    # Run all demos in a single event loop
    asyncio.run(run_demos())

