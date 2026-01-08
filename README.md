```diff
-Working in progress
```

# Distributed Serving Engine

A vanilla-Python prototype showing patterns and building blocks for distributed AI serving (including KV cache management, disaggregated inference, batching, sharding, and runtime optimizations)

## High-level Features

- **Coordinator / Orchestrator**: receives inference requests, performs routing, model selection, and scheduling.
- **Worker nodes**: run model sessions and handle batched inference. Each worker exposes a simple RPC API (TCP sockets / asyncio) implemented with stdlib.
- **Load Balancer**: distributes requests across worker nodes using configurable strategies (round-robin, least connections, etc.) and monitors worker health.
- **KV cache management**: shared request/response cache with LRU eviction; optional persistence
- **Disaggregated inference**: separate processes for model serving and for pre/post-processing to demonstrate vertical split of concerns.
- **Sharding & routing**: split model or requests across workers (simple hash-based sharding), plus fallback routing.
- **Batching & coalescing**: combine compatible queries into larger batches to improve throughput.
- **Lightweight telemetry**: request tracing and basic metrics (latency, throughput) using stdlib `time` and logging.


## Folder Layout

```
distributed-serving-engine/
├── README.md
├── src/
│   ├── coordinator.py
│   ├── router.py
│   ├── load_balancer.py
│   ├── batcher.py
│   ├── worker.py
│   ├── kvstore.py
│   ├── model_registry.py
│   ├── preproc.py
│   ├── postproc.py
│   └── utils.py
├── examples/
│   ├── example_client.py
│   └── demo_config.yaml
├── tests/
│   └── test_basic_flow.py
├── docs/
│   └── design.md
└── run.sh
```


## Requirements

- Python 3.10+ (stdlib-only prototype). Optional: `uvloop`, `aiohttp`, `grpcio` if you want to replace stdlib primitives later.
- For real model inference, plug in PyTorch/TF/ONNX; the orchestration still uses vanilla Python.


## Components / Modules

### coordinator.py
- Starts an asyncio TCP server (or HTTP endpoint) to accept inference requests.
- Validates request format, consults `kvstore` for cache hits.
- If miss, pushes request to `batcher` and returns a `request_id` for polling or pushes result back to client.
- Responsible for global scheduling decisions and retries.

### router.py
- Implements model-to-worker mapping.
- Hash-based sharding: e.g., `shard = hash(request.key) % N`.
- Handles model/shard routing and failover.

### load_balancer.py
- Distributes requests across worker nodes using configurable strategies:
  - Round-robin
  - Least connections
  - Random
  - Least latency
- Monitors worker health with configurable timeouts and failure thresholds
- Tracks performance metrics (latency, error rates, etc.)
- Supports dynamic worker registration/deregistration

### batcher.py
- Buffers requests per target-model.
- Flush policy: max batch size OR max latency.
- Coalesces inputs into a batch payload; decompresses model outputs into per-request results.

### worker.py
- Loads model artifacts (mocked in vanilla prototype).
- Runs preproc -> infer -> postproc pipeline.
- Can register with coordinator (simple registration handshake).

### kvstore.py
- In-memory dict + doubly linked list for LRU eviction.
- TTL support per entry and optional persistence.
- API: `get(key)`, `set(key, value, ttl=None)`, `delete(key)`, `exists(key)`.

### model_registry.py
- Stores model metadata: version, shards, input schema, batchable flag, quantized flag.
- Used by router/coordinator to route correctly and optimize batching.

### preproc.py / postproc.py
- Pure Python transformations (tokenization mock, normalization).
- Demonstrate push-down of CPU-bound preprocessing to a separate process to reduce worker load.

### utils.py
- Serialization helpers: `serialize(obj)`, `deserialize(bytes)` using `pickle` + length-prefixed frames.
- Simple tracer for request IDs and timings.

## Run instructions (shell)

1. Clone the repo
2. `python -m venv .venv && source .venv/bin/activate`
3. Start one or more workers: `python src/worker.py --port 9001 --model default`
4. Start coordinator: `python src/coordinator.py --listen-port 9000 --worker 127.0.0.1:9001`
5. Try the examples : `python examples/*.py`

