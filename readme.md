# Distributed Inference Engine

> Lightweight, vanilla-Python prototype showing patterns and building blocks for distributed AI inference: KV cache management, disaggregated inference, batching, sharding, and runtime optimizations.

---

## Project overview

**Goal.** Provide a small, well-documented reference implementation (no heavy dependencies) of a distributed inference engine. It demonstrates architectural patterns and optimizations you can use to scale model inference across machines/processes while keeping the core orchestration and tooling in plain Python.

**Audience.** Engineers and researchers who want a clear, minimal, educational system that can be extended to production by swapping in fast inference backends (PyTorch, TensorFlow, ONNX Runtime, Triton, etc.).


## High-level features (showcased)

- **Coordinator / Orchestrator** — receives inference requests, performs routing, model selection, and scheduling.
- **Worker nodes** — run model sessions and handle batched inference. Each worker exposes a simple RPC API (TCP sockets / asyncio) implemented with stdlib.
- **KV cache management** — shared request/response cache with LRU eviction; optional persistence via `sqlite3`.
- **Disaggregated inference** — separate processes for model serving and for pre/post-processing to demonstrate vertical split of concerns.
- **Sharding & routing** — split model or requests across workers (simple hash-based sharding), plus fallback routing.
- **Batching & coalescing** — combine compatible queries into larger batches to improve throughput.
- **Lightweight telemetry** — request tracing and basic metrics (latency, throughput) using stdlib `time` and logging.


## Architecture diagram (conceptual)

Coordinator (accepts requests) -> Router -> KV Cache -> Batcher -> Worker Pool (multiple worker processes / machines)

Workers: Preprocessor -> Model Inference -> Postprocessor

KV Cache and Model Registry are accessible by coordinator and workers.


## Why "vanilla" Python?

This repo focuses on architecture and control-plane logic in plain Python so you can:
- Understand distributed patterns without the noise of complex frameworks
- Replace components later with high-performance alternatives
- Learn core trade-offs (consistency, latency vs throughput, caching)


## Folder layout (shell)

```
distributed-inference-engine/
├── README.md                # this file
├── src/
│   ├── coordinator.py       # main API server + scheduler
│   ├── router.py            # routing + sharding logic
│   ├── batcher.py           # request coalescing and batching
│   ├── worker.py            # worker process that loads model and serves requests
│   ├── kvstore.py           # in-memory KV cache with LRU + optional sqlite persistence
│   ├── model_registry.py    # registry for model metadata, endpoints and shards
│   ├── preproc.py           # example pre-processing utilities
│   ├── postproc.py          # example post-processing utilities
│   └── utils.py             # helpers: serialization, tracing, metrics
├── examples/
│   ├── example_client.py    # simple client that sends requests
│   └── demo_config.yaml     # sample configuration
├── tests/
│   └── test_basic_flow.py
├── docs/
│   └── design.md
└── run.sh
```


## Requirements

- Python 3.10+ (stdlib-only prototype). Optional: `uvloop`, `aiohttp`, `grpcio` if you want to replace stdlib primitives later.
- For real model inference, plug in PyTorch/TF/ONNX; the orchestration still uses vanilla Python.


## Components (detailed shell)

### coordinator.py
- Starts an asyncio TCP server (or HTTP endpoint) to accept inference requests.
- Validates request format, consults `kvstore` for cache hits.
- If miss, pushes request to `batcher` and returns a `request_id` for polling or pushes result back to client.
- Responsible for global scheduling decisions and retries.

### router.py
- Implements model-to-worker mapping.
- Hash-based sharding: e.g., `shard = hash(request.key) % N`.
- Handles worker health checks and failover.

### batcher.py
- Buffers requests per target-model.
- Flush policy: max batch size OR max latency.
- Coalesces inputs into a batch payload; decompresses model outputs into per-request results.

### worker.py
- Loads model artifacts (mocked in vanilla prototype).
- Implements `serve()` method exposing a simple request handler (asyncio TCP or UNIX socket).
- Runs preproc -> infer -> postproc pipeline.
- Can register with coordinator (simple registration handshake).

### kvstore.py
- In-memory dict + doubly linked list for LRU eviction.
- TTL support per entry and optional persistence using `sqlite3` for warm restart.
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


## Design choices & tradeoffs

- **Batching vs latency:** larger batches increase throughput but increase tail latency. Make batching policy configurable per-model.
- **Consistency:** KV cache is eventually consistent across nodes in this prototype. For stronger consistency, replace with a consensus-backed store.
- **Sharding:** simple hash sharding is easy but uneven; consistent hashing or range sharding can be added later.
- **Fault tolerance:** coordinator retries on worker failures; consider adding a replicated model registry or leader election for production.


## Optimizations demonstrated

- **KV cache (result caching):** Cache results for repeated inputs (e.g., repeated prompts). Use LRU + TTL. Optional persistent cache via `sqlite3` for warm restarts.
- **Request coalescing & batching:** group similar requests to amortize model compute.
- **Disaggregated pipeline:** separate pre/post-processing from the actual model worker so GPU workers focus on model compute.
- **Sharded model serving:** split model across workers (conceptual) — e.g., tensor slices or layer-based partitioning. In prototype, show sharded routing and reassembly steps.
- **Adaptive batching:** dynamically adjust batch size based on current latency and queue depth.
- **Lightweight compression:** compress batched payloads with `zlib` for network-efficient transfer.
- **Zero-copy / memory mapping:** show optional pathway using `mmap` for large shared tensors between processes.


## Example API (shell)

```python
# example_client.py (sketch)
from socket import socket, AF_INET, SOCK_STREAM
import pickle

def send_request(input_text, model_name="default"):
    msg = {"model": model_name, "input": input_text}
    b = pickle.dumps(msg)
    s = socket(AF_INET, SOCK_STREAM)
    s.connect(("127.0.0.1", 9000))
    s.send(len(b).to_bytes(8, "big") + b)
    # read response length + payload
    L = int.from_bytes(s.recv(8), "big")
    payload = s.recv(L)
    resp = pickle.loads(payload)
    s.close()
    return resp
```


## Development / run instructions (shell)

1. Clone the repo
2. `python -m venv .venv && source .venv/bin/activate`
3. Start one or more workers: `python src/worker.py --port 9001 --model default`
4. Start coordinator: `python src/coordinator.py --listen-port 9000 --worker 127.0.0.1:9001`
5. Try the example client: `python examples/example_client.py`


## Testing & validation

- `tests/test_basic_flow.py` should spin up an in-process coordinator and worker (using `multiprocessing`) and assert request->response flow and KV cache hits.


## Extension ideas (future work)

- Integrate with real inference frameworks (PyTorch/TensorFlow/ONNX) — keep orchestration code unchanged.
- Add secure channels (TLS) for RPC.
- Replace coordinator with leader-election for HA using `multiprocessing`-based consensus or `etcd`.
- Add adaptive routing that moves "hot" models to fewer workers for better cache locality.
- Integrate GPU-aware scheduling and resource accounting.


## TODO: Minimal milestones (roadmap)

1. Implement `kvstore` with LRU + TTL + sqlite persistence.
2. Implement `worker` with fake model that sleeps to simulate inference latency.
3. Implement `coordinator` + `router` + `batcher` and a simple socket-based protocol.
4. Wire up `examples/example_client.py` and tests.
5. Add disaggregated preproc process and show performance benefits in a micro-benchmark.


## License

MIT


---

*(This README is a skeleton. Want me to fill in any file (e.g., implement `kvstore.py` or `batcher.py`) now?)*

