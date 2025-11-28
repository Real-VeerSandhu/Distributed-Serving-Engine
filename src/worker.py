"""
Worker implementation for the Distributed Inference Engine.

This module provides a worker that can load models, handle inference requests,
and communicate with the coordinator.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a model."""
    model_name: str
    model_path: str
    batch_size: int = 1
    max_batch_size: int = 32
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None


class FakeModel:
    """A fake model that simulates inference with configurable latency."""
    
    def __init__(self, config: ModelConfig):
        """Initialize the fake model."""
        self.config = config
        self.model_name = config.model_name
        self.batch_size = config.batch_size
        self.max_batch_size = config.max_batch_size
        self.input_schema = config.input_schema or {}
        self.output_schema = config.output_schema or {}
        
        # Simulated model parameters
        self._latency = 0.1  # Base latency in seconds
        self._latency_std = 0.02  # Standard deviation for latency variation
        
    async def predict(self, inputs: Any) -> Any:
        """
        Simulate model inference with configurable latency.
        
        Args:
            inputs: Input data for the model
            
        Returns:
            Processed output from the model
        """
        # Simulate computation time with some random variation
        latency = max(0, self._latency + (time.time() % 0.04 - 0.02))
        await asyncio.sleep(latency)
        
        # For now, just echo back the input with some metadata
        return {
            "model": self.model_name,
            "output": inputs,
            "metadata": {
                "latency": latency,
                "batch_size": len(inputs) if isinstance(inputs, (list, tuple)) else 1,
                "timestamp": time.time()
            }
        }


class Worker:
    """Worker that loads models and handles inference requests."""
    
    def __init__(self, worker_id: str, host: str = "0.0.0.0", port: int = 0):
        """Initialize the worker."""
        self.worker_id = worker_id
        self.host = host
        self.port = port
        self.models: Dict[str, FakeModel] = {}
        self.server = None
        self._stop_event = asyncio.Event()
        
    async def start(self) -> int:
        """Start the worker server and return the actual port number."""
        self.server = await asyncio.start_server(
            self._handle_connection,
            host=self.host,
            port=self.port
        )
        
        # Get the actual port if port 0 was used (OS-assigned port)
        self.port = self.server.sockets[0].getsockname()[1]
        logger.info(f"Worker {self.worker_id} listening on {self.host}:{self.port}")
        return self.port
        
    async def stop(self) -> None:
        """Stop the worker server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info(f"Worker {self.worker_id} stopped")
            
    async def _handle_connection(self, reader: asyncio.StreamReader, 
                               writer: asyncio.StreamWriter) -> None:
        """Handle incoming client connections."""
        try:
            # Read the request
            data = await reader.read(4096)
            if not data:
                return
                
            # Parse the request
            try:
                request = json.loads(data.decode())
            except json.JSONDecodeError:
                logger.error("Invalid JSON received")
                return
                
            # Process the request
            response = await self._process_request(request)
            
            # Send the response
            writer.write(json.dumps(response).encode())
            await writer.drain()
            
        except Exception as e:
            logger.error(f"Error handling connection: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
            
    async def _process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process an inference request."""
        try:
            model_name = request.get("model")
            inputs = request.get("inputs")
            
            if not model_name or inputs is None:
                return {"error": "Invalid request: missing model or inputs"}
                
            if model_name not in self.models:
                return {"error": f"Model {model_name} not found"}
                
            # Get predictions
            model = self.models[model_name]
            outputs = await model.predict(inputs)
            
            return {
                "model": model_name,
                "outputs": outputs,
                "worker_id": self.worker_id,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {"error": str(e), "success": False}
            
    def load_model(self, config: ModelConfig) -> bool:
        """Load a model into the worker."""
        try:
            self.models[config.model_name] = FakeModel(config)
            logger.info(f"Loaded model {config.model_name} on worker {self.worker_id}")
            return True
        except Exception as e:
            logger.error(f"Error loading model {config.model_name}: {e}")
            return False
            
    def unload_model(self, model_name: str) -> bool:
        """Unload a model from the worker."""
        if model_name in self.models:
            del self.models[model_name]
            logger.info(f"Unloaded model {model_name} from worker {self.worker_id}")
            return True
        return False


async def main():
    """Example usage of the Worker class."""
    # Create and start a worker
    worker = Worker(worker_id="worker-1", host="127.0.0.1", port=9001)
    port = await worker.start()
    
    # Load a model
    config = ModelConfig(
        model_name="test-model",
        model_path="/path/to/model",
        batch_size=8,
        max_batch_size=32
    )
    worker.load_model(config)
    
    try:
        # Keep the worker running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down worker...")
    finally:
        await worker.stop()


if __name__ == "__main__":
    asyncio.run(main())