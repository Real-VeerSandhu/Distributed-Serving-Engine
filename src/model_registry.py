"""
Model Registry for the Distributed Inference Engine.

This module provides a registry for model metadata, versions, and sharding information.
It's used by the coordinator and router to manage model deployment and request routing.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum, auto
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Status of a model deployment."""
    LOADING = auto()
    READY = auto()
    UPDATING = auto()
    FAILED = auto()
    UNLOADING = auto()


@dataclass
class ModelShard:
    """Represents a shard of a model."""
    shard_id: int
    worker_id: str  # e.g., "worker-1" or "ip:port"
    status: ModelStatus = ModelStatus.READY
    load: float = 0.0  # Current load on this shard (0-1)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional shard-specific metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert shard to a dictionary for serialization."""
        return {
            "shard_id": self.shard_id,
            "worker_id": self.worker_id,
            "status": self.status.name,
            "load": self.load,
            "metadata": self.metadata
        }


@dataclass
class ModelVersion:
    """Represents a specific version of a model."""
    version: str
    model_path: str
    input_schema: Dict[str, Any]  # Expected input schema
    output_schema: Dict[str, Any]  # Expected output schema
    batch_size: int = 1
    max_batch_size: int = 32
    quantized: bool = False
    shards: List[ModelShard] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional version metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert version to a dictionary for serialization."""
        return {
            "version": self.version,
            "model_path": self.model_path,
            "batch_size": self.batch_size,
            "max_batch_size": self.max_batch_size,
            "quantized": self.quantized,
            "shards": [shard.to_dict() for shard in self.shards],
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "metadata": self.metadata
        }


class ModelRegistry:
    """Manages model metadata, versions, and sharding information."""

    def __init__(self):
        """Initialize the model registry."""
        self._models: Dict[str, Dict[str, ModelVersion]] = {}  # model_name -> version -> ModelVersion
        self._worker_models: Dict[str, Set[Tuple[str, str]]] = {}  # worker_id -> set((model_name, version))
        self._model_hashes: Dict[str, str] = {}  # model_name:version -> hash

    def register_model(
        self,
        model_name: str,
        version: str,
        model_path: str,
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any],
        batch_size: int = 1,
        max_batch_size: int = 32,
        quantized: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a new model version."""
        if model_name not in self._models:
            self._models[model_name] = {}

        model_version = ModelVersion(
            version=version,
            model_path=model_path,
            input_schema=input_schema,
            output_schema=output_schema,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            quantized=quantized,
            metadata=metadata or {}
        )

        self._models[model_name][version] = model_version
        self._update_model_hash(model_name, version)

    def add_shard(
        self,
        model_name: str,
        version: str,
        shard_id: int,
        worker_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ModelShard:
        """Add a shard for a specific model version."""
        if model_name not in self._models or version not in self._models[model_name]:
            raise ValueError(f"Model {model_name} version {version} not found")

        # Check if shard already exists
        existing_shard = next((s for s in self._models[model_name][version].shards 
                             if s.shard_id == shard_id), None)
        if existing_shard:
            return existing_shard

        shard = ModelShard(
            shard_id=shard_id,
            worker_id=worker_id,
            metadata=metadata or {}
        )
        
        self._models[model_name][version].shards.append(shard)
        
        # Update worker tracking
        if worker_id not in self._worker_models:
            self._worker_models[worker_id] = set()
        self._worker_models[worker_id].add((model_name, version))

        return shard

    def get_shard_for_key(self, model_name: str, version: str, key: str) -> Optional[ModelShard]:
        """Get the appropriate shard for a given key using consistent hashing."""
        if model_name not in self._models or version not in self._models[model_name]:
            return None

        version_data = self._models[model_name][version]
        if not version_data.shards:
            return None

        # Simple hash-based sharding
        key_hash = int(hashlib.md5(key.encode()).hexdigest(), 16)
        shard_idx = key_hash % len(version_data.shards)
        return version_data.shards[shard_idx]

    def get_model_version(self, model_name: str, version: str) -> Optional[ModelVersion]:
        """Get a specific model version."""
        return self._models.get(model_name, {}).get(version)

    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self._models.keys())

    def list_versions(self, model_name: str) -> List[str]:
        """List all versions for a model."""
        return list(self._models.get(model_name, {}).keys())

    def get_worker_models(self, worker_id: str) -> List[Tuple[str, str]]:
        """Get all (model_name, version) tuples for a worker."""
        return list(self._worker_models.get(worker_id, set()))

    def _update_model_hash(self, model_name: str, version: str) -> None:
        """Update the hash for a model version."""
        model = self._models[model_name][version]
        model_dict = model.to_dict()
        # Remove shards from hash calculation as they can change without changing the model
        model_dict["shards"] = []
        model_str = json.dumps(model_dict, sort_keys=True)
        self._model_hashes[f"{model_name}:{version}"] = hashlib.md5(model_str.encode()).hexdigest()

    def get_model_hash(self, model_name: str, version: str) -> Optional[str]:
        """Get the hash of a model version's metadata."""
        return self._model_hashes.get(f"{model_name}:{version}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the registry to a dictionary for serialization."""
        return {
            "models": {
                model_name: {
                    version: version_data.to_dict()
                    for version, version_data in versions.items()
                }
                for model_name, versions in self._models.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelRegistry':
        """Create a ModelRegistry from a dictionary."""
        registry = cls()
        # Implementation would deserialize the data and populate the registry
        # This is a simplified version
        return registry