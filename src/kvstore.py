"""
Key-Value Cache

This module provides a high-performance, in-memory key-value cache
designed specifically for AI inference workloads.
"""

import time
import json
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Generic
from dataclasses import dataclass
from collections import OrderedDict

T = TypeVar('T')
ST = TypeVar('ST')

@dataclass
class CacheEntry(Generic[T]):
    """Represents a single cache entry with metadata."""
    value: T
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl: Optional[float] = None  # in seconds

class KVCache:
    """
    High-performance key-value cache for AI inference.
    
    Features:
    - Support for multiple eviction policies (LRU, LFU, FIFO)
    - Batch operations for better performance
    - TTL support
    - Statistics tracking
    - Thread-safe operations
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        eviction_policy: str = "lru",
        default_ttl: Optional[float] = None
    ):
        """
        Initialize the KV Cache.
        
        Args:
            max_size: Maximum number of items to store
            eviction_policy: Cache eviction policy ('lru', 'lfu', or 'fifo')
            default_ttl: Default time-to-live in seconds for new entries
        """
        self.max_size = max_size
        self.eviction_policy = eviction_policy.lower()
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self._setup_eviction_policy()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def _setup_eviction_policy(self):
        """Initialize the appropriate eviction policy data structure."""
        if self.eviction_policy == "lru":
            self._access_order = OrderedDict()
        elif self.eviction_policy == "lfu":
            self._access_count = {}
        # FIFO uses the dict's insertion order (Python 3.7+)
    
    def _update_access(self, key: str):
        """Update access metadata based on eviction policy."""
        if self.eviction_policy == "lru":
            # Move to end (most recently used)
            if key in self._access_order:
                self._access_order.move_to_end(key)
            else:
                self._access_order[key] = time.time()
        elif self.eviction_policy == "lfu":
            self._access_count[key] = self._access_count.get(key, 0) + 1
    
    def _evict_if_needed(self):
        """Evict items if we've reached max size."""
        if len(self.cache) < self.max_size:
            return
            
        self.evictions += 1
        
        if self.eviction_policy == "lru":
            # Remove least recently used
            key, _ = next(iter(self._access_order.items()))
            del self._access_order[key]
            del self.cache[key]
        elif self.eviction_policy == "lfu":
            # Remove least frequently used
            key = min(self._access_count.items(), key=lambda x: x[1])[0]
            del self._access_count[key]
            del self.cache[key]
        else:  # FIFO
            # Remove oldest item
            key = next(iter(self.cache))
            del self.cache[key]
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        batch: bool = False
    ) -> None:
        """
        Set a key-value pair in the cache.
        
        Args:
            key: The key to set
            value: The value to cache
            ttl: Optional time-to-live in seconds
            batch: If True, doesn't update access time
        """
        if key in self.cache:
            self.delete(key)
            
        self._evict_if_needed()
        
        entry = CacheEntry(
            value=value,
            created_at=time.time(),
            last_accessed=time.time(),
            ttl=ttl if ttl is not None else self.default_ttl
        )
        
        self.cache[key] = entry
        if not batch:
            self._update_access(key)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the cache.
        
        Args:
            key: The key to look up
            default: Default value if key is not found or expired
            
        Returns:
            The cached value or default if not found/expired
        """
        entry = self.cache.get(key)
        if entry is None:
            self.misses += 1
            return default
            
        # Check if expired
        if self._is_expired(entry):
            self.delete(key)
            self.misses += 1
            return default
            
        # Update access metadata
        entry.last_accessed = time.time()
        entry.access_count += 1
        self._update_access(key)
        self.hits += 1
        
        return entry.value
    
    def batch_get(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values at once."""
        return {key: self.get(key) for key in keys}
    
    def batch_set(self, items: Dict[str, Any], ttl: Optional[float] = None) -> None:
        """Set multiple key-value pairs at once."""
        for key, value in items.items():
            self.set(key, value, ttl=ttl, batch=True)
        # Update access order for all keys at once
        for key in items:
            self._update_access(key)
    
    def delete(self, key: str) -> bool:
        """Delete a key from the cache."""
        if key in self.cache:
            del self.cache[key]
            if self.eviction_policy == "lru" and key in self._access_order:
                del self._access_order[key]
            elif self.eviction_policy == "lfu" and key in self._access_count:
                del self._access_count[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all items from the cache."""
        self.cache.clear()
        if hasattr(self, '_access_order'):
            self._access_order.clear()
        if hasattr(self, '_access_count'):
            self._access_count.clear()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if an entry has expired."""
        if entry.ttl is None:
            return False
        return (time.time() - entry.created_at) > entry.ttl
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "max_size": self.max_size,
            "evictions": self.evictions,
            "eviction_policy": self.eviction_policy
        }
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        if key not in self.cache:
            return False
        if self._is_expired(self.cache[key]):
            self.delete(key)
            return False
        return True
    
    def __len__(self) -> int:
        """Get number of non-expired items in cache."""
        # Remove expired items first
        expired = [k for k, v in self.cache.items() if self._is_expired(v)]
        for k in expired:
            self.delete(k)
        return len(self.cache)

# Alias for backward compatibility
KVStore = KVCache
create_kv_store = KVCache
