import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class CacheEntry:
    """Represents a single cache entry with value and metadata."""
    value: Any
    created_at: float
    ttl: Optional[float] = None  # in seconds


class KVStore:
    """
    In-memory key-value store with LRU eviction.
    
    Features:
    - LRU (Least Recently Used) eviction when max_size is reached
    - TTL (Time To Live) support for cache entries
    - Thread-safe operations
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize the KVStore.
        
        Args:
            max_size: Maximum number of items to store before evicting the least recently used
        """
        self.max_size = max_size
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
    
    def _evict_if_needed(self) -> None:
        """Evict the least recently used item if we've reached max size."""
        if len(self.cache) >= self.max_size:
            # Remove the first item (least recently used)
            self.cache.popitem(last=False) # LRU eviction
    
    def _get_current_time(self) -> float:
        """Get current time in seconds since epoch."""
        return time.time()
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if an entry has expired."""
        if entry.ttl is None:
            return False
        return (self._get_current_time() - entry.created_at) > entry.ttl
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize a value for storage."""
        import pickle
        return pickle.dumps(value)
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize a value from storage."""
        import pickle
        return pickle.loads(data)
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Set a key-value pair in the store with an optional TTL.
        
        Args:
            key: The key to set
            value: The value to store
            ttl: Time to live in seconds (None for no expiration)
        """
        # Check if we need to evict an item
        if key not in self.cache:
            self._evict_if_needed()
        
        # Create and store the new entry
        entry = CacheEntry(
            value=value,
            created_at=self._get_current_time(),
            ttl=ttl
        )
        
        # Update the cache
        self.cache[key] = entry
        self.cache.move_to_end(key)  # Mark as recently used
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the store by key.
        
        Args:
            key: The key to look up
            default: Default value to return if key is not found or expired
            
        Returns:
            The stored value or default if not found/expired
        """
        if key not in self.cache:
            return default
        
        entry = self.cache[key]
        
        # Check if the entry has expired
        if self._is_expired(entry):
            self.delete(key)
            return default
        
        # Move to end to mark as recently used
        self.cache.move_to_end(key)
        return entry.value
    
    def delete(self, key: str) -> bool:
        """
        Delete a key from the store.
        
        Args:
            key: The key to delete
            
        Returns:
            True if the key was deleted, False if it didn't exist
        """
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def exists(self, key: str) -> bool:
        """
        Check if a key exists and is not expired.
        
        Args:
            key: The key to check
            
        Returns:
            True if the key exists and is not expired, False otherwise
        """
        if key not in self.cache:
            return False
            
        entry = self.cache[key]
        if self._is_expired(entry):
            self.delete(key)
            return False
            
        return True
    
    def clear(self) -> None:
        """Clear all entries from the store."""
        self.cache.clear()
    
    def close(self) -> None:
        """Close the store and release resources."""
        self.clear()
    
    def __contains__(self, key: str) -> bool:
        """Check if a key exists and is not expired."""
        return self.exists(key)
    
    def __getitem__(self, key: str) -> Any:
        """Get a value by key, raising KeyError if not found or expired."""
        value = self.get(key)
        if value is None and key not in self.cache:
            raise KeyError(key)
        return value
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set a key-value pair with no TTL."""
        self.set(key, value)
    
    def __delitem__(self, key: str) -> None:
        """Delete a key, raising KeyError if not found."""
        if not self.delete(key):
            raise KeyError(key)
    
    def __len__(self) -> int:
        """Get the number of non-expired entries in the store."""
        # Remove expired entries
        expired_keys = [
            key for key, entry in self.cache.items()
            if self._is_expired(entry)
        ]
        for key in expired_keys:
            del self.cache[key]
            
        return len(self.cache)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close the store."""
        self.close()


def create_kv_store(max_size: int = 1000) -> KVStore:
    """
    Create a new KVStore instance.
    
    Args:
        max_size: Maximum number of items to store
        
    Returns:
        A new KVStore instance
    """
    return KVStore(max_size=max_size)
    